import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from collections import defaultdict

from data import EMDataset
from model import UnetModel
from visualize import plot_segmentation_result

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data/raw")
CHECKPOINT_PATH = Path("models/checkpoint.pth")
PATCH_SIZE      = 256
FULL_SIZE       = 512


# ── Metrics ───────────────────────────────────────────────────────────────────
def dice_score(pred, label):
    """Dice score for the membrane class (class = 1).
    pred, label: 2D numpy arrays with values in {0, 1}
    """
    intersection = np.logical_and(pred == 1, label == 1).sum()
    return (2 * intersection) / (pred.sum() + label.sum() + 1e-8)


def iou_score(pred, label):
    """IoU for the membrane class (class = 1)."""
    intersection = np.logical_and(pred == 1, label == 1).sum()
    union        = np.logical_or( pred == 1, label == 1).sum()
    return intersection / (union + 1e-8)


# ── Stitching ─────────────────────────────────────────────────────────────────
def stitch(patches_with_coords, full_size=FULL_SIZE):
    """Place patches back into a full-size image.
    patches_with_coords: list of (patch, row, col)
    """
    canvas = np.zeros((full_size, full_size), dtype=patches_with_coords[0][0].dtype)
    for patch, r, c in patches_with_coords:
        canvas[r:r+PATCH_SIZE, c:c+PATCH_SIZE] = patch
    return canvas


FIGURES_DIR = Path("reports/figures")


# ── Main evaluation ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    # --- Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device}")

    model = UnetModel().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_statedict"])
    model.eval()
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}\n")

    # --- Load test data (no augmentation) ---
    # We need patch coordinates for stitching, so we access patches directly
    test_data = EMDataset(DATA_DIR, split="test", augment=False)

    # Group patches by image index so we can stitch them back together.
    # test_data.patches is a list of dicts with keys: image, label, image_idx, row, col
    groups = defaultdict(list)
    for patch_dict in test_data.patches:
        groups[patch_dict["image_idx"]].append(patch_dict)

    all_dice = []
    all_iou  = []

    for img_idx, patches in sorted(groups.items()):

        image_patches = []  # for stitching the input image
        label_patches = []  # for stitching the ground truth
        pred_patches  = []  # for stitching the prediction

        for patch_dict in patches:
            image_np = patch_dict["image"]   # (256, 256) float32
            label_np = patch_dict["label"]   # (256, 256) int64
            r, c     = patch_dict["row"], patch_dict["col"]

            # Run model on this patch
            image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 256, 256)
            with torch.no_grad():
                logits = model(image_tensor)                          # (1, 2, 256, 256)
                pred_np = logits.argmax(dim=1).squeeze().cpu().numpy()  # (256, 256)

            image_patches.append((image_np, r, c))
            label_patches.append((label_np, r, c))
            pred_patches.append((pred_np,  r, c))

        # Stitch all four patches back into 512x512
        full_image = stitch(image_patches)
        full_label = stitch(label_patches)
        full_pred  = stitch(pred_patches)

        # Compute metrics on the full image
        dice = dice_score(full_pred, full_label)
        iou  = iou_score( full_pred, full_label)
        all_dice.append(dice)
        all_iou.append(iou)

        print(f"Image {img_idx:02d}  Dice: {dice:.3f}  IoU: {iou:.3f}")

        # Show stitched result
        save_path = FIGURES_DIR / f"eval_image{img_idx:02d}.png"
        plot_segmentation_result(
            full_image, full_label, full_pred,
            title=f"Test image {img_idx:02d}  —  Dice: {dice:.3f}  IoU: {iou:.3f}",
            save_path=save_path,
        )
        print(f"Saved → {save_path}")

    # --- Summary ---
    print(f"\nMean Dice : {np.mean(all_dice):.3f}  ± {np.std(all_dice):.3f}")
    print(f"Mean IoU  : {np.mean(all_iou):.3f}  ± {np.std(all_iou):.3f}")