import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from collections import defaultdict

from data import EMDataset
from model import UnetModel

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


# ── Visualise one stitched result ─────────────────────────────────────────────
FIGURES_DIR = Path("reports/figures")


def plot_stitched(image, label, pred, img_idx, dice, iou):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Image {img_idx:02d}")
    axes[0].axis("off")

    axes[1].imshow(label, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Label (ground truth)")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction\nDice: {dice:.3f}  IoU: {iou:.3f}")
    axes[2].axis("off")

    plt.suptitle(f"Test image {img_idx:02d}", fontsize=13)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / f"eval_image{img_idx:02d}.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved → {FIGURES_DIR / f'eval_image{img_idx:02d}.png'}")


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
        plot_stitched(full_image, full_label, full_pred, img_idx, dice, iou)

    # --- Summary ---
    print(f"\nMean Dice : {np.mean(all_dice):.3f}  ± {np.std(all_dice):.3f}")
    print(f"Mean IoU  : {np.mean(all_iou):.3f}  ± {np.std(all_iou):.3f}")