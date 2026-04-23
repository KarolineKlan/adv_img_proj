import torch
import numpy as np
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
FIGURES_DIR     = Path("reports/figures")
PROB_MAPS_DIR   = Path("data/prob_maps")  # max_flow.py reads .npy files from here


# ── Metrics ───────────────────────────────────────────────────────────────────
# Convention: labels are membrane=0, background=1 (matching the raw PNG files
# where membranes are dark/black and background is white/bright).
# Metrics evaluate the membrane class (== 0).

def dice_score(pred, label):
    """Dice score for the membrane class (membrane = 0).
    pred, label: 2D numpy arrays with values in {0, 1}
    """
    intersection = np.logical_and(pred == 0, label == 0).sum()
    return (2 * intersection) / ((pred == 0).sum() + (label == 0).sum() + 1e-8)


def iou_score(pred, label):
    """IoU for the membrane class (membrane = 0)."""
    intersection = np.logical_and(pred == 0, label == 0).sum()
    union        = np.logical_or( pred == 0, label == 0).sum()
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

    # --- Create output dirs ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROB_MAPS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load test data (no augmentation) ---
    test_data = EMDataset(DATA_DIR, split="test", augment=False)

    # Group patches by image index so we can stitch them back together
    groups = defaultdict(list)
    for patch_dict in test_data.patches:
        groups[patch_dict["image_idx"]].append(patch_dict)

    all_dice = []
    all_iou  = []

    for img_idx, patches in sorted(groups.items()):

        image_patches = []  # for stitching the input image
        label_patches = []  # for stitching the ground truth
        pred_patches  = []  # for stitching the hard prediction (argmax)
        prob_patches  = []  # for stitching P(membrane) — fed into MRF

        for patch_dict in patches:
            image_np = patch_dict["image"]   # (256, 256) float32
            label_np = patch_dict["label"]   # (256, 256) int64 — membrane=0, background=1
            r, c     = patch_dict["row"], patch_dict["col"]

            image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(image_tensor)                             # (1, 2, 256, 256)

                # Hard prediction — argmax gives 0 (membrane) or 1 (background)
                pred_np = logits.argmax(dim=1).squeeze().cpu().numpy()  # (256, 256)

                # Soft probability — same forward pass, no extra cost
                # Identical to Week 10: prob_val = torch.nn.functional.softmax(lgt_val, dim=1)
                # Channel 0 = P(membrane), channel 1 = P(background) — matches label convention
                probs   = torch.nn.functional.softmax(logits, dim=1)   # (1, 2, 256, 256)
                prob_np = probs[0, 0].cpu().numpy()                     # (256, 256) P(membrane)

            image_patches.append((image_np, r, c))
            label_patches.append((label_np, r, c))
            pred_patches.append((pred_np,   r, c))
            prob_patches.append((prob_np,   r, c))

        # Stitch all patches back into 512x512
        full_image = stitch(image_patches)
        full_label = stitch(label_patches)
        full_pred  = stitch(pred_patches)
        full_prob  = stitch(prob_patches)  # float32 in [0, 1] — P(membrane)

        # Compute metrics on membrane class (== 0)
        dice = dice_score(full_pred, full_label)
        iou  = iou_score( full_pred, full_label)
        all_dice.append(dice)
        all_iou.append(iou)
        print(f"Image {img_idx:02d}  Dice: {dice:.3f}  IoU: {iou:.3f}")

        # Save probability maps — max_flow.py loads these directly:
        #   cap_sink   = np.load("data/prob_maps/prob_membrane_XX.npy")   → P(membrane)
        #   cap_source = np.load("data/prob_maps/prob_background_XX.npy") → P(background)
        np.save(PROB_MAPS_DIR / f"prob_membrane_{img_idx:02d}.npy",   full_prob)
        np.save(PROB_MAPS_DIR / f"prob_background_{img_idx:02d}.npy", 1 - full_prob)
        print(f"  Saved → prob_membrane_{img_idx:02d}.npy, prob_background_{img_idx:02d}.npy")

        # Show and save stitched segmentation result
        save_path = FIGURES_DIR / f"eval_image{img_idx:02d}.png"
        plot_segmentation_result(
            full_image, full_label, full_pred,
            title=f"Test image {img_idx:02d}  —  Dice: {dice:.3f}  IoU: {iou:.3f}",
            save_path=save_path,
        )
        print(f"  Saved → {save_path}")

    # --- Summary ---
    print(f"\nMean Dice : {np.mean(all_dice):.3f}  ± {np.std(all_dice):.3f}")
    print(f"Mean IoU  : {np.mean(all_iou):.3f}  ± {np.std(all_iou):.3f}")