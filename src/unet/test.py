import torch
import numpy as np
from pathlib import Path
from PIL import Image

from model import UnetModel
from visualize_test import plot_test_result


# ── Settings ────────────────────────────────────────────────────────────────
TEST_DIR        = Path("data/raw/test_images")
CHECKPOINT_PATH = Path("models/checkpoint.pth")
FIGURES_DIR     = Path("reports/figures")

PATCH_SIZE = 256


# ── Load model ──────────────────────────────────────────────────────────────
def load_model(device):
    model = UnetModel().to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_statedict"])

    model.eval()
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    return model


# ── Main inference ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model = load_model(device)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    test_images = sorted(TEST_DIR.glob("*.png"))

    print(f"Found {len(test_images)} test images")

    # ── inference loop ───────────────────────────────────────────────────────
    for img_path in test_images:

        # Load image
        image = Image.open(img_path).convert("L")
        image_np = np.array(image, dtype=np.float32) / 255.0

        pred_patches = []

        # ── 2x2 patch inference ───────────────────────────────────────────────
        for r in [0, 256]:
            for c in [0, 256]:

                patch = image_np[r:r + PATCH_SIZE, c:c + PATCH_SIZE]

                patch_tensor = (
                    torch.from_numpy(patch)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )

                with torch.no_grad():
                    logits = model(patch_tensor)
                    pred = logits.argmax(dim=1).squeeze().cpu().numpy()

                pred_patches.append((pred, r, c))

        # ── stitch prediction ────────────────────────────────────────────────
        full_pred = np.zeros((512, 512), dtype=np.float32)

        for pred, r, c in pred_patches:
            full_pred[r:r + PATCH_SIZE, c:c + PATCH_SIZE] = pred

        # ensure clean type (IMPORTANT safety step)
        full_pred = np.asarray(full_pred, dtype=np.int64)

        # ── save visualization ───────────────────────────────────────────────
        save_path = FIGURES_DIR / f"{img_path.stem}_pred.png"

        plot_test_result(
            image_np,
            full_pred,
            title=img_path.stem,
            save_path=save_path,
        )

        print(f"Saved → {save_path}")