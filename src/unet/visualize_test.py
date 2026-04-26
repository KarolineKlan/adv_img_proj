from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


FIGURES_DIR = Path("reports/figures")


# ── Colors ────────────────────────────────────────────────────────────────
MEMBRANE_COLOR   = np.array([0.000, 0.533, 0.208], dtype=np.float32)
BACKGROUND_COLOR = np.array([0.600, 0.000, 0.000], dtype=np.float32)


# ── Overlay ───────────────────────────────────────────────────────────────
def make_overlay(image, pred, alpha=0.6):

    image = np.asarray(image, dtype=np.float32)
    pred  = np.asarray(pred, dtype=np.int64)

    gray = np.stack([image, image, image], axis=-1)

    mask = (pred == 1)[..., None]

    color = np.where(
        mask,
        MEMBRANE_COLOR.reshape(1, 1, 3),
        BACKGROUND_COLOR.reshape(1, 1, 3)
    )

    return (alpha * color + (1 - alpha) * gray).astype(np.float32)


# ── Main plotting function ────────────────────────────────────────────────
def plot_test_result(image, pred, title="", save_path=None):

    overlay = make_overlay(image, pred)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if title:
        plt.suptitle(title, fontsize=14)

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
    else:
        plt.show()