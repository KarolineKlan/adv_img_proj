from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


FIGURES_DIR = Path("reports/figures")


def plot_predictions(model, dataset, device, n=4, epoch=None):
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9))

    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    with torch.no_grad():
        for col, idx in enumerate(indices):
            image, label = dataset[idx]
            logits = model(image.unsqueeze(0).to(device))          # (1, 2, H, W)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()   # (H, W)

            axes[0, col].imshow(image[0], cmap="gray")
            axes[0, col].set_title(f"Image (patch {idx})")
            axes[0, col].axis("off")

            axes[1, col].imshow(label, cmap="gray", vmin=0, vmax=1)
            axes[1, col].set_title("Label")
            axes[1, col].axis("off")

            axes[2, col].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[2, col].set_title("Prediction")
            axes[2, col].axis("off")

    title = f"Epoch {epoch}" if epoch is not None else "Predictions"
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"predictions_epoch{epoch:04d}.png" if epoch is not None else "predictions.png"
    plt.savefig(FIGURES_DIR / fname, dpi=100)
    plt.close()


MEMBRANE_COLOR   = np.array([0.000, 0.533, 0.208], dtype=np.float32)  # #008835
BACKGROUND_COLOR = np.array([0.600, 0.000, 0.000], dtype=np.float32)  # #990000


def make_overlay(image, pred_mask, alpha=0.60):
    """Blend corporate red/green colors with the grayscale image.

    Background pixels are tinted red (#990000), membrane pixels green (#008835).
    Alpha controls how strong the color is vs. the underlying grayscale.

    Args:
        image    : (H, W) float32 grayscale in [0, 1]
        pred_mask: (H, W) int array with values {0, 1}, 1 = membrane
        alpha    : color blend strength [0 = grayscale only, 1 = solid color]
    Returns:
        (H, W, 3) float32 RGB array
    """
    gray3    = np.stack([image, image, image], axis=-1)          # (H, W, 3)
    membrane = (pred_mask == 1)[..., np.newaxis]                 # (H, W, 1)

    color = np.where(membrane, MEMBRANE_COLOR, BACKGROUND_COLOR) # (H, W, 3)
    return (alpha * color + (1 - alpha) * gray3).astype(np.float32)


def plot_segmentation_result(image, label, pred, title="", save_path=None):
    """Four-panel plot: Input | Ground Truth | Prediction | Overlay.

    Args:
        image    : (H, W) float32 grayscale in [0, 1]
        label    : (H, W) int ground truth mask
        pred     : (H, W) int predicted mask
        title    : suptitle string
        save_path: Path to save to; if None the figure is shown interactively
    """
    overlay = make_overlay(image, pred)

    _, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        (image,   "gray", "Input"),
        (label,   "gray", "Ground Truth"),
        (pred,    "gray", "Prediction"),
        (overlay, None,   "Overlay"),
    ]
    for ax, (data, cmap, name) in zip(axes, panels):
        ax.imshow(data, cmap=cmap, vmin=0, vmax=1) if cmap else ax.imshow(data)
        ax.set_title(name)
        ax.axis("off")

    if title:
        plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train loss", lw=2)
    plt.plot(val_losses,   label="Val loss",   lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "losses.png", dpi=100)
    plt.close()