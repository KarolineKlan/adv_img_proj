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