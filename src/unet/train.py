import torch
from torch.utils.data import DataLoader
from pathlib import Path

from data import EMDataset
from model import UnetModel
from visualize import plot_predictions, plot_losses

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/raw")
NR_EPOCHS  = 30
BATCH_SIZE = 8
LR         = 1e-4

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ── Data ──────────────────────────────────────────────────────────────────────
train_data = EMDataset(DATA_DIR, split="train", augment=True)
val_data   = EMDataset(DATA_DIR, split="val",   augment=False)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Train patches: {len(train_data)}, Val patches: {len(val_data)}")

# ── Model, loss, optimiser ────────────────────────────────────────────────────
model     = UnetModel().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── Show predictions before any training ──────────────────────────────────────
print("\nPredictions before training:")
plot_predictions(model, val_data, device, n=4, epoch=0)

# ── Training loop ─────────────────────────────────────────────────────────────
train_losses = []
val_losses   = []

for epoch in range(NR_EPOCHS):

    # --- Train ---
    model.train()
    epoch_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # --- Validate ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            val_loss += criterion(logits, labels).item()

    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}/{NR_EPOCHS}  "
          f"train loss: {train_losses[-1]:.4f}  "
          f"val loss:   {val_losses[-1]:.4f}")

    # --- Visualise every 2 epochs ---
    if (epoch + 1) % 2 == 0:
        plot_predictions(model, val_data, device, n=4, epoch=epoch+1)
        plot_losses(train_losses, val_losses)

# ── Final plots ───────────────────────────────────────────────────────────────
plot_predictions(model, val_data, device, n=4, epoch=NR_EPOCHS)
plot_losses(train_losses, val_losses)

# ── Save checkpoint ───────────────────────────────────────────────────────────
checkpoint_path = Path("models/checkpoint.pth")
checkpoint_path.parent.mkdir(exist_ok=True)
torch.save({
    "epoch":              NR_EPOCHS,
    "model_statedict":    model.state_dict(),
    "optimizer_statedict": optimizer.state_dict(),
    "train_losses":       train_losses,
    "val_losses":         val_losses,
}, checkpoint_path)
print(f"\nCheckpoint saved to {checkpoint_path}")