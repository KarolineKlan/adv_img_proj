from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import elastic_transform, gaussian_blur
from torchvision.transforms import InterpolationMode
import PIL.Image
import random


def elastic_deform(image, label, alpha=34.0, sigma=4.0, seed=None):
    """Apply identical elastic deformation to image and label.

    Uses torchvision's elastic_transform with a shared displacement field so
    image and label stay aligned. Bilinear interpolation for the image,
    nearest-neighbour for the label to preserve integer class values.
    """
    h, w = image.shape
    # Generate and smooth displacement field (no scipy needed)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    noise = torch.randn(2, h, w, generator=generator)
    kernel_size = 2 * int(4 * sigma) + 1          # 6-sigma rule, forced odd
    # Divide by image size: elastic_transform expects normalized [-1, 1] coordinates
    dx = gaussian_blur(noise[0:1], kernel_size, sigma) * alpha / w  # (1, H, W)
    dy = gaussian_blur(noise[1:2], kernel_size, sigma) * alpha / h
    displacement = torch.stack([dx.squeeze(0), dy.squeeze(0)], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    img_t = torch.from_numpy(image).unsqueeze(0)                   # (1, H, W)
    lbl_t = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)

    img_def = elastic_transform(img_t, displacement, InterpolationMode.BILINEAR).squeeze(0).numpy()
    lbl_def = elastic_transform(lbl_t, displacement, InterpolationMode.NEAREST).squeeze(0).numpy().astype(np.int64)
    return img_def, lbl_def

#we decide to do a 23/5/2 split 

TRAIN_INDICES = [i for i in range(1, 24)]   # 1–23
VAL_INDICES   = [i for i in range(24, 29)]  # 24–28
TEST_INDICES  = [i for i in range(29, 31)]  # 29–30


class EMDataset(Dataset):
    """Dataset of EM membrane patches for training/validation/testing.

    Each 512x512 image is split into four 256x256 patches (2x2 grid).
    Images are grayscale, normalised to [0, 1].
    Labels are binary: 1 = membrane, 0 = background.

    Usage:
        train_data = EMDataset("data/raw", split="train", augment=True)
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

        for images, labels in train_loader:
            # images: (B, 1, 256, 256)
            # labels: (B, 256, 256)
    """

    def __init__(self, data_dir, split="train", augment=False):
        self.augment = augment
        self.patches = []   # list of (image_patch, label_patch) numpy arrays

        if split == "train":
            indices = TRAIN_INDICES
        elif split == "val":
            indices = VAL_INDICES
        else:
            indices = TEST_INDICES

        for idx in indices:
            image = np.array(
                PIL.Image.open(Path(data_dir) / "train_images" / f"train_{idx:02d}.png").convert("L"),
                dtype=np.float32
            ) / 255.0

            label = np.array(
                PIL.Image.open(Path(data_dir) / "train_labels" / f"labels_{idx:02d}.png").convert("L")
            )
            # dark membranes = 0, and white cell/background = 1
            label = (label > 128).astype(np.int64)

            # Split 512x512 into four 256x256 patches
            for r in [0, 256]:
                for c in [0, 256]:
                    self.patches.append({
                        "image":     image[r:r+256, c:c+256],
                        "label":     label[r:r+256, c:c+256],
                        "image_idx": idx,
                        "row":       r,
                        "col":       c,
                    })

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        image = self.patches[idx]["image"]
        label = self.patches[idx]["label"]
        image = image.copy()
        label = label.copy()

        if self.augment:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()
            if random.random() > 0.5:
                image = np.flipud(image).copy()
                label = np.flipud(label).copy()
            k = random.randint(0, 3)
            image = np.rot90(image, k).copy()
            label = np.rot90(label, k).copy()
            if random.random() > 0.5:
                image, label = elastic_deform(image, label)

        image = torch.tensor(image).unsqueeze(0)  # (1, 256, 256)
        label = torch.tensor(label)               # (256, 256)

        return image, label


# ------------------------------------------------------------------------------------------------
###### Alt herunder er bare til at få et par plots og printe lidt for at se om det virker ###################

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    DATA_DIR = "data/raw"

    # --- Check sizes ---
    train_data = EMDataset(DATA_DIR, split="train", augment=True)
    val_data   = EMDataset(DATA_DIR, split="val",   augment=False)
    test_data  = EMDataset(DATA_DIR, split="test",  augment=False)

    print(f"Train patches : {len(train_data)}  (expect {23 * 4} = 88)")
    print(f"Val patches   : {len(val_data)}   (expect {5 * 4} = 20)")
    print(f"Test patches  : {len(test_data)}   (expect {2 * 4} = 8)")

    # --- Check a single sample ---
    image, label = train_data[0]
    print(f"\nSingle sample:")
    print(f"  image shape : {image.shape}  (expect [1, 256, 256])")
    print(f"  label shape : {label.shape}  (expect [256, 256])")
    print(f"  image range : [{image.min():.3f}, {image.max():.3f}]  (expect [0, 1])")
    print(f"  label values: {label.unique().tolist()}  (expect [0, 1])")

    # --- Check a batch from the DataLoader ---
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    images, labels = next(iter(train_loader))
    print(f"\nBatch from DataLoader:")
    print(f"  images : {images.shape}  (expect [8, 1, 256, 256])")
    print(f"  labels : {labels.shape}  (expect [8, 256, 256])")

####### Plot stitching and patches ##############

    # --- Stitch the four patches of the first training image back to 512x512 ---
    first_idx = train_data.patches[0]["image_idx"]
    first_patches = [p for p in train_data.patches if p["image_idx"] == first_idx]
    stitched_image = np.zeros((512, 512), dtype=np.float32)
    stitched_label = np.zeros((512, 512), dtype=np.int64)
    for p in first_patches:
        r, c = p["row"], p["col"]
        stitched_image[r:r+256, c:c+256] = p["image"]
        stitched_label[r:r+256, c:c+256] = p["label"]

    # --- Visualise: 2x2 patches + stitched, mirrored for image (cols 0-1) and label (cols 2-3) ---
    # Sort patches so layout is: (0,0) (0,256) / (256,0) (256,256)
    first_patches = sorted(first_patches, key=lambda p: (p["row"], p["col"]))

    fig = plt.figure(figsize=(14, 11))
    gs  = fig.add_gridspec(3, 4, hspace=0.1, wspace=0.05)

    patch_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # grid (row, col) for each patch

    for p, (gr, gc) in zip(first_patches, patch_positions):
        patch_title = f"idx={p['image_idx']}  r={p['row']}, c={p['col']}"

        ax_img = fig.add_subplot(gs[gr, gc])
        ax_img.imshow(p["image"], cmap="gray")
        ax_img.set_title(patch_title, fontsize=8)
        ax_img.axis("off")

        ax_lbl = fig.add_subplot(gs[gr, gc + 2])
        ax_lbl.imshow(p["label"], cmap="gray", vmin=0, vmax=1)
        ax_lbl.set_title(patch_title, fontsize=8)
        ax_lbl.axis("off")

    ax_si = fig.add_subplot(gs[2, :2])
    ax_si.imshow(stitched_image, cmap="gray")
    ax_si.set_title("Stitched back image", fontsize=10)
    ax_si.axis("off")

    ax_sl = fig.add_subplot(gs[2, 2:])
    ax_sl.imshow(stitched_label, cmap="gray", vmin=0, vmax=1)
    ax_sl.set_title("Stitched back label", fontsize=10)
    ax_sl.axis("off")

    # Column group titles
    fig.text(0.27, 0.97, f"Image {first_idx:02d}", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.73, 0.97, f"Label {first_idx:02d}",  ha="center", fontsize=12, fontweight="bold")

    fig.suptitle("EMDataset sanity check — train split", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()

######## PLOT AUGMENTATIONS ############
    # --- Augmentation showcase: each column is a specific transform ---
    aug_transforms = {
        "Original":   lambda x: x,
        "Flip H":     lambda x: np.fliplr(x).copy(),
        "Flip V":     lambda x: np.flipud(x).copy(),
        "Rot 90°":    lambda x: np.rot90(x, 1).copy(),
        "Rot 180°":   lambda x: np.rot90(x, 2).copy(),
        "Rot 270°":   lambda x: np.rot90(x, 3).copy(),
    }

    showcase_patches = [0, 4, 8, 12] # one patch from each of the first four training images
    fig2, axes2 = plt.subplots(len(showcase_patches), len(aug_transforms), figsize=(14, 9))

    for row, patch_idx in enumerate(showcase_patches):
        p = train_data.patches[patch_idx]
        for col, (name, fn) in enumerate(aug_transforms.items()):
            ax = axes2[row, col]
            ax.imshow(fn(p["image"]), cmap="gray")
            if row == 0:
                ax.set_title(name, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(
                    f"idx={p['image_idx']}\nr={p['row']}, c={p['col']}", fontsize=8
                )
                ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                ax.spines[:].set_visible(False)
            else:
                ax.axis("off")

    fig2.suptitle("Augmentation showcase — specific transforms per column", fontsize=13)
    plt.tight_layout()
    plt.show()

    ######## POSTER AUGMENTATION FIGURE ############
    # Single patch shown with each augmentation type — designed for poster use
    poster_patch = train_data.patches[0]
    img = poster_patch["image"]
    img_elastic, _ = elastic_deform(img, poster_patch["label"], seed=42)

    poster_transforms = {
        "Original":            img,
        "Flip \n horizontal":     np.fliplr(img).copy(),
        "Flip \n vertical":       np.flipud(img).copy(),
        "Elastic \n deformation": img_elastic,
        "90° \n rotation":        np.rot90(img, 1).copy(),
        "180° \n rotation":       np.rot90(img, 2).copy(),
        "270° \n rotation":       np.rot90(img, 3).copy(),
    }

    fig3, axes3 = plt.subplots(1, len(poster_transforms), figsize=(20, 5))
    for ax, (name, data) in zip(axes3, poster_transforms.items()):
        ax.imshow(data, cmap="gray")
        ax.set_title(name, fontsize=16, fontweight="bold", pad=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("reports/figures/augmentation_poster.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → reports/figures/augmentation_poster.png")