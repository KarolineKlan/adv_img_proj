import argparse
import numpy as np
import matplotlib.pyplot as plt
import maxflow
from scipy.signal import find_peaks

# Import metrics from evaluate.py — these now correctly evaluate membrane class (== 0)
from evaluate_probs import dice_score, iou_score

PROB_MAPS_DIR = "data/prob_maps"

# ── Convention (confirmed by diagnostic) ──────────────────────────────────────
# Raw PNG labels: membrane = black = 0.0, background = white = 1.0
# After load_mask (> 0.5): membrane = 0, background = 1
# prob_membrane.npy: high values = high P(membrane)  — bright = membrane
# All labelings produced here must follow membrane=0, background=1


def load_mask(image_id):
    """Load ground truth mask with confirmed convention: membrane=0, background=1.
    Raw PNG: white background (1.0), black membranes (0.0).
    After > 0.5: background=1, membrane=0. Confirmed correct by diagnostic.
    """
    mask_raw = plt.imread(f"data/raw/train_labels/labels_{image_id}.png")
    return (mask_raw > 0.5).astype(np.uint8)  # background=1, membrane=0


def run_mrf_intensity(image, mask, beta):
    """MRF using raw pixel intensity as the data term (baseline).

    The graph cut assigns source=0 or sink=1 to each pixel.
    After maxflow, get_grid_segments returns True where pixel is on the source side.
    We check the result mean to confirm orientation and invert if needed
    so the output always has membrane=0, background=1.
    """
    N, M = image.shape
    d    = image.flatten()

    hist, bin_edges = np.histogram(d, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peaks, _ = find_peaks(hist, distance=5)
    mu_values = np.sort(bin_centers[peaks])

    if len(mu_values) < 2:
        print("Error: Could not find two distinct peaks in histogram.")
        return None, None, None, None, None

    mu1 = mu_values[0]   # darker  = membrane
    mu2 = mu_values[-1]  # lighter = background
    print(f"[Intensity MRF] Means — membrane: {mu1:.2f}, background: {mu2:.2f}")

    g     = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((N, M))
    struct_h = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    struct_v = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    g.add_grid_edges(nodes, beta, structure=struct_h, symmetric=True)
    g.add_grid_edges(nodes, beta, structure=struct_v, symmetric=True)

    # Pixels close to mu1 (dark/membrane) are cheaper to assign to source
    # Pixels close to mu2 (bright/background) are cheaper to assign to sink
    cap_source = (image - mu1)**2
    cap_sink   = (image - mu2)**2
    g.add_grid_tedges(nodes, cap_source, cap_sink)

    g.maxflow()
    labeling = g.get_grid_segments(nodes).astype(np.uint8)

    # Ensure membrane=0, background=1
    # If most pixels are 0 then background is 0 — needs inversion
    if labeling.mean() < 0.5:
        labeling = 1 - labeling

    dice = dice_score(labeling, mask)
    iou  = iou_score(labeling,  mask)
    print(f"[Intensity MRF] Dice: {dice:.4f}  IoU: {iou:.4f}")
    return labeling, dice, iou, mu1, mu2


def run_unet_only(image_id, mask):
    """Standalone U-Net prediction from saved probability maps.
    prob_membrane has high values where membrane is likely.
    argmin gives 0 (membrane wins) where P(membrane) > P(background).
    """
    prob_membrane   = np.load(f"{PROB_MAPS_DIR}/prob_membrane_{image_id}.npy")
    prob_background = np.load(f"{PROB_MAPS_DIR}/prob_background_{image_id}.npy")

    # Where P(membrane) > P(background), assign 0 (membrane). Otherwise 1 (background).
    labeling = np.where(prob_membrane > prob_background, 0, 1).astype(np.uint8)

    dice = dice_score(labeling, mask)
    iou  = iou_score(labeling,  mask)
    print(f"[U-Net only]   Dice: {dice:.4f}  IoU: {iou:.4f}")
    return labeling, prob_membrane, dice, iou


def run_mrf_unet(image, mask, image_id, beta):
    """MRF using U-Net probability maps as the data term.

    Log-likelihood formulation: cost = -log P(label | pixel).
    Low probability → high cost → graph cut avoids that assignment.
    cap_source = cost of assigning to source (background)
    cap_sink   = cost of assigning to sink   (membrane)
    """
    prob_membrane   = np.load(f"{PROB_MAPS_DIR}/prob_membrane_{image_id}.npy")
    prob_background = np.load(f"{PROB_MAPS_DIR}/prob_background_{image_id}.npy")

    N, M = image.shape
    g     = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((N, M))
    struct_h = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    struct_v = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    g.add_grid_edges(nodes, beta, structure=struct_h, symmetric=True)
    g.add_grid_edges(nodes, beta, structure=struct_v, symmetric=True)

    cap_source = -np.log(prob_background + 1e-6)  # cost of labelling as background
    cap_sink   = -np.log(prob_membrane   + 1e-6)  # cost of labelling as membrane
    g.add_grid_tedges(nodes, cap_source, cap_sink)

    g.maxflow()
    labeling = g.get_grid_segments(nodes).astype(np.uint8)

    # Ensure membrane=0, background=1
    if labeling.mean() < 0.5:
        labeling = 1 - labeling

    dice = dice_score(labeling, mask)
    iou  = iou_score(labeling,  mask)
    print(f"[U-Net + MRF]  Dice: {dice:.4f}  IoU: {iou:.4f}")
    return labeling, dice, iou


def plot_comparison(image, mask, labeling_mrf, labeling_unet, labeling_comb,
                    dice_mrf, iou_mrf, dice_unet, iou_unet, dice_comb, iou_comb, args):
    """5-panel segmentation comparison.
    All labelings have membrane=0 (black) and background=1 (white).
    """
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))

    axs[0].imshow(image, cmap='gray')
    axs[0].set_title(f"Original (ID: {args.image_id})")

    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Ground Truth\n(white=background, black=membrane)")

    axs[2].imshow(labeling_mrf, cmap='gray')
    axs[2].set_title(f"Intensity MRF\nDice: {dice_mrf:.3f}  IoU: {iou_mrf:.3f}")

    axs[3].imshow(labeling_unet, cmap='gray')
    axs[3].set_title(f"U-Net only\nDice: {dice_unet:.3f}  IoU: {iou_unet:.3f}")

    axs[4].imshow(labeling_comb, cmap='gray')
    axs[4].set_title(f"U-Net + MRF\nDice: {dice_comb:.3f}  IoU: {iou_comb:.3f}")

    for ax in axs:
        ax.axis('off')

    plt.suptitle(f"Segmentation comparison  —  beta={args.beta}", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_probability_analysis(image, prob_membrane, mu1, mu2, args):
    """Three plots showing what the U-Net probability map looks like."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # ── Plot 1: Intensity histogram with MRF means ────────────────────────────
    d = image.flatten()
    axs[0].hist(d, bins=50, color='gray', alpha=0.8)
    axs[0].axvline(mu1, color='r', linestyle='--', lw=2, label=f'Membrane mean ({mu1:.2f})')
    axs[0].axvline(mu2, color='b', linestyle='--', lw=2, label=f'Background mean ({mu2:.2f})')
    axs[0].set_title("Intensity histogram\n(MRF data term)")
    axs[0].set_xlabel("Pixel intensity")
    axs[0].set_ylabel("Frequency")
    axs[0].legend()

    # ── Plot 2: U-Net P(membrane) histogram ───────────────────────────────────
    # Most pixels should be near 0 (background), smaller peak near 1 (membrane)
    axs[1].hist(prob_membrane.flatten(), bins=100, color='steelblue', alpha=0.8)
    axs[1].axvline(0.5, color='r', linestyle='--', lw=2, label='Decision threshold (0.5)')
    axs[1].set_title("U-Net P(membrane) histogram\n(network confidence)")
    axs[1].set_xlabel("P(membrane)")
    axs[1].set_ylabel("Frequency")
    axs[1].set_xlim(0, 1)
    axs[1].legend()

    # ── Plot 3: Probability map overlaid on image ─────────────────────────────
    # hot_r: dark = high P(membrane), matching visual convention where membranes are dark
    axs[2].imshow(image, cmap='gray')
    overlay = axs[2].imshow(prob_membrane, cmap='hot_r', alpha=0.5, vmin=0, vmax=1)
    plt.colorbar(overlay, ax=axs[2], fraction=0.046, pad=0.04, label='P(membrane)')
    axs[2].set_title("P(membrane) overlaid on image\n(dark = high membrane probability)")
    axs[2].axis('off')

    plt.suptitle(f"Probability analysis — image {args.image_id}", fontsize=13)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRF segmentation comparison")
    parser.add_argument('--image_id', type=str, default='29',
                        help='Image index, e.g. 29 or 30')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Smoothness constant (higher = smoother regions)')
    args = parser.parse_args()

    image = plt.imread(f"data/raw/train_images/train_{args.image_id}.png")
    mask  = load_mask(args.image_id)

    # Quick sanity check — membrane should be a minority of pixels
    membrane_fraction = (mask == 0).mean()
    print(f"Membrane fraction in mask: {membrane_fraction:.3f}  (expect ~0.10–0.25)")

    # Run all three methods
    labeling_mrf,  dice_mrf,  iou_mrf, mu1, mu2      = run_mrf_intensity(image, mask, args.beta)
    labeling_unet, prob_membrane, dice_unet, iou_unet = run_unet_only(args.image_id, mask)
    labeling_comb, dice_comb, iou_comb                = run_mrf_unet(image, mask, args.image_id, args.beta)

    print(f"\nSummary for image {args.image_id}:")
    print(f"  Intensity MRF : Dice {dice_mrf:.4f}  IoU {iou_mrf:.4f}")
    print(f"  U-Net only    : Dice {dice_unet:.4f}  IoU {iou_unet:.4f}")
    print(f"  U-Net + MRF   : Dice {dice_comb:.4f}  IoU {iou_comb:.4f}")

    plot_comparison(image, mask, labeling_mrf, labeling_unet, labeling_comb,
                    dice_mrf, iou_mrf, dice_unet, iou_unet, dice_comb, iou_comb, args)

    plot_probability_analysis(image, prob_membrane, mu1, mu2, args)