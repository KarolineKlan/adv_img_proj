import argparse
import numpy as np
import matplotlib.pyplot as plt
import maxflow
from scipy.signal import find_peaks
from evaluate import dice_score, iou_score

def run_segmentation(args):
    img_path = f"data/raw/train_images/train_{args.image_id}.png"
    msk_path = f"data/raw/train_labels/labels_{args.image_id}.png"
        
    image = plt.imread(img_path)
    mask = plt.imread(msk_path) 

    N, M = image.shape
    d = image.flatten()

    # 2. Histogram Analysis to find mu1 and mu2
    hist, bin_edges = np.histogram(d, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peaks, _ = find_peaks(hist, distance=5)
    
    # Sort peaks by intensity to distinguish background from foreground
    mu_values = np.sort(bin_centers[peaks])
    
    if len(mu_values) < 2:
        print("Error: Could not find at least two distinct peaks in histogram.")
        return
        
    mu1 = mu_values[0]  # Darker (Background)
    mu2 = mu_values[-1] # Lighter (Foreground)
    print(f"Detected Means - Background: {mu1:.2f}, Foreground: {mu2:.2f}")

    # 3. Build the Graph
    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((N, M))

    # Add 2D Smoothing (Horizontal and Vertical)
    # This prevents the horizontal line artifacts
    struct_h = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    struct_v = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    
    g.add_grid_edges(nodes, args.beta, structure=struct_h, symmetric=True)
    g.add_grid_edges(nodes, args.beta, structure=struct_v, symmetric=True)

    # 4. Set Terminal Edges (The Data Term)
    # Source (connected to mu2) and Sink (connected to mu1)
    cap_source = (image - mu1)**2
    cap_sink = (image - mu2)**2
    g.add_grid_tedges(nodes, cap_source, cap_sink)

    # 5. Solve and Extract Result
    g.maxflow()
    labeling_image = g.get_grid_segments(nodes).astype(np.uint8)
    labeling_image = 1 - labeling_image  # Invert: 1 for foreground, 0 for background

    # 6. Evaluation
    dice = dice_score(labeling_image, mask)
    iou = iou_score(labeling_image, mask)
    print(f"Results for ID {args.image_id}: Dice={dice:.4f}, IoU={iou:.4f}")

    overlay_image = image.copy()
    overlay_image = np.stack([overlay_image]*3, axis=-1)  # Convert to RGB for visualization
    overlay_image[mask == 1] = [255, 0, 0]  # Red for true mask
    overlay_image[labeling_image == 1] += [0, 255, 0]  # Green for predicted mask (overlaps will be yellow)


    # 7. Visualization
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title(f"Original (ID: {args.image_id})")
    
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Ground Truth Mask")
    
    axs[2].imshow(labeling_image, cmap='gray')
    axs[2].set_title(f"Graph Cut (beta={args.beta})")

    axs[3].imshow(overlay_image)
    axs[3].set_title("Overlay (Red=True, Green=Predicted)")

    
    for ax in axs:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Graph Cut Image Segmentation")
    
    parser.add_argument('--image_id', type=str, default='30', 
                        help='ID number for image and mask files')
    parser.add_argument('--beta', type=float, default=0.0001, 
                        help='Smoothing constant (higher = smoother regions)')
    
    args = parser.parse_args()
    run_segmentation(args)