import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch

def overlay_mask_on_image(image, mask, obj_id=None, random_color=False):
    """Overlay segmentation mask on the original image without using matplotlib."""
    if random_color:
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id + 1
        color = (np.array(cmap(cmap_idx)[:3]) * 255).astype(np.uint8)

    # Ensure mask is in (height, width) format
    mask = np.squeeze(mask)  # Remove singleton dimensions (e.g., (1, H, W) -> (H, W))
    
    if mask.shape[:2] != image.shape[:2]:  # Ensure mask and image match in size
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape}")

    # Convert grayscale mask to a 3-channel color mask
    mask_rgb = np.zeros_like(image, dtype=np.uint8)
    mask_rgb[mask > 0] = color  # Apply color to masked regions
    
    # Blend original image and mask overlay
    alpha = 0.6  # Transparency factor
    blended = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)

    return blended

def show_points(coords, labels, ax, marker_size=200):
    """Plot positive (green) and negative (red) points on an axis."""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    if pos_points.size > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1],
                   color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
    if neg_points.size > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1],
                   color='red', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

def save_plot(fig, save_path):
    """Save a matplotlib figure and close it."""
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def plot_first_frame(first_frame, object_points, output_dir):
    """Plot and save the first frame with overlaid label points for each object."""
    for obj_id, data in object_points.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(f"First Frame Labels for Object {obj_id}")
        ax.imshow(first_frame)
        show_points(data["points"], data["labels"], ax)
        save_plot(fig, os.path.join(output_dir, f"clicks_object_{obj_id}.jpg"))
        plt.close(fig)


def save_first_frame_segmentation(first_frame, masks_to_show, tracking_plots_dir):
    """Overlay segmentation masks on the first frame and save the images."""
    for obj_id, mask in masks_to_show.items():
        overlay_img = overlay_mask_on_image(first_frame, mask, obj_id=obj_id)
        Image.fromarray(overlay_img).save(os.path.join(tracking_plots_dir, f"segmentation_{obj_id}.jpg"))
        
        
def plot_centres3d_over_time(centers3d_over_time, output_folder):
    """Plot the object centres over time (X, Y, and Z separately) and save the figure."""
    if not centers3d_over_time:
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    for obj_id, tensor in centers3d_over_time.items():
        array = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor  # Convert tensor to NumPy if needed
        frame_indices = np.arange(array.shape[0])  # Assuming frames are sequential

        if array.ndim == 2 and array.shape[1] == 3:  # Ensure valid 3D shape
            ax1.plot(frame_indices, array[:, 1], marker='o', label=f"Object {obj_id} (X)")
            ax2.plot(frame_indices, array[:, 0], marker='o', label=f"Object {obj_id} (Y)")
            ax3.plot(frame_indices, array[:, 2], marker='o', label=f"Object {obj_id} (Z)")

    # Set labels and legends
    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("Centre X Position")
    ax1.legend()

    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Centre Y Position")
    ax2.legend()

    ax3.set_xlabel("Frame Index")
    ax3.set_ylabel("Centre Z Position (Depth)")
    ax3.legend()

    plt.suptitle("Object Centres Over Time (3D)")
    
    # Save the plot
    centre_plot_path = os.path.join(output_folder, "centres3d_over_time.jpg")
    plt.savefig(centre_plot_path, bbox_inches="tight")
    plt.close(fig)


def plot_max_distance_over_time(max_distance_over_time, output_folder, experiment):
    """Plot max distances over time (for holonomic pendulum) and save the figure."""
    if not max_distance_over_time:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for obj_id, distances in max_distance_over_time.items():
        distances = sorted(distances, key=lambda x: x[0])
        frame_indices = [d[0] for d in distances]
        max_distances = np.array([d[1] for d in distances], dtype=np.float32)
        ax.plot(frame_indices, max_distances, marker='o', label=f"Object {obj_id}")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Max Distance")
    ax.legend()
    plt.suptitle(f"Max Distance Over Time ({experiment})")
    max_distance_plot_path = os.path.join(output_folder, "max_distance_over_time.jpg")
    plt.savefig(max_distance_plot_path, bbox_inches="tight")
    plt.close(fig)
