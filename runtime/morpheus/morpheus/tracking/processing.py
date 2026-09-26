import json
import re
from scipy.spatial.distance import pdist
from skimage.measure import label, regionprops
import numpy as np
import os
from .plotting import overlay_mask_on_image
from .compute_angles import compute_angles_mask
from ..device import get_device, hf_pipeline_device
from PIL import Image
import pickle
import torch
from transformers import pipeline
from torchvision import transforms

class DepthProcessorV2:
    def __init__(self, device=None):
        self.device = device if device is not None else get_device()
        # Using the correct model identifier that works with transformers pipeline
        # The V2 models are available under LiheYoung namespace
        self.depth_anything = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Large-hf",
            device=hf_pipeline_device(self.device)
        )
        self.to_tensor = transforms.ToTensor()

    def predict_depth(self, image):
        depth_output = self.depth_anything(image)
        return self.to_tensor(depth_output["depth"]).squeeze().to(self.device)

class DepthProcessor:
    def __init__(self, encoder='vitl', device=None):
        self.mapper = {"vits": "small", "vitb": "base", "vitl": "large"}
        self.device = device if device is not None else get_device()
        self.depth_anything = pipeline(
            task="depth-estimation",
            model=f"nielsr/depth-anything-{self.mapper[encoder]}",
            device=hf_pipeline_device(self.device)
        )
        self.to_tensor = transforms.ToTensor()
    
    def predict_depth(self, image):
        depth_output = self.depth_anything(image)
        return self.to_tensor(depth_output["depth"]).squeeze().to(self.device)
    
def transform_keys(data):
    """Recursively transform keys to retain only 'video_X' format."""
    new_data = {}
    for key, value in data.items():
        match = re.match(r"(video_\d+)", key)
        new_key = match.group(1) if match else key
        if new_key in new_data and isinstance(new_data[new_key], dict) and isinstance(value, dict):
            new_data[new_key].update(value)
        else:
            new_data[new_key] = transform_keys(value) if isinstance(value, dict) else value
    return new_data

def load_labels(labels_path):
    """Load and transform labels from a JSON file."""
    with open(labels_path, "r") as f:
        data = json.load(f)
    return transform_keys(data)

def compute_mask_properties(mask, experiment):
    """
    Compute the centroid (centre) of a mask, and—for holonomic pendulum—its
    maximum distance (using regionprops' major_axis_length as a proxy).
    """
    mask = np.squeeze(mask)
    mask_bool = mask > 0
    indices = np.argwhere(mask_bool)
    centre = tuple(np.mean(indices, axis=0)) if indices.size > 0 else (None, None)
    max_distance = np.nan
    
    # we only need max distance for holonic pendulum and second object
    if indices.size > 0:
        # for holonomic pendulum, we use the maximum distance between points, as the major axis length was no working well for the very thin maks
        if experiment == "holonomic_pendulum":
            max_distance = np.max(pdist(indices, metric='euclidean')) if indices.shape[0] > 1 else 0.0
        # for double pendulum, we use the major axis length as a proxy for the maximum distance
        elif experiment == "double_pendulum":
            labeled = label(mask_bool)
            props = regionprops(labeled)
            max_distance = props[0].major_axis_length if props else 0.0
    return centre, max_distance

def process_object_points(label_info):
    """Convert label info into a dictionary of object points and labels."""
    object_points = {}
    for obj_key, obj_data in label_info.items():
        obj_id = int(obj_key.split("_")[-1])
        pos = np.array(obj_data.get("positive", []), dtype=np.float32)
        neg = np.array(obj_data.get("negative", []), dtype=np.float32)
        pts, labs = [], []
        if pos.size > 0:
            pts.append(pos)
            labs.append(np.ones(len(pos), dtype=int))
        if neg.size > 0:
            pts.append(neg)
            labs.append(np.zeros(len(neg), dtype=int))
        if pts:
            object_points[obj_id] = {
                "points": np.concatenate(pts, axis=0),
                "labels": np.concatenate(labs, axis=0)
            }
    return object_points

def to_array(x, size):
    #{obj_id: [(frame_idx, (y, x)), ...]} -> {obj_id: np.array([[y, x], ...])} and None for empty
    n_values = len(x[0][1])
    arr = np.full((size, n_values), None, dtype=np.float32)
    for frame_idx, values in x:
        if 0 <= frame_idx < size:
            arr[frame_idx] = list(values)
        else:
            raise ValueError(f"Frame index {frame_idx} is out of bounds for array of size {size}")
    return arr

def process_segmentation_frames(frames_dir, video_segments, tracking_plots_dir, info):
    """
    Process and save segmentation overlays for every 10th frame.
    Computes centres and, if applicable, max distances over time.
    """
    # Sort frame names numerically.
    frame_names = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")],
        key=lambda p: int(os.path.splitext(p)[0])
    )
    centers_over_time = {}       # {obj_id: [(frame_idx, (y, x)), ...]}
    masks_over_time = {}         # {frame_idx: {obj_id: mask}}
    max_distance_over_time = {}  # {obj_id: [max_distances]}

    experiment = info["experiment"]
    
    # Map experiment names for cosmos-transferred videos to base experiment types
    EXPERIMENTS_MATCH_LABELS = {
        "falling_ball_basketball": "falling_ball",
        "falling_ball_baseball": "falling_ball",
        "falling_ball_pingpong": "falling_ball",
        "projectile_beach_volleyball": "projectile",
        "projectile_kitchen_lemon": "projectile",
        "projectile_pinecone": "projectile",
        "holonomic_pendulum_pendulum": "holonomic_pendulum",
        "holonomic_pendulum_wrecking_ball": "holonomic_pendulum",
        "rolling_full_can_aluminum_barrel": "rolling_full_can",
        "rolling_full_can_barrel": "rolling_full_can",
        "rolling_full_can_beverage_can": "rolling_full_can",
        "sliding_book_brick": "sliding_book",
        "sliding_book_delivery_truck_crate": "sliding_book",
        "sliding_book_inclined_shelf": "sliding_book",
    }
    
    # Use mapped experiment for processing logic
    base_experiment = EXPERIMENTS_MATCH_LABELS.get(experiment, experiment)
    
    # Ensure output directory for frames exists.
    frames_output = os.path.join(tracking_plots_dir, "frames")
    os.makedirs(frames_output, exist_ok=True)

    save_every = 10
    
    for out_frame_idx in range(0, len(frame_names)):
        frame_path = os.path.join(frames_dir, frame_names[out_frame_idx])
        frame_img = np.array(Image.open(frame_path))

        if out_frame_idx in video_segments:
            masks_over_time[out_frame_idx] = {}
            for obj_id, mask in video_segments[out_frame_idx].items():
                frame_img = overlay_mask_on_image(frame_img, mask, obj_id=obj_id)
                centre, max_distance = compute_mask_properties(mask, base_experiment)
                masks_over_time[out_frame_idx][obj_id] = mask
                if centre[0] is not None:
                    centers_over_time.setdefault(obj_id, []).append((out_frame_idx, centre))
                if base_experiment in ["holonomic_pendulum", "double_pendulum"]:
                    max_distance_over_time.setdefault(obj_id, []).append(max_distance)

        if out_frame_idx % save_every == 0:
            save_path = os.path.join(frames_output, f"frame_{out_frame_idx:05d}_segmentation.jpg")
            Image.fromarray(frame_img).save(save_path)
    
    thetas, is_most_probable = None, None     
    
    # after all masks are processed, we can compute angles for double pendulum
    if base_experiment == "double_pendulum":
        # Compute angles, also returns wheter the pivot point was the most probable on or could actually be computed
        # Also we have new centres that only have common timesteps
        thetas,  is_most_probable = compute_angles_mask(masks_over_time)
    return centers_over_time, masks_over_time, max_distance_over_time, thetas, is_most_probable

def process_and_save_depth_v2_frames(frames_dir, output_folder, depth_processor):
    """
    Create depths for each experiment with DepthAnything-v2 this is used as a modality input to COSMOS-transfer.
    """
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(x.split('.')[0])
    )
    depth_over_time = {}
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        image = Image.open(frame_path).convert('RGB')
        depth = depth_processor.predict_depth(image)
        depth_over_time[frame_file] = depth
    
    depth_tensor = torch.stack(list(depth_over_time.values()), dim=0)
    #save the depth tensor as a pickle file
    print(f"Saving depth.pkl with shape {depth_tensor.shape}")
    with open(os.path.join(output_folder, "depth.pkl"), "wb") as f:
        pickle.dump(depth_tensor, f)

def process_depth_frames(frames_dir, masks_over_time, centers_over_time, depth_processor):
    """
    Process depth for each frame using DepthAnything and compute average depth for each object's mask.
    Returns depth_over_time dictionary mapping object IDs to frame-wise depth values.
    """
    depth_over_time = {}

    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(x.split('.')[0])
    )

    max_frame_idx = len(frame_files)
    objects_names = masks_over_time[0].keys()
    depth_over_time = {obj_name: np.nan * np.ones(max_frame_idx) for obj_name in objects_names}
    
    for frame_idx, frame_file in enumerate(frame_files):
        assert f"{frame_idx:05d}.jpg" == frame_file
        
        if frame_idx in masks_over_time:
            frame_path = os.path.join(frames_dir, frame_file)
            image = Image.open(frame_path).convert('RGB')
            depth = depth_processor.predict_depth(image)
        
        
            # we also need to check if the masks are in the common timesteps
            for obj_id, mask in masks_over_time[frame_idx].items():
                mask_tensor = torch.from_numpy(mask).to(depth.device)
                masked_depth = depth * mask_tensor
                avg_depth = masked_depth.sum() / (mask_tensor.sum() + 1e-6)
                depth_over_time[obj_id][frame_idx] = avg_depth.item()
        
    centers3d_over_time = {}

    for key in centers_over_time.keys():
        txy = centers_over_time[key]
        z_over_time = depth_over_time[key]
        xy_array = to_array(txy, z_over_time.shape[0])
        xyz = np.concatenate([xy_array, z_over_time[:, None]], axis=1, dtype=np.float32)
        centers3d_over_time[key] = torch.tensor(xyz)
        assert centers3d_over_time[key].shape[1] == 3, "Should be xyz"

    return centers3d_over_time

def save_segmentation_masks(output_folder, masks_over_time):
    """Save computed segmentation masks as pickle files."""
    mask_over_time = {}
    for frame_idx, frame_masks in masks_over_time.items():
        mask_over_time[frame_idx] = {}
        for obj_id, mask in frame_masks.items():
            mask_over_time[frame_idx][obj_id] = mask
    
    #stack masks per frame
    mask_list = []
    for frame_idx in range(len(mask_over_time)):
        mask_list.append([mask_over_time[frame_idx][obj_id].squeeze(0) for obj_id in mask_over_time[frame_idx]])
    
    masks_np = np.stack(mask_list, axis=0)
    masks_tensor = torch.from_numpy(masks_np).to(get_device())

    for obj_id in range(masks_tensor.shape[1]):
        with open(os.path.join(output_folder, f"mask_obj_{obj_id}.pkl"), "wb") as f:
            print(f"Saving mask_obj_{obj_id}.pkl with shape {masks_tensor[:, obj_id].shape}")
            pickle.dump(masks_tensor[:, obj_id], f)
   

def save_tracking_data(output_folder, centers3d_over_time, max_distance_over_time, thetas):
    """Save computed centres, masks, and (if applicable) max distances as pickle files."""
    for obj_id, center3d_data in centers3d_over_time.items():
        print(f"Saving centres3d_obj_{obj_id}.pkl")
        
        with open(os.path.join(output_folder, f"centres3d_obj_{obj_id}.pkl"), "wb") as f:
            pickle.dump(center3d_data, f)
        
    if max_distance_over_time:
        print("Saving max_distance.pkl")
        with open(os.path.join(output_folder, "max_distance.pkl"), "wb") as f:
            pickle.dump(max_distance_over_time, f)

    if thetas is not None:
        print("Saving angles.pkl")
        with open(os.path.join(output_folder, "angles.pkl"), "wb") as f:
            pickle.dump(thetas, f)
            
def load_centers3d_over_time(output_folder):
    """Load the centers3d_over_time dictionary from the output folder."""
    centers3d_over_time = {}
    centers_obj_idx = {'centres3d_obj_1.pkl': 1, 'centres3d_obj_2.pkl': 2}
    for files in os.listdir(output_folder):
        if "centres3d_obj_1" in files or "centres3d_obj_2" in files:
            with open(os.path.join(output_folder, files), "rb") as f:
                centers3d_over_time[centers_obj_idx[files]] = pickle.load(f)
    return centers3d_over_time