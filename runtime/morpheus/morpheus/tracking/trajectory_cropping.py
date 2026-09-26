#!/usr/bin/env python3
"""
Utility functions for trajectory cropping in the tracking pipeline.
Automatically crops trajectories for falling/bouncing/sliding experiments.
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Import matplotlib - fail if not available
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_pickle_with_torch_conversion(filepath):
    """
    Load pickle file with torch tensor conversion.
    """
    import torch
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Convert torch tensors to numpy arrays if needed
    if hasattr(data, 'detach'):
        data = data.detach().cpu().numpy()
    elif isinstance(data, list):
        converted_data = []
        for item in data:
            if item is not None and hasattr(item, 'detach'):
                converted_data.append(item.detach().cpu().numpy())
            else:
                converted_data.append(item)
        data = converted_data
    
    return data


def extract_trajectory_coordinates(pickle_data):
    """
    Extract x,y coordinates from pickle data.
    NOTE: Data is stored as [Y, X, Z] format, not [X, Y, Z]!
    """
    if pickle_data is None:
        raise ValueError("Pickle data is None")
    
    # Extract coordinates - data is stored as [Y, X, Z] not [X, Y, Z]!
    data_len = len(pickle_data)
    t = np.arange(data_len)
    y = np.array([item[0] if item is not None else np.nan for item in pickle_data])  # First element is Y!
    x = np.array([item[1] if item is not None else np.nan for item in pickle_data])  # Second element is X!
    
    # Find valid start index (where data is not NaN)
    valid_indices = np.where(~np.isnan(x))[0]
    if len(valid_indices) == 0:
        raise ValueError("No valid coordinates found in data")
        
    valid_start_idx = valid_indices[0]
    x = x[valid_start_idx:]
    y = y[valid_start_idx:]
    t = t[valid_start_idx:]
    
    # Interpolate missing values
    df = pd.DataFrame({'t': t, 'x': x, 'y': y}).interpolate(method='linear')
    
    # Return the y coordinates (we focus on y for falling detection)
    y_coords = df['y'].to_numpy()
    x_coords = df['x'].to_numpy()
    
    # Remove any remaining NaN values
    valid_mask = ~np.isnan(y_coords)
    y_coords = y_coords[valid_mask]
    x_coords = x_coords[valid_mask]
    
    return x_coords, y_coords


def _get_stopping_description(experiment_name):
    """Get description text for stopping criteria based on experiment type."""
    if 'falling' in experiment_name.lower():
        return "FALLING: Stop when approaching landing point (max Y)."
    elif 'bouncing' in experiment_name.lower():
        return "BOUNCING: Use valley detection to find all local maxima (valleys) after ≥20 frames, return first valley within 10% of starting y value (first frame), can be higher or lower."
    elif 'sliding' in experiment_name.lower():
        return "SLIDING: Stop when center approaches area (x >= 1000, y >= 750)."
    else:
        return f"{experiment_name.upper()}: Default crop - keep all frames (no specific stopping criteria)."


def find_crop_frame(y_coords, experiment_type, min_frames=20, x_coords=None):
    """
    Find the frame where to crop the video based on experiment type.
    
    FALLING: Find when object reaches 90% of landing point (max Y)
    BOUNCING: Use valley detection to find all local maxima (valleys) after min_frames,
              return first valley within 10% of starting y value (first frame)
    SLIDING: Stop when center approaches area (x >= 1000, y >= 750)
    OTHER: Default crop - keep all frames (returns last frame)
    
    Since higher Y = lower on screen.
    """
    if y_coords is None:
        raise ValueError("Y coordinates are None")
    if len(y_coords) < min_frames:
        raise ValueError(f"Not enough frames: {len(y_coords)} < {min_frames}")
    
    min_y = np.min(y_coords)  # Top of trajectory (starting point)
    max_y = np.max(y_coords)  # Bottom of trajectory (lowest point)
    
    if 'falling' in experiment_type.lower():
        # FALLING LOGIC: Crop when approaching landing point (max Y)
        threshold = max_y - 0.1 * (max_y - min_y)  # 10% from bottom (landing point)
        
        # Find first frame where y is within 10% of maximum (bottom/landing point)
        for frame_idx, y in enumerate(y_coords):
            if y >= threshold and frame_idx >= min_frames:
                return frame_idx
        
        # If never reaches threshold after min_frames, return last frame
        return len(y_coords) - 1
                
    elif 'bouncing' in experiment_type.lower():
        # BOUNCING LOGIC: Use valley detection to find when object returns to starting height
        # Find all valleys (local maxima in Y = lowest points in trajectory) after min_frames
        # Return first valley within 10% of starting y value
        
        # Get the starting y value (first frame)
        y_start = y_coords[0]
        
        # Calculate threshold range: within 10% of starting y value, can be higher or lower
        threshold_low = y_start * 0.9   # 10% below starting (Y smaller = higher on screen)
        threshold_high = y_start * 1.1  # 10% above starting (Y larger = lower on screen)
        
        # Find all valleys (local maxima in Y coordinates)
        # A valley is where Y is greater than both neighbors (object at lowest point)
        valleys = []
        for i in range(min_frames, len(y_coords) - 1):
            # Check if this is a local maximum (valley in trajectory)
            # Y should be greater than immediate neighbors
            if y_coords[i] > y_coords[i - 1] and y_coords[i] > y_coords[i + 1]:
                valleys.append(i)
        
        # Find first valley within threshold range
        for valley_idx in valleys:
            if threshold_low <= y_coords[valley_idx] <= threshold_high:
                return valley_idx
        
        # If no valley found within threshold, return last frame
        return len(y_coords) - 1
    
    elif 'sliding' in experiment_type.lower():
        # SLIDING LOGIC: Stop when center approaches area (x >= 1000, y >= 750)
        if x_coords is None:
            raise ValueError("X coordinates are required for sliding_book experiments")
        
        if len(x_coords) != len(y_coords):
            raise ValueError(f"X and Y coordinates must have same length: {len(x_coords)} != {len(y_coords)}")
        
        # Area boundaries: x >= 1000, y >= 750
        x_threshold = 1000
        y_threshold = 750
        
        # Find first frame after min_frames where center enters the area
        for frame_idx in range(min_frames, len(y_coords)):
            if x_coords[frame_idx] >= x_threshold and y_coords[frame_idx] >= y_threshold:
                return frame_idx
        
        # If never enters the area, return last frame
        return len(y_coords) - 1
    
    else:
        # Default for unknown experiment types: keep all frames
        return len(y_coords) - 1


def create_cropped_trajectory_sanity_plot(x_full, y_full, x_cropped, y_cropped, crop_frame, output_png_file, title, experiment_info):
    """Create a sanity check plot comparing original vs cropped trajectory."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left plot: Time series comparison
    frames_full = list(range(len(y_full)))
    frames_cropped = list(range(len(y_cropped)))
    
    ax1.plot(frames_full, y_full, 'b-', linewidth=2, label='Original trajectory', alpha=0.7)
    ax1.plot(frames_cropped, y_cropped, 'r-', linewidth=3, label='Cropped trajectory (for scoring)', alpha=0.9)
    ax1.axvline(x=crop_frame, color='red', linestyle='--', linewidth=2, label=f'Crop point (frame {crop_frame})', alpha=0.8)
    
    ax1.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Vertical Position (pixels - top to bottom)', fontsize=12, fontweight='bold')
    ax1.set_title('Trajectory Comparison: Original vs Cropped', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()  # Reverse Y-axis so lower Y values (top of screen) appear at top
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: 2D trajectory comparison (square plot)
    if x_full is not None and x_cropped is not None:
        ax2.plot(x_full, y_full, 'b-', linewidth=2, label='Original flight path', alpha=0.7)
        ax2.plot(x_cropped, y_cropped, 'r-', linewidth=3, label='Cropped flight path', alpha=0.9)
        ax2.scatter(x_full[0], y_full[0], color='green', s=100, label='Start point', zorder=5)
        ax2.scatter(x_cropped[-1], y_cropped[-1], color='red', s=150, label=f'Crop endpoint (frame {crop_frame})', zorder=5, marker='X')
        
        ax2.set_xlabel('Horizontal Position (pixels)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Vertical Position (pixels - top to bottom)', fontsize=12, fontweight='bold')
        ax2.set_title('2D Flight Path: Original vs Cropped', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Make it square like the main plot
        all_x = list(x_full) + list(x_cropped)
        all_y = list(y_full) + list(y_cropped)
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        max_range = max(x_range, y_range)
        
        x_center = (max(all_x) + min(all_x)) / 2
        y_center = (max(all_y) + min(all_y)) / 2
        
        padding = max_range * 0.1
        ax2.set_xlim(x_center - max_range/2 - padding, x_center + max_range/2 + padding)
        ax2.set_ylim(y_center - max_range/2 - padding, y_center + max_range/2 + padding)
        ax2.set_aspect('equal', adjustable='box')
        ax2.invert_yaxis()  # Reverse Y-axis so lower Y values (top of screen) appear at top
    
    # Add info box
    textstr = f'''CROPPED TRAJECTORY FOR SCORING:
Model: {experiment_info.get('model', 'Unknown')}
Experiment: {experiment_info.get('experiment', 'Unknown')}
Original frames: {len(y_full)}
Cropped frames: {len(y_cropped)} (0 to {crop_frame})
Cropped data saved as: centres3d_obj_1_cropped.pkl
Ready for scoring pipeline!'''
    
    props = dict(boxstyle='round,pad=0.7', facecolor='lightblue', alpha=0.9, 
                 edgecolor='darkblue', linewidth=2)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_png_file, dpi=200, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.close()
    
    return output_png_file


def should_crop_experiment(experiment_name):
    """
    Check if an experiment should be cropped based on its name.
    Returns True for falling, bouncing, or sliding experiments (case-insensitive).
    """
    exp_lower = experiment_name.lower()
    return 'falling' in exp_lower or 'bouncing' in exp_lower or 'sliding' in exp_lower


def crop_trajectory_if_needed(output_folder, experiment_name, info):
    """
    Crop trajectory for falling/bouncing/sliding experiments if needed.
    Skips cropping for real-world videos.
    
    Args:
        output_folder: Path to the tracking output folder containing centres3d_obj_1.pkl
        experiment_name: Name of the experiment
        info: Dictionary containing experiment metadata (model, conditioning, prompt_type, etc.)
    
    Returns:
        True if trajectory was cropped, False otherwise
    """
    # Check if this experiment should be cropped
    if not should_crop_experiment(experiment_name):
        return False
    
    # Skip cropping for real-world videos
    if info.get('model') == 'real-world':
        return False
    
    # Check if pickle file exists
    pickle_path = os.path.join(output_folder, 'centres3d_obj_1.pkl')
    if not os.path.exists(pickle_path):
        print(f"Warning: No pickle file found for cropping: {pickle_path}")
        return False
    
    # Load trajectory data
    pickle_data = load_pickle_with_torch_conversion(pickle_path)
    
    # Extract coordinates
    x_coords, y_coords = extract_trajectory_coordinates(pickle_data)
    
    if len(y_coords) == 0:
        print(f"Warning: No coordinate points extracted for {experiment_name}")
        return False
    
    # Find crop frame
    crop_frame = find_crop_frame(y_coords, experiment_name, min_frames=20, x_coords=x_coords)
    
    # Handle None case (shouldn't happen, but be safe)
    if crop_frame is None or not isinstance(crop_frame, (int, np.integer)):
        return False
    
    # Check if a crop was actually made (crop_frame is not the last frame)
    if crop_frame >= len(y_coords) - 1:
        return False
    
    print(f"  Cropping trajectory: {crop_frame}/{len(y_coords)} frames", flush=True)
    
    # Crop the original pickle data to only include frames up to crop_frame
    cropped_data = pickle_data[:crop_frame + 1]  # +1 to include the crop frame itself
    
    # Save as centres3d_obj_1_cropped.pkl
    cropped_pickle_path = os.path.join(output_folder, 'centres3d_obj_1_cropped.pkl')
    with open(cropped_pickle_path, 'wb') as f:
        pickle.dump(cropped_data, f)
    
    # Extract coordinates for sanity check plot
    x_coords_full, y_coords_full = extract_trajectory_coordinates(pickle_data)
    x_coords_cropped, y_coords_cropped = extract_trajectory_coordinates(cropped_data)
    
    # Create sanity check plot showing original vs cropped
    sanity_plot_path = os.path.join(output_folder, 'cropped_trajectory_sanity_check.png')
    
    # Extract info for plot
    model_name = info.get('model', 'Unknown')
    prompt_type = info.get('prompt_type', 'Unknown')
    video_dir = os.path.basename(output_folder)
    
    create_cropped_trajectory_sanity_plot(
        x_coords_full, y_coords_full, 
        x_coords_cropped, y_coords_cropped, 
        crop_frame, sanity_plot_path, 
        f"{model_name} - {experiment_name} - {video_dir}", 
        {'model': model_name, 'experiment': experiment_name, 'prompt_type': prompt_type}
    )
    
    # Calculate thresholds for result data
    if 'falling' in experiment_name.lower():
        threshold_value = float(np.max(y_coords) - 0.1 * (np.max(y_coords) - np.min(y_coords)))
    elif 'bouncing' in experiment_name.lower():
        # For bouncing, use starting y value (first frame)
        y_start = y_coords[0]
        threshold_value = {
            "threshold_low": float(y_start * 0.9),
            "threshold_high": float(y_start * 1.1)
        }
    elif 'sliding' in experiment_name.lower():
        # For sliding_book, store the area boundaries
        threshold_value = {
            "x_threshold": 1000,
            "y_threshold": 750,
            "description": "Stop when center enters area: x >= 1000, y >= 750"
        }
    else:
        # For other types, no specific threshold
        threshold_value = None
    
    # Create comprehensive result data
    result_data = {
        "experiment": experiment_name,
        "model": info.get('model', 'Unknown'),
        "conditioning": info.get('conditioning', 'Unknown'),
        "prompt_type": info.get('prompt_type', 'Unknown'),
        "video_dir": video_dir,
        "total_frames": len(y_coords),
        "crop_frame": int(crop_frame),
        "N_frame": int(crop_frame),  # Last frame to keep (stopping point)
        "stopping_criteria": {
            "method": "physics_based_stopping",
            "experiment_type": experiment_name,
            "y_min": float(np.min(y_coords)),
            "y_max": float(np.max(y_coords)),
            "threshold_10_percent": threshold_value,
            "min_frames_required": 20,
            "description": _get_stopping_description(experiment_name)
        },
        "trajectory_data": {
            "data_source": "Tracking data from centres3d_obj_1.pkl",
            "y_coordinates": y_coords.tolist(),
            "x_coordinates": x_coords.tolist() if x_coords is not None else None,
            "note": "Trajectory data extracted from SAM2 tracking results"
        },
        "processing_info": {
            "processed_date": datetime.now().strftime("%Y-%m-%d"),
            "script_version": "integrated_tracking_pipeline",
            "pickle_file_path": pickle_path,
            "note": "Uses trajectory data from SAM2 tracking - integrated into tracking pipeline"
        }
    }
    
    # Save JSON result
    json_output = os.path.join(output_folder, 'falling_object_stopping_analysis.json')
    with open(json_output, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    return True

