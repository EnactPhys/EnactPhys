import torch
import numpy as np

def filter_trajectory(xyz_coords: dict, gap_threshold: float = 0.15, min_trajectory_length: float = 0.15, min_valid_percentage: float = 0.15) -> dict:
    discard_details = {}
    for obj_id, coords in xyz_coords.items():
        discard_details[obj_id] = {
            "is_discarded": False, 
            "discard_reason": None,
            "shortened_trajectory": False  # Initialize with default value and fix typo
        }
        #get only x and y coordinates
        #find sequences of valid frames(where object is tracked)
        valid_mask = ~torch.isnan(coords[:,0])
        valid_frames = torch.where(valid_mask)[0]

        valid_percentage = len(valid_frames) / len(coords)

        if valid_percentage < min_valid_percentage:
            discard_details[obj_id]["is_discarded"] = True
            discard_details[obj_id]["discard_reason"] = f"None frames too many smaller than min. valid_threshold: {min_valid_percentage}"
        
        # Find gaps between valid frame sequences
        gaps = valid_frames[1:] - valid_frames[:-1]
        min_gap_threshold = int(len(coords) * gap_threshold)
        min_length_threshold = int(len(coords) * min_trajectory_length)
        
        # Find all significant gaps
        significant_gaps = [i for i, gap in enumerate(gaps) if gap > min_gap_threshold]

        if not significant_gaps:
            continue
        # Split trajectory into segments
        segments = []        
        # Add first segment if it exists
        if significant_gaps:
            segments.append((valid_frames[0], valid_frames[significant_gaps[0]]))

        # Add middle segments
        for i in range(len(significant_gaps)-1):
            start = valid_frames[significant_gaps[i]+1]
            end = valid_frames[significant_gaps[i+1]]
            segments.append((start, end))
        
        # Add last segment if it exists
        if significant_gaps:
            segments.append((valid_frames[significant_gaps[-1]+1], valid_frames[-1]))
        
        # Find the longest segment
        segment_lengths = [(end - start + 1) for start, end in segments]
        longest_segment_idx = max(range(len(segments)), key=lambda i: segment_lengths[i])
        start_frame, end_frame = segments[longest_segment_idx]

        # Extract the longest segment
        filtered_coords = coords[start_frame:end_frame+1]
        
        # Remove any remaining NaN values
        valid_mask = ~torch.isnan(filtered_coords).any(dim=1)
        filtered_coords = filtered_coords[valid_mask]

        if len(filtered_coords) < min_length_threshold:
            discard_details[obj_id]["is_discarded"] = True
            discard_details[obj_id]["discard_reason"] = "Longest segment of object detection too short"

        else:
            discard_details[obj_id]["is_discarded"] = False
            discard_details[obj_id]["shortened_trajectory"] = True
    return discard_details