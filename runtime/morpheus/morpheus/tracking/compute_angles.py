from collections import defaultdict
import numpy as np

def get_pivot_point(masks):
    """
    Get the center of the mask, which is an intersection of the masks at each frame.
    If there is no intersection, return the most probable point.
    """
    # Compute the intersection across all frames
    intersection_mask = np.all(masks, axis=0)
    indices = np.argwhere(intersection_mask)
    is_most_probable = False
    if len(indices) > 0:
        center = np.mean(indices, axis=0)
    else:
        probability_map = np.sum(masks, axis=0)
        # Find the most probable point (i.e., the maximum value in the probability map)
        max_value = np.max(probability_map)
        most_probable_indices = np.argwhere(probability_map == max_value)
        center = np.mean(most_probable_indices, axis=0)  # Average in case of multiple max values
        is_most_probable = True
    return center, is_most_probable


def convert_to_np_array(center):
    return np.asarray([[x, y] for idx, (x, y) in sorted(center.items(), key=lambda x: x[0])])

def get_angles(centers, obj_masks):
    """Compute angles and track if the pivot point was the most probable."""
    o1, is_most_probable = get_pivot_point(obj_masks[2])
    if o1 is None:
        raise ValueError("Pivot point is None")
    
    m1, m2 = centers[2], centers[1]
    assert len(m1) == len(m2), f"Lengths of m1 ({len(m1)}) and m2 ({len(m2)}) do not match."
    
    # O2 = 2M1 - O1
    o2 = 2 * m1 - o1
    # O3 = 2M2 - O2
    o3 = 2 * m2 - o2
    # v1 = O2 - O1
    v1 = o2 - o1
    # v2 = O3 - O2
    v2 = o3 - o2
    # Angle theta1 between v1 and the y-axis
    theta1 = np.arctan2(v1[:, 1], v1[:, 0])
    # Angle theta2 between v2 and the y-axis
    theta2 = np.arctan2(v2[:, 1], v2[:, 0])
    # Angle between the two vectors
    # Duplicate o1 to match the shape of o2 and o3
    o1_expanded = np.tile(o1, (len(o2), 1))

    return np.stack([theta1, theta2], axis=0), (o1_expanded, o2, o3), is_most_probable

def get_mask_center(mask: np.ndarray):
    indices = np.argwhere(mask)
    return tuple(np.mean(indices, axis=0)) if indices.size > 0 else None

def compute_centers(masks):
    centers = defaultdict(list)
    for t in masks:
        for obj_id in masks[t]:
            center_point = get_mask_center(masks[t][obj_id].squeeze())
            if center_point is not None:
                centers[obj_id].append((t, center_point))
    centers = {obj_id: dict(centers[obj_id]) for obj_id in centers}
    return centers


def fill_in_missing_values(centers):
    # Fill in missing values in the centers
    # Get the trajectory start and end, which is the minimum and maximum time steps common to all objects
    obj2idx = {obj_id: [idx for idx in centers[obj_id]] for obj_id in centers}
    trajectory_start = max([min(obj2idx[obj_id]) for obj_id in centers])  # Latest start time
    trajectory_end = min([max(obj2idx[obj_id]) for obj_id in centers])      # Earliest end time
    common_timesteps = set(range(trajectory_start, trajectory_end + 1))
    print(f"Trajectory start: {trajectory_start}, Trajectory end: {trajectory_end}")
    # Fill in missing values by interpolating between closest time steps
    for obj_id in centers:
        for i in range(trajectory_start, trajectory_end + 1):
            if i not in obj2idx[obj_id]:
                print(f"Filling in missing value for {obj_id} at {i}")
                left_values = [x for x in obj2idx[obj_id] if x < i]
                right_values = [x for x in obj2idx[obj_id] if x > i]
                left_closest = max(left_values) if left_values else None
                right_closest = min(right_values) if right_values else None
                if left_closest is None and right_closest is None:
                    raise ValueError(f"No closest values found for trajectory")
                elif left_closest is None:
                    centers[obj_id][i] = centers[obj_id][right_closest]
                elif right_closest is None:
                    print(f"No right closest value for {obj_id} at {i}, using left closest")
                    centers[obj_id][i] = centers[obj_id][left_closest]
                else:
                    print(f"Interpolating missing value for {obj_id} at {i}")
                    interpolation_factor = (i - left_closest) / (right_closest - left_closest)
                    interpolated = (1 - interpolation_factor) * np.array(centers[obj_id][left_closest]) \
                                   + interpolation_factor * np.array(centers[obj_id][right_closest])
                    centers[obj_id][i] = interpolated.tolist()
    # Trim centers to include only the common time steps
    for obj_id in centers:
        centers[obj_id] = {t: centers[obj_id][t] for t in sorted(common_timesteps)}
    return centers, sorted(common_timesteps)

def compute_angles_mask(masks):
    """Process masks.pkl in a folder, compute angles, and store pivot point information."""
    centers = compute_centers(masks)
    centers, common_timesteps = fill_in_missing_values(centers)
    centers = {obj_id: convert_to_np_array(centers[obj_id]) for obj_id in centers}
    obj_masks = {idx: np.concatenate([masks[timestep][idx] for timestep in masks], axis=0) for idx in centers}
    
    thetas, _, is_most_probable = get_angles(centers, obj_masks)
    
    # add keys of the the common timesteps to thetas to get {timestep: {obj_id: theta}}
    thetas_dict = {timestep: {} for timestep in common_timesteps}

    for idx, t in enumerate(common_timesteps):
        thetas_dict[t][1] = thetas[0][idx]
        thetas_dict[t][2] = thetas[1][idx]
            
    pivot_json = {"pivot_point": is_most_probable}
    return thetas_dict, pivot_json