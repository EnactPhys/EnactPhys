import os
import cv2
import json
import numpy as np
import torch
# ----- CONFIGURATION (used only for processing functions) -----
# NOTE: The target (real-world) resolution is fixed at 1280x1024.
REAL_WORLD_RESOLUTION = (1280, 1024)


COSMOS_SCALE_FACTOR=0.6875

# Valid image extensions.
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Metadata keywords (all matching is case-insensitive)
CONDITIONING_KEYWORDS = {
    "single_frame_conditioning": "single_frame_conditioning",
    "multi_frame_conditioning": "multi_frame_conditioning",
    "keyframe_interpolation": "keyframe_interpolation",
}
MODEL_KEYWORDS = {
    "kling-turbo": "Kling-Turbo",
    "wan": "WAN-2.1",
    "cosmos-predict2": "COSMOS-predict2",
    "cosmos-predict1": "COSMOS-predict1",
    # "cosmos": "COSMOS", #do not mix the order it is important 
    "ltx": "LTX",
    "pyramid-flow": "PyramidalFlow",
    "cogvideo": "CogVideo",
    "veo3-fast": "Veo3-fast",
    "veo3": "Veo3"
}
ENHANCEMENT_KEYWORDS = {
    "plain": "plain",
    "regular": "plain",
    "enhanced": "enhanced",
    "glm4" : "enhanced"
}
EXPERIMENT_KEYWORDS = [
    "falling_ball",
    "projectile",
    "non_holonomic_pendulum",
    "holonomic_pendulum",
    "double_pendulum",
    "bouncing_ball",
    "rolling_empty_can",
    "rolling_full_can",
    "rolling_orange",
    "sliding_book",
    "falling_apple",
    "falling_marker",
    "falling_tape",
    "spring",
    "collision_equal",
    "collision_small_hits_big",
    "collision_big_hits_small"
]

CROP_SIZE_VEO3 = (900, 720)  # (width, height)

# For Kling-Turbo we empirically determined that the model outputs 1920x1080 with
# vertical content and black side bars. The effective content area is 1350x1080
# (crop 285px from left and right), which we then resize to 1280x1024.
CROP_SIZE_KLING_TURBO = (1350, 1080)  # (width, height)

def cosmos_postprocesing_inverse(frames):
    """
    Postprocesses the video for the Cosmos model by applying inverse operations.
    
    Inverse of the Cosmos preprocessing:
    1. Remove black padding (center crop from 1280x704 to 880x704)
    2. Scale up by inverse factor (1/0.6875 ≈ 1.4545) to get back to 1280x1024
    
    Args:
        frames: Input frames - can be torch.Tensor or numpy array
               Expected shape: (T, H, W, C) for batch or (H, W, C) for single frame
               For torch tensor: (C, T, H, W) or (T, H, W, C) for batch, (C, H, W) for single
               Expected input resolution: 1280x704
               
    Returns:
        opencv_frames: Frames in OpenCV-compatible format (numpy array, uint8, BGR)
                      Shape: (T, H, W, C) for batch or (H, W, C) for single frame
                      where H=1024, W=1280
    """
    # Convert to numpy if torch tensor
    if isinstance(frames, torch.Tensor):
        # Handle different tensor formats
        if frames.dim() == 4 and frames.shape[0] == 3:  # (C, T, H, W)
            frames = frames.permute(1, 2, 3, 0)  # (T, H, W, C)
        elif frames.dim() == 3 and frames.shape[0] == 3:  # (C, H, W) - single frame
            frames = frames.permute(1, 2, 0)  # (H, W, C)
        frames = frames.cpu().numpy()
    
    # Handle single frame case - convert to batch of 1 frame
    single_frame = False
    if frames.ndim == 3:
        single_frame = True
        frames = frames[np.newaxis, ...]  # Add batch dimension: (H, W, C) -> (1, H, W, C)
    
    # Ensure frames is in correct format
    if frames.ndim != 4:
        raise ValueError(f"Expected 3D or 4D frames array, got shape: {frames.shape}")
    
    T, H, W, C = frames.shape
    
    # Expected input resolution from Cosmos processing
    if H != 704 or W != 1280:
        print(f"Warning: Expected input size 1280x704, got {W}x{H}")
    
    # Process each frame
    processed_frames = []
    
    for i in range(T):
        frame = frames[i]
        
        # Ensure frame is in uint8 format and proper range
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:  # Assume normalized [0,1]
                frame = (frame * 255).astype(np.uint8)
            elif frame.max() <= 255:  # Already in [0,255] range
                frame = frame.astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Step 1: Remove black padding (center crop from 1280x704 to 880x704)
        # The original scaling was: 1280 * 0.6875 = 880, 1024 * 0.6875 = 703.36 ≈ 704
        # So we need to crop to 880x704
        crop_w = int(1280 * COSMOS_SCALE_FACTOR)  # 880
        crop_h = 704  # Height was already at target
        
        # Center crop
        start_x = (W - crop_w) // 2
        end_x = start_x + crop_w
        start_y = (H - crop_h) // 2 
        end_y = start_y + crop_h
        
        cropped_frame = frame[start_y:end_y, start_x:end_x]
        
        # Step 2: Scale up by inverse factor to get back to 1280x1024
        target_w, target_h = REAL_WORLD_RESOLUTION  # 1280x1024
        
        resized_frame = cv2.resize(cropped_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        processed_frames.append(resized_frame)
    
    # Convert list to numpy array
    opencv_frames = np.array(processed_frames)
    
    # If input was a single frame, return single frame
    if single_frame:
        return opencv_frames[0]
    
    return opencv_frames


def extract_metadata_from_path(rel_path):
    """
    From the relative path (from the input folder), extract:
      generation, model, enhancement, and experiment.
    Returns a tuple: (generation, model, enhancement, experiment).
    """
    lower_path = rel_path.lower()
    conditioning = next((mapped for key, mapped in CONDITIONING_KEYWORDS.items() if key in lower_path), None)
    model = next((mapped for key, mapped in MODEL_KEYWORDS.items() if key in lower_path), None)
    enhancement = next((mapped for key, mapped in ENHANCEMENT_KEYWORDS.items() if key in lower_path), None)
    experiment = next((exp for exp in EXPERIMENT_KEYWORDS if exp in lower_path), None)
    # video_num = rel_path.split("/")[4] # i did this for LTX, but this does not with PyramidFlow/Cogvideo
        
    if not enhancement:
        enhancement = "plain" 

    # make a clear print statement
    # print("Conditioning: ", conditioning)
    # print("Model: ", model)
    # print("Enhancement: ", enhancement)
    # print("Experiment: ", experiment)
    # print("Video Number: ", video_num)

    return conditioning, model, enhancement, experiment # video_num

def create_video_from_frames(frames_folder, output_video_path, fps=30):
    """
    Given a folder containing frames (assumed sorted alphanumerically),
    create an MP4 video using OpenCV's VideoWriter.
    """
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.lower().endswith(".jpg")])
    if not frame_files:
        print(f"No frames found in {frames_folder} to create video.")
        return

    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    if first_frame is None:
        print(f"Error reading first frame in {frames_folder}")
        return
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error reading frame {frame_path}")
            continue
        out_video.write(frame)
    out_video.release()
    # print(f"Created video: {output_video_path}")



def _center_crop_resize(frame, crop_size, out_size, upscale=True):
    """Center-crop to crop_size, then resize to out_size. Optionally upscale first if needed."""
    crop_w, crop_h = crop_size
    h, w = frame.shape[:2]

    # Upscale to ensure the crop fits
    if upscale and (w < crop_w or h < crop_h):
        scale = max(crop_w / w, crop_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = frame.shape[:2]

    # Center crop
    x0 = max((w - crop_w) // 2, 0)
    y0 = max((h - crop_h) // 2, 0)
    crop = frame[y0:y0 + crop_h, x0:x0 + crop_w]

    # If rounding made it 1px off, pad
    if crop.shape[1] != crop_w or crop.shape[0] != crop_h:
        canvas = np.zeros((crop_h, crop_w, 3), dtype=frame.dtype)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        crop = canvas

    # Final resize
    interp = cv2.INTER_AREA if (out_size[0] < crop_w or out_size[1] < crop_h) else cv2.INTER_LINEAR
    out = cv2.resize(crop, out_size, interpolation=interp)
    return out


def transform_frame(frame, transformation):
    """
    Applies the inverse transformation on a single frame according to the model.
    
    Inverse operations:
      - "CogVideo": Assumes source size 960x768; scales up uniformly (factor ≈1.3333) to 1280x1024.
      - "LTX": Assumes source size 960x736; pads 16 pixels on top and bottom (to 960x768),
               then resizes to 1280x1024.
      - "Pyramidal": Assumes source size 1280x768; scales height by 1.3333 to 1024 (width remains).
      - "COSMOS": Assumes source size 1280x704; scales height by ~1.4545 to 1024 (width remains).
    """
    final_size = REAL_WORLD_RESOLUTION
    if transformation == "CogVideo":
        transformed = cv2.resize(frame, final_size, interpolation=cv2.INTER_LINEAR)
    elif transformation == "Kling-Turbo":
        # Kling-Turbo outputs 1920x1080 with black side bars. We center-crop to
        # 1350x1080 to remove the padding and then resize to 1280x1024.
        transformed = _center_crop_resize(frame, CROP_SIZE_KLING_TURBO, final_size, upscale=True)
    elif transformation == "Veo3" or transformation == "Veo3-fast":
        transformed = _center_crop_resize(frame, CROP_SIZE_VEO3, final_size, upscale=True)
    elif transformation == "LTX":
        h, w = frame.shape[:2]
        if w != 960 or h != 736:
            print(f"Warning: Unexpected LTX frame size ({w}x{h}).")
        padded = cv2.copyMakeBorder(frame, 16, 16, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        transformed = cv2.resize(padded, final_size, interpolation=cv2.INTER_LINEAR)
    elif transformation == "Pyramidal":
        transformed = cv2.resize(frame, final_size, interpolation=cv2.INTER_LINEAR)
    elif transformation == "COSMOS":
        transformed = cv2.resize(frame, final_size, interpolation=cv2.INTER_LINEAR)
    elif transformation == "COSMOS-predict2":
        transformed = cosmos_postprocesing_inverse(frame)
    elif transformation == "COSMOS-predict1":
        # no need for any transformation
        transformed = frame
    elif transformation == "WAN-2.1":
        transformed = cv2.resize(frame, final_size, interpolation=cv2.INTER_LINEAR)
    else:
        transformed = cv2.resize(frame, final_size, interpolation=cv2.INTER_LINEAR)
    return transformed

def process_mp4_file(mp4_path, rel_video, output_dir, video_num, use_cache=False):
    """
    Processes an MP4 file directly without saving the unresized frames.
    Reads each frame from the MP4, applies model-dependent inverse resizing,
    saves resized frames into the output 'frames' subfolder, then creates a video
    and writes metadata JSON.
    """
    conditioning, model, enhancement, experiment = extract_metadata_from_path(rel_video)
    if None in (conditioning, model, enhancement, experiment):
        print(f"Missing metadata in path: {rel_video}. Skipping file {mp4_path}.")
        return
    
    out_video_folder = os.path.join(output_dir, conditioning, model, enhancement, experiment, str(video_num))
    json_path = os.path.join(out_video_folder, "info.json")

    if use_cache and os.path.exists(json_path):
        print(f"Skipping MP4 (cached output found).")
        return
    
    os.makedirs(out_video_folder, exist_ok=True)
    frames_out_folder = os.path.join(out_video_folder, "frames_for_tracking")
    os.makedirs(frames_out_folder, exist_ok=True)

    transformation = model if model != "PyramidalFlow" else "Pyramidal"
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"Error opening MP4 file: {mp4_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None or frame.size == 0:
            print(f"Skipping empty or corrupted frame {frame_count} in {mp4_path}")
        else:
            transformed_frame = transform_frame(frame, transformation)
            frame_filename = os.path.join(frames_out_folder, f"{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, transformed_frame)
        frame_count += 1
    cap.release()
    # print(f"Processed MP4 {mp4_path}: {frame_count} resized frames saved to {frames_out_folder}")

    output_video_path = os.path.join(out_video_folder, "output_video.mp4")
    create_video_from_frames(frames_out_folder, output_video_path)

    metadata = {
        "experiment": experiment,
        "model": model,
        "conditioning": conditioning,
        "prompt_type": enhancement,
        "path_to_video": os.path.relpath(output_video_path, output_dir),
        "path_to_frames": os.path.relpath(frames_out_folder, output_dir)
    }

    with open(json_path, "w") as jf:
        json.dump(metadata, jf, indent=4)
    print(f"Wrote metadata JSON to {json_path}")
