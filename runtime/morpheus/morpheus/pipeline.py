"""End-to-end tracking + scoring pipeline.

Walks a canonical input tree (``<conditioning>/<model>/<prompt_type>/<experiment>/
<video_dir>/`` for generated videos, plus a separate ``<real_world_subdir>/
<experiment>/<video_dir>/`` branch for real-world footage) and, per video:

1. loads ``frames_for_tracking/`` and resolves SAM2 click labels,
2. tracks the object(s) with SAM2 and lifts centroids to 3D via DepthAnything,
3. crops falling/bouncing/sliding trajectories and flags discardable ones,
4. optionally scores the (cropped) trajectory into ``combined_scores.json``.

This is the device-agnostic successor of the original ``tracking_and_scores.py``.
The SAM2 predictor is built once (for CUDA/MPS/CPU) and reused across videos.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
from PIL import Image

from .taxonomy import (
    CONDITIONING_TYPES,
    EXPERIMENTS_MATCH_LABELS,
    EXP_NAME_MAP,
    MODEL_NAMES,
)
from .tracking.processing import (
    load_labels,
    process_object_points,
    process_segmentation_frames,
    process_depth_frames,
    save_tracking_data,
    save_segmentation_masks,
    process_and_save_depth_v2_frames,
    DepthProcessorV2,
)
from .tracking.plotting import (
    save_first_frame_segmentation,
    plot_first_frame,
    plot_centres3d_over_time,
)
from .tracking.discard_stats import calculate_permanence_stats
from .tracking.discard_filtering import filter_trajectory
from .tracking.trajectory_cropping import crop_trajectory_if_needed, should_crop_experiment
from .scoring.combined import calculate_combined_score

# Full set of methods / experiments the walker understands (incl. cosmos-transferred
# sub-variants). Callers restrict these via the CLI filters.
ALL_METHODS = {"real-world", *MODEL_NAMES}
ALL_CONDITIONING = set(CONDITIONING_TYPES)
ALL_EXPERIMENTS = {
    "holonomic_pendulum_pendulum", "holonomic_pendulum_wrecking_ball",
    "rolling_full_can_aluminum_barrel", "rolling_full_can_barrel", "rolling_full_can_beverage_can",
    "sliding_book_brick", "sliding_book_delivery_truck_crate", "sliding_book_inclined_shelf",
    "falling_ball_basketball", "falling_ball_baseball", "falling_ball_pingpong",
    "projectile_beach_volleyball", "projectile_kitchen_lemon", "projectile_pinecone",
    "falling_ball", "bouncing_ball", "non_holonomic", "double_pendulum", "holonomic_pendulum",
    "projectile", "falling_apple", "falling_tape", "falling_marker", "sliding_book",
    "rolling_orange", "rolling_empty_can", "rolling_full_can",
    "collision_equal", "collision_big_hits_small", "collision_small_hits_big", "spring",
}


def process_video_folder(video_folder, labels_json, tracker, args):
    """Track and (optionally) score a single ``<video_dir>``."""
    rel_path = os.path.relpath(video_folder, args.input_dir)
    print(f"Processing video folder: {video_folder}")
    parts = rel_path.split(os.sep)

    if args.video_number is not None:
        if "real-world" in video_folder:
            video_name = parts[-1]
            try:
                video_number = int(video_name.split("_")[1])
            except (IndexError, ValueError):
                print(f"Unexpected video name format for real-world video: {video_name}")
                return
        else:
            video_number = int(parts[-1])
        if video_number != args.video_number:
            print(f"Skipping video {video_number} because it is not the video number {args.video_number}")
            return
    if len(parts) < 2:
        print(f"Unexpected folder structure: {video_folder}")
        return

    tracking_output_folder = os.path.join(args.output_dir, rel_path)
    if (
        args.use_cache
        and os.path.exists(tracking_output_folder)
        and os.listdir(tracking_output_folder)
        and os.path.exists(os.path.join(tracking_output_folder, "combined_scores.json"))
    ):
        print(f"Skipping cached folder: {tracking_output_folder}")
        return

    if (
        not args.use_cached_tracking
        or not os.path.exists(tracking_output_folder)
        or not os.listdir(tracking_output_folder)
        or not os.path.exists(os.path.join(tracking_output_folder, "centres3d_obj_1.pkl"))
    ):
        os.makedirs(tracking_output_folder, exist_ok=True)

        frames_dir = os.path.join(video_folder, "frames_for_tracking")
        info_json = os.path.join(video_folder, "info.json")

        if not os.path.exists(info_json):
            print(f"Info.json not found in {video_folder}")
            return
        try:
            with open(info_json, "r") as f:
                info = json.load(f)
        except json.JSONDecodeError:
            print(f"Failed to parse info.json in {video_folder}")
            return

        experiment = info["experiment"]
        if experiment == "non_holonomic_pendulum":
            experiment = "non_holonomic"
            info["experiment"] = "non_holonomic"
        video_key_str = parts[-1]

        if info["model"] == "real-world":
            video_number = int(video_key_str.split("_")[1])
            video_key = f"video_{video_number}"
        else:
            if info.get("conditioning") == "keyframe_interpolation":
                video_number = int(video_key_str.split("_")[0]) if "_" in video_key_str else int(video_key_str)
                video_key = f"video_{video_number}"
            else:
                video_key = "video_5" if experiment == "holonomic_pendulum" else "video_0"

        # Cosmos-transferred sub-variants: derive video number from the "<n>_seed..." folder
        # and normalise the experiment to its canonical label bucket.
        for prefix, canon in (
            (("falling_ball_basketball", "falling_ball_baseball", "falling_ball_pingpong"), None),
            (("projectile_beach_volleyball", "projectile_kitchen_lemon", "projectile_pinecone"), None),
            (("holonomic_pendulum_pendulum", "holonomic_pendulum_wrecking_ball"), "holonomic_pendulum"),
            (("rolling_full_can_aluminum_barrel", "rolling_full_can_barrel", "rolling_full_can_beverage_can"), "rolling_full_can"),
            (("sliding_book_brick", "sliding_book_delivery_truck_crate", "sliding_book_inclined_shelf"), "sliding_book"),
        ):
            if info["experiment"] in prefix:
                print(f"For cosmos-transfered videos overriding video number from: {video_key_str}")
                video_number = int(video_key_str.split("_seed")[0])
                video_key = f"video_{video_number}"
                if canon is not None:
                    experiment = canon

        temp_experiment = None
        if info["experiment"] in EXPERIMENTS_MATCH_LABELS:
            temp_experiment = EXPERIMENTS_MATCH_LABELS[info["experiment"]]
            if temp_experiment not in labels_json:
                print(f"Label info missing for mapped experiment '{temp_experiment}'")
                return
        elif experiment not in labels_json:
            print(f"Label info missing for experiment '{experiment}' and video '{video_key}'")
            return
        elif video_key not in labels_json[experiment]:
            print(f"Label info missing for experiment '{experiment}' and video '{video_key}'")
            return

        label_info = labels_json[temp_experiment][video_key] if temp_experiment else labels_json[experiment][video_key]

        object_points = process_object_points(label_info)
        if not object_points:
            print(f"No labeled points for video {video_key}")
            return

        tracking_plots_dir = os.path.join(tracking_output_folder, "tracking_plots")
        os.makedirs(os.path.join(tracking_plots_dir, "frames"), exist_ok=True)

        first_frame_path = os.path.join(frames_dir, "00000.jpg")
        if not os.path.exists(first_frame_path):
            print(f"First frame not found: {first_frame_path}")
            return
        try:
            first_frame = np.array(Image.open(first_frame_path))
        except Exception as e:
            print(f"Error loading first frame {first_frame_path}: {str(e)}")
            return
        plot_first_frame(first_frame, object_points, tracking_plots_dir)

        inference_state, masks_to_show = tracker.initialize_tracking(frames_dir, object_points)
        save_first_frame_segmentation(first_frame, masks_to_show, tracking_plots_dir)

        video_segments = tracker.propagate(inference_state)
        centers_over_time, masks_over_time, max_distance_over_time, thetas, is_most_probable = (
            process_segmentation_frames(frames_dir, video_segments, tracking_plots_dir, info)
        )

        centers3d_over_time = process_depth_frames(
            frames_dir, masks_over_time, centers_over_time, tracker.depth_processor
        )

        plot_centres3d_over_time(centers3d_over_time, tracking_output_folder)

        if args.save_segmentation_masks:
            save_segmentation_masks(tracking_output_folder, masks_over_time)
        if args.save_depth_masks_v2:
            process_and_save_depth_v2_frames(frames_dir, tracking_output_folder, args.depth_processor_v2)

        save_tracking_data(tracking_output_folder, centers3d_over_time, max_distance_over_time, thetas)

        # Extract conditioning / prompt_type from the path if info.json lacks them.
        path_parts = rel_path.split(os.sep)
        conditioning_type = info.get("conditioning", None)
        prompt_type = info.get("prompt_type", None)
        if not conditioning_type and len(path_parts) >= 1:
            if any(x in path_parts[0] for x in CONDITIONING_TYPES):
                conditioning_type = path_parts[0]
            elif "real-world" in path_parts[0]:
                conditioning_type = "real-world"
        if not prompt_type and len(path_parts) >= 3:
            if not any(x in path_parts[2] for x in [*CONDITIONING_TYPES, "real-world"]):
                prompt_type = path_parts[2]

        crop_info = info.copy()
        if conditioning_type:
            crop_info["conditioning"] = conditioning_type
        if prompt_type:
            crop_info["prompt_type"] = prompt_type

        original_experiment = info["experiment"]
        crop_trajectory_if_needed(tracking_output_folder, original_experiment, crop_info)

        discard_details = filter_trajectory(
            centers3d_over_time, gap_threshold=0.15, min_trajectory_length=0.15, min_valid_percentage=0.15
        )
        filtering_info = {
            **info,
            "trajectory_filtering": {
                "discard_details": discard_details,
                "gap_threshold": 0.15,
                "min_trajectory_length": 0.15,
                "min_valid_percentage": 0.15,
            },
            "is_most_probable": is_most_probable,
        }
        with open(os.path.join(tracking_output_folder, "filtering_info.json"), "w") as f:
            json.dump(filtering_info, f, indent=2)

    if args.calculate_scores:
        print(f"Calculating scores for {video_folder}...", flush=True)
        calculate_statistical_score = not args.only_physical_score
        calculate_physical_score = not args.only_statistical_score

        with open(os.path.join(video_folder, "info.json"), "r") as f:
            info = json.load(f)
        experiment = info["experiment"]

        cropped_pkl_path = os.path.join(tracking_output_folder, "centres3d_obj_1_cropped.pkl")
        if should_crop_experiment(experiment) and os.path.exists(cropped_pkl_path):
            path = cropped_pkl_path
            print(f"  Using cropped trajectory for scoring: {cropped_pkl_path}", flush=True)
        else:
            path = os.path.join(tracking_output_folder, "centres3d_obj_1.pkl")
        path_obj_2 = os.path.join(tracking_output_folder, "centres3d_obj_2.pkl")
        distance_pkl_path = os.path.join(tracking_output_folder, "max_distance.pkl")
        angle_pkl_path = os.path.join(tracking_output_folder, "angles.pkl")

        base_experiment = EXPERIMENTS_MATCH_LABELS.get(experiment, experiment)

        if (
            calculate_statistical_score
            and args.resume_statistical_score
            and os.path.exists(os.path.join(tracking_output_folder, "combined_scores.json"))
        ):
            with open(os.path.join(tracking_output_folder, "combined_scores.json"), "r") as f:
                previous_combined = json.load(f)
                if previous_combined.get("statistical_score") is not None:
                    print(f"Skipping {video_folder}: already has a statistical score.", flush=True)
                    return

        mapped_experiment = EXP_NAME_MAP.get(experiment, experiment)

        if base_experiment == "holonomic_pendulum":
            combined = calculate_combined_score(
                path, experiment, info["model"], distance_pkl_path=distance_pkl_path,
                physical_score=calculate_physical_score, statistical_score=calculate_statistical_score,
            )
        elif base_experiment == "double_pendulum":
            combined = calculate_combined_score(
                path, experiment, info["model"], distance_pkl_path=distance_pkl_path,
                angle_pkl_path=angle_pkl_path, obj_2_centers_pkl_path=path_obj_2,
                physical_score=calculate_physical_score, statistical_score=calculate_statistical_score,
            )
        elif mapped_experiment == "collision":
            combined = calculate_combined_score(
                path, experiment, info["model"], obj_2_centers_pkl_path=path_obj_2,
                physical_score=calculate_physical_score, statistical_score=calculate_statistical_score,
                videos_dir=video_folder,
            )
        else:
            combined = calculate_combined_score(
                path, experiment, info["model"],
                physical_score=calculate_physical_score, statistical_score=calculate_statistical_score,
            )

        combined_with_info = {**info, **combined}
        json_path = os.path.join(tracking_output_folder, "combined_scores.json")
        if os.path.exists(json_path):
            previous_combined = json.load(open(json_path))
            combined_with_info = {**previous_combined, **combined_with_info}
        with open(json_path, "w") as f:
            json.dump(combined_with_info, f, indent=2)
    print(f"Finished processing video: {video_folder}")


def process_video_folder_resilient(video_folder, labels_json, tracker, args):
    """Process one sample without letting a terminal sample failure abort its shard."""
    rel_path = os.path.relpath(video_folder, args.input_dir)
    output_folder = os.path.join(args.output_dir, rel_path)
    failure_path = os.path.join(output_folder, "FAILURE.json")
    try:
        process_video_folder(video_folder, labels_json, tracker, args)
        if os.path.exists(failure_path):
            os.remove(failure_path)
    except Exception as exc:
        import traceback

        os.makedirs(output_folder, exist_ok=True)
        failure = {
            "status": "terminal_failure_zero",
            "sample_key": rel_path,
            "exception_type": type(exc).__name__,
            "reason": str(exc),
            "traceback": traceback.format_exc(),
        }
        with open(failure_path, "w") as handle:
            json.dump(failure, handle, indent=2)
        print(
            f"Terminal sample failure recorded at {failure_path}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def process_all_videos(tracker, args, conditioning, methods, experiments):
    """Iterate the input tree, dispatching each ``<video_dir>`` to the tracker/scorer."""
    labels_json = load_labels(args.labels_json)
    input_dir = args.input_dir

    skip_methods = ALL_METHODS - set(methods)
    skip_experiments = ALL_EXPERIMENTS - set(experiments)
    skip_conditioning = ALL_CONDITIONING - set(conditioning)

    if args.save_depth_masks_v2:
        args.depth_processor_v2 = DepthProcessorV2()

    # Real-world branch (separate, no conditioning/model/prompt levels).
    if "real-world" in methods:
        real_world_dir = os.path.join(input_dir, args.real_world_subdir)
        if os.path.exists(real_world_dir):
            for experiment in os.listdir(real_world_dir):
                if experiment in (".cache", ".gitignore", ".gitattributes") or ".git" in experiment:
                    continue
                if experiment in skip_experiments:
                    continue
                experiment_path = os.path.join(real_world_dir, experiment)
                for video_folder in glob.glob(os.path.join(experiment_path, "*")):
                    if os.path.isdir(video_folder):
                        process_video_folder_resilient(video_folder, labels_json, tracker, args)

    # Generated-videos branch.
    for conditioning_type in conditioning:
        if conditioning_type in skip_conditioning:
            continue
        conditioning_path = os.path.join(input_dir, conditioning_type)
        if not os.path.exists(conditioning_path):
            continue
        for model in methods:
            if model == "real-world":
                continue
            if conditioning_type == "keyframe_interpolation" and model not in ("CogVideo", "WAN-2.1"):
                continue
            if model in skip_methods:
                continue
            model_path = os.path.join(conditioning_path, model)
            if not os.path.exists(model_path):
                continue
            for prompt_type in os.listdir(model_path):
                prompt_path = os.path.join(model_path, prompt_type)
                if not os.path.isdir(prompt_path):
                    continue
                for experiment in experiments:
                    if experiment in skip_experiments:
                        continue
                    if model != "real-world" and experiment == "non_holonomic":
                        experiment_folder_name = "non_holonomic_pendulum"
                    else:
                        experiment_folder_name = experiment
                    experiment_path = os.path.join(prompt_path, experiment_folder_name)
                    if not os.path.exists(experiment_path):
                        continue
                    for video_folder in glob.glob(os.path.join(experiment_path, "*")):
                        if os.path.isdir(video_folder):
                            process_video_folder_resilient(video_folder, labels_json, tracker, args)

    print("Finished processing all videos.")

    if args.calculate_permanence_stats:
        if "real-world" in methods:
            base_path = os.path.join(args.output_dir, "real_world")
            for exp in experiments:
                print(f"Calculating permanence stats for real-world/{exp}")
                calculate_permanence_stats(os.path.join(base_path, exp), recompute_filtering_info=True)
        for cond in conditioning:
            for method in methods:
                if method == "real-world":
                    continue
                base_path = os.path.join(args.output_dir, cond, method)
                if not os.path.exists(base_path):
                    continue
                for prompt_type in [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]:
                    prompt_type_path = os.path.join(base_path, prompt_type)
                    for exp in experiments:
                        print(f"Calculating permanence stats for {cond}/{method}/{prompt_type}/{exp}")
                        calculate_permanence_stats(os.path.join(prompt_type_path, exp), recompute_filtering_info=True)
