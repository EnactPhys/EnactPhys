"""Physics-IQ per-video metric extraction with the recorded frame convention."""
import json
import sys
from pathlib import Path
import imageio.v2 as imageio

def pad49(source: Path, target: Path) -> None:
    reader = imageio.get_reader(source)
    writer = imageio.get_writer(
        target,
        fps=24,
        codec="libx264",
        quality=7,
        macro_block_size=None,
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    frames = []
    try:
        for frame in reader:
            frames.append(frame)
            writer.append_data(frame)
        if len(frames) != 49:
            raise ValueError(f"{source}: expected 49 frames, got {len(frames)}")
        for _ in range(71):
            writer.append_data(frames[-1])
    finally:
        reader.close()
        writer.close()

def score_worker(job: dict) -> dict:
    sys.path.insert(0, job["benchmark_code_root"])
    from physiq.binary_mask_generator import generate_mask
    from physiq.calculate_and_write_metrics_to_csv import _build_view_paths, process_view

    task_id = job["task_id"]
    media = Path(job["media_root"]) / task_id
    video_dir = media / "video"
    mask_dir = media / "mask"
    result_path = media / "VIEW_METRICS.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    video_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    video = video_dir / job["output_video_name"]
    if not video.is_file():
        pad49(Path(job["generated_video"]), video)
    expected_mask = mask_dir / job["expected_mask_name"]
    if not expected_mask.is_file():
        generate_mask(str(video), str(mask_dir / job["output_video_name"]), False)
    if not expected_mask.is_file():
        raise FileNotFoundError(expected_mask)
    paths = _build_view_paths(
        job["scenario"],
        job["view"],
        job["take1_id"],
        job["take2_id"],
        24,
        job["real_folder"],
        str(video_dir),
        job["real_masks"],
        str(mask_dir),
    )
    metrics = process_view(paths, job["view"], 0, 120, 120)
    if not metrics:
        raise RuntimeError(f"no metrics for {task_id}")
    result = {
        "task_id": task_id,
        "benchmark_id": job["benchmark_id"],
        "search_variant": job["search_variant"],
        "parameter_variant": job.get("parameter_variant", ""),
        "scenario": job["scenario"] if job["scenario"].endswith(".mp4") else job["scenario"] + ".mp4",
        "view": job["view"],
        "output_video_name": job["output_video_name"],
        "video": str(video),
        "metrics": metrics,
        "condition_summary_json": job["condition_summary_json"],
        "prompt_source": job["prompt_source"],
        "generation_seed": int(job["generation_seed"]),
    }
    result_path.write_text(json.dumps(result) + "\n")
    return result
