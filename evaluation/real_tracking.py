#!/usr/bin/env python3
"""Persistent SAM2 worker for the 49-frame Wan and 57-frame official PhyCo outputs."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def read_frames(path: Path) -> tuple[list[np.ndarray], float, int, int]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"cannot decode {path}")
    return frames, fps, width, height


def mask_metrics(mask: np.ndarray) -> dict[str, object]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    components: list[dict[str, object]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < 20:
            continue
        component_mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        components.append({
            "area": area,
            "bbox_xywh": [x, y, width, height],
            "centroid_xy": [float(centroids[label][0]), float(centroids[label][1])],
            "aspect": float(width / max(height, 1)),
            "circularity": 0.0 if perimeter <= 0 else float(4 * math.pi * area / perimeter**2),
        })
    components.sort(key=lambda item: int(item["area"]), reverse=True)
    total_area = int(sum(int(item["area"]) for item in components))
    if components and total_area:
        cx = sum(float(item["centroid_xy"][0]) * int(item["area"]) for item in components) / total_area
        cy = sum(float(item["centroid_xy"][1]) * int(item["area"]) for item in components) / total_area
        centroid = [float(cx), float(cy)]
    else:
        centroid = [None, None]
    return {
        "component_count": len(components),
        "total_area": total_area,
        "largest_fraction": float(int(components[0]["area"]) / total_area) if components and total_area else 0.0,
        "centroid_xy": centroid,
        "components": components,
    }


def track_one(task: dict[str, object], predictor, args: argparse.Namespace) -> dict[str, object]:
    output = args.output_root / "trajectories" / str(task["model"]) / str(task["phase"]) / f"{task['task_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("status") == "COMPLETE" and payload.get("task", {}).get("task_id") == task["task_id"]:
            return {"status": "SKIP_COMPLETE", "output": str(output)}

    frames, fps, width, height = read_frames(Path(str(task["video"])))
    expected = (int(task["expected_frames"]), int(task["expected_width"]), int(task["expected_height"]))
    if (len(frames), width, height) != expected:
        raise ValueError(
            f"video contract: {task['video']} got={len(frames)},{width}x{height} expected={expected[0]},{expected[1]}x{expected[2]}"
        )
    with np.load(str(task["prompt_mask"]), allow_pickle=False) as archive:
        prompts = np.asarray(archive["segmentation"]) > 0
    if prompts.shape != (int(task["object_count"]), height, width):
        raise ValueError(f"prompt shape {prompts.shape} for {task['task_id']}")
    images = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]).astype(np.float32) / 255.0
    started = time.perf_counter()
    state = predictor.init_state(images=images, device="cuda:0")
    try:
        for object_index, prompt in enumerate(prompts):
            predictor.add_new_mask(
                state, frame_idx=0, obj_id=object_index,
                mask=torch.from_numpy(prompt.astype(np.uint8)),
            )
        records: list[dict[int, dict[str, object]]] = [dict() for _ in frames]
        frame0_masks: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            for frame_index, object_ids, logits in predictor.propagate_in_video(state):
                for slot, object_id in enumerate(object_ids):
                    mask = logits[slot].detach().float().cpu().squeeze().numpy() > 0.0
                    if mask.shape != (height, width):
                        mask = cv2.resize(
                            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    records[int(frame_index)][int(object_id)] = mask_metrics(mask)
                    if int(frame_index) == 0:
                        frame0_masks[int(object_id)] = mask
        if any(len(frame) != int(task["object_count"]) for frame in records):
            raise RuntimeError("SAM2 did not return every object on every frame")
        frame0_ious = []
        for object_index, prompt in enumerate(prompts):
            predicted = frame0_masks[object_index]
            union = int(np.logical_or(prompt, predicted).sum())
            predicted_iou = float(np.logical_and(prompt, predicted).sum() / union) if union else 0.0
            predicted_area = int(records[0][object_index]["total_area"])
            # Frame zero is an observed conditioning mask, not a tracker
            # prediction.  Keep it exact so tiny real objects are not eroded by
            # SAM2 before displacement is measured; SAM2 propagation is used
            # from frame one onward.
            records[0][object_index] = mask_metrics(prompt)
            frame0_ious.append({
                "object_index": object_index,
                "prompt_area": int(prompt.sum()),
                "predicted_area": predicted_area,
                "nonempty": int(prompt.sum()) > 0,
                "iou": 1.0,
                "sam2_predicted_iou": predicted_iou,
                "frame0_source": "exact_prompt_mask",
            })
        if any(not item["nonempty"] for item in frame0_ious):
            raise RuntimeError(f"empty frame-0 prompt mask: {frame0_ious}")
        elapsed = time.perf_counter() - started
        payload = {
            "status": "COMPLETE",
            "task": task,
            "video_sha256": "not-computed",
            "future_masks_used": False,
            "frame0_records_use_exact_prompt": True,
            "tracker": {
                "name": "Meta SAM2 persistent multi-object video predictor",
                "checkpoint": str(args.checkpoint),
                "config": args.config,
                "device": "cuda:0",
                "elapsed_s": elapsed,
            },
            "video_contract": {"frames": len(frames), "fps": fps, "width": width, "height": height},
            "frame0_contract": frame0_ious,
            "records": [
                {"frame_index": index, "objects": {str(key): value for key, value in sorted(frame.items())}}
                for index, frame in enumerate(records)
            ],
        }
        output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
        return {"status": "PASS", "output": str(output), "elapsed_s": elapsed}
    finally:
        try:
            predictor.reset_state(state)
        except Exception:
            pass
        del state, images, frames
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="sam2_hiera_l.yaml")
    parser.add_argument("--worker-id", type=int, required=True)
    args = parser.parse_args()
    tasks = json.loads(args.selection.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONPATH", str(args.code_root))
    import sys
    sys.path.insert(0, str(args.code_root.resolve()))
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda:0")
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    started = time.time()
    for offset, task in enumerate(tasks):
        try:
            result = track_one(task, predictor, args)
            rows.append({
                "index": task["index"], "model": task["model"], "phase": task["phase"],
                "task_id": task["task_id"], **result,
            })
            print(json.dumps({
                "worker": args.worker_id, "done": offset + 1, "total": len(tasks),
                "task": task["task_id"], "model": task["model"], "status": result["status"],
            }), flush=True)
        except Exception as error:
            failure = {
                "index": task["index"], "task_id": task["task_id"], "model": task["model"],
                "phase": task["phase"], "error": repr(error),
            }
            failures.append(failure)
            rows.append({**failure, "status": "UNEVALUABLE_TRACKER_FAILURE"})
            print(json.dumps({
                "worker": args.worker_id, **failure, "status": "UNEVALUABLE_TRACKER_FAILURE",
            }), flush=True)
    complete = {
        "status": "PASS" if not failures else "PASS_WITH_UNEVALUABLE",
        "worker_id": args.worker_id,
        "task_count": len(tasks),
        "completed": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_s": time.time() - started,
        "rows": rows,
    }
    (args.output_root / f"WORKER_{args.worker_id:02d}_COMPLETE.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
