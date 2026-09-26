"""Extract Real parameter-control trajectories from generated videos with SAM2."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--videos", type=Path, required=True, help="Directory containing task_id.mp4 files")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sam2-checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--model-name", default="enactphys")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        p.error("Invalid shard selection")
    if Path(args.model_name).name != args.model_name:
        p.error("Invalid model name")
    dataset = args.dataset.resolve()
    with (dataset / "real/parameter_control.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if args.seed is None or int(r["seed"]) == args.seed]
    if not rows:
        p.error("No matching tasks")
    rows = rows[args.shard_index::args.shard_count]
    import numpy as np
    tasks = []
    for row in rows:
        if Path(row["task_id"]).name != row["task_id"]:
            raise ValueError("Invalid task ID")
        video = args.videos.resolve() / (row["task_id"] + ".mp4")
        mask = (dataset / row["tracking_mask"]).resolve()
        if not mask.is_relative_to(dataset):
            raise ValueError("Mask path escapes dataset")
        for path in [video, mask]:
            if not path.is_file():
                raise FileNotFoundError(path)
        with np.load(mask, allow_pickle=False) as data:
            count = len(data["segmentation"])
        tasks.append(dict(index=row["task_id"], task_id=row["task_id"], model=args.model_name,
                     phase=row["group_id"], video=str(video), prompt_mask=str(mask),
                     object_count=count, expected_frames=49, expected_width=768, expected_height=448,
                     condition_path=str(dataset / row["condition_path"])))
    if args.dry_run:
        print(json.dumps(dict(tasks=len(tasks), status="INPUTS_VALIDATED")))
        return
    if not args.sam2_checkpoint.is_file():
        raise FileNotFoundError(args.sam2_checkpoint)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection = output / ("selection_%d.json" % args.shard_index)
    contents = json.dumps(tasks, indent=2) + "\n"
    if selection.exists() and selection.read_text() != contents:
        raise ValueError("Existing output has a different input selection")
    if not selection.exists():
        selection.write_text(contents)
    cmd = [sys.executable, str(ROOT / "evaluation/real_tracking.py"),
           "--selection", str(selection), "--output-root", str(output),
           "--code-root", str(ROOT / "evaluation/vendor/sam2"),
           "--checkpoint", str(args.sam2_checkpoint.resolve()), "--worker-id", str(args.shard_index)]
    subprocess.run(cmd, env=dict(os.environ), check=True)
    status = json.loads((output / ("WORKER_%02d_COMPLETE.json" % args.shard_index)).read_text())
    if status["failure_count"]:
        raise SystemExit("Tracking failures recorded; inspect the worker completion report")


if __name__ == "__main__":
    main()
