"""Generate EnactPhys videos from the published PhysDelta task manifests."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ["sim/parameter_control", "sim/invariance", "sim/object_control",
          "real/parameter_control", "real/object_control", "real/friction_diagnostic",
          "real/quality", "sim/plausibility", "sim/video_quality"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--track", choices=TRACKS, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--task-id")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        p.error("Require 0 <= shard-index < shard-count")
    dataset = args.dataset.resolve()
    with (dataset / (args.track + ".csv")).open(newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if (args.seed is None or int(r["seed"]) == args.seed)
                and (not args.task_id or r["task_id"] == args.task_id)]
    if not rows:
        p.error("No tasks match the selection")
    if len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate task IDs")
    for row in rows:
        if Path(row["task_id"]).name != row["task_id"]:
            raise ValueError("Invalid task ID")
        for key in ["image", "video", "condition_path", "tracking_mask"]:
            asset = (dataset / row[key]).resolve()
            if not asset.is_relative_to(dataset):
                raise ValueError("Input path escapes dataset root")
            if not asset.is_file():
                raise FileNotFoundError(asset)
            row[key] = str(asset)
        row.update(split="PhysDelta", cfg_scale="1.2", num_inference_steps="30", sigma_shift="5.0")
    selected = rows[args.shard_index::args.shard_count]
    if args.dry_run:
        print(json.dumps(dict(mode="input_validation", track=args.track, total_tasks=len(rows),
              shard_tasks=len(selected), seeds=sorted({int(r["seed"]) for r in selected}),
              sampling=dict(frames=49, width=768, height=448, cfg_scale=1.2,
                            steps=30, sigma_shift=5.0), missing_assets=0), indent=2))
        return
    if not selected:
        return
    for file in [args.checkpoint / "trainable_model.safetensors", args.checkpoint / "complete.json"]:
        if not file.is_file():
            raise FileNotFoundError(file)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / ("generation_manifest_shard_%d.csv" % args.shard_index)
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue()
    if manifest.exists() and manifest.read_text() != content.replace("\r\n", "\n"):
        raise ValueError("Existing output directory has a different task manifest")
    if not manifest.exists():
        with manifest.open("x", newline="") as f:
            f.write(content)
    cmd = [sys.executable, str(ROOT / "scripts/generate_adapter.py"),
           "--manifest", str(manifest), "--model-root", str(args.base_model.resolve()),
           "--checkpoint", str(args.checkpoint.resolve()), "--architecture", "h2_mass",
           "--mass-encoder", "mlp", "--cfg-scale", "1.2",
           "--output-root", str(output / "videos"), "--shard-index", str(args.shard_index),
           "--shard-count", str(args.shard_count)]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "runtime/inference"))
    subprocess.run(cmd, env=env, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
