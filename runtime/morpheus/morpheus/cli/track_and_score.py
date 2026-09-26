"""CLI: SAM2-track and physics-score a tree of videos.

Reproduce the paper's real-world numbers with, e.g.::

    morpheus-track-and-score \\
        --input-dir  /path/to/data \\
        --output-dir /path/to/results \\
        --methods real-world --calculate-scores

(the real-world split lives under ``<input-dir>/<real-world-subdir>/<experiment>/...``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..taxonomy import (
    CONDITIONING_TYPES,
    MODEL_NAMES,
    REAL_WORLD_DEFAULT_EXPERIMENTS,
)

# Path defaults are computed here (torch-free) so `--help` stays lightweight; the SAM2
# tracker (which imports torch) is only imported inside main().
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_JSON = str(_REPO_ROOT / "prompts" / "reference_images" / "labels.json")
DEFAULT_CHECKPOINT = str(_REPO_ROOT / "checkpoints" / "sam2.1_hiera_large.pt")
DEFAULT_CONFIG_DIR = str(_REPO_ROOT / "configs")
DEFAULT_MODEL_CFG = "sam2.1_hiera_l.yaml"

_ALL_METHODS = ["real-world", *MODEL_NAMES]
_ALL_EXPERIMENTS = sorted({
    *REAL_WORLD_DEFAULT_EXPERIMENTS, "non_holonomic", "double_pendulum",
    "holonomic_pendulum_pendulum", "holonomic_pendulum_wrecking_ball",
    "rolling_full_can_aluminum_barrel", "rolling_full_can_barrel", "rolling_full_can_beverage_can",
    "sliding_book_brick", "sliding_book_delivery_truck_crate", "sliding_book_inclined_shelf",
    "falling_ball_basketball", "falling_ball_baseball", "falling_ball_pingpong",
    "projectile_beach_volleyball", "projectile_kitchen_lemon", "projectile_pinecone",
})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track objects with SAM2 and compute physics scores.")
    p.add_argument("--input-dir", required=True, help="Root directory containing the video tree.")
    p.add_argument("--output-dir", required=True, help="Where to write tracking results and scores.")

    p.add_argument("--calculate-scores", action="store_true", help="Compute physics scores after tracking.")
    p.add_argument("--only-physical-score", action="store_true",
                   help="Compute only the Physical Invariance score (keep existing Dynamical).")
    p.add_argument("--only-statistical-score", action="store_true",
                   help="Compute only the Dynamical (statistical) score (keep existing Physical).")

    p.add_argument("--use-cache", action="store_true",
                   help="Skip a video that already has combined_scores.json.")
    p.add_argument("--use-cached-tracking", action="store_true",
                   help="Reuse existing tracking pickles; only (re)compute scores.")
    p.add_argument("--resume-statistical-score", action="store_true",
                   help="Skip videos that already have a statistical score.")

    p.add_argument("--save-segmentation-masks", action="store_true", help="Persist SAM2 masks.")
    p.add_argument("--save-depth-masks-v2", action="store_true", help="Export DepthAnything-v2 depth maps.")
    p.add_argument("--calculate-permanence-stats", action="store_true",
                   help="Aggregate trajectory permanence statistics after processing.")

    p.add_argument("--conditioning", nargs="+", choices=CONDITIONING_TYPES, default=list(CONDITIONING_TYPES),
                   help="Conditioning regimes to process (generated videos).")
    p.add_argument("--methods", nargs="+", choices=_ALL_METHODS, default=["real-world"],
                   help="Video sources to process. Default: real-world only.")
    p.add_argument("--experiments", nargs="+", choices=_ALL_EXPERIMENTS,
                   default=list(REAL_WORLD_DEFAULT_EXPERIMENTS),
                   help="Experiments to process. Default: the 9 reported real-world buckets "
                        "(non_holonomic excluded).")
    p.add_argument("--video-number", type=int, default=None, help="Process only this video index.")

    p.add_argument("--real-world-subdir", default="real-world-cropped",
                   help="Sub-directory under --input-dir holding real-world videos "
                        "(e.g. real-world-cropped_iccv_2025).")
    p.add_argument("--labels-json", default=DEFAULT_LABELS_JSON, help="SAM2 click-labels JSON.")

    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None,
                   help="Force a device. Default: auto (cuda -> mps -> cpu).")
    p.add_argument("--sam2-checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--sam2-config-dir", default=DEFAULT_CONFIG_DIR)
    p.add_argument("--sam2-model-cfg", default=DEFAULT_MODEL_CFG)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    # process_video_folder still reads a few attributes under their original names.
    args.depth_processor_v2 = None

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Processing videos in {args.input_dir}")

    # Imported here so `--help` works without importing torch/sam2.
    from ..tracking.sam2_tracker import LazyTracker
    from ..pipeline import process_all_videos

    tracker = LazyTracker(
        checkpoint=args.sam2_checkpoint,
        config_dir=args.sam2_config_dir,
        model_cfg=args.sam2_model_cfg,
        device=args.device,
    )

    process_all_videos(
        tracker,
        args,
        conditioning=tuple(args.conditioning),
        methods=tuple(args.methods),
        experiments=tuple(args.experiments),
    )


if __name__ == "__main__":
    main()
