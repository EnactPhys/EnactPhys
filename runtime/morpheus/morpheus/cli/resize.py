"""CLI: resize generated videos to the real-world reference resolution.

Example::

    morpheus-resize --input-dir ./generated_videos --output-dir ./resized_videos --use-cache
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resize generated videos to 1280x1024 for fair comparison.")
    p.add_argument("--input-dir", required=True, help="Root folder of source MP4s (model name must be in the path).")
    p.add_argument("--output-dir", required=True, help="Destination for resized frames/videos/metadata.")
    p.add_argument("--use-cache", action="store_true", help="Skip a clip whose info.json already exists.")
    p.add_argument("--num-workers", type=int, default=None, help="Process pool size (default: CPUs - 1).")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from ..resizing.resize import resize_tree

    resize_tree(args.input_dir, args.output_dir, use_cache=args.use_cache, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
