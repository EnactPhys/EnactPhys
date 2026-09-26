"""Walk a tree of generated videos and resize each to 1280x1024.

Produces the canonical input shape expected by the tracking/scoring pipeline:
``<output_dir>/<conditioning>/<model>/<prompt_type>/<experiment>/<video_dir>/`` with
``frames_for_tracking/``, ``output_video.mp4`` and ``info.json``.

Note: ``<video_dir>`` is written as a bare integer (``0``, ``1``, ...), whereas the
real-world reference tree uses ``video_0`` / ``video_0_fps30``. Both forms exist on
disk in the original data; check what an input tree actually contains.

Pure CPU / OpenCV -- no GPU required. Model detection is by path-string matching
(see ``MODEL_KEYWORDS`` in :mod:`morpheus.resizing.transforms`), so the model name
must appear somewhere in the input path.
"""

from __future__ import annotations

import os
from multiprocessing import Pool, cpu_count

from .transforms import process_mp4_file


def resize_tree(input_dir: str, output_dir: str, use_cache: bool = False, num_workers: int | None = None):
    """Recursively resize every ``.mp4`` under ``input_dir`` into ``output_dir``."""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1 if cpu_count() > 1 else 1)
    print(f"Using {num_workers} worker processes.")

    tasks = []
    with Pool(processes=num_workers) as pool:
        for root, dirs, files in os.walk(input_dir):
            if ".git" in dirs:
                dirs.remove(".git")
            dirs.sort()
            # Only leaf folders (no subdirectories) hold the source clips.
            if not dirs:
                mp4_files = [f for f in files if f.lower().endswith(".mp4")]
                # video number is hardcoded to 0 since it is not in all output paths (e.g. seed-named)
                video_num = 0
                for mp4_file in mp4_files:
                    mp4_path = os.path.join(root, mp4_file)
                    base_name = os.path.splitext(mp4_file)[0]
                    rel_video = os.path.relpath(os.path.join(root, base_name), input_dir)
                    tasks.append(
                        pool.apply_async(process_mp4_file, (mp4_path, rel_video, output_dir, video_num, use_cache))
                    )
                    video_num += 1

        print(f"Submitted {len(tasks)} tasks to the pool. Waiting for completion...")
        for i, task in enumerate(tasks):
            task.get()  # re-raise any worker exception
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{len(tasks)} tasks.")

    print("All videos processed: resized frames, output videos, and metadata JSON files created.")
