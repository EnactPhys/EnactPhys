"""Resize generated videos to the real-world reference resolution (1280x1024).

Each model emits video at its own native resolution/aspect ratio; ``transform_frame``
applies the per-model inverse transform so trajectory-based physics scores are
comparable across models and against the real-world footage.
"""

from .transforms import REAL_WORLD_RESOLUTION, transform_frame, process_mp4_file
from .resize import resize_tree

__all__ = ["REAL_WORLD_RESOLUTION", "transform_frame", "process_mp4_file", "resize_tree"]
