"""Device-agnostic SAM2 video tracking wrapper.

In the original research code the SAM2 predictor and depth model were only built
when a CUDA device was present, so the tracker could not run on Apple-silicon (MPS)
or CPU at all. Here the predictor is built for whatever device
:func:`morpheus.device.get_device` selects, so tracking runs anywhere SAM2 does.
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import numpy as np

from sam2.build_sam import build_sam2_video_predictor

from ..device import get_device
from .processing import DepthProcessor

# Repo layout: <repo_root>/morpheus/tracking/sam2_tracker.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = str(_REPO_ROOT / "checkpoints" / "sam2.1_hiera_large.pt")
DEFAULT_CONFIG_DIR = str(_REPO_ROOT / "configs")
DEFAULT_MODEL_CFG = "sam2.1_hiera_l.yaml"


class Sam2Tracker:
    """Builds a SAM2 video predictor and propagates object masks across frames.

    Parameters
    ----------
    checkpoint : path to the SAM2.1 hiera-large checkpoint.
    config_dir : directory holding the Hydra SAM2 config (``sam2.1_hiera_l.yaml``).
    model_cfg  : config file name resolved by Hydra within ``config_dir``.
    device     : torch device; defaults to CUDA -> MPS -> CPU.
    load_depth : whether to also build the DepthAnything depth model (needed to lift
                 2D centroids to 3D). Set False for tracking-only / mask export.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        config_dir: str = DEFAULT_CONFIG_DIR,
        model_cfg: str = DEFAULT_MODEL_CFG,
        device=None,
        load_depth: bool = True,
    ):
        # Resolve once; a forced device (e.g. "cpu") applies to both SAM2 and depth.
        self.device = get_device(prefer=device) if isinstance(device, (str, type(None))) else device

        # Hydra keeps a global singleton; clear it so re-initialisation is safe.
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir)

        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=str(self.device))
        self.depth_processor = DepthProcessor(device=self.device) if load_depth else None

    def initialize_tracking(self, frames_dir, object_points):
        """Seed SAM2 with click points on the first frame.

        Returns the inference state and the first-frame masks (for overlay plots).
        """
        inference_state = self.predictor.init_state(video_path=frames_dir)
        self.predictor.reset_state(inference_state)
        masks_to_show = {}
        for obj_id, data in object_points.items():
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                points=data["points"],
                labels=data["labels"],
            )
            masks_to_show[obj_id] = (out_mask_logits[obj_id - 1] > 0.0).cpu().numpy()
        return inference_state, masks_to_show

    def propagate(self, inference_state):
        """Run segmentation propagation over the whole clip and collect masks."""
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            for i, out_obj_id in enumerate(out_obj_ids):
                video_segments.setdefault(out_frame_idx, {})[out_obj_id] = (
                    out_mask_logits[i] > 0.0
                ).cpu().numpy()
        return video_segments


class LazyTracker:
    """Defers building the SAM2 predictor + depth model until the first video needs it.

    This keeps scoring-only runs (``--use-cached-tracking`` over existing trajectory
    pickles) from paying the multi-hundred-MB checkpoint load.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._tracker: Sam2Tracker | None = None

    def _ensure(self) -> Sam2Tracker:
        if self._tracker is None:
            self._tracker = Sam2Tracker(**self._kwargs)
        return self._tracker

    @property
    def depth_processor(self):
        return self._ensure().depth_processor

    def initialize_tracking(self, frames_dir, object_points):
        return self._ensure().initialize_tracking(frames_dir, object_points)

    def propagate(self, inference_state):
        return self._ensure().propagate(inference_state)
