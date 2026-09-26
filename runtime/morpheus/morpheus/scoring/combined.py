"""Combine the physical-invariance and dynamical scores for a single video.

This is the public scoring entry point. It merges:

* **Physical Invariance Score** (``physical_score``) -- rule-based conservation checks
  over sliding time windows, from :mod:`morpheus.scoring.physical_score`.
* **Dynamical Score** (``statistical_score``) -- PINN fit to the equation of motion,
  ``1 - min(NMSE, 1)``, from :mod:`morpheus.scoring.dynamical_score`.

The output JSON keys keep the original code names (``physical_score`` /
``statistical_score``) so results diff directly against the paper's
``score_breakdown_summary.csv`` columns. See :mod:`morpheus.taxonomy` for the
paper<->code naming note.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from ..taxonomy import EXP_NAME_MAP, MODEL_NAMES, phenomenon_for
from .physical_score import calculate_one_video_score
from .dynamical_score import run_pin_framework


def calculate_combined_score(
    centers_pkl_path,
    exp_name,
    data_source_name,
    stillness_penalty=False,
    obj_2_centers_pkl_path=None,
    distance_pkl_path=None,
    angle_pkl_path=None,
    physical_score=True,
    statistical_score=True,
    videos_dir=None,
):
    """Calculate and combine physical and statistical scores into a unified dict."""
    category = "VGM_trajectories" if data_source_name in MODEL_NAMES else "real_world_trajectories"

    # Physical Invariance Score (conservation checks).
    if physical_score:
        physical_results = calculate_one_video_score(
            centers_pkl_path,
            EXP_NAME_MAP[exp_name],
            category,
            "test",
            stillness_penalty,
            obj_2_centers_pkl_path,
            distance_pkl_path,
            angle_pkl_path=angle_pkl_path,
            videos_dir=videos_dir,
        )
        print(f"Physical score: {physical_results}")
    else:
        physical_results = {}

    # Dynamical Score (PINN fit).
    if statistical_score:
        # Reset every per-video PINN to the same state.  The upstream scorer
        # otherwise depends on process history and can give different scores
        # for byte-identical videos.
        dynamical_seed = int(os.environ.get("MORPHEUS_PINN_SEED", "3407"))
        random.seed(dynamical_seed)
        np.random.seed(dynamical_seed)
        torch.manual_seed(dynamical_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(dynamical_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        phenomenon = phenomenon_for(exp_name)
        print(f"Running PIN framework for {phenomenon}...")
        print(f"Centers pkl path: {centers_pkl_path}", flush=True)
        stat = run_pin_framework(
            centers_pkl_path,
            phenomenon,
            verbose=True,
            n_epochs=200000,
            lr=1e-3,
            obj_2_centers_pkl_path=obj_2_centers_pkl_path,
            angles_pkl_path=angle_pkl_path,
        )
        mse = stat["mse"]
        nmse = stat["nmse"]
        print(f"MSE: {mse}, NMSE: {nmse}, statistical_score: {1 - min(nmse, 1)}")
        statistical_results = {
            "individual_statistical_scores": {"MSE": mse, "NMSE": nmse},
            "statistical_score": 1 - min(nmse, 1),
            "dynamical_seed": dynamical_seed,
        }
    else:
        statistical_results = {}

    return {**physical_results, **statistical_results}
