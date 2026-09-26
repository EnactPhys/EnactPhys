"""Single source of truth for the experiment / phenomenon / model taxonomy.

Every stage of the pipeline (generation, resizing, tracking, scoring) keys off a
fixed **experiment x conditioning x prompt-type** taxonomy. In the original research
codebase these maps were copy-pasted across ``scores.py``, ``scores_ablation.py`` and
``tracking_and_scores.py``; here they live in exactly one place.

Naming note (paper <-> code):
    * paper "Dynamical Score"          == code ``statistical_score``  (PINN fit, S_dyn = max(1-NMSE, 0))
    * paper "Physical Invariance Score" == code ``physical_score``     (sliding-window conservation)
The JSON output keys keep the *code* names so results diff directly against the
paper's ``score_breakdown_summary.csv`` columns (``statistical_score_mean`` /
``physical_score_mean``).
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------
# Video sources (models) and the real-world reference
# --------------------------------------------------------------------------------------

#: Generative models evaluated in the benchmark.
MODEL_NAMES = [
    "WAN-2.1",
    "Veo3-fast",
    "Veo3",
    "COSMOS-predict2",
    "COSMOS-predict1",
    "CogVideo",
    "PyramidalFlow",
    "LTX",
    "Kling-Turbo",
]

#: All scoring sources, including the real-world reference footage.
VIDEO_SOURCE = ["real-world", *MODEL_NAMES]

#: Conditioning regimes (must match generation ``--cond_num_frames``).
CONDITIONING_TYPES = [
    "single_frame_conditioning",
    "multi_frame_conditioning",
    "keyframe_interpolation",
]

#: Prompt styles.
PROMPT_TYPES = ["plain", "enhanced"]


# --------------------------------------------------------------------------------------
# Experiment -> phenomenon bucket (drives which PINN / physical checks run)
# --------------------------------------------------------------------------------------

#: Maps a raw experiment (incl. real-world sub-variants) to its physics phenomenon.
#: The phenomenon selects the statistical (PINN) model in ``scoring/pinns``.
PHENOMENON_MAP = {
    "falling_ball_basketball": "freefall",
    "falling_ball_baseball": "freefall",
    "falling_ball_pingpong": "freefall",
    "projectile_beach_volleyball": "projectile",
    "projectile_kitchen_lemon": "projectile",
    "projectile_pinecone": "projectile",
    "falling_ball": "freefall",
    "falling_apple": "freefall",
    "falling_tape": "freefall",
    "falling_marker": "freefall",
    "rolling_orange": "sliding_object",
    "rolling_empty_can": "sliding_object",
    "rolling_full_can": "sliding_object",
    "projectile": "projectile",
    "sliding_book": "sliding_object",
    "holonomic_pendulum": "pendulum",
    "holonomic": "pendulum",
    "bouncing_ball": "bouncingball",
    "non_holonomic_pendulum": "pendulum",
    "non_holonomic": "pendulum",
    "double_pendulum": "doublependulum",
    "collision_equal": "collision",
    "collision_small_hits_big": "collision",
    "collision_big_hits_small": "collision",
    "spring": "spring",
    # cosmos-transferred sub-variants
    "holonomic_pendulum_pendulum": "pendulum",
    "holonomic_pendulum_wrecking_ball": "pendulum",
    "rolling_full_can_aluminum_barrel": "sliding_object",
    "rolling_full_can_barrel": "sliding_object",
    "rolling_full_can_beverage_can": "sliding_object",
    "sliding_book_brick": "sliding_object",
    "sliding_book_delivery_truck_crate": "sliding_object",
    "sliding_book_inclined_shelf": "sliding_object",
}

#: Maps a raw experiment to the canonical experiment name used by the physical score.
EXP_NAME_MAP = {
    "falling_ball_basketball": "falling_ball",
    "falling_ball_baseball": "falling_ball",
    "falling_ball_pingpong": "falling_ball",
    "projectile_beach_volleyball": "projectile",
    "projectile_kitchen_lemon": "projectile",
    "projectile_pinecone": "projectile",
    "falling_ball": "falling_ball",
    "falling_apple": "falling_ball",
    "falling_tape": "falling_ball",
    "falling_marker": "falling_ball",
    "projectile": "projectile",
    "bouncing_ball": "bouncing_ball",
    "sliding_book": "sliding_book",
    "rolling_orange": "sliding_book",
    "rolling_empty_can": "sliding_book",
    "rolling_full_can": "sliding_book",
    "holonomic_pendulum": "holonomic_pendulum",
    "non_holonomic": "non-holonomic_pendulum",  # folder name -> experiment name
    "non_holonomic_pendulum": "non-holonomic_pendulum",
    "double_pendulum": "double_pendulum",
    "collision_equal": "collision",
    "collision_small_hits_big": "collision",
    "collision_big_hits_small": "collision",
    "spring": "spring",
    # cosmos-transferred sub-variants
    "holonomic_pendulum_pendulum": "holonomic_pendulum",
    "holonomic_pendulum_wrecking_ball": "holonomic_pendulum",
    "rolling_full_can_aluminum_barrel": "sliding_book",
    "rolling_full_can_barrel": "sliding_book",
    "rolling_full_can_beverage_can": "sliding_book",
    "sliding_book_brick": "sliding_book",
    "sliding_book_delivery_truck_crate": "sliding_book",
    "sliding_book_inclined_shelf": "sliding_book",
}

#: Cosmos-transferred experiment folder -> experiment whose SAM2 click-labels to reuse.
EXPERIMENTS_MATCH_LABELS = {
    "falling_ball_basketball": "falling_ball",
    "falling_ball_baseball": "falling_ball",
    "falling_ball_pingpong": "falling_ball",
    "projectile_beach_volleyball": "projectile",
    "projectile_kitchen_lemon": "projectile",
    "projectile_pinecone": "projectile",
    "holonomic_pendulum_pendulum": "holonomic_pendulum",
    "holonomic_pendulum_wrecking_ball": "holonomic_pendulum",
    "rolling_full_can_aluminum_barrel": "rolling_full_can",
    "rolling_full_can_barrel": "rolling_full_can",
    "rolling_full_can_beverage_can": "rolling_full_can",
    "sliding_book_brick": "sliding_book",
    "sliding_book_delivery_truck_crate": "sliding_book",
    "sliding_book_inclined_shelf": "sliding_book",
}

#: All experiment keys recognised by the pipeline.
EXPERIMENT_NAMES = list(PHENOMENON_MAP.keys())

#: The 17 canonical experiments (one prompt file each), no sub-variants.
CANONICAL_EXPERIMENTS = [
    "falling_ball",
    "falling_apple",
    "falling_marker",
    "falling_tape",
    "bouncing_ball",
    "projectile",
    "holonomic_pendulum",
    "non_holonomic_pendulum",
    "double_pendulum",
    "spring",
    "sliding_book",
    "rolling_full_can",
    "rolling_empty_can",
    "rolling_orange",
    "collision_equal",
    "collision_big_hits_small",
    "collision_small_hits_big",
]

#: Real-world experiments reported in the paper.
#:
#: ``non_holonomic_pendulum`` is intentionally excluded: it is commented out of every
#: results table in the paper and is not among the 9 reported phenomenon buckets. It
#: remains fully scoreable (present in the maps above) but is not scored by default.
REAL_WORLD_DEFAULT_EXPERIMENTS = [
    "falling_ball",
    "falling_apple",
    "falling_marker",
    "falling_tape",
    "bouncing_ball",
    "projectile",
    "holonomic_pendulum",
    "double_pendulum",
    "spring",
    "sliding_book",
    "rolling_full_can",
    "rolling_empty_can",
    "rolling_orange",
    "collision_equal",
    "collision_big_hits_small",
    "collision_small_hits_big",
]


def phenomenon_for(experiment: str) -> str:
    """Return the phenomenon bucket for an experiment, or raise if unknown."""
    try:
        return PHENOMENON_MAP[experiment]
    except KeyError as exc:
        raise ValueError(f"No phenomenon mapping for experiment '{experiment}'") from exc


def canonical_experiment(experiment: str) -> str:
    """Return the canonical experiment name used by the physical score."""
    return EXP_NAME_MAP.get(experiment, experiment)
