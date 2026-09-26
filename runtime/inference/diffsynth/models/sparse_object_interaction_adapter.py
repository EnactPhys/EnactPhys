"""Variable-cardinality sparse object interaction adapter.

Only first-frame masks and explicitly-present F/g/mu/e controls enter the
forward path. Future masks remain loss-only localization targets.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn


H2_ABLATION_MODES = (
    "full", "no_rw_supervision", "no_temporal_attention", "no_state_carry",
    "no_contact_supervision", "no_reader_route_carry", "no_motion_losses",
    "always_on_gate", "shared_contact_gate",
)


def h2_ablation_mode() -> str:
    mode = os.environ.get("PHYSICAL_WM_H2_ABLATION", "full")
    if mode not in H2_ABLATION_MODES:
        raise ValueError(f"unknown H2 ablation: {mode!r}")
    return mode


FORWARD_CONDITION_KEYS = (
    "first_frame_masks",
    "object_valid_mask",
    "force",
    "force_present",
    "mu",
    "mu_present",
    "restitution",
    "restitution_present",
)

TEMPORAL_FORWARD_CONDITION_KEYS = FORWARD_CONDITION_KEYS + (
    "gravity",
    "gravity_present",
)


def _truncate_attention_to_prefix_mass(
    attention: torch.Tensor,
    support_mass: float,
) -> torch.Tensor:
    """Keep the shortest score-sorted prefix reaching ``support_mass``."""
    if not 0.0 < support_mass <= 1.0:
        raise ValueError("writer support mass must be in (0, 1]")
    attention_float = attention.float()
    cumulative_before = attention_float.cumsum(dim=-1) - attention_float
    keep = cumulative_before < support_mass
    truncated = attention_float * keep
    truncated = truncated / truncated.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return truncated.to(dtype=attention.dtype)


TEMPORAL_SCHEDULE_FORWARD_CONDITION_KEYS = TEMPORAL_FORWARD_CONDITION_KEYS + (
    "force_schedule",
)
TEMPORAL_PAIR_SCHEDULE_FORWARD_CONDITION_KEYS = (
    TEMPORAL_SCHEDULE_FORWARD_CONDITION_KEYS + ("platform_mask",)
)
TEMPORAL_NO_INTERACTION_SCHEDULE_FORWARD_CONDITION_KEYS = (
    TEMPORAL_SCHEDULE_FORWARD_CONDITION_KEYS + ("platform_mask",)
)
TEMPORAL_INDEPENDENT_EDGE_SCHEDULE_FORWARD_CONDITION_KEYS = (
    TEMPORAL_SCHEDULE_FORWARD_CONDITION_KEYS
    + (
        "platform_mask",
        "edge_mu",
        "edge_mu_present",
        "edge_restitution",
        "edge_restitution_present",
    )
)
INDEPENDENT_EDGE_CONDITION_CONTRACT = (
    "continuous-writer-independent-undirected-edge-r4-v1"
)
DIAGNOSTIC_SPATIAL_ORACLE_MODES = (
    "predicted",
    "oracle_read",
    "oracle_write",
    "oracle_both",
)
WRITE_GATE_MODES = ("legacy", "bounded_v1", "erase_then_write_v1")


def _environment_flag(name: str, default: bool = False) -> bool:
    """Read an opt-in boolean switch from the environment.

    Only explicit values are accepted so a typo cannot silently select the
    wrong architecture in a launched run.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {raw!r}")


def _environment_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    value = float(default if raw is None or not raw.strip() else raw.strip())
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _straight_through_support_gate(
    logits: torch.Tensor,
    object_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return exact binary forward gates with sigmoid straight-through gradients."""
    if logits.ndim != 4:
        raise ValueError("support gate logits must be [B,T,N,K]")
    if tuple(object_valid.shape) != (logits.shape[0], logits.shape[2]):
        raise ValueError("support gate object_valid must be [B,N]")
    probabilities = torch.sigmoid(logits)
    hard = probabilities >= 0.5
    valid = object_valid[:, None, :, None]
    hard = hard & valid
    all_off = valid & ~hard.any(dim=-1, keepdim=True)
    fallback = torch.zeros_like(hard)
    fallback.scatter_(-1, logits.argmax(dim=-1, keepdim=True), True)
    hard = hard | (fallback & all_off)
    straight_through = hard.to(probabilities.dtype) + probabilities - probabilities.detach()
    straight_through = straight_through * valid.to(probabilities.dtype)
    return straight_through, hard


def _normalize_gated_scores(
    scores: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    shifted = scores - scores.amax(dim=-1, keepdim=True)
    mass = shifted.exp() * gate
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _zero_last(module: nn.Sequential) -> None:
    layer = module[-1]
    if not isinstance(layer, nn.Linear):
        raise TypeError("expected an MLP ending in nn.Linear")
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class SparsePhysicsResidualEncoder(nn.Module):
    """Encode only declared interventions; absence is never encoded as zero."""

    def __init__(self, object_dim: int):
        super().__init__()
        self.force = _mlp(3, object_dim, object_dim)
        self.mu = _mlp(1, object_dim, object_dim)
        self.restitution = _mlp(1, object_dim, object_dim)
        for module in (self.force, self.mu, self.restitution):
            _zero_last(module)

    def forward(
        self,
        force: torch.Tensor,
        force_present: torch.Tensor,
        mu: torch.Tensor,
        mu_present: torch.Tensor,
        restitution: torch.Tensor,
        restitution_present: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> torch.Tensor:
        magnitude = force[..., :1]
        # A present zero-force condition is invariant to its placeholder
        # direction. Presence and numeric zero therefore retain distinct
        # meanings without leaking the unused direction into the token.
        force_vector = torch.cat((magnitude, magnitude * force[..., 1:]), dim=-1)
        valid = object_valid[..., None]
        return valid * (
            force_present[..., None] * self.force(force_vector)
            + mu_present[..., None] * self.mu(mu[..., None])
            + restitution_present[..., None]
            * self.restitution(restitution[..., None])
        )


class ParallelSparseObjectRouter(nn.Module):
    """Find every object at every time on a low-dimensional coarse grid."""

    def __init__(
        self,
        wan_dim: int,
        object_dim: int,
        locator_dim: int,
        topk: int,
        max_times: int = 13,
        previous_frame_tracking: bool = False,
    ):
        super().__init__()
        self.object_dim = object_dim
        self.locator_dim = locator_dim
        self.topk = int(topk)
        self.previous_frame_tracking = bool(previous_frame_tracking)
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        self.input_norm = nn.LayerNorm(wan_dim)
        self.value_projection = nn.Linear(wan_dim, object_dim)
        self.query_projection = nn.Linear(object_dim, locator_dim)
        self.key_projection = nn.Linear(wan_dim, locator_dim)
        self.position_encoder = _mlp(5, object_dim, object_dim)
        self.time_embedding = nn.Embedding(max_times, object_dim)
        self.query_norm = nn.LayerNorm(object_dim)
        self.read_projection = nn.Linear(object_dim, object_dim)
        self.location_bias = nn.Parameter(torch.tensor(-4.0))
        self.hard_support_enabled = _environment_flag(
            "PHYSICAL_WM_HARD_SUPPORT", False
        )
        self.correctable_route_carry = _environment_flag(
            "PHYSICAL_WM_CORRECTABLE_ROUTE_CARRY", False
        )
        self.route_prior_alpha = _environment_float(
            "PHYSICAL_WM_ROUTE_PRIOR_ALPHA", 1.0
        )
        if self.route_prior_alpha < 0.0:
            raise ValueError("PHYSICAL_WM_ROUTE_PRIOR_ALPHA must be non-negative")
        self.support_gate = None
        if self.hard_support_enabled:
            # Optional architecture branches must not perturb initialization of
            # parameters shared with the comparison arm.
            with torch.random.fork_rng(devices=[]):
                self.support_gate = _mlp(locator_dim * 2, locator_dim, 1)
        if self.support_gate is not None:
            # p=0.5 and threshold >= starts with the exact old Top-K support.
            _zero_last(self.support_gate)
        self.last_support_gate_logits: Optional[torch.Tensor] = None
        self.last_support_gate_mask: Optional[torch.Tensor] = None

    @staticmethod
    def _mask_geometry(masks: torch.Tensor) -> torch.Tensor:
        """Return normalized centroid, spread, and area for each mask."""
        _, _, height, width = masks.shape
        dtype, device = masks.dtype, masks.device
        y = torch.linspace(0.0, 1.0, height, dtype=dtype, device=device)
        x = torch.linspace(0.0, 1.0, width, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        mass = masks.sum(dim=(-2, -1)).clamp_min(1e-6)
        cx = (masks * xx).sum(dim=(-2, -1)) / mass
        cy = (masks * yy).sum(dim=(-2, -1)) / mass
        sx = torch.sqrt(
            (masks * (xx - cx[..., None, None]).square()).sum(dim=(-2, -1))
            / mass
            + 1e-6
        )
        sy = torch.sqrt(
            (masks * (yy - cy[..., None, None]).square()).sum(dim=(-2, -1))
            / mass
            + 1e-6
        )
        area = mass / float(height * width)
        return torch.stack((cx, cy, sx, sy, area), dim=-1)

    @staticmethod
    def _gather_candidates(
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        batch, times, objects, candidates = indices.shape
        batch_index = torch.arange(batch, device=values.device)[:, None, None, None]
        time_index = torch.arange(times, device=values.device)[None, :, None, None]
        batch_index = batch_index.expand(batch, times, objects, candidates)
        time_index = time_index.expand(batch, times, objects, candidates)
        return values[batch_index, time_index, indices]

    def read_slots_at_indices(
        self,
        hidden: torch.Tensor,
        first_frame_masks: torch.Tensor,
        object_valid: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        candidate_indices: torch.Tensor,
        candidate_attention: torch.Tensor,
    ) -> torch.Tensor:
        """Re-read object slots at fixed indices for a diagnostic intervention.

        The spatial indices may differ from the router prediction, but the
        original candidate-attention weights are retained.  This keeps Top-K
        cardinality and read mass fixed while changing only spatial support.
        The helper deliberately supports only the parallel/no-TrackPrev route;
        recursively tracked reads would require replaying the whole recurrence.
        """
        if self.previous_frame_tracking:
            raise RuntimeError(
                "fixed-index diagnostic read is unsupported with previous-frame tracking"
            )
        batch, objects = first_frame_masks.shape[:2]
        expected_prefix = (batch, frames, objects)
        if tuple(candidate_indices.shape[:3]) != expected_prefix:
            raise ValueError(
                "diagnostic candidate indices must start with [B,T,N], got "
                f"{tuple(candidate_indices.shape)}"
            )
        if tuple(candidate_attention.shape) != tuple(candidate_indices.shape):
            raise ValueError("diagnostic attention and indices must have equal shape")
        if candidate_indices.dtype != torch.long:
            raise ValueError("diagnostic candidate indices must be torch.long")
        if torch.any((candidate_indices < 0) | (candidate_indices >= height * width)):
            raise ValueError("diagnostic candidate indices are outside the Wan grid")

        coarse_height, coarse_width = first_frame_masks.shape[2:]
        if height % coarse_height or width % coarse_width:
            raise ValueError("coarse mask grid must divide the full Wan token grid")
        scale_y = height // coarse_height
        scale_x = width // coarse_width
        normalized_full = self.input_norm(hidden).reshape(
            batch, frames, height, width, hidden.shape[-1]
        )
        coarse_hidden = normalized_full.reshape(
            batch,
            frames,
            coarse_height,
            scale_y,
            coarse_width,
            scale_x,
            hidden.shape[-1],
        ).mean(dim=(3, 5))
        mass = first_frame_masks.sum(dim=(-2, -1))
        if torch.any(object_valid & (mass <= 1e-6)):
            raise ValueError("every valid diagnostic object must be visible at frame zero")
        masks = first_frame_masks.flatten(-2)
        raw_anchors = torch.einsum(
            "bog,bgd->bod",
            masks / mass.clamp_min(1e-6)[..., None],
            coarse_hidden[:, 0].flatten(1, 2),
        )
        anchors = self.value_projection(raw_anchors) + self.position_encoder(
            self._mask_geometry(first_frame_masks)
        )
        time_ids = torch.arange(frames, device=hidden.device)
        queries = anchors[:, None] + self.time_embedding(time_ids)[None, :, None]
        candidate_hidden = self._gather_candidates(
            normalized_full.flatten(2, 3), candidate_indices
        )
        candidate_values = self.value_projection(candidate_hidden)
        read = (candidate_attention[..., None] * candidate_values).sum(dim=-2)
        slots = queries + self.read_projection(read)
        return slots * object_valid[:, None, :, None]

    def forward(
        self,
        hidden: torch.Tensor,
        first_frame_masks: torch.Tensor,
        object_valid: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        previous_route_indices: Optional[torch.Tensor] = None,
        previous_route_attention: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, objects = first_frame_masks.shape[:2]
        self.last_support_gate_logits = None
        self.last_support_gate_mask = None
        if hidden.shape[1] != frames * height * width:
            raise ValueError("sparse router requires plain Wan video tokens")
        coarse_height, coarse_width = first_frame_masks.shape[2:]
        if height % coarse_height or width % coarse_width:
            raise ValueError(
                "coarse mask grid must divide the full Wan token grid: "
                f"coarse={(coarse_height, coarse_width)} full={(height, width)}"
            )
        scale_y = height // coarse_height
        scale_x = width // coarse_width
        if frames > self.time_embedding.num_embeddings:
            raise ValueError(f"too many latent times: {frames}")
        mass = first_frame_masks.sum(dim=(-2, -1))
        if torch.any(object_valid & (mass <= 1e-6)):
            raise ValueError("every valid object mask must contain visible pixels")

        normalized_full = self.input_norm(hidden).reshape(
            batch, frames, height, width, hidden.shape[-1]
        )
        full_key = self.key_projection(normalized_full)
        coarse_hidden = normalized_full.reshape(
            batch,
            frames,
            coarse_height,
            scale_y,
            coarse_width,
            scale_x,
            hidden.shape[-1],
        ).mean(dim=(3, 5))
        coarse_key = full_key.reshape(
            batch,
            frames,
            coarse_height,
            scale_y,
            coarse_width,
            scale_x,
            self.locator_dim,
        ).mean(dim=(3, 5))
        masks = first_frame_masks.flatten(-2)
        raw_anchors = torch.einsum(
            "bog,bgd->bod",
            masks / mass.clamp_min(1e-6)[..., None],
            coarse_hidden[:, 0].flatten(1, 2),
        )
        anchors = self.value_projection(raw_anchors) + self.position_encoder(
            self._mask_geometry(first_frame_masks)
        )
        time_ids = torch.arange(frames, device=hidden.device)
        coarse_key_flat = coarse_key.flatten(2, 3)
        if self.previous_frame_tracking:
            if previous_route_indices is not None or previous_route_attention is not None:
                raise ValueError("cross-block route carry is incompatible with TrackPrev")
            # The first-frame mask remains the identity anchor.  For t>0 the
            # query also reads the selected appearance from t-1.  This path is
            # is causal by temporal index and never consumes future
            # segmentation labels (the Wan hidden itself remains bidirectional).
            slots_per_time = []
            scores_per_time = []
            indices_per_time = []
            attention_per_time = []
            gate_logits_per_time = []
            gate_mask_per_time = []
            previous_read = None
            normalized_grid = normalized_full.flatten(2, 3)
            full_key_grid = full_key.flatten(2, 3)
            candidate_count = min(self.topk, coarse_height * coarse_width)
            offset_y = torch.arange(scale_y, device=hidden.device)
            offset_x = torch.arange(scale_x, device=hidden.device)
            for time_index in range(frames):
                query_state = anchors + self.time_embedding(time_ids[time_index])
                if previous_read is not None:
                    query_state = query_state + self.read_projection(previous_read)
                query = self.query_projection(self.query_norm(query_state))
                scores = torch.einsum(
                    "bod,bgd->bog", query, coarse_key_flat[:, time_index]
                ) / math.sqrt(self.locator_dim)
                _, coarse_indices = scores.topk(candidate_count, dim=-1)
                coarse_y = coarse_indices // coarse_width
                coarse_x = coarse_indices % coarse_width
                full_y = coarse_y[..., None, None] * scale_y + offset_y[
                    None, None, None, :, None
                ]
                full_x = coarse_x[..., None, None] * scale_x + offset_x[
                    None, None, None, None, :
                ]
                expanded_indices = (full_y * width + full_x).flatten(-3)
                expanded_keys = self._gather_candidates(
                    full_key_grid[:, time_index : time_index + 1],
                    expanded_indices[:, None],
                ).squeeze(1)
                expanded_scores = torch.einsum(
                    "bod,bokd->bok", query, expanded_keys
                ) / math.sqrt(self.locator_dim)
                final_count = min(self.topk, expanded_indices.shape[-1])
                refined_scores, refined_offsets = expanded_scores.topk(
                    final_count, dim=-1
                )
                candidate_indices = expanded_indices.gather(-1, refined_offsets)
                candidate_hidden = self._gather_candidates(
                    normalized_grid[:, time_index : time_index + 1],
                    candidate_indices[:, None],
                ).squeeze(1)
                candidate_values = self.value_projection(candidate_hidden)
                if self.support_gate is not None:
                    candidate_keys = torch.gather(
                        expanded_keys,
                        -2,
                        refined_offsets[..., None].expand(
                            *refined_offsets.shape, self.locator_dim
                        ),
                    )
                    gate_logits = self.support_gate(
                        torch.cat(
                            (
                                query[..., None, :].expand_as(candidate_keys),
                                candidate_keys,
                            ),
                            dim=-1,
                        )
                    ).squeeze(-1)
                    gate, hard_gate = _straight_through_support_gate(
                        gate_logits[:, None], object_valid
                    )
                    candidate_attention = _normalize_gated_scores(
                        refined_scores, gate[:, 0]
                    )
                    gate_logits_per_time.append(gate_logits)
                    gate_mask_per_time.append(hard_gate[:, 0])
                else:
                    candidate_attention = refined_scores.softmax(dim=-1)
                    candidate_attention = (
                        candidate_attention * object_valid[:, :, None]
                    )
                read = (
                    candidate_attention[..., None] * candidate_values
                ).sum(dim=-2)
                slot = (query_state + self.read_projection(read)) * object_valid[
                    ..., None
                ]
                slots_per_time.append(slot)
                scores_per_time.append(scores)
                indices_per_time.append(candidate_indices)
                attention_per_time.append(candidate_attention)
                previous_read = read
            scores = torch.stack(scores_per_time, dim=1)
            if self.support_gate is not None:
                self.last_support_gate_logits = torch.stack(
                    gate_logits_per_time, dim=1
                )
                self.last_support_gate_mask = torch.stack(
                    gate_mask_per_time, dim=1
                )
            return (
                torch.stack(slots_per_time, dim=1),
                (scores + self.location_bias).reshape(
                    batch, frames, objects, coarse_height, coarse_width
                ),
                torch.stack(indices_per_time, dim=1),
                torch.stack(attention_per_time, dim=1),
                coarse_key_flat,
            )

        queries = anchors[:, None] + self.time_embedding(time_ids)[None, :, None]
        query = self.query_projection(self.query_norm(queries))
        # Correlation is computed once for all B/T/N queries on the 14x24
        # coarse grid. Full-D Wan values are gathered only inside selected
        # coarse cells below.
        scores = torch.einsum("btod,btgd->btog", query, coarse_key_flat)
        scores = scores / math.sqrt(self.locator_dim)
        route_prior_full = None
        if previous_route_indices is not None:
            if not self.correctable_route_carry:
                raise ValueError("previous route supplied while route carry is disabled")
            if previous_route_attention is None:
                raise ValueError("route carry requires previous route attention")
            expected_prefix = (batch, frames, objects)
            if tuple(previous_route_indices.shape[:3]) != expected_prefix:
                raise ValueError("previous route indices must start with [B,T,N]")
            if tuple(previous_route_attention.shape) != tuple(previous_route_indices.shape):
                raise ValueError("previous route indices and attention must match")
            if torch.any(
                (previous_route_indices < 0)
                | (previous_route_indices >= height * width)
            ):
                raise ValueError("previous route contains an out-of-grid index")
            if not torch.isfinite(previous_route_attention).all():
                raise ValueError("previous route attention must be finite")
            if torch.any(previous_route_attention < 0):
                raise ValueError("previous route attention must be non-negative")
            route_prior_full = scores.new_zeros(
                batch, frames, objects, height * width
            )
            route_prior_full.scatter_add_(
                -1,
                previous_route_indices,
                previous_route_attention.to(dtype=scores.dtype),
            )
            route_prior_full = torch.nn.functional.max_pool2d(
                route_prior_full.reshape(-1, 1, height, width),
                kernel_size=3,
                stride=1,
                padding=1,
            ).reshape(batch, frames, objects, height, width)
            route_prior_full = route_prior_full * object_valid[:, None, :, None, None]
            route_prior_coarse = route_prior_full.reshape(
                batch,
                frames,
                objects,
                coarse_height,
                scale_y,
                coarse_width,
                scale_x,
            ).amax(dim=(4, 6))
            scores = scores + self.route_prior_alpha * route_prior_coarse.flatten(-2)

        candidate_count = min(self.topk, coarse_height * coarse_width)
        _, coarse_indices = scores.topk(
            candidate_count, dim=-1
        )
        coarse_y = coarse_indices // coarse_width
        coarse_x = coarse_indices % coarse_width
        offset_y = torch.arange(scale_y, device=hidden.device)
        offset_x = torch.arange(scale_x, device=hidden.device)
        full_y = coarse_y[..., None, None] * scale_y + offset_y[None, None, None, None, :, None]
        full_x = coarse_x[..., None, None] * scale_x + offset_x[None, None, None, None, None, :]
        expanded_indices = (full_y * width + full_x).flatten(-3)
        # Refine the coarse pool with low-D full-grid keys first. Only the
        # final global Top-K positions read full-D Wan values and receive the
        # sparse residual, rather than all cells under every coarse candidate.
        expanded_keys = self._gather_candidates(
            full_key.flatten(2, 3), expanded_indices
        )
        expanded_scores = torch.einsum(
            "btod,btokd->btok", query, expanded_keys
        ) / math.sqrt(self.locator_dim)
        if route_prior_full is not None:
            expanded_prior = torch.gather(
                route_prior_full.flatten(-2), -1, expanded_indices
            )
            expanded_scores = (
                expanded_scores + self.route_prior_alpha * expanded_prior
            )
        final_count = min(self.topk, expanded_indices.shape[-1])
        refined_scores, refined_offsets = expanded_scores.topk(
            final_count, dim=-1
        )
        candidate_indices = expanded_indices.gather(-1, refined_offsets)
        normalized_grid = normalized_full.flatten(2, 3)
        candidate_hidden = self._gather_candidates(
            normalized_grid, candidate_indices
        )
        candidate_values = self.value_projection(candidate_hidden)
        if self.support_gate is not None:
            candidate_keys = torch.gather(
                expanded_keys,
                -2,
                refined_offsets[..., None].expand(
                    *refined_offsets.shape, self.locator_dim
                ),
            )
            gate_logits = self.support_gate(
                torch.cat(
                    (query[..., None, :].expand_as(candidate_keys), candidate_keys),
                    dim=-1,
                )
            ).squeeze(-1)
            gate, hard_gate = _straight_through_support_gate(
                gate_logits, object_valid
            )
            candidate_attention = _normalize_gated_scores(refined_scores, gate)
            self.last_support_gate_logits = gate_logits
            self.last_support_gate_mask = hard_gate
        else:
            candidate_attention = refined_scores.softmax(dim=-1)
            candidate_attention = candidate_attention * object_valid[:, None, :, None]
        read = (candidate_attention[..., None] * candidate_values).sum(dim=-2)
        slots = queries + self.read_projection(read)
        slots = slots * object_valid[:, None, :, None]

        location_logits = (scores + self.location_bias).reshape(
            batch, frames, objects, coarse_height, coarse_width
        )
        return (
            slots,
            location_logits,
            candidate_indices,
            candidate_attention,
            coarse_key_flat,
        )


class IndependentLocalWriterRouter(nn.Module):
    """Predict one independent, spatially local write support per object-time.

    The Writer deliberately does not consume the Reader's candidate indices or
    weights.  It first scores the complete Wan token grid from the updated
    object state, chooses one global seed, and then limits sparse writeback to
    the configurable neighbourhood around that seed.  The seed is a routing
    reference rather than a hand-coded object centroid.
    """

    def __init__(
        self,
        wan_dim: int,
        object_dim: int,
        locator_dim: int,
        topk: int,
        radius: int = 1,
    ):
        super().__init__()
        self.topk = int(topk)
        self.radius = int(radius)
        if self.topk <= 0:
            raise ValueError("writer topk must be positive")
        if self.radius < 0:
            raise ValueError("writer radius must be non-negative")
        self.input_norm = nn.LayerNorm(wan_dim)
        self.query_norm = nn.LayerNorm(object_dim)
        self.query_projection = nn.Linear(object_dim, locator_dim)
        self.key_projection = nn.Linear(wan_dim, locator_dim)
        self.locator_dim = int(locator_dim)
        self.hard_support_enabled = _environment_flag(
            "PHYSICAL_WM_HARD_SUPPORT", False
        )
        self.correctable_route_carry = _environment_flag(
            "PHYSICAL_WM_CORRECTABLE_ROUTE_CARRY", False
        )
        self.support_gate = None
        if self.hard_support_enabled:
            with torch.random.fork_rng(devices=[]):
                self.support_gate = _mlp(locator_dim * 2, locator_dim, 1)
        if self.support_gate is not None:
            _zero_last(self.support_gate)
        self.last_support_gate_logits: Optional[torch.Tensor] = None
        self.last_support_gate_mask: Optional[torch.Tensor] = None
        self.last_support_gate_allowed_mask: Optional[torch.Tensor] = None
        support_mass = os.environ.get(
            "PHYSICAL_WM_WRITER_SUPPORT_MASS", "1.0"
        ).strip()
        self.support_mass = float(support_mass)
        if not 0.0 < self.support_mass <= 1.0:
            raise ValueError(
                "PHYSICAL_WM_WRITER_SUPPORT_MASS must be in (0, 1]"
            )

    def forward(
        self,
        hidden: torch.Tensor,
        updated_objects: torch.Tensor,
        object_valid: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        anchor_indices: Optional[torch.Tensor] = None,
        anchor_attention: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, times, objects, _ = updated_objects.shape
        if times != frames or hidden.shape[1] != frames * height * width:
            raise ValueError("writer inputs do not match the Wan token grid")
        if tuple(object_valid.shape) != (batch, objects):
            raise ValueError("writer object_valid must be [B,N]")

        self.last_support_gate_logits = None
        self.last_support_gate_mask = None
        self.last_support_gate_allowed_mask = None
        normalized = self.input_norm(hidden).reshape(
            batch, frames, height * width, hidden.shape[-1]
        )
        keys = self.key_projection(normalized)
        queries = self.query_projection(self.query_norm(updated_objects))
        scores = torch.einsum("btod,btgd->btog", queries, keys)
        scores = scores / math.sqrt(self.locator_dim)

        # Exactly one global seed is selected for each valid object-time.  The
        # following Top-K is restricted to its local square, so two distant
        # score modes cannot both receive a non-zero write weight.
        if anchor_indices is not None:
            if not self.correctable_route_carry:
                raise ValueError("Writer anchor supplied while route carry is disabled")
            if anchor_attention is None or anchor_attention.shape != anchor_indices.shape:
                raise ValueError("Writer anchor indices and attention must match")
            if tuple(anchor_indices.shape[:3]) != (batch, times, objects):
                raise ValueError("Writer anchors must start with [B,T,N]")
            anchor_weight = anchor_attention.float()
            anchor_mass = anchor_weight.sum(dim=-1).clamp_min(1.0e-6)
            anchor_y = torch.div(anchor_indices, width, rounding_mode="floor").float()
            anchor_x = (anchor_indices % width).float()
            seed_y = ((anchor_weight * anchor_y).sum(dim=-1) / anchor_mass).round().long()
            seed_x = ((anchor_weight * anchor_x).sum(dim=-1) / anchor_mass).round().long()
            seed_y = seed_y.clamp(0, height - 1)
            seed_x = seed_x.clamp(0, width - 1)
            seed_indices = seed_y * width + seed_x
        else:
            seed_indices = scores.argmax(dim=-1)
            seed_y = seed_indices // width
            seed_x = seed_indices % width
        cell_indices = torch.arange(height * width, device=hidden.device)
        cell_y = cell_indices // width
        cell_x = cell_indices % width
        allowed = (
            (cell_y - seed_y[..., None]).abs() <= self.radius
        ) & ((cell_x - seed_x[..., None]).abs() <= self.radius)
        masked_scores = scores.masked_fill(
            ~allowed, torch.finfo(scores.dtype).min
        )
        candidate_count = min(self.topk, height * width)
        selected_scores, candidate_indices = masked_scores.topk(
            candidate_count, dim=-1
        )
        if self.support_gate is not None:
            candidate_keys = ParallelSparseObjectRouter._gather_candidates(
                keys, candidate_indices
            )
            gate_logits = self.support_gate(
                torch.cat(
                    (
                        queries[..., None, :].expand_as(candidate_keys),
                        candidate_keys,
                    ),
                    dim=-1,
                )
            ).squeeze(-1)
            selected_allowed = torch.gather(allowed, -1, candidate_indices)
            gate_logits = gate_logits.masked_fill(
                ~selected_allowed, torch.finfo(gate_logits.dtype).min
            )
            gate, hard_gate = _straight_through_support_gate(
                gate_logits, object_valid
            )
            candidate_attention = _normalize_gated_scores(selected_scores, gate)
            self.last_support_gate_logits = gate_logits
            self.last_support_gate_mask = hard_gate
            self.last_support_gate_allowed_mask = selected_allowed
        else:
            candidate_attention = selected_scores.softmax(dim=-1)
        if self.support_mass < 1.0:
            candidate_attention = _truncate_attention_to_prefix_mass(
                candidate_attention, self.support_mass
            )
        candidate_attention = (
            candidate_attention * object_valid[:, None, :, None]
        )
        logits = scores.reshape(batch, frames, objects, height, width)
        return logits, candidate_indices, candidate_attention, seed_indices


class IdentityCompetitiveTemplateRouter(ParallelSparseObjectRouter):
    """Causal identity routing on a 2x locator grid above Wan tokens.

    The first-frame simulator mask is the immutable identity/template anchor.
    At later latent times, bilinearly lifted low-dimensional Wan keys compete
    on the 2x mask grid across valid object slots plus one parameter-free
    locator background.  The resulting per-object center translates exactly
    one copy of the first-frame mask.  That single template is then projected
    by deterministic area integration to the Wan token grid for both reading
    and residual writeback.  No future mask, velocity, confidence state, or
    contact label enters the forward path.
    """

    def __init__(
        self,
        wan_dim: int,
        object_dim: int,
        locator_dim: int,
        topk: int,
        max_times: int = 13,
    ):
        # Retain the historical parameter names/shapes so the unchanged
        # temporal, interaction, and condition paths can be warm-started.  The
        # old Top-K parameter is kept only as part of that constructor contract;
        # this router never performs sparse independent Top-K selection.
        super().__init__(
            wan_dim,
            object_dim,
            locator_dim,
            topk,
            max_times=max_times,
            previous_frame_tracking=True,
        )
        # One bounded scalar mixes the previous-frame appearance read into the
        # permanent first-frame identity anchor.  This is not an object
        # confidence/no-write gate and cannot turn conditioning off.
        self.tracking_mix_logit = nn.Parameter(torch.zeros(()))
        self.joint_slot_competition = True
        self.single_center_template_writeback = True
        self.high_resolution_locator = True
        self.wan_write_grid_projection = True

    @staticmethod
    def _spatial_centers(weights: torch.Tensor) -> torch.Tensor:
        """Return deterministic normalized (x, y) soft centers."""
        height, width = weights.shape[-2:]
        dtype, device = weights.dtype, weights.device
        y = torch.linspace(0.0, 1.0, height, dtype=dtype, device=device)
        x = torch.linspace(0.0, 1.0, width, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        mass = weights.sum(dim=(-2, -1)).clamp_min(1e-6)
        cx = (weights * xx).sum(dim=(-2, -1)) / mass
        cy = (weights * yy).sum(dim=(-2, -1)) / mass
        return torch.stack((cx, cy), dim=-1)

    @staticmethod
    def _translate_template(
        template: torch.Tensor,
        initial_centers: torch.Tensor,
        current_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Translate one mask template without learning scale or deformation."""
        batch, objects, height, width = template.shape
        dtype, device = template.dtype, template.device
        y = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
        x = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        base_grid = torch.stack((xx, yy), dim=-1)
        grid = base_grid[None, None].expand(
            batch, objects, height, width, 2
        ).clone()
        # grid_sample maps each output coordinate back into the input.  A
        # positive desired output shift therefore subtracts the displacement
        # from the sampling grid.  Centers use [0,1], hence the factor of two.
        displacement = 2.0 * (current_centers - initial_centers)
        grid[..., 0] -= displacement[..., 0, None, None]
        grid[..., 1] -= displacement[..., 1, None, None]
        translated = torch.nn.functional.grid_sample(
            template.reshape(batch * objects, 1, height, width),
            grid.reshape(batch * objects, height, width, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return translated.reshape(batch, objects, height, width)

    @staticmethod
    def _joint_object_ownership(
        object_logits: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Compete all valid slots against one parameter-free background."""
        invalid_logit = torch.finfo(object_logits.dtype).min
        masked = object_logits.masked_fill(~valid[..., None], invalid_logit)
        background = masked.new_zeros(masked.shape[0], 1, masked.shape[-1])
        return torch.cat((masked, background), dim=1).softmax(dim=1)[
            :, : masked.shape[1]
        ] * valid[..., None]

    @staticmethod
    def _lift_locator_grid(
        grid: torch.Tensor,
        output_height: int,
        output_width: int,
    ) -> torch.Tensor:
        """Bilinearly lift low-D Wan keys without changing the backbone."""
        batch, frames, height, width, channels = grid.shape
        lifted = torch.nn.functional.interpolate(
            grid.permute(0, 1, 4, 2, 3).reshape(
                batch * frames, channels, height, width
            ),
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=True,
        )
        return lifted.reshape(
            batch, frames, channels, output_height, output_width
        ).permute(0, 1, 3, 4, 2)

    @staticmethod
    def _project_template_to_wan(
        template: torch.Tensor,
        height: int,
        width: int,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Area-project one high-res template to the Wan read/write grid."""
        batch, objects = template.shape[:2]
        projected = torch.nn.functional.interpolate(
            template.reshape(batch * objects, 1, *template.shape[-2:]),
            size=(height, width),
            mode="area",
        ).reshape(batch, objects, height, width)
        return (
            projected
            / projected.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            * valid[..., None, None]
        )

    def forward(
        self,
        hidden: torch.Tensor,
        first_frame_masks: torch.Tensor,
        object_valid: torch.Tensor,
        frames: int,
        height: int,
        width: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, objects, mask_height, mask_width = first_frame_masks.shape
        if hidden.shape[1] != frames * height * width:
            raise ValueError("identity router requires plain Wan video tokens")
        if (mask_height, mask_width) != (2 * height, 2 * width):
            raise ValueError(
                "identity router requires a 2x locator mask grid above Wan: "
                f"mask={(mask_height, mask_width)} Wan={(height, width)}"
            )
        if frames > self.time_embedding.num_embeddings:
            raise ValueError(f"too many latent times: {frames}")
        valid = object_valid.to(dtype=torch.bool)
        mask_mass = first_frame_masks.sum(dim=(-2, -1))
        if torch.any(valid & (mask_mass <= 1e-6)):
            raise ValueError("every valid object mask must contain visible pixels")

        normalized = self.input_norm(hidden).reshape(
            batch, frames, height, width, hidden.shape[-1]
        )
        locator_grid = self._lift_locator_grid(
            self.key_projection(normalized), mask_height, mask_width
        ).flatten(2, 3)
        value_grid = self.value_projection(normalized).flatten(2, 3)
        template = first_frame_masks * valid[..., None, None]
        initial_wan_routing = self._project_template_to_wan(
            template, height, width, valid
        )
        raw_anchors = torch.einsum(
            "bohw,bhwd->bod", initial_wan_routing, normalized[:, 0]
        )
        anchors = self.value_projection(raw_anchors) + self.position_encoder(
            self._mask_geometry(first_frame_masks)
        )
        initial_centers = self._spatial_centers(template)
        time_ids = torch.arange(frames, device=hidden.device)

        slots_per_time = []
        logits_per_time = []
        routing_per_time = []
        previous_read = None
        for time_index in range(frames):
            query_state = anchors + self.time_embedding(time_ids[time_index])
            if previous_read is not None:
                tracking_mix = torch.sigmoid(self.tracking_mix_logit)
                query_state = query_state + tracking_mix * self.read_projection(
                    previous_read
                )
            query = self.query_projection(self.query_norm(query_state))
            object_logits = torch.einsum(
                "bod,bgd->bog", query, locator_grid[:, time_index]
            ) / math.sqrt(self.locator_dim)
            object_logits = object_logits + self.location_bias
            # One zero-logit locator background/dustbin participates only in
            # spatial ownership.  It is distinct from the projection-level
            # strict-zero NULL K/V used by the later interaction attention.
            ownership = self._joint_object_ownership(object_logits, valid)
            ownership_map = ownership.reshape(
                batch, objects, mask_height, mask_width
            )

            if time_index == 0:
                routing_map = initial_wan_routing
                location_probability = template.clamp(0.0, 1.0)
            else:
                center_weights = ownership_map / ownership_map.sum(
                    dim=(-2, -1), keepdim=True
                ).clamp_min(1e-6)
                current_centers = self._spatial_centers(center_weights)
                translated = self._translate_template(
                    template, initial_centers, current_centers
                )
                # Competition determines only one center per object.  Once the
                # center is chosen, both read and write use exactly the same
                # translated t0 template; ownership must not reshape or
                # reweight the template internally.
                location_probability = translated.clamp(0.0, 1.0)
                routing_map = self._project_template_to_wan(
                    translated, height, width, valid
                )
            location_logits = torch.logit(
                location_probability.clamp(min=1e-6, max=1.0 - 1e-6)
            )

            routing = routing_map.flatten(-2)
            read = torch.einsum(
                "bog,bgd->bod", routing, value_grid[:, time_index]
            )
            slot = (query_state + self.read_projection(read)) * valid[..., None]
            slots_per_time.append(slot)
            logits_per_time.append(location_logits)
            routing_per_time.append(routing)
            previous_read = read

        return (
            torch.stack(slots_per_time, dim=1),
            torch.stack(logits_per_time, dim=1),
            None,
            torch.stack(routing_per_time, dim=1),
            locator_grid,
        )


class ObjectInteractionBlock(nn.Module):
    """One object-query attention over objects plus two fixed background K/V."""

    def __init__(self, object_dim: int, num_heads: int, ffn_ratio: int = 2):
        super().__init__()
        self.attention_norm = nn.LayerNorm(object_dim)
        self.attention = nn.MultiheadAttention(
            object_dim, num_heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(object_dim)
        self.ffn = _mlp(object_dim, object_dim * ffn_ratio, object_dim)

    def forward(
        self,
        objects: torch.Tensor,
        background: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> torch.Tensor:
        normalized_objects = self.attention_norm(objects)
        key_value = torch.cat((normalized_objects, background), dim=1)
        key_padding = torch.cat(
            (
                ~object_valid,
                torch.zeros(
                    object_valid.shape[0],
                    background.shape[1],
                    dtype=torch.bool,
                    device=object_valid.device,
                ),
            ),
            dim=1,
        )
        delta, _ = self.attention(
            normalized_objects,
            key_value,
            key_value,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        valid = object_valid[..., None]
        objects = objects + valid * delta
        objects = objects + valid * self.ffn(self.ffn_norm(objects))
        return objects * valid


class SparseObjectInteractionGroup(nn.Module):
    """One Wan insertion: route, pool, interact once, decode, and write back."""

    def __init__(
        self,
        wan_dim: int,
        object_dim: int,
        locator_dim: int,
        num_heads: int,
        topk: int,
    ):
        super().__init__()
        self.condition_encoder = SparsePhysicsResidualEncoder(object_dim)
        self.router = ParallelSparseObjectRouter(
            wan_dim, object_dim, locator_dim, topk
        )
        self.temporal_norm = nn.LayerNorm(object_dim)
        self.temporal_score = nn.Linear(object_dim, 1)
        self.background_queries = nn.Parameter(torch.randn(2, locator_dim) * 0.02)
        self.background_value = nn.Linear(locator_dim, object_dim)
        self.background_norm = nn.LayerNorm(object_dim)
        self.interaction = ObjectInteractionBlock(object_dim, num_heads)
        self.time_decoder = _mlp(object_dim, object_dim * 2, object_dim)
        # Project only O object vectors, then scatter them to the selected Wan
        # positions. A bias would create a global residual at empty positions.
        self.output_projection = nn.Linear(object_dim, wan_dim, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        self.noise_gate = _mlp(1, 32, 1)
        self.residual_scale = nn.Parameter(torch.ones(()))

    def _pool_background(
        self,
        locator_grid: torch.Tensor,
        location_logits: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid = object_valid[:, None, :, None, None]
        occupancy = (
            torch.sigmoid(location_logits) * valid
        ).amax(dim=2).flatten(1).detach()
        flattened = locator_grid.flatten(1, 2)
        scores = torch.einsum("qd,bgd->bqg", self.background_queries, flattened)
        scores = scores / math.sqrt(flattened.shape[-1])
        scores = scores + (1.0 - occupancy).clamp_min(1e-4).log()[:, None]
        attention = scores.softmax(dim=-1)
        pooled_low_dim = torch.einsum("bqg,bgd->bqd", attention, flattened)
        background = self.background_value(pooled_low_dim)
        return self.background_norm(background)

    def forward(
        self,
        hidden: torch.Tensor,
        first_frame_masks: torch.Tensor,
        object_valid: torch.Tensor,
        physics_residual: torch.Tensor,
        timestep: torch.Tensor,
        frames: int,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            slots,
            location_logits,
            candidate_indices,
            candidate_attention,
            locator_grid,
        ) = self.router(
            hidden,
            first_frame_masks,
            object_valid,
            frames,
            height,
            width,
        )
        temporal_weight = self.temporal_score(
            self.temporal_norm(slots)
        ).squeeze(-1).softmax(dim=1)
        pooled = torch.einsum("bto,btod->bod", temporal_weight, slots)
        pooled = pooled * object_valid[..., None]
        conditioned = pooled + physics_residual
        background = self._pool_background(
            locator_grid, location_logits, object_valid
        )
        updated = self.interaction(conditioned, background, object_valid)

        decoded = self.time_decoder(slots + updated[:, None])
        decoded = decoded * object_valid[:, None, :, None]
        batch, times, objects, candidates = candidate_indices.shape
        grid_cells = height * width
        denominator = candidate_attention.new_zeros(batch, times, grid_cells)
        flattened_indices = candidate_indices.reshape(
            batch, times, objects * candidates
        )
        denominator.scatter_add_(
            -1,
            flattened_indices,
            candidate_attention.reshape(batch, times, objects * candidates),
        )
        candidate_denominator = self.router._gather_candidates(
            denominator, candidate_indices
        ).clamp_min(1.0)
        normalized_write = candidate_attention / candidate_denominator
        projected_objects = self.output_projection(decoded)
        candidate_residuals = (
            normalized_write[..., None] * projected_objects[..., None, :]
        ).reshape(batch, times, objects * candidates, hidden.shape[-1])
        projected_grid = hidden.new_zeros(
            batch, times, grid_cells, hidden.shape[-1], dtype=projected_objects.dtype
        )
        projected_grid.scatter_add_(
            2,
            flattened_indices[..., None].expand(
                batch, times, objects * candidates, hidden.shape[-1]
            ),
            candidate_residuals,
        )
        time_mask = projected_grid.new_ones(1, times, 1, 1)
        time_mask[:, 0] = 0
        projected = (projected_grid * time_mask).reshape(
            batch, times * grid_cells, hidden.shape[-1]
        )
        normalized_timestep = (
            timestep.float().reshape(hidden.shape[0], -1).mean(dim=1, keepdim=True)
            / 1000.0
        ).clamp(0.0, 1.0)
        gate = torch.sigmoid(self.noise_gate(normalized_timestep))[:, None]
        hidden = hidden + (
            gate * self.residual_scale * projected
        ).to(dtype=hidden.dtype)
        return hidden, location_logits


class SparseObjectInteractionAdapter(nn.Module):
    """Configured unshared single-pass groups for a 30-block Wan backbone."""

    def __init__(
        self,
        wan_dim: int,
        injection_blocks: Iterable[int] = (2, 6, 10, 14, 18, 22, 25, 27),
        object_dim: int = 512,
        locator_dim: int = 128,
        num_heads: int = 8,
        topk: int = 8,
    ):
        super().__init__()
        self.injection_blocks = tuple(int(index) for index in injection_blocks)
        if (
            not self.injection_blocks
            or len(set(self.injection_blocks)) != len(self.injection_blocks)
            or any(index < 0 for index in self.injection_blocks)
        ):
            raise ValueError("v2 requires non-empty distinct Wan insertion blocks")
        if object_dim % num_heads:
            raise ValueError("object_dim must be divisible by num_heads")
        self.groups = nn.ModuleDict(
            {
                str(index): SparseObjectInteractionGroup(
                    wan_dim, object_dim, locator_dim, num_heads, topk
                )
                for index in self.injection_blocks
            }
        )

    @staticmethod
    def _batched(
        value: torch.Tensor,
        batch: int,
        name: str,
        unbatched_ndim: int,
    ) -> torch.Tensor:
        if value.ndim == 0:
            raise ValueError(f"{name} must not be scalar")
        if value.ndim == unbatched_ndim:
            value = value.unsqueeze(0)
        if value.shape[0] == 1 and batch > 1:
            value = value.expand((batch,) + tuple(value.shape[1:]))
        if value.shape[0] != batch:
            raise ValueError(f"{name} batch must be {batch}, got {value.shape[0]}")
        return value

    def forward(
        self,
        block_index: int,
        hidden: torch.Tensor,
        condition: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        frames: int,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = str(block_index)
        if key not in self.groups:
            raise KeyError(block_index)
        missing = set(FORWARD_CONDITION_KEYS) - set(condition)
        if missing:
            raise ValueError(f"sparse object condition missing {sorted(missing)}")

        parameter = next(self.parameters())
        hidden_fp32 = hidden.to(dtype=parameter.dtype)
        batch = hidden.shape[0]
        unbatched_ndims = {
            "first_frame_masks": 3,
            "object_valid_mask": 1,
            "force": 2,
            "force_present": 1,
            "mu": 1,
            "mu_present": 1,
            "restitution": 1,
            "restitution_present": 1,
        }
        tensors = {
            name: self._batched(
                condition[name], batch, name, unbatched_ndims[name]
            ).to(hidden.device)
            for name in FORWARD_CONDITION_KEYS
        }
        first_masks = tensors["first_frame_masks"].to(dtype=hidden_fp32.dtype)
        if first_masks.ndim != 4:
            raise ValueError("first_frame_masks must be [B,N,Hc,Wc]")
        objects = first_masks.shape[1]
        expected = {
            "object_valid_mask": (batch, objects),
            "force": (batch, objects, 3),
            "force_present": (batch, objects),
            "mu": (batch, objects),
            "mu_present": (batch, objects),
            "restitution": (batch, objects),
            "restitution_present": (batch, objects),
        }
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(tensors[name].shape)}")
        if not all(torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("sparse object forward inputs must be finite")
        object_valid = tensors["object_valid_mask"] > 0.5
        presence = {
            name: tensors[name].to(dtype=hidden_fp32.dtype)
            for name in ("force_present", "mu_present", "restitution_present")
        }
        for name, value in presence.items():
            if torch.any((value < 0) | (value > 1)):
                raise ValueError(f"{name} must be in [0,1]")
            if torch.any((value > 0.5) & ~object_valid):
                raise ValueError(f"{name} cannot enable a padded object")
        force_input = tensors["force"]
        force = force_input.to(dtype=hidden_fp32.dtype)
        mu = tensors["mu"].to(dtype=hidden_fp32.dtype)
        restitution = tensors["restitution"].to(dtype=hidden_fp32.dtype)
        if torch.any((first_masks < 0) | (first_masks > 1)):
            raise ValueError("first_frame_masks must be in [0,1]")
        # Each admitted source keeps its own frozen normalization.  Do not
        # impose one legacy dataset's 360/340 upper bound on mixed-source F.
        if torch.any(force[..., 0] < 0):
            raise ValueError("normalized force magnitude must be nonnegative")
        active_force = (presence["force_present"] > 0.5) & (force[..., 0] > 0)
        if active_force.any():
            norms = force[..., 1:][active_force].norm(dim=-1)
            if torch.any((norms < 0.9) | (norms > 1.1)):
                raise ValueError("active force direction norm must be near one")
        if torch.any((mu < 0) | (mu > 2)):
            raise ValueError("mu must be in [0,2]")
        if torch.any((restitution < 0) | (restitution > 1)):
            raise ValueError("restitution must be in [0,1]")

        group = self.groups[key]
        physics_residual = group.condition_encoder(
            force,
            presence["force_present"],
            mu,
            presence["mu_present"],
            restitution,
            presence["restitution_present"],
            object_valid.to(dtype=hidden_fp32.dtype),
        )
        hidden_fp32, logits = group(
            hidden_fp32,
            first_masks,
            object_valid,
            physics_residual,
            timestep,
            frames,
            height,
            width,
        )
        return hidden_fp32.to(dtype=hidden.dtype), logits


class SparseUnaryTemporalResidualEncoder(nn.Module):
    """Encode object-bound F/g residuals before same-object temporal attention.

    Both inputs use ``[normalized magnitude, direction x, direction y]`` and
    explicit presence masks.  Without a schedule, F is an initial intervention
    at latent time zero.  With ``force_schedule[B,T,N]``, the same physical F
    is gated at every latent time by its exogenous per-slot duty fraction.
    Gravity remains active at every latent time.  Numeric zero and absence stay
    distinct for both quantities.
    """

    def __init__(self, object_dim: int):
        super().__init__()
        self.force = _mlp(3, object_dim, object_dim)
        self.gravity = _mlp(3, object_dim, object_dim)
        _zero_last(self.force)
        _zero_last(self.gravity)
        self.mass_enabled = os.environ.get("PHYSICAL_WM_OBJECT_MASS", "0") == "1"
        if self.mass_enabled:
            self.mass_encoder_kind = os.environ.get("PHYSICAL_WM_MASS_ENCODER", "linear")
            if self.mass_encoder_kind == "mlp":
                self.mass = _mlp(1, object_dim, object_dim)
                _zero_last(self.mass)
            elif self.mass_encoder_kind == "linear":
                self.mass = nn.Linear(1, object_dim)
                nn.init.zeros_(self.mass.weight)
                nn.init.zeros_(self.mass.bias)
            else:
                raise ValueError(f"unknown mass encoder: {self.mass_encoder_kind}")

    def forward(
        self,
        force: torch.Tensor,
        force_present: torch.Tensor,
        gravity: torch.Tensor,
        gravity_present: torch.Tensor,
        object_valid: torch.Tensor,
        times: int,
        force_schedule: Optional[torch.Tensor] = None,
        mass: Optional[torch.Tensor] = None,
        mass_present: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        force_magnitude = force[..., :1]
        gravity_magnitude = gravity[..., :1]
        force_vector = torch.cat(
            (force_magnitude, force_magnitude * force[..., 1:]), dim=-1
        )
        gravity_vector = torch.cat(
            (gravity_magnitude, gravity_magnitude * gravity[..., 1:]), dim=-1
        )
        valid = object_valid[..., None]
        force_residual = (
            valid * force_present[..., None] * self.force(force_vector)
        )
        if force_schedule is None:
            # Preserve the original temporal route when no schedule is
            # supplied: F is an initial intervention at latent time zero.
            force_schedule = force.new_zeros(
                force.shape[0], times, force.shape[1]
            )
            force_schedule[:, 0] = 1.0
        elif tuple(force_schedule.shape) != (force.shape[0], times, force.shape[1]):
            raise ValueError(
                "force_schedule must be [B,T,N], got "
                f"{tuple(force_schedule.shape)}"
            )
        force_schedule = force_schedule.to(dtype=force_residual.dtype)
        scheduled_force = force_schedule[..., None] * force_residual[:, None]
        gravity_residual = (
            valid * gravity_present[..., None] * self.gravity(gravity_vector)
        )
        residual = gravity_residual[:, None].expand(-1, times, -1, -1).clone()
        residual = residual + scheduled_force
        if self.mass_enabled:
            if mass is None or mass_present is None:
                raise ValueError("object mass requires mass and mass_present")
            mass_residual = valid * mass_present[..., None] * self.mass(mass[..., None])
            residual = residual + mass_residual[:, None]
        return residual


class SameObjectTemporalBlock(nn.Module):
    """Transformer block over time for one object at a time.

    With ``causal=True`` a latent time may only attend to itself and earlier
    latent times.  This restricts information visibility inside the physics
    branch only; the Wan backbone, the Reader and the write-back all stay
    bidirectional, so every frame is still denoised jointly at the same noise
    level.  The diagonal stays visible, so no query row is fully masked.
    """

    def __init__(
        self,
        object_dim: int,
        num_heads: int,
        ffn_ratio: int = 2,
        causal: bool = False,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(object_dim)
        self.attention = nn.MultiheadAttention(
            object_dim, num_heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(object_dim)
        self.ffn = _mlp(object_dim, object_dim * ffn_ratio, object_dim)
        self.causal = bool(causal)
        # Evaluation-only override.  Plain Python state keeps the state_dict
        # key set identical whether or not the ablation is used.
        self.causal_runtime_enabled = True
        self.attention_enabled = h2_ablation_mode() != "no_temporal_attention"

    def _causal_mask(
        self, times: int, device: torch.device
    ) -> torch.Tensor:
        return torch.ones(
            times, times, dtype=torch.bool, device=device
        ).triu(diagonal=1)

    def forward(
        self,
        slots: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, times, objects, dimension = slots.shape
        sequence = slots.permute(0, 2, 1, 3).reshape(
            batch * objects, times, dimension
        )
        if self.attention_enabled:
            normalized = self.attention_norm(sequence)
            attention_mask = (
                self._causal_mask(times, sequence.device)
                if self.causal and self.causal_runtime_enabled
                else None
            )
            delta, _ = self.attention(
                normalized, normalized, normalized,
                need_weights=False, attn_mask=attention_mask,
            )
            sequence = sequence + delta
        sequence = sequence + self.ffn(self.ffn_norm(sequence))
        sequence = sequence.reshape(batch, objects, times, dimension).permute(
            0, 2, 1, 3
        )
        return sequence * object_valid[:, None, :, None]


class TemporalObjectInteractionBlock(nn.Module):
    """Per-time object attention with factorized e/mu sender/receiver modulation.

    Q and K depend only on the normalized object state.  Restitution (e) and
    friction (mu) affect the object Value sent by each object and the residual
    strength applied to the message received by each object.  In the explicit
    no-interaction route, one pooled support remains a real candidate, self is
    masked, and a parameter-free NULL is appended after projection with
    exactly-zero K/V.  Legacy Temporal behavior is retained when disabled.
    """

    def __init__(
        self,
        object_dim: int,
        num_heads: int,
        ffn_ratio: int = 2,
        no_interaction: bool = False,
        mu_message_gate_alpha: Optional[float] = None,
    ):
        super().__init__()
        if object_dim % num_heads:
            raise ValueError("object_dim must be divisible by num_heads")
        self.object_dim = object_dim
        self.num_heads = num_heads
        self.head_dim = object_dim // num_heads
        self.no_interaction = bool(no_interaction)
        self.mu_message_gate_alpha = (
            None
            if mu_message_gate_alpha is None
            else float(mu_message_gate_alpha)
        )
        if (
            self.mu_message_gate_alpha is not None
            and not 0.0 < self.mu_message_gate_alpha <= 1.0
        ):
            raise ValueError("mu message gate alpha must be in (0, 1]")
        self.attention_norm = nn.LayerNorm(object_dim)
        self.query_projection = nn.Linear(object_dim, object_dim)
        self.key_projection = nn.Linear(object_dim, object_dim)
        self.value_projection = nn.Linear(object_dim, object_dim)
        # A NULL-only aggregate must remain exactly zero.  A projection bias
        # would turn the zero Value into a non-zero interaction residual.
        self.output_projection = nn.Linear(
            object_dim, object_dim, bias=not self.no_interaction
        )
        self.e_modulation = _mlp(1, object_dim, object_dim * 3)
        self.mu_modulation = _mlp(1, object_dim, object_dim * 3)
        _zero_last(self.e_modulation)
        _zero_last(self.mu_modulation)
        self.ffn_norm = nn.LayerNorm(object_dim)
        self.ffn = _mlp(object_dim, object_dim * ffn_ratio, object_dim)
        self.capture_attention = False
        self.last_attention: Optional[torch.Tensor] = None
        self.last_qk_raw: Optional[dict[str, torch.Tensor]] = None
        # This is a separate evaluation receipt from attention capture.  It
        # remains off in the ordinary forward path and stores no graph state.
        self.capture_diagnostic_state = False
        self.last_interaction_message: Optional[torch.Tensor] = None
        self.last_e_modulation_effect: Optional[torch.Tensor] = None
        self.last_mu_modulation_effect: Optional[torch.Tensor] = None
        self.last_interaction_ffn_delta: Optional[torch.Tensor] = None
        self.last_interaction_total_delta: Optional[torch.Tensor] = None
        # Evaluation-only causal probe. Plain Python state deliberately keeps
        # checkpoints and the ordinary forward graph unchanged when unset.
        self.edge_causal_intervention: Optional[dict] = None
        self.last_edge_causal_receipt: Optional[dict] = None

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.reshape(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        objects: torch.Tensor,
        support: torch.Tensor,
        object_valid: torch.Tensor,
        mu: torch.Tensor,
        mu_present: torch.Tensor,
        restitution: torch.Tensor,
        restitution_present: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(objects)
        query = self._heads(self.query_projection(normalized))
        object_key = self._heads(self.key_projection(normalized))
        unmodulated_object_value = self.value_projection(normalized)
        object_value = unmodulated_object_value

        e_modulation = (
            restitution_present[..., None]
            * self.e_modulation(restitution[..., None])
        )
        mu_modulation = (
            mu_present[..., None] * self.mu_modulation(mu[..., None])
        )
        e_scale_v, e_shift_v, e_gate = torch.chunk(e_modulation, 3, dim=-1)
        mu_scale_v, mu_shift_v, mu_gate = torch.chunk(mu_modulation, 3, dim=-1)
        if self.mu_message_gate_alpha is not None:
            alpha = self.mu_message_gate_alpha
            # Unit slope at zero retains ordinary numeric sensitivity.  The
            # largest safe symmetric alpha is 1: (1 + gate) then stays in
            # [0, 2], so mu cannot invert the entire received object message.
            mu_gate = alpha * torch.tanh(mu_gate / alpha)
        scale_v = e_scale_v + mu_scale_v
        shift_v = e_shift_v + mu_shift_v
        delta_gate_message = e_gate + mu_gate
        e_delta_value = e_scale_v * unmodulated_object_value + e_shift_v
        mu_delta_value = mu_scale_v * unmodulated_object_value + mu_shift_v
        object_value = (1.0 + scale_v) * object_value + shift_v
        object_value = self._heads(object_value)
        e_delta_value_heads = self._heads(e_delta_value)
        mu_delta_value_heads = self._heads(mu_delta_value)

        if self.no_interaction and support.shape[1] != 1:
            raise ValueError("no-interaction route requires one support candidate")
        support_key = self._heads(self.key_projection(support))
        support_value = self._heads(self.value_projection(support))
        if self.no_interaction:
            # NULL is not an embedding passed through a biased projection.  Its
            # K/V are created at the final attention shape and remain zero.
            null_key = object_key.new_zeros(
                object_key.shape[0], object_key.shape[1], 1, object_key.shape[-1]
            )
            null_value = object_value.new_zeros(
                object_value.shape[0], object_value.shape[1], 1, object_value.shape[-1]
            )
            key = torch.cat((object_key, support_key, null_key), dim=2)
            value = torch.cat((object_value, support_value, null_value), dim=2)
        else:
            key = torch.cat((object_key, support_key), dim=2)
            value = torch.cat((object_value, support_value), dim=2)
        raw_attention_logits = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)
        attention_logits = raw_attention_logits
        invalid_keys = torch.cat(
            (
                ~object_valid,
                torch.zeros(
                    object_valid.shape[0],
                    support.shape[1] + int(self.no_interaction),
                    dtype=torch.bool,
                    device=object_valid.device,
                ),
            ),
            dim=-1,
        )[:, None, None, :]
        attention_logits = attention_logits.masked_fill(
            invalid_keys, torch.finfo(attention_logits.dtype).min
        )
        if self.no_interaction:
            self_mask = torch.eye(
                objects.shape[1],
                key.shape[2],
                dtype=torch.bool,
                device=objects.device,
            )[None, None]
            attention_logits = attention_logits.masked_fill(
                self_mask, torch.finfo(attention_logits.dtype).min
            )
        if self.capture_attention:
            self.last_qk_raw = {
                "query": query.detach().float().cpu(),
                "key": key.detach().float().cpu(),
                "raw_logits": raw_attention_logits.detach().float().cpu(),
                "masked_logits": attention_logits.detach().float().cpu(),
            }
        else:
            self.last_qk_raw = None
        diagnostic = self.edge_causal_intervention
        self.last_edge_causal_receipt = None
        selected_rows = None
        if diagnostic is not None:
            mode = diagnostic.get("mode")
            if mode not in {
                "delete_edge",
                "swap_sender",
                "ablate_e_path",
                "ablate_mu_path",
                "ablate_e_object_path",
                "ablate_mu_object_path",
                "oracle_timing_only",
                "oracle_timing_partner",
            }:
                raise ValueError(f"unsupported edge causal mode: {mode!r}")
            temporal_steps = diagnostic.get("temporal_steps")
            receiver = diagnostic.get("receiver_object")
            sender = diagnostic.get("sender_object")
            latent_times = diagnostic.get("latent_times")
            replacement_sender = diagnostic.get("replacement_sender_object")
            if (
                not isinstance(temporal_steps, int)
                or temporal_steps <= 0
                or objects.shape[0] % temporal_steps
                or not isinstance(receiver, int)
                or not isinstance(sender, int)
                or isinstance(receiver, bool)
                or isinstance(sender, bool)
                or not 0 <= receiver < objects.shape[1]
                or not 0 <= sender < objects.shape[1]
                or (
                    receiver == sender
                    and mode not in {
                        "ablate_e_object_path", "ablate_mu_object_path"
                    }
                )
            ):
                raise ValueError("invalid edge causal shape/object contract")
            if mode == "swap_sender" and (
                not isinstance(replacement_sender, int)
                or isinstance(replacement_sender, bool)
                or not 0 <= replacement_sender < objects.shape[1]
                or replacement_sender in {receiver, sender}
            ):
                raise ValueError("invalid replacement sender contract")
            try:
                latent_times = tuple(int(value) for value in latent_times)
            except (TypeError, ValueError) as error:
                raise ValueError("edge causal latent_times must be integers") from error
            if (
                (not latent_times and not mode.startswith("oracle_timing"))
                or len(set(latent_times)) != len(latent_times)
                or any(not 0 <= value < temporal_steps for value in latent_times)
            ):
                raise ValueError("edge causal latent_times are outside video time")
            row_time = torch.arange(objects.shape[0], device=objects.device) % temporal_steps
            selected_rows = torch.zeros_like(row_time, dtype=torch.bool)
            for latent_time in latent_times:
                selected_rows |= row_time == latent_time
            if mode == "delete_edge":
                attention_logits = attention_logits.clone()
                attention_logits[selected_rows, :, receiver, sender] = torch.finfo(
                    attention_logits.dtype
                ).min
            elif mode == "swap_sender":
                attention_logits = attention_logits.clone()
                sender_logits = attention_logits[
                    selected_rows, :, receiver, sender
                ].clone()
                replacement_logits = attention_logits[
                    selected_rows, :, receiver, replacement_sender
                ].clone()
                attention_logits[
                    selected_rows, :, receiver, sender
                ] = replacement_logits
                attention_logits[
                    selected_rows, :, receiver, replacement_sender
                ] = sender_logits
            elif mode in {"oracle_timing_only", "oracle_timing_partner"}:
                # The final key is the explicit zero K/V NULL alternative.
                # Outside the simulator-audited event window, the receiver
                # may attend only to NULL. The partner variant additionally
                # limits that window to NULL versus the true sender.
                attention_logits = attention_logits.clone()
                dtype_min = torch.finfo(attention_logits.dtype).min
                outside_rows = ~selected_rows
                attention_logits[outside_rows, :, receiver, :-1] = dtype_min
                if mode == "oracle_timing_partner":
                    sender_logits = attention_logits[
                        selected_rows, :, receiver, sender
                    ].clone()
                    attention_logits[selected_rows, :, receiver, :-1] = dtype_min
                    attention_logits[
                        selected_rows, :, receiver, sender
                    ] = sender_logits
        attention = attention_logits.softmax(dim=-1)
        if self.capture_attention:
            self.last_attention = attention.detach().float().cpu()
        else:
            self.last_attention = None
        message_heads = torch.matmul(attention, value)
        if diagnostic is not None and diagnostic["mode"] in {
            "ablate_e_path", "ablate_mu_path",
            "ablate_e_object_path", "ablate_mu_object_path",
        }:
            # Edge modes remove one sender->receiver contribution. Object modes
            # remove one physical object's outgoing parameter-conditioned Value
            # from every receiver. Q/K and the other parameter remain exact.
            parameter_delta = (
                e_delta_value_heads
                if diagnostic["mode"] in {
                    "ablate_e_path", "ablate_e_object_path"
                }
                else mu_delta_value_heads
            )
            sender_delta = -parameter_delta[:, :, sender]
            message_heads = message_heads.clone()
            if diagnostic["mode"].endswith("_object_path"):
                correction = attention[:, :, :, sender, None] * sender_delta[:, :, None]
                message_heads[selected_rows] += correction[selected_rows]
            else:
                correction = attention[:, :, receiver, sender, None] * sender_delta
                message_heads[selected_rows, :, receiver] += correction[selected_rows]
        message = message_heads.transpose(1, 2).reshape(
            objects.shape[0], objects.shape[1], self.object_dim
        )
        message = self.output_projection(message)
        effective_gate = delta_gate_message
        if diagnostic is not None and diagnostic["mode"] in {
            "ablate_e_path", "ablate_mu_path",
            "ablate_e_object_path", "ablate_mu_object_path",
        }:
            # Remove exactly the selected receiver-gate term at selected times.
            effective_gate = effective_gate.clone()
            parameter_gate = (
                e_gate
                if diagnostic["mode"] in {
                    "ablate_e_path", "ablate_e_object_path"
                }
                else mu_gate
            )
            effective_gate[selected_rows, receiver] -= parameter_gate[
                selected_rows, receiver
            ]
        message = (1.0 + effective_gate) * message
        self.last_e_modulation_effect = None
        self.last_mu_modulation_effect = None
        if self.capture_diagnostic_state and diagnostic is None:
            def message_with_parameter_removed(
                parameter_delta: torch.Tensor,
                remaining_gate: torch.Tensor,
            ) -> torch.Tensor:
                counterfactual_value = value.clone()
                counterfactual_value[:, :, :objects.shape[1]] -= parameter_delta
                counterfactual_heads = torch.matmul(attention, counterfactual_value)
                counterfactual = counterfactual_heads.transpose(1, 2).reshape(
                    objects.shape[0], objects.shape[1], self.object_dim
                )
                counterfactual = self.output_projection(counterfactual)
                return (1.0 + remaining_gate) * counterfactual

            no_e_message = message_with_parameter_removed(
                e_delta_value_heads, mu_gate
            )
            no_mu_message = message_with_parameter_removed(
                mu_delta_value_heads, e_gate
            )
            self.last_e_modulation_effect = (message - no_e_message).detach().float().cpu()
            self.last_mu_modulation_effect = (message - no_mu_message).detach().float().cpu()
        if diagnostic is not None:
            self.last_edge_causal_receipt = {
                "mode": diagnostic["mode"],
                "receiver_object": receiver,
                "sender_object": sender,
                **(
                    {"replacement_sender_object": replacement_sender}
                    if diagnostic["mode"] == "swap_sender"
                    else {}
                ),
                "temporal_steps": temporal_steps,
                "latent_times": latent_times,
                "selected_row_count": int(selected_rows.sum().item()),
                **(
                    {
                        "outside_row_count": int((~selected_rows).sum().item()),
                        "non_null_key_count": int(key.shape[2] - 1),
                    }
                    if diagnostic["mode"].startswith("oracle_timing")
                    else {}
                ),
                "attention_unchanged": diagnostic["mode"] in {
                    "ablate_e_path", "ablate_mu_path",
                    "ablate_e_object_path", "ablate_mu_object_path",
                },
                "object_wide_parameter_path": diagnostic["mode"].endswith(
                    "_object_path"
                ),
            }

        valid = object_valid[..., None]
        message_delta = valid * message
        after_message = objects + message_delta
        ffn_delta = valid * self.ffn(self.ffn_norm(after_message))
        output = (after_message + ffn_delta) * valid
        if self.capture_diagnostic_state:
            self.last_interaction_message = message_delta.detach().float().cpu()
            self.last_interaction_ffn_delta = ffn_delta.detach().float().cpu()
            self.last_interaction_total_delta = (output - objects).detach().float().cpu()
        else:
            self.last_interaction_message = None
            self.last_interaction_ffn_delta = None
            self.last_interaction_total_delta = None
        return output


class IndependentUndirectedEdgeInteractionBlock(nn.Module):
    """Independent undirected gates with direction-specific state messages.

    Objects and the single pooled support token use the same message functions.
    The legacy route uses one independent sigmoid relation gate per unordered
    pair. The three-contact route bypasses Q/K and predicts independent base,
    restitution, and friction gates from the two directed pair states. Neither
    route normalizes gates across partners or introduces a NULL edge.

    Restitution and friction are edge-local conditions. They read the same
    pair state but own fully separate encoders, AdaLN parameters, and residual
    output projections. Presence zero removes only that physical residual; it
    never disables the edge itself.
    """

    def __init__(
        self,
        object_dim: int,
        num_heads: int,
        ffn_ratio: int = 2,
        pair_event_supervision: bool = False,
        shared_contact_gate: bool = False,
        three_contact_gates: bool = False,
        always_on_object_object_gate: bool = False,
        shared_three_contact_gate: bool = False,
    ):
        super().__init__()
        if object_dim % num_heads:
            raise ValueError("object_dim must be divisible by num_heads")
        self.object_dim = int(object_dim)
        self.num_heads = int(num_heads)
        self.head_dim = object_dim // num_heads
        self.pair_event_supervision = bool(pair_event_supervision)
        self.shared_contact_gate = bool(shared_contact_gate)
        self.three_contact_gates = bool(three_contact_gates)
        self.always_on_object_object_gate = bool(
            always_on_object_object_gate
        )
        self.shared_three_contact_gate = bool(shared_three_contact_gate)
        if sum(
            (
                self.pair_event_supervision,
                self.shared_contact_gate,
                self.three_contact_gates,
            )
        ) > 1:
            raise ValueError(
                "legacy pair-event, shared-contact, and three-contact routes "
                "are mutually exclusive"
            )
        if self.always_on_object_object_gate and not self.three_contact_gates:
            raise ValueError(
                "always-on object-object gates require the three-contact route"
            )
        if self.shared_three_contact_gate and not self.three_contact_gates:
            raise ValueError(
                "shared three-contact gate requires the three-contact route"
            )
        if self.shared_three_contact_gate and self.always_on_object_object_gate:
            raise ValueError("shared and always-on gate ablations are exclusive")
        self.contact_head_enabled = (
            self.pair_event_supervision
            or self.shared_contact_gate
            or self.three_contact_gates
        )
        self.e_pair_event_loss_mode = os.environ.get(
            "PHYSICAL_WM_E_PAIR_EVENT_LOSS_MODE", "hard_slot"
        ).strip().lower()
        if self.e_pair_event_loss_mode not in {
            "hard_slot",
            "event_distribution_w1",
            "weak_pair",
            "weak_pair_crossblock_union",
            "coarse_window_pair",
        }:
            raise ValueError(
                "unsupported PHYSICAL_WM_E_PAIR_EVENT_LOSS_MODE"
            )
        self.node_norm = nn.LayerNorm(object_dim)
        self.query_projection = nn.Linear(object_dim, object_dim)
        self.key_projection = nn.Linear(object_dim, object_dim)
        self.sender_projection = nn.Linear(object_dim, object_dim)
        self.receiver_projection = nn.Linear(object_dim, object_dim)
        self.pair_norm = nn.LayerNorm(object_dim)
        self.base_output = nn.Linear(object_dim, object_dim, bias=False)
        if self.three_contact_gates:
            # The A/B experiment must differ only by the three new heads.
            # Construct them without advancing the process-global RNG so all
            # later common Reader/Interaction/Writer parameters remain byte
            # identical to the original independent-edge arm at the same seed.
            with torch.random.fork_rng(devices=[]):
                self.base_contact_head = _mlp(
                    object_dim * 2, object_dim, 1
                )
                if not self.shared_three_contact_gate:
                    self.e_contact_head = _mlp(
                        object_dim * 2, object_dim, 1
                    )
                    self.mu_contact_head = _mlp(
                        object_dim * 2, object_dim, 1
                    )
            heads = [self.base_contact_head]
            if not self.shared_three_contact_gate:
                heads.extend((self.e_contact_head, self.mu_contact_head))
            for head in heads:
                _zero_last(head)
            # These parameters remain in legacy checkpoints for byte-level
            # compatibility, but the three-contact route never evaluates QK.
            self.query_projection.requires_grad_(False)
            self.key_projection.requires_grad_(False)
        elif self.contact_head_enabled:
            # Dedicated symmetric pair gates.  They are independent of the
            # generic edge gate and see only current predicted state and
            # forward-only kinematics; simulator GT is loss-only.
            self.e_pair_event_head = nn.Linear(object_dim + 2, 1)
            nn.init.zeros_(self.e_pair_event_head.weight)
            nn.init.zeros_(self.e_pair_event_head.bias)
            if not self.shared_contact_gate:
                self.mu_pair_event_head = nn.Linear(object_dim + 2, 1)
                nn.init.zeros_(self.mu_pair_event_head.weight)
                nn.init.constant_(
                    self.e_pair_event_head.bias,
                    (
                        -4.63
                        if self.e_pair_event_loss_mode
                        == "weak_pair_crossblock_union"
                        else -2.5
                        if self.e_pair_event_loss_mode
                        in {"weak_pair", "coarse_window_pair"}
                        else 4.0
                    ),
                )
                nn.init.constant_(self.mu_pair_event_head.bias, 4.0)

        # The two physical axes intentionally share no learned weights.
        self.e_adaln = _mlp(1, object_dim, object_dim * 3)
        self.mu_adaln = _mlp(1, object_dim, object_dim * 3)
        self.e_output = nn.Linear(object_dim, object_dim, bias=False)
        self.mu_output = nn.Linear(object_dim, object_dim, bias=False)
        _zero_last(self.e_adaln)
        _zero_last(self.mu_adaln)

        self.output_projection = nn.Linear(object_dim, object_dim, bias=False)
        self.ffn_norm = nn.LayerNorm(object_dim)
        self.ffn = _mlp(object_dim, object_dim * ffn_ratio, object_dim)
        self.capture_edges = False
        self.last_edge_gate: Optional[torch.Tensor] = None
        self.last_edge_logit: Optional[torch.Tensor] = None
        self.last_edge_base_rms: Optional[torch.Tensor] = None
        self.last_edge_e_residual_rms: Optional[torch.Tensor] = None
        self.last_edge_mu_residual_rms: Optional[torch.Tensor] = None
        self.last_edge_gated_base_rms: Optional[torch.Tensor] = None
        self.last_edge_gated_e_residual_rms: Optional[torch.Tensor] = None
        self.last_edge_gated_mu_residual_rms: Optional[torch.Tensor] = None
        self.last_base_contact_logit: Optional[torch.Tensor] = None
        self.last_base_contact_gate: Optional[torch.Tensor] = None
        self.last_e_pair_event_logit: Optional[torch.Tensor] = None
        self.last_mu_pair_event_logit: Optional[torch.Tensor] = None
        self.last_e_pair_event_gate: Optional[torch.Tensor] = None
        self.last_mu_pair_event_gate: Optional[torch.Tensor] = None
        self.last_executed_base_contact_gate: Optional[torch.Tensor] = None
        self.last_executed_e_pair_event_gate: Optional[torch.Tensor] = None
        self.last_executed_mu_pair_event_gate: Optional[torch.Tensor] = None
        self.last_edge_base_aggregate_rms: Optional[torch.Tensor] = None
        self.last_edge_e_aggregate_rms: Optional[torch.Tensor] = None
        self.last_edge_mu_aggregate_rms: Optional[torch.Tensor] = None
        # Evaluation-only state-chain receipts.  These are plain runtime
        # attributes (no checkpoint state) and are populated only when the
        # continuous mediation diagnostic requests them.
        self.capture_diagnostic_state = False
        self.last_interaction_message: Optional[torch.Tensor] = None
        self.last_e_modulation_effect: Optional[torch.Tensor] = None
        self.last_mu_modulation_effect: Optional[torch.Tensor] = None
        self.last_interaction_ffn_delta: Optional[torch.Tensor] = None
        self.last_interaction_total_delta: Optional[torch.Tensor] = None
        # Evaluation-only causal probe.  It is unset for every training and
        # ordinary inference call, and therefore adds no checkpoint state.
        self.edge_causal_intervention: Optional[dict] = None
        self.last_edge_causal_receipt: Optional[dict] = None

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = value.shape
        return value.reshape(
            batch, nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)

    @staticmethod
    def _full_edge_tensor(value: torch.Tensor) -> torch.Tensor:
        """Expand object rows [B,N,N+1] to symmetric [B,N+1,N+1]."""
        if value.ndim != 3 or value.shape[2] != value.shape[1] + 1:
            raise ValueError("edge tensor must be [B,N,N+1]")
        batch, objects, _ = value.shape
        object_block = value[:, :, :objects]
        if not torch.allclose(object_block, object_block.transpose(1, 2)):
            raise ValueError("object-object edge tensor must be symmetric")
        full = value.new_zeros(batch, objects + 1, objects + 1)
        full[:, :objects, :] = value
        full[:, objects, :objects] = value[:, :, objects]
        return full

    @staticmethod
    def _physical_residual(
        pair_state: torch.Tensor,
        value: torch.Tensor,
        presence: torch.Tensor,
        encoder: nn.Sequential,
        output: nn.Linear,
    ) -> torch.Tensor:
        parameters = presence[..., None] * encoder(value[..., None])
        scale, shift, gate = torch.chunk(parameters, 3, dim=-1)
        delta = scale * pair_state + shift
        # At zero initialization the residual is exactly zero while the
        # scale/shift branch still receives gradients. Zeroing both this gate
        # and the residual output would make the entire physical path dead.
        return (2.0 * torch.sigmoid(gate)) * output(delta)

    @staticmethod
    def _undirected_head_logits(
        pair_state: torch.Tensor,
        head: nn.Module,
    ) -> torch.Tensor:
        """Evaluate one MLP once per unordered edge and mirror its scalar."""
        _, nodes, _, _ = pair_state.shape
        left, right = torch.triu_indices(
            nodes, nodes, offset=1, device=pair_state.device
        )
        pair_input = torch.cat(
            (pair_state[:, left, right], pair_state[:, right, left]), dim=-1
        )
        unique = head(pair_input).squeeze(-1)
        full = pair_state.new_zeros(pair_state.shape[:3])
        full[:, left, right] = unique
        full[:, right, left] = unique
        return full

    def forward(
        self,
        objects: torch.Tensor,
        support: torch.Tensor,
        object_valid: torch.Tensor,
        edge_mu: torch.Tensor,
        edge_mu_present: torch.Tensor,
        edge_restitution: torch.Tensor,
        edge_restitution_present: torch.Tensor,
        node_position: Optional[torch.Tensor] = None,
        temporal_steps: Optional[int] = None,
    ) -> torch.Tensor:
        if support.shape[1] != 1:
            raise ValueError("independent-edge route requires one support token")
        if objects.ndim != 3 or support.ndim != 3:
            raise ValueError("objects and support must be [B,N,D]")
        batch, object_count, dimension = objects.shape
        if dimension != self.object_dim or support.shape != (batch, 1, dimension):
            raise ValueError("object/support shape differs from edge block contract")
        if object_valid.shape != (batch, object_count):
            raise ValueError("object_valid must be [B,N]")
        if self.contact_head_enabled and not self.three_contact_gates:
            if node_position is None:
                raise ValueError("pair event gates require node_position [B,N+1,2]")
            if tuple(node_position.shape) != (batch, object_count + 1, 2):
                raise ValueError("pair event gates require node_position [B,N+1,2]")
            if (
                not isinstance(temporal_steps, int)
                or isinstance(temporal_steps, bool)
                or temporal_steps <= 0
                or batch % temporal_steps
            ):
                raise ValueError("pair event gates require a valid temporal_steps")
        elif self.three_contact_gates:
            if node_position is not None or temporal_steps is not None:
                raise ValueError(
                    "three-contact gates forbid position and temporal kinematics"
                )
        elif node_position is not None or temporal_steps is not None:
            raise ValueError("kinematic pair inputs supplied while supervision is off")

        nodes = torch.cat((objects, support), dim=1)
        normalized = self.node_norm(nodes)
        sender = self.sender_projection(normalized)[:, None, :, :]
        receiver = self.receiver_projection(normalized)[:, :, None, :]
        pair_state = self.pair_norm(sender + receiver)
        node_valid = torch.cat(
            (
                object_valid,
                torch.ones(
                    batch, 1, dtype=torch.bool, device=objects.device
                ),
            ),
            dim=1,
        )
        edge_valid = node_valid[:, :, None] & node_valid[:, None, :]
        diagonal = torch.eye(
            object_count + 1, dtype=torch.bool, device=objects.device
        )[None]
        edge_valid = edge_valid & ~diagonal
        base_contact_logit = None
        e_pair_event_logit = None
        mu_pair_event_logit = None
        if self.three_contact_gates:
            base_contact_logit = self._undirected_head_logits(
                pair_state, self.base_contact_head
            )
            if self.shared_three_contact_gate:
                e_pair_event_logit = base_contact_logit
                mu_pair_event_logit = base_contact_logit
            else:
                e_pair_event_logit = self._undirected_head_logits(
                    pair_state, self.e_contact_head
                )
                mu_pair_event_logit = self._undirected_head_logits(
                    pair_state, self.mu_contact_head
                )
            edge_logit = base_contact_logit
        else:
            query = self._heads(self.query_projection(normalized))
            key = self._heads(self.key_projection(normalized))
            directed_logit = torch.matmul(
                query, key.transpose(-1, -2)
            ) / math.sqrt(self.head_dim)
            edge_logit = (
                0.5 * (directed_logit + directed_logit.transpose(-1, -2))
            ).mean(dim=1)
        edge_gate = torch.sigmoid(edge_logit) * edge_valid.to(edge_logit.dtype)
        predicted_edge_gate = edge_gate
        object_object_valid = torch.zeros_like(edge_valid)
        object_object_valid[:, :object_count, :object_count] = edge_valid[
            :, :object_count, :object_count
        ]
        if self.always_on_object_object_gate:
            edge_gate = torch.where(
                object_object_valid,
                torch.ones_like(edge_gate),
                edge_gate,
            )
        diagnostic = self.edge_causal_intervention
        self.last_edge_causal_receipt = None
        e_path_ablation = None
        mu_path_ablation = None
        if diagnostic is not None:
            mode = diagnostic.get("mode")
            temporal_steps = diagnostic.get("temporal_steps")
            receiver = diagnostic.get("receiver_object")
            sender = diagnostic.get("sender_object")
            latent_times = diagnostic.get("latent_times")
            node_count = object_count + 1
            if mode not in {
                "delete_edge",
                "flatten_edge_time",
                "reverse_edge_time",
                "ablate_e_path",
                "ablate_mu_path",
            }:
                raise ValueError(
                    "independent-edge causal probe supports delete_edge, "
                    "flatten_edge_time, reverse_edge_time, ablate_e_path, "
                    "or ablate_mu_path; "
                    f"got {mode!r}"
                )
            if (
                not isinstance(temporal_steps, int)
                or isinstance(temporal_steps, bool)
                or temporal_steps <= 0
                or batch % temporal_steps
                or not isinstance(receiver, int)
                or not isinstance(sender, int)
                or isinstance(receiver, bool)
                or isinstance(sender, bool)
                or not 0 <= receiver < node_count
                or not 0 <= sender < node_count
                or receiver == sender
                or (
                    mode == "ablate_mu_path"
                    and not (
                        (receiver < object_count and sender == object_count)
                        or (sender < object_count and receiver == object_count)
                    )
                )
                or not isinstance(latent_times, (tuple, list))
                or not latent_times
                or any(
                    not isinstance(time, int)
                    or isinstance(time, bool)
                    or not 0 <= time < temporal_steps
                    for time in latent_times
                )
            ):
                raise ValueError("invalid independent-edge causal contract")
            time_index = torch.arange(batch, device=objects.device) % temporal_steps
            edge_gate = edge_gate.clone()
            if mode in {"delete_edge", "ablate_e_path", "ablate_mu_path"}:
                selected_rows = torch.zeros(
                    batch, dtype=torch.bool, device=objects.device
                )
                for latent_time in latent_times:
                    selected_rows |= time_index == latent_time
                if mode == "delete_edge":
                    before = edge_gate[selected_rows, receiver, sender].detach()
                    edge_gate[selected_rows, receiver, sender] = 0
                    edge_gate[selected_rows, sender, receiver] = 0
                    self.last_edge_causal_receipt = {
                        "mode": mode,
                        "receiver_object": receiver,
                        "sender_object": sender,
                        "latent_times": tuple(latent_times),
                        "temporal_steps": temporal_steps,
                        "selected_row_count": int(selected_rows.sum().item()),
                        "predicted_gate_mean": float(before.float().mean().item()),
                        "actual_gate_nonzero_count": int(
                            torch.count_nonzero(
                                edge_gate[selected_rows, receiver, sender]
                            ).item()
                        ),
                        "undirected_reverse_nonzero_count": int(
                            torch.count_nonzero(
                                edge_gate[selected_rows, sender, receiver]
                            ).item()
                        ),
                    }
                else:
                    if not self.contact_head_enabled:
                        raise RuntimeError(
                            f"{mode} requires pair-level event supervision"
                        )
                    if mode == "ablate_e_path":
                        e_path_ablation = (selected_rows, receiver, sender)
                        parameter_path = "e"
                    else:
                        mu_path_ablation = (selected_rows, receiver, sender)
                        parameter_path = "mu"
                    self.last_edge_causal_receipt = {
                        "mode": mode,
                        "receiver_object": receiver,
                        "sender_object": sender,
                        "latent_times": tuple(latent_times),
                        "temporal_steps": temporal_steps,
                        "selected_row_count": int(selected_rows.sum().item()),
                        "parameter_path": parameter_path,
                        "typed_residual_hard_zero": True,
                        "general_relation_gate_unchanged": True,
                        **(
                            {"mu_path_unchanged": True}
                            if mode == "ablate_e_path"
                            else {"e_path_unchanged": True}
                        ),
                    }
            else:
                if tuple(latent_times) != tuple(range(temporal_steps)):
                    raise ValueError(
                        "edge-time intervention must cover every latent time"
                    )
                predicted = edge_gate[:, receiver, sender].reshape(
                    batch // temporal_steps, temporal_steps
                )
                if mode == "flatten_edge_time":
                    executed = predicted.mean(dim=1, keepdim=True).expand_as(predicted)
                else:
                    executed = torch.flip(predicted, dims=(1,))
                flat_executed = executed.reshape(batch)
                edge_gate[:, receiver, sender] = flat_executed
                edge_gate[:, sender, receiver] = flat_executed
                predicted_float = predicted.detach().float()
                executed_float = executed.detach().float()
                mean_tolerance = {
                    torch.bfloat16: 2.0 ** -8,
                    torch.float16: 2.0 ** -10,
                }.get(edge_gate.dtype, 1.0e-6)
                mean_drift = float(
                    (
                        predicted_float.mean(dim=1)
                        - executed_float.mean(dim=1)
                    ).abs().max().item()
                )
                self.last_edge_causal_receipt = {
                    "mode": mode,
                    "receiver_object": receiver,
                    "sender_object": sender,
                    "latent_times": tuple(latent_times),
                    "temporal_steps": temporal_steps,
                    "selected_row_count": batch,
                    "predicted_gate_mean": float(predicted_float.mean().item()),
                    "actual_gate_mean": float(executed_float.mean().item()),
                    "predicted_temporal_std": float(
                        predicted_float.std(dim=1, unbiased=False).mean().item()
                    ),
                    "actual_temporal_std": float(
                        executed_float.std(dim=1, unbiased=False).mean().item()
                    ),
                    "gate_dtype": str(edge_gate.dtype),
                    "mean_preservation_max_abs": mean_drift,
                    "mean_preservation_tolerance": mean_tolerance,
                    "mean_preservation_pass": mean_drift <= mean_tolerance,
                }

        if self.contact_head_enabled and not self.three_contact_gates:
            # The message state is direction-specific, but event gates are
            # undirected: explicitly symmetrize the state feature before the
            # shared scalar heads and use symmetric distance/approach terms.
            symmetric_pair_state = 0.5 * (
                pair_state + pair_state.transpose(1, 2)
            )
            edge_distance = torch.cdist(
                node_position.float(), node_position.float()
            ).to(pair_state.dtype)
            sequence_distance = edge_distance.reshape(
                batch // temporal_steps,
                temporal_steps,
                object_count + 1,
                object_count + 1,
            )
            sequence_approach = torch.zeros_like(sequence_distance)
            sequence_approach[:, 1:] = (
                sequence_distance[:, :-1] - sequence_distance[:, 1:]
            )
            edge_approach = sequence_approach.reshape_as(edge_distance)
            pair_gate_input = torch.cat(
                (
                    symmetric_pair_state,
                    edge_distance[..., None],
                    edge_approach[..., None],
                ),
                dim=-1,
            )
            e_pair_event_logit = self.e_pair_event_head(pair_gate_input).squeeze(-1)
            if self.shared_contact_gate:
                # Start from the historical generic-relation prediction and
                # learn one contact residual from state, distance and approach.
                # The supervised and free-learning arms execute this same graph.
                e_pair_event_logit = edge_logit + e_pair_event_logit
                mu_pair_event_logit = e_pair_event_logit
            else:
                mu_pair_event_logit = self.mu_pair_event_head(
                    pair_gate_input
                ).squeeze(-1)
        full_mu = self._full_edge_tensor(edge_mu)
        full_mu_present = self._full_edge_tensor(edge_mu_present)
        full_e = self._full_edge_tensor(edge_restitution)
        full_e_present = self._full_edge_tensor(edge_restitution_present)
        if self.contact_head_enabled:
            # Restitution is typed to both object-object and object-support
            # edges.  Legacy event-gate runs restricted friction to support;
            # shared/three-contact runs honor every valid endpoint-present mu
            # edge.  The support node is the final row/column of the tensor.
            object_object = torch.zeros_like(full_e_present)
            object_object[:, :object_count, :object_count] = 1.0
            object_table = torch.zeros_like(full_mu_present)
            object_table[:, :object_count, object_count] = 1.0
            object_table[:, object_count, :object_count] = 1.0
            typed_e = torch.maximum(object_object, object_table)
            if self.shared_contact_gate or self.three_contact_gates:
                # Contact is independent of which material coefficients happen
                # to be present.  Endpoint e/mu conditions may exist on either
                # object-object or object-support edges; absence only zeros that
                # typed residual, never the learned contact decision.
                full_e_present = full_e_present * edge_valid.to(
                    full_e_present.dtype
                )
                full_mu_present = full_mu_present * edge_valid.to(
                    full_mu_present.dtype
                )
            else:
                full_e_present = full_e_present * typed_e
                full_mu_present = full_mu_present * object_table
        base = self.base_output(pair_state)
        e_residual = self._physical_residual(
            pair_state,
            full_e,
            full_e_present,
            self.e_adaln,
            self.e_output,
        )
        mu_residual = self._physical_residual(
            pair_state,
            full_mu,
            full_mu_present,
            self.mu_adaln,
            self.mu_output,
        )
        if self.three_contact_gates:
            base_contact_gate = edge_gate
            e_pair_event_gate = (
                torch.sigmoid(e_pair_event_logit)
                * edge_valid.to(e_pair_event_logit.dtype)
            )
            mu_pair_event_gate = (
                torch.sigmoid(mu_pair_event_logit)
                * edge_valid.to(mu_pair_event_logit.dtype)
            )
            predicted_e_pair_event_gate = e_pair_event_gate
            predicted_mu_pair_event_gate = mu_pair_event_gate
            if self.always_on_object_object_gate:
                e_pair_event_gate = torch.where(
                    object_object_valid,
                    torch.ones_like(e_pair_event_gate),
                    e_pair_event_gate,
                )
                mu_pair_event_gate = torch.where(
                    object_object_valid,
                    torch.ones_like(mu_pair_event_gate),
                    mu_pair_event_gate,
                )
            executed_e_pair_event_gate = e_pair_event_gate
            executed_mu_pair_event_gate = mu_pair_event_gate
            if diagnostic is not None and diagnostic.get("mode") == "delete_edge":
                time_index = torch.arange(batch, device=objects.device) % temporal_steps
                selected_rows = torch.zeros(
                    batch, dtype=torch.bool, device=objects.device
                )
                for latent_time in diagnostic["latent_times"]:
                    selected_rows |= time_index == latent_time
                receiver_index = diagnostic["receiver_object"]
                sender_index = diagnostic["sender_object"]
                executed_e_pair_event_gate = e_pair_event_gate.clone()
                executed_mu_pair_event_gate = mu_pair_event_gate.clone()
                predicted_e = e_pair_event_gate[
                    selected_rows, receiver_index, sender_index
                ].detach()
                predicted_mu = mu_pair_event_gate[
                    selected_rows, receiver_index, sender_index
                ].detach()
                for gate in (
                    executed_e_pair_event_gate,
                    executed_mu_pair_event_gate,
                ):
                    gate[selected_rows, receiver_index, sender_index] = 0
                    gate[selected_rows, sender_index, receiver_index] = 0
                self.last_edge_causal_receipt.update(
                    {
                        "base_contact_hard_zero": True,
                        "e_contact_hard_zero": True,
                        "mu_contact_hard_zero": True,
                        "predicted_e_contact_gate_mean": float(
                            predicted_e.float().mean().item()
                        ),
                        "predicted_mu_contact_gate_mean": float(
                            predicted_mu.float().mean().item()
                        ),
                        "actual_e_contact_gate_nonzero_count": int(
                            torch.count_nonzero(
                                executed_e_pair_event_gate[
                                    selected_rows, receiver_index, sender_index
                                ]
                            ).item()
                        ),
                        "e_contact_reverse_nonzero_count": int(
                            torch.count_nonzero(
                                executed_e_pair_event_gate[
                                    selected_rows, sender_index, receiver_index
                                ]
                            ).item()
                        ),
                        "actual_mu_contact_gate_nonzero_count": int(
                            torch.count_nonzero(
                                executed_mu_pair_event_gate[
                                    selected_rows, receiver_index, sender_index
                                ]
                            ).item()
                        ),
                        "mu_contact_reverse_nonzero_count": int(
                            torch.count_nonzero(
                                executed_mu_pair_event_gate[
                                    selected_rows, sender_index, receiver_index
                                ]
                            ).item()
                        ),
                    }
                )
            if e_path_ablation is not None:
                selected_rows, receiver_index, sender_index = e_path_ablation
                executed_e_pair_event_gate = e_pair_event_gate.clone()
                executed_e_pair_event_gate[
                    selected_rows, receiver_index, sender_index
                ] = 0
                executed_e_pair_event_gate[
                    selected_rows, sender_index, receiver_index
                ] = 0
            if mu_path_ablation is not None:
                selected_rows, receiver_index, sender_index = mu_path_ablation
                executed_mu_pair_event_gate = mu_pair_event_gate.clone()
                executed_mu_pair_event_gate[
                    selected_rows, receiver_index, sender_index
                ] = 0
                executed_mu_pair_event_gate[
                    selected_rows, sender_index, receiver_index
                ] = 0
            base_aggregate = (base_contact_gate[..., None] * base).sum(dim=2)
            e_aggregate = (
                executed_e_pair_event_gate[..., None] * e_residual
            ).sum(dim=2)
            mu_aggregate = (
                executed_mu_pair_event_gate[..., None] * mu_residual
            ).sum(dim=2)
            aggregate = base_aggregate + e_aggregate + mu_aggregate
        elif self.shared_contact_gate:
            contact_gate = (
                torch.sigmoid(e_pair_event_logit)
                * edge_valid.to(e_pair_event_logit.dtype)
            )
            predicted_e_pair_event_gate = contact_gate
            predicted_mu_pair_event_gate = contact_gate
            e_pair_event_gate = contact_gate
            mu_pair_event_gate = contact_gate
            executed_contact_gate = contact_gate
            if diagnostic is not None and diagnostic.get("mode") == "delete_edge":
                time_index = torch.arange(batch, device=objects.device) % temporal_steps
                selected_rows = torch.zeros(
                    batch, dtype=torch.bool, device=objects.device
                )
                for latent_time in diagnostic["latent_times"]:
                    selected_rows |= time_index == latent_time
                receiver = diagnostic["receiver_object"]
                sender_index = diagnostic["sender_object"]
                before = contact_gate[selected_rows, receiver, sender_index].detach()
                executed_contact_gate = contact_gate.clone()
                executed_contact_gate[selected_rows, receiver, sender_index] = 0
                executed_contact_gate[selected_rows, sender_index, receiver] = 0
                self.last_edge_causal_receipt.update(
                    {
                        "predicted_contact_gate_mean": float(
                            before.float().mean().item()
                        ),
                        "actual_contact_gate_nonzero_count": int(
                            torch.count_nonzero(
                                executed_contact_gate[
                                    selected_rows, receiver, sender_index
                                ]
                            ).item()
                        ),
                        "shared_contact_hard_zero": True,
                    }
                )
            executed_e_residual = e_residual
            if e_path_ablation is not None:
                selected_rows, receiver, sender_index = e_path_ablation
                executed_e_residual = e_residual.clone()
                before = executed_e_residual[
                    selected_rows, receiver, sender_index
                ].detach()
                executed_e_residual[selected_rows, receiver, sender_index] = 0
                executed_e_residual[selected_rows, sender_index, receiver] = 0
                self.last_edge_causal_receipt.update(
                    {
                        "predicted_e_residual_rms": float(
                            before.float().square().mean().sqrt().item()
                        ),
                        "actual_e_residual_nonzero_count": int(
                            torch.count_nonzero(
                                executed_e_residual[
                                    selected_rows, receiver, sender_index
                                ]
                            ).item()
                        ),
                        "undirected_reverse_nonzero_count": int(
                            torch.count_nonzero(
                                executed_e_residual[
                                    selected_rows, sender_index, receiver
                                ]
                            ).item()
                        ),
                        "shared_contact_gate_unchanged": True,
                    }
                )
            executed_mu_residual = mu_residual
            if mu_path_ablation is not None:
                selected_rows, receiver, sender_index = mu_path_ablation
                executed_mu_residual = mu_residual.clone()
                before = executed_mu_residual[
                    selected_rows, receiver, sender_index
                ].detach()
                executed_mu_residual[selected_rows, receiver, sender_index] = 0
                executed_mu_residual[selected_rows, sender_index, receiver] = 0
                self.last_edge_causal_receipt.update(
                    {
                        "predicted_mu_residual_rms": float(
                            before.float().square().mean().sqrt().item()
                        ),
                        "actual_mu_residual_nonzero_count": int(
                            torch.count_nonzero(
                                executed_mu_residual[
                                    selected_rows, receiver, sender_index
                                ]
                            ).item()
                        ),
                        "undirected_reverse_nonzero_count": int(
                            torch.count_nonzero(
                                executed_mu_residual[
                                    selected_rows, sender_index, receiver
                                ]
                            ).item()
                        ),
                        "shared_contact_gate_unchanged": True,
                    }
                )
            base_aggregate = (executed_contact_gate[..., None] * base).sum(dim=2)
            e_aggregate = (
                executed_contact_gate[..., None] * executed_e_residual
            ).sum(dim=2)
            mu_aggregate = (
                executed_contact_gate[..., None] * executed_mu_residual
            ).sum(dim=2)
            aggregate = base_aggregate + e_aggregate + mu_aggregate
        elif self.pair_event_supervision:
            e_pair_event_gate = (
                torch.sigmoid(e_pair_event_logit)
                * edge_valid.to(e_pair_event_logit.dtype)
                * typed_e.to(e_pair_event_logit.dtype)
            )
            mu_pair_event_gate = (
                torch.sigmoid(mu_pair_event_logit)
                * edge_valid.to(mu_pair_event_logit.dtype)
                * object_table.to(mu_pair_event_logit.dtype)
            )
            predicted_e_pair_event_gate = e_pair_event_gate
            predicted_mu_pair_event_gate = mu_pair_event_gate
            executed_e_pair_event_gate = e_pair_event_gate
            if e_path_ablation is not None:
                selected_rows, receiver, sender = e_path_ablation
                before = e_pair_event_gate[
                    selected_rows, receiver, sender
                ].detach()
                executed_e_pair_event_gate = e_pair_event_gate.clone()
                executed_e_pair_event_gate[
                    selected_rows, receiver, sender
                ] = 0
                executed_e_pair_event_gate[
                    selected_rows, sender, receiver
                ] = 0
                self.last_edge_causal_receipt.update(
                    {
                        "predicted_e_gate_mean": float(
                            before.float().mean().item()
                        ),
                        "actual_e_gate_nonzero_count": int(
                            torch.count_nonzero(
                                executed_e_pair_event_gate[
                                    selected_rows, receiver, sender
                                ]
                            ).item()
                        ),
                        "undirected_reverse_nonzero_count": int(
                            torch.count_nonzero(
                                executed_e_pair_event_gate[
                                    selected_rows, sender, receiver
                                ]
                            ).item()
                        ),
                    }
                )
            executed_mu_pair_event_gate = mu_pair_event_gate
            if mu_path_ablation is not None:
                selected_rows, receiver, sender = mu_path_ablation
                before = mu_pair_event_gate[
                    selected_rows, receiver, sender
                ].detach()
                executed_mu_pair_event_gate = mu_pair_event_gate.clone()
                executed_mu_pair_event_gate[
                    selected_rows, receiver, sender
                ] = 0
                executed_mu_pair_event_gate[
                    selected_rows, sender, receiver
                ] = 0
                self.last_edge_causal_receipt.update(
                    {
                        "predicted_mu_gate_mean": float(
                            before.float().mean().item()
                        ),
                        "actual_mu_gate_nonzero_count": int(
                            torch.count_nonzero(
                                executed_mu_pair_event_gate[
                                    selected_rows, receiver, sender
                                ]
                            ).item()
                        ),
                        "undirected_reverse_nonzero_count": int(
                            torch.count_nonzero(
                                executed_mu_pair_event_gate[
                                    selected_rows, sender, receiver
                                ]
                            ).item()
                        ),
                    }
                )
            # Three independently aggregated paths.  In particular, typed
            # residuals are not multiplied by the generic edge gate.
            base_aggregate = (edge_gate[..., None] * base).sum(dim=2)
            e_aggregate = (
                executed_e_pair_event_gate[..., None] * e_residual
            ).sum(dim=2)
            mu_aggregate = (
                executed_mu_pair_event_gate[..., None] * mu_residual
            ).sum(dim=2)
            aggregate = base_aggregate + e_aggregate + mu_aggregate
        else:
            e_pair_event_gate = None
            predicted_e_pair_event_gate = None
            predicted_mu_pair_event_gate = None
            executed_e_pair_event_gate = None
            mu_pair_event_gate = None
            base_aggregate = (edge_gate[..., None] * base).sum(dim=2)
            e_aggregate = (edge_gate[..., None] * e_residual).sum(dim=2)
            mu_aggregate = (edge_gate[..., None] * mu_residual).sum(dim=2)
            aggregate = base_aggregate + e_aggregate + mu_aggregate
        message_delta = self.output_projection(aggregate)
        after_message = nodes + message_delta
        ffn_delta = self.ffn(self.ffn_norm(after_message))
        updated = after_message + ffn_delta
        updated = updated * node_valid[..., None].to(updated.dtype)
        if self.capture_diagnostic_state:
            # The public continuous-state trace consumes receiver-level
            # [B*T,N,D] tensors.  Keep the three message families separate in
            # the receipt while preserving the exact forward computation.
            self.last_interaction_message = message_delta[:, :object_count].detach().float().cpu()
            self.last_e_modulation_effect = self.output_projection(
                e_aggregate
            )[:, :object_count].detach().float().cpu()
            self.last_mu_modulation_effect = self.output_projection(
                mu_aggregate
            )[:, :object_count].detach().float().cpu()
            self.last_interaction_ffn_delta = ffn_delta[:, :object_count].detach().float().cpu()
            self.last_interaction_total_delta = (
                updated[:, :object_count] - objects
            ).detach().float().cpu()
        else:
            self.last_interaction_message = None
            self.last_e_modulation_effect = None
            self.last_mu_modulation_effect = None
            self.last_interaction_ffn_delta = None
            self.last_interaction_total_delta = None
        if self.capture_edges:
            # Preserve the unmodified model prediction as the descriptive
            # diagnostic; the causal receipt separately attests execution.
            self.last_edge_gate = predicted_edge_gate.detach()
            self.last_edge_logit = edge_logit.detach()
            gate = predicted_edge_gate[..., None]
            self.last_edge_base_rms = base.detach().float().square().mean(-1).sqrt()
            self.last_edge_e_residual_rms = (
                e_residual.detach().float().square().mean(-1).sqrt()
            )
            self.last_edge_mu_residual_rms = (
                mu_residual.detach().float().square().mean(-1).sqrt()
            )
            self.last_edge_gated_base_rms = (
                (gate * base).detach().float().square().mean(-1).sqrt()
            )
            self.last_edge_gated_mu_residual_rms = (
                (
                    (
                        mu_pair_event_gate[..., None]
                        if self.contact_head_enabled
                        else gate
                    )
                    * mu_residual
                ).detach().float().square().mean(-1).sqrt()
            )
            self.last_edge_gated_e_residual_rms = (
                (
                    (
                        e_pair_event_gate[..., None]
                        if self.contact_head_enabled
                        else gate
                    )
                    * e_residual
                ).detach().float().square().mean(-1).sqrt()
            )
            self.last_e_pair_event_logit = (
                None if e_pair_event_logit is None else e_pair_event_logit
            )
            self.last_mu_pair_event_logit = (
                None if mu_pair_event_logit is None else mu_pair_event_logit
            )
            self.last_e_pair_event_gate = (
                None
                if predicted_e_pair_event_gate is None
                else predicted_e_pair_event_gate.detach()
            )
            self.last_mu_pair_event_gate = (
                None
                if predicted_mu_pair_event_gate is None
                else predicted_mu_pair_event_gate.detach()
            )
            self.last_edge_base_aggregate_rms = base_aggregate.detach().float().square().mean(-1).sqrt()
            self.last_edge_e_aggregate_rms = e_aggregate.detach().float().square().mean(-1).sqrt()
            self.last_edge_mu_aggregate_rms = mu_aggregate.detach().float().square().mean(-1).sqrt()
        else:
            self.last_edge_gate = None
            self.last_edge_logit = None
            self.last_edge_base_rms = None
            self.last_edge_e_residual_rms = None
            self.last_edge_mu_residual_rms = None
            self.last_edge_gated_base_rms = None
            self.last_edge_gated_e_residual_rms = None
            self.last_edge_gated_mu_residual_rms = None
            self.last_e_pair_event_logit = e_pair_event_logit
            self.last_mu_pair_event_logit = mu_pair_event_logit
            self.last_e_pair_event_gate = e_pair_event_gate
            self.last_mu_pair_event_gate = mu_pair_event_gate
            self.last_edge_base_aggregate_rms = None
            self.last_edge_e_aggregate_rms = None
            self.last_edge_mu_aggregate_rms = None
        self.last_base_contact_logit = base_contact_logit
        self.last_base_contact_gate = (
            predicted_edge_gate if self.three_contact_gates else None
        )
        self.last_executed_base_contact_gate = (
            base_contact_gate.detach() if self.three_contact_gates else None
        )
        self.last_executed_e_pair_event_gate = (
            executed_e_pair_event_gate.detach()
            if self.three_contact_gates
            else None
        )
        self.last_executed_mu_pair_event_gate = (
            executed_mu_pair_event_gate.detach()
            if self.three_contact_gates
            else None
        )
        return updated[:, :object_count]


class TemporalPairPhysicsInteractionBlock(nn.Module):
    """State-only context attention plus post-selection pair physics.

    Material conditions never enter Q, K, V, the ordinary context message, or
    a receiver-wide gate.  They are read only after the per-time attention has
    selected a non-self partner.  The original softmax mass is retained: object
    non-self weights are masked, never re-normalized, so self/support attention
    remains an explicit no-object-interaction alternative.
    """

    def __init__(
        self,
        object_dim: int,
        num_heads: int,
        condition_dim: int = 64,
        ffn_ratio: int = 2,
    ):
        super().__init__()
        if object_dim % num_heads:
            raise ValueError("object_dim must be divisible by num_heads")
        self.object_dim = object_dim
        self.num_heads = num_heads
        self.head_dim = object_dim // num_heads
        self.attention_norm = nn.LayerNorm(object_dim)
        self.support_norm = nn.LayerNorm(object_dim)
        self.query_projection = nn.Linear(object_dim, object_dim)
        self.key_projection = nn.Linear(object_dim, object_dim)
        self.value_projection = nn.Linear(object_dim, object_dim)
        self.output_projection = nn.Linear(object_dim, object_dim)
        # Pair scalars: e_i, e_j, e_i*e_j, e_pair_present,
        # mu_i, mu_j, mu_i*mu_j, mu_i_present, mu_j_present, support_j.
        self.condition_encoder = _mlp(10, condition_dim, condition_dim)
        pair_input_dim = object_dim * 4 + condition_dim
        self.pair_response = nn.Sequential(
            nn.Linear(pair_input_dim, object_dim),
            nn.SiLU(),
            nn.Linear(object_dim, object_dim),
            nn.SiLU(),
            nn.Linear(object_dim, object_dim),
        )
        _zero_last(self.pair_response)
        self.ffn_norm = nn.LayerNorm(object_dim)
        self.ffn = _mlp(object_dim, object_dim * ffn_ratio, object_dim)
        self.capture_attention = False
        self.last_attention: Optional[torch.Tensor] = None

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, times, length, _ = value.shape
        return value.reshape(
            batch, times, length, self.num_heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)

    @staticmethod
    def _pair_mask(
        object_valid: torch.Tensor,
        partner_count: int,
    ) -> torch.Tensor:
        batch, objects = object_valid.shape
        if partner_count != objects + 1:
            raise ValueError("pair physics requires exactly one support partner")
        partner_valid = torch.cat(
            (
                object_valid,
                torch.ones(batch, 1, dtype=torch.bool, device=object_valid.device),
            ),
            dim=-1,
        )
        mask = object_valid[:, :, None] & partner_valid[:, None, :]
        diagonal = torch.eye(
            objects, partner_count, dtype=torch.bool, device=object_valid.device
        )
        return mask & ~diagonal[None]

    def _pair_condition_features(
        self,
        mu: torch.Tensor,
        mu_present: torch.Tensor,
        restitution: torch.Tensor,
        restitution_present: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, objects = mu.shape
        zeros = mu.new_zeros(batch, 1)
        object_mu = mu * mu_present
        partner_mu = torch.cat((object_mu, zeros), dim=-1)
        partner_mu_present = torch.cat((mu_present, zeros), dim=-1)

        object_e = restitution * restitution_present
        partner_e = torch.cat((object_e, zeros), dim=-1)
        partner_e_present = torch.cat((restitution_present, zeros), dim=-1)

        e_pair_present = (
            restitution_present[:, :, None]
            * partner_e_present[:, None, :]
        )
        e_i = object_e[:, :, None] * e_pair_present
        e_j = partner_e[:, None, :] * e_pair_present
        mu_i = object_mu[:, :, None].expand(-1, -1, objects + 1)
        mu_j = partner_mu[:, None, :].expand(-1, objects, -1)
        mu_i_present = mu_present[:, :, None].expand(-1, -1, objects + 1)
        mu_j_present = partner_mu_present[:, None, :].expand(-1, objects, -1)
        support_j = mu.new_zeros(batch, objects, objects + 1)
        support_j[..., -1] = 1.0
        scalars = torch.stack(
            (
                e_i,
                e_j,
                e_i * e_j,
                e_pair_present,
                mu_i,
                mu_j,
                mu_i * mu_j,
                mu_i_present,
                mu_j_present,
                support_j,
            ),
            dim=-1,
        )
        # Object-object edges are active for either declared material.  The
        # support edge intentionally uses receiver mu only; no platform e is
        # invented for datasets that do not provide it.
        material_active = (
            (e_pair_present > 0.5)
            | (mu_i_present > 0.5)
            | (mu_j_present > 0.5)
        )
        return self.condition_encoder(scalars), material_active

    def pair_physics_residual(
        self,
        normalized_objects: torch.Tensor,
        normalized_support: torch.Tensor,
        attention: torch.Tensor,
        object_valid: torch.Tensor,
        mu: torch.Tensor,
        mu_present: torch.Tensor,
        restitution: torch.Tensor,
        restitution_present: torch.Tensor,
    ) -> torch.Tensor:
        """Return [B,T,N,D] without changing the original attention mass."""
        batch, times, objects, dimension = normalized_objects.shape
        partners = torch.cat((normalized_objects, normalized_support), dim=2)
        partner_count = partners.shape[2]
        receiver = normalized_objects[:, :, :, None, :].expand(
            -1, -1, -1, partner_count, -1
        )
        partner = partners[:, :, None, :, :].expand(
            -1, -1, objects, -1, -1
        )
        condition, material_active = self._pair_condition_features(
            mu, mu_present, restitution, restitution_present
        )
        condition = condition[:, None].expand(-1, times, -1, -1, -1)
        pair_feature = torch.cat(
            (
                receiver,
                partner,
                partner - receiver,
                receiver * partner,
                condition,
            ),
            dim=-1,
        )
        response = self.pair_response(pair_feature)
        pair_mask = self._pair_mask(object_valid, partner_count)
        pair_mask = pair_mask & material_active
        # Mean only over heads.  Do not normalize across partners after masking.
        pair_weight = attention.mean(dim=2) * pair_mask[:, None].to(attention.dtype)
        return torch.einsum("btij,btijd->btid", pair_weight, response)

    def forward(
        self,
        objects: torch.Tensor,
        support: torch.Tensor,
        object_valid: torch.Tensor,
        mu: torch.Tensor,
        mu_present: torch.Tensor,
        restitution: torch.Tensor,
        restitution_present: torch.Tensor,
    ) -> torch.Tensor:
        normalized_objects = self.attention_norm(objects)
        normalized_support = self.support_norm(support)
        query = self._heads(self.query_projection(normalized_objects))
        object_key = self._heads(self.key_projection(normalized_objects))
        object_value = self._heads(self.value_projection(normalized_objects))
        support_key = self._heads(self.key_projection(normalized_support))
        support_value = self._heads(self.value_projection(normalized_support))
        key = torch.cat((object_key, support_key), dim=3)
        value = torch.cat((object_value, support_value), dim=3)
        attention_logits = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)
        invalid_keys = torch.cat(
            (
                ~object_valid,
                torch.zeros(
                    object_valid.shape[0],
                    1,
                    dtype=torch.bool,
                    device=object_valid.device,
                ),
            ),
            dim=-1,
        )[:, None, None, None, :]
        attention_logits = attention_logits.masked_fill(
            invalid_keys, torch.finfo(attention_logits.dtype).min
        )
        attention = attention_logits.softmax(dim=-1)
        if self.capture_attention:
            self.last_attention = attention.detach().float().cpu()
        else:
            self.last_attention = None
        context = torch.matmul(attention, value).permute(0, 1, 3, 2, 4).reshape(
            objects.shape[0], objects.shape[1], objects.shape[2], self.object_dim
        )
        context = self.output_projection(context)
        physics = self.pair_physics_residual(
            normalized_objects,
            normalized_support,
            attention,
            object_valid,
            mu,
            mu_present,
            restitution,
            restitution_present,
        )
        valid = object_valid[:, None, :, None]
        objects = objects + valid * (context + physics)
        objects = objects + valid * self.ffn(self.ffn_norm(objects))
        return objects * valid


class SparseObjectInteractionTemporalGroup(nn.Module):
    """Temporal version of one Wan insertion, retaining the sparse router."""

    def __init__(
        self,
        wan_dim: int,
        object_dim: int,
        locator_dim: int,
        num_heads: int,
        topk: int,
        pair_physics: bool = False,
        platform_support: bool = False,
        previous_frame_tracking: bool = False,
        joint_template_routing: bool = False,
        mu_message_gate_alpha: Optional[float] = None,
        decoupled_writer: bool = False,
        continuous_object_state: bool = False,
        writer_radius: int = 1,
        write_gate_mode: Optional[str] = None,
        causal_object_temporal: bool = False,
        event_after_effect: bool = False,
        independent_edge: bool = False,
        pair_event_supervision: bool = False,
        shared_contact_gate: bool = False,
        three_contact_gates: bool = False,
        always_on_object_object_gate: bool = False,
        shared_three_contact_gate: bool = False,
    ):
        super().__init__()
        self.condition_encoder = SparseUnaryTemporalResidualEncoder(object_dim)
        self.joint_template_routing = bool(joint_template_routing)
        if self.joint_template_routing and not previous_frame_tracking:
            raise ValueError("joint template routing requires causal previous-frame tracking")
        router_class = (
            IdentityCompetitiveTemplateRouter
            if self.joint_template_routing
            else ParallelSparseObjectRouter
        )
        router_kwargs = {}
        if not self.joint_template_routing:
            router_kwargs["previous_frame_tracking"] = previous_frame_tracking
        self.router = router_class(
            wan_dim,
            object_dim,
            locator_dim,
            topk,
            **router_kwargs,
        )
        self.decoupled_writer = bool(decoupled_writer)
        self.continuous_object_state = bool(continuous_object_state)
        self.hard_support_enabled = _environment_flag(
            "PHYSICAL_WM_HARD_SUPPORT", False
        )
        self.hard_support_loss_weight = _environment_float(
            "PHYSICAL_WM_HARD_SUPPORT_LOSS_WEIGHT", 0.3
        )
        if self.hard_support_loss_weight < 0.0:
            raise ValueError(
                "PHYSICAL_WM_HARD_SUPPORT_LOSS_WEIGHT must be non-negative"
            )
        self.correctable_route_carry = _environment_flag(
            "PHYSICAL_WM_CORRECTABLE_ROUTE_CARRY", False
        )
        self.position_dependent_writer_value = _environment_flag(
            "PHYSICAL_WM_POSITION_DEPENDENT_WRITER_VALUE", False
        )
        self.relative_position_writer_value = _environment_flag(
            "PHYSICAL_WM_RELATIVE_POSITION_WRITER_VALUE", False
        )
        if (
            self.position_dependent_writer_value
            and self.relative_position_writer_value
        ):
            raise ValueError(
                "legacy local-hidden and relative-position Writer values are exclusive"
            )
        if (
            self.position_dependent_writer_value
            or self.relative_position_writer_value
        ) and not self.decoupled_writer:
            raise ValueError("position-dependent Writer value requires independent Writer")
        self.last_updated_objects: Optional[torch.Tensor] = None
        if self.continuous_object_state:
            self.previous_state_norm = nn.LayerNorm(object_dim)
            self.current_read_norm = nn.LayerNorm(object_dim)
            self.state_fusion_gate = nn.Linear(object_dim * 2, object_dim)
            # Start by retaining most of the explicit state while still
            # allowing the current Wan read to correct it.  The gate remains
            # fully trainable and bounded in [0, 1].
            nn.init.zeros_(self.state_fusion_gate.weight)
            nn.init.ones_(self.state_fusion_gate.bias)
        self.write_gate_mode = (
            write_gate_mode
            if write_gate_mode is not None
            else os.environ.get("PHYSICAL_WM_WRITE_GATE_MODE", "legacy")
        ).strip().lower()
        if self.write_gate_mode not in WRITE_GATE_MODES:
            raise ValueError(
                f"unsupported write gate mode: {self.write_gate_mode!r}; "
                f"expected one of {WRITE_GATE_MODES}"
            )
        if self.write_gate_mode == "bounded_v1" and not self.decoupled_writer:
            raise ValueError("bounded_v1 requires the independent Writer route")
        self.writer = (
            IndependentLocalWriterRouter(
                wan_dim,
                object_dim,
                locator_dim,
                topk,
                radius=writer_radius,
            )
            if self.decoupled_writer
            else None
        )
        self.last_writer_logits: Optional[torch.Tensor] = None
        self.last_writer_indices: Optional[torch.Tensor] = None
        self.last_writer_attention: Optional[torch.Tensor] = None
        self.last_writer_seed_indices: Optional[torch.Tensor] = None
        # Loss-only receipts for the route that actually produced the Reader
        # state.  They are plain runtime attributes and add no checkpoint
        # parameters or forward inputs.
        self.last_reader_indices: Optional[torch.Tensor] = None
        self.last_reader_attention: Optional[torch.Tensor] = None
        self.last_reader_grid_shape: Optional[Tuple[int, int]] = None
        self.last_reader_support_gate_logits: Optional[torch.Tensor] = None
        self.last_reader_support_gate_mask: Optional[torch.Tensor] = None
        self.last_writer_support_gate_logits: Optional[torch.Tensor] = None
        self.last_writer_support_gate_mask: Optional[torch.Tensor] = None
        self.last_writer_support_gate_allowed_mask: Optional[torch.Tensor] = None
        self.last_write_gate: Optional[torch.Tensor] = None
        self.temporal = SameObjectTemporalBlock(
            object_dim, num_heads, causal=causal_object_temporal
        )
        self.causal_object_temporal = bool(causal_object_temporal)
        # Event after-effect: what the per-time object interaction did at an
        # earlier latent time is carried forward into every later latent time
        # as an exponentially decaying state contribution.  A time-parallel
        # strictly-lower-triangular accumulation, not a serial loop.  Only the
        # two scalars below are new parameters, so the read-out is directly
        # interpretable and existing checkpoints stay loadable when disabled.
        self.event_after_effect = bool(event_after_effect)
        if self.event_after_effect:
            self.event_beta = nn.Parameter(torch.tensor(0.25))
            self.event_decay_logit = nn.Parameter(torch.zeros(()))
        # Evaluation-only ablation switch; plain Python keeps state_dict keys
        # identical.  Norm capture stays off during training so no per-forward
        # device synchronisation is introduced.
        self.event_after_effect_runtime_enabled = True
        self.capture_event_after_effect_norms = False
        self.last_event_history_norm: Optional[float] = None
        self.last_event_after_effect_norm: Optional[float] = None
        self.pair_physics = bool(pair_physics)
        self.platform_support = bool(platform_support)
        self.independent_edge = bool(independent_edge)
        self.pair_event_supervision = bool(pair_event_supervision)
        self.shared_contact_gate = bool(shared_contact_gate)
        self.three_contact_gates = bool(three_contact_gates)
        self.always_on_object_object_gate = bool(
            always_on_object_object_gate
        )
        self.shared_three_contact_gate = bool(shared_three_contact_gate)
        self.contact_head_enabled = (
            self.pair_event_supervision
            or self.shared_contact_gate
            or self.three_contact_gates
        )
        # Evaluation-only, non-persistent state used by the frozen pair-e
        # window diagnostic.  These are plain Python attributes so enabling
        # the probe never changes the checkpoint key set.
        self.e_value_intervention: Optional[dict] = None
        self.last_e_value_intervention_receipt: Optional[dict] = None
        if self.contact_head_enabled and not self.independent_edge:
            raise ValueError("pair event supervision requires independent edges")
        if sum((self.pair_physics, self.platform_support, self.independent_edge)) > 1:
            raise ValueError("pair, NULL-platform, and independent-edge routes are exclusive")
        support_count = (
            1
            if self.pair_physics or self.platform_support or self.independent_edge
            else 2
        )
        # Keep historical parameter names and shapes for legacy Temporal/Pair.
        self.background_queries = nn.Parameter(
            torch.randn(support_count, locator_dim) * 0.02
        )
        self.background_value = nn.Linear(locator_dim, object_dim)
        self.background_norm = nn.LayerNorm(object_dim)
        self.interaction = (
            TemporalPairPhysicsInteractionBlock(object_dim, num_heads)
            if self.pair_physics
            else (
                IndependentUndirectedEdgeInteractionBlock(
                    object_dim,
                    num_heads,
                    pair_event_supervision=self.pair_event_supervision,
                    shared_contact_gate=self.shared_contact_gate,
                    three_contact_gates=self.three_contact_gates,
                    always_on_object_object_gate=(
                        self.always_on_object_object_gate
                    ),
                    shared_three_contact_gate=self.shared_three_contact_gate,
                )
                if self.independent_edge
                else TemporalObjectInteractionBlock(
                    object_dim,
                    num_heads,
                    no_interaction=self.platform_support,
                    mu_message_gate_alpha=mu_message_gate_alpha,
                )
            )
        )
        # The per-time object state is already available; no pooled-object time
        # decoder is needed.  Keep the existing zero-init sparse writeback.
        self.output_projection = nn.Linear(object_dim, wan_dim, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        if self.position_dependent_writer_value:
            with torch.random.fork_rng(devices=[]):
                self.local_writer_state_projection = nn.Linear(
                    object_dim, locator_dim
                )
                self.local_writer_hidden_projection = nn.Linear(
                    wan_dim, locator_dim
                )
                self.local_writer_position_projection = nn.Linear(2, locator_dim)
                self.local_writer_output_projection = nn.Linear(
                    locator_dim, wan_dim, bias=False
                )
            nn.init.zeros_(self.local_writer_output_projection.weight)
        if self.relative_position_writer_value:
            # W1([z; r]) uses D_h = D_z.  Creating the opt-in branch inside a
            # forked RNG keeps every parameter shared with H2 bit-identical at
            # initialization.  The zeroed final layer makes the entire branch
            # exactly zero until training updates it.
            with torch.random.fork_rng(devices=[]):
                self.relative_position_writer_mlp = nn.Sequential(
                    nn.Linear(object_dim + 2, object_dim),
                    nn.SiLU(),
                    nn.Linear(object_dim, wan_dim),
                )
            nn.init.zeros_(self.relative_position_writer_mlp[-1].weight)
            nn.init.zeros_(self.relative_position_writer_mlp[-1].bias)
        self.noise_gate = _mlp(1, 32, 1)
        self.erase_gate = (
            _mlp(wan_dim * 2, 128, 1)
            if self.write_gate_mode == "erase_then_write_v1"
            else None
        )
        self.residual_scale = nn.Parameter(torch.ones(()))
        self.diagnostic_spatial_oracle_mode = "predicted"
        self._diagnostic_spatial_oracle_labels: Optional[torch.Tensor] = None
        self.last_diagnostic_spatial_routes: Optional[dict] = None
        # Evaluation-only receipt for the continuous-state mediation probe.
        # This is deliberately plain Python state: checkpoint keys must remain
        # byte-for-byte compatible with the training route.
        self.last_continuous_state_mediation_trace: Optional[dict] = None

    def event_after_effect_parameters(self) -> Optional[dict]:
        """Read out the learned persistence scalars for evidence reporting."""
        if not self.event_after_effect:
            return None
        with torch.no_grad():
            return {
                "beta": float(self.event_beta.detach().float().item()),
                "decay_lambda": float(
                    torch.sigmoid(self.event_decay_logit.detach().float()).item()
                ),
            }

    def _relative_position_writer_values(
        self,
        updated: torch.Tensor,
        projected_objects: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_attention: torch.Tensor,
        width: int,
    ) -> torch.Tensor:
        """Return W0(z) + W1([z; p-c]) on the existing Writer support.

        Coordinates are raw feature-grid ``(x, y)`` values.  The center uses
        the original H2 Writer attention, and W1 reads no Wan hidden feature.
        """
        if not self.relative_position_writer_value:
            raise RuntimeError("relative-position Writer value is not enabled")
        candidate_y = torch.div(
            candidate_indices, width, rounding_mode="floor"
        ).to(dtype=projected_objects.dtype)
        candidate_x = (candidate_indices % width).to(
            dtype=projected_objects.dtype
        )
        route_weight = candidate_attention.to(projected_objects.dtype)
        epsilon = 1.0e-6
        route_mass = route_weight.sum(dim=-1, keepdim=True) + epsilon
        center_x = (route_weight * candidate_x).sum(
            dim=-1, keepdim=True
        ) / route_mass
        center_y = (route_weight * candidate_y).sum(
            dim=-1, keepdim=True
        ) / route_mass
        relative_position = torch.stack(
            (candidate_x - center_x, candidate_y - center_y), dim=-1
        )
        expanded_state = updated[..., None, :].expand(
            *relative_position.shape[:-1], updated.shape[-1]
        )
        writer_input = torch.cat((expanded_state, relative_position), dim=-1)
        return projected_objects[..., None, :] + self.relative_position_writer_mlp(
            writer_input
        )

    def _event_history_weights(
        self, times: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Strictly-lower-triangular exponential-moving-average weights.

        ``W[t, s] = (1 - lambda) * lambda ** (t - 1 - s)`` for ``s < t`` and 0
        otherwise.  Row ``t`` sums to ``1 - lambda ** t``, so the accumulated
        history is bounded by the largest single interaction delta and the scale
        does not drift with ``lambda``.  Row 0 is all zero: the first latent
        time has no past.
        """
        decay = torch.sigmoid(self.event_decay_logit.to(dtype=dtype))
        index = torch.arange(times, device=device, dtype=dtype)
        exponent = index[:, None] - index[None, :] - 1.0
        allowed = exponent >= 0.0
        weights = (1.0 - decay) * torch.pow(
            decay, exponent.clamp(min=0.0)
        )
        return weights * allowed.to(dtype=dtype)

    def _apply_event_after_effect(
        self,
        updated: torch.Tensor,
        pre_interaction: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Carry each latent time's interaction effect into later latent times.

        The per-time interaction delta is what the object graph did at that
        latent time.  When the NULL route wins the delta is near zero, so a
        latent time with no real event contributes no after-effect: the same
        gate that mutes the current message also decides whether a durable
        state change is created.  The accumulation is a single ``[T, T]``
        matrix product, so all latent times stay computed in parallel.
        """
        self.last_event_history_norm = None
        self.last_event_after_effect_norm = None
        if not self.event_after_effect or not self.event_after_effect_runtime_enabled:
            return updated
        times = updated.shape[1]
        delta = updated - pre_interaction
        weights = self._event_history_weights(
            times, updated.device, updated.dtype
        )
        history = torch.einsum("ts,bsnd->btnd", weights, delta)
        after_effect = self.event_beta.to(dtype=updated.dtype) * history
        after_effect = after_effect * object_valid[:, None, :, None].to(
            after_effect.dtype
        )
        # A single latent time yields an all-zero weight matrix rather than an
        # early return: both scalars must stay in the autograd graph every
        # step, because the formal run uses find_unused_parameters=false.
        if self.capture_event_after_effect_norms:
            with torch.no_grad():
                self.last_event_history_norm = float(
                    history.detach().float().norm().item()
                )
                self.last_event_after_effect_norm = float(
                    after_effect.detach().float().norm().item()
                )
        return updated + after_effect

    def configure_diagnostic_spatial_oracle(
        self,
        mode: str = "predicted",
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        """Configure an explicit future-mask oracle for read/write diagnosis.

        This is disabled by default and is not part of the model forward
        contract.  It exists only to localize a consistency failure after
        training.  Normal inference never supplies or consumes these labels.
        """
        if mode not in DIAGNOSTIC_SPATIAL_ORACLE_MODES:
            raise ValueError(f"unsupported diagnostic spatial oracle mode: {mode!r}")
        if mode == "predicted":
            if labels is not None:
                raise ValueError("predicted mode must not receive future labels")
            self._diagnostic_spatial_oracle_labels = None
        else:
            if not isinstance(labels, torch.Tensor) or labels.ndim not in (4, 5):
                raise ValueError(
                    "oracle labels must be [T,N,H,W] or [B,T,N,H,W]"
                )
            labels = labels.detach().float().cpu().contiguous()
            if not torch.isfinite(labels).all():
                raise ValueError("oracle labels must be finite")
            if torch.any((labels < 0) | (labels > 1)):
                raise ValueError("oracle labels must be in [0,1]")
            self._diagnostic_spatial_oracle_labels = labels
        self.diagnostic_spatial_oracle_mode = mode
        self.last_diagnostic_spatial_routes = None

    @staticmethod
    def _oracle_candidate_indices(
        labels: torch.Tensor,
        first_frame_masks: torch.Tensor,
        object_valid: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        candidates: int,
    ) -> torch.Tensor:
        batch, objects = first_frame_masks.shape[:2]
        labels = labels.to(device=first_frame_masks.device, dtype=first_frame_masks.dtype)
        if labels.ndim == 4:
            labels = labels.unsqueeze(0)
        if labels.shape[0] == 1 and batch > 1:
            labels = labels.expand(batch, -1, -1, -1, -1)
        expected = (batch, frames, objects, height, width)
        if tuple(labels.shape) != expected:
            raise ValueError(
                f"oracle labels must match current Wan grid {expected}, got "
                f"{tuple(labels.shape)}"
            )
        if not torch.allclose(
            labels[:, 0], first_frame_masks, atol=1e-4, rtol=1e-4
        ):
            raise ValueError("oracle labels are not bound to the supplied frame-0 masks")
        mass = labels.sum(dim=(-2, -1))
        if torch.any(object_valid[:, None] & (mass <= 1e-6)):
            raise ValueError("every valid object-time oracle mask must be non-empty")
        if not 0 < candidates <= height * width:
            raise ValueError("invalid diagnostic Top-K cardinality")

        y = torch.arange(height, device=labels.device, dtype=labels.dtype)
        x = torch.arange(width, device=labels.device, dtype=labels.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        safe_mass = mass.clamp_min(1e-6)
        cy = (labels * yy).sum(dim=(-2, -1)) / safe_mass
        cx = (labels * xx).sum(dim=(-2, -1)) / safe_mass
        distance = (
            (yy - cy[..., None, None]).square()
            + (xx - cx[..., None, None]).square()
        ) / float(max(height * height + width * width, 1))
        # All true support cells rank before any filler cell.  If a small ball
        # occupies fewer than K Wan tokens, remaining indices stay nearest to
        # its GT centroid instead of depending on arbitrary zero-score ties.
        ranking = torch.where(labels > 0, 2.0 + labels, -distance)
        indices = ranking.flatten(-2).topk(candidates, dim=-1).indices
        return torch.where(
            object_valid[:, None, :, None], indices, torch.zeros_like(indices)
        ).long()

    def _pool_support(
        self,
        locator_grid: torch.Tensor,
        location_logits: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid = object_valid[:, None, :, None, None]
        occupancy = (
            torch.sigmoid(location_logits) * valid
        ).amax(dim=2).flatten(1).detach()
        flattened = locator_grid.flatten(1, 2)
        scores = torch.einsum("qd,bgd->bqg", self.background_queries, flattened)
        scores = scores / math.sqrt(flattened.shape[-1])
        scores = scores + (1.0 - occupancy).clamp_min(1e-4).log()[:, None]
        attention = scores.softmax(dim=-1)
        pooled_low_dim = torch.einsum("bqg,bgd->bqd", attention, flattened)
        support = self.background_value(pooled_low_dim)
        return self.background_norm(support)

    def _pool_support_per_time(
        self,
        locator_grid: torch.Tensor,
        location_logits: torch.Tensor,
        object_valid: torch.Tensor,
        platform_mask: torch.Tensor,
        return_position: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        valid = object_valid[:, None, :, None, None]
        occupancy = (
            torch.sigmoid(location_logits) * valid
        ).amax(dim=2).flatten(2).detach()
        batch, times, grid_cells, _ = locator_grid.shape
        height, width = location_logits.shape[-2:]
        if grid_cells != height * width:
            raise ValueError(
                "router locator grid must match the coarse location grid: "
                f"cells={grid_cells} versus {(height, width)}"
            )
        platform = torch.nn.functional.interpolate(
            platform_mask[:, None],
            size=(height, width),
            mode="nearest",
        ).flatten(1)
        eligible_support = platform > 0.5
        allowed_support = (1.0 - occupancy).clamp_min(1e-4)
        # Some admitted legacy/static conditions have no explicit platform
        # segmentation even though the visible scene contains a support
        # surface (for example, overhead pool-table clips).  Preserve exactly
        # one support candidate by falling back, per affected sample only, to
        # the existing non-object complement weighting.  Missingness comes
        # from the explicit frame-0 platform mask, while the support value and
        # occupancy weight use only the same-time current latent.  This neither
        # introduces a learned background token nor changes the separate
        # projection-level strict-zero NULL candidate.
        missing_platform = ~eligible_support.any(dim=-1)
        if torch.any(missing_platform):
            eligible_support = torch.where(
                missing_platform[:, None],
                torch.ones_like(eligible_support),
                eligible_support,
            )
        scores = torch.einsum(
            "qd,btgd->btqg", self.background_queries, locator_grid
        ) / math.sqrt(locator_grid.shape[-1])
        scores = scores.masked_fill(
            ~eligible_support[:, None, None],
            torch.finfo(scores.dtype).min,
        )
        scores = scores + allowed_support.log()[:, :, None]
        attention = scores.softmax(dim=-1)
        pooled_low_dim = torch.einsum("btqg,btgd->btqd", attention, locator_grid)
        pooled = self.background_norm(self.background_value(pooled_low_dim))
        if not return_position:
            return pooled
        cell = torch.arange(height * width, device=attention.device)
        cell_y = torch.div(cell, width, rounding_mode="floor").float()
        cell_x = (cell % width).float()
        weight = attention.float()
        center_y = torch.einsum("btqg,g->btq", weight, cell_y)
        center_x = torch.einsum("btqg,g->btq", weight, cell_x)
        position = torch.stack(
            (
                center_x / float(max(width - 1, 1)),
                center_y / float(max(height - 1, 1)),
            ),
            dim=-1,
        )
        return pooled, position

    @staticmethod
    def _pool_candidate_positions(
        candidate_indices: torch.Tensor,
        candidate_attention: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if candidate_indices.shape != candidate_attention.shape:
            raise ValueError("candidate indices and attention must have equal shape")
        row = torch.div(candidate_indices, width, rounding_mode="floor")
        column = candidate_indices % width
        weight = candidate_attention.float()
        total = weight.sum(dim=-1).clamp_min(1e-6)
        center_y = (weight * row.float()).sum(dim=-1) / total
        center_x = (weight * column.float()).sum(dim=-1) / total
        return torch.stack(
            (
                center_x / float(max(width - 1, 1)),
                center_y / float(max(height - 1, 1)),
            ),
            dim=-1,
        )

    @staticmethod
    def _bounded_write_gate(
        normalized_timestep: torch.Tensor,
        raw_gate: torch.Tensor,
        writer_attention: torch.Tensor,
        reader_indices: torch.Tensor,
        writer_seed_indices: torch.Tensor,
        object_valid: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Return a non-parametric, per-time write amplitude.

        The intervention is intentionally conservative.  It uses the existing
        trainable noise gate, then multiplies it by (i) a monotone denoising
        progress factor, (ii) Writer support concentration, and (iii) spatial
        agreement between the original Reader top-1 support and the independent
        Writer seed.  It changes only the residual amplitude; route indices,
        object slots, and supervision remain untouched.
        """
        if writer_attention.ndim != 4:
            raise ValueError(
                "writer_attention must be [B,T,N,K], got "
                f"{tuple(writer_attention.shape)}"
            )
        if reader_indices.shape != writer_attention.shape:
            raise ValueError("Reader and Writer route shapes must match")
        expected_seed_shape = writer_attention.shape[:3]
        if tuple(writer_seed_indices.shape) != tuple(expected_seed_shape):
            raise ValueError(
                "writer_seed_indices must be [B,T,N], got "
                f"{tuple(writer_seed_indices.shape)}"
            )
        batch, times, objects, candidates = writer_attention.shape
        if tuple(object_valid.shape) != (batch, objects):
            raise ValueError("object_valid must be [B,N]")
        if candidates <= 1:
            concentration = torch.ones(
                batch, times, objects, device=writer_attention.device,
                dtype=writer_attention.dtype,
            )
        else:
            probabilities = writer_attention.float().clamp_min(1e-8)
            entropy = -(
                probabilities * probabilities.log()
            ).sum(dim=-1) / math.log(candidates)
            concentration = (1.0 - entropy).clamp(0.0, 1.0).to(
                dtype=writer_attention.dtype
            )

        # A peaked local distribution is useful but not sufficient: an
        # independent Writer that jumps far from the Reader is still risky.
        # Compare only the two top-1 seeds, with a soft spatial tolerance.
        reader_seed = reader_indices[..., 0].long()
        writer_seed = writer_seed_indices.long()
        reader_y = reader_seed // width
        reader_x = reader_seed % width
        writer_y = writer_seed // width
        writer_x = writer_seed % width
        distance = torch.sqrt(
            (reader_y.float() - writer_y.float()).square()
            + (reader_x.float() - writer_x.float()).square()
        )
        agreement = torch.exp(-distance / 2.0).to(dtype=writer_attention.dtype)

        valid = object_valid[:, None, :].to(dtype=writer_attention.dtype)
        valid_count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        # Keep the two config-declared factors separate.  In particular,
        # concentration is the one-minus-normalized-entropy confidence; it
        # must not be replaced by max probability or mixed with agreement.
        writer_confidence = (
            concentration * valid
        ).sum(dim=-1, keepdim=True) / valid_count
        reader_writer_seed_agreement = (
            agreement * valid
        ).sum(dim=-1, keepdim=True) / valid_count

        if tuple(normalized_timestep.shape) != (batch, 1):
            raise ValueError(
                "normalized_timestep must be [B,1], got "
                f"{tuple(normalized_timestep.shape)}"
            )
        if tuple(raw_gate.shape) != (batch, 1):
            raise ValueError(
                f"raw_gate must be [B,1], got {tuple(raw_gate.shape)}"
            )
        progress = (1.0 - normalized_timestep).clamp(0.0, 1.0)
        # Keep a small late-independent floor so the intervention cannot turn
        # into a hard no-write rule on a difficult but valid sample.
        time_factor = progress.square().clamp_min(0.05).unsqueeze(1)
        confidence_factor = 0.25 + 0.75 * writer_confidence
        agreement_factor = 0.25 + 0.75 * reader_writer_seed_agreement
        gate = (
            0.85
            * raw_gate[:, None, :]
            * time_factor
            * confidence_factor
            * agreement_factor
        )
        # Return [B,T,1,1] explicitly so batch size never participates in
        # broadcasting the per-example gate.
        return gate.clamp(0.0, 0.85).unsqueeze(-1)

    @staticmethod
    def _apply_continuous_state_mediation(
        updated: torch.Tensor,
        object_valid: torch.Tensor,
        mediation: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """Replace one outgoing state before the Writer, fail-closed.

        The donor receipt is intentionally CPU-detached when bound.  Moving it
        back here makes the intervention evaluation-only and prevents an
        accidental autograd path into a previous DiT call.
        """
        mode = mediation.get("mode")
        if mode not in {
            "target_swap",
            "wrong_object",
            "random_matched_stats",
        }:
            raise ValueError(f"unsupported continuous mediation mode: {mode!r}")
        donor_state = mediation.get("donor_state")
        if not isinstance(donor_state, torch.Tensor):
            raise ValueError("continuous mediation donor state must be a tensor")
        if tuple(donor_state.shape) != tuple(updated.shape):
            raise ValueError(
                "continuous mediation donor state must match outgoing state: "
                f"{tuple(donor_state.shape)} versus {tuple(updated.shape)}"
            )
        if not torch.isfinite(donor_state).all():
            raise ValueError("continuous mediation donor state must be finite")
        target_object = mediation.get("target_object")
        donor_object = mediation.get("source_object", mediation.get("donor_object"))
        if (
            not isinstance(target_object, int)
            or not isinstance(donor_object, int)
            or isinstance(target_object, bool)
            or isinstance(donor_object, bool)
        ):
            raise ValueError("continuous mediation object ids must be integers")
        objects = updated.shape[2]
        if not 0 <= target_object < objects or not 0 <= donor_object < objects:
            raise ValueError("continuous mediation object id is outside current slots")
        if not bool(object_valid[:, target_object].all()):
            raise ValueError("continuous mediation target object must be valid")
        if not bool(object_valid[:, donor_object].all()):
            raise ValueError("continuous mediation donor object must be valid")
        if mode == "wrong_object" and donor_object == target_object:
            raise ValueError("wrong_object mediation requires a distinct donor object")

        donor_state = donor_state.to(device=updated.device, dtype=updated.dtype)
        donor_value = donor_state[:, :, donor_object, :]
        if mode == "random_matched_stats":
            random_seed = mediation.get("random_seed")
            if not isinstance(random_seed, int) or isinstance(random_seed, bool):
                raise ValueError("random_matched_stats requires an integer random_seed")
            generator = torch.Generator(device=updated.device)
            generator.manual_seed(random_seed)
            noise = torch.randn(
                donor_value.shape,
                device=updated.device,
                dtype=updated.dtype,
                generator=generator,
            )
            noise = noise - noise.mean(dim=-1, keepdim=True)
            noise = noise / noise.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            mean = donor_value.mean(dim=-1, keepdim=True)
            std = donor_value.std(dim=-1, keepdim=True, unbiased=False)
            replacement = mean + noise * std
        else:
            replacement = donor_value

        # Do not mutate the interaction output in place: the receipt below is
        # a faithful pre-Writer state and the caller owns the original tensor.
        mediated = updated.clone()
        mediated[:, :, target_object, :] = replacement
        return mediated, {
            "mode": mode,
            "target_object": target_object,
            "source_object": donor_object,
            "random_seed": mediation.get("random_seed"),
        }

    def forward(
        self,
        hidden: torch.Tensor,
        first_frame_masks: torch.Tensor,
        object_valid: torch.Tensor,
        unary_temporal_residual: torch.Tensor,
        platform_mask: Optional[torch.Tensor],
        mu: torch.Tensor,
        mu_present: torch.Tensor,
        restitution: torch.Tensor,
        restitution_present: torch.Tensor,
        timestep: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        previous_object_state: Optional[torch.Tensor] = None,
        previous_route_indices: Optional[torch.Tensor] = None,
        previous_route_attention: Optional[torch.Tensor] = None,
        state_mediation: Optional[dict] = None,
        edge_mu: Optional[torch.Tensor] = None,
        edge_mu_present: Optional[torch.Tensor] = None,
        edge_restitution: Optional[torch.Tensor] = None,
        edge_restitution_present: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.last_write_gate = None
        self.last_continuous_state_mediation_trace = None
        (
            slots,
            location_logits,
            candidate_indices,
            candidate_attention,
            locator_grid,
        ) = self.router(
            hidden,
            first_frame_masks,
            object_valid,
            frames,
            height,
            width,
            previous_route_indices=previous_route_indices,
            previous_route_attention=previous_route_attention,
        )
        self.last_reader_indices = candidate_indices
        self.last_reader_attention = candidate_attention
        self.last_reader_grid_shape = (height, width)
        self.last_reader_support_gate_logits = self.router.last_support_gate_logits
        self.last_reader_support_gate_mask = self.router.last_support_gate_mask
        reader_state = slots if state_mediation is not None else None
        predicted_candidate_indices = candidate_indices
        reader_candidate_indices = candidate_indices
        mode = self.diagnostic_spatial_oracle_mode
        oracle_indices = None
        read_indices = candidate_indices
        if mode != "predicted":
            if candidate_indices is None:
                raise RuntimeError(
                    "spatial oracle diagnosis requires the sparse Top-K router"
                )
            labels = self._diagnostic_spatial_oracle_labels
            if labels is None:
                raise RuntimeError("spatial oracle mode has no bound future labels")
            oracle_indices = self._oracle_candidate_indices(
                labels,
                first_frame_masks,
                object_valid,
                frames,
                height,
                width,
                candidate_indices.shape[-1],
            )
            read_indices = (
                oracle_indices
                if mode in {"oracle_read", "oracle_both"}
                else candidate_indices
            )
            if mode in {"oracle_read", "oracle_both"}:
                slots = self.router.read_slots_at_indices(
                    hidden,
                    first_frame_masks,
                    object_valid,
                    frames,
                    height,
                    width,
                    read_indices,
                    candidate_attention,
                )
        if previous_object_state is not None:
            if not self.continuous_object_state:
                raise ValueError(
                    "previous_object_state requires continuous_object_state"
                )
            if tuple(previous_object_state.shape) != tuple(slots.shape):
                raise ValueError(
                    "previous_object_state must match current Reader slots: "
                    f"{tuple(previous_object_state.shape)} versus {tuple(slots.shape)}"
                )
            previous = self.previous_state_norm(previous_object_state)
            current = self.current_read_norm(slots)
            retain = torch.sigmoid(
                self.state_fusion_gate(torch.cat((previous, current), dim=-1))
            )
            slots = retain * previous_object_state + (1.0 - retain) * slots
            slots = slots * object_valid[:, None, :, None].to(slots.dtype)
        fused_state = slots if state_mediation is not None else None
        if unary_temporal_residual.shape[:3] != slots.shape[:3]:
            raise ValueError(
                "F/g temporal residual must match [B,T,N], got "
                f"{tuple(unary_temporal_residual.shape[:3])} versus "
                f"{tuple(slots.shape[:3])}"
            )
        slots = slots + unary_temporal_residual
        temporal_input = slots if state_mediation is not None else None
        slots = self.temporal(slots, object_valid)
        temporal_delta = (
            slots - temporal_input if temporal_input is not None else None
        )

        batch, times, objects, _ = slots.shape
        node_position = None
        if self.pair_physics or self.platform_support or self.independent_edge:
            spatial_contact_features = (
                self.contact_head_enabled and not self.three_contact_gates
            )
            support_result = self._pool_support_per_time(
                locator_grid,
                location_logits,
                object_valid,
                platform_mask,
                return_position=spatial_contact_features,
            )
            if spatial_contact_features:
                support, support_position = support_result
                if read_indices is None:
                    raise RuntimeError("pair event supervision requires Reader indices")
                object_position = self._pool_candidate_positions(
                    read_indices, candidate_attention, height, width
                )
                node_position = torch.cat((object_position, support_position), dim=2)
            else:
                support = support_result
        interaction_input = slots if state_mediation is not None else None
        if isinstance(
            self.interaction,
            (TemporalObjectInteractionBlock, IndependentUndirectedEdgeInteractionBlock),
        ):
            self.interaction.capture_diagnostic_state = state_mediation is not None
        if self.pair_physics:
            updated = self.interaction(
                slots,
                support,
                object_valid,
                mu,
                mu_present,
                restitution,
                restitution_present,
            )
        else:
            if self.platform_support or self.independent_edge:
                support = support.reshape(
                    batch * times, support.shape[2], support.shape[3]
                )
            else:
                support = self._pool_support(
                    locator_grid, location_logits, object_valid
                )
                support = support[:, None].expand(
                    batch, times, support.shape[1], support.shape[2]
                ).reshape(batch * times, support.shape[1], support.shape[2])
            per_time_objects = slots.reshape(batch * times, objects, -1)
            per_time_valid = object_valid[:, None].expand(
                batch, times, objects
            ).reshape(batch * times, objects)
            if self.independent_edge:
                edge_values = (
                    edge_mu,
                    edge_mu_present,
                    edge_restitution,
                    edge_restitution_present,
                )
                if any(value is None for value in edge_values):
                    raise ValueError("independent-edge route requires all edge conditions")
                per_time_edges = [
                    value[:, None]
                    .expand(batch, times, objects, objects + 1)
                    .reshape(batch * times, objects, objects + 1)
                    for value in edge_values
                ]
                self.last_e_value_intervention_receipt = None
                if self.e_value_intervention is not None:
                    spec = self.e_value_intervention
                    left, right = spec["object_pair"]
                    latent_times = tuple(spec["latent_times"])
                    replacement = float(spec["replacement_pair_e"])
                    per_time_e = per_time_edges[2].reshape(
                        batch, times, objects, objects + 1
                    ).clone()
                    before = per_time_e[:, latent_times, left, right].detach().float()
                    per_time_e[:, latent_times, left, right] = replacement
                    # Object-object rows are stored in both directions.  An
                    # object-support value has no support receiver row in the
                    # compact [N,N+1] condition and is symmetrized later by
                    # _full_edge_tensor.
                    if right < objects:
                        per_time_e[:, latent_times, right, left] = replacement
                    per_time_edges[2] = per_time_e.reshape(
                        batch * times, objects, objects + 1
                    )
                    self.last_e_value_intervention_receipt = {
                        "route": "independent_edge_pair_e",
                        "object_pair": (left, right),
                        "latent_times": latent_times,
                        "replacement_pair_e": replacement,
                        "source_pair_e": float(spec["source_pair_e"]),
                        "before_mean": float(before.mean().item()),
                        "after_mean": replacement,
                        "selected_value_count": int(
                            before.numel() * (2 if right < objects else 1)
                        ),
                    }
                updated = self.interaction(
                    per_time_objects,
                    support,
                    per_time_valid,
                    *per_time_edges,
                    node_position=(
                        None
                        if node_position is None
                        else node_position.reshape(batch * times, objects + 1, 2)
                    ),
                    temporal_steps=(times if node_position is not None else None),
                ).reshape(batch, times, objects, -1)
            else:
                per_time_mu = mu[:, None].expand(batch, times, objects).reshape(
                    batch * times, objects
                )
                per_time_mu_present = mu_present[:, None].expand(
                    batch, times, objects
                ).reshape(batch * times, objects)
                per_time_restitution = restitution[:, None].expand(
                    batch, times, objects
                ).reshape(batch * times, objects)
                per_time_restitution_present = restitution_present[:, None].expand(
                    batch, times, objects
                ).reshape(batch * times, objects)
                updated = self.interaction(
                    per_time_objects,
                    support,
                    per_time_valid,
                    per_time_mu,
                    per_time_mu_present,
                    per_time_restitution,
                    per_time_restitution_present,
                ).reshape(batch, times, objects, -1)

        updated = self._apply_event_after_effect(updated, slots, object_valid)
        interaction_total_delta = (
            updated - interaction_input if interaction_input is not None else None
        )
        mediation_receipt = None
        if state_mediation is not None and state_mediation.get("mode") is not None:
            updated, mediation_receipt = self._apply_continuous_state_mediation(
                updated, object_valid, state_mediation
            )
        self.last_updated_objects = updated

        if self.writer is not None:
            (
                self.last_writer_logits,
                candidate_indices,
                candidate_attention,
                self.last_writer_seed_indices,
            ) = self.writer(
                hidden,
                updated,
                object_valid,
                frames,
                height,
                width,
                anchor_indices=(
                    reader_candidate_indices
                    if self.correctable_route_carry
                    else None
                ),
                anchor_attention=(
                    self.last_reader_attention
                    if self.correctable_route_carry
                    else None
                ),
            )
            self.last_writer_support_gate_logits = (
                self.writer.last_support_gate_logits
            )
            self.last_writer_support_gate_mask = self.writer.last_support_gate_mask
            self.last_writer_support_gate_allowed_mask = (
                self.writer.last_support_gate_allowed_mask
            )
            predicted_writer_indices = candidate_indices
        else:
            predicted_writer_indices = candidate_indices
            self.last_writer_logits = None
            self.last_writer_seed_indices = None
            self.last_writer_support_gate_logits = None
            self.last_writer_support_gate_mask = None
            self.last_writer_support_gate_allowed_mask = None

        if mode != "predicted":
            if oracle_indices is None or read_indices is None:
                raise RuntimeError("spatial oracle routes were not initialized")
            write_indices = (
                oracle_indices
                if mode in {"oracle_write", "oracle_both"}
                else predicted_writer_indices
            )
            candidate_indices = write_indices
            self.last_diagnostic_spatial_routes = {
                "mode": mode,
                # Retain the legacy alias for old non-decoupled diagnostics.
                "predicted_indices": predicted_candidate_indices.detach().cpu(),
                "predicted_read_indices": predicted_candidate_indices.detach().cpu(),
                "predicted_write_indices": predicted_writer_indices.detach().cpu(),
                "oracle_indices": oracle_indices.detach().cpu(),
                "read_indices": read_indices.detach().cpu(),
                "write_indices": write_indices.detach().cpu(),
                "candidate_attention": candidate_attention.detach().float().cpu(),
            }
        else:
            self.last_diagnostic_spatial_routes = None

        if self.writer is not None:
            # During an oracle-write diagnostic this is the route actually
            # consumed by scatter_add_, while logits/weights/seeds remain the
            # independent Writer's predictions.
            self.last_writer_indices = candidate_indices
            self.last_writer_attention = candidate_attention
        else:
            self.last_writer_indices = None
            self.last_writer_attention = None

        grid_cells = height * width
        projected_objects = self.output_projection(updated)
        if candidate_indices is None:
            if candidate_attention.shape != (batch, times, objects, grid_cells):
                raise ValueError(
                    "dense template routing must be [B,T,N,H*W], got "
                    f"{tuple(candidate_attention.shape)}"
                )
            denominator = candidate_attention.sum(dim=2).clamp_min(1.0)
            normalized_write = candidate_attention / denominator[:, :, None]
            projected_grid = torch.einsum(
                "btog,btod->btgd", normalized_write, projected_objects
            )
        else:
            denominator = candidate_attention.new_zeros(batch, times, grid_cells)
            flattened_indices = candidate_indices.reshape(
                batch, times, objects * candidate_indices.shape[-1]
            )
            denominator.scatter_add_(
                -1,
                flattened_indices,
                candidate_attention.reshape(
                    batch, times, objects * candidate_attention.shape[-1]
                ),
            )
            candidate_denominator = self.router._gather_candidates(
                denominator, candidate_indices
            ).clamp_min(1.0)
            normalized_write = candidate_attention / candidate_denominator
            writer_values = projected_objects[..., None, :]
            if self.position_dependent_writer_value:
                if self.writer is None:
                    raise RuntimeError(
                        "position-dependent Writer value requires Writer route"
                    )
                normalized_hidden = self.writer.input_norm(hidden).reshape(
                    batch, times, grid_cells, hidden.shape[-1]
                )
                candidate_hidden = self.router._gather_candidates(
                    normalized_hidden, candidate_indices
                )
                candidate_y = torch.div(
                    candidate_indices, width, rounding_mode="floor"
                ).to(dtype=projected_objects.dtype)
                candidate_x = (candidate_indices % width).to(
                    dtype=projected_objects.dtype
                )
                route_weight = candidate_attention.to(projected_objects.dtype)
                route_mass = route_weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
                center_x = (route_weight * candidate_x).sum(
                    dim=-1, keepdim=True
                ) / route_mass
                center_y = (route_weight * candidate_y).sum(
                    dim=-1, keepdim=True
                ) / route_mass
                relative_position = torch.stack(
                    (
                        (candidate_x - center_x) / float(max(width - 1, 1)),
                        (candidate_y - center_y) / float(max(height - 1, 1)),
                    ),
                    dim=-1,
                )
                local_feature = torch.nn.functional.silu(
                    self.local_writer_state_projection(updated)[..., None, :]
                    + self.local_writer_hidden_projection(candidate_hidden)
                    + self.local_writer_position_projection(relative_position)
                )
                writer_values = writer_values + self.local_writer_output_projection(
                    local_feature
                )
            elif self.relative_position_writer_value:
                writer_values = self._relative_position_writer_values(
                    updated,
                    projected_objects,
                    candidate_indices,
                    candidate_attention,
                    width,
                )
            candidate_residuals = (
                normalized_write[..., None] * writer_values
            ).reshape(
                batch,
                times,
                objects * candidate_indices.shape[-1],
                hidden.shape[-1],
            )
            projected_grid = hidden.new_zeros(
                batch,
                times,
                grid_cells,
                hidden.shape[-1],
                dtype=projected_objects.dtype,
            )
            projected_grid.scatter_add_(
                2,
                flattened_indices[..., None].expand(
                    batch,
                    times,
                    objects * candidate_indices.shape[-1],
                    hidden.shape[-1],
                ),
                candidate_residuals,
            )
        time_mask = projected_grid.new_ones(1, times, 1, 1)
        time_mask[:, 0] = 0
        projected = (projected_grid * time_mask).reshape(
            batch, times * grid_cells, hidden.shape[-1]
        )
        normalized_timestep = (
            timestep.float().reshape(hidden.shape[0], -1).mean(dim=1, keepdim=True)
            / 1000.0
        ).clamp(0.0, 1.0)
        raw_gate = torch.sigmoid(self.noise_gate(normalized_timestep))
        if self.write_gate_mode == "erase_then_write_v1":
            hidden_grid = hidden.reshape(
                batch, times, grid_cells, hidden.shape[-1]
            )
            local_support = (denominator > 0).to(dtype=projected_grid.dtype)
            erase_logits = self.erase_gate(
                torch.cat(
                    (
                        hidden_grid.to(dtype=projected_grid.dtype),
                        projected_grid,
                    ),
                    dim=-1,
                )
            )
            erase_gate = torch.sigmoid(erase_logits) * local_support[..., None]
            erase_gate = erase_gate * time_mask
            self.last_write_gate = erase_gate.detach()
            mixed = (
                hidden_grid.to(dtype=projected_grid.dtype) * (1.0 - erase_gate)
                + projected_grid * erase_gate
            )
            hidden = mixed.reshape_as(hidden).to(dtype=hidden.dtype)
        elif self.write_gate_mode == "bounded_v1":
            if reader_candidate_indices is None:
                raise RuntimeError("bounded_v1 requires sparse Reader indices")
            if self.last_writer_attention is None or self.last_writer_seed_indices is None:
                raise RuntimeError("bounded_v1 requires Writer route diagnostics")
            gate = self._bounded_write_gate(
                normalized_timestep,
                raw_gate,
                self.last_writer_attention,
                reader_candidate_indices,
                self.last_writer_seed_indices,
                object_valid,
                height,
                width,
            )
            self.last_write_gate = gate.detach()
            projected = (projected_grid * time_mask * gate).reshape(
                batch, times * grid_cells, hidden.shape[-1]
            )
            hidden = hidden + (
                self.residual_scale * projected
            ).to(dtype=hidden.dtype)
        else:
            gate = raw_gate[:, None]
            self.last_write_gate = raw_gate.detach().view(batch, 1, 1, 1)
            hidden = hidden + (
                gate * self.residual_scale * projected
            ).to(dtype=hidden.dtype)
        if state_mediation is not None:
            def receipt_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                return None if value is None else value.detach().float().cpu().clone()

            def receipt_interaction(
                value: Optional[torch.Tensor],
            ) -> Optional[torch.Tensor]:
                if value is None:
                    return None
                # TemporalObjectInteractionBlock runs the ordinary/no-
                # interaction route as [B*T,N,D]; receipts must restore the
                # public state-chain layout before comparing its two terms to
                # the group-level [B,T,N,D] total delta.
                if value.ndim == 3 and tuple(value.shape) == (
                    batch * times,
                    objects,
                    updated.shape[-1],
                ):
                    value = value.reshape(batch, times, objects, -1)
                if tuple(value.shape) != tuple(updated.shape):
                    raise RuntimeError(
                        "continuous mediation interaction receipt has unexpected shape: "
                        f"{tuple(value.shape)} versus {tuple(updated.shape)}"
                    )
                return receipt_tensor(value)

            self.last_continuous_state_mediation_trace = {
                "call_index": state_mediation["call_index"],
                "block_index": state_mediation["block_index"],
                "reader_state": receipt_tensor(reader_state),
                "previous_state": receipt_tensor(previous_object_state),
                "fused_state": receipt_tensor(fused_state),
                "unary_residual": receipt_tensor(unary_temporal_residual),
                "temporal_delta": receipt_tensor(temporal_delta),
                "interaction_message": receipt_interaction(
                    getattr(self.interaction, "last_interaction_message", None)
                ),
                "e_modulation_effect": receipt_interaction(
                    getattr(self.interaction, "last_e_modulation_effect", None)
                ),
                "mu_modulation_effect": receipt_interaction(
                    getattr(self.interaction, "last_mu_modulation_effect", None)
                ),
                "interaction_ffn_delta": receipt_interaction(
                    getattr(self.interaction, "last_interaction_ffn_delta", None)
                ),
                "interaction_total_delta": receipt_tensor(interaction_total_delta),
                # Backward-compatible alias for older consumers of the first
                # diagnostic receipt; it is unambiguously the total update.
                "interaction_delta": receipt_tensor(interaction_total_delta),
                "updated_outgoing_state": receipt_tensor(updated),
                "outgoing_state": receipt_tensor(updated),
                "writer_object_residual": receipt_tensor(projected_objects),
                "writer_route": {
                    "indices": receipt_tensor(self.last_writer_indices),
                    "attention": receipt_tensor(self.last_writer_attention),
                    "seed_indices": receipt_tensor(self.last_writer_seed_indices),
                },
                "writer_gate": receipt_tensor(self.last_write_gate),
                "mediation": mediation_receipt,
            }
        return hidden, location_logits


class SparseObjectInteractionTemporalAdapter(SparseObjectInteractionAdapter):
    """V2-compatible route with same-object temporal then per-time interaction."""

    def __init__(
        self,
        wan_dim: int,
        injection_blocks: Iterable[int] = (2, 6, 10, 14, 18, 22, 25, 27),
        object_dim: int = 512,
        locator_dim: int = 128,
        num_heads: int = 8,
        topk: int = 8,
        force_schedule: bool = False,
        pair_physics: bool = False,
        platform_support: bool = False,
        previous_frame_tracking: bool = False,
        joint_template_routing: bool = False,
        mu_message_gate_alpha: Optional[float] = None,
        decoupled_writer: bool = False,
        continuous_object_state: bool = False,
        writer_radius: int = 1,
        write_gate_mode: Optional[str] = None,
        causal_object_temporal: Optional[bool] = None,
        event_after_effect: Optional[bool] = None,
        independent_edge: bool = False,
        pair_event_supervision: Optional[bool] = None,
        shared_contact_gate: Optional[bool] = None,
        three_contact_gates: Optional[bool] = None,
        always_on_object_object_gate: Optional[bool] = None,
        shared_three_contact_gate: Optional[bool] = None,
    ):
        # Do not call the parent constructor: it would create the old pooled
        # groups.  Static validation/helpers from the parent remain reusable.
        nn.Module.__init__(self)
        self.injection_blocks = tuple(int(index) for index in injection_blocks)
        self.force_schedule = bool(force_schedule)
        self.pair_physics = bool(pair_physics)
        self.platform_support = bool(platform_support)
        self.independent_edge = bool(independent_edge)
        self.disable_gravity = _environment_flag("PHYSICAL_WM_DISABLE_GRAVITY")
        self.pair_event_supervision = (
            _environment_flag("PHYSICAL_WM_PAIR_EVENT_SUPERVISION")
            if pair_event_supervision is None
            else bool(pair_event_supervision)
        )
        self.shared_contact_gate = (
            _environment_flag("PHYSICAL_WM_SHARED_CONTACT_GATE")
            if shared_contact_gate is None
            else bool(shared_contact_gate)
        )
        self.three_contact_gates = (
            _environment_flag("PHYSICAL_WM_THREE_CONTACT_GATES")
            if three_contact_gates is None
            else bool(three_contact_gates)
        )
        self.always_on_object_object_gate = (
            _environment_flag("PHYSICAL_WM_ALWAYS_ON_OBJECT_OBJECT_GATE")
            if always_on_object_object_gate is None
            else bool(always_on_object_object_gate)
        )
        self.shared_three_contact_gate = (
            _environment_flag("PHYSICAL_WM_SHARED_THREE_CONTACT_GATE")
            if shared_three_contact_gate is None
            else bool(shared_three_contact_gate)
        )
        if sum(
            (
                self.pair_event_supervision,
                self.shared_contact_gate,
                self.three_contact_gates,
            )
        ) > 1:
            raise ValueError(
                "legacy pair-event, shared-contact, and three-contact routes "
                "are mutually exclusive"
            )
        if self.always_on_object_object_gate and not self.three_contact_gates:
            raise ValueError(
                "always-on object-object gates require the three-contact route"
            )
        if self.shared_three_contact_gate and not self.three_contact_gates:
            raise ValueError(
                "shared three-contact gate requires the three-contact route"
            )
        if self.shared_three_contact_gate and self.always_on_object_object_gate:
            raise ValueError("shared and always-on gate ablations are exclusive")
        self.e_pair_event_loss_mode = os.environ.get(
            "PHYSICAL_WM_E_PAIR_EVENT_LOSS_MODE", "hard_slot"
        ).strip().lower()
        if self.e_pair_event_loss_mode not in {
            "hard_slot",
            "event_distribution_w1",
            "weak_pair",
            "weak_pair_crossblock_union",
            "coarse_window_pair",
        }:
            raise ValueError(
                "unsupported PHYSICAL_WM_E_PAIR_EVENT_LOSS_MODE"
            )
        self.continuous_object_state = bool(continuous_object_state)
        self.hard_support_enabled = _environment_flag(
            "PHYSICAL_WM_HARD_SUPPORT", False
        )
        self.hard_support_loss_weight = _environment_float(
            "PHYSICAL_WM_HARD_SUPPORT_LOSS_WEIGHT", 0.3
        )
        if self.hard_support_loss_weight < 0.0:
            raise ValueError(
                "PHYSICAL_WM_HARD_SUPPORT_LOSS_WEIGHT must be non-negative"
            )
        self.correctable_route_carry = _environment_flag(
            "PHYSICAL_WM_CORRECTABLE_ROUTE_CARRY", False
        )
        if self.correctable_route_carry and not self.continuous_object_state:
            raise ValueError("cross-block route carry requires continuous object state")
        if self.continuous_object_state and not decoupled_writer:
            raise ValueError(
                "continuous object state currently requires the independent Writer"
            )
        self._continuous_state: Optional[torch.Tensor] = None
        self.h2_ablation = h2_ablation_mode()
        self.state_carry_enabled = self.h2_ablation != "no_state_carry"
        self.reader_route_carry_enabled = self.h2_ablation != "no_reader_route_carry"
        self._continuous_route_indices: Optional[torch.Tensor] = None
        self._continuous_route_attention: Optional[torch.Tensor] = None
        self._next_continuous_stage = 0
        # All mediation state is evaluation-only Python state.  In particular,
        # do not register a buffer/module here: existing checkpoints must keep
        # exactly the same state_dict key set when this probe is unused.
        self._continuous_state_call_index = 0
        self._continuous_state_mediation_enabled = False
        self._continuous_state_mediation_donor: Optional[dict] = None
        self._continuous_state_mediation_plan: Optional[dict] = None
        self._continuous_state_mediation_applied: set[tuple[int, int]] = set()
        self.continuous_state_mediation_trace: list[dict] = []
        self.write_gate_mode = (
            write_gate_mode
            if write_gate_mode is not None
            else os.environ.get("PHYSICAL_WM_WRITE_GATE_MODE", "legacy")
        ).strip().lower()
        # Temporal-evolution route.  Both default to the historical behavior and
        # are opt-in through the environment, exactly like write_gate_mode, so
        # every existing checkpoint keeps an identical state_dict key set.
        self.causal_object_temporal = (
            _environment_flag("PHYSICAL_WM_CAUSAL_OBJECT_TEMPORAL")
            if causal_object_temporal is None
            else bool(causal_object_temporal)
        )
        self.event_after_effect = (
            _environment_flag("PHYSICAL_WM_EVENT_AFTER_EFFECT")
            if event_after_effect is None
            else bool(event_after_effect)
        )
        if (
            self.pair_event_supervision
            or self.shared_contact_gate
            or self.three_contact_gates
        ) and not (
            self.independent_edge
            and decoupled_writer
            and self.continuous_object_state
            and not self.event_after_effect
        ):
            raise ValueError(
                "pair event supervision requires Independent Edge with "
                "hidden-aware Continuous Writer and event-after-effect off"
            )
        if (self.platform_support or self.independent_edge) and not self.force_schedule:
            raise ValueError("support-token temporal routes require force_schedule")
        if (
            not self.injection_blocks
            or len(set(self.injection_blocks)) != len(self.injection_blocks)
            or any(index < 0 for index in self.injection_blocks)
        ):
            raise ValueError(
                "temporal sparse route requires non-empty distinct blocks"
            )
        if object_dim % num_heads:
            raise ValueError("object_dim must be divisible by num_heads")
        self.groups = nn.ModuleDict(
            {
                str(index): SparseObjectInteractionTemporalGroup(
                    wan_dim,
                    object_dim,
                    locator_dim,
                    num_heads,
                    topk,
                    pair_physics=self.pair_physics,
                    platform_support=self.platform_support,
                    independent_edge=self.independent_edge,
                    pair_event_supervision=self.pair_event_supervision,
                    shared_contact_gate=self.shared_contact_gate,
                    three_contact_gates=self.three_contact_gates,
                    always_on_object_object_gate=(
                        self.always_on_object_object_gate
                    ),
                    shared_three_contact_gate=self.shared_three_contact_gate,
                    previous_frame_tracking=previous_frame_tracking,
                    joint_template_routing=joint_template_routing,
                    mu_message_gate_alpha=mu_message_gate_alpha,
                    decoupled_writer=decoupled_writer,
                    continuous_object_state=(
                        self.continuous_object_state
                        and index != self.injection_blocks[0]
                    ),
                    writer_radius=writer_radius,
                    write_gate_mode=self.write_gate_mode,
                    causal_object_temporal=self.causal_object_temporal,
                    event_after_effect=self.event_after_effect,
                )
                for index in self.injection_blocks
            }
        )

    def configure_temporal_evolution_ablation(
        self,
        event_after_effect: Optional[bool] = None,
        causal_object_temporal: Optional[bool] = None,
        capture_norms: Optional[bool] = None,
    ) -> dict:
        """Toggle the temporal-evolution route at inference without reloading.

        Only plain Python attributes are touched, so the checkpoint and the
        state_dict key set stay unchanged.  ``None`` leaves a switch alone.
        """
        for group in self.groups.values():
            if event_after_effect is not None:
                group.event_after_effect_runtime_enabled = bool(event_after_effect)
            if causal_object_temporal is not None:
                group.temporal.causal_runtime_enabled = bool(causal_object_temporal)
            if capture_norms is not None:
                group.capture_event_after_effect_norms = bool(capture_norms)
        return {
            "event_after_effect_built": self.event_after_effect,
            "causal_object_temporal_built": self.causal_object_temporal,
            "event_after_effect_active": [
                bool(group.event_after_effect and group.event_after_effect_runtime_enabled)
                for group in self.groups.values()
            ],
            "causal_object_temporal_active": [
                bool(group.temporal.causal and group.temporal.causal_runtime_enabled)
                for group in self.groups.values()
            ],
        }

    def configure_e_value_intervention(
        self, intervention: Optional[dict] = None
    ) -> None:
        """Replace one symmetric object-pair e value in selected latent slots."""
        if intervention is not None:
            if not self.independent_edge:
                raise RuntimeError("pair-e value intervention requires Independent Edge")
            if not isinstance(intervention, dict):
                raise ValueError("e value intervention must be a dict or None")
            pair = intervention.get("object_pair")
            latent_times = intervention.get("latent_times")
            replacement = intervention.get("replacement_pair_e")
            source = intervention.get("source_pair_e")
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in pair
                )
                or pair[0] == pair[1]
                or not isinstance(latent_times, (tuple, list))
                or not latent_times
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 0 <= value < 13
                    for value in latent_times
                )
                or len(set(latent_times)) != len(latent_times)
                or not isinstance(replacement, (int, float))
                or isinstance(replacement, bool)
                or not 0.0 <= float(replacement) <= 1.0
                or not isinstance(source, (int, float))
                or isinstance(source, bool)
                or not 0.0 <= float(source) <= 1.0
                or abs(float(source) - float(replacement)) < 0.5
            ):
                raise ValueError("invalid e value intervention contract")
        for group in self.groups.values():
            group.e_value_intervention = (
                None if intervention is None else dict(intervention)
            )
            group.last_e_value_intervention_receipt = None

    def e_value_intervention_receipts(self) -> dict[int, dict]:
        receipts: dict[int, dict] = {}
        for block in self.injection_blocks:
            receipt = self.groups[str(block)].last_e_value_intervention_receipt
            if receipt is None:
                raise RuntimeError(
                    f"e value intervention did not execute at block {block}"
                )
            receipts[int(block)] = dict(receipt)
        return receipts

    def event_after_effect_report(self) -> dict:
        """Per-injection-block learned persistence scalars and last-call norms."""
        report = {
            "event_after_effect": self.event_after_effect,
            "causal_object_temporal": self.causal_object_temporal,
            "blocks": {},
        }
        for index, group in zip(self.injection_blocks, self.groups.values()):
            entry = group.event_after_effect_parameters() or {}
            entry["last_history_norm"] = group.last_event_history_norm
            entry["last_after_effect_norm"] = group.last_event_after_effect_norm
            report["blocks"][int(index)] = entry
        return report

    def reset_continuous_state(self) -> None:
        """Reset the explicit object chain at the start of each DiT call."""
        self._continuous_state = None
        self._continuous_route_indices = None
        self._continuous_route_attention = None
        self._next_continuous_stage = 0
        # WanVideoDiT invokes this exactly once per DiT call.  The diagnostic
        # session resets this counter before each donor/recipient generation,
        # so both enumerate the same denoising calls as 1..N.
        self._continuous_state_call_index += 1

    def configure_edge_causal_diagnostic(
        self, intervention: Optional[dict] = None
    ) -> None:
        """Configure an evaluation-only QK-edge/e-message causal probe.

        The same frozen intervention is applied at every denoising call while
        each group's ``latent_times`` selects the physical video-time window.
        Passing ``None`` restores the ordinary inference path.
        """
        if intervention is not None and not isinstance(intervention, dict):
            raise ValueError("edge causal intervention must be a dict or None")
        supported = (
            TemporalObjectInteractionBlock,
            IndependentUndirectedEdgeInteractionBlock,
        )
        for group in self.groups.values():
            if not isinstance(group.interaction, supported):
                if intervention is None:
                    # Reset is intentionally valid for every architecture.  A
                    # multi-task evaluator calls it before ordinary predicted
                    # inference even when the active interaction block never
                    # supported this legacy diagnostic.
                    continue
                raise ValueError("edge causal diagnosis requires a supported interaction block")
            if (
                intervention is not None
                and isinstance(
                    group.interaction, IndependentUndirectedEdgeInteractionBlock
                )
                and intervention.get("mode") not in {
                    "delete_edge",
                    "flatten_edge_time",
                    "reverse_edge_time",
                    "ablate_e_path",
                    "ablate_mu_path",
                }
            ):
                raise ValueError(
                    "independent-edge causal probe supports delete_edge, "
                    "flatten_edge_time, reverse_edge_time, ablate_e_path, "
                    "or ablate_mu_path"
                )
            group.interaction.edge_causal_intervention = (
                None if intervention is None else dict(intervention)
            )
            group.interaction.last_edge_causal_receipt = None

    def edge_causal_diagnostic_receipts(self) -> dict[int, dict]:
        """Return one exact receipt per injected block after generation."""
        receipts: dict[int, dict] = {}
        for block in self.injection_blocks:
            interaction = self.groups[str(block)].interaction
            receipt = interaction.last_edge_causal_receipt
            if receipt is None:
                raise RuntimeError(
                    f"edge causal diagnostic did not execute at block {block}"
                )
            receipts[int(block)] = dict(receipt)
        return receipts

    def configure_continuous_state_mediation_diagnostic(
        self,
        enabled: bool = True,
        donor_trace: Optional[list[dict]] = None,
        intervention: Optional[dict] = None,
    ) -> None:
        """Enable/disable the evaluation-only Continuous Writer receipt.

        When enabled, every injected block emits a CPU-detached receipt.  A
        donor receipt and a frozen intervention plan can optionally be
        configured afterwards with the explicit methods below.  Normal training and
        inference leave this disabled, so their tensor operations are
        unchanged.
        """
        if not isinstance(enabled, bool):
            raise ValueError("continuous mediation enabled must be a bool")
        if enabled and not self.continuous_object_state:
            raise ValueError(
                "continuous mediation diagnosis requires continuous_object_state"
            )
        self._continuous_state_mediation_enabled = enabled
        self._continuous_state_call_index = 0
        self.continuous_state_mediation_trace = []
        self._continuous_state_mediation_donor = None
        self._continuous_state_mediation_plan = None
        self._continuous_state_mediation_applied = set()
        if donor_trace is not None:
            self.bind_continuous_state_mediation_donor_trace(donor_trace)
        if intervention is not None:
            if not isinstance(intervention, dict):
                raise ValueError("continuous mediation intervention must be a dict")
            self.configure_continuous_state_mediation_intervention(**intervention)

    def bind_continuous_state_mediation_donor_trace(
        self,
        donor_trace: list[dict],
    ) -> None:
        """Bind a previously recorded CPU receipt as an immutable donor."""
        if not self._continuous_state_mediation_enabled:
            raise RuntimeError("enable continuous mediation diagnosis before binding a donor")
        if not isinstance(donor_trace, (list, tuple)) or not donor_trace:
            raise ValueError("continuous mediation donor trace must be a non-empty list")
        indexed = {}
        for entry in donor_trace:
            if not isinstance(entry, dict):
                raise ValueError("continuous mediation donor trace entries must be dicts")
            call_index = entry.get("call_index")
            block_index = entry.get("block_index")
            state = entry.get("updated_outgoing_state", entry.get("outgoing_state"))
            if (
                not isinstance(call_index, int)
                or isinstance(call_index, bool)
                or not isinstance(block_index, int)
                or isinstance(block_index, bool)
                or block_index not in self.injection_blocks
                or not isinstance(state, torch.Tensor)
                or state.ndim != 4
                or not torch.isfinite(state).all()
            ):
                raise ValueError("continuous mediation donor trace has an invalid receipt")
            key = (call_index, block_index)
            if key in indexed:
                raise ValueError("continuous mediation donor trace has duplicate call/block")
            indexed[key] = state.detach().float().cpu().clone().contiguous()
        self._continuous_state_mediation_donor = indexed
        self._continuous_state_mediation_plan = None
        self._continuous_state_mediation_applied = set()

    def configure_continuous_state_mediation_intervention(
        self,
        *,
        call_index: int,
        block_index: int,
        target_object: int,
        mode: str = "target_swap",
        source_object: Optional[int] = None,
        wrong_object: Optional[int] = None,
        random_seed: int = 0,
    ) -> None:
        """Schedule one point; retained as a compatibility wrapper for plan."""
        # ``wrong_object`` was the old name for the source slot.  Keep that
        # spelling only for single-point callers while the plan uses the clear
        # source-object -> target-recipient contract.
        if source_object is None and wrong_object is not None:
            source_object = wrong_object
        self.configure_continuous_state_mediation_plan(
            call_indices=(call_index,),
            block_indices=(block_index,),
            target_object=target_object,
            mode=mode,
            source_object=source_object,
            wrong_object=wrong_object,
            random_seed=random_seed,
        )

    def configure_continuous_state_mediation_plan(
        self,
        *,
        call_indices: Iterable[int] | str,
        block_indices: Iterable[int] | str,
        target_object: int,
        mode: str = "target_swap",
        source_object: Optional[int] = None,
        wrong_object: Optional[int] = None,
        random_seed: int = 0,
    ) -> None:
        """Freeze a complete multi-call/block mediation plan.

        ``source_object`` is the physical donor object.  For ``target_swap``
        it must equal ``target_object``.  For ``wrong_object``,
        ``target_object`` is deliberately the wrong recipient and must differ.
        ``call_indices`` and ``block_indices`` accept a non-empty iterable of
        integers or ``"all"``.  ``"all"`` means every call/block present in
        the bound donor trace; the Cartesian plan must exist completely before
        inference starts.  Each planned point is applied exactly once, and
        :meth:`continuous_state_mediation_plan_status` reports completion.
        """
        if not self._continuous_state_mediation_enabled:
            raise RuntimeError("enable continuous mediation diagnosis before intervention")
        donor = self._continuous_state_mediation_donor
        if donor is None:
            raise RuntimeError("bind a continuous mediation donor trace first")
        if mode not in {"target_swap", "wrong_object", "random_matched_stats"}:
            raise ValueError(f"unsupported continuous mediation mode: {mode!r}")
        if not isinstance(target_object, int) or isinstance(target_object, bool):
            raise ValueError("continuous mediation target object must be an integer")
        if not isinstance(random_seed, int) or isinstance(random_seed, bool):
            raise ValueError("continuous mediation random_seed must be an integer")
        if source_object is None and wrong_object is not None:
            source_object = wrong_object
        if not isinstance(source_object, int) or isinstance(source_object, bool):
            raise ValueError("continuous mediation mode requires source_object")

        donor_calls = sorted({call for call, _ in donor})
        donor_blocks = sorted({block for _, block in donor})

        def resolve(values: Iterable[int] | str, available: list[int], name: str) -> tuple[int, ...]:
            if values == "all":
                return tuple(available)
            if isinstance(values, str):
                raise ValueError(f"continuous mediation {name} must be integers or 'all'")
            try:
                resolved = tuple(values)
            except TypeError as error:
                raise ValueError(
                    f"continuous mediation {name} must be a non-empty iterable"
                ) from error
            if not resolved or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in resolved
            ):
                raise ValueError(
                    f"continuous mediation {name} must be a non-empty integer iterable"
                )
            if len(set(resolved)) != len(resolved) or any(value not in available for value in resolved):
                raise ValueError(f"continuous mediation {name} is outside bound donor trace")
            return tuple(sorted(resolved))

        calls = resolve(call_indices, donor_calls, "call_indices")
        if block_indices != "all":
            try:
                requested_blocks = tuple(block_indices)
            except TypeError as error:
                raise ValueError(
                    "continuous mediation block_indices must be a non-empty iterable"
                ) from error
            if any(block not in self.injection_blocks for block in requested_blocks):
                raise ValueError(
                    "continuous mediation block_indices are outside the configured route"
                )
        blocks = resolve(block_indices, donor_blocks, "block_indices")
        if any(block not in self.injection_blocks for block in blocks):
            raise ValueError("continuous mediation block_indices are outside the configured route")
        keys = {(call, block) for call in calls for block in blocks}
        if not keys.issubset(donor):
            missing = sorted(keys - set(donor))
            raise ValueError(
                "continuous mediation donor is missing requested plan keys: "
                f"{missing}"
            )
        for key in keys:
            state = donor[key]
            if (
                target_object < 0
                or source_object < 0
                or target_object >= state.shape[2]
                or source_object >= state.shape[2]
            ):
                raise ValueError("continuous mediation object id is outside donor slots")
        if mode == "target_swap" and source_object != target_object:
            raise ValueError("target_swap requires source_object == target_object")
        if mode == "wrong_object" and source_object == target_object:
            raise ValueError("wrong_object requires source and target objects to differ")
        self._continuous_state_mediation_plan = {
            key: {
                "call_index": key[0],
                "block_index": key[1],
                "target_object": target_object,
                "source_object": source_object,
                "mode": mode,
                "random_seed": random_seed,
            }
            for key in keys
        }
        self._continuous_state_mediation_applied = set()

    def continuous_state_mediation_plan_status(self) -> dict:
        """Return frozen/applied receipt counts for end-of-generation checks."""
        expected = (
            set()
            if self._continuous_state_mediation_plan is None
            else set(self._continuous_state_mediation_plan)
        )
        applied = set(self._continuous_state_mediation_applied)
        return {
            "expected_count": len(expected),
            "applied_count": len(applied),
            "complete": applied == expected,
            "expected_keys": tuple(sorted(expected)),
            "applied_keys": tuple(sorted(applied)),
            "missing_keys": tuple(sorted(expected - applied)),
        }

    def _continuous_state_mediation_for(
        self, block_index: int
    ) -> Optional[dict]:
        if not self._continuous_state_mediation_enabled:
            return None
        result = {
            "call_index": self._continuous_state_call_index,
            "block_index": int(block_index),
        }
        key = (self._continuous_state_call_index, int(block_index))
        plan = self._continuous_state_mediation_plan
        intervention = None if plan is None else plan.get(key)
        if intervention is None:
            return result
        if key in self._continuous_state_mediation_applied:
            raise RuntimeError("continuous mediation plan point was applied more than once")
        donor = self._continuous_state_mediation_donor
        if donor is None:
            raise RuntimeError("continuous mediation intervention has no bound donor")
        donor_state = donor.get(key)
        if donor_state is None:
            raise RuntimeError("continuous mediation donor receipt disappeared")
        return {**result, **intervention, "donor_state": donor_state}

    def _mark_continuous_state_mediation_applied(self, receipt: dict) -> None:
        mediation = receipt.get("mediation")
        if mediation is None:
            return
        key = (receipt["call_index"], receipt["block_index"])
        if self._continuous_state_mediation_plan is None or key not in self._continuous_state_mediation_plan:
            raise RuntimeError("continuous mediation applied outside frozen plan")
        if key in self._continuous_state_mediation_applied:
            raise RuntimeError("continuous mediation plan point was applied more than once")
        self._continuous_state_mediation_applied.add(key)

    def configure_diagnostic_spatial_oracle(
        self,
        mode: str = "predicted",
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        """Apply one diagnostic read/write intervention to every configured group."""
        for group in self.groups.values():
            group.configure_diagnostic_spatial_oracle(mode, labels)

    def forward(
        self,
        block_index: int,
        hidden: torch.Tensor,
        condition: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        frames: int,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = str(block_index)
        if key not in self.groups:
            raise KeyError(block_index)
        if self.continuous_object_state:
            if self._next_continuous_stage >= len(self.injection_blocks):
                raise RuntimeError(
                    "continuous object stages already completed; reset before a new DiT call"
                )
            expected_block = self.injection_blocks[self._next_continuous_stage]
            if int(block_index) != expected_block:
                raise RuntimeError(
                    "continuous object stages cannot be skipped or reordered: "
                    f"expected block {expected_block}, got {block_index}"
                )
        if self.independent_edge:
            if condition.get("contract") != INDEPENDENT_EDGE_CONDITION_CONTRACT:
                raise ValueError(
                    "independent-edge condition contract differs from "
                    f"{INDEPENDENT_EDGE_CONDITION_CONTRACT!r}"
                )
            forward_keys = TEMPORAL_INDEPENDENT_EDGE_SCHEDULE_FORWARD_CONDITION_KEYS
        elif self.pair_physics:
            forward_keys = TEMPORAL_PAIR_SCHEDULE_FORWARD_CONDITION_KEYS
        elif self.platform_support:
            forward_keys = TEMPORAL_NO_INTERACTION_SCHEDULE_FORWARD_CONDITION_KEYS
        elif self.force_schedule:
            forward_keys = TEMPORAL_SCHEDULE_FORWARD_CONDITION_KEYS
        else:
            forward_keys = TEMPORAL_FORWARD_CONDITION_KEYS
        mass_enabled = os.environ.get("PHYSICAL_WM_OBJECT_MASS", "0") == "1"
        if mass_enabled:
            forward_keys = forward_keys + ("mass", "mass_present")
        missing = set(forward_keys) - set(condition)
        if missing:
            raise ValueError(f"sparse object condition missing {sorted(missing)}")

        parameter = next(self.parameters())
        hidden_fp32 = hidden.to(dtype=parameter.dtype)
        batch = hidden.shape[0]
        unbatched_ndims = {
            "first_frame_masks": 3,
            "platform_mask": 2,
            "object_valid_mask": 1,
            "force": 2,
            "force_present": 1,
            "force_schedule": 2,
            "gravity": 2,
            "gravity_present": 1,
            "mu": 1,
            "mu_present": 1,
            "restitution": 1,
            "restitution_present": 1,
            "edge_mu": 2,
            "edge_mu_present": 2,
            "edge_restitution": 2,
            "edge_restitution_present": 2,
        }
        if mass_enabled:
            unbatched_ndims.update(mass=1, mass_present=1)
        tensors = {
            name: self._batched(
                condition[name], batch, name, unbatched_ndims[name]
            ).to(hidden.device)
            for name in forward_keys
        }
        first_masks = tensors["first_frame_masks"].to(dtype=hidden_fp32.dtype)
        platform_mask = (
            tensors["platform_mask"].to(dtype=hidden_fp32.dtype)
            if self.pair_physics or self.platform_support or self.independent_edge
            else None
        )
        if first_masks.ndim != 4:
            raise ValueError("first_frame_masks must be [B,N,Hc,Wc]")
        objects = first_masks.shape[1]
        expected = {
            "object_valid_mask": (batch, objects),
            "force": (batch, objects, 3),
            "force_present": (batch, objects),
            **(
                {"force_schedule": (batch, frames, objects)}
                if self.force_schedule
                else {}
            ),
            "gravity": (batch, objects, 3),
            "gravity_present": (batch, objects),
            "mu": (batch, objects),
            "mu_present": (batch, objects),
            "restitution": (batch, objects),
            "restitution_present": (batch, objects),
            **(
                {
                    "edge_mu": (batch, objects, objects + 1),
                    "edge_mu_present": (batch, objects, objects + 1),
                    "edge_restitution": (batch, objects, objects + 1),
                    "edge_restitution_present": (batch, objects, objects + 1),
                }
                if self.independent_edge
                else {}
            ),
            **(
                {"platform_mask": (batch,) + tuple(first_masks.shape[-2:])}
                if self.pair_physics or self.platform_support or self.independent_edge
                else {}
            ),
        }
        if mass_enabled:
            expected.update(mass=(batch, objects), mass_present=(batch, objects))
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != shape:
                raise ValueError(
                    f"{name} must be {shape}, got {tuple(tensors[name].shape)}"
                )
        if not all(torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("sparse object forward inputs must be finite")
        object_valid = tensors["object_valid_mask"] > 0.5
        presence = {
            name: tensors[name].to(dtype=hidden_fp32.dtype)
            for name in (
                "force_present",
                "gravity_present",
                "mu_present",
                "restitution_present",
            )
        }
        for name, value in presence.items():
            if torch.any((value < 0) | (value > 1)):
                raise ValueError(f"{name} must be in [0,1]")
            if torch.any((value > 0.5) & ~object_valid):
                raise ValueError(f"{name} cannot enable a padded object")
        edge_values = None
        if self.independent_edge:
            edge_values = {
                name: tensors[name].to(dtype=hidden_fp32.dtype)
                for name in (
                    "edge_mu",
                    "edge_mu_present",
                    "edge_restitution",
                    "edge_restitution_present",
                )
            }
            partner_valid = torch.cat(
                (
                    object_valid,
                    torch.ones(batch, 1, dtype=torch.bool, device=hidden.device),
                ),
                dim=1,
            )
            edge_valid = object_valid[:, :, None] & partner_valid[:, None, :]
            diagonal = torch.eye(
                objects, objects + 1, dtype=torch.bool, device=hidden.device
            )[None]
            edge_valid = edge_valid & ~diagonal
            for name in ("edge_mu_present", "edge_restitution_present"):
                value = edge_values[name]
                if torch.any((value < 0) | (value > 1)):
                    raise ValueError(f"{name} must be in [0,1]")
                if torch.any((value > 0.5) & ~edge_valid):
                    raise ValueError(f"{name} cannot enable invalid or self edges")
            for name in ("edge_mu", "edge_restitution"):
                value = edge_values[name]
                if torch.any(value.masked_select(~edge_valid) != 0):
                    raise ValueError(f"{name} must be zero on invalid or self edges")
            if torch.any((edge_values["edge_mu"] < 0) | (edge_values["edge_mu"] > 2)):
                raise ValueError("edge_mu must be in [0,2]")
            if torch.any(
                (edge_values["edge_restitution"] < 0)
                | (edge_values["edge_restitution"] > 1)
            ):
                raise ValueError("edge_restitution must be in [0,1]")
            for name, value in edge_values.items():
                object_block = value[:, :, :objects]
                if not torch.allclose(object_block, object_block.transpose(1, 2)):
                    raise ValueError(f"{name} object-object block must be symmetric")
        force_input = tensors["force"]
        force = force_input.to(dtype=hidden_fp32.dtype)
        force_schedule = (
            tensors["force_schedule"].to(dtype=hidden_fp32.dtype)
            if self.force_schedule
            else None
        )
        gravity = tensors["gravity"].to(dtype=hidden_fp32.dtype)
        mu = tensors["mu"].to(dtype=hidden_fp32.dtype)
        restitution = tensors["restitution"].to(dtype=hidden_fp32.dtype)
        if torch.any((first_masks < 0) | (first_masks > 1)):
            raise ValueError("first_frame_masks must be in [0,1]")
        if platform_mask is not None:
            if torch.any((platform_mask < 0) | (platform_mask > 1)):
                raise ValueError("platform_mask must be in [0,1]")
            object_union = first_masks.amax(dim=1)
            if torch.any(platform_mask * object_union > 1e-4):
                raise ValueError("platform_mask must exclude every ball pixel")
        # The mixed training sources intentionally use source-local
        # normalization.  Only the universal nonnegative-magnitude invariant
        # belongs here; source-specific upper bounds belong in data audits.
        if torch.any(force[..., 0] < 0):
            raise ValueError("normalized force magnitude must be nonnegative")
        if force_schedule is not None:
            if torch.any((force_schedule < 0) | (force_schedule > 1)):
                raise ValueError("force_schedule must be in [0,1]")
            if torch.any(
                force_schedule
                * (~object_valid[:, None, :]).to(force_schedule.dtype)
                > 0
            ):
                raise ValueError("force_schedule cannot enable a padded object")
        active_force = (presence["force_present"] > 0.5) & (force[..., 0] > 0)
        if active_force.any():
            norms = force[..., 1:][active_force].norm(dim=-1)
            if torch.any((norms < 0.9) | (norms > 1.1)):
                raise ValueError("active force direction norm must be near one")
        if torch.any((gravity[..., 0] < 0) | (gravity[..., 0] > 1)):
            raise ValueError("normalized gravity magnitude must be in [0,1]")
        present_gravity = presence["gravity_present"] > 0.5
        if present_gravity.any():
            norms = gravity[..., 1:][present_gravity].norm(dim=-1)
            # A zero-magnitude sweep may retain either a unit direction or an
            # all-zero placeholder.  Any other norm is a codec error (notably
            # a raw Cartesian value such as [0, 0, -9.8]).
            valid_direction = (norms <= 1.0e-6) | (
                (norms >= 0.9) & (norms <= 1.1)
            )
            if torch.any(~valid_direction):
                raise ValueError(
                    "present gravity direction norm must be zero or near one"
                )
        if torch.any((mu < 0) | (mu > 2)):
            raise ValueError("mu must be in [0,2]")
        if torch.any((restitution < 0) | (restitution > 1)):
            raise ValueError("restitution must be in [0,1]")

        group = self.groups[key]
        state_mediation = self._continuous_state_mediation_for(block_index)
        if self.disable_gravity:
            gravity = torch.zeros_like(gravity)
            presence["gravity_present"] = torch.zeros_like(
                presence["gravity_present"]
            )
        mass_kwargs = {}
        if mass_enabled:
            mass = tensors["mass"].to(dtype=hidden_fp32.dtype)
            mass_present = tensors["mass_present"].to(dtype=hidden_fp32.dtype)
            if not torch.isfinite(mass).all() or torch.any((mass < 0) | (mass > 1)):
                raise ValueError("mass must be finite and normalized to [0,1]")
            if not torch.isfinite(mass_present).all() or torch.any((mass_present != 0) & (mass_present != 1)):
                raise ValueError("mass_present must be binary")
            if torch.any((mass_present > 0) & ~object_valid):
                raise ValueError("mass_present cannot enable padded objects")
            mass_kwargs = dict(mass=mass, mass_present=mass_present)
        unary_temporal_residual = group.condition_encoder(
            force,
            presence["force_present"],
            gravity,
            presence["gravity_present"],
            object_valid.to(dtype=hidden_fp32.dtype),
            frames,
            force_schedule=force_schedule,
            **mass_kwargs,
        )
        hidden_fp32, logits = group(
            hidden_fp32,
            first_masks,
            object_valid,
            unary_temporal_residual,
            platform_mask,
            mu,
            presence["mu_present"],
            restitution,
            presence["restitution_present"],
            timestep,
            frames,
            height,
            width,
            previous_object_state=(
                self._continuous_state
                if self.continuous_object_state and self.state_carry_enabled
                else None
            ),
            previous_route_indices=(
                self._continuous_route_indices
                if self.correctable_route_carry and self.reader_route_carry_enabled
                else None
            ),
            previous_route_attention=(
                self._continuous_route_attention
                if self.correctable_route_carry and self.reader_route_carry_enabled
                else None
            ),
            state_mediation=state_mediation,
            edge_mu=(edge_values["edge_mu"] if edge_values is not None else None),
            edge_mu_present=(
                edge_values["edge_mu_present"] if edge_values is not None else None
            ),
            edge_restitution=(
                edge_values["edge_restitution"] if edge_values is not None else None
            ),
            edge_restitution_present=(
                edge_values["edge_restitution_present"]
                if edge_values is not None
                else None
            ),
        )
        if state_mediation is not None:
            receipt = group.last_continuous_state_mediation_trace
            if receipt is None:
                raise RuntimeError("continuous mediation group produced no receipt")
            self.continuous_state_mediation_trace.append(receipt)
            self._mark_continuous_state_mediation_applied(receipt)
        if self.continuous_object_state:
            if group.last_updated_objects is None:
                raise RuntimeError("continuous group produced no updated object state")
            self._continuous_state = group.last_updated_objects
            self._next_continuous_stage += 1
        if self.correctable_route_carry:
            if group.last_reader_indices is None or group.last_reader_attention is None:
                raise RuntimeError("route-carry group produced no Reader route")
            self._continuous_route_indices = group.last_reader_indices
            self._continuous_route_attention = group.last_reader_attention
        return hidden_fp32.to(dtype=hidden.dtype), logits
