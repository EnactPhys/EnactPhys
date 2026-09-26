"""Paper-faithful PhyParam control branch with explicit ambiguity boundaries.

This module implements only claims that are recoverable from arXiv:2607.18924:

* a trainable Control-DiT branch parallel to a frozen Wan backbone;
* first-frame object masks serialized beside noisy video tokens;
* five local physical tokens per object (force direction, force magnitude,
  mass, friction, restitution) plus one global gravity token;
* shared spatial RoPE for mask and first-frame video tokens;
* restricted physical cross-attention that prevents object-to-object
  attribute leakage; and
* dense, zero-initialized residuals injected into the frozen backbone.

The paper does not publish code or enough detail to identify layer count,
normalization ranges, residual scaling, or the exact gravity tokenization.
Those choices are therefore constructor arguments and are documented in the
adjacent AMBIGUITY_LOG.md rather than being presented as official details.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


LOCAL_TOKEN_NAMES = (
    "force_direction",
    "force_magnitude",
    "mass",
    "friction",
    "restitution",
)
LOCAL_TOKENS_PER_OBJECT = len(LOCAL_TOKEN_NAMES)
MAX_NORMALIZED_FORCE = 360.0 / 340.0
MAX_SERIALIZED_NORMALIZED_FORCE = float(
    torch.tensor(MAX_NORMALIZED_FORCE, dtype=torch.bfloat16).float()
)
MAX_FRICTION = 2.0
REQUIRED_CONDITION_KEYS = (
    "first_frame_masks",
    "object_valid_mask",
    "force",
    "force_present",
    "gravity",
    "gravity_present",
    "mu",
    "mu_present",
    "restitution",
    "restitution_present",
)


def harmonic_embedding(values: torch.Tensor, bands: int) -> torch.Tensor:
    """Dyadic harmonic embedding from the paper's scalar encoding equation."""
    if bands <= 0:
        raise ValueError(f"harmonic bands must be positive, got {bands}")
    frequencies = torch.pow(
        2.0,
        torch.arange(bands, dtype=torch.float64, device=values.device),
    ) * math.pi
    phase = values.to(torch.float64).unsqueeze(-1) * frequencies
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1).flatten(-2).to(values.dtype)


def _rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Wan-compatible rotary application without importing wan_video_dit."""
    x = x.reshape(x.shape[0], x.shape[1], num_heads, x.shape[2] // num_heads)
    complex_x = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    complex_freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    return torch.view_as_real(complex_x * complex_freqs).flatten(2).to(x.dtype)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def _as_batched(
    value: torch.Tensor,
    *,
    batched_ndim: int,
    name: str,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    value = value.to(device=device)
    if value.ndim == batched_ndim - 1:
        value = value.unsqueeze(0)
    if value.ndim != batched_ndim:
        raise ValueError(
            f"{name} must have {batched_ndim - 1} or {batched_ndim} dimensions, "
            f"got {tuple(value.shape)}"
        )
    if value.shape[0] == 1 and batch > 1:
        value = value.expand(batch, *value.shape[1:])
    if value.shape[0] != batch:
        raise ValueError(
            f"{name} batch {value.shape[0]} does not match denoiser batch {batch}"
        )
    return value


class RoutedSelfAttention(nn.Module):
    """Wan self-attention weights with key padding support for object padding."""

    def __init__(self, base_attention: nn.Module):
        super().__init__()
        self.num_heads = int(base_attention.num_heads)
        self.q = copy.deepcopy(base_attention.q)
        self.k = copy.deepcopy(base_attention.k)
        self.v = copy.deepcopy(base_attention.v)
        self.o = copy.deepcopy(base_attention.o)
        self.norm_q = copy.deepcopy(base_attention.norm_q)
        self.norm_k = copy.deepcopy(base_attention.norm_k)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> torch.Tensor:
        q = _rope_apply(self.norm_q(self.q(x)), freqs, self.num_heads)
        k = _rope_apply(self.norm_k(self.k(x)), freqs, self.num_heads)
        v = self.v(x)
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        safe_valid = token_valid.clone()
        empty = ~safe_valid.any(dim=1)
        safe_valid[empty, 0] = True
        bias = torch.zeros(
            x.shape[0], 1, 1, x.shape[1], dtype=q.dtype, device=x.device
        ).masked_fill(~safe_valid[:, None, None, :], float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.o(out) * token_valid.to(out.dtype).unsqueeze(-1)


class RestrictedPhysicalCrossAttention(nn.Module):
    """Cross-attention with an explicit per-query physical routing matrix."""

    def __init__(self, base_attention: nn.Module):
        super().__init__()
        self.num_heads = int(base_attention.num_heads)
        self.has_image_input = bool(
            getattr(base_attention, "has_image_input", False)
        )
        self.image_token_count = 257 if self.has_image_input else 0
        self.q = copy.deepcopy(base_attention.q)
        self.k = copy.deepcopy(base_attention.k)
        self.v = copy.deepcopy(base_attention.v)
        self.o = copy.deepcopy(base_attention.o)
        self.norm_q = copy.deepcopy(base_attention.norm_q)
        self.norm_k = copy.deepcopy(base_attention.norm_k)
        if self.has_image_input:
            self.k_img = copy.deepcopy(base_attention.k_img)
            self.v_img = copy.deepcopy(base_attention.v_img)
            self.norm_k_img = copy.deepcopy(base_attention.norm_k_img)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        allowed: torch.Tensor | None,
    ) -> torch.Tensor:
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        bias = None
        if allowed is not None:
            safe_allowed = allowed.clone()
            invalid_rows = ~safe_allowed.any(dim=-1)
            safe_allowed[:, :, 0] |= invalid_rows
            bias = torch.zeros(
                q.shape[0], 1, q.shape[2], k.shape[2], dtype=q.dtype, device=q.device
            ).masked_fill(~safe_allowed[:, None], float("-inf"))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        allowed: torch.Tensor,
        query_valid: torch.Tensor,
    ) -> torch.Tensor:
        if allowed.shape != (queries.shape[0], queries.shape[1], context.shape[1]):
            raise ValueError(
                "physical routing mask shape mismatch: "
                f"got {tuple(allowed.shape)}, expected "
                f"{(queries.shape[0], queries.shape[1], context.shape[1])}"
            )
        q = self.norm_q(self.q(queries))
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        row_valid = allowed.any(dim=-1) & query_valid
        image_count = min(self.image_token_count, context.shape[1])
        main_context = context[:, image_count:]
        main_allowed = allowed[:, :, image_count:]
        k = self.norm_k(self.k(main_context))
        v = self.v(main_context)
        out = self._attention(q, k, v, main_allowed)
        if image_count:
            image_context = context[:, :image_count]
            image_allowed = allowed[:, :, :image_count]
            image_k = self.norm_k_img(self.k_img(image_context))
            image_v = self.v_img(image_context)
            out = out + self._attention(q, image_k, image_v, image_allowed)
        out = out.permute(0, 2, 1, 3).reshape(
            queries.shape[0], queries.shape[1], -1
        )
        return self.o(out) * row_valid.to(out.dtype).unsqueeze(-1)


class PhyParamControlBlock(nn.Module):
    """One copied Wan block with paper-specified physical attention routing."""

    def __init__(self, base_block: nn.Module):
        super().__init__()
        self.self_attn = RoutedSelfAttention(base_block.self_attn)
        self.cross_attn = RestrictedPhysicalCrossAttention(base_block.cross_attn)
        self.norm1 = copy.deepcopy(base_block.norm1)
        self.norm2 = copy.deepcopy(base_block.norm2)
        self.norm3 = copy.deepcopy(base_block.norm3)
        self.ffn = copy.deepcopy(base_block.ffn)
        self.modulation = nn.Parameter(base_block.modulation.detach().clone())

    def _modulation(self, t_mod: torch.Tensor) -> Sequence[torch.Tensor]:
        has_seq = t_mod.ndim == 4
        chunk_dim = 2 if has_seq else 1
        chunks = (self.modulation.to(t_mod) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            chunks = tuple(value.squeeze(2) for value in chunks)
        return chunks

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        token_valid: torch.Tensor,
        cross_allowed: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._modulation(t_mod)
        self_input = _modulate(self.norm1(hidden), shift_msa, scale_msa)
        hidden = hidden + gate_msa * self.self_attn(self_input, freqs, token_valid)
        hidden = hidden + self.cross_attn(
            self.norm3(hidden), context, cross_allowed, token_valid
        )
        mlp_input = _modulate(self.norm2(hidden), shift_mlp, scale_mlp)
        hidden = hidden + gate_mlp * self.ffn(mlp_input)
        return hidden * token_valid.to(hidden.dtype).unsqueeze(-1)


class PhyParamConditionEncoder(nn.Module):
    """Convert the existing F/g/mu/e + first-mask contract into paper tokens."""

    def __init__(
        self,
        dim: int,
        harmonic_bands: int = 8,
        default_mass_normalized: float = 0.5,
        max_objects: int = 8,
    ):
        super().__init__()
        if not 0.0 <= default_mass_normalized <= 1.0:
            raise ValueError("default normalized mass must be in [0,1]")
        self.dim = int(dim)
        self.harmonic_bands = int(harmonic_bands)
        self.default_mass_normalized = float(default_mass_normalized)
        if max_objects <= 0:
            raise ValueError("PhyParam max_objects must be positive")
        self.max_objects = int(max_objects)
        self.mask_projection = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.direction_projection = nn.Sequential(
            nn.Linear(6 * harmonic_bands, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(2 * harmonic_bands, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.gravity_projection = nn.Sequential(
            nn.Linear(8 * harmonic_bands, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.local_type_embedding = nn.Embedding(LOCAL_TOKENS_PER_OBJECT, dim)
        self.gravity_type_embedding = nn.Parameter(torch.zeros(1, 1, dim))

    def _scalar_token(
        self,
        value: torch.Tensor,
        type_id: int,
        scale: float = 1.0,
    ) -> torch.Tensor:
        embedded = harmonic_embedding(value / float(scale), self.harmonic_bands)
        token = self.scalar_projection(embedded.to(self.scalar_projection[0].weight.dtype))
        type_ids = torch.full(value.shape, type_id, dtype=torch.long, device=value.device)
        return token + self.local_type_embedding(type_ids)

    def forward(
        self,
        condition: Mapping[str, torch.Tensor],
        *,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        missing = set(REQUIRED_CONDITION_KEYS) - set(condition)
        if missing:
            raise ValueError(f"PhyParam condition missing {sorted(missing)}")
        masks = _as_batched(
            condition["first_frame_masks"], batched_ndim=4, name="first_frame_masks",
            batch=batch, device=device,
        ).float()
        valid = _as_batched(
            condition["object_valid_mask"], batched_ndim=2, name="object_valid_mask",
            batch=batch, device=device,
        ).float()
        force = _as_batched(
            condition["force"], batched_ndim=3, name="force", batch=batch, device=device
        ).float()
        force_present = _as_batched(
            condition["force_present"], batched_ndim=2, name="force_present",
            batch=batch, device=device,
        ).float()
        gravity = _as_batched(
            condition["gravity"], batched_ndim=3, name="gravity", batch=batch, device=device
        ).float()
        gravity_present = _as_batched(
            condition["gravity_present"], batched_ndim=2, name="gravity_present",
            batch=batch, device=device,
        ).float()
        mu = _as_batched(
            condition["mu"], batched_ndim=2, name="mu", batch=batch, device=device
        ).float()
        mu_present = _as_batched(
            condition["mu_present"], batched_ndim=2, name="mu_present",
            batch=batch, device=device,
        ).float()
        restitution = _as_batched(
            condition["restitution"], batched_ndim=2, name="restitution",
            batch=batch, device=device,
        ).float()
        restitution_present = _as_batched(
            condition["restitution_present"], batched_ndim=2,
            name="restitution_present", batch=batch, device=device,
        ).float()
        objects = masks.shape[1]
        if objects <= 0 or objects > self.max_objects:
            raise ValueError(
                f"PhyParam object count must be in [1,{self.max_objects}], got {objects}"
            )
        expected_object_shapes = {
            "object_valid_mask": valid.shape[1],
            "force": force.shape[1],
            "force_present": force_present.shape[1],
            "gravity": gravity.shape[1],
            "gravity_present": gravity_present.shape[1],
            "mu": mu.shape[1],
            "mu_present": mu_present.shape[1],
            "restitution": restitution.shape[1],
            "restitution_present": restitution_present.shape[1],
        }
        bad = {name: count for name, count in expected_object_shapes.items() if count != objects}
        if bad:
            raise ValueError(f"PhyParam object cardinality mismatch: masks={objects}, others={bad}")
        tensors = (masks, valid, force, force_present, gravity, gravity_present, mu, mu_present, restitution, restitution_present)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("PhyParam condition contains non-finite values")
        if torch.any((masks < 0) | (masks > 1)):
            raise ValueError("first_frame_masks must be in [0,1]")
        for name, value in (
            ("object_valid_mask", valid), ("force_present", force_present),
            ("gravity_present", gravity_present), ("mu_present", mu_present),
            ("restitution_present", restitution_present),
        ):
            if torch.any((value < 0) | (value > 1)):
                raise ValueError(f"{name} must be in [0,1]")
        if force.shape[-1] != 3 or gravity.shape[-1] != 3:
            raise ValueError("existing-data adapter expects force/gravity vectors with 3 values")

        object_valid = valid > 0.5
        for name, presence in (
            ("force_present", force_present),
            ("gravity_present", gravity_present),
            ("mu_present", mu_present),
            ("restitution_present", restitution_present),
        ):
            if torch.any((presence > 0.5) & ~object_valid):
                raise ValueError(f"{name} cannot enable a padded object")
        padded = ~object_valid
        for name, value in (
            ("first_frame_masks", masks),
            ("force", force),
            ("gravity", gravity),
            ("mu", mu),
            ("restitution", restitution),
        ):
            if padded.any() and torch.count_nonzero(value[padded]).item() != 0:
                raise ValueError(f"padded object has nonzero {name}")
        resized_masks = F.interpolate(
            masks.reshape(batch * objects, 1, *masks.shape[-2:]),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, objects, height * width, 1)
        resized_masks = resized_masks * object_valid[:, :, None, None]
        mask_tokens = self.mask_projection(
            resized_masks.to(self.mask_projection[0].weight.dtype)
        ).to(dtype)

        direction = torch.cat(
            (force[..., 1:3], torch.zeros_like(force[..., :1])), dim=-1
        )
        direction_norm = direction.norm(dim=-1, keepdim=True)
        direction = torch.where(
            direction_norm > 1e-6,
            direction / direction_norm.clamp_min(1e-6),
            direction,
        )
        direction_token = self.direction_projection(
            harmonic_embedding(direction, self.harmonic_bands)
            .flatten(-2)
            .to(self.direction_projection[0].weight.dtype)
        ) + self.local_type_embedding(
            torch.zeros(direction.shape[:-1], dtype=torch.long, device=direction.device)
        )
        mass = condition.get("mass")
        mass_present = condition.get("mass_present")
        if mass is None:
            mass = torch.full_like(mu, self.default_mass_normalized)
            mass_present = torch.ones_like(valid)
        else:
            mass = _as_batched(
                mass, batched_ndim=2, name="mass", batch=batch, device=device
            ).float()
            if mass_present is None:
                mass_present = torch.ones_like(mass)
            else:
                mass_present = _as_batched(
                    mass_present, batched_ndim=2, name="mass_present",
                    batch=batch, device=device,
                ).float()
        for name, value in (
            ("mass", mass),
            ("mass_present", mass_present),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        for name, value in (
            ("mass_present", mass_present),
        ):
            if torch.any((value < 0) | (value > 1)):
                raise ValueError(f"{name} must be in [0,1]")
        scalar_ranges = (
            (
                "force_magnitude",
                force[..., 0],
                MAX_SERIALIZED_NORMALIZED_FORCE,
            ),
            ("gravity_magnitude", gravity[..., 0], 1.0),
            ("mass", mass, 1.0),
            ("mu", mu, MAX_FRICTION),
            ("restitution", restitution, 1.0),
        )
        for name, value, upper in scalar_ranges:
            if torch.any((value < 0) | (value > upper)):
                raise ValueError(f"{name} must be in [0,{upper}]")
        active_force = (force_present > 0.5) & (force[..., 0] > 0)
        if active_force.any():
            norms = force[..., 1:3][active_force].norm(dim=-1)
            if torch.any((norms < 0.9) | (norms > 1.1)):
                raise ValueError("active force direction norm must be near one")
        active_gravity = (gravity_present > 0.5) & (gravity[..., 0] > 0)
        if active_gravity.any():
            norms = gravity[..., 1:3][active_gravity].norm(dim=-1)
            if torch.any((norms < 0.9) | (norms > 1.1)):
                raise ValueError("active gravity direction norm must be near one")
        local_tokens = torch.stack(
            (
                direction_token,
                self._scalar_token(force[..., 0], 1, MAX_NORMALIZED_FORCE),
                self._scalar_token(mass, 2),
                self._scalar_token(mu, 3, MAX_FRICTION),
                self._scalar_token(restitution, 4),
            ),
            dim=2,
        ).to(dtype)
        local_valid = torch.stack(
            (
                force_present,
                force_present,
                mass_present,
                mu_present,
                restitution_present,
            ),
            dim=2,
        ) > 0.5
        local_valid = local_valid & object_valid.unsqueeze(-1)

        # The current dataset stores the scene-global gravity once per object.
        # Collapse only after proving all valid copies agree, so data drift fails closed.
        global_gravity = torch.zeros(batch, 3, device=device)
        global_gravity_valid = torch.zeros(batch, dtype=torch.bool, device=device)
        for batch_index in range(batch):
            selected = object_valid[batch_index] & (gravity_present[batch_index] > 0.5)
            if selected.any():
                copies = gravity[batch_index, selected]
                if not torch.allclose(copies, copies[:1].expand_as(copies), atol=1e-5, rtol=1e-5):
                    raise ValueError("per-object gravity copies disagree; cannot form global PhyParam token")
                global_gravity[batch_index] = copies[0]
                global_gravity_valid[batch_index] = True
        gravity_direction = torch.cat(
            (
                global_gravity[..., 1:3],
                torch.zeros_like(global_gravity[..., :1]),
            ),
            dim=-1,
        )
        gravity_direction_norm = gravity_direction.norm(dim=-1, keepdim=True)
        gravity_direction = torch.where(
            gravity_direction_norm > 1e-6,
            gravity_direction / gravity_direction_norm.clamp_min(1e-6),
            gravity_direction,
        )
        gravity_harmonic = torch.cat(
            (
                harmonic_embedding(gravity_direction, self.harmonic_bands).flatten(-2),
                harmonic_embedding(global_gravity[..., 0], self.harmonic_bands),
            ),
            dim=-1,
        )
        gravity_token = self.gravity_projection(
            gravity_harmonic.to(self.gravity_projection[0].weight.dtype)
        ).unsqueeze(1) + self.gravity_type_embedding
        return mask_tokens, local_tokens, local_valid, gravity_token.to(dtype), global_gravity_valid


class PhyParamFeatureSupervisor(nn.Module):
    """Project multi-depth Control-DiT features into a cached DINO feature space.

    The paper identifies a frozen DINOv3 teacher but does not publish the
    checkpoint, tapped ControlNet depths, or fusion rule.  This implementation
    uses independently projected, equally averaged Control-DiT depths.  DINO
    targets are precomputed and detached; no DINO parameters live in the
    training graph.
    """

    def __init__(
        self,
        dim: int,
        feature_dim: int,
        tap_blocks: Sequence[int],
        target_grid: tuple[int, int] = (14, 14),
        temporal_min_norm: float = 1e-6,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("DINO feature dimension must be positive")
        if not tap_blocks:
            raise ValueError("at least one Control-DiT feature tap is required")
        self.feature_dim = int(feature_dim)
        self.tap_blocks = tuple(int(index) for index in tap_blocks)
        self.target_grid = tuple(int(value) for value in target_grid)
        if len(self.target_grid) != 2 or min(self.target_grid) <= 0:
            raise ValueError("DINO target grid must contain two positive values")
        self.temporal_min_norm = float(temporal_min_norm)
        self.heads = nn.ModuleDict(
            {
                str(index): nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, self.feature_dim),
                )
                for index in self.tap_blocks
            }
        )
        self._features: list[torch.Tensor] = []

    def reset(self) -> None:
        self._features = []

    def capture(
        self,
        block_index: int,
        hidden: torch.Tensor,
        *,
        video_token_count: int,
        frames: int,
        height: int,
        width: int,
    ) -> None:
        key = str(int(block_index))
        if key not in self.heads:
            return
        video = hidden[:, :video_token_count]
        if video.shape[1] != frames * height * width:
            raise ValueError("captured Control-DiT feature has an invalid token count")
        video = video.reshape(video.shape[0], frames, height, width, video.shape[-1])
        if (height, width) != self.target_grid:
            video = F.interpolate(
                video.permute(0, 1, 4, 2, 3).flatten(0, 1),
                size=self.target_grid,
                mode="bilinear",
                align_corners=False,
            ).unflatten(0, (hidden.shape[0], frames)).permute(0, 1, 3, 4, 2)
        projected = self.heads[key](video)
        self._features.append(
            projected
        )

    def _canonical_target(
        self,
        target: torch.Tensor | Mapping[str, torch.Tensor],
        *,
        batch: int,
        frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        spatial_shape = None
        if isinstance(target, Mapping):
            spatial_shape = target.get("spatial_shape")
            target = target.get("features")
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                "PhyParam DINO target must be a tensor or {'features': tensor}"
            )
        target = target.to(device=device, dtype=torch.float32).detach()
        if target.ndim == 3:
            target = target.unsqueeze(0)
        if target.ndim == 4:
            # [B,T,N,C], unbatched [T,C,H,W], or an explicitly described
            # unbatched [T,H,W,C] mapping.
            if target.shape[1] == self.feature_dim and target.shape[-1] != self.feature_dim:
                target = target.permute(0, 2, 3, 1).unsqueeze(0)
            elif target.shape[-1] == self.feature_dim:
                if (
                    spatial_shape is not None
                    and target.shape[1] == int(spatial_shape[0])
                    and target.shape[2] == int(spatial_shape[1])
                ):
                    target = target.unsqueeze(0)
                    return self._finish_target(
                        target, batch=batch, frames=frames, device=device
                    )
                batch_size, time, patches, channels = target.shape
                if spatial_shape is None:
                    side = math.isqrt(patches)
                    if side * side != patches:
                        raise ValueError(
                            "flattened DINO patches require spatial_shape when not square"
                        )
                    target = target.reshape(batch_size, time, side, side, channels)
                else:
                    target_height, target_width = (
                        int(spatial_shape[0]), int(spatial_shape[1])
                    )
                    if target_height * target_width != patches:
                        raise ValueError("DINO spatial_shape does not match patch count")
                    target = target.reshape(
                        batch_size, time, target_height, target_width, channels
                    )
            else:
                raise ValueError("cannot infer four-dimensional DINO feature layout")
        elif target.ndim == 5:
            if target.shape[-1] == self.feature_dim:
                pass
            elif target.shape[2] == self.feature_dim:
                target = target.permute(0, 1, 3, 4, 2)
            else:
                raise ValueError("DINO feature dimension does not match configured feature_dim")
        else:
            raise ValueError(
                "DINO features must be [T,N,C], [B,T,N,C], [T,C,H,W], "
                "[B,T,H,W,C], or [B,T,C,H,W]"
            )
        return self._finish_target(target, batch=batch, frames=frames, device=device)

    def _finish_target(
        self,
        target: torch.Tensor,
        *,
        batch: int,
        frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        if target.shape[0] == 1 and batch > 1:
            target = target.expand(batch, *target.shape[1:])
        if target.shape[0] != batch:
            raise ValueError(
                f"DINO target batch {target.shape[0]} does not match model batch {batch}"
            )
        if target.shape[1] != frames:
            indices = torch.linspace(
                0, target.shape[1] - 1, frames, device=device
            ).round().long()
            target = target.index_select(1, indices)
        if not torch.isfinite(target).all():
            raise ValueError("DINO target contains non-finite values")
        return target

    def losses(
        self,
        target: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if len(self._features) != len(self.tap_blocks):
            raise RuntimeError(
                "Control-DiT feature capture is incomplete: "
                f"got {len(self._features)}, expected {len(self.tap_blocks)}"
            )
        prediction = torch.stack(self._features, dim=0).mean(dim=0).float()
        teacher = self._canonical_target(
            target,
            batch=prediction.shape[0],
            frames=prediction.shape[1],
            device=prediction.device,
        )
        if prediction.shape[2:4] != teacher.shape[2:4]:
            batch, frames, height, width, channels = prediction.shape
            prediction = F.interpolate(
                prediction.permute(0, 1, 4, 2, 3).reshape(
                    batch * frames, channels, height, width
                ),
                size=teacher.shape[2:4],
                mode="bilinear",
                align_corners=False,
            ).reshape(
                batch, frames, channels, teacher.shape[2], teacher.shape[3]
            ).permute(0, 1, 3, 4, 2)
        feature_cosine = F.cosine_similarity(
            prediction, teacher, dim=-1, eps=1e-6
        )
        feature_loss = 1.0 - feature_cosine.mean()

        prediction_delta = prediction[:, 1:] - prediction[:, :-1]
        teacher_delta = teacher[:, 1:] - teacher[:, :-1]
        temporal_valid = teacher_delta.norm(dim=-1) > self.temporal_min_norm
        temporal_cosine = F.cosine_similarity(
            prediction_delta, teacher_delta, dim=-1, eps=1e-6
        )
        if temporal_valid.any():
            temporal_loss = 1.0 - temporal_cosine[temporal_valid].mean()
        else:
            temporal_loss = prediction.sum() * 0.0
        metrics = {
            "phyparam_feature_cosine": feature_cosine.mean().detach(),
            "phyparam_temporal_cosine": (
                temporal_cosine[temporal_valid].mean().detach()
                if temporal_valid.any()
                else prediction.new_zeros(())
            ),
            "phyparam_temporal_valid_fraction": temporal_valid.float().mean().detach(),
            "phyparam_feature_taps_captured": prediction.new_tensor(
                float(len(self._features))
            ),
        }
        return feature_loss, temporal_loss, metrics


@dataclass(frozen=True)
class PhyParamControlState:
    hidden: torch.Tensor
    context: torch.Tensor
    freqs: torch.Tensor
    token_valid: torch.Tensor
    cross_allowed: torch.Tensor
    video_token_count: int
    frames: int
    height: int
    width: int


class PhyParamControlDiT(nn.Module):
    """Unified paper-reimplementation branch producing dense Wan residuals."""

    def __init__(
        self,
        base_blocks: Sequence[nn.Module],
        *,
        dim: int,
        num_control_blocks: int | None = None,
        harmonic_bands: int = 8,
        default_mass_normalized: float = 0.5,
        max_objects: int = 8,
        dino_feature_dim: int = 4096,
        feature_tap_blocks: Sequence[int] | None = None,
        enable_feature_supervision: bool = True,
        dino_target_grid: tuple[int, int] = (14, 14),
        temporal_min_norm: float = 1e-6,
    ):
        super().__init__()
        count = len(base_blocks) if num_control_blocks is None else int(num_control_blocks)
        if count <= 0 or count > len(base_blocks):
            raise ValueError(
                f"num_control_blocks must be in [1,{len(base_blocks)}], got {count}"
            )
        self.num_control_blocks = count
        if feature_tap_blocks is None:
            feature_tap_blocks = tuple(
                sorted(
                    {
                        max(0, math.ceil(count * fraction / 3) - 1)
                        for fraction in (1, 2, 3)
                    }
                )
            )
        feature_tap_blocks = tuple(int(index) for index in feature_tap_blocks)
        if any(index < 0 or index >= count for index in feature_tap_blocks):
            raise ValueError(
                f"feature taps {feature_tap_blocks} must be within [0,{count - 1}]"
            )
        self.condition_encoder = PhyParamConditionEncoder(
            dim=dim,
            harmonic_bands=harmonic_bands,
            default_mass_normalized=default_mass_normalized,
            max_objects=max_objects,
        )
        self.blocks = nn.ModuleList(
            [PhyParamControlBlock(base_blocks[index]) for index in range(count)]
        )
        self.output_projections = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(count)]
        )
        for projection in self.output_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.feature_supervision_enabled = bool(enable_feature_supervision)
        self.feature_supervisor = (
            PhyParamFeatureSupervisor(
                dim=dim,
                feature_dim=dino_feature_dim,
                tap_blocks=feature_tap_blocks,
                target_grid=dino_target_grid,
                temporal_min_norm=temporal_min_norm,
            )
            if self.feature_supervision_enabled
            else None
        )
        self._initialize_from_scratch()

    def _initialize_from_scratch(self) -> None:
        """Discard copied Wan values while retaining the exact block architecture.

        Copying constructs a shape-compatible Control-DiT, but the formal run
        contract requires every trainable value to start independently of the
        frozen backbone.  Standard PyTorch modules use their native reset;
        Wan modulation follows its published constructor distribution.  Dense
        residual projections remain zero so step zero equals frozen Wan.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                module.reset_parameters()
            elif module.__class__.__name__ == "RMSNorm" and hasattr(module, "weight"):
                nn.init.ones_(module.weight)
        for block in self.blocks:
            nn.init.normal_(
                block.modulation,
                mean=0.0,
                std=block.modulation.shape[-1] ** -0.5,
            )
        for projection in self.output_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def prepare(
        self,
        video_hidden: torch.Tensor,
        text_context: torch.Tensor,
        condition: Mapping[str, torch.Tensor],
        freqs: torch.Tensor,
        *,
        frames: int,
        height: int,
        width: int,
    ) -> PhyParamControlState:
        if self.feature_supervisor is not None:
            self.feature_supervisor.reset()
        batch, video_tokens, _ = video_hidden.shape
        if video_tokens != frames * height * width:
            raise ValueError(
                f"video token count {video_tokens} != frames*height*width "
                f"({frames}*{height}*{width})"
            )
        mask_tokens, local_tokens, local_valid, gravity_token, gravity_valid = (
            self.condition_encoder(
                condition,
                batch=batch,
                height=height,
                width=width,
                device=video_hidden.device,
                dtype=video_hidden.dtype,
            )
        )
        objects = mask_tokens.shape[1]
        spatial = height * width
        mask_sequence = mask_tokens.reshape(batch, objects * spatial, -1)
        hidden = torch.cat((video_hidden, mask_sequence), dim=1)

        local_sequence = local_tokens.reshape(batch, objects * LOCAL_TOKENS_PER_OBJECT, -1)
        context = torch.cat((text_context, local_sequence, gravity_token), dim=1)
        text_tokens = text_context.shape[1]
        gravity_index = text_tokens + objects * LOCAL_TOKENS_PER_OBJECT
        object_valid = local_valid.any(dim=-1)
        token_valid = torch.cat(
            (
                torch.ones(batch, video_tokens, dtype=torch.bool, device=video_hidden.device),
                object_valid[:, :, None].expand(-1, -1, spatial).reshape(batch, -1),
            ),
            dim=1,
        )
        context_valid = torch.cat(
            (
                torch.ones(batch, text_tokens, dtype=torch.bool, device=video_hidden.device),
                local_valid.reshape(batch, -1),
                gravity_valid[:, None],
            ),
            dim=1,
        )
        allowed = torch.zeros(
            batch, hidden.shape[1], context.shape[1], dtype=torch.bool,
            device=video_hidden.device,
        )
        allowed[:, :video_tokens, :text_tokens] = True
        allowed[:, :video_tokens, gravity_index] = gravity_valid[:, None]
        for object_index in range(objects):
            query_start = video_tokens + object_index * spatial
            query_end = query_start + spatial
            attr_start = text_tokens + object_index * LOCAL_TOKENS_PER_OBJECT
            attr_end = attr_start + LOCAL_TOKENS_PER_OBJECT
            allowed[:, query_start:query_end, :text_tokens] = True
            allowed[:, query_start:query_end, attr_start:attr_end] = local_valid[
                :, object_index, None, :
            ]
        allowed &= context_valid[:, None, :]
        allowed &= token_valid[:, :, None]

        first_frame_freqs = freqs[:spatial]
        mask_freqs = first_frame_freqs.repeat(objects, 1, 1)
        combined_freqs = torch.cat((freqs, mask_freqs), dim=0)
        return PhyParamControlState(
            hidden=hidden,
            context=context,
            freqs=combined_freqs,
            token_valid=token_valid,
            cross_allowed=allowed,
            video_token_count=video_tokens,
            frames=frames,
            height=height,
            width=width,
        )

    def capture_feature(
        self,
        block_index: int,
        hidden: torch.Tensor,
        state: PhyParamControlState,
    ) -> None:
        if self.feature_supervisor is None:
            return
        self.feature_supervisor.capture(
            block_index,
            hidden,
            video_token_count=state.video_token_count,
            frames=state.frames,
            height=state.height,
            width=state.width,
        )

    def feature_losses(
        self,
        target: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.feature_supervisor is None:
            raise RuntimeError("PhyParam feature supervision is disabled")
        return self.feature_supervisor.losses(target)

    def forward_block(
        self,
        block_index: int,
        state: PhyParamControlState,
        t_mod: torch.Tensor,
    ) -> tuple[PhyParamControlState, torch.Tensor | None]:
        if block_index >= self.num_control_blocks:
            return state, None
        hidden = self.blocks[block_index](
            state.hidden,
            state.context,
            t_mod,
            state.freqs,
            state.token_valid,
            state.cross_allowed,
        )
        self.capture_feature(block_index, hidden, state)
        dense_residual = self.output_projections[block_index](
            hidden[:, : state.video_token_count]
        )
        return (
            PhyParamControlState(
                hidden=hidden,
                context=state.context,
                freqs=state.freqs,
                token_valid=state.token_valid,
                cross_allowed=state.cross_allowed,
                video_token_count=state.video_token_count,
                frames=state.frames,
                height=state.height,
                width=state.width,
            ),
            dense_residual,
        )
