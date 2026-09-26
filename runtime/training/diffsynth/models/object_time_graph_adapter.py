"""Learned object-time graph adapter for F/mu/e-conditioned video generation.

The forward path is intentionally restricted to first-frame object masks and
the F/mu/e controls. Future object masks and contact labels are training
targets owned by the loss; they are never read by this module.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F


FORWARD_CONDITION_KEYS = (
    "first_frame_masks",
    "force",
    "force_target",
    "mu",
    "restitution",
)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _zero_last_linear(module: nn.Sequential) -> None:
    layer = module[-1]
    if not isinstance(layer, nn.Linear):
        raise TypeError("expected an MLP ending in nn.Linear")
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class PhysicsConditionEncoder(nn.Module):
    """Shared raw-value encoders; no condition tokens or Fourier features."""

    def __init__(self, object_dim: int):
        super().__init__()
        self.force_magnitude = _mlp(1, object_dim, object_dim)
        self.force_direction = _mlp(2, object_dim, object_dim)
        self.force_fusion = _mlp(object_dim * 2, object_dim, object_dim)
        self.mu = _mlp(1, object_dim, object_dim)
        self.restitution = _mlp(1, object_dim, object_dim)

    def forward(
        self,
        force: torch.Tensor,
        mu: torch.Tensor,
        restitution: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if force.ndim != 2 or force.shape[-1] != 3:
            raise ValueError(f"force must be [B,3], got {tuple(force.shape)}")
        if mu.ndim != 2 or mu.shape[-1] != 2:
            raise ValueError(f"mu must contain mu_A/mu_B as [B,2], got {tuple(mu.shape)}")
        if restitution.ndim != 2 or restitution.shape[-1] != 1:
            raise ValueError(
                f"restitution must be [B,1], got {tuple(restitution.shape)}"
            )
        magnitude = force[:, :1]
        force_feature = self.force_fusion(
            torch.cat(
                (
                    self.force_magnitude(magnitude),
                    self.force_direction(force[:, 1:]),
                ),
                dim=-1,
            )
        )
        return (
            force_feature,
            self.mu(mu[..., None]),
            self.restitution(restitution),
        )


class LearnedObjectRouter(nn.Module):
    """Use first-frame anchors to find both objects at every latent time."""

    def __init__(
        self,
        wan_dim: int,
        object_dim: int,
        num_heads: int,
        max_times: int = 13,
        num_objects: int = 2,
    ):
        super().__init__()
        if object_dim % num_heads:
            raise ValueError("object_dim must be divisible by num_heads")
        self.object_dim = object_dim
        self.num_heads = num_heads
        self.head_dim = object_dim // num_heads
        self.num_objects = num_objects
        self.input_norm = nn.LayerNorm(wan_dim)
        self.input_projection = nn.Linear(wan_dim, object_dim)
        self.object_embedding = nn.Embedding(num_objects, object_dim)
        self.time_embedding = nn.Embedding(max_times, object_dim)
        self.query_norm = nn.LayerNorm(object_dim)
        self.key_norm = nn.LayerNorm(object_dim)
        self.query_projection = nn.Linear(object_dim, object_dim)
        self.key_projection = nn.Linear(object_dim, object_dim)
        self.value_projection = nn.Linear(object_dim, object_dim)
        self.read_projection = nn.Linear(object_dim, object_dim)
        # A sigmoid probability near zero is safer than writing 0.5 everywhere
        # before localization has learned. The constant does not affect the
        # spatial softmax used for reading.
        self.location_bias = nn.Parameter(torch.tensor(-4.0))

    def forward(
        self,
        hidden: torch.Tensor,
        first_frame_masks: torch.Tensor,
        frames: int,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = hidden.shape[0]
        if hidden.shape[1] != frames * height * width:
            raise ValueError(
                "object router requires plain Wan video tokens without "
                "prepended reference tokens"
            )
        expected_masks = (batch, self.num_objects, height, width)
        if tuple(first_frame_masks.shape) != expected_masks:
            raise ValueError(
                f"first_frame_masks must be {expected_masks}, got "
                f"{tuple(first_frame_masks.shape)}"
            )
        if frames > self.time_embedding.num_embeddings:
            raise ValueError(f"too many latent times: {frames}")

        grid = self.input_projection(self.input_norm(hidden)).reshape(
            batch, frames, height * width, self.object_dim
        )
        masks = first_frame_masks.flatten(-2)
        mass = masks.sum(dim=-1, keepdim=True)
        if torch.any(mass <= 1e-6):
            raise ValueError("every first-frame object mask must contain visible pixels")
        anchors = torch.einsum(
            "bog,bgd->bod",
            masks / mass,
            grid[:, 0],
        )
        object_ids = torch.arange(self.num_objects, device=hidden.device)
        time_ids = torch.arange(frames, device=hidden.device)
        queries = (
            anchors[:, None]
            + self.object_embedding(object_ids)[None, None]
            + self.time_embedding(time_ids)[None, :, None]
        )

        query = self.query_projection(self.query_norm(queries)).reshape(
            batch, frames, self.num_objects, self.num_heads, self.head_dim
        )
        key = self.key_projection(self.key_norm(grid)).reshape(
            batch, frames, height * width, self.num_heads, self.head_dim
        )
        value = self.value_projection(grid).reshape(
            batch, frames, height * width, self.num_heads, self.head_dim
        )
        scores = torch.einsum("btohd,btghd->bthog", query, key)
        scores = scores / math.sqrt(self.head_dim)
        read_attention = scores.softmax(dim=-1)
        read = torch.einsum("bthog,btghd->btohd", read_attention, value)
        read = self.read_projection(read.flatten(-2))
        slots = queries + read

        # The same learned compatibility drives localization supervision and
        # local writeback. No future mask enters this calculation.
        location_logits_flat = scores.mean(dim=2) + self.location_bias
        write_attention = torch.sigmoid(location_logits_flat)
        location_logits = location_logits_flat.reshape(
            batch, frames, self.num_objects, height, width
        )
        return slots, location_logits, write_attention


class ObjectTimeGraphBlock(nn.Module):
    """F/mu-modulated temporal update plus contact-gated e relation exchange."""

    def __init__(
        self,
        object_dim: int,
        num_heads: int,
        ffn_dim: int,
        enable_force_kick: bool = False,
    ):
        super().__init__()
        self.object_dim = object_dim
        self.temporal_norm = nn.LayerNorm(object_dim, elementwise_affine=False)
        self.temporal_attention = nn.MultiheadAttention(
            object_dim, num_heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(object_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(object_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, object_dim),
        )

        # mu and target-bound F remain separate until their AdaLN parameters
        # are added, which keeps the routes directly ablatable.
        self.mu_modulation = _mlp(object_dim, object_dim, object_dim * 6)
        self.force_modulation = _mlp(object_dim, object_dim, object_dim * 6)
        _zero_last_linear(self.mu_modulation)
        _zero_last_linear(self.force_modulation)
        self.force_kick = (
            _mlp(object_dim, object_dim, object_dim)
            if enable_force_kick
            else None
        )

        relation_input_dim = object_dim * 4
        self.contact_head = _mlp(relation_input_dim, object_dim, 1)
        nn.init.constant_(self.contact_head[-1].bias, -2.0)
        self.relation_encoder = _mlp(relation_input_dim, object_dim * 2, object_dim)
        self.relation_norm = nn.LayerNorm(object_dim, elementwise_affine=False)
        self.e_modulation = _mlp(object_dim, object_dim, object_dim * 4)
        self.relation_ffn = _mlp(object_dim, object_dim * 2, object_dim)
        self.relation_to_a = nn.Linear(object_dim, object_dim)
        self.relation_to_b = nn.Linear(object_dim, object_dim)

    @staticmethod
    def _relation_features(slots: torch.Tensor) -> torch.Tensor:
        if slots.shape[2] != 2:
            raise ValueError("v05 Object-Time Graph currently requires exactly two objects")
        a, b = slots[:, :, 0], slots[:, :, 1]
        return torch.cat((a, b, a - b, a * b), dim=-1)

    def _temporal_update(
        self,
        slots: torch.Tensor,
        force_feature: torch.Tensor,
        force_magnitude: torch.Tensor,
        mu_features: torch.Tensor,
        force_target: torch.Tensor,
        apply_initial_force_kick: bool = False,
    ) -> torch.Tensor:
        batch, times, objects, dim = slots.shape
        if objects != 2 or dim != self.object_dim:
            raise ValueError(
                f"slots must be [B,T,2,{self.object_dim}], got {tuple(slots.shape)}"
            )
        if tuple(force_target.shape) != (batch, objects):
            raise ValueError(
                f"force_target must be {(batch, objects)}, got "
                f"{tuple(force_target.shape)}"
            )

        target = force_target[..., None]
        if tuple(force_magnitude.shape) != (batch, 1):
            raise ValueError(
                f"force_magnitude must be {(batch, 1)}, got "
                f"{tuple(force_magnitude.shape)}"
            )
        force_gate = target * force_magnitude[:, None]
        force_by_object = force_feature[:, None].expand(-1, objects, -1)
        modulation = self.mu_modulation(mu_features)
        modulation = (
            modulation
            + force_gate * self.force_modulation(force_by_object)
        )
        (
            temporal_shift,
            temporal_scale,
            temporal_gate,
            ffn_shift,
            ffn_scale,
            ffn_gate,
        ) = modulation.chunk(6, dim=-1)

        if apply_initial_force_kick:
            if times < 2:
                raise ValueError("initial force kick requires at least two latent times")
            if self.force_kick is None:
                raise RuntimeError("force kick requested from a block without kick parameters")
            kick = force_gate * self.force_kick(force_by_object)
            slots = slots.clone()
            slots[:, 1] = slots[:, 1] + kick

        # Temporal attention is independent for A and B. mu_A/mu_B act over
        # all latent times; target-only F modulation is exactly zero for the
        # non-target object.
        sequence = slots.permute(0, 2, 1, 3)
        temporal_input = self.temporal_norm(sequence)
        temporal_input = (
            temporal_input * (1.0 + temporal_scale[:, :, None])
            + temporal_shift[:, :, None]
        )
        flat = temporal_input.reshape(batch * objects, times, dim)
        temporal_delta, _ = self.temporal_attention(
            flat, flat, flat, need_weights=False
        )
        temporal_delta = temporal_delta.reshape(batch, objects, times, dim)
        sequence = sequence + torch.tanh(temporal_gate[:, :, None]) * temporal_delta

        ffn_input = self.ffn_norm(sequence)
        ffn_input = (
            ffn_input * (1.0 + ffn_scale[:, :, None])
            + ffn_shift[:, :, None]
        )
        sequence = sequence + torch.tanh(ffn_gate[:, :, None]) * self.ffn(ffn_input)
        return sequence.permute(0, 2, 1, 3)

    def forward(
        self,
        slots: torch.Tensor,
        contact_slots: torch.Tensor,
        force_feature: torch.Tensor,
        force_magnitude: torch.Tensor,
        mu_features: torch.Tensor,
        restitution_feature: torch.Tensor,
        force_target: torch.Tensor,
        apply_initial_force_kick: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = self._temporal_update(
            slots,
            force_feature,
            force_magnitude,
            mu_features,
            force_target,
            apply_initial_force_kick=apply_initial_force_kick,
        )
        # Contact inference owns a parallel e-free state. The generation state
        # may already contain an earlier relation update, but ContactHead never
        # consumes that state, so block 2 cannot use block 1's e as a shortcut.
        contact_slots = self._temporal_update(
            contact_slots,
            force_feature,
            force_magnitude,
            mu_features,
            force_target,
            apply_initial_force_kick=apply_initial_force_kick,
        )

        contact_features = self._relation_features(contact_slots)
        # ContactHead is evaluated before e is introduced. This is the central
        # anti-leakage guarantee for autonomous impact judgment.
        contact_logits = self.contact_head(contact_features)
        contact_probability = torch.sigmoid(contact_logits)

        relation_features = self._relation_features(slots)
        relation = self.relation_encoder(relation_features)
        e_scale, e_shift, e_gate_a, e_gate_b = self.e_modulation(
            restitution_feature
        ).chunk(4, dim=-1)
        relation = self.relation_norm(relation)
        relation = (
            relation * (1.0 + e_scale[:, None])
            + e_shift[:, None]
        )
        message = self.relation_ffn(relation)
        delta_a = (
            contact_probability
            * torch.sigmoid(e_gate_a[:, None])
            * self.relation_to_a(message)
        )
        delta_b = (
            contact_probability
            * torch.sigmoid(e_gate_b[:, None])
            * self.relation_to_b(message)
        )
        slots = slots.clone()
        slots[:, :, 0] = slots[:, :, 0] + delta_a
        slots[:, :, 1] = slots[:, :, 1] + delta_b
        return slots, contact_slots, contact_logits


class ObjectTimeGraphGroup(nn.Module):
    """One Wan insertion group: learned route, two graph blocks, writeback."""

    def __init__(
        self,
        wan_dim: int,
        object_dim: int,
        num_heads: int,
        ffn_dim: int,
        graph_blocks: int,
        enable_initial_force_kick: bool,
    ):
        super().__init__()
        if graph_blocks != 2:
            raise ValueError("v05 freezes exactly two graph blocks per group")
        self.router = LearnedObjectRouter(
            wan_dim=wan_dim,
            object_dim=object_dim,
            num_heads=num_heads,
        )
        self.blocks = nn.ModuleList([
            ObjectTimeGraphBlock(
                object_dim,
                num_heads,
                ffn_dim,
                enable_force_kick=(
                    enable_initial_force_kick and block_id == 0
                ),
            )
            for block_id in range(graph_blocks)
        ])
        self.output_projection = nn.Linear(object_dim, wan_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.noise_gate = _mlp(1, 32, 1)
        self.residual_scale = nn.Parameter(torch.ones(()))

    def forward(
        self,
        hidden: torch.Tensor,
        first_frame_masks: torch.Tensor,
        force_feature: torch.Tensor,
        force_magnitude: torch.Tensor,
        mu_features: torch.Tensor,
        restitution_feature: torch.Tensor,
        force_target: torch.Tensor,
        timestep: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        apply_initial_force_kick: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots, attention_logits, write_attention = self.router(
            hidden, first_frame_masks, frames, height, width
        )
        original_slots = slots
        contact_slots = slots
        contact_logits = []
        for block_id, block in enumerate(self.blocks):
            slots, contact_slots, logits = block(
                slots,
                contact_slots,
                force_feature=force_feature,
                force_magnitude=force_magnitude,
                mu_features=mu_features,
                restitution_feature=restitution_feature,
                force_target=force_target,
                apply_initial_force_kick=(
                    apply_initial_force_kick and block_id == 0
                ),
            )
            contact_logits.append(logits)

        delta_slots = slots - original_slots
        # Normalize only overlapping object responsibilities. Values below a
        # total responsibility of one stay small, preserving background.
        write = write_attention / write_attention.sum(
            dim=2, keepdim=True
        ).clamp_min(1.0)
        delta_grid = torch.einsum("btog,btod->btgd", write, delta_slots)
        # Never alter the clean conditioning frame.
        delta_grid[:, 0] = 0
        projected = self.output_projection(
            delta_grid.reshape(hidden.shape[0], frames * height * width, -1)
        )

        if timestep.numel() % hidden.shape[0] != 0:
            raise ValueError(
                f"timestep elements {timestep.numel()} are not divisible by "
                f"batch {hidden.shape[0]}"
            )
        normalized_timestep = (
            timestep.float().reshape(hidden.shape[0], -1).mean(dim=1, keepdim=True)
            / 1000.0
        ).clamp(0.0, 1.0)
        gate = torch.sigmoid(self.noise_gate(normalized_timestep))[:, None]
        hidden = hidden + (
            gate * self.residual_scale * projected
        ).to(dtype=hidden.dtype)
        return hidden, attention_logits, torch.stack(contact_logits, dim=0)


class ObjectTimeGraphAdapter(nn.Module):
    """Eight learned-routing groups mounted densely in the 30-block Wan."""

    def __init__(
        self,
        wan_dim: int,
        injection_blocks: Iterable[int] = (2, 6, 10, 14, 18, 22, 25, 27),
        graph_blocks_per_group: int = 2,
        object_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
    ):
        super().__init__()
        self.injection_blocks = tuple(int(index) for index in injection_blocks)
        if self.injection_blocks != (2, 6, 10, 14, 18, 22, 25, 27):
            raise ValueError(
                "v05 freezes injection blocks to "
                "(2,6,10,14,18,22,25,27)"
            )
        self.condition_encoder = PhysicsConditionEncoder(object_dim)
        self.groups = nn.ModuleDict(
            {
                str(index): ObjectTimeGraphGroup(
                    wan_dim=wan_dim,
                    object_dim=object_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    graph_blocks=graph_blocks_per_group,
                    enable_initial_force_kick=(
                        index == self.injection_blocks[0]
                    ),
                )
                for index in self.injection_blocks
            }
        )

    @staticmethod
    def _batch_tensor(
        value: torch.Tensor,
        batch: int,
        trailing_shape: Tuple[int, ...],
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if value.ndim == len(trailing_shape):
            value = value.unsqueeze(0)
        if value.shape[0] == 1 and batch > 1:
            value = value.expand((batch,) + tuple(value.shape[1:]))
        expected = (batch,) + trailing_shape
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must be {expected}, got {tuple(value.shape)}")
        return value.to(device=device, dtype=dtype)

    def forward(
        self,
        block_index: int,
        hidden: torch.Tensor,
        condition: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        frames: int,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = str(block_index)
        if key not in self.groups:
            raise KeyError(block_index)
        missing = set(FORWARD_CONDITION_KEYS) - set(condition)
        if missing:
            raise ValueError(
                f"object-time graph forward condition missing {sorted(missing)}"
            )

        parameter = next(self.parameters())
        hidden_fp32 = hidden.to(dtype=parameter.dtype)
        batch = hidden.shape[0]
        device, dtype = hidden.device, hidden_fp32.dtype
        # Explicit whitelist: labels may coexist in the outer dictionary, but
        # no future target is copied into the forward condition.
        first_masks = self._batch_tensor(
            condition["first_frame_masks"],
            batch, (2, height, width), "first_frame_masks", device, dtype
        )
        force = self._batch_tensor(
            condition["force"], batch, (3,), "force", device, dtype
        )
        force_target = self._batch_tensor(
            condition["force_target"], batch, (2,), "force_target", device, dtype
        )
        mu = self._batch_tensor(
            condition["mu"], batch, (2,), "mu", device, dtype
        )
        restitution = self._batch_tensor(
            condition["restitution"], batch, (1,), "restitution", device, dtype
        )
        tensors = (first_masks, force, force_target, mu, restitution)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("object-time graph forward inputs must be finite")
        if first_masks.min() < 0 or first_masks.max() > 1:
            raise ValueError("first_frame_masks must be in [0,1]")
        if force[:, :1].min() < 0 or force[:, :1].max() > 1:
            raise ValueError("normalized force magnitude must be in [0,1]")
        active_force = force[:, 0] > 0
        if active_force.any():
            direction_norm = force[active_force, 1:].norm(dim=-1)
            if torch.any((direction_norm < 0.9) | (direction_norm > 1.1)):
                raise ValueError("active force direction norm must be near one")
        if torch.any((force_target < 0) | (force_target > 1)):
            raise ValueError("force_target must be a one-hot vector")
        if not torch.allclose(
            force_target.sum(dim=-1),
            torch.ones(batch, device=device, dtype=dtype),
        ):
            raise ValueError("force_target must select exactly one object")
        if torch.any((mu < 0) | (mu > 2)):
            raise ValueError("mu_A/mu_B must be in [0,2]")
        if torch.any((restitution < 0) | (restitution > 1)):
            raise ValueError("restitution must be in [0,1]")
        force_feature, mu_features, restitution_feature = self.condition_encoder(
            force, mu, restitution
        )
        hidden_fp32, attention_logits, contact_logits = self.groups[key](
            hidden=hidden_fp32,
            first_frame_masks=first_masks,
            force_feature=force_feature,
            force_magnitude=force[:, :1],
            mu_features=mu_features,
            restitution_feature=restitution_feature,
            force_target=force_target,
            timestep=timestep,
            frames=frames,
            height=height,
            width=width,
            apply_initial_force_kick=(block_index == self.injection_blocks[0]),
        )
        return hidden_fp32.to(dtype=hidden.dtype), attention_logits, contact_logits
