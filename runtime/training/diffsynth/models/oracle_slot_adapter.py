"""Oracle-routed typed object-slot adapter for the first ELSA-inspired POC.

The router is deliberately deterministic: per-frame instance masks are
compiled offline to the Wan token grid.  The trainable module only performs
Lift -> typed Inject/Carry/Exchange -> Project.  Ground-truth impact labels are
never consumed here; they are read only by the training loss.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class TypedSlotMixerCore(nn.Module):
    """Shared Inject/Carry/Exchange computation used at all insertion depths."""

    def __init__(
        self,
        slot_dim: int = 256,
        effect_dim: int = 128,
        max_times: int = 13,
        max_objects: int = 8,
        geometry_dim: int = 6,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.effect_dim = effect_dim
        self.geometry_dim = geometry_dim

        self.time_embedding = nn.Embedding(max_times, slot_dim)
        self.object_embedding = nn.Embedding(max_objects, slot_dim)
        self.geometry_encoder = _mlp(geometry_dim, effect_dim, slot_dim)
        self.slot_norm = nn.LayerNorm(slot_dim)

        self.force_encoder = _mlp(3, effect_dim, effect_dim)
        self.gravity_encoder = _mlp(1, effect_dim, effect_dim)
        self.friction_encoder = _mlp(1, effect_dim, effect_dim)
        self.restitution_encoder = _mlp(1, effect_dim, effect_dim)

        self.force_branch = _mlp(slot_dim + effect_dim, effect_dim, effect_dim)
        self.gravity_branch = _mlp(slot_dim + effect_dim, effect_dim, effect_dim)
        self.friction_branch = _mlp(
            slot_dim + effect_dim * 2, effect_dim * 2, effect_dim
        )
        self.carry_input = nn.Linear(slot_dim, effect_dim)
        self.carry = nn.GRUCell(effect_dim, effect_dim)

        relation_input_dim = slot_dim * 2 + effect_dim * 2 + geometry_dim + 2
        self.relation_encoder = _mlp(
            relation_input_dim, effect_dim * 2, effect_dim
        )
        self.impact_head = _mlp(effect_dim * 2, effect_dim, 1)
        nn.init.constant_(self.impact_head[-1].bias, -2.0)

        message_input_dim = slot_dim * 2 + effect_dim * 3
        self.impact_value = _mlp(
            message_input_dim, effect_dim * 2, effect_dim
        )
        self.effect_norm = nn.LayerNorm(effect_dim)
        self.effect_decoder = _mlp(
            slot_dim + effect_dim, slot_dim * 2, slot_dim
        )

    @staticmethod
    def mask_geometry(masks: torch.Tensor) -> torch.Tensor:
        """Return centroid, area, spatial spread and visibility from soft masks.

        Args:
            masks: [B,T,O,H,W], values in [0,1].
        Returns:
            [B,T,O,6] containing cx, cy, area, std_x, std_y, visible.
        """
        batch, times, objects, height, width = masks.shape
        dtype, device = masks.dtype, masks.device
        y = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
        x = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        mass = masks.sum(dim=(-2, -1)).clamp_min(1e-6)
        cx = (masks * xx).sum(dim=(-2, -1)) / mass
        cy = (masks * yy).sum(dim=(-2, -1)) / mass
        var_x = (masks * (xx - cx[..., None, None]).square()).sum(
            dim=(-2, -1)
        ) / mass
        var_y = (masks * (yy - cy[..., None, None]).square()).sum(
            dim=(-2, -1)
        ) / mass
        area = mass / float(height * width)
        visible = (mass > 1e-4).to(dtype)
        return torch.stack(
            (cx, cy, area, var_x.sqrt(), var_y.sqrt(), visible), dim=-1
        ).reshape(batch, times, objects, 6)

    def _pair_relation(
        self,
        slots: torch.Tensor,
        states: torch.Tensor,
        geometry: torch.Tensor,
        velocity: torch.Tensor,
        i: int,
        j: int,
    ) -> torch.Tensor:
        return self.relation_encoder(
            torch.cat(
                (
                    slots[:, i],
                    slots[:, j],
                    states[:, i],
                    states[:, j],
                    geometry[:, i] - geometry[:, j],
                    velocity[:, i] - velocity[:, j],
                ),
                dim=-1,
            )
        )

    def forward(
        self,
        visual_slots: torch.Tensor,
        condition: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mix object slots independently for one Wan insertion depth.

        visual_slots is [B,T,O,slot_dim].  Returned impact logits are
        [B,T-1,P], with P=O*(O-1)/2 unordered pairs.
        """
        batch, times, objects, _ = visual_slots.shape
        if times > self.time_embedding.num_embeddings:
            raise ValueError(f"too many latent times: {times}")
        if objects > self.object_embedding.num_embeddings:
            raise ValueError(f"too many objects: {objects}")

        masks = condition["masks"].float()
        geometry = self.mask_geometry(masks)
        time_ids = torch.arange(times, device=visual_slots.device)
        object_ids = torch.arange(objects, device=visual_slots.device)
        slots = self.slot_norm(
            visual_slots
            + self.time_embedding(time_ids)[None, :, None]
            + self.object_embedding(object_ids)[None, None]
            + self.geometry_encoder(geometry)
        )

        force = self.force_encoder(condition["force"].float())
        gravity = self.gravity_encoder(condition["gravity"].float())
        friction = self.friction_encoder(condition["friction"].float())
        restitution = self.restitution_encoder(
            condition["restitution"].float()
        )
        target = condition["force_target"].float()
        active = condition["force_active"].float()

        states = torch.zeros(
            batch, objects, self.effect_dim,
            dtype=slots.dtype, device=slots.device,
        )
        output_states = [states]
        previous_relations = {}
        impact_logits = []
        centroid_velocity = torch.zeros(
            batch, times, objects, 2,
            dtype=geometry.dtype, device=geometry.device,
        )
        centroid_velocity[:, 1:] = (
            geometry[:, 1:, :, :2] - geometry[:, :-1, :, :2]
        )

        for tau in range(1, times):
            current_slots = slots[:, tau]
            f_value = self.force_branch(
                torch.cat(
                    (current_slots, force[:, None].expand(-1, objects, -1)),
                    dim=-1,
                )
            )
            f_gate = target[:, :, None] * active[:, tau, None, None]
            g_value = self.gravity_branch(
                torch.cat(
                    (current_slots, gravity[:, None].expand(-1, objects, -1)),
                    dim=-1,
                )
            )
            mu_value = self.friction_branch(
                torch.cat(
                    (
                        current_slots,
                        states,
                        friction[:, None].expand(-1, objects, -1),
                    ),
                    dim=-1,
                )
            )
            local_effect = f_gate * f_value + g_value + mu_value
            carry_input = self.carry_input(current_slots) + local_effect
            carried = self.carry(
                carry_input.reshape(batch * objects, self.effect_dim),
                states.reshape(batch * objects, self.effect_dim),
            ).reshape(batch, objects, self.effect_dim)

            incoming = torch.zeros_like(carried)
            current_logits = []
            for i in range(objects):
                for j in range(i + 1, objects):
                    relation = self._pair_relation(
                        current_slots,
                        carried,
                        geometry[:, tau],
                        centroid_velocity[:, tau],
                        i,
                        j,
                    )
                    previous = previous_relations.get((i, j))
                    if previous is None:
                        previous = torch.zeros_like(relation)
                    logit = self.impact_head(
                        torch.cat((previous, relation), dim=-1)
                    ).squeeze(-1)
                    gate = torch.sigmoid(logit)[:, None]
                    current_logits.append(logit)
                    previous_relations[(i, j)] = relation

                    msg_i_to_j = self.impact_value(
                        torch.cat(
                            (
                                current_slots[:, i],
                                current_slots[:, j],
                                carried[:, i],
                                carried[:, j],
                                restitution,
                            ),
                            dim=-1,
                        )
                    )
                    msg_j_to_i = self.impact_value(
                        torch.cat(
                            (
                                current_slots[:, j],
                                current_slots[:, i],
                                carried[:, j],
                                carried[:, i],
                                restitution,
                            ),
                            dim=-1,
                        )
                    )
                    incoming[:, j] = incoming[:, j] + gate * msg_i_to_j
                    incoming[:, i] = incoming[:, i] + gate * msg_j_to_i

            states = self.effect_norm(carried + incoming)
            output_states.append(states)
            impact_logits.append(torch.stack(current_logits, dim=-1))

        effect_states = torch.stack(output_states, dim=1)
        delta_slots = self.effect_decoder(
            torch.cat((slots, effect_states), dim=-1)
        )
        return delta_slots, torch.stack(impact_logits, dim=1)


class OracleRoutedSlotAdapter(nn.Module):
    """Layer-specific Lift/Project shells around one shared typed mixer."""

    def __init__(
        self,
        wan_dim: int,
        injection_blocks: Iterable[int] = (7, 15, 23),
        slot_dim: int = 256,
        effect_dim: int = 128,
    ):
        super().__init__()
        self.injection_blocks = tuple(int(x) for x in injection_blocks)
        self.slot_dim = slot_dim
        self.shared_mixer = TypedSlotMixerCore(slot_dim, effect_dim)
        self.input_norms = nn.ModuleDict(
            {str(i): nn.LayerNorm(wan_dim) for i in self.injection_blocks}
        )
        self.input_projections = nn.ModuleDict(
            {str(i): nn.Linear(wan_dim, slot_dim) for i in self.injection_blocks}
        )
        self.output_projections = nn.ModuleDict(
            {str(i): nn.Linear(slot_dim, wan_dim) for i in self.injection_blocks}
        )
        self.noise_gates = nn.ModuleDict(
            {str(i): _mlp(1, 32, 1) for i in self.injection_blocks}
        )
        self.residual_scales = nn.ParameterDict(
            {str(i): nn.Parameter(torch.ones(())) for i in self.injection_blocks}
        )
        for projection in self.output_projections.values():
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

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
        if key not in self.input_projections:
            raise KeyError(block_index)
        if hidden.shape[1] != frames * height * width:
            raise ValueError(
                "oracle slot adapter requires plain Wan video tokens without "
                "prepended reference tokens"
            )

        parameter = next(self.parameters())
        hidden_fp32 = hidden.to(dtype=parameter.dtype)
        masks = condition["masks"]
        if masks.ndim == 4:
            masks = masks.unsqueeze(0)
        masks = masks.to(
            device=hidden.device, dtype=hidden_fp32.dtype
        )
        expected = (hidden.shape[0], frames, masks.shape[2], height, width)
        if tuple(masks.shape) != expected:
            raise ValueError(
                f"oracle masks must be {expected}, got {tuple(masks.shape)}"
            )
        condition_fp32 = {}
        for key_, value in condition.items():
            if key_ == "impact_labels":
                continue
            if key_ != "masks" and value.ndim == 1:
                value = value.unsqueeze(0)
            condition_fp32[key_] = value.to(
                device=hidden.device, dtype=hidden_fp32.dtype
            )
        condition_fp32["masks"] = masks

        grid = self.input_projections[key](
            self.input_norms[key](hidden_fp32)
        ).reshape(hidden.shape[0], frames, height * width, self.slot_dim)
        flat_masks = masks.flatten(-2)
        read = flat_masks / flat_masks.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        slots = torch.einsum("btog,btgd->btod", read, grid)
        delta_slots, logits = self.shared_mixer(slots, condition_fp32)

        write = flat_masks / flat_masks.sum(dim=2, keepdim=True).clamp_min(1.0)
        delta_grid = torch.einsum("btog,btod->btgd", write, delta_slots)
        delta_grid[:, 0] = 0
        projected = self.output_projections[key](
            delta_grid.reshape(hidden.shape[0], frames * height * width, -1)
        )
        # TI2V's separated-timestep path expands one diffusion timestep to a
        # value per video token (first-frame tokens are clean zeros).  The
        # adapter gate is layer/sample scoped, so reduce that token schedule
        # back to one scalar per sample before the MLP.  Treating the token
        # dimension as a batch dimension would broadcast to [L,L,D].
        if timestep.numel() % hidden.shape[0] != 0:
            raise ValueError(
                f"timestep elements {timestep.numel()} are not divisible by "
                f"batch {hidden.shape[0]}"
            )
        normalized_timestep = (
            timestep.float().reshape(hidden.shape[0], -1).mean(dim=1, keepdim=True)
            / 1000.0
        ).clamp(0.0, 1.0)
        noise_gate = torch.sigmoid(
            self.noise_gates[key](normalized_timestep)
        )[:, None]
        residual = (
            noise_gate * self.residual_scales[key] * projected
        ).to(dtype=hidden.dtype)
        return hidden + residual, logits
