"""4-GPU model-parallel decoder for the Wan2.2 VAE38.

The upstream VAE decoder keeps causal convolution state in mutable Python
lists.  That is safe for the original single-device implementation, but it
is not a safe boundary for activation checkpointing or cross-device stages.
This module keeps the legacy decoder untouched and exposes a parallel path
with an immutable-at-the-call-boundary cache state.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .wan_video_vae import (
    CACHE_T,
    AttentionBlock,
    CausalConv3d,
    ResidualBlock,
    Up_ResidualBlock,
    count_conv3d,
    unpatchify,
)


CacheEntry = torch.Tensor | str | None


@dataclass(frozen=True)
class DecoderCacheState:
    """Explicit causal state at a stage boundary.

    The tuple itself is never mutated by the model-parallel path.  Existing
    legacy blocks still receive a short-lived list copy because their public
    API predates this wrapper; mutations stay inside one stage call and are
    returned as a new state.
    """

    entries: tuple[CacheEntry, ...]
    next_index: int = 0


def _state_after_legacy_block(
    cache: list[CacheEntry], index: list[int]
) -> DecoderCacheState:
    return DecoderCacheState(tuple(cache), int(index[0]))


def _direct_causal_conv(
    layer: CausalConv3d,
    x: torch.Tensor,
    state: DecoderCacheState,
) -> tuple[torch.Tensor, DecoderCacheState]:
    """Apply a top-level causal conv without mutating the input state."""

    cache = list(state.entries)
    index = state.next_index
    previous = cache[index]
    cache_x = x[:, :, -CACHE_T:, :, :].clone()
    if (
        cache_x.shape[2] < 2
        and isinstance(previous, torch.Tensor)
    ):
        cache_x = torch.cat(
            [
                previous[:, :, -1, :, :].unsqueeze(2).to(
                    cache_x.device, non_blocking=True
                ),
                cache_x,
            ],
            dim=2,
        )
    output = layer(x, previous)
    cache[index] = cache_x
    return output, DecoderCacheState(tuple(cache), index + 1)


class WanDecoderStage(nn.Module):
    """One resolution stage with explicit causal state input/output."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        device: torch.device,
        checkpoint_stateless: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.device = torch.device(device)
        self.checkpoint_stateless = bool(checkpoint_stateless)
        self.to(self.device)
        self.cache_size = count_conv3d(self)

    def initial_state(self) -> DecoderCacheState:
        return DecoderCacheState((None,) * self.cache_size, 0)

    def _checkpoint_attention(self, layer: AttentionBlock, x: torch.Tensor) -> torch.Tensor:
        if (
            self.checkpoint_stateless
            and torch.is_grad_enabled()
            and x.requires_grad
        ):
            return checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    def _checkpoint_cache_free_residual(
        self,
        layer: ResidualBlock,
        x: torch.Tensor,
        state: DecoderCacheState,
    ) -> tuple[torch.Tensor, DecoderCacheState] | None:
        # Current Wan ResidualBlock instances contain CausalConv3d and are
        # therefore stateful.  Keep this branch for genuinely cache-free
        # residual variants without applying checkpoint to the mutable path.
        if count_conv3d(layer) != 0:
            return None
        if (
            self.checkpoint_stateless
            and torch.is_grad_enabled()
            and x.requires_grad
        ):
            def pure_forward(value: torch.Tensor) -> torch.Tensor:
                return layer(value, None, [0])[0]

            return checkpoint(pure_forward, x, use_reentrant=False), state
        return layer(x, None, [0])[0], state

    def forward(
        self,
        x: torch.Tensor,
        state: DecoderCacheState,
        *,
        first_chunk: bool = False,
    ) -> tuple[torch.Tensor, DecoderCacheState]:
        if x.device != self.device:
            raise RuntimeError(
                f"WanDecoderStage expected {self.device}, got activation on {x.device}"
            )
        # Wan resets _conv_idx for every latent time step.  Only the cache
        # entries survive across calls; carrying next_index forward would
        # address the cache list past its end on the second frame.
        current = DecoderCacheState(state.entries, 0)
        for layer in self.layers:
            if isinstance(layer, CausalConv3d):
                x, current = _direct_causal_conv(layer, x, current)
            elif isinstance(layer, ResidualBlock):
                cache_free = self._checkpoint_cache_free_residual(layer, x, current)
                if cache_free is not None:
                    x, current = cache_free
                else:
                    cache = list(current.entries)
                    index = [current.next_index]
                    x, cache, index = layer(x, cache, index)
                    current = _state_after_legacy_block(cache, index)
            elif isinstance(layer, Up_ResidualBlock):
                cache = list(current.entries)
                index = [current.next_index]
                x, cache, index = layer(
                    x,
                    cache,
                    index,
                    first_chunk=first_chunk,
                )
                current = _state_after_legacy_block(cache, index)
            elif isinstance(layer, AttentionBlock):
                x = self._checkpoint_attention(layer, x)
            else:
                x = layer(x)
        return x, DecoderCacheState(current.entries, 0)


def memory_snapshot(devices: Sequence[torch.device]) -> dict[str, dict[str, float]]:
    """Return synchronized per-GPU allocator evidence."""

    result: dict[str, dict[str, float]] = {}
    for device in devices:
        torch.cuda.synchronize(device)
        result[str(device)] = {
            "allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
            "reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
            "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device)
            / (1024**3),
        }
    return result


class WanVideoVAE38ModelParallel(nn.Module):
    """Split one Wan2.2 VAE38 decoder over exactly four GPUs.

    The base VAE is loaded once on CPU and its decoder children are moved in
    place.  No stage receives a second copy of the VAE.  Stage boundaries are
    resolution boundaries, not parameter-count boundaries:

    - GPU 0: decoder input convolution, middle blocks, lowest-resolution up;
    - GPU 1: next temporal/spatial upsampling stage;
    - GPU 2: high-resolution spatial stage;
    - GPU 3: final high-resolution stage and output head.
    """

    def __init__(
        self,
        base_vae: nn.Module,
        devices: Sequence[str | torch.device],
        *,
        checkpoint_stateless: bool = True,
        autocast_bf16: bool = True,
    ) -> None:
        super().__init__()
        if len(devices) != 4:
            raise ValueError("WanVideoVAE38ModelParallel requires exactly four devices")
        self.devices = tuple(torch.device(device) for device in devices)
        if any(device.type != "cuda" for device in self.devices):
            raise ValueError("all model-parallel devices must be CUDA devices")
        if not hasattr(base_vae, "model") or not hasattr(base_vae.model, "decoder"):
            raise TypeError("base_vae must be a loaded WanVideoVAE38 instance")
        self.base_vae = base_vae
        self.core = base_vae.model
        self.autocast_bf16 = bool(autocast_bf16)

        decoder = self.core.decoder
        if len(decoder.upsamples) != 4:
            raise ValueError(
                f"expected four Wan decoder upsampling stages, got {len(decoder.upsamples)}"
            )

        # Move the only copy of every decoder child to its stage device.
        self.core.conv2.to(self.devices[0])
        self.scale = [
            value.to(device=self.devices[0], dtype=torch.bfloat16)
            if isinstance(value, torch.Tensor)
            else value
            for value in base_vae.scale
        ]
        self.stages = nn.ModuleList(
            [
                WanDecoderStage(
                    [decoder.conv1, *list(decoder.middle), decoder.upsamples[0]],
                    self.devices[0],
                    checkpoint_stateless=checkpoint_stateless,
                ),
                WanDecoderStage(
                    [decoder.upsamples[1]],
                    self.devices[1],
                    checkpoint_stateless=checkpoint_stateless,
                ),
                WanDecoderStage(
                    [decoder.upsamples[2]],
                    self.devices[2],
                    checkpoint_stateless=checkpoint_stateless,
                ),
                WanDecoderStage(
                    [decoder.upsamples[3], *list(decoder.head)],
                    self.devices[3],
                    checkpoint_stateless=checkpoint_stateless,
                ),
            ]
        )

    def parameter_placement(self) -> dict[str, list[str]]:
        def devices_for(module: nn.Module, *, buffers: bool = False) -> list[str]:
            tensors = module.buffers() if buffers else module.parameters()
            return sorted({str(tensor.device) for tensor in tensors})

        placement: dict[str, list[str]] = {}
        for stage_index, stage in enumerate(self.stages):
            placement[f"stage{stage_index}"] = devices_for(stage)
        placement["conv2"] = devices_for(self.core.conv2)
        return placement

    def buffer_placement(self) -> dict[str, list[str]]:
        def devices_for(module: nn.Module) -> list[str]:
            return sorted({str(buffer.device) for buffer in module.buffers()})

        placement: dict[str, list[str]] = {}
        for stage_index, stage in enumerate(self.stages):
            placement[f"stage{stage_index}"] = devices_for(stage)
        placement["conv2"] = devices_for(self.core.conv2)
        return placement

    def _decode_impl(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, dict[str, dict[str, float]]]]:
        device0 = self.devices[0]
        if isinstance(self.scale[0], torch.Tensor):
            z = latent.to(device0, non_blocking=True)
            z = z / self.scale[1].view(1, self.core.z_dim, 1, 1, 1) + self.scale[0].view(
                1, self.core.z_dim, 1, 1, 1
            )
        else:
            z = latent.to(device0, non_blocking=True)
            z = z / self.scale[1] + self.scale[0]

        latent_features = self.core.conv2(z)
        states = [stage.initial_state() for stage in self.stages]
        decoded_parts: list[torch.Tensor] = []
        profile: dict[str, dict[str, dict[str, float]]] = {}

        for index in range(int(z.shape[2])):
            x, states[0] = self.stages[0](
                latent_features[:, :, index : index + 1, :, :],
                states[0],
                first_chunk=index == 0,
            )
            x = x.to(self.devices[1], non_blocking=True)
            x, states[1] = self.stages[1](x, states[1], first_chunk=index == 0)
            x = x.to(self.devices[2], non_blocking=True)
            x, states[2] = self.stages[2](x, states[2], first_chunk=index == 0)
            x = x.to(self.devices[3], non_blocking=True)
            x, states[3] = self.stages[3](x, states[3], first_chunk=index == 0)
            decoded_parts.append(x)

            if index == int(z.shape[2]) - 1:
                profile["after_stage0"] = memory_snapshot([self.devices[0]])
                profile["after_stage1"] = memory_snapshot([self.devices[1]])
                profile["after_stage2"] = memory_snapshot([self.devices[2]])
                profile["after_stage3"] = memory_snapshot([self.devices[3]])

        video = unpatchify(torch.cat(decoded_parts, dim=2), patch_size=2)
        # Keep output and graph in BF16.  The caller owns any FP32 loss
        # reduction; there is no whole-video float conversion here.
        return video.clamp_(-1, 1), profile

    def decode(
        self,
        latent: torch.Tensor,
        *,
        return_profile: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, dict[str, dict[str, float]]]]:
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.autocast_bf16
            else contextlib.nullcontext()
        )
        with context:
            video, profile = self._decode_impl(latent)
        if return_profile:
            return video, profile
        return video

    forward = decode
