import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .oracle_slot_adapter import OracleRoutedSlotAdapter
from .object_time_graph_adapter import (
    FORWARD_CONDITION_KEYS,
    ObjectTimeGraphAdapter,
)
from .sparse_object_interaction_adapter import (
    FORWARD_CONDITION_KEYS as SPARSE_OBJECT_FORWARD_KEYS,
    TEMPORAL_FORWARD_CONDITION_KEYS as SPARSE_OBJECT_TEMPORAL_FORWARD_KEYS,
    TEMPORAL_INDEPENDENT_EDGE_SCHEDULE_FORWARD_CONDITION_KEYS as SPARSE_OBJECT_TEMPORAL_INDEPENDENT_EDGE_SCHEDULE_FORWARD_KEYS,
    TEMPORAL_PAIR_SCHEDULE_FORWARD_CONDITION_KEYS as SPARSE_OBJECT_TEMPORAL_PAIR_SCHEDULE_FORWARD_KEYS,
    TEMPORAL_NO_INTERACTION_SCHEDULE_FORWARD_CONDITION_KEYS as SPARSE_OBJECT_TEMPORAL_NO_INTERACTION_SCHEDULE_FORWARD_KEYS,
    TEMPORAL_SCHEDULE_FORWARD_CONDITION_KEYS as SPARSE_OBJECT_TEMPORAL_SCHEDULE_FORWARD_KEYS,
    SparseObjectInteractionAdapter,
    SparseObjectInteractionTemporalAdapter,
)
from .phyparam_control_dit import PhyParamControlDiT, PhyParamControlState
import math
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter
from ..core.gradient import gradient_checkpoint_forward
from .wantodance import WanToDanceRotaryEmbedding, WanToDanceMusicEncoderLayer

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE and q.is_cuda:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE and q.is_cuda:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE and q.is_cuda:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def harmonic_embedding_1d(num_bands: int, position: torch.Tensor) -> torch.Tensor:
    """Dyadic sin/cos features for one scalar normalized to [0, 1]."""
    if num_bands <= 0:
        raise ValueError(f"num_bands must be positive, got {num_bands}")
    frequencies = torch.pow(
        2.0,
        torch.arange(num_bands, dtype=torch.float64, device=position.device),
    ) * math.pi
    phase = torch.outer(position.to(torch.float64), frequencies)
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1).flatten(1).to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def set_to_torch_norm(models):
    for model in models:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.use_torch_norm = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_torch_norm = False
        self.normalized_shape = (dim,)

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        if self.use_torch_norm:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:        
            # Preserve the activation dtype even when a trainable FP32 master
            # weight is used under BF16 autocast (required by FlashAttention).
            return self.norm(x.float()).to(dtype) * self.weight.to(dtype=dtype)


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class ForceWindowCrossAttention(nn.Module):
    """Frame-wise, spatially aligned force attention.

    Video and force tokens are partitioned into the same non-overlapping spatial
    windows. A video token can therefore query force tokens only from its own
    frame and local window; no scalar x/y coordinate token is used.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.window_size = window_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.attn = CrossAttention(dim, num_heads, eps, has_image_input=False)
        # Exact clean-base equivalence at step zero.
        nn.init.zeros_(self.attn.o.weight)
        nn.init.zeros_(self.attn.o.bias)

    def forward(
        self,
        x: torch.Tensor,
        force_tokens: torch.Tensor,
        support: torch.Tensor,
        frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        window = self.window_size
        if height % window or width % window:
            raise ValueError(
                f"force attention grid {(height, width)} must be divisible by window {window}"
            )
        batch, _, dim = x.shape
        x_grid = x.reshape(batch, frames, height, width, dim)
        force_grid = force_tokens.reshape(batch, frames, height, width, dim)
        support_grid = support.reshape(batch, frames, height, width, 1)
        x_windows = rearrange(
            x_grid,
            "b t (nh wh) (nw ww) d -> (b t nh nw) (wh ww) d",
            wh=window,
            ww=window,
        )
        force_windows = rearrange(
            force_grid,
            "b t (nh wh) (nw ww) d -> (b t nh nw) (wh ww) d",
            wh=window,
            ww=window,
        )
        support_windows = rearrange(
            support_grid,
            "b t (nh wh) (nw ww) c -> (b t nh nw) (wh ww) c",
            wh=window,
            ww=window,
        )
        residual = self.attn(self.norm(x_windows), force_windows)
        # Keep the new residual spatially local. Later frozen self-attention can
        # propagate the intervention to the rest of the scene.
        residual = residual * support_windows.to(dtype=residual.dtype)
        return rearrange(
            residual,
            "(b t nh nw) (wh ww) d -> b (t nh wh nw ww) d",
            b=batch,
            t=frames,
            nh=height // window,
            nw=width // window,
            wh=window,
            ww=window,
        )


GLOBAL_PHYSICS_TYPES = ("g", "mu", "e")
GLOBAL_PHYSICS_TYPE_TO_ID = {name: idx for idx, name in enumerate(GLOBAL_PHYSICS_TYPES)}
GLOBAL_PHYSICS_HARMONIC_BANDS = 8
GLOBAL_PHYSICS_MAX = {"g": 40.0, "mu": 1.0, "e": 1.0}
GLOBAL_PHYSICS_SCALE = {name: 1.0 / maximum for name, maximum in GLOBAL_PHYSICS_MAX.items()}


class PhysicsTokenEncoder(nn.Module):
    """Builds one KV token per (type, raw_value) physics scalar.

    token = MLP_val(harmonic_embed(raw_value / MAX[type])) + type_embed[type_id]
    """

    def __init__(
        self,
        dim: int,
        num_types: int = len(GLOBAL_PHYSICS_TYPES),
        harmonic_bands: int = GLOBAL_PHYSICS_HARMONIC_BANDS,
    ):
        super().__init__()
        self.dim = dim
        self.harmonic_bands = harmonic_bands
        self.type_embed = nn.Embedding(num_types, dim)
        self.mlp_val = nn.Sequential(
            nn.Linear(2 * harmonic_bands, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, scaled_values: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        # scaled_values: [M] float: already SCALE[type]*raw_value. type_ids: [M] long.
        harmonic = harmonic_embedding_1d(self.harmonic_bands, scaled_values)
        value_embed = self.mlp_val(harmonic.to(self.mlp_val[0].weight.dtype))
        return value_embed + self.type_embed(type_ids)


class GlobalPhysicsCrossAttn(nn.Module):
    """Global scalar-physics cross attention: video hidden states as query,
    a variable-length set of g/mu/e physics tokens as KV.

    Fully independent weights from ForceWindowCrossAttention (separate q/k/v/o,
    separate zero-init, no shared projections or attention weights); it happens
    to be mounted at the same Wan blocks, but that is a placement choice, not
    shared computation. flash_attention()/AttentionModule/CrossAttention in
    this file have no key_padding_mask support (the F path never needs one,
    since its windows are always fully dense), so this module does its own
    masked attention with F.scaled_dot_product_attention(attn_mask=...).
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_q = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm_kv = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.q_rms = RMSNorm(dim, eps=eps)
        self.k_rms = RMSNorm(dim, eps=eps)
        # Exact clean-base equivalence at step zero (same convention as
        # ForceWindowCrossAttention): the residual this module adds is
        # identically zero until training moves these weights.
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """x: [B,L,D] video hidden states (query). tokens: [B,N,D] physics KV,
        already padded to this batch's max N. key_padding_mask: [B,N] bool,
        True = real token, False = padding. N must be >= 1 (callers must use
        the N=0 shortcut, i.e. skip calling this module entirely, when a batch
        has zero physics tokens for every sample -- see
        WanModel.prepare_global_physics_tokens, which returns None in that
        case exactly like patchify_with_force does for an absent force map)."""
        batch, length, dim = x.shape
        n = tokens.shape[1]
        if n == 0:
            # Defensive guard: an empty KV set has nothing to attend to. Callers
            # are expected to short-circuit before this (state is None), but a
            # zero-length tensor must never reach scaled_dot_product_attention.
            return torch.zeros_like(x)
        valid_sample = key_padding_mask.any(dim=1)  # [B] any real (unpadded) token at all
        q = self.q_rms(self.q(self.norm_q(x)))
        k = self.k_rms(self.k(self.norm_kv(tokens)))
        v = self.v(self.norm_kv(tokens))
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        # N=0 guard for samples with zero valid tokens (all-masked row): force
        # unmask slot 0 so softmax never sees an all -inf row (which produces
        # NaN), then zero that row's output afterwards regardless of what the
        # (otherwise unused) attention read. Samples with >=1 real token are
        # unaffected since safe_mask only changes rows where valid_sample is
        # already False.
        safe_mask = key_padding_mask.clone()
        safe_mask[~valid_sample, 0] = True
        attn_mask = torch.zeros(batch, 1, 1, n, dtype=q.dtype, device=q.device)
        attn_mask = attn_mask.masked_fill(~safe_mask[:, None, None, :], float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.o(out)
        out = out * valid_sample.to(dtype=out.dtype)[:, None, None]
        return out


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def _modulation(self, t_mod):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def forward_self_attention(self, x, t_mod, freqs):
        shift_msa, scale_msa, gate_msa, _, _, _ = self._modulation(t_mod)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        return x

    def forward_cross_attention_ffn(self, x, context, t_mod):
        _, _, _, shift_mlp, scale_mlp, gate_mlp = self._modulation(t_mod)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x

    def forward(self, x, context, t_mod, freqs):
        x = self.forward_self_attention(x, t_mod, freqs)
        return self.forward_cross_attention_ffn(x, context, t_mod)


class FrictionZeroConv2d(nn.Module):
    """Zero-initialised 1x1 residual projection used by the side branch."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FrictionControlNetBranch(nn.Module):
    """One PhyCo-style branch for a frozen-VAE friction-map latent.

    The semantic condition is one float map.  It is repeated to RGB only at the
    frozen Wan VAE boundary, producing a 48-channel latent.  This module keeps
    the trainable adapter deliberately small: one 1x1 projection, five copied
    Wan blocks, and one zero-initialised output projection per copied block.
    """

    def __init__(self, latent_channels: int, dim: int, num_blocks: int):
        super().__init__()
        self.latent_channels = latent_channels
        self.num_blocks = num_blocks
        self.condition_projection = nn.Sequential(
            nn.Conv2d(latent_channels, dim, kernel_size=1),
            nn.GELU(approximate="tanh"),
        )
        self.blocks = nn.ModuleList()
        self.output_projections = nn.ModuleList(
            [FrictionZeroConv2d(dim) for _ in range(num_blocks)]
        )

    def initialize_from_base_blocks(self, base_blocks: nn.ModuleList) -> None:
        if len(base_blocks) < self.num_blocks:
            raise ValueError(
                f"requested {self.num_blocks} ControlNet blocks from only {len(base_blocks)} base blocks"
            )
        self.blocks = nn.ModuleList(
            [copy.deepcopy(block) for block in base_blocks[: self.num_blocks]]
        )

    def initial_state(
        self,
        control_latents: torch.Tensor,
        frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if control_latents.ndim != 5:
            raise ValueError(
                "control latents must have shape [B, C, T, H, W]"
            )
        if control_latents.shape[1] != self.latent_channels:
            raise ValueError(
                f"expected {self.latent_channels} VAE channels, got {control_latents.shape[1]}"
            )
        control_frames = int(control_latents.shape[2])
        if control_frames not in (1, frames):
            raise ValueError(
                "control latent time must be static T=1 or match the Wan token time: "
                f"control T={control_frames}, Wan T={frames}"
            )
        projection_parameter = next(self.condition_projection.parameters())
        # Apply the existing 2-D projection independently to every latent time
        # slice.  Treating T as part of the batch preserves temporal order and
        # introduces no temporal layers, embeddings, or new parameters.
        batch, channels, _, latent_height, latent_width = control_latents.shape
        spatial = F.interpolate(
            control_latents.permute(0, 2, 1, 3, 4)
            .reshape(batch * control_frames, channels, latent_height, latent_width)
            .to(device=projection_parameter.device),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=spatial.is_cuda,
        ):
            spatial = self.condition_projection(spatial)
        dim = spatial.shape[1]
        spatial = spatial.reshape(batch, control_frames, dim, height, width)
        if control_frames == 1 and frames != 1:
            spatial = spatial.expand(-1, frames, -1, -1, -1)
        return spatial.permute(0, 1, 3, 4, 2).reshape(
            batch, frames * height * width, dim
        )

    def project_output(
        self,
        block_index: int,
        state: torch.Tensor,
        frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch, _, dim = state.shape
        feature = (
            state.reshape(batch, frames, height, width, dim)
            .permute(0, 1, 4, 2, 3)
            .reshape(batch * frames, dim, height, width)
        )
        feature = self.output_projections[block_index](feature)
        return (
            feature.reshape(batch, frames, dim, height, width)
            .permute(0, 1, 3, 4, 2)
            .reshape(batch, frames * height * width, dim)
        )


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


def wantodance_torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)
    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = wantodance_torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


class WanToDanceInjector(nn.Module):
    def __init__(self, all_modules, all_modules_names, dim=2048, num_heads=32, inject_layer=[0, 27]):
        super().__init__()
        self.injected_block_id = {}
        injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, DiTBlock):
                for inject_id in inject_layer:
                    if f'root.transformer_blocks.{inject_id}' == mod_name:
                        self.injected_block_id[inject_id] = injector_id
                        injector_id += 1

        self.injector = nn.ModuleList(
            [
                CrossAttention(
                    dim=dim,
                    num_heads=num_heads,
                )
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_feat = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6,)
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_vec = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6,)
                for _ in range(injector_id)
            ]
        )


class WanModel(torch.nn.Module):

    _repeated_blocks = ["DiTBlock"]

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
        friction_controlnet: bool = False,
        friction_controlnet_blocks: int = 5,
        friction_controlnet_latent_channels: int = 48,
        restitution_controlnet: bool = False,
        restitution_controlnet_blocks: int = 5,
        restitution_controlnet_latent_channels: int = 48,
        force_controlnet: bool = False,
        force_controlnet_blocks: int = 5,
        force_controlnet_latent_channels: int = 48,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.friction_controlnet = (
            FrictionControlNetBranch(
                latent_channels=friction_controlnet_latent_channels,
                dim=dim,
                num_blocks=min(friction_controlnet_blocks, num_layers),
            )
            if friction_controlnet
            else None
        )
        self.restitution_controlnet = (
            FrictionControlNetBranch(
                latent_channels=restitution_controlnet_latent_channels,
                dim=dim,
                num_blocks=min(restitution_controlnet_blocks, num_layers),
            )
            if restitution_controlnet
            else None
        )
        self.force_controlnet = (
            FrictionControlNetBranch(
                latent_channels=force_controlnet_latent_channels,
                dim=dim,
                num_blocks=min(force_controlnet_blocks, num_layers),
            )
            if force_controlnet
            else None
        )
        # Added only after strict clean-Wan loading by enable_force_conditioning.
        self.force_condition_mode = None
        self.force_condition_channels = 3
        self.force_condition_projection = None
        self.force_condition_attention = nn.ModuleDict()
        self.force_condition_attention_blocks = ()
        # Added only after strict clean-Wan loading by
        # enable_global_physics_cross_attn. Fully independent from the
        # force_condition_* block above (separate weights, separate zero-init,
        # separate freeze bookkeeping) -- see GlobalPhysicsCrossAttn.
        self.global_physics_enabled = False
        self.global_physics_attention = nn.ModuleDict()
        self.global_physics_attention_blocks = ()
        self.global_physics_token_encoder = None
        self.oracle_slot_adapter = None
        self.oracle_slot_enabled = False
        self.oracle_slot_impact_logits = []
        self.oracle_slot_last_impact_logits = None
        self.oracle_slot_last_impact_labels = None
        self.object_time_graph_adapter = None
        self.object_time_graph_enabled = False
        self.object_time_graph_attention_logits = []
        self.object_time_graph_contact_logits = []
        self.object_time_graph_last_attention_logits = None
        self.object_time_graph_last_attention_labels = None
        self.object_time_graph_last_contact_logits = None
        self.object_time_graph_last_contact_labels = None
        self.object_time_graph_last_losses = None
        self.sparse_object_adapter = None
        self.sparse_object_enabled = False
        self.sparse_object_architecture = None
        self.sparse_object_attention_logits = []
        self.sparse_object_reader_outputs = []
        self.sparse_object_writer_outputs = []
        self.sparse_object_reader_gate_outputs = []
        self.sparse_object_writer_gate_outputs = []
        self.sparse_object_write_gates = []
        self.sparse_object_e_pair_event_logits = []
        self.sparse_object_mu_pair_event_logits = []
        self.sparse_object_base_contact_logits = []
        self.sparse_object_pair_attentions = []
        self.sparse_object_pair_qk_raw = []
        self.sparse_object_last_attention_logits = None
        self.sparse_object_last_attention_labels = None
        self.sparse_object_last_losses = None
        # Added only after the clean Wan checkpoint is loaded. This is a
        # paper-faithful reimplementation boundary, not an official release.
        self.phyparam_control_dit = None
        self.phyparam_control_enabled = False
        self.phyparam_last_losses = None
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads

        if wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            end = int(22350 / 8 + 0.5) # 149f * 30fps * 5s = 22350
            self.freqs = precompute_freqs_cis_3d(head_dim, end=end)
        else:
            self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        self.prepare_wantodance(in_dim, dim, num_heads, has_image_pos_emb, out_dim, patch_size, eps,
                                wantodance_enable_music_inject, wantodance_music_inject_layers, wantodance_enable_refimage, wantodance_enable_refface,
                                wantodance_enable_global, wantodance_enable_dynamicfps, wantodance_enable_unimodel)

    def enable_oracle_slot_adapter(
        self,
        injection_blocks=(7, 15, 23),
        slot_dim: int = 256,
        effect_dim: int = 128,
    ) -> None:
        if self.oracle_slot_adapter is None:
            self.oracle_slot_adapter = OracleRoutedSlotAdapter(
                self.dim, injection_blocks, slot_dim, effect_dim
            )
        self.oracle_slot_enabled = True
        reference = next(self.blocks.parameters())
        self.oracle_slot_adapter.to(device=reference.device, dtype=torch.float32)
        self.freeze_base_for_oracle_slot_adapter()

    def freeze_base_for_oracle_slot_adapter(self) -> None:
        if self.oracle_slot_adapter is None:
            return
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("oracle_slot_adapter."))

    def reset_oracle_slot_impact_logits(self) -> None:
        self.oracle_slot_impact_logits = []

    def apply_oracle_slot_adapter(
        self,
        block_index,
        hidden,
        condition,
        timestep,
        frames,
        height,
        width,
    ):
        if (
            condition is None
            or not self.oracle_slot_enabled
            or str(block_index) not in self.oracle_slot_adapter.input_projections
        ):
            return hidden
        hidden, logits = self.oracle_slot_adapter(
            block_index, hidden, condition, timestep, frames, height, width
        )
        self.oracle_slot_impact_logits.append(logits)
        return hidden

    def oracle_slot_impact_loss(
        self,
        labels: torch.Tensor,
        positive_weight: float | None = None,
    ) -> torch.Tensor:
        if not self.oracle_slot_impact_logits:
            raise RuntimeError("oracle slot ImpactHead produced no logits")
        logits = torch.stack(self.oracle_slot_impact_logits, dim=0).mean(dim=0)
        labels = labels.to(device=logits.device, dtype=logits.dtype)
        if labels.ndim == logits.ndim - 1:
            labels = labels.unsqueeze(0)
        if tuple(labels.shape) != tuple(logits.shape):
            raise ValueError(
                f"impact labels {tuple(labels.shape)} != logits {tuple(logits.shape)}"
            )
        self.oracle_slot_last_impact_logits = logits.detach()
        self.oracle_slot_last_impact_labels = labels.detach()
        if positive_weight is None:
            positives = labels.sum()
            negatives = labels.numel() - positives
            positive_weight = (
                float((negatives / positives.clamp_min(1.0)).detach())
                if positives.item() > 0
                else 1.0
            )
        pos_weight = torch.tensor(
            positive_weight, dtype=logits.dtype, device=logits.device
        )
        return F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight
        )

    def enable_object_time_graph_adapter(
        self,
        injection_blocks=(2, 6, 10, 14, 18, 22, 25, 27),
        graph_blocks_per_group: int = 2,
        object_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
    ) -> None:
        if len(self.blocks) != 30:
            raise ValueError(
                "Object-Time Graph v05 is frozen for the 30-block Wan "
                f"backbone, got {len(self.blocks)} blocks"
            )
        if self.object_time_graph_adapter is None:
            self.object_time_graph_adapter = ObjectTimeGraphAdapter(
                wan_dim=self.dim,
                injection_blocks=injection_blocks,
                graph_blocks_per_group=graph_blocks_per_group,
                object_dim=object_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
            )
        self.object_time_graph_enabled = True
        reference = next(self.blocks.parameters())
        self.object_time_graph_adapter.to(
            device=reference.device, dtype=torch.float32
        )
        self.freeze_base_for_object_time_graph_adapter()

    def freeze_base_for_object_time_graph_adapter(self) -> None:
        if self.object_time_graph_adapter is None:
            return
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                name.startswith("object_time_graph_adapter.")
            )

    def reset_object_time_graph_outputs(self) -> None:
        self.object_time_graph_attention_logits = []
        self.object_time_graph_contact_logits = []
        self.object_time_graph_last_attention_logits = None
        self.object_time_graph_last_attention_labels = None
        self.object_time_graph_last_contact_logits = None
        self.object_time_graph_last_contact_labels = None
        self.object_time_graph_last_losses = None

    def apply_object_time_graph_adapter(
        self,
        block_index,
        hidden,
        condition,
        timestep,
        frames,
        height,
        width,
    ):
        if (
            condition is None
            or not self.object_time_graph_enabled
            or str(block_index) not in self.object_time_graph_adapter.groups
        ):
            return hidden
        missing = set(FORWARD_CONDITION_KEYS) - set(condition)
        if missing:
            raise ValueError(
                f"object-time graph condition missing {sorted(missing)}"
            )
        forward_condition = {
            key: condition[key] for key in FORWARD_CONDITION_KEYS
        }
        hidden, attention_logits, contact_logits = (
            self.object_time_graph_adapter(
                block_index,
                hidden,
                forward_condition,
                timestep,
                frames,
                height,
                width,
            )
        )
        self.object_time_graph_attention_logits.append(attention_logits)
        self.object_time_graph_contact_logits.append(contact_logits)
        return hidden

    @staticmethod
    def _balanced_bce_with_logits(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        positives = labels.sum()
        negatives = labels.numel() - positives
        positive_weight = (
            negatives / positives.clamp_min(1.0)
            if positives.item() > 0
            else torch.ones((), device=logits.device, dtype=logits.dtype)
        )
        positive_weight = positive_weight.to(
            device=logits.device, dtype=logits.dtype
        )
        return F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=positive_weight
        )

    def object_time_graph_auxiliary_losses(
        self,
        attention_labels: torch.Tensor,
        contact_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.object_time_graph_attention_logits:
            raise RuntimeError("object-time graph router produced no attention logits")
        if not self.object_time_graph_contact_logits:
            raise RuntimeError("object-time graph ContactHead produced no logits")

        # [groups,B,T,O,H,W]
        attention_logits = torch.stack(
            self.object_time_graph_attention_logits, dim=0
        )
        attention_labels = attention_labels.to(
            device=attention_logits.device, dtype=attention_logits.dtype
        )
        if attention_labels.ndim == 4:
            attention_labels = attention_labels.unsqueeze(0)
        expected_attention = attention_logits.shape[1:]
        if tuple(attention_labels.shape) != tuple(expected_attention):
            raise ValueError(
                f"attention labels {tuple(attention_labels.shape)} != "
                f"{tuple(expected_attention)}"
            )
        expanded_attention_labels = attention_labels.unsqueeze(0).expand_as(
            attention_logits
        )
        # t=0 mask is a forward input. Only future positions measure learned
        # routing, so the localization loss deliberately starts at t=1.
        attention_loss = self._balanced_bce_with_logits(
            attention_logits[:, :, 1:],
            expanded_attention_labels[:, :, 1:],
        )

        # Each group contains exactly two Graph Blocks:
        # [groups,blocks,B,T,1].
        contact_logits = torch.stack(
            self.object_time_graph_contact_logits, dim=0
        )
        contact_labels = contact_labels.to(
            device=contact_logits.device, dtype=contact_logits.dtype
        )
        expected_contact = contact_logits.shape[2:]
        if contact_labels.ndim == 2:
            contact_labels = contact_labels.unsqueeze(0)
        if tuple(contact_labels.shape) != tuple(expected_contact):
            raise ValueError(
                f"contact labels {tuple(contact_labels.shape)} != "
                f"{tuple(expected_contact)}"
            )
        expanded_contact_labels = contact_labels[None, None].expand_as(
            contact_logits
        )
        contact_loss = self._balanced_bce_with_logits(
            contact_logits, expanded_contact_labels
        )

        self.object_time_graph_last_attention_logits = (
            attention_logits.mean(dim=0).detach()
        )
        self.object_time_graph_last_attention_labels = attention_labels.detach()
        self.object_time_graph_last_contact_logits = (
            contact_logits.mean(dim=(0, 1)).detach()
        )
        self.object_time_graph_last_contact_labels = contact_labels.detach()
        return attention_loss, contact_loss

    def enable_phyparam_control_dit(
        self,
        num_control_blocks: int | None = None,
        harmonic_bands: int = 8,
        default_mass_normalized: float = 0.5,
        max_objects: int = 8,
        dino_feature_dim: int = 4096,
        feature_tap_blocks: tuple[int, ...] | None = None,
        enable_feature_supervision: bool = True,
        dino_target_grid: tuple[int, int] = (14, 14),
        temporal_min_norm: float = 1e-6,
    ) -> None:
        conflicting = []
        if self.sparse_object_enabled:
            conflicting.append("sparse_object_adapter")
        if self.object_time_graph_enabled:
            conflicting.append("object_time_graph_adapter")
        if self.oracle_slot_enabled:
            conflicting.append("oracle_slot_adapter")
        if self.force_condition_mode is not None:
            conflicting.append("force_conditioning")
        if self.global_physics_enabled:
            conflicting.append("global_physics_cross_attn")
        if self.friction_controlnet is not None:
            conflicting.append("friction_controlnet")
        if self.restitution_controlnet is not None:
            conflicting.append("restitution_controlnet")
        if self.force_controlnet is not None:
            conflicting.append("force_controlnet")
        if conflicting:
            raise ValueError(
                "PhyParam paper reimplementation must be isolated from other "
                f"conditioning branches, found {conflicting}"
            )
        if self.phyparam_control_dit is None:
            self.phyparam_control_dit = PhyParamControlDiT(
                self.blocks,
                dim=self.dim,
                num_control_blocks=num_control_blocks,
                harmonic_bands=harmonic_bands,
                default_mass_normalized=default_mass_normalized,
                max_objects=max_objects,
                dino_feature_dim=dino_feature_dim,
                feature_tap_blocks=feature_tap_blocks,
                enable_feature_supervision=enable_feature_supervision,
                dino_target_grid=dino_target_grid,
                temporal_min_norm=temporal_min_norm,
            )
        self.phyparam_control_enabled = True
        reference = next(self.blocks.parameters())
        # The formal PhyParam contract uses BF16 throughout.  Follow the
        # already-loaded Wan backbone dtype instead of silently retaining a
        # second 5B parameter copy in FP32 on every data-parallel rank.
        self.phyparam_control_dit.to(device=reference.device, dtype=reference.dtype)
        self.freeze_base_for_phyparam_control_dit()

    def freeze_base_for_phyparam_control_dit(self) -> None:
        if self.phyparam_control_dit is None:
            return
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("phyparam_control_dit."))

    def prepare_phyparam_control(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        condition: Optional[dict],
        freqs: torch.Tensor,
        frames: int,
        height: int,
        width: int,
    ) -> Optional[PhyParamControlState]:
        if condition is None:
            if self.phyparam_control_enabled:
                raise ValueError(
                    "PhyParam control is enabled but sparse_object_condition is missing"
                )
            return None
        if not self.phyparam_control_enabled or self.phyparam_control_dit is None:
            raise ValueError(
                "a PhyParam condition was supplied but the control branch is disabled"
            )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=hidden.is_cuda,
        ):
            return self.phyparam_control_dit.prepare(
                hidden,
                context.to(dtype=hidden.dtype),
                condition,
                freqs,
                frames=frames,
                height=height,
                width=width,
            )

    def apply_phyparam_control_block(
        self,
        block_index: int,
        state: Optional[PhyParamControlState],
        t_mod: torch.Tensor,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> tuple[Optional[PhyParamControlState], Optional[torch.Tensor]]:
        if (
            state is None
            or self.phyparam_control_dit is None
            or block_index >= self.phyparam_control_dit.num_control_blocks
        ):
            return state, None
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=state.hidden.is_cuda,
        ):
            branch_t_mod = t_mod
            if t_mod.ndim == 4 and t_mod.shape[1] == state.video_token_count:
                spatial = state.height * state.width
                mask_tokens = state.hidden.shape[1] - state.video_token_count
                if mask_tokens % spatial != 0:
                    raise ValueError("PhyParam mask-token count is not divisible by the spatial grid")
                objects = mask_tokens // spatial
                mask_t_mod = t_mod[:, :spatial].repeat(1, objects, 1, 1)
                branch_t_mod = torch.cat((t_mod, mask_t_mod), dim=1)
            if branch_t_mod.ndim == 4 and branch_t_mod.shape[1] != state.hidden.shape[1]:
                raise ValueError(
                    "PhyParam separated timestep length does not match video+mask tokens"
                )
            hidden = gradient_checkpoint_forward(
                self.phyparam_control_dit.blocks[block_index],
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                state.hidden,
                state.context,
                branch_t_mod.to(dtype=state.hidden.dtype),
                state.freqs,
                state.token_valid,
                state.cross_allowed,
            )
            residual = self.phyparam_control_dit.output_projections[block_index](
                hidden[:, : state.video_token_count]
            )
            self.phyparam_control_dit.capture_feature(block_index, hidden, state)
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
            residual,
        )

    def phyparam_feature_losses(
        self,
        target: torch.Tensor | dict,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if not self.phyparam_control_enabled or self.phyparam_control_dit is None:
            raise RuntimeError("PhyParam feature loss requested while branch is disabled")
        return self.phyparam_control_dit.feature_losses(target)

    def enable_sparse_object_interaction_adapter(
        self,
        injection_blocks=(2, 6, 10, 14, 18, 22, 25, 27),
        object_dim: int = 512,
        locator_dim: int = 128,
        num_heads: int = 8,
        topk: int = 8,
        architecture: str = "v2",
    ) -> None:
        if self.phyparam_control_enabled:
            raise ValueError(
                "Sparse Object Interaction and PhyParam control cannot be enabled together"
            )
        if len(self.blocks) != 30:
            raise ValueError(
                "Sparse Object Interaction v2 requires the 30-block Wan "
                f"backbone, got {len(self.blocks)} blocks"
            )
        injection_blocks = tuple(int(index) for index in injection_blocks)
        if (
            not injection_blocks
            or len(set(injection_blocks)) != len(injection_blocks)
            or any(index < 0 or index >= len(self.blocks) for index in injection_blocks)
        ):
            raise ValueError(
                "sparse object injection blocks must be non-empty, distinct, "
                f"and inside [0, {len(self.blocks) - 1}]"
            )
        if architecture not in {
            "v2",
            "temporal",
            "temporal_schedule",
            "temporal_no_interaction_schedule",
            "temporal_no_interaction_schedule_mugate",
            "temporal_no_interaction_schedule_mugate_decoupled_writer",
            "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
            "temporal_independent_edge_schedule_decoupled_writer_continuous",
            "temporal_no_interaction_schedule_trackprev_mugate",
            "temporal_no_interaction_schedule_ictr",
            "temporal_pair_schedule",
        }:
            raise ValueError(
                "sparse object architecture must be 'v2', 'temporal', "
                "'temporal_schedule', 'temporal_no_interaction_schedule', "
                "'temporal_no_interaction_schedule_mugate', "
                "'temporal_no_interaction_schedule_mugate_decoupled_writer', "
                "'temporal_no_interaction_schedule_mugate_decoupled_writer_continuous', "
                "'temporal_independent_edge_schedule_decoupled_writer_continuous', "
                "'temporal_no_interaction_schedule_trackprev_mugate', "
                "'temporal_no_interaction_schedule_ictr', "
                "or 'temporal_pair_schedule'"
            )
        if self.sparse_object_adapter is None:
            adapter_class = (
                SparseObjectInteractionTemporalAdapter
                if architecture
                in {
                    "temporal", "temporal_schedule",
                    "temporal_no_interaction_schedule",
                    "temporal_no_interaction_schedule_mugate",
                    "temporal_no_interaction_schedule_mugate_decoupled_writer",
                    "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                    "temporal_independent_edge_schedule_decoupled_writer_continuous",
                    "temporal_no_interaction_schedule_trackprev_mugate",
                    "temporal_no_interaction_schedule_ictr",
                    "temporal_pair_schedule",
                }
                else SparseObjectInteractionAdapter
            )
            adapter_kwargs = {
                "wan_dim": self.dim,
                "injection_blocks": injection_blocks,
                "object_dim": object_dim,
                "locator_dim": locator_dim,
                "num_heads": num_heads,
                "topk": topk,
            }
            if architecture in {
                "temporal",
                "temporal_schedule",
                "temporal_no_interaction_schedule",
                "temporal_no_interaction_schedule_mugate",
                "temporal_no_interaction_schedule_mugate_decoupled_writer",
                "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                "temporal_independent_edge_schedule_decoupled_writer_continuous",
                "temporal_no_interaction_schedule_trackprev_mugate",
                "temporal_no_interaction_schedule_ictr",
                "temporal_pair_schedule",
            }:
                adapter_kwargs["force_schedule"] = architecture in {
                    "temporal_schedule",
                    "temporal_no_interaction_schedule",
                    "temporal_no_interaction_schedule_mugate",
                    "temporal_no_interaction_schedule_mugate_decoupled_writer",
                    "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                    "temporal_independent_edge_schedule_decoupled_writer_continuous",
                    "temporal_no_interaction_schedule_trackprev_mugate",
                    "temporal_no_interaction_schedule_ictr",
                    "temporal_pair_schedule",
                }
                adapter_kwargs["pair_physics"] = (
                    architecture == "temporal_pair_schedule"
                )
                adapter_kwargs["independent_edge"] = architecture == (
                    "temporal_independent_edge_schedule_decoupled_writer_continuous"
                )
                adapter_kwargs["platform_support"] = (
                    architecture in {
                        "temporal_no_interaction_schedule",
                        "temporal_no_interaction_schedule_mugate",
                        "temporal_no_interaction_schedule_mugate_decoupled_writer",
                        "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                        "temporal_no_interaction_schedule_trackprev_mugate",
                        "temporal_no_interaction_schedule_ictr",
                    }
                )
                adapter_kwargs["previous_frame_tracking"] = (
                    architecture in {
                        "temporal_no_interaction_schedule_trackprev_mugate",
                        "temporal_no_interaction_schedule_ictr",
                    }
                )
                adapter_kwargs["joint_template_routing"] = (
                    architecture == "temporal_no_interaction_schedule_ictr"
                )
                adapter_kwargs["mu_message_gate_alpha"] = (
                    1.0
                    if architecture in {
                        "temporal_no_interaction_schedule_mugate",
                        "temporal_no_interaction_schedule_mugate_decoupled_writer",
                        "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                        "temporal_no_interaction_schedule_trackprev_mugate",
                        "temporal_no_interaction_schedule_ictr",
                    }
                    else None
                )
                adapter_kwargs["decoupled_writer"] = architecture in {
                    "temporal_no_interaction_schedule_mugate_decoupled_writer",
                    "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                    "temporal_independent_edge_schedule_decoupled_writer_continuous",
                }
                adapter_kwargs["continuous_object_state"] = (
                    architecture in {
                        "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
                        "temporal_independent_edge_schedule_decoupled_writer_continuous",
                    }
                )
            self.sparse_object_adapter = adapter_class(
                **adapter_kwargs,
            )
            self.sparse_object_architecture = architecture
        elif self.sparse_object_architecture != architecture:
            raise ValueError(
                "sparse object adapter already initialized with architecture "
                f"{self.sparse_object_architecture!r}, requested {architecture!r}"
            )
        self.sparse_object_enabled = True
        reference = next(self.blocks.parameters())
        self.sparse_object_adapter.to(device=reference.device, dtype=torch.float32)
        self.freeze_base_for_sparse_object_adapter()

    def freeze_base_for_sparse_object_adapter(self) -> None:
        if self.sparse_object_adapter is None:
            return
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("sparse_object_adapter."))
        if getattr(self.sparse_object_adapter, "three_contact_gates", False):
            # The relation-QK modules remain only as inactive legacy checkpoint
            # keys.  They are absent from the three-contact forward and must
            # not be handed to DDP as trainable unused parameters.
            for group in self.sparse_object_adapter.groups.values():
                group.interaction.query_projection.requires_grad_(False)
                group.interaction.key_projection.requires_grad_(False)
        # Construct all original modules before freezing so retained parameters
        # receive exactly the same seeded initialization as the Full model.
        for group in self.sparse_object_adapter.groups.values():
            temporal = getattr(group, "temporal", None)
            if temporal is not None and not getattr(temporal, "attention_enabled", True):
                temporal.attention_norm.requires_grad_(False)
                temporal.attention.requires_grad_(False)
            if not getattr(self.sparse_object_adapter, "state_carry_enabled", True):
                for name in ("previous_state_norm", "current_read_norm", "state_fusion_gate"):
                    module = getattr(group, name, None)
                    if module is not None:
                        module.requires_grad_(False)

    def reset_sparse_object_outputs(self) -> None:
        self.sparse_object_attention_logits = []
        self.sparse_object_reader_outputs = []
        self.sparse_object_writer_outputs = []
        self.sparse_object_reader_gate_outputs = []
        self.sparse_object_writer_gate_outputs = []
        self.sparse_object_write_gates = []
        self.sparse_object_e_pair_event_logits = []
        self.sparse_object_mu_pair_event_logits = []
        self.sparse_object_base_contact_logits = []
        self.sparse_object_pair_attentions = []
        self.sparse_object_pair_qk_raw = []
        self.sparse_object_last_attention_logits = None
        self.sparse_object_last_attention_labels = None
        self.sparse_object_last_losses = None
        if (
            self.sparse_object_adapter is not None
            and hasattr(self.sparse_object_adapter, "reset_continuous_state")
        ):
            self.sparse_object_adapter.reset_continuous_state()

    def apply_sparse_object_adapter(
        self,
        block_index,
        hidden,
        condition,
        timestep,
        frames,
        height,
        width,
    ):
        if (
            condition is None
            or not self.sparse_object_enabled
            or str(block_index) not in self.sparse_object_adapter.groups
        ):
            return hidden
        if self.sparse_object_architecture == "temporal_pair_schedule":
            forward_keys = SPARSE_OBJECT_TEMPORAL_PAIR_SCHEDULE_FORWARD_KEYS
        elif self.sparse_object_architecture == (
            "temporal_independent_edge_schedule_decoupled_writer_continuous"
        ):
            forward_keys = (
                SPARSE_OBJECT_TEMPORAL_INDEPENDENT_EDGE_SCHEDULE_FORWARD_KEYS
            )
        elif self.sparse_object_architecture in {
            "temporal_no_interaction_schedule",
            "temporal_no_interaction_schedule_mugate",
            "temporal_no_interaction_schedule_mugate_decoupled_writer",
            "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
            "temporal_no_interaction_schedule_trackprev_mugate",
            "temporal_no_interaction_schedule_ictr",
        }:
            forward_keys = SPARSE_OBJECT_TEMPORAL_NO_INTERACTION_SCHEDULE_FORWARD_KEYS
        elif self.sparse_object_architecture == "temporal_schedule":
            forward_keys = SPARSE_OBJECT_TEMPORAL_SCHEDULE_FORWARD_KEYS
        elif self.sparse_object_architecture == "temporal":
            forward_keys = SPARSE_OBJECT_TEMPORAL_FORWARD_KEYS
        else:
            forward_keys = SPARSE_OBJECT_FORWARD_KEYS
        if os.environ.get("PHYSICAL_WM_OBJECT_MASS", "0") == "1":
            forward_keys = forward_keys + ("mass", "mass_present")
        missing = set(forward_keys) - set(condition)
        if missing:
            raise ValueError(f"sparse object condition missing {sorted(missing)}")
        forward_condition = {
            key: condition[key] for key in forward_keys
        }
        if self.sparse_object_architecture == (
            "temporal_independent_edge_schedule_decoupled_writer_continuous"
        ):
            if "contract" not in condition:
                raise ValueError("independent-edge condition missing contract")
            forward_condition["contract"] = condition["contract"]
        hidden, attention_logits = self.sparse_object_adapter(
            block_index,
            hidden,
            forward_condition,
            timestep,
            frames,
            height,
            width,
        )
        self.sparse_object_attention_logits.append(attention_logits)
        group = self.sparse_object_adapter.groups[str(block_index)]
        reader_indices = getattr(group, "last_reader_indices", None)
        if reader_indices is not None:
            self.sparse_object_reader_outputs.append(
                (
                    reader_indices,
                    getattr(group, "last_reader_attention", None),
                    getattr(group, "last_reader_grid_shape", None),
                )
            )
        reader_gate_logits = getattr(
            group, "last_reader_support_gate_logits", None
        )
        if reader_gate_logits is not None:
            self.sparse_object_reader_gate_outputs.append(
                (
                    reader_indices,
                    reader_gate_logits,
                    group.last_reader_support_gate_mask,
                    group.last_reader_attention,
                    group.last_reader_grid_shape,
                    torch.ones_like(
                        group.last_reader_support_gate_mask,
                        dtype=torch.bool,
                    ),
                )
            )
        if group.last_writer_logits is not None:
            self.sparse_object_writer_outputs.append(
                (
                    group.last_writer_logits,
                    group.last_writer_indices,
                    group.last_writer_attention,
                    group.last_writer_seed_indices,
                )
            )
        writer_gate_logits = getattr(
            group, "last_writer_support_gate_logits", None
        )
        if writer_gate_logits is not None:
            self.sparse_object_writer_gate_outputs.append(
                (
                    group.last_writer_indices,
                    writer_gate_logits,
                    group.last_writer_support_gate_mask,
                    group.last_writer_attention,
                    (height, width),
                    group.last_writer_support_gate_allowed_mask,
                )
            )
        if group.last_write_gate is not None:
            self.sparse_object_write_gates.append(group.last_write_gate)
        interaction = getattr(group, "interaction", None)
        if getattr(interaction, "last_e_pair_event_logit", None) is not None:
            batch = hidden.shape[0]
            nodes = interaction.last_e_pair_event_logit.shape[-1]
            expected = (batch * frames, nodes, nodes)
            if tuple(interaction.last_e_pair_event_logit.shape) != expected:
                raise RuntimeError("pair-event logits disagree with [B*T,N+1,N+1]")
            self.sparse_object_e_pair_event_logits.append(
                interaction.last_e_pair_event_logit.reshape(batch, frames, nodes, nodes)
            )
            self.sparse_object_mu_pair_event_logits.append(
                interaction.last_mu_pair_event_logit.reshape(batch, frames, nodes, nodes)
            )
            if getattr(interaction, "last_base_contact_logit", None) is not None:
                self.sparse_object_base_contact_logits.append(
                    interaction.last_base_contact_logit.reshape(
                        batch, frames, nodes, nodes
                    )
                )
        if self.sparse_object_architecture in {
            "temporal_pair_schedule",
            "temporal_no_interaction_schedule",
            "temporal_no_interaction_schedule_mugate",
            "temporal_no_interaction_schedule_mugate_decoupled_writer",
            "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
            "temporal_no_interaction_schedule_trackprev_mugate",
            "temporal_no_interaction_schedule_ictr",
        }:
            pair_attention = self.sparse_object_adapter.groups[
                str(block_index)
            ].interaction.last_attention
            if pair_attention is not None:
                self.sparse_object_pair_attentions.append(pair_attention)
            pair_qk_raw = self.sparse_object_adapter.groups[
                str(block_index)
            ].interaction.last_qk_raw
            if pair_qk_raw is not None:
                self.sparse_object_pair_qk_raw.append(pair_qk_raw)
        return hidden

    def sparse_object_pair_event_losses(
        self,
        e_labels: torch.Tensor,
        e_valid: torch.Tensor,
        mu_labels: torch.Tensor,
        mu_valid: torch.Tensor,
        object_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Loss-only supervision for typed symmetric pair event gates.

        ``hard_slot`` reproduces exact simulator pair/time BCE.
        ``event_distribution_w1`` supervises one block-averaged probability
        distribution over legal pair/time events plus a null event with a
        single temporal-distance objective. ``weak_pair``
        keeps the contact slot latent per block. ``weak_pair_crossblock_union``
        regularizes the joint temporal support of all Reader blocks.
        ``coarse_window_pair`` reveals only a clipped +/-1-slot window.
        """
        if not self.sparse_object_e_pair_event_logits:
            raise RuntimeError("pair-event route produced no e/mu logits")
        full_reference = self.sparse_object_e_pair_event_logits[0]
        batch, times, nodes, _ = full_reference.shape
        objects = nodes - 1

        def batched(value: torch.Tensor, ndim: int) -> torch.Tensor:
            value = value.to(device=full_reference.device, dtype=torch.float32)
            return value.unsqueeze(0) if value.ndim == ndim - 1 else value

        e_target = batched(e_labels, 4)
        e_mask = batched(e_valid, 4) > 0.5
        mu_target = batched(mu_labels, 4)
        mu_mask = batched(mu_valid, 4) > 0.5
        valid_objects = batched(object_valid_mask, 2) > 0.5
        expected = (batch, times, objects, nodes)
        for name, value in {
            "e_pair_event_labels": e_target,
            "e_pair_event_valid": e_mask,
            "mu_pair_event_labels": mu_target,
            "mu_pair_event_valid": mu_mask,
        }.items():
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must be {expected}, got {tuple(value.shape)}")
        if tuple(valid_objects.shape) != (batch, objects):
            raise ValueError("object_valid_mask disagrees with pair labels")
        partner_valid = torch.cat(
            (valid_objects, torch.ones(batch, 1, dtype=torch.bool, device=full_reference.device)),
            dim=1,
        )
        pair_valid = valid_objects[:, None, :, None] & partner_valid[:, None, None, :]
        diagonal = torch.eye(objects, nodes, dtype=torch.bool, device=full_reference.device)
        pair_valid = pair_valid & ~diagonal[None, None]
        object_object_upper = torch.zeros(objects, nodes, dtype=torch.bool, device=full_reference.device)
        object_object_upper[:, :objects] = torch.triu(
            torch.ones(objects, objects, dtype=torch.bool, device=full_reference.device),
            diagonal=1,
        )
        object_table = torch.zeros(objects, nodes, dtype=torch.bool, device=full_reference.device)
        object_table[:, objects] = True
        e_object_labels = e_target[:, :, :, :objects]
        e_object_valid = batched(e_valid, 4)[:, :, :, :objects]
        if not torch.equal(e_object_labels, e_object_labels.transpose(-1, -2)):
            raise ValueError("e pair labels must be symmetric")
        if not torch.equal(e_object_valid, e_object_valid.transpose(-1, -2)):
            raise ValueError("e pair valid mask must be symmetric")
        if torch.any(torch.diagonal(e_object_valid, dim1=-2, dim2=-1)):
            raise ValueError("e pair diagonal must be invalid")
        # Restitution is meaningful on both object-object and object-support
        # edges.  Object-object entries are symmetric and counted once below;
        # support has no receiver row, so each object-support edge has one
        # supervised entry in the final partner column.
        typed_e_valid = pair_valid
        typed_mu_valid = pair_valid & object_table[None, None]
        if torch.any(e_target.masked_select(~typed_e_valid) != 0):
            raise ValueError(
                "e pair positives exist outside object-object or object-support pairs"
            )
        if torch.any(mu_target.masked_select(~typed_mu_valid) != 0):
            raise ValueError("mu pair positives exist outside object-table pairs")
        if torch.any(e_target > batched(e_valid, 4).to(e_target.dtype)):
            raise ValueError("e pair positives must be marked valid")
        if torch.any(mu_target > batched(mu_valid, 4).to(mu_target.dtype)):
            raise ValueError("mu pair positives must be marked valid")
        e_mask = e_mask & pair_valid & (
            object_object_upper[None, None] | object_table[None, None]
        )
        mu_mask = mu_mask & pair_valid & object_table[None, None]

        def loss_for(logits_list, target, mask):
            if not mask.any():
                # General-video and F-only rows intentionally carry no e/mu
                # answer.  A microbatch containing only such rows must skip
                # the typed event BCE while leaving every other training loss
                # active.  The scalar stays connected to the event heads so
                # DDP sees a stable graph with exactly zero gradient.
                return sum(full_logits.float().sum() * 0.0 for full_logits in logits_list)
            losses = []
            for full_logits in logits_list:
                if tuple(full_logits.shape) != (batch, times, nodes, nodes):
                    raise ValueError("pair-event heads disagree across Reader blocks")
                if not torch.allclose(full_logits, full_logits.transpose(-1, -2), atol=1e-5):
                    raise ValueError("pair-event logits must be symmetric")
                logits = full_logits[:, :, :objects, :]
                item = F.binary_cross_entropy_with_logits(
                    logits.float(), target, reduction="none"
                )
                positive_mask = mask & (target > 0.5)
                negative_mask = mask & ~positive_mask
                class_losses = []
                if positive_mask.any():
                    class_losses.append(item[positive_mask].mean())
                if negative_mask.any():
                    class_losses.append(item[negative_mask].mean())
                losses.append(torch.stack(class_losses).mean())
            return torch.stack(losses).mean()

        e_mode = getattr(
            getattr(self, "sparse_object_adapter", None),
            "e_pair_event_loss_mode",
            "hard_slot",
        )
        if e_mode == "hard_slot":
            e_loss = loss_for(
                self.sparse_object_e_pair_event_logits, e_target, e_mask
            )
            weak_metrics = {}
        elif e_mode == "event_distribution_w1":
            # One probability distribution per supervised sample over every
            # legal undirected pair/time event and one fixed-logit null event.
            # The exact first-contact slot remains the sole zero-cost event;
            # neighbouring slots are still wrong, but their cost grows with
            # temporal distance.  Averaging logits before the softmax avoids
            # imposing the same answer independently on all Reader blocks.
            if len(self.sparse_object_e_pair_event_logits) != 8:
                raise ValueError(
                    "event-distribution supervision requires exactly eight Reader blocks"
                )
            supervised = e_mask.flatten(1).any(dim=1)
            if not supervised.any():
                e_loss = sum(
                    full_logits.float().sum() * 0.0
                    for full_logits in self.sparse_object_e_pair_event_logits
                )
                weak_metrics = {
                    "e_pair_event_distribution_expected_distance": (
                        full_reference.new_zeros(())
                    ),
                    "e_pair_event_distribution_target_probability": (
                        full_reference.new_zeros(())
                    ),
                    "e_pair_event_distribution_null_probability": (
                        full_reference.new_zeros(())
                    ),
                    "e_pair_event_distribution_supervised_fraction": (
                        full_reference.new_zeros(())
                    ),
                }
            else:
                block_logits = []
                for full_logits in self.sparse_object_e_pair_event_logits:
                    if tuple(full_logits.shape) != (batch, times, nodes, nodes):
                        raise ValueError("pair-event heads disagree across Reader blocks")
                    if not torch.allclose(
                        full_logits, full_logits.transpose(-1, -2), atol=1e-5
                    ):
                        raise ValueError("pair-event logits must be symmetric")
                    block_logits.append(full_logits.float()[:, :, :objects, :])
                mean_logits = torch.stack(block_logits).mean(dim=0)[supervised]
                event_mask = e_mask[supervised]
                positive = (e_target[supervised] > 0.5) & event_mask
                positive_count = positive.flatten(1).sum(dim=1)
                if torch.any(positive_count > 1):
                    raise ValueError(
                        "event-distribution target must contain at most one first-contact event"
                    )
                hit = positive_count == 1

                flat_logits = mean_logits.flatten(1).masked_fill(
                    ~event_mask.flatten(1), -torch.inf
                )
                # Anchor null at the same logit as H0's frozen fresh-head
                # initialization (4.0).  Forward initialization therefore
                # remains exactly matched, while Hit and Miss samples both
                # start with useful event-vs-null gradients.
                null_logits = torch.full(
                    (flat_logits.shape[0], 1),
                    4.0,
                    dtype=flat_logits.dtype,
                    device=flat_logits.device,
                )
                probability = torch.softmax(
                    torch.cat((flat_logits, null_logits), dim=1), dim=1
                )

                # Every event costs one by default.  For Hit samples only the
                # correct pair receives the normalized |t-t*| temporal cost;
                # wrong pairs and null retain unit cost.  For Miss samples all
                # events cost one and null costs zero.
                event_cost = torch.ones_like(mean_logits)
                if hit.any():
                    positive_index = positive.flatten(1).float().argmax(dim=1)
                    target_time = positive_index // (objects * nodes)
                    target_pair_index = positive_index % (objects * nodes)
                    target_left = target_pair_index // nodes
                    target_right = target_pair_index % nodes
                    time_index = torch.arange(
                        times, device=mean_logits.device
                    ).view(1, times, 1, 1)
                    left_index = torch.arange(
                        objects, device=mean_logits.device
                    ).view(1, 1, objects, 1)
                    right_index = torch.arange(
                        nodes, device=mean_logits.device
                    ).view(1, 1, 1, nodes)
                    same_pair = (
                        (left_index == target_left[:, None, None, None])
                        & (right_index == target_right[:, None, None, None])
                        & hit[:, None, None, None]
                    )
                    temporal_cost = (
                        time_index - target_time[:, None, None, None]
                    ).abs().to(mean_logits.dtype) / max(times - 1, 1)
                    event_cost = torch.where(
                        same_pair, temporal_cost, event_cost
                    )
                flat_cost = event_cost.flatten(1)
                null_cost = hit.to(flat_cost.dtype).unsqueeze(1)
                joint_cost = torch.cat((flat_cost, null_cost), dim=1)
                expected_distance = (probability * joint_cost).sum(dim=1)
                # Unlabelled rows have an exact graph-connected zero and are
                # included in the frozen full-microbatch mean.  This keeps the
                # per-row definition literal under mixed labelled/unlabelled
                # batches and prevents rank-local composition from silently
                # changing the reduction contract.
                e_loss = expected_distance.sum() / batch

                target_probability = probability.new_zeros(probability.shape[0])
                if hit.any():
                    target_probability[hit] = probability[
                        hit, positive.flatten(1)[hit].float().argmax(dim=1)
                    ]
                weak_metrics = {
                    "e_pair_event_distribution_expected_distance": (
                        expected_distance.detach().mean()
                    ),
                    "e_pair_event_distribution_target_probability": (
                        target_probability[hit].detach().mean()
                        if hit.any() else probability.new_zeros(())
                    ),
                    "e_pair_event_distribution_null_probability": (
                        probability[:, -1].detach().mean()
                    ),
                    "e_pair_event_distribution_supervised_fraction": (
                        supervised.float().detach().mean()
                    ),
                }
        elif e_mode == "weak_pair":
            # One binary bag label per undirected pair: whether that pair
            # contacted at any of the 13 slots.  Exact simulator time labels
            # are collapsed before comparison and never enter model forward.
            pair_target = (e_target > 0.5).any(dim=1)
            pair_mask = e_mask.any(dim=1)
            if not pair_mask.any():
                raise ValueError("weak pair-event supervision has no valid pairs")
            block_losses = []
            block_presence = []
            block_mass = []
            block_tv = []
            for full_logits in self.sparse_object_e_pair_event_logits:
                if tuple(full_logits.shape) != (batch, times, nodes, nodes):
                    raise ValueError("pair-event heads disagree across Reader blocks")
                if not torch.allclose(
                    full_logits, full_logits.transpose(-1, -2), atol=1e-5
                ):
                    raise ValueError("pair-event logits must be symmetric")
                logits = full_logits.float()[:, :, :objects, :]
                selected_logits = logits.permute(0, 2, 3, 1)[pair_mask]
                selected = torch.sigmoid(selected_logits)
                selected_target = pair_target[pair_mask]
                # Stable noisy-OR bag BCE over exactly 13 time instances.
                # Keep the negative bag in log space: probability-space OR
                # rounds to one for modest positive logits and would then
                # clamp away the all-Miss gradient.
                log_no_event = F.logsigmoid(-selected_logits).sum(dim=-1)
                positive_item = -torch.log(-torch.expm1(log_no_event))
                negative_item = -log_no_event
                item = torch.where(
                    selected_target, positive_item, negative_item
                )
                positive = selected_target
                negative = ~positive
                classes = []
                if positive.any():
                    classes.append(item[positive].mean())
                if negative.any():
                    classes.append(item[negative].mean())
                presence = torch.stack(classes).mean()
                if positive.any():
                    positive_gates = selected[positive]
                    mass = positive_gates.sum(dim=-1).mean()
                    padded = F.pad(positive_gates, (1, 1), value=0.0)
                    tv = (padded[:, 1:] - padded[:, :-1]).abs().sum(dim=-1).mean()
                else:
                    mass = presence.new_zeros(())
                    tv = presence.new_zeros(())
                block_losses.append(presence + 0.15 * mass + 0.03 * tv)
                block_presence.append(presence)
                block_mass.append(mass)
                block_tv.append(tv)
            global_step = int(getattr(self, "sparse_object_global_step", 0))
            regularizer_scale = min(1.0, max(0.0, global_step / 600.0))
            presence_mean = torch.stack(block_presence).mean()
            mass_mean = torch.stack(block_mass).mean()
            tv_mean = torch.stack(block_tv).mean()
            e_loss = (
                torch.stack(block_losses).mean()
                if regularizer_scale == 1.0
                else presence_mean + regularizer_scale * (
                    0.15 * mass_mean + 0.03 * tv_mean
                )
            )
            weak_metrics = {
                "e_pair_event_weak_presence": presence_mean.detach(),
                "e_pair_event_weak_positive_mass": mass_mean.detach(),
                "e_pair_event_weak_positive_tv": tv_mean.detach(),
                "e_pair_event_weak_regularizer_scale": full_reference.new_tensor(
                    regularizer_scale
                ),
                "e_pair_event_weak_block_count": full_reference.new_tensor(
                    len(block_losses)
                ),
                "e_pair_event_weak_positive_pair_fraction": pair_target[
                    pair_mask
                ].float().mean().detach(),
                "e_pair_event_weak_crossblock_union": full_reference.new_zeros(()),
                "e_pair_event_coarse_window": full_reference.new_zeros(()),
            }
        elif e_mode == "weak_pair_crossblock_union":
            # Joint support across all blocks: spreading different block peaks
            # over time increases union mass, while multiple blocks may still
            # cooperate inside the same short window.
            pair_target = (e_target > 0.5).any(dim=1)
            pair_mask = e_mask.any(dim=1)
            if not pair_mask.any():
                raise ValueError("cross-block weak supervision has no valid pairs")
            selected_logits = []
            for full_logits in self.sparse_object_e_pair_event_logits:
                if tuple(full_logits.shape) != (batch, times, nodes, nodes):
                    raise ValueError("pair-event heads disagree across Reader blocks")
                if not torch.allclose(
                    full_logits, full_logits.transpose(-1, -2), atol=1e-5
                ):
                    raise ValueError("pair-event logits must be symmetric")
                logits = full_logits.float()[:, :, :objects, :]
                selected_logits.append(logits.permute(0, 2, 3, 1)[pair_mask])
            # [blocks, valid_pairs, time]. All OR operations stay in log space.
            selected_logits = torch.stack(selected_logits)
            selected_target = pair_target[pair_mask]
            log_no_event_by_time = F.logsigmoid(-selected_logits).sum(dim=0)
            log_no_event = log_no_event_by_time.sum(dim=-1)
            item = torch.where(
                selected_target,
                -torch.log(-torch.expm1(log_no_event)),
                -log_no_event,
            )
            classes = []
            if selected_target.any():
                classes.append(item[selected_target].mean())
            if (~selected_target).any():
                classes.append(item[~selected_target].mean())
            presence = torch.stack(classes).mean()
            if selected_target.any():
                union_gate = -torch.expm1(log_no_event_by_time)
                positive_gate = union_gate[selected_target]
                mass = positive_gate.sum(dim=-1).mean()
                padded = F.pad(positive_gate, (1, 1), value=0.0)
                tv = (padded[:, 1:] - padded[:, :-1]).abs().sum(dim=-1).mean()
            else:
                mass = presence.new_zeros(())
                tv = presence.new_zeros(())
            global_step = int(getattr(self, "sparse_object_global_step", 0))
            regularizer_scale = min(1.0, max(0.0, global_step / 600.0))
            e_loss = presence + regularizer_scale * (0.15 * mass + 0.03 * tv)
            weak_metrics = {
                "e_pair_event_weak_presence": presence.detach(),
                "e_pair_event_weak_positive_mass": mass.detach(),
                "e_pair_event_weak_positive_tv": tv.detach(),
                "e_pair_event_weak_regularizer_scale": full_reference.new_tensor(
                    regularizer_scale
                ),
                "e_pair_event_weak_block_count": full_reference.new_tensor(
                    len(self.sparse_object_e_pair_event_logits)
                ),
                "e_pair_event_weak_positive_pair_fraction": selected_target.float()
                .mean()
                .detach(),
                "e_pair_event_weak_crossblock_union": full_reference.new_ones(()),
                "e_pair_event_coarse_window": full_reference.new_zeros(()),
            }
        elif e_mode == "coarse_window_pair":
            # Loss-only +/-1-slot MIL window.  The precise contact slot is not
            # required to open; any one location inside the window may satisfy
            # the positive bag, while outside locations are explicitly closed.
            pair_target = (e_target > 0.5).any(dim=1)
            pair_mask = e_mask.any(dim=1)
            if not pair_mask.any():
                raise ValueError("coarse-window supervision has no valid pairs")
            exact = (e_target > 0.5).permute(0, 2, 3, 1)[pair_mask]
            selected_target = pair_target[pair_mask]
            positive_counts = exact.sum(dim=-1)
            if torch.any(positive_counts[selected_target] != 1):
                raise ValueError("coarse-window positive pair must have one first-contact slot")
            window = F.max_pool1d(
                exact.float().unsqueeze(1), kernel_size=3, stride=1, padding=1
            ).squeeze(1) > 0.5
            block_losses = []
            for full_logits in self.sparse_object_e_pair_event_logits:
                if tuple(full_logits.shape) != (batch, times, nodes, nodes):
                    raise ValueError("pair-event heads disagree across Reader blocks")
                if not torch.allclose(
                    full_logits, full_logits.transpose(-1, -2), atol=1e-5
                ):
                    raise ValueError("pair-event logits must be symmetric")
                logits = full_logits.float()[:, :, :objects, :]
                selected_logits = logits.permute(0, 2, 3, 1)[pair_mask]
                classes = []
                if selected_target.any():
                    positive_logits = selected_logits[selected_target]
                    positive_window = window[selected_target]
                    log_no_inside = (
                        F.logsigmoid(-positive_logits)
                        * positive_window.to(positive_logits.dtype)
                    ).sum(dim=-1)
                    inside = -torch.log(-torch.expm1(log_no_inside))
                    outside_mask = ~positive_window
                    outside = (
                        F.softplus(positive_logits)
                        * outside_mask.to(positive_logits.dtype)
                    ).sum(dim=-1) / outside_mask.sum(dim=-1).clamp_min(1)
                    classes.append((0.5 * (inside + outside)).mean())
                if (~selected_target).any():
                    classes.append(
                        F.softplus(selected_logits[~selected_target])
                        .mean(dim=-1)
                        .mean()
                    )
                block_losses.append(torch.stack(classes).mean())
            e_loss = torch.stack(block_losses).mean()
            weak_metrics = {
                "e_pair_event_weak_presence": e_loss.detach(),
                "e_pair_event_weak_positive_mass": full_reference.new_zeros(()),
                "e_pair_event_weak_positive_tv": full_reference.new_zeros(()),
                "e_pair_event_weak_regularizer_scale": full_reference.new_zeros(()),
                "e_pair_event_weak_block_count": full_reference.new_tensor(
                    len(block_losses)
                ),
                "e_pair_event_weak_positive_pair_fraction": selected_target.float()
                .mean()
                .detach(),
                "e_pair_event_weak_crossblock_union": full_reference.new_zeros(()),
                "e_pair_event_coarse_window": full_reference.new_ones(()),
                "e_pair_event_coarse_window_radius": full_reference.new_tensor(1.0),
            }
        else:
            raise ValueError(f"unsupported e pair event loss mode {e_mode!r}")
        mu_loss = loss_for(self.sparse_object_mu_pair_event_logits, mu_target, mu_mask)
        mean_e = torch.stack(self.sparse_object_e_pair_event_logits).mean(0)[:, :, :objects]
        mean_mu = torch.stack(self.sparse_object_mu_pair_event_logits).mean(0)[:, :, :objects]

        def masked_mean_or_zero(value, mask):
            if mask.any():
                return value[mask].mean().detach()
            return full_reference.new_zeros(())

        metrics = {
            "e_pair_event_positive_fraction": masked_mean_or_zero(e_target, e_mask),
            "mu_pair_event_positive_fraction": masked_mean_or_zero(mu_target, mu_mask),
            "e_pair_event_gate_mean": masked_mean_or_zero(
                torch.sigmoid(mean_e.float()), e_mask
            ),
            "mu_pair_event_gate_mean": masked_mean_or_zero(
                torch.sigmoid(mean_mu.float()), mu_mask
            ),
            "e_pair_event_valid_count": e_mask.sum().detach(),
            "mu_pair_event_valid_count": mu_mask.sum().detach(),
            "e_pair_event_loss_mode_weak": full_reference.new_tensor(
                float(e_mode in {"weak_pair", "weak_pair_crossblock_union"})
            ),
            "e_pair_event_loss_mode_crossblock_union": full_reference.new_tensor(
                float(e_mode == "weak_pair_crossblock_union")
            ),
            "e_pair_event_loss_mode_coarse_window": full_reference.new_tensor(
                float(e_mode == "coarse_window_pair")
            ),
            "e_pair_event_loss_mode_distribution_w1": full_reference.new_tensor(
                float(e_mode == "event_distribution_w1")
            ),
            **weak_metrics,
        }
        return e_loss, mu_loss, metrics

    def sparse_object_contact_loss(
        self,
        contact_labels: torch.Tensor,
        contact_valid: torch.Tensor,
        object_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Hard loss-only supervision for one shared OO/OS contact timeline."""
        if not self.sparse_object_e_pair_event_logits:
            raise RuntimeError("shared-contact route produced no contact logits")
        reference = self.sparse_object_e_pair_event_logits[0]
        batch, times, nodes, _ = reference.shape
        objects = nodes - 1

        def batched(value: torch.Tensor, ndim: int) -> torch.Tensor:
            value = value.to(device=reference.device, dtype=torch.float32)
            return value.unsqueeze(0) if value.ndim == ndim - 1 else value

        target = batched(contact_labels, 4)
        valid = batched(contact_valid, 4) > 0.5
        valid_objects = batched(object_valid_mask, 2) > 0.5
        expected = (batch, times, objects, nodes)
        if tuple(target.shape) != expected or tuple(valid.shape) != expected:
            raise ValueError(
                f"contact_label/contact_valid must be {expected}, got "
                f"{tuple(target.shape)} and {tuple(valid.shape)}"
            )
        if tuple(valid_objects.shape) != (batch, objects):
            raise ValueError("object_valid_mask disagrees with contact labels")
        oo_target = target[..., :objects]
        oo_valid = valid[..., :objects]
        if not torch.equal(oo_target, oo_target.transpose(-1, -2)):
            raise ValueError("object-object contact_label must be symmetric")
        if not torch.equal(oo_valid, oo_valid.transpose(-1, -2)):
            raise ValueError("object-object contact_valid must be symmetric")
        if torch.any(torch.diagonal(oo_valid, dim1=-2, dim2=-1)):
            raise ValueError("self contact must be invalid")
        if torch.any(target > valid.to(target.dtype)):
            raise ValueError("positive contact must be marked valid")

        partner_valid = torch.cat(
            (
                valid_objects,
                torch.ones(
                    batch, 1, dtype=torch.bool, device=reference.device
                ),
            ),
            dim=1,
        )
        pair_valid = (
            valid_objects[:, None, :, None]
            & partner_valid[:, None, None, :]
        )
        upper_or_support = torch.zeros(
            objects, nodes, dtype=torch.bool, device=reference.device
        )
        upper_or_support[:, :objects] = torch.triu(
            torch.ones(
                objects, objects, dtype=torch.bool, device=reference.device
            ),
            diagonal=1,
        )
        upper_or_support[:, objects] = True
        mask = valid & pair_valid & upper_or_support[None, None]

        losses = []
        for full_logits in self.sparse_object_e_pair_event_logits:
            if tuple(full_logits.shape) != (batch, times, nodes, nodes):
                raise ValueError("contact heads disagree across Reader blocks")
            if not torch.allclose(
                full_logits, full_logits.transpose(-1, -2), atol=1e-5
            ):
                raise ValueError("contact logits must be symmetric")
            if not mask.any():
                losses.append(full_logits.float().sum() * 0.0)
                continue
            item = F.binary_cross_entropy_with_logits(
                full_logits.float()[:, :, :objects, :],
                target,
                reduction="none",
            )
            positive = mask & (target > 0.5)
            negative = mask & ~positive
            classes = []
            if positive.any():
                classes.append(item[positive].mean())
            if negative.any():
                classes.append(item[negative].mean())
            losses.append(torch.stack(classes).mean())
        loss = torch.stack(losses).mean()
        mean_gate = torch.sigmoid(
            torch.stack(self.sparse_object_e_pair_event_logits).mean(0).float()
        )[:, :, :objects, :]
        zero = reference.new_zeros(())
        metrics = {
            "contact_positive_fraction": (
                target[mask].mean().detach() if mask.any() else zero
            ),
            "contact_gate_mean": (
                mean_gate[mask].mean().detach() if mask.any() else zero
            ),
            "contact_valid_count": mask.sum().detach(),
            "contact_block_count": reference.new_tensor(
                len(self.sparse_object_e_pair_event_logits)
            ),
        }
        return loss, metrics

    def sparse_object_three_contact_losses(
        self,
        contact_labels: torch.Tensor,
        contact_valid: torch.Tensor,
        e_contact_labels: torch.Tensor,
        e_contact_valid: torch.Tensor,
        mu_contact_labels: torch.Tensor,
        mu_contact_valid: torch.Tensor,
        object_valid_mask: torch.Tensor,
        edge_restitution_present: torch.Tensor,
        edge_mu_present: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Loss-only supervision for independent base/e/mu contact heads."""
        if not self.sparse_object_base_contact_logits:
            raise RuntimeError("three-contact route produced no base logits")
        reference = self.sparse_object_base_contact_logits[0]
        batch, times, nodes, _ = reference.shape
        objects = nodes - 1

        def batched(value: torch.Tensor, ndim: int, *, boolean: bool = False):
            value = value.to(device=reference.device)
            if value.ndim == ndim - 1:
                value = value.unsqueeze(0)
            return value > 0.5 if boolean else value.float()

        targets = {
            "base": batched(contact_labels, 4),
            "e": batched(e_contact_labels, 4),
            "mu": batched(mu_contact_labels, 4),
        }
        valid_by_head = {
            "base": batched(contact_valid, 4, boolean=True),
            "e": batched(e_contact_valid, 4, boolean=True),
            "mu": batched(mu_contact_valid, 4, boolean=True),
        }
        valid_objects = batched(object_valid_mask, 2, boolean=True)
        e_present = batched(edge_restitution_present, 3, boolean=True)
        mu_present = batched(edge_mu_present, 3, boolean=True)
        expected = (batch, times, objects, nodes)
        for name in ("base", "e", "mu"):
            target = targets[name]
            valid = valid_by_head[name]
            if tuple(target.shape) != expected or tuple(valid.shape) != expected:
                raise ValueError(
                    f"{name} contact labels disagree with [B,T,N,N+1]"
                )
            if not torch.equal(
                target[..., :objects], target[..., :objects].transpose(-1, -2)
            ):
                raise ValueError(f"object-object {name} contact_label must be symmetric")
            if not torch.equal(
                valid[..., :objects], valid[..., :objects].transpose(-1, -2)
            ):
                raise ValueError(f"object-object {name} contact_valid must be symmetric")
            if torch.any(target > valid.to(target.dtype)):
                raise ValueError(f"positive {name} contact must be marked valid")
        for name, present in (("e", e_present), ("mu", mu_present)):
            if tuple(present.shape) != (batch, objects, nodes):
                raise ValueError(
                    f"edge_{name}_present must be [B,N,N+1], got {tuple(present.shape)}"
                )
            if not torch.equal(
                present[..., :objects],
                present[..., :objects].transpose(-1, -2),
            ):
                raise ValueError(f"object-object edge_{name}_present must be symmetric")
        if tuple(valid_objects.shape) != (batch, objects):
            raise ValueError("object_valid_mask disagrees with three-contact labels")
        partner_valid = torch.cat(
            (
                valid_objects,
                torch.ones(
                    batch, 1, dtype=torch.bool, device=reference.device
                ),
            ),
            dim=1,
        )
        pair_valid = (
            valid_objects[:, None, :, None]
            & partner_valid[:, None, None, :]
        )
        upper_or_support = torch.zeros(
            objects, nodes, dtype=torch.bool, device=reference.device
        )
        upper_or_support[:, :objects] = torch.triu(
            torch.ones(
                objects, objects, dtype=torch.bool, device=reference.device
            ),
            diagonal=1,
        )
        upper_or_support[:, objects] = True
        base_pair_mask = pair_valid & upper_or_support[None, None]
        masks = {
            "base": valid_by_head["base"] & base_pair_mask,
            "e": valid_by_head["e"] & base_pair_mask & e_present[:, None],
            "mu": valid_by_head["mu"] & base_pair_mask & mu_present[:, None],
        }
        logits_by_head = {
            "base": self.sparse_object_base_contact_logits,
            "e": self.sparse_object_e_pair_event_logits,
            "mu": self.sparse_object_mu_pair_event_logits,
        }
        block_count = len(self.sparse_object_base_contact_logits)
        if any(len(values) != block_count for values in logits_by_head.values()):
            raise ValueError("three-contact heads disagree on Reader block count")

        def head_loss(name: str):
            mask = masks[name]
            target = targets[name]
            losses = []
            for logits in logits_by_head[name]:
                if tuple(logits.shape) != (batch, times, nodes, nodes):
                    raise ValueError("three-contact logits disagree across Reader blocks")
                if not torch.allclose(
                    logits, logits.transpose(-1, -2), atol=1e-5
                ):
                    raise ValueError("three-contact logits must be symmetric")
                if not mask.any():
                    losses.append(logits.float().sum() * 0.0)
                    continue
                item = F.binary_cross_entropy_with_logits(
                    logits.float()[:, :, :objects, :],
                    target,
                    reduction="none",
                )
                positive = mask & (target > 0.5)
                negative = mask & ~positive
                classes = []
                if positive.any():
                    classes.append(item[positive].mean())
                if negative.any():
                    classes.append(item[negative].mean())
                losses.append(torch.stack(classes).mean())
            return torch.stack(losses).mean()

        base_loss = head_loss("base")
        e_loss = head_loss("e")
        mu_loss = head_loss("mu")
        metrics = {"three_contact_block_count": reference.new_tensor(block_count)}
        for name, mask in masks.items():
            mean_gate = torch.sigmoid(
                torch.stack(logits_by_head[name]).mean(0).float()
            )[:, :, :objects, :]
            zero = reference.new_zeros(())
            metrics.update(
                {
                    f"{name}_contact_positive_fraction": (
                        targets[name][mask].mean().detach() if mask.any() else zero
                    ),
                    f"{name}_contact_gate_mean": (
                        mean_gate[mask].mean().detach() if mask.any() else zero
                    ),
                    f"{name}_contact_valid_count": mask.sum().detach(),
                }
            )
        return base_loss, e_loss, mu_loss, metrics

    def enable_sparse_pair_attention_capture(self, enabled: bool = True) -> None:
        if self.sparse_object_architecture not in {
            "temporal_pair_schedule",
            "temporal_no_interaction_schedule",
            "temporal_no_interaction_schedule_mugate",
            "temporal_no_interaction_schedule_mugate_decoupled_writer",
            "temporal_no_interaction_schedule_mugate_decoupled_writer_continuous",
            "temporal_no_interaction_schedule_trackprev_mugate",
            "temporal_no_interaction_schedule_ictr",
        }:
            raise RuntimeError(
                "pair-attention capture requires temporal_pair_schedule or "
                "a temporal_no_interaction_schedule route"
            )
        for group in self.sparse_object_adapter.groups.values():
            group.interaction.capture_attention = bool(enabled)

    def sparse_object_localization_loss(
        self,
        attention_labels: torch.Tensor,
        object_valid_mask: torch.Tensor,
        supervision_mode: str = "balanced_bce",
    ) -> Tuple[torch.Tensor, dict]:
        if not self.sparse_object_attention_logits:
            raise RuntimeError("sparse object router produced no attention logits")
        logits = torch.stack(self.sparse_object_attention_logits, dim=0)
        labels = attention_labels.to(device=logits.device, dtype=logits.dtype)
        valid = object_valid_mask.to(device=logits.device) > 0.5
        if labels.ndim == 4:
            labels = labels.unsqueeze(0)
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if tuple(labels.shape) != tuple(logits.shape[1:]):
            raise ValueError(
                f"attention labels {tuple(labels.shape)} != {tuple(logits.shape[1:])}"
            )
        if tuple(valid.shape) != (logits.shape[1], logits.shape[3]):
            raise ValueError(
                f"object_valid_mask {tuple(valid.shape)} != "
                f"{(logits.shape[1], logits.shape[3])}"
            )
        batch_labels = labels
        labels = batch_labels.unsqueeze(0).expand_as(logits)
        valid_spatial = valid[None, :, None, :, None, None].expand_as(logits)
        # The first-frame masks are forward inputs, so learned localization is
        # measured only on future latent frames.
        future_logits = logits[:, :, 1:]
        future_labels = labels[:, :, 1:]
        future_valid = valid_spatial[:, :, 1:]
        if supervision_mode == "balanced_bce":
            positives = (future_labels * future_valid).sum()
            valid_count = future_valid.sum().clamp_min(1)
            negatives = valid_count - positives
            positive_weight = (negatives / positives.clamp_min(1.0)).to(
                logits.dtype
            )
            elementwise = F.binary_cross_entropy_with_logits(
                future_logits,
                future_labels,
                pos_weight=positive_weight,
                reduction="none",
            )
            loss = (elementwise * future_valid).sum() / valid_count
            route_loss = loss
            read_loss = logits.new_zeros(())
        elif supervision_mode == "mask_route_read":
            if not self.sparse_object_reader_outputs:
                raise RuntimeError(
                    "mask_route_read supervision requires actual Reader routes"
                )
            reader_groups = len(self.sparse_object_reader_outputs)
            if reader_groups != logits.shape[0]:
                raise RuntimeError(
                    "Reader route receipts do not match localization groups: "
                    f"{reader_groups} versus {logits.shape[0]}"
                )

            flat_future_logits = future_logits.flatten(-2)
            flat_future_labels = future_labels.flatten(-2)
            label_mass = flat_future_labels.sum(dim=-1)
            active = valid[:, None, :].expand(
                logits.shape[1], logits.shape[2] - 1, logits.shape[3]
            ) & (label_mass[0] > 1.0e-6)
            active_groups = active[None].expand(logits.shape[0], -1, -1, -1)
            route_target = flat_future_labels / label_mass.clamp_min(1.0e-6)[
                ..., None
            ]
            route_ce = -(
                route_target * flat_future_logits.log_softmax(dim=-1)
            ).sum(dim=-1)
            route_loss = (
                route_ce * active_groups
            ).sum() / active_groups.sum().clamp_min(1)
            route_loss = route_loss / math.log(max(flat_future_logits.shape[-1], 2))

            indices = torch.stack(
                [output[0] for output in self.sparse_object_reader_outputs], dim=0
            )
            attention = torch.stack(
                [output[1] for output in self.sparse_object_reader_outputs], dim=0
            )
            grid_shapes = [output[2] for output in self.sparse_object_reader_outputs]
            if any(shape != grid_shapes[0] for shape in grid_shapes):
                raise RuntimeError("Reader groups disagree on the full grid shape")
            full_height, full_width = grid_shapes[0]
            if not isinstance(full_height, int) or not isinstance(full_width, int):
                raise RuntimeError("Reader grid shape must contain Python integers")
            if tuple(indices.shape[:4]) != tuple(logits.shape[:4]):
                raise RuntimeError(
                    "Reader route indices do not match [groups,B,T,N]: "
                    f"{tuple(indices.shape[:4])} versus {tuple(logits.shape[:4])}"
                )
            if tuple(attention.shape) != tuple(indices.shape):
                raise RuntimeError("Reader route indices and weights must match")
            if torch.any((indices < 0) | (indices >= full_height * full_width)):
                raise RuntimeError("Reader route contains an out-of-grid index")

            full_labels = batch_labels
            if tuple(full_labels.shape[-2:]) != (full_height, full_width):
                original_shape = full_labels.shape
                full_labels = F.interpolate(
                    full_labels.reshape(-1, 1, *original_shape[-2:]),
                    size=(full_height, full_width),
                    mode="area",
                ).reshape(*original_shape[:3], full_height, full_width)
            binary_full_labels = (full_labels >= 0.1).to(
                dtype=logits.dtype
            ).flatten(-2)
            expanded_full_labels = binary_full_labels[None].expand(
                logits.shape[0], -1, -1, -1, -1
            )
            selected_target = torch.gather(
                expanded_full_labels, -1, indices
            )
            on_target_mass = (attention * selected_target).sum(dim=-1)
            off_target_mass = 1.0 - on_target_mass[:, :, 1:]
            read_loss = (
                off_target_mass * active_groups
            ).sum() / active_groups.sum().clamp_min(1)
            loss = 0.5 * (route_loss + read_loss)
        else:
            raise ValueError(
                f"unsupported Reader supervision mode: {supervision_mode!r}"
            )

        mean_logits = logits.mean(dim=0)
        prediction = torch.sigmoid(mean_logits[:, 1:]) >= 0.5
        target = batch_labels[:, 1:] >= 0.1
        metric_valid = valid[:, None, :, None, None]
        prediction = prediction & metric_valid
        target = target & metric_valid
        intersection = (prediction & target).sum().float()
        union = (prediction | target).sum().float()
        target_count = target.sum().float()
        metrics = {
            "finder_iou": intersection / union.clamp_min(1.0),
            "finder_recall": intersection / target_count.clamp_min(1.0),
            "reader_route_loss": route_loss.detach(),
            "reader_off_target_mass": read_loss.detach(),
            "reader_on_target_mass": (1.0 - read_loss).detach(),
            "reader_supervision_mask_route_read": logits.new_tensor(
                float(supervision_mode == "mask_route_read")
            ),
        }
        self.sparse_object_last_attention_logits = mean_logits.detach()
        self.sparse_object_last_attention_labels = batch_labels.detach()
        return loss, metrics

    def sparse_object_writer_loss(
        self,
        attention_labels: torch.Tensor,
        object_valid_mask: torch.Tensor,
        supervision_mode: str = "mask_route_write",
    ) -> Tuple[torch.Tensor, dict]:
        """Supervise the Writer's global choice and its actual sparse writes."""
        if not self.sparse_object_writer_outputs:
            raise RuntimeError("independent sparse Writer produced no routing outputs")
        logits = torch.stack(
            [output[0] for output in self.sparse_object_writer_outputs], dim=0
        )
        indices = torch.stack(
            [output[1] for output in self.sparse_object_writer_outputs], dim=0
        )
        attention = torch.stack(
            [output[2] for output in self.sparse_object_writer_outputs], dim=0
        )
        seeds = torch.stack(
            [output[3] for output in self.sparse_object_writer_outputs], dim=0
        )
        labels = attention_labels.to(device=logits.device, dtype=logits.dtype)
        valid = object_valid_mask.to(device=logits.device) > 0.5
        if labels.ndim == 4:
            labels = labels.unsqueeze(0)
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        expected_prefix = (logits.shape[1], logits.shape[2], logits.shape[3])
        if tuple(labels.shape[:3]) != expected_prefix:
            raise ValueError(
                f"writer labels {tuple(labels.shape[:3])} != {expected_prefix}"
            )
        if tuple(valid.shape) != (logits.shape[1], logits.shape[3]):
            raise ValueError("writer object_valid_mask does not match [B,N]")
        if tuple(labels.shape[-2:]) != tuple(logits.shape[-2:]):
            original_shape = labels.shape
            labels = F.interpolate(
                labels.reshape(-1, 1, *original_shape[-2:]),
                size=logits.shape[-2:],
                mode="area",
            ).reshape(*original_shape[:3], *logits.shape[-2:])

        groups, batch, times, objects, height, width = logits.shape
        grid_cells = height * width
        future_logits = logits[:, :, 1:].flatten(-2)
        future_labels = labels[:, 1:].flatten(-2)
        label_mass = future_labels.sum(dim=-1)
        active = valid[:, None, :].expand(batch, times - 1, objects)
        active = active & (label_mass > 1e-6)
        if supervision_mode not in {"mask_route_write", "centroid_ce"}:
            raise ValueError(
                f"unsupported Writer supervision mode: {supervision_mode!r}"
            )
        target = future_labels / label_mass.clamp_min(1e-6)[..., None]
        if supervision_mode == "centroid_ce":
            y = torch.arange(height, device=logits.device, dtype=logits.dtype)
            x = torch.arange(width, device=logits.device, dtype=logits.dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            future_label_maps = labels[:, 1:]
            mass_2d = future_label_maps.sum(dim=(-2, -1)).clamp_min(1e-6)
            centroid_y = (
                future_label_maps * yy
            ).sum(dim=(-2, -1)) / mass_2d
            centroid_x = (
                future_label_maps * xx
            ).sum(dim=(-2, -1)) / mass_2d
            centroid_y = centroid_y.round().long().clamp(0, height - 1)
            centroid_x = centroid_x.round().long().clamp(0, width - 1)
            centroid_index = centroid_y * width + centroid_x
            target = torch.nn.functional.one_hot(
                centroid_index, num_classes=grid_cells
            ).to(dtype=logits.dtype)
        route_ce = -(
            target[None] * future_logits.log_softmax(dim=-1)
        ).sum(dim=-1)
        active_groups = active[None].expand(groups, -1, -1, -1)
        route_loss = (
            route_ce * active_groups
        ).sum() / active_groups.sum().clamp_min(1)
        route_loss = route_loss / math.log(max(grid_cells, 2))

        binary_labels = (labels >= 0.1).to(dtype=logits.dtype).flatten(-2)
        expanded_labels = binary_labels[None].expand(groups, -1, -1, -1, -1)
        selected_target = torch.gather(expanded_labels, -1, indices)
        on_target_mass = (attention * selected_target).sum(dim=-1)
        future_write_active = active[None].expand(groups, -1, -1, -1)
        off_target_mass = 1.0 - on_target_mass[:, :, 1:]
        write_loss = (
            off_target_mass * future_write_active
        ).sum() / future_write_active.sum().clamp_min(1)

        seed_target = torch.gather(
            expanded_labels, -1, seeds[..., None]
        ).squeeze(-1)
        seed_hit = (
            seed_target[:, :, 1:] * future_write_active
        ).sum() / future_write_active.sum().clamp_min(1)
        total = (
            route_loss
            if supervision_mode == "centroid_ce"
            else 0.5 * (route_loss + write_loss)
        )
        metrics = {
            "writer_route_ce_normalized": route_loss.detach(),
            "writer_off_target_mass": write_loss.detach(),
            "writer_seed_hit_rate": seed_hit.detach(),
            "writer_on_target_mass": (1.0 - write_loss).detach(),
            "writer_supervision_centroid_ce": logits.new_tensor(
                float(supervision_mode == "centroid_ce")
            ),
        }
        return total, metrics

    def sparse_object_hard_support_loss(
        self,
        attention_labels: torch.Tensor,
        object_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Supervise Reader and Writer binary support inside their fixed Top-8."""

        def one_head(outputs, prefix: str):
            if not outputs:
                raise RuntimeError(f"hard support enabled but {prefix} gate produced no outputs")
            indices = torch.stack([output[0] for output in outputs], dim=0)
            logits = torch.stack([output[1] for output in outputs], dim=0)
            hard = torch.stack([output[2] for output in outputs], dim=0).bool()
            attention = torch.stack([output[3] for output in outputs], dim=0)
            grid_shapes = [output[4] for output in outputs]
            allowed = torch.stack([output[5] for output in outputs], dim=0).bool()
            if any(shape != grid_shapes[0] for shape in grid_shapes):
                raise RuntimeError(f"{prefix} support-gate groups disagree on grid shape")
            if tuple(indices.shape) != tuple(logits.shape):
                raise RuntimeError(f"{prefix} support-gate indices/logits disagree")
            if tuple(hard.shape) != tuple(logits.shape):
                raise RuntimeError(f"{prefix} support-gate hard mask disagrees")
            if tuple(attention.shape) != tuple(logits.shape):
                raise RuntimeError(f"{prefix} support-gate attention disagrees")
            if tuple(allowed.shape) != tuple(logits.shape):
                raise RuntimeError(f"{prefix} support-gate allowed mask disagrees")

            groups, batch, times, objects, candidates = logits.shape
            labels = attention_labels.to(device=logits.device, dtype=logits.dtype)
            valid = object_valid_mask.to(device=logits.device) > 0.5
            if labels.ndim == 4:
                labels = labels.unsqueeze(0)
            if valid.ndim == 1:
                valid = valid.unsqueeze(0)
            if tuple(labels.shape[:3]) != (batch, times, objects):
                raise ValueError(f"{prefix} gate labels do not match [B,T,N]")
            if tuple(valid.shape) != (batch, objects):
                raise ValueError(f"{prefix} gate valid mask does not match [B,N]")
            height, width = grid_shapes[0]
            if tuple(labels.shape[-2:]) != (height, width):
                original_shape = labels.shape
                labels = F.interpolate(
                    labels.reshape(-1, 1, *original_shape[-2:]),
                    size=(height, width),
                    mode="area",
                ).reshape(*original_shape[:3], height, width)
            binary = (labels >= 0.1).to(dtype=logits.dtype).flatten(-2)
            target = torch.gather(
                binary[None].expand(groups, -1, -1, -1, -1),
                -1,
                indices,
            )
            label_nonempty = binary.sum(dim=-1) > 0
            active_route = valid[:, None, :] & label_nonempty
            active = (
                active_route[None, ..., None].expand_as(logits) & allowed
            ).clone()
            active[:, :, 0] = False
            target_bool_raw = target > 0.5
            positive_active = active & target_bool_raw
            negative_active = active & ~target_bool_raw
            positive_loss = F.softplus(-logits).masked_select(positive_active)
            negative_loss = F.softplus(logits).masked_select(negative_active)
            class_losses = []
            if positive_loss.numel():
                class_losses.append(positive_loss.mean())
            if negative_loss.numel():
                class_losses.append(negative_loss.mean())
            # Equal weight per class when both occur.  If a batch contains only
            # one class, keep supervising that class instead of assigning it a
            # zero pos_weight or silently dropping the objective.
            loss = (
                torch.stack(class_losses).mean()
                if class_losses
                else logits.sum() * 0.0
            )

            prediction = hard & active
            target_bool = target_bool_raw & active
            true_positive = (prediction & target_bool).sum().float()
            predicted_positive = prediction.sum().float()
            target_positive = target_bool.sum().float()
            precision = true_positive / predicted_positive.clamp_min(1.0)
            recall = true_positive / target_positive.clamp_min(1.0)
            f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-6)
            valid_routes = active_route[None, :, 1:].expand(
                groups, batch, times - 1, objects
            )
            support_count = hard[:, :, 1:].sum(dim=-1).float()[valid_routes]
            raw_all_off = ~(logits[:, :, 1:] >= 0.0).any(dim=-1)
            fallback_values = raw_all_off[valid_routes].float()
            fallback_rate = (
                fallback_values.mean()
                if fallback_values.numel()
                else logits.new_zeros(())
            )
            off_gate_weight = attention.masked_select(~hard).abs()
            off_gate_max = (
                off_gate_weight.max()
                if off_gate_weight.numel()
                else logits.new_zeros(())
            )
            full_positive = binary[:, 1:].sum(dim=-1)
            full_active = active_route[:, 1:]
            candidate_recall_values = (
                (target[:, :, 1:] * allowed[:, :, 1:]).sum(dim=-1)
                / full_positive[None].clamp_min(1.0)
            )[full_active[None].expand(groups, -1, -1, -1)]
            candidate_recall = (
                candidate_recall_values.mean()
                if candidate_recall_values.numel()
                else logits.new_zeros(())
            )
            support_mean = (
                support_count.mean()
                if support_count.numel()
                else logits.new_zeros(())
            )
            support_min = (
                support_count.min()
                if support_count.numel()
                else logits.new_zeros(())
            )
            support_max = (
                support_count.max()
                if support_count.numel()
                else logits.new_zeros(())
            )
            return loss, {
                f"{prefix}_support_gate_bce": loss.detach(),
                f"{prefix}_support_gate_precision": precision.detach(),
                f"{prefix}_support_gate_recall": recall.detach(),
                f"{prefix}_support_gate_f1": f1.detach(),
                f"{prefix}_support_count_mean": support_mean.detach(),
                f"{prefix}_support_count_min": support_min.detach(),
                f"{prefix}_support_count_max": support_max.detach(),
                f"{prefix}_support_active_route_count": logits.new_tensor(
                    float(support_count.numel())
                ),
                f"{prefix}_support_fallback_rate": fallback_rate.detach(),
                f"{prefix}_support_off_gate_max": off_gate_max.detach(),
                f"{prefix}_support_candidate_gt_recall": candidate_recall.detach(),
            }

        reader_loss, reader_metrics = one_head(
            self.sparse_object_reader_gate_outputs, "reader"
        )
        writer_loss, writer_metrics = one_head(
            self.sparse_object_writer_gate_outputs, "writer"
        )
        total = 0.5 * (reader_loss + writer_loss)
        return total, {
            "hard_support_gate_loss": total.detach(),
            **reader_metrics,
            **writer_metrics,
        }

    def enable_friction_controlnet(
        self,
        num_blocks: int = 5,
        latent_channels: int = 48,
    ) -> None:
        """Attach the branch after strict Wan checkpoint loading."""
        if self.friction_controlnet is None:
            self.friction_controlnet = FrictionControlNetBranch(
                latent_channels=latent_channels,
                dim=self.dim,
                num_blocks=min(num_blocks, len(self.blocks)),
            )
            reference = next(self.blocks.parameters())
            self.friction_controlnet.to(device=reference.device)
        self.friction_controlnet.initialize_from_base_blocks(self.blocks)
        # The base is BF16, but ordinary AdamW at 1e-5 cannot reliably update
        # BF16 weights around typical magnitudes. Keep every trainable branch
        # parameter in FP32; residuals are cast back only at base injection.
        self.friction_controlnet.to(device=reference.device, dtype=torch.float32)
        self.freeze_base_for_friction_controlnet()

    def freeze_base_for_friction_controlnet(self) -> None:
        if self.friction_controlnet is None:
            return
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("friction_controlnet."))

    def enable_restitution_controlnet(
        self,
        num_blocks: int = 5,
        latent_channels: int = 48,
    ) -> None:
        """Attach an independently trained restitution branch beside friction."""
        if self.restitution_controlnet is None:
            self.restitution_controlnet = FrictionControlNetBranch(
                latent_channels=latent_channels,
                dim=self.dim,
                num_blocks=min(num_blocks, len(self.blocks)),
            )
            reference = next(self.blocks.parameters())
            self.restitution_controlnet.to(device=reference.device)
        else:
            reference = next(self.blocks.parameters())
        self.restitution_controlnet.initialize_from_base_blocks(self.blocks)
        self.restitution_controlnet.to(device=reference.device, dtype=torch.float32)

    def enable_force_controlnet(
        self,
        num_blocks: int = 5,
        latent_channels: int = 48,
    ) -> None:
        """Attach an independently trainable force branch beside existing branches."""
        reference = next(self.blocks.parameters())
        if self.force_controlnet is None:
            self.force_controlnet = FrictionControlNetBranch(
                latent_channels=latent_channels,
                dim=self.dim,
                num_blocks=min(num_blocks, len(self.blocks)),
            )
            self.force_controlnet.to(device=reference.device)
        self.force_controlnet.initialize_from_base_blocks(self.blocks)
        self.force_controlnet.to(device=reference.device, dtype=torch.float32)
        self.freeze_base_for_force_controlnet()

    def freeze_base_for_force_controlnet(self) -> None:
        if self.force_controlnet is None:
            return
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("force_controlnet."))

    def enable_force_conditioning(
        self,
        mode: str,
        channels: int = 3,
        attention_blocks: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 28),
        window_size: int = 2,
    ) -> None:
        """Attach one of two non-ControlNet force interfaces after base loading."""
        if mode not in ("latent_concat", "cross_attention"):
            raise ValueError(f"unsupported force condition mode: {mode}")
        if self.force_condition_mode is not None and self.force_condition_mode != mode:
            raise RuntimeError(
                f"force conditioning already enabled as {self.force_condition_mode}"
            )
        reference = next(self.blocks.parameters())
        self.force_condition_mode = mode
        self.force_condition_channels = channels
        self.force_condition_attention_blocks = tuple(
            block for block in attention_blocks if block < len(self.blocks)
        )
        if mode == "latent_concat":
            # This zero-initialised convolution is algebraically equivalent to
            # expanding patch_embedding from in_dim to in_dim+channels, copying
            # the clean-Wan slice, and zero-initialising the new channel slice.
            self.force_condition_projection = nn.Conv3d(
                channels,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=False,
            )
            nn.init.zeros_(self.force_condition_projection.weight)
        else:
            self.force_condition_projection = nn.Conv3d(
                channels,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=False,
            )
            nn.init.xavier_uniform_(self.force_condition_projection.weight)
            self.force_condition_attention = nn.ModuleDict({
                str(block): ForceWindowCrossAttention(
                    self.dim,
                    self.blocks[block].num_heads,
                    window_size=window_size,
                )
                for block in self.force_condition_attention_blocks
            })
        self.force_condition_projection.to(device=reference.device, dtype=torch.float32)
        self.force_condition_attention.to(device=reference.device, dtype=torch.float32)

    def freeze_base_for_force_conditioning(self) -> None:
        """Train the new interface plus optional LoRA weights, freeze clean Wan."""
        if self.force_condition_mode is None:
            return
        for name, parameter in self.named_parameters():
            trainable = (
                name.startswith("force_condition_")
                or "lora_A" in name
                or "lora_B" in name
            )
            parameter.requires_grad_(trainable)

    def enable_global_physics_cross_attn(
        self,
        attention_blocks: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 28),
        harmonic_bands: int = GLOBAL_PHYSICS_HARMONIC_BANDS,
    ) -> None:
        """Attach the independent global g/mu/e physics cross-attention interface.

        Mounted at the same eight evenly-spaced block indices as the F path.
        There is no weight sharing with force_condition_attention.
        """
        if self.global_physics_enabled:
            raise RuntimeError("global physics conditioning already enabled")
        reference = next(self.blocks.parameters())
        self.global_physics_attention_blocks = tuple(
            block for block in attention_blocks if block < len(self.blocks)
        )
        self.global_physics_token_encoder = PhysicsTokenEncoder(
            self.dim,
            num_types=len(GLOBAL_PHYSICS_TYPES),
            harmonic_bands=harmonic_bands,
        )
        self.global_physics_attention = nn.ModuleDict({
            str(block): GlobalPhysicsCrossAttn(self.dim, self.blocks[block].num_heads)
            for block in self.global_physics_attention_blocks
        })
        self.global_physics_enabled = True
        self.global_physics_token_encoder.to(device=reference.device, dtype=torch.float32)
        self.global_physics_attention.to(device=reference.device, dtype=torch.float32)

    def freeze_base_for_force_and_physics_conditioning(self) -> None:
        """Train F cross-attn + LoRA + GlobalPhysicsCrossAttn; freeze everything
        else. This is one atomic pass over every parameter (not a call to
        freeze_base_for_force_conditioning() followed by a second pass that
        only *adds* the physics prefix), because two separate passes would be
        call-order dependent: whichever ran second would reset every name it
        does not recognise back to requires_grad=False, silently re-freezing
        whatever the first pass had just enabled. Call this instead of
        freeze_base_for_force_conditioning() whenever global physics
        conditioning is enabled (train.py's WanTrainingModule does so)."""
        if self.force_condition_mode is None and not self.global_physics_enabled:
            return
        for name, parameter in self.named_parameters():
            trainable = (
                name.startswith("force_condition_")
                or name.startswith("global_physics_")
                or "lora_A" in name
                or "lora_B" in name
            )
            parameter.requires_grad_(trainable)

    def prepare_global_physics_tokens(
        self,
        phys_tokens: Optional[list],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Turn this sample's [(type_str, raw_value), ...] list into a KV state.

        Mirrors validate_force_condition/patchify_with_force's convention: the
        caller passes one sample's (unbatched) token list -- matching how
        force_condition_tensor arrives without a batch dim -- and this method
        adds the batch dimension itself, broadcasting to `batch` (>1 only
        happens under CFG-style duplication of the same sample, never from
        multiple distinct dataset rows, since training batches exactly one
        dataset item per process). Missing physics quantities must never
        appear as a zero-valued token: the source-side builder simply omits
        them, so `phys_tokens` legitimately ranges from an empty list (pure-F
        sources) up to K entries.
        """
        if not self.global_physics_enabled:
            return None
        tokens_list = list(phys_tokens) if phys_tokens else []
        n = len(tokens_list)
        # DDP correctness: this state must be produced (and the module below
        # invoked) identically on every rank every step, even when this
        # sample carries zero physics tokens (pure-F sources) -- otherwise,
        # under find_unused_parameters=False, the module's/encoder's params
        # go "unused" on some ranks but not others in the same step, which
        # hangs the corresponding ALLREDUCE until the NCCL watchdog aborts.
        # A fully-masked dummy token below is value-equivalent to skipping
        # (GlobalPhysicsCrossAttn zeroes its output for an all-padded row)
        # but keeps every rank's participation uniform.
        dummy = n == 0
        if dummy:
            n = 1
        encoder = self.global_physics_token_encoder
        encoder_dtype = next(encoder.parameters()).dtype
        scaled_values = torch.zeros(n, dtype=torch.float32, device=device)
        type_ids = torch.zeros(n, dtype=torch.long, device=device)
        if not dummy:
            for i, (type_name, raw_value) in enumerate(tokens_list):
                if type_name not in GLOBAL_PHYSICS_TYPE_TO_ID:
                    raise ValueError(f"unknown global physics token type: {type_name!r}")
                raw_value = float(raw_value)
                scaled_value = GLOBAL_PHYSICS_SCALE[type_name] * raw_value
                if not math.isfinite(raw_value) or not 0.0 <= scaled_value <= 1.0:
                    raise ValueError(
                        f"global physics token {type_name}={raw_value} is outside "
                        f"the fixed range [0, {GLOBAL_PHYSICS_MAX[type_name]}]"
                    )
                scaled_values[i] = scaled_value
                type_ids[i] = GLOBAL_PHYSICS_TYPE_TO_ID[type_name]
        if not torch.isfinite(scaled_values).all():
            raise ValueError("global physics token values contain non-finite entries")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            flat_tokens = encoder(scaled_values.to(encoder_dtype), type_ids)
        tokens = flat_tokens.unsqueeze(0).to(dtype=dtype)  # [1, N, D]
        key_padding_mask = (
            torch.zeros(1, n, dtype=torch.bool, device=device)
            if dummy
            else torch.ones(1, n, dtype=torch.bool, device=device)
        )
        if batch > 1:
            tokens = tokens.expand(batch, -1, -1)
            key_padding_mask = key_padding_mask.expand(batch, -1)
        return tokens, key_padding_mask

    def apply_global_physics_cross_attention(
        self,
        block_index: int,
        x: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if (
            state is None
            or not self.global_physics_enabled
            or str(block_index) not in self.global_physics_attention
        ):
            return x
        tokens, key_padding_mask = state
        module = self.global_physics_attention[str(block_index)]
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=x.is_cuda,
        ):
            residual = module(
                x,
                tokens.to(dtype=x.dtype),
                key_padding_mask,
            )
        return x + residual.to(dtype=x.dtype)

    def validate_force_condition(
        self,
        condition: Optional[torch.Tensor],
        batch: int,
        frames: int,
        height: int,
        width: int,
    ) -> Optional[torch.Tensor]:
        if condition is None:
            return None
        if self.force_condition_mode is None:
            raise ValueError("force condition tensor supplied while conditioning is disabled")
        if condition.ndim == 4:
            condition = condition.unsqueeze(0)
        if condition.ndim != 5 or condition.shape[1] != self.force_condition_channels:
            raise ValueError(
                "force condition must be [B,3,T,H,W] or [3,T,H,W], got "
                f"{tuple(condition.shape)}"
            )
        if condition.shape[0] == 1 and batch > 1:
            condition = condition.expand(batch, -1, -1, -1, -1)
        if condition.shape[0] != batch:
            raise ValueError(f"force batch {condition.shape[0]} != video batch {batch}")
        if tuple(condition.shape[2:]) != (frames, height, width):
            raise ValueError(
                f"force grid {tuple(condition.shape[2:])} != video latent grid "
                f"{(frames, height, width)}"
            )
        if not torch.isfinite(condition).all():
            raise ValueError("force condition contains non-finite values")
        support = condition[:, :1].float()
        signed = condition[:, 1:].float()
        if (
            float(support.min()) < 0.0
            or float(support.max()) > 1.0
            or float(signed.min()) < -1.0
            or float(signed.max()) > 1.0
        ):
            raise ValueError("force support/vector channels violate [-1,1] contract")
        if torch.count_nonzero(signed.masked_select(support.expand_as(signed) == 0)).item() != 0:
            raise ValueError("force vector is nonzero outside support")
        return condition

    def patchify_with_force(
        self,
        x: torch.Tensor,
        force_condition: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        base = self.patch_embedding(x)
        condition = self.validate_force_condition(
            force_condition,
            batch=x.shape[0],
            frames=x.shape[2],
            height=x.shape[3],
            width=x.shape[4],
        )
        if condition is None:
            return base, None
        projection_parameter = next(self.force_condition_projection.parameters())
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=condition.is_cuda,
        ):
            projected = self.force_condition_projection(
                condition.to(device=projection_parameter.device)
            )
        if self.force_condition_mode == "latent_concat":
            return base + projected.to(dtype=base.dtype), None
        support = F.avg_pool3d(
            condition[:, :1].float(),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).clamp_(0.0, 1.0)
        force_tokens = rearrange(projected, "b d t h w -> b (t h w) d").contiguous()
        support_tokens = rearrange(support, "b c t h w -> b (t h w) c").contiguous()
        return base, (force_tokens, support_tokens)

    def apply_force_condition_attention(
        self,
        block_index: int,
        x: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor]],
        frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if (
            state is None
            or self.force_condition_mode != "cross_attention"
            or str(block_index) not in self.force_condition_attention
        ):
            return x
        force_tokens, support = state
        module = self.force_condition_attention[str(block_index)]
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=x.is_cuda,
        ):
            residual = module(
                x,
                force_tokens.to(dtype=x.dtype),
                support,
                frames,
                height,
                width,
            )
        return x + residual.to(dtype=x.dtype)

    def prepare_force_controlnet(
        self,
        control_latents: Optional[torch.Tensor],
        frames: int,
        height: int,
        width: int,
        batch: int,
    ) -> Optional[torch.Tensor]:
        if control_latents is None:
            return None
        if self.force_controlnet is None:
            raise ValueError("force_control_latents were supplied but the branch is disabled")
        if not self.force_controlnet.blocks:
            raise RuntimeError("the force branch was not copied from the loaded Wan blocks")
        if control_latents.ndim == 4:
            control_latents = control_latents.unsqueeze(0)
        if control_latents.ndim != 5:
            raise ValueError(
                "force_control_latents must have shape [C, T, H, W] or [B, C, T, H, W]"
            )
        if control_latents.shape[0] == 1 and batch > 1:
            control_latents = control_latents.expand(batch, -1, -1, -1, -1)
        if control_latents.shape[0] != batch:
            raise ValueError(
                f"control batch {control_latents.shape[0]} does not match denoiser batch {batch}"
            )
        return self.force_controlnet.initial_state(control_latents, frames, height, width)

    def apply_force_controlnet_block(
        self,
        block_index: int,
        state: Optional[torch.Tensor],
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            state is None
            or self.force_controlnet is None
            or block_index >= self.force_controlnet.num_blocks
        ):
            return state, None
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=state.is_cuda,
        ):
            state = gradient_checkpoint_forward(
                self.force_controlnet.blocks[block_index],
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                state,
                context.to(dtype=state.dtype),
                t_mod.to(dtype=state.dtype),
                freqs,
            )
            residual = self.force_controlnet.project_output(
                block_index,
                state,
                frames,
                height,
                width,
            )
        return state, residual

    def prepare_restitution_controlnet(
        self,
        control_latents: Optional[torch.Tensor],
        frames: int,
        height: int,
        width: int,
        batch: int,
    ) -> Optional[torch.Tensor]:
        if control_latents is None:
            return None
        if self.restitution_controlnet is None:
            raise ValueError(
                "restitution_control_latents were supplied but the branch is disabled"
            )
        if not self.restitution_controlnet.blocks:
            raise RuntimeError("the restitution branch was not copied from the loaded Wan blocks")
        if control_latents.ndim == 4:
            control_latents = control_latents.unsqueeze(0)
        if control_latents.ndim != 5:
            raise ValueError(
                "restitution_control_latents must have shape [C, T, H, W] or [B, C, T, H, W]"
            )
        if control_latents.shape[0] == 1 and batch > 1:
            control_latents = control_latents.expand(batch, -1, -1, -1, -1)
        if control_latents.shape[0] != batch:
            raise ValueError(
                f"control batch {control_latents.shape[0]} does not match denoiser batch {batch}"
            )
        return self.restitution_controlnet.initial_state(
            control_latents,
            frames,
            height,
            width,
        )

    def apply_restitution_controlnet_block(
        self,
        block_index: int,
        state: Optional[torch.Tensor],
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            state is None
            or self.restitution_controlnet is None
            or block_index >= self.restitution_controlnet.num_blocks
        ):
            return state, None
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=state.is_cuda,
        ):
            state = gradient_checkpoint_forward(
                self.restitution_controlnet.blocks[block_index],
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                state,
                context.to(dtype=state.dtype),
                t_mod.to(dtype=state.dtype),
                freqs,
            )
            residual = self.restitution_controlnet.project_output(
                block_index,
                state,
                frames,
                height,
                width,
            )
        return state, residual

    def prepare_friction_controlnet(
        self,
        control_latents: Optional[torch.Tensor],
        frames: int,
        height: int,
        width: int,
        batch: int,
    ) -> Optional[torch.Tensor]:
        if control_latents is None:
            return None
        if self.friction_controlnet is None:
            raise ValueError(
                "friction_control_latents were supplied but the branch is disabled"
            )
        if not self.friction_controlnet.blocks:
            raise RuntimeError("the friction branch was not copied from the loaded Wan blocks")
        if control_latents.ndim == 4:
            control_latents = control_latents.unsqueeze(0)
        if control_latents.ndim != 5:
            raise ValueError(
                "friction_control_latents must have shape [C, T, H, W] or [B, C, T, H, W]"
            )
        if control_latents.shape[0] == 1 and batch > 1:
            control_latents = control_latents.expand(batch, -1, -1, -1, -1)
        if control_latents.shape[0] != batch:
            raise ValueError(
                f"control batch {control_latents.shape[0]} does not match denoiser batch {batch}"
            )
        return self.friction_controlnet.initial_state(
            control_latents,
            frames,
            height,
            width,
        )

    def apply_friction_controlnet_block(
        self,
        block_index: int,
        state: Optional[torch.Tensor],
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            state is None
            or self.friction_controlnet is None
            or block_index >= self.friction_controlnet.num_blocks
        ):
            return state, None
        # FP32 parameters provide master-weight updates; BF16 autocast keeps the
        # copied attention blocks on Wan's efficient FlashAttention path.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=state.is_cuda,
        ):
            state = gradient_checkpoint_forward(
                self.friction_controlnet.blocks[block_index],
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                state,
                context.to(dtype=state.dtype),
                t_mod.to(dtype=state.dtype),
                freqs,
            )
            residual = self.friction_controlnet.project_output(
                block_index,
                state,
                frames,
                height,
                width,
            )
        return state, residual

    def prepare_wantodance(
        self,
        in_dim, dim, num_heads, has_image_pos_emb, out_dim, patch_size, eps,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
    ):
        if wantodance_enable_music_inject:
            all_modules, all_modules_names = wantodance_torch_dfs(self.blocks, parent_name="root.transformer_blocks")
            self.music_injector = WanToDanceInjector(all_modules, all_modules_names, dim=dim, num_heads=num_heads, inject_layer=wantodance_music_inject_layers)
        if wantodance_enable_refimage:
            self.img_emb_refimage = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if wantodance_enable_refface:
            self.img_emb_refface = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if wantodance_enable_global or wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            music_feature_dim = 35
            ff_size = 1024
            dropout = 0.1
            latent_dim = 256
            nhead = 4
            activation = F.gelu
            rotary = WanToDanceRotaryEmbedding(dim=latent_dim)
            self.music_projection = nn.Linear(music_feature_dim, latent_dim)
            self.music_encoder = nn.Sequential()
            for _ in range(2):
                self.music_encoder.append(
                    WanToDanceMusicEncoderLayer(
                        d_model=latent_dim,
                        nhead=nhead,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        batch_first=True,
                        rotary=rotary,
                        device='cuda',
                    )
                )
        if wantodance_enable_unimodel:
            self.patch_embedding_global = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        if wantodance_enable_unimodel:
            self.head_global = Head(dim, out_dim, patch_size, eps)
        self.wantodance_enable_music_inject = wantodance_enable_music_inject
        self.wantodance_enable_refimage = wantodance_enable_refimage
        self.wantodance_enable_refface = wantodance_enable_refface
        self.wantodance_enable_global = wantodance_enable_global
        self.wantodance_enable_dynamicfps = wantodance_enable_dynamicfps
        self.wantodance_enable_unimodel = wantodance_enable_unimodel

    def wantodance_after_transformer_block(self, block_idx, hidden_states):
        if self.wantodance_enable_music_inject:
            if block_idx in self.music_injector.injected_block_id.keys():
                audio_attn_id = self.music_injector.injected_block_id[block_idx]
                audio_emb = self.merged_audio_emb  # b f n c
                num_frames = audio_emb.shape[1]
                input_hidden_states = hidden_states.clone()  # b (f h w) c
                input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)
                attn_hidden_states = self.music_injector.injector_pre_norm_feat[audio_attn_id](input_hidden_states)
                audio_emb = rearrange(audio_emb, "b t c -> (b t) 1 c", t=num_frames)
                attn_audio_emb = audio_emb
                residual_out = self.music_injector.injector[audio_attn_id](attn_hidden_states, attn_audio_emb)
                residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
                hidden_states = hidden_states + residual_out
        return hidden_states

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None, enable_wantodance_global=False):
        if enable_wantodance_global:
            x = self.patch_embedding_global(x)
        else:
            x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs
                )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
