"""Single-forward auxiliary losses for Sparse Object Interaction v2."""

from __future__ import annotations

import torch


def dynamic_weighted_flow_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    clean_latents: torch.Tensor,
    maximum_weight: float = 5.0,
):
    """Extra single-sample FM loss weighted by GT latent frame changes."""
    if clean_latents.ndim != 5:
        raise ValueError("clean_latents must be [B,C,T,H,W]")
    motion = (clean_latents[:, :, 1:] - clean_latents[:, :, :-1]).abs()
    motion = motion.float().mean(dim=1, keepdim=True)
    if prediction.shape[2] != motion.shape[2]:
        raise ValueError(
            f"prediction time {prediction.shape[2]} != motion time {motion.shape[2]}"
        )
    mean_motion = motion.mean(dim=(2, 3, 4), keepdim=True)
    weights = torch.where(
        mean_motion > 1e-8,
        motion / mean_motion.clamp_min(1e-8),
        torch.zeros_like(motion),
    ).clamp_(0.0, float(maximum_weight)).detach()
    squared_error = (prediction.float() - target.float()).square()
    expanded_weights = weights.expand_as(squared_error)
    loss = (squared_error * expanded_weights).sum() / expanded_weights.sum().clamp_min(1.0)
    stats = {
        "dynamic_weight_mean": weights.mean().detach(),
        "dynamic_weight_max": weights.amax().detach(),
        "dynamic_weight_nonzero_ratio": (weights > 0).float().mean().detach(),
    }
    return loss, stats


def temporal_residual_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """First-order temporal consistency of the FM residual, no extra forward."""
    residual = prediction.float() - target.float()
    if residual.shape[2] < 2:
        return residual.sum() * 0.0
    temporal_residual = residual[:, :, 1:] - residual[:, :, :-1]
    return torch.nn.functional.smooth_l1_loss(
        temporal_residual,
        torch.zeros_like(temporal_residual),
    )
