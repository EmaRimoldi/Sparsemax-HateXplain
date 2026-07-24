from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def sparsemax_numpy(values: np.ndarray) -> np.ndarray:
    """Project a one-dimensional NumPy vector onto the probability simplex."""
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("sparsemax_numpy expects a non-empty one-dimensional vector")
    shifted = scores - scores.max()
    sorted_scores = np.sort(shifted)[::-1]
    cumulative = np.cumsum(sorted_scores)
    support = 1 + np.arange(1, scores.size + 1) * sorted_scores > cumulative
    support_size = int(support.sum())
    tau = (cumulative[support_size - 1] - 1.0) / support_size
    return np.maximum(shifted - tau, 0.0)


def sparsemax(scores: Tensor, dim: int = -1, mask: Tensor | None = None) -> Tensor:
    """Differentiable sparsemax with optional boolean validity mask."""
    if not scores.is_floating_point():
        raise TypeError("sparsemax expects floating-point scores")

    if mask is not None:
        valid = mask.to(dtype=torch.bool, device=scores.device)
        valid = torch.broadcast_to(valid, scores.shape)
        if not torch.all(valid.any(dim=dim)):
            raise ValueError("every sparsemax slice must contain at least one valid element")
        floor = torch.finfo(scores.dtype).min / 2
        scores = scores.masked_fill(~valid, floor)
    else:
        valid = None

    shifted = scores - scores.max(dim=dim, keepdim=True).values
    sorted_scores = torch.sort(shifted, dim=dim, descending=True).values
    cumulative = sorted_scores.cumsum(dim)

    size = scores.size(dim)
    rank_shape = [1] * scores.ndim
    rank_shape[dim] = size
    ranks = torch.arange(1, size + 1, device=scores.device, dtype=scores.dtype)
    ranks = ranks.reshape(rank_shape)

    support = 1 + ranks * sorted_scores > cumulative
    support_size = support.sum(dim=dim, keepdim=True)
    tau = (cumulative.gather(dim, support_size - 1) - 1) / support_size
    output = torch.clamp(shifted - tau, min=0)

    if valid is not None:
        output = output.masked_fill(~valid, 0)
    return output


def sparsemax_loss(
    scores: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Fenchel--Young sparsemax loss for a distribution-valued target."""
    if scores.shape != target.shape:
        raise ValueError("scores and target must have identical shapes")
    if scores.ndim != 2:
        raise ValueError("sparsemax_loss expects tensors shaped [batch, sequence]")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"unsupported reduction: {reduction}")

    if mask is None:
        valid = torch.ones_like(scores, dtype=torch.bool)
    else:
        valid = mask.to(device=scores.device, dtype=torch.bool)
        if valid.shape != scores.shape:
            raise ValueError("mask must match scores")

    floor = torch.finfo(scores.dtype).min / 2
    masked_scores = scores.masked_fill(~valid, floor)
    probabilities = sparsemax(masked_scores, dim=-1, mask=valid)
    support = probabilities > 0
    support_size = support.sum(dim=-1).clamp_min(1).to(scores.dtype)

    safe_scores = scores.masked_fill(~valid, 0)
    target = target.to(scores.dtype).masked_fill(~valid, 0)
    tau = ((safe_scores * support).sum(dim=-1) - 1.0) / support_size

    cross_term = -(target * safe_scores).sum(dim=-1)
    quadratic = 0.5 * (
        ((safe_scores.square()) * support).sum(dim=-1) - support_size * tau.square()
    )
    constant = 0.5 * target.square().sum(dim=-1)
    losses = cross_term + quadratic + constant

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


def soft_target_cross_entropy(
    values: Tensor,
    target: Tensor,
    mask: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Historical baseline loss: log-softmax over post-softmax attention values."""
    if values.shape != target.shape or values.shape != mask.shape:
        raise ValueError("values, target, and mask must have identical shapes")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"unsupported reduction: {reduction}")

    losses = []
    for row_values, row_target, row_mask in zip(values, target, mask, strict=True):
        valid = row_mask.to(dtype=torch.bool)
        if not valid.any():
            raise ValueError("every row must contain at least one valid token")
        log_probabilities = torch.log_softmax(row_values[valid], dim=0)
        losses.append(-(row_target[valid] * log_probabilities).sum())
    result = torch.stack(losses)

    if reduction == "none":
        return result
    if reduction == "sum":
        return result.sum()
    return result.mean()
