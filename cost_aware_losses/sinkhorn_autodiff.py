"""
cost_aware_losses.sinkhorn_autodiff
==================================

Entropic OT-loss for classification using a Sinkhorn solver, with
**full autodiff through the Sinkhorn iterations**.

Compared to :class:`~cost_aware_losses.sinkhorn_envelope.SinkhornEnvelopeLoss`,
this variant allows gradients to flow through the iterative Sinkhorn updates.
This can be more "end-to-end", but comes with:
- higher memory usage (saving intermediate tensors for backprop),
- potentially less stable gradients for long iteration counts.

Author
------
Warith Harchaoui <wharchaoui@nexton-group.com>

Usage
-----
>>> import torch
>>> from cost_aware_losses import SinkhornFullAutodiffLoss
>>> scores = torch.randn(8, 4, requires_grad=True)
>>> y = torch.randint(0, 4, (8,))
>>> C = torch.rand(8, 4, 4)
>>> loss = SinkhornFullAutodiffLoss(max_iter=50)(scores, y, C=C)
>>> loss.backward()
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .base import CostAwareLoss

def _sinkhorn_plan(
    p: Tensor,
    q: Tensor,
    C: Tensor,
    eps: Tensor,
    *,
    max_iter: int,
) -> Tensor:
    """
    Compute the entropic OT plan via Sinkhorn scaling (balanced OT).

    Parameters
    ----------
    p, q:
        Source and target distributions, shape (B, K). Each row must sum to 1.
    C:
        Cost matrices, shape (B, K, K).
    eps:
        Entropic regularization parameter ε, shape (B,) or scalar tensor.
    max_iter:
        Number of Sinkhorn iterations.

    Returns
    -------
    Tensor
        Transport plans P of shape (B, K, K).

    Notes
    -----
    This is the classical balanced entropic OT with Gibbs kernel:

        K = exp(-C / ε)
        P = diag(u) K diag(v)

    updated by:
        u = p / (K v)
        v = q / (K^T u)

    We clamp denominators for numerical stability.
    """
    if p.ndim != 2 or q.ndim != 2:
        raise ValueError("p and q must have shape (B, K).")
    if C.ndim != 3:
        raise ValueError("C must have shape (B, K, K).")

    B, K = p.shape
    eps_b = eps.view(-1, 1, 1) if eps.ndim == 1 else eps

    # Gibbs kernel
    Kmat = torch.exp(-C / eps_b)  # (B,K,K)

    # Initialize scalings
    u = torch.ones((B, K), device=p.device, dtype=p.dtype) / K
    v = torch.ones((B, K), device=p.device, dtype=p.dtype) / K

    # Small constant to avoid division by 0
    tiny = torch.as_tensor(1e-12, device=p.device, dtype=p.dtype)

    for _ in range(int(max_iter)):
        Kv = torch.bmm(Kmat, v.unsqueeze(-1)).squeeze(-1)  # (B,K)
        u = p / torch.clamp(Kv, min=tiny)

        KTu = torch.bmm(Kmat.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)  # (B,K)
        v = q / torch.clamp(KTu, min=tiny)

    P = u.unsqueeze(2) * Kmat * v.unsqueeze(1)
    return P


def _entropy_kl_objective(P: Tensor, p: Tensor, q: Tensor, eps: Tensor, C: Tensor) -> Tensor:
    """
    Entropic OT primal objective with KL regularization against p ⊗ q.

    Objective:
        <P, C> + ε * KL(P || p ⊗ q)

    where:
        KL(P || p⊗q) = Σ_ij P_ij [log P_ij - log p_i - log q_j]

    Parameters
    ----------
    P:
        Transport plan (B, K, K).
    p, q:
        Marginals (B, K).
    eps:
        ε of shape (B,) or scalar.
    C:
        Cost matrix (B, K, K).

    Returns
    -------
    Tensor
        Objective values per example (B,).
    """
    tiny = torch.as_tensor(1e-12, device=P.device, dtype=P.dtype)
    P_safe = torch.clamp(P, min=tiny)
    p_safe = torch.clamp(p, min=tiny)
    q_safe = torch.clamp(q, min=tiny)

    cost_term = (P * C).sum(dim=(1, 2))

    # KL term: Σ_ij P_ij (log P_ij - log p_i - log q_j)
    logP = torch.log(P_safe)
    logp = torch.log(p_safe).unsqueeze(2)
    logq = torch.log(q_safe).unsqueeze(1)
    kl = (P_safe * (logP - logp - logq)).sum(dim=(1, 2))

    if eps.ndim == 0:
        return cost_term + eps * kl
    return cost_term + eps * kl


class SinkhornFullAutodiffLoss(CostAwareLoss):
    """
    Sinkhorn OT-loss with full autodiff through Sinkhorn iterations.

    Parameters
    ----------
    epsilon_mode, epsilon, epsilon_scale, epsilon_min:
        See :class:`~cost_aware_losses.base.CostAwareLoss`.
    max_iter:
        Number of Sinkhorn iterations.
    label_smoothing:
        Small label smoothing applied to the one-hot target distribution q.
    """

    def __init__(
        self,
        *,
        epsilon_mode: str = "offdiag_mean",
        epsilon: Optional[float] = None,
        epsilon_scale: float = 1.0,
        epsilon_min: float = 1e-8,
        max_iter: int = 50,
        label_smoothing: float = 1e-3,
    ) -> None:
        super().__init__(
            epsilon_mode=epsilon_mode,  # type: ignore[arg-type]
            epsilon=epsilon,
            epsilon_scale=epsilon_scale,
            epsilon_min=epsilon_min,
        )
        self.max_iter = int(max_iter)
        self.label_smoothing = float(label_smoothing)

    def _loss_per_example(self, scores: Tensor, targets: Tensor, Cb: Tensor) -> Tensor:
        """
        Per-example loss values with full autodiff.

        Returns
        -------
        Tensor
            Loss vector (B,).
        """
        B, K = scores.shape
        eps = self.compute_epsilon(Cb)

        p = torch.softmax(scores, dim=1)

        q = torch.zeros((B, K), device=scores.device, dtype=scores.dtype)
        q.scatter_(1, targets.view(-1, 1), 1.0)

        if self.label_smoothing > 0:
            delta = self.label_smoothing
            q = (1.0 - delta) * q + delta / float(K)

        # Full autodiff plan.
        P = _sinkhorn_plan(p, q, Cb, eps, max_iter=self.max_iter)
        return _entropy_kl_objective(P, p, q, eps, Cb)
