"""
cost_aware_losses.sinkhorn_fenchel_young
========================================

Sinkhorn–Fenchel–Young (formerly CACIS) loss for cost-sensitive classification.

This loss is designed for settings where:
- misclassification costs are asymmetric and meaningful (€, time, risk, ...),
- costs may be example-dependent,
- we want gradients that *do not* backpropagate through an inner solver.

Mathematical summary
--------------------
Given scores f ∈ R^K, label y, and cost matrix C ∈ R^{K×K}, Sinkhorn–Fenchel–Young defines:

    ℓ(y, f; C, ε) = Ω*_{C,ε}(f) − f_y

where Ω* is the Fenchel conjugate of a convex regularizer Ω_{C,ε}.
The conjugate term is computed through an inner optimization solved by
Frank–Wolfe on the simplex, but **we do not differentiate through the solver**.

See ``docs/math.md`` for a concise derivation and interpretation.

Usage
-----
>>> import torch
>>> from cost_aware_losses import SinkhornFenchelYoungLoss
>>> scores = torch.randn(16, 5, requires_grad=True)
>>> y = torch.randint(0, 5, (16,))
>>> C = torch.rand(16, 5, 5)
>>> loss = SinkhornFenchelYoungLoss(solver_iter=30)(scores, y, C=C)
>>> loss.backward()
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .base import CostAwareLoss


def _solve_qp_on_simplex(M: Tensor, n_iter: int) -> Tensor:
    """
    Approximate the minimizer of a quadratic form over the simplex via Frank–Wolfe.

    Solves:
        min_{α ∈ Δ_K} αᵀ M α
    where Δ_K = {α ∈ R^K : α_i >= 0, Σ α_i = 1}.

    Parameters
    ----------
    M:
        Positive (semi-)definite matrix batch of shape (B, K, K).
    n_iter:
        Number of Frank–Wolfe iterations.

    Returns
    -------
    Tensor
        Approximate minimizer α of shape (B, K), each row in the simplex.
    """
    B, K, _ = M.shape
    alpha = torch.full((B, K), 1.0 / K, device=M.device, dtype=M.dtype)

    for it in range(int(n_iter)):
        # Gradient of αᵀ M α is 2 M α (M is symmetric by construction).
        grad = 2.0 * torch.bmm(M, alpha.unsqueeze(-1)).squeeze(-1)

        # Linear minimization oracle on the simplex -> pick vertex with min gradient.
        idx = grad.argmin(dim=1)
        s = torch.zeros_like(alpha)
        s.scatter_(1, idx.unsqueeze(1), 1.0)

        # Standard FW step size γ = 2 / (t + 2)
        step = 2.0 / (it + 2.0)
        alpha = (1.0 - step) * alpha + step * s

    return alpha


class SinkhornFenchelYoungLoss(CostAwareLoss):
    """
    Sinkhorn–Fenchel–Young loss (implicit differentiation).

    Parameters
    ----------
    epsilon_mode, epsilon, epsilon_scale, epsilon_min:
        See :class:`~cost_aware_losses.base.CostAwareLoss`.
    solver_iter:
        Number of Frank–Wolfe iterations used to approximate the inner optimum.

    Notes
    -----
    Backpropagation does **not** go through the solver: α is computed under
    ``torch.no_grad()`` and used in an implicit-gradient construction.
    """

    def __init__(
        self,
        *,
        epsilon_mode: str = "offdiag_mean",
        epsilon: Optional[float] = None,
        epsilon_scale: float = 2.0,
        epsilon_min: float = 1e-8,
        solver_iter: int = 50,
    ) -> None:
        super().__init__(
            epsilon_mode=epsilon_mode,  # type: ignore[arg-type]
            epsilon=epsilon,
            epsilon_scale=epsilon_scale,
            epsilon_min=epsilon_min,
        )
        self.solver_iter = int(solver_iter)

    def _conjugate_term(self, scores: Tensor, Cb: Tensor, eps: Tensor) -> Tensor:
        """
        Evaluate the Fenchel conjugate term Ω*(f) per example.

        For Sinkhorn–Fenchel–Young:
            Ω*(f) = -ε log( min_{α ∈ Δ} αᵀ M α )
        where
            M_{ij} = exp(-(f_i + f_j + c_{ij}) / ε).

        Parameters
        ----------
        scores:
            Model scores (B, K).
        Cb:
            Batched cost matrices (B, K, K).
        eps:
            Temperature vector (B,).

        Returns
        -------
        Tensor
            Conjugate values (B,).
        """
        B, K = scores.shape

        # Build exponent in a numerically stable way.
        f_i = (0.5 * scores).unsqueeze(2)  # (B,K,1)
        f_j = (0.5 * scores).unsqueeze(1)  # (B,1,K)
        exponent = -(f_i + f_j + Cb) / eps.view(-1, 1, 1)

        # Log-sum-exp shift for stability before exponentiating.
        shift = exponent.amax(dim=(1, 2), keepdim=True)
        logM = exponent - shift
        M = torch.exp(logM)

        # Inner solve (no gradients through solver).
        with torch.no_grad():
            alpha = _solve_qp_on_simplex(M, self.solver_iter)

        # Compute log(αᵀ M α) in log-domain:
        # log Σ_i Σ_j α_i α_j M_ij = LSE(log α_i + log α_j + log M_ij)
        neginf = torch.tensor(-float("inf"), device=scores.device, dtype=scores.dtype)
        loga = torch.where(alpha > 0, torch.log(alpha), neginf)
        term = loga.unsqueeze(2) + loga.unsqueeze(1) + logM  # (B,K,K)
        logval = torch.logsumexp(term.view(B, K * K), dim=1)

        return -eps * (logval + shift.squeeze(-1).squeeze(-1))

    def _loss_per_example(self, scores: Tensor, targets: Tensor, Cb: Tensor) -> Tensor:
        """
        Per-example Sinkhorn–Fenchel–Young loss: Ω*(f) - f_y.

        Parameters
        ----------
        scores:
            (B, K)
        targets:
            (B,)
        Cb:
            (B, K, K)

        Returns
        -------
        Tensor
            Loss vector (B,).
        """
        eps = self.compute_epsilon(Cb)
        conj = self._conjugate_term(scores, Cb, eps)

        f_y = scores.gather(1, targets.view(-1, 1)).squeeze(1)
        return conj - f_y
