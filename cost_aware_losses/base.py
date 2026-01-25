"""
cost_aware_losses.base
=====================

Shared utilities and an abstract base class for cost-aware classification losses.

The core design goal is to factor out the *stable* ingredients common to all
cost-aware objectives in this repository:

- Cost-matrix handling (global vs per-example costs)
- Robust temperature/regularization scaling via off-diagonal statistics
- A consistent forward signature:

    forward(scores, targets, C=None) -> loss

Where:
- scores has shape (B, K)
- targets has shape (B,)
- C is either (K, K) or (B, K, K) or None (default uniform costs)

Notes
-----
In the Sinkhorn-regularized OT literature (and in POT), the entropic
regularization parameter is commonly denoted ``reg``. In this repository we use
``epsilon`` as the temperature/regularization scale. For balanced entropic OT
(with KL regularization), these correspond directly:

    reg = epsilon

See ``docs/math.md`` for a concise explanation.

Author
------
Warith Harchaoui <wharchaoui@nexton-group.com>

Usage
-----
Subclass :class:`CostAwareLoss` and implement :meth:`_loss_per_example`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
from torch import Tensor
from torch.nn import Module


_EpsilonMode = Literal["constant", "offdiag_mean", "offdiag_median", "offdiag_max"]


def off_diagonal_stat(C: Tensor, stat: Literal["mean", "median", "max"]) -> Tensor:
    """
    Compute a statistic of off-diagonal entries of a cost matrix.

    Parameters
    ----------
    C:
        Cost matrix of shape (K, K) or batch of shape (B, K, K).
    stat:
        Statistic to compute across off-diagonal entries: ``mean``, ``median``, or ``max``.

    Returns
    -------
    Tensor
        Scalar tensor if C is (K, K), or tensor of shape (B,) if C is (B, K, K).

    Raises
    ------
    ValueError
        If C does not have exactly 2 or 3 dimensions.
    """
    if C.ndim == 2:
        K = C.shape[0]
        mask = ~torch.eye(K, dtype=torch.bool, device=C.device)
        vals = C[mask]
        if stat == "mean":
            return vals.mean()
        if stat == "median":
            return vals.median()
        if stat == "max":
            return vals.max()

    if C.ndim == 3:
        B, K, _ = C.shape
        mask = ~torch.eye(K, dtype=torch.bool, device=C.device)
        mask = mask.unsqueeze(0).expand(B, K, K)
        vals = C[mask].view(B, K * K - K)
        if stat == "mean":
            return vals.mean(dim=1)
        if stat == "median":
            return vals.median(dim=1).values
        if stat == "max":
            return vals.max(dim=1).values

    raise ValueError(f"C must have shape (K,K) or (B,K,K), got {tuple(C.shape)}.")


class CostAwareLoss(Module, ABC):
    """
    Abstract base class for cost-aware classification losses.

    Parameters
    ----------
    epsilon_mode:
        How to compute the temperature/regularization scale ε.
    epsilon:
        Constant ε value when ``epsilon_mode='constant'``.
    epsilon_scale:
        Multiplicative factor applied to the data-driven ε statistic.
    epsilon_min:
        Lower bound to prevent numerical issues.
    epsilon_schedule:
        Optional scheduling strategy: None, "exponential_decay".
        When enabled, epsilon is multiplied by a time-varying factor.
    schedule_start_mult:
        Starting multiplier for epsilon schedule (default: 10.0).
    schedule_end_mult:
        Ending multiplier for epsilon schedule (default: 0.1).
    total_epochs:
        Total number of training epochs (required if epsilon_schedule is set).
    """

    def __init__(
        self,
        *,
        epsilon_mode: _EpsilonMode = "offdiag_mean",
        epsilon: Optional[float] = None,
        epsilon_scale: float = 1.0,
        epsilon_min: float = 1e-8,
        epsilon_schedule: Optional[str] = None,
        schedule_start_mult: float = 10.0,
        schedule_end_mult: float = 0.1,
        total_epochs: Optional[int] = None,
    ) -> None:
        super().__init__()

        if epsilon_mode == "constant" and epsilon is None:
            raise ValueError("epsilon must be provided when epsilon_mode='constant'.")
        
        if epsilon_schedule is not None and epsilon_schedule not in ("exponential_decay",):
            raise ValueError(f"Unknown epsilon_schedule: {epsilon_schedule}")
        
        if epsilon_schedule is not None and total_epochs is None:
            raise ValueError("total_epochs must be provided when epsilon_schedule is set.")

        self.epsilon_mode = epsilon_mode
        self.epsilon = epsilon
        self.epsilon_scale = float(epsilon_scale)
        self.epsilon_min = float(epsilon_min)
        
        # Scheduling
        self.epsilon_schedule = epsilon_schedule
        self.schedule_start_mult = float(schedule_start_mult)
        self.schedule_end_mult = float(schedule_end_mult)
        self.total_epochs = int(total_epochs) if total_epochs is not None else None
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """
        Update the current epoch for epsilon scheduling.
        
        Parameters
        ----------
        epoch:
            Current epoch (0-indexed).
        """
        self.current_epoch = int(epoch)
    
    def _compute_schedule_multiplier(self) -> float:
        """
        Compute the schedule multiplier for current epoch.
        
        Returns
        -------
        float
            Multiplier to apply to base epsilon.
        """
        if self.epsilon_schedule is None:
            return 1.0
        
        if self.epsilon_schedule == "exponential_decay":
            # Exponential decay: start_mult * (end_mult / start_mult) ^ (t / T)
            # where t = current_epoch, T = total_epochs - 1
            if self.total_epochs is None or self.total_epochs <= 1:
                return 1.0
            
            progress = float(self.current_epoch) / float(self.total_epochs - 1)
            progress = min(max(progress, 0.0), 1.0)  # Clamp to [0, 1]
            
            ratio = self.schedule_end_mult / self.schedule_start_mult
            multiplier = self.schedule_start_mult * (ratio ** progress)
            return multiplier
        
        return 1.0
    
    def compute_epsilon(self, C: Tensor) -> Tensor:
        """
        Determine the temperature/regularization parameter ε from the cost matrix.
        
        Applies scheduling if enabled.

        Parameters
        ----------
        C:
            Cost matrix (K, K) or (B, K, K).

        Returns
        -------
        Tensor
            Scalar ε if C is (K, K), or shape (B,) if C is (B, K, K).
        """
        # Compute base epsilon
        if self.epsilon_mode == "constant":
            eps = torch.as_tensor(self.epsilon, device=C.device, dtype=C.dtype)
        else:
            stat = self.epsilon_mode.replace("offdiag_", "")
            eps = off_diagonal_stat(C, stat)  # type: ignore[arg-type]

        # Apply static scale
        eps = eps * self.epsilon_scale
        
        # Apply schedule multiplier
        schedule_mult = self._compute_schedule_multiplier()
        eps = eps * schedule_mult
        
        return torch.clamp(eps, min=self.epsilon_min)

    @staticmethod
    def default_uniform_cost(K: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
        Default uniform misclassification costs: 1 off-diagonal, 0 on diagonal.

        Parameters
        ----------
        K:
            Number of classes.
        device, dtype:
            Tensor placement.

        Returns
        -------
        Tensor
            Cost matrix of shape (K, K).
        """
        C = torch.ones((K, K), device=device, dtype=dtype)
        C.fill_diagonal_(0.0)
        return C

    @staticmethod
    def ensure_batched_cost(C: Tensor, batch_size: int) -> Tensor:
        """
        Ensure C is a batched tensor (B, K, K).

        Parameters
        ----------
        C:
            Cost matrix (K,K) or (B,K,K).
        batch_size:
            Target batch size B.

        Returns
        -------
        Tensor
            Batched cost matrix of shape (B, K, K).
        """
        if C.ndim == 2:
            return C.unsqueeze(0).expand(batch_size, -1, -1)
        if C.ndim == 3:
            if C.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch: scores B={batch_size} but C has B={C.shape[0]}.")
            return C
        raise ValueError(f"C must have shape (K,K) or (B,K,K), got {tuple(C.shape)}.")

    @abstractmethod
    def _loss_per_example(self, scores: Tensor, targets: Tensor, Cb: Tensor) -> Tensor:
        """
        Compute per-example loss values.

        Implementations should return a tensor of shape (B,).

        Notes
        -----
        The base class will take the mean over the batch and return a scalar loss.
        """
        raise NotImplementedError

    def forward(self, scores: Tensor, targets: Tensor, C: Optional[Tensor] = None) -> Tensor:
        """
        Compute the batch-mean cost-aware loss.

        Parameters
        ----------
        scores:
            Model outputs (logits or scores), shape (B, K).
        targets:
            Ground-truth integer labels in [0, K-1], shape (B,).
        C:
            Optional cost matrix. Either (K, K) for a global cost or (B, K, K)
            for example-dependent costs. If None, uses a uniform off-diagonal
            cost matrix.

        Returns
        -------
        Tensor
            Scalar tensor equal to the mean loss across the batch.
        """
        if scores.ndim != 2:
            raise ValueError(f"scores must have shape (B,K), got {tuple(scores.shape)}.")
        if targets.ndim != 1:
            raise ValueError(f"targets must have shape (B,), got {tuple(targets.shape)}.")
        if scores.shape[0] != targets.shape[0]:
            raise ValueError("scores and targets must share the same batch dimension.")

        B, K = scores.shape
        device, dtype = scores.device, scores.dtype

        if C is None:
            C = self.default_uniform_cost(K, device=device, dtype=dtype)

        Cb = self.ensure_batched_cost(C, batch_size=B)
        loss_vec = self._loss_per_example(scores, targets, Cb)
        return loss_vec.mean()
