"""
examples.utils
========================

Utilities for the IEEE-CIS fraud detection benchmark.

This module focuses on:
- logging configuration
- device selection (CUDA / MPS / CPU)
- a TrainingState container for iteration-based metric tracking
- plotting metric trajectories over iterations (cost/regret, AP, etc.)
- plotting Precision–Recall curves
- checkpoint serialization helpers (save/load TrainingState)

Design notes
------------
Fraud detection is highly imbalanced, so **Precision–Recall** metrics (PR-AUC)
are often more informative than ROC-AUC. Additionally, in cost-aware
settings, it is crucial to track **business-aligned regret/cost** metrics rather
than relying on raw training losses, which may not be comparable across different
surrogate objectives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import matplotlib.pyplot as plt


PathLike = Union[str, Path]


# =============================================================================
# Logging / device
# =============================================================================

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure basic logging.

    Parameters
    ----------
    level:
        Logging level (e.g., ``logging.INFO``).
    """
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def get_device() -> torch.device:
    """
    Return the best available PyTorch device.

    Returns
    -------
    torch.device
        ``cuda`` if available, else ``mps`` if available, else ``cpu``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Training state
# =============================================================================

@dataclass
class TrainingState:
    """
    Container for iteration-based metric tracking.

    Attributes
    ----------
    batch_size:
        Training batch size (used in plot labels).
    current_iter:
        Global optimizer step count.
    train_ema:
        Exponential moving averages of training-batch metrics.
        Keys are metric names; values are sequences aligned with ``train_ema_iters``.
    train_ema_iters:
        Iteration indices for EMA points.
    probe_points:
        Periodic evaluation points on a fixed validation probe set.
        Keys are metric names; values are sequences aligned with ``probe_iters``.
    probe_iters:
        Iteration indices for probe evaluation points.
    epoch_iters:
        Iteration indices at epoch boundaries (for vertical lines in plots).
    """
    batch_size: int = 0
    current_iter: int = 0

    train_ema: Dict[str, List[float]] = field(default_factory=dict)
    train_ema_iters: List[int] = field(default_factory=list)

    probe_points: Dict[str, List[float]] = field(default_factory=dict)
    probe_iters: List[int] = field(default_factory=list)

    epoch_iters: List[int] = field(default_factory=lambda: [0])


def training_state_to_dict(state: TrainingState) -> Dict[str, Any]:
    """
    Convert TrainingState to a JSON-serializable dictionary.

    Parameters
    ----------
    state:
        The TrainingState instance.

    Returns
    -------
    dict
        Serializable dictionary.
    """
    return {
        "batch_size": int(state.batch_size),
        "current_iter": int(state.current_iter),
        "train_ema": {k: list(map(float, v)) for k, v in state.train_ema.items()},
        "train_ema_iters": list(map(int, state.train_ema_iters)),
        "probe_points": {k: list(map(float, v)) for k, v in state.probe_points.items()},
        "probe_iters": list(map(int, state.probe_iters)),
        "epoch_iters": list(map(int, state.epoch_iters)),
    }


def training_state_from_dict(d: Dict[str, Any]) -> TrainingState:
    """
    Restore a TrainingState from a dictionary.

    Parameters
    ----------
    d:
        Dictionary produced by :func:`training_state_to_dict`.

    Returns
    -------
    TrainingState
        Reconstructed state.
    """
    state = TrainingState(
        batch_size=int(d.get("batch_size", 0)),
        current_iter=int(d.get("current_iter", 0)),
    )
    state.train_ema = {str(k): list(map(float, v)) for k, v in (d.get("train_ema") or {}).items()}
    state.train_ema_iters = list(map(int, d.get("train_ema_iters") or []))
    state.probe_points = {str(k): list(map(float, v)) for k, v in (d.get("probe_points") or {}).items()}
    state.probe_iters = list(map(int, d.get("probe_iters") or []))
    state.epoch_iters = list(map(int, d.get("epoch_iters") or [0]))
    return state


# =============================================================================
# EMA helpers
# =============================================================================

def ema_update(prev: Optional[float], x: float, alpha: float) -> float:
    """
    Update an exponential moving average.

    Parameters
    ----------
    prev:
        Previous EMA value, or None if uninitialized.
    x:
        New observation.
    alpha:
        Smoothing factor in (0, 1]. Larger alpha = less smoothing.

    Returns
    -------
    float
        Updated EMA value.
    """
    if prev is None:
        return float(x)
    return float(alpha) * float(x) + (1.0 - float(alpha)) * float(prev)


# =============================================================================
# Plotting
# =============================================================================

def plot_metric_trajectory(
    *,
    iters: Sequence[int],
    values: Sequence[float],
    out_path: PathLike,
    title: str,
    ylabel: str,
    epoch_iters: Optional[Sequence[int]] = None,
    y_quantile_max: Optional[float] = 0.98,
) -> None:
    """
    Plot a single metric vs training iterations.

    Parameters
    ----------
    iters:
        Iteration indices.
    values:
        Metric values (same length as iters).
    out_path:
        PNG file path to save.
    title:
        Plot title.
    ylabel:
        Y-axis label.
    epoch_iters:
        Optional epoch boundary iteration indices; drawn as light vertical lines.
    y_quantile_max:
        If provided, sets the upper y-limit to this quantile to improve readability.
        Use None to avoid cropping (e.g., AP in [0,1]).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    it = np.asarray(list(iters), dtype=int)
    val = np.asarray(list(values), dtype=float)

    plt.figure(figsize=(12, 4.5))
    plt.plot(it, val, linewidth=2)

    if epoch_iters:
        first = True
        for e in epoch_iters:
            plt.axvline(e, linestyle=":", alpha=0.25, label="Epoch" if first else None)
            first = False
        if len(epoch_iters) > 1:
            plt.legend(loc="best")

    plt.xlabel("Optimizer iterations")
    plt.ylabel(ylabel)
    plt.title(title)

    if y_quantile_max is not None and val.size > 0:
        top = float(np.quantile(val, float(y_quantile_max)))
        bottom = float(np.min(val))
        pad = 0.1 * (top - bottom + 1e-12)
        plt.ylim(bottom - pad, top + pad)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    *,
    out_path: PathLike,
    title: str,
    average_precision: Optional[float] = None,
) -> None:
    """
    Plot a precision–recall curve.

    Parameters
    ----------
    precision:
        Precision values (sklearn).
    recall:
        Recall values (sklearn).
    out_path:
        Destination PNG path.
    title:
        Plot title.
    average_precision:
        Optional Average Precision (AP) for annotation.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2)

    if average_precision is not None:
        plt.title(f"{title}\nAP={average_precision:.4f}")
    else:
        plt.title(title)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0.0, 1.05)
    plt.xlim(0.0, 1.0)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
