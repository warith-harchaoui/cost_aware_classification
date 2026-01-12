"""
examples.tabular_models
=================================

Tabular models for fraud detection experiments.

This file isolates model definitions from the training script, so the benchmark
runner can remain stable while the model architecture evolves over time.

Naming choices
--------------
We use a task-facing name (`TabularRiskModel`) rather than `MLPModel`, because
in future iterations this may be replaced by:
- embedding-based models for categoricals,
- FT-Transformer / TabTransformer-style architectures,
- monotonic networks,
- or other specialized tabular models.

The class currently supports two backbones:
- ``linear`` : a single linear layer
- ``mlp``    : a configurable MLP

Usage
-----
>>> model = TabularRiskModel(TabularModelConfig(input_dim=128, backbone="mlp", hidden_dims=(256, 128), dropout=0.1, K=2))
>>> logits = model(x)  # (B, 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor


BackboneName = Literal["linear", "mlp"]


@dataclass(frozen=True)
class TabularModelConfig:
    """
    Configuration for :class:`TabularRiskModel`.

    Attributes
    ----------
    input_dim:
        Number of input features.
    backbone:
        Backbone family name: "linear" or "mlp".
    hidden_dims:
        Hidden layer sizes for the MLP backbone.
    dropout:
        Dropout probability applied between hidden layers (MLP only).
    use_batchnorm:
        Whether to use BatchNorm1d between linear layers (MLP only).
    K:
        Number of output classes (default=2).
    """
    input_dim: int
    backbone: BackboneName = "mlp"
    hidden_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.0
    use_batchnorm: bool = False
    K: int = 2


class TabularRiskModel(nn.Module):
    """
    Risk model for tabular fraud detection.

    The model outputs logits for K classes.

    Parameters
    ----------
    config:
        A :class:`TabularModelConfig` instance.

    Notes
    -----
    The output dimension is defined by config.K.
    """

    def __init__(self, config: TabularModelConfig) -> None:
        super().__init__()
        self.config = config
        self.net = self._build_network(config)

    @staticmethod
    def _build_network(config: TabularModelConfig) -> nn.Module:
        """
        Build the network specified by the config.

        Returns
        -------
        nn.Module
            A module mapping (B, input_dim) -> (B, K).
        """
        if config.backbone == "linear":
            return nn.Linear(config.input_dim, config.K)

        if config.backbone == "mlp":
            layers: list[nn.Module] = []
            in_dim = config.input_dim
            for h in config.hidden_dims:
                layers.append(nn.Linear(in_dim, int(h)))
                if config.use_batchnorm:
                    layers.append(nn.BatchNorm1d(int(h)))
                layers.append(nn.ReLU(inplace=True))
                if config.dropout and config.dropout > 0:
                    layers.append(nn.Dropout(p=float(config.dropout)))
                in_dim = int(h)
            layers.append(nn.Linear(in_dim, config.K))
            return nn.Sequential(*layers)

        raise ValueError(f"Unknown backbone: {config.backbone}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input features of shape (B, D).

        Returns
        -------
        Tensor
            Logits of shape (B, 2).
        """
        return self.net(x)
