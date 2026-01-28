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

    @torch.no_grad()
    def initialize_output_bias(self, target_prevalence: float) -> None:
        """
        Initialize the output layer's bias to reflect the target prevalence.
        
        This helps with calibration at the start of training, especially for
        imbalanced datasets.
        
        Parameters
        ----------
        target_prevalence : float
            The fraction of positive samples (class 1) in the training set.
        """
        if target_prevalence <= 0 or target_prevalence >= 1:
            return

        # For softmax (K=2): 
        # P(y=1) = exp(z1) / (exp(z0) + exp(z1))
        # If we set z0 = 0, then z1 = log(p / (1-p))
        bias_val = float(torch.log(torch.tensor(target_prevalence / (1.0 - target_prevalence))))
        
        # Access the final linear layer
        final_layer: nn.Linear
        if isinstance(self.net, nn.Linear):
            final_layer = self.net
        else:
            # For nn.Sequential, it's the last element
            final_layer = self.net[-1] # type: ignore

        if hasattr(final_layer, "bias") and final_layer.bias is not None:
            # Initialize: class 0 bias to 0, class 1 bias to bias_val
            final_layer.bias.zero_()
            final_layer.bias[1] = bias_val

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

def round_nearest_upper_power_of_2(x: int) -> int:
    """
    Round an integer to the nearest upper power of 2.
    
    Parameters
    ----------
    x : int
        The integer to round
        
    Returns
    -------
    int
        The nearest upper power of 2
    """
    return 2 ** (x - 1).bit_length()

def compute_smart_architecture_defaults(
    input_dim: int,
    n_train: int,
    n_classes: int = 2,
) -> Tuple[Tuple[int, ...], float]:
    """
    Compute smart architecture defaults using classic ML heuristics.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    n_train : int
        Number of training samples
    n_classes : int
        Number of output classes (default: 2 for binary)
        
    Returns
    -------
    hidden_dims : Tuple[int, ...]
        Recommended hidden layer dimensions
    dropout : float
        Recommended dropout rate
        
    Heuristics
    ----------
    - First layer: ~2/3 of input_dim (captures main patterns)
    - Second layer: ~1/3 of input_dim (compression toward output)
    - Third layer: bridge layer if needed (optional)
    - Pyramid structure: gradually decrease toward n_classes
    - Capacity constraint: total params should be < n_train to avoid overfitting
    - Dropout: higher for smaller datasets, 0.2-0.5 typical range
    
    Examples
    --------
    >>> hidden_dims, dropout = compute_smart_architecture_defaults(544, 413378)
    >>> hidden_dims
    (384, 192)
    >>> dropout
    0.1
    """
    # Classic pyramid heuristic: start at ~2/3 input, compress to output
    layer1 = max(int(input_dim * 0.67), n_classes * 2)  # At least 2x output
    layer2 = max(int(input_dim * 0.33), n_classes * 2)  # At least 2x output
    
    # # Round to nice numbers (multiples of 32 or 64 for GPU efficiency)
    # def round_to_nice(x: int) -> int:
    #     if x >= 256:
    #         return ((x + 31) // 32) * 32  # Round to nearest 32
    #     elif x >= 64:
    #         return ((x + 15) // 16) * 16  # Round to nearest 16
    #     else:
    #         return ((x + 7) // 8) * 8  # Round to nearest 8

    
    layer1 = round_nearest_upper_power_of_2(layer1)
    layer2 = round_nearest_upper_power_of_2(layer2)
    
    # Ensure decreasing pyramid
    if layer2 >= layer1:
        layer2 = max(layer1 // 2, n_classes * 2)
        layer2 = round_nearest_upper_power_of_2(layer2) 
    
    # Estimate total parameters (rough)
    # params ≈ input_dim * layer1 + layer1 * layer2 + layer2 * n_classes
    total_params = input_dim * layer1 + layer1 * layer2 + layer2 * n_classes
    
    # If capacity too high relative to n_train, scale down
    capacity_ratio = total_params / n_train
    if capacity_ratio > 2.0:  # Rule of thumb: params < 2 * n_train
        scale_factor = (2.0 / capacity_ratio) ** 0.5  # Square root for gradual reduction
        layer1 = round_nearest_upper_power_of_2(int(layer1 * scale_factor))
        layer2 = round_nearest_upper_power_of_2(int(layer2 * scale_factor))
    
    # Determine optimal dropout based on dataset size
    # Larger datasets → less dropout needed
    # Smaller datasets → more dropout to prevent overfitting
    samples_per_feature = n_train / input_dim
    if samples_per_feature < 10:
        dropout_rate = 0.5  # High dropout for very small datasets
    elif samples_per_feature < 50:
        dropout_rate = 0.3  # Moderate dropout
    elif samples_per_feature < 200:
        dropout_rate = 0.2  # Light dropout
    else:
        dropout_rate = 0.1  # Minimal dropout for large datasets
    
    return (layer1, layer2), dropout_rate
