"""
cost_aware_losses
=================

A small research/engineering toolkit for **cost-aware classification**:

- Sinkhorn-Fenchel-Young (implicit Fenchel–Young loss with Frank–Wolfe inner solver)
- Sinkhorn entropic OT losses:
  - envelope-style gradients
  - full autodiff through Sinkhorn iterations
  - POT library integration

Author
------
Warith Harchaoui <wharchaoui@nexton-group.com>

See:
- docs/math.md for derivations and the ε ↔ reg mapping,
- docs/fraud_business_and_cost_matrix.md for the IEEE-CIS fraud value model,
- examples/fraud_detection.py for an end-to-end benchmark script.
"""

from .sinkhorn_fenchel_young import SinkhornFenchelYoungLoss
from .sinkhorn_envelope import SinkhornEnvelopeLoss
from .sinkhorn_autodiff import SinkhornFullAutodiffLoss
from .sinkhorn_pot import SinkhornPOTLoss

__all__ = [
    "SinkhornFenchelYoungLoss",
    "SinkhornEnvelopeLoss",
    "SinkhornFullAutodiffLoss",
    "SinkhornPOTLoss",
]
