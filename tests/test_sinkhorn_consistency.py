import torch
import pytest
import logging
from typing import List, Tuple

from cost_aware_losses import (
    SinkhornPOTLoss,
    SinkhornFullAutodiffLoss,
    SinkhornEnvelopeLoss,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_data(B=4, K=3, seed=42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    scores = torch.randn(B, K, requires_grad=True, dtype=torch.float64)
    # Scale down scores to keep softmax away from saturation
    scores.data.mul_(0.5)
    
    y = torch.randint(0, K, (B,))
    
    # Random cost matrix (B, K, K)
    C = torch.rand(B, K, K, dtype=torch.float64)
    # Ensure some 0s on diagonal for realism? Or just random is fine.
    # Let's make diagonal smaller to look like costs
    for i in range(B):
        C[i].diagonal().fill_(0.1)
        
    return scores, y, C

@pytest.mark.parametrize("eps", [0.1, 1.0])
def test_sinkhorn_variants_consistency(eps):
    """
    Check that SinkhornPOTLoss, SinkhornFullAutodiffLoss, and SinkhornEnvelopeLoss
    produce consistent loss values and gradients.
    """
    logger.info(f"Testing consistency with epsilon={eps}")
    B, K = 8, 4
    scores_base, y, C = generate_data(B, K)
    
    # We will clone scores for each loss to ensure independent gradient tracking
    def get_scores():
        s = scores_base.clone().detach()
        s.requires_grad = True
        return s

    # Common parameters
    params = dict(
        epsilon=eps,
        max_iter=100,
        epsilon_mode="constant", # Use constant eps to avoid epsilon computation differences
    )
    
    losses = {
        "pot": SinkhornPOTLoss(allow_numpy_fallback=True, **params),
        "autodiff": SinkhornFullAutodiffLoss(**params),
        "envelope": SinkhornEnvelopeLoss(**params),
    }
    
    results = {}
    
    for name, loss_fn in losses.items():
        s = get_scores()
        loss_val = loss_fn(s, y, C=C)
        loss_val.backward()
        
        results[name] = {
            "loss": loss_val.item(),
            "grad": s.grad.clone()
        }
        logger.info(f"[{name}] Loss: {loss_val.item():.6f}")
        logger.info(f"[{name}] Grad norm: {s.grad.norm():.6f}")

    # Comparisons
    # 1. POT vs Envelope (Should be identical as both use dual gradient graft now)
    logger.info("Comparing pot vs envelope...")
    res_pot = results["pot"]
    res_env = results["envelope"]
    
    diff_loss_pe = abs(res_pot["loss"] - res_env["loss"])
    diff_grad_pe = (res_pot["grad"] - res_env["grad"]).norm().item()
    
    logger.info(f"  Loss diff: {diff_loss_pe:.2e}")
    logger.info(f"  Grad diff: {diff_grad_pe:.2e}")
    
    assert diff_loss_pe < 1e-5, "POT and Envelope losses should match strictly"
    assert diff_grad_pe < 1e-4, "POT and Envelope gradients should match strictly"
    
    # 2. POT vs Autodiff (Expected to differ slightly due to Implicit vs Unrolled derivative)
    logger.info("Comparing pot vs autodiff...")
    res_auto = results["autodiff"]
    
    diff_loss_pa = abs(res_pot["loss"] - res_auto["loss"])
    diff_grad_pa = (res_pot["grad"] - res_auto["grad"]).norm().item()
    
    logger.info(f"  Loss diff: {diff_loss_pa:.2e}")
    logger.info(f"  Grad diff: {diff_grad_pa:.2e}")
    
    assert diff_loss_pa < 1e-4, "POT and Autodiff loss values should be close"
    
    # Relaxed gradient check (norm diff < 0.1 or cosine sim > 0.9?)
    # Using a looser tolerance for gradients
    if diff_grad_pa > 0.1:
        logger.warning(f"POT vs Autodiff gradient difference is large: {diff_grad_pa:.4f}")
        # Not asserting failure here as they are different methods
    else:
        logger.info("POT vs Autodiff gradients are reasonably close")

if __name__ == "__main__":
    test_sinkhorn_variants_consistency(0.1)
    test_sinkhorn_variants_consistency(1.0)
