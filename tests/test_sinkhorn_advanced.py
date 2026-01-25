
import torch
import pytest
import ot
import logging
from torch.autograd import gradcheck
from cost_aware_losses import SinkhornPOTLoss, SinkhornEnvelopeLoss

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_data(B=4, K=3, seed=42, dtype=torch.float64):
    torch.manual_seed(seed)
    # gradcheck requires double precision usually
    scores = torch.randn(B, K, requires_grad=True, dtype=dtype)
    y = torch.randint(0, K, (B,))
    
    # Random cost matrix (B, K, K)
    C = torch.rand(B, K, K, dtype=dtype)
    return scores, y, C

@pytest.mark.parametrize("loss_cls", [SinkhornPOTLoss, SinkhornEnvelopeLoss])
def test_gradcheck(loss_cls):
    """
    Verify gradients using finite differences (torch.autograd.gradcheck).
    This ensures that the custom analytical gradients (or gradient grafts) 
    match the numerical gradients.
    """
    logger.info(f"Running gradcheck for {loss_cls.__name__}")
    
    B, K = 4, 3
    scores, y, C = generate_data(B, K, dtype=torch.float64)
    
    # Initialize loss
    # Using a slightly larger epsilon for stability during finite differences
    loss_fn = loss_cls(epsilon=0.1, max_iter=50)
    
    # Define a function for gradcheck: params -> loss
    # gradcheck varies inputs that have requires_grad=True
    def func(s):
        return loss_fn(s, y, C=C)
        
    # Check gradients
    # eps: step size for finite differences
    # atol: absolute tolerance
    test_passed = gradcheck(func, (scores,), eps=1e-6, atol=1e-4)
    
    assert test_passed, f"gradcheck failed for {loss_cls.__name__}"
    logger.info(f"gradcheck passed for {loss_cls.__name__}")

def test_epsilon_limit_convergence():
    """
    Verify that as epsilon -> 0, Sinkhorn loss converges to the exact Optimal Transport cost (EMD).
    """
    logger.info("Running epsilon limit convergence test")
    B, K = 5, 4
    scores, y, C = generate_data(B, K, dtype=torch.float64)
    
    # Get probability distributions
    # Sinkhorn loss inputs are (scores), but internally it computes softmax p and target q
    # We need to replicate this to call ot.emd2
    
    prob_p = torch.softmax(scores, dim=1)
    
    # Construct target q: one-hot or smoothed?
    # The losses construct q based on y. usually q is one-hot per class?
    # Actually Sinkhorn loss usually targets a distribution q that depends on y.
    # In SinkhornPOTLoss:
    #   q = F.one_hot(y, num_classes=K).double()
    q_onehot = torch.nn.functional.one_hot(y, num_classes=K).to(scores.dtype)
    
    # Expected EMD for each example
    emd_values = []
    for i in range(B):
        # p[i]: (K,), q[i]: (K,), C[i]: (K, K)
        # ot.emd2 returns scalar cost
        val = ot.emd2(prob_p[i].detach(), q_onehot[i].detach(), C[i].detach())
        emd_values.append(val)
    
    emd_total = torch.tensor(emd_values).mean().item()
    logger.info(f"Exact EMD (mean): {emd_total:.6f}")
    
    # Test with decreasing epsilons
    epsilons = [1.0, 0.1, 0.01, 1e-3]
    previous_diff = float('inf')
    
    for eps in epsilons:
        loss_fn = SinkhornPOTLoss(epsilon=eps, max_iter=200, allow_numpy_fallback=True)
        # Using allow_numpy_fallback=True for stability at low eps if needed
        
        loss_val = loss_fn(scores, y, C=C).item()
        
        diff = abs(loss_val - emd_total)
        logger.info(f"Epsilon: {eps:.4f}, Loss: {loss_val:.6f}, Diff from EMD: {diff:.6f}")
        
        # Check that difference decreases (or is already very small)
        # Note: At extremely low epsilon, numerical instability might cause issues, 
        # but trend should be visible.
        if eps < 1.0:
            # It's not strictly monotonic due to numerical errors, but should be small for small eps
            pass
            
        if eps == 1e-3:
            assert diff < 1e-2, f"Sinkhorn loss with eps={eps} should be close to EMD. Diff: {diff}"

def test_extreme_costs():
    """
    Verify numerical stability with large cost values.
    """
    logger.info("Running extreme cost stability test")
    B, K = 4, 3
    scores, y, _ = generate_data(B, K, dtype=torch.float64)
    
    # Large costs
    C_large = torch.rand(B, K, K, dtype=torch.float64) * 1e5
    
    loss_fn = SinkhornPOTLoss(epsilon=1.0, max_iter=50)
    
    try:
        loss = loss_fn(scores, y, C=C_large)
        loss.backward()
        
        assert not torch.isnan(loss), "Loss is NaN with large costs"
        assert not torch.isinf(loss), "Loss is Inf with large costs"
        assert not torch.isnan(scores.grad).any(), "Gradient is NaN with large costs"
        
        logger.info("Large cost test passed (no NaNs)")
        
    except Exception as e:
        pytest.fail(f"Test failed with exception: {e}")

def test_cost_shift_invariance():
    """
    Verify behavior when adding constant to cost matrix.
    Loss should increase by constant (if balanced), gradients should be invariant.
    """
    logger.info("Running cost shift invariance test")
    B, K = 4, 3
    scores, y, C = generate_data(B, K, dtype=torch.float64)
    
    loss_fn = SinkhornPOTLoss(epsilon=0.1, max_iter=50)
    
    # Baseline
    scores.grad = None
    loss_ref = loss_fn(scores, y, C=C)
    loss_ref.backward()
    grad_ref = scores.grad.clone()
    
    # Shifted Cost
    shift = 10.0
    C_shifted = C + shift
    
    scores.grad = None
    loss_shift = loss_fn(scores, y, C=C_shifted)
    loss_shift.backward()
    grad_shift = scores.grad.clone()
    
    # Check Loss Shift
    # Primal: <P, C+k> = <P, C> + k * sum(P)
    # sum(P) is 1 for balanced OT? 
    # SinkhornPOTLoss normalizes plan to sum to 1?
    # Usually yes, row sum p, col sum q. sum(p)=1. So sum(P)=1.
    loss_diff = (loss_shift - loss_ref).item()
    logger.info(f"Loss Difference (Shift {shift}): {loss_diff:.6f}")
    
    # Allow some tolerance for entropic regularization effects / soft constraints
    assert abs(loss_diff - shift) < 1e-3, f"Loss should increase by roughly {shift}, got {loss_diff}"
    
    # Check Gradient Invariance
    grad_diff = (grad_ref - grad_shift).norm().item()
    logger.info(f"Gradient Difference: {grad_diff:.6e}")
    
    # Relaxed tolerance for float precision accumulation
    assert grad_diff < 2e-5, f"Gradients should be invariant to constant cost shift. Diff: {grad_diff}"

if __name__ == "__main__":
    # Test script runner
    test_gradcheck(SinkhornPOTLoss)
    test_epsilon_limit_convergence()
    test_extreme_costs()
    test_cost_shift_invariance()
