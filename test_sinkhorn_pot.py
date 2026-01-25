"""
Test script for SinkhornPOTLoss

Author
------
Warith Harchaoui <wharchaoui@nexton-group.com>
"""
import torch
from cost_aware_losses import SinkhornPOTLoss

def test_sinkhorn_pot_loss():
    """Test basic functionality of SinkhornPOTLoss."""
    print("Testing SinkhornPOTLoss...")
    
    # Create test data
    batch_size = 8
    num_classes = 4
    
    scores = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Create a simple cost matrix (uniform off-diagonal costs)
    C = torch.ones(num_classes, num_classes)
    C.fill_diagonal_(0.0)
    
    print(f"Scores shape: {scores.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Cost matrix shape: {C.shape}")
    
    # Create loss function
    loss_fn = SinkhornPOTLoss(max_iter=50)
    
    # Compute loss
    print("\nComputing loss...")
    loss = loss_fn(scores, targets, C=C)
    
    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    
    # Test backward pass
    print("\nTesting backward pass...")
    loss.backward()
    
    print(f"Scores gradient shape: {scores.grad.shape}")
    print(f"Scores gradient norm: {scores.grad.norm().item():.6f}")
    
    print("\n✓ SinkhornPOTLoss test passed!")
    

def test_batched_cost_matrix():
    """Test with batched cost matrices."""
    print("\n" + "="*60)
    print("Testing with batched cost matrix...")
    
    batch_size = 4
    num_classes = 3
    
    scores = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Create per-example cost matrices
    C = torch.rand(batch_size, num_classes, num_classes)
    for i in range(batch_size):
        C[i].fill_diagonal_(0.0)
    
    print(f"Batched cost matrix shape: {C.shape}")
    
    loss_fn = SinkhornPOTLoss(max_iter=50)
    loss = loss_fn(scores, targets, C=C)
    
    print(f"Loss value: {loss.item():.6f}")
    
    loss.backward()
    print(f"Gradients computed successfully")
    print("\n✓ Batched cost matrix test passed!")


def test_comparison_with_envelope():
    """Compare POT implementation with custom envelope implementation."""
    print("\n" + "="*60)
    print("Comparing SinkhornPOTLoss with SinkhornEnvelopeLoss...")
    
    from cost_aware_losses import SinkhornEnvelopeLoss
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    batch_size = 8
    num_classes = 4
    
    scores = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    C = torch.ones(num_classes, num_classes)
    C.fill_diagonal_(0.0)
    
    # POT implementation
    loss_pot = SinkhornPOTLoss(max_iter=100, epsilon_mode="constant", epsilon=0.1)
    loss_value_pot = loss_pot(scores, targets, C=C)
    
    # Custom envelope implementation
    scores_envelope = scores.clone().detach().requires_grad_(True)
    loss_envelope = SinkhornEnvelopeLoss(max_iter=100, epsilon_mode="constant", epsilon=0.1)
    loss_value_envelope = loss_envelope(scores_envelope, targets, C=C)
    
    print(f"POT loss: {loss_value_pot.item():.6f}")
    print(f"Envelope loss: {loss_value_envelope.item():.6f}")
    print(f"Absolute difference: {abs(loss_value_pot.item() - loss_value_envelope.item()):.6f}")
    
    # Note: Small differences are expected due to numerical precision and stopping criteria
    print("\n✓ Comparison test completed!")


if __name__ == "__main__":
    test_sinkhorn_pot_loss()
    test_batched_cost_matrix()
    test_comparison_with_envelope()
    print("\n" + "="*60)
    print("All tests passed successfully! ✓")
