"""
Quick test for epsilon scheduling functionality.

Author
------
Warith Harchaoui <wharchaoui@nexton-group.com>
"""
import torch
from cost_aware_losses import SinkhornPOTLoss

# Test exponential decay schedule
print("Testing epsilon scheduling...")

# Create loss with exponential decay
loss_fn = SinkhornPOTLoss(
    epsilon_mode="offdiag_mean",
    epsilon_schedule="exponential_decay",
    schedule_start_mult=10.0,
    schedule_end_mult=0.1,
    total_epochs=10
)

# Simple cost matrix
C = torch.tensor([[0.0, 1.0], [2.0, 0.0]])

print(f"\nBase epsilon (from cost matrix): {loss_fn.compute_epsilon(C).item():.4f}")

# Test schedule over epochs
print("\nEpsilon values over 10 epochs:")
print("Epoch | Multiplier | Epsilon")
print("------|------------|--------")

for epoch in range(10):
    loss_fn.set_epoch(epoch)
    eps = loss_fn.compute_epsilon(C).item()
    mult = loss_fn._compute_schedule_multiplier()
    print(f"  {epoch}   |   {mult:6.3f}   | {eps:.4f}")

# Verify: should be approximately 10x base at epoch 0
# and approximately 0.1x base at epoch 9
base_eps = loss_fn.compute_epsilon(C).item() / loss_fn._compute_schedule_multiplier()
loss_fn.set_epoch(0)
eps_start = loss_fn.compute_epsilon(C).item()
loss_fn.set_epoch(9)
eps_end = loss_fn.compute_epsilon(C).item()

print(f"\nVerification:")
print(f"Base epsilon: {base_eps:.4f}")
print(f"Epoch 0:  {eps_start:.4f}  (should be ~{10*base_eps:.4f}, ratio: {eps_start/base_eps:.2f})")
print(f"Epoch 9:  {eps_end:.4f}  (should be ~{0.1*base_eps:.4f}, ratio: {eps_end/base_eps:.2f})")

if abs(eps_start - 10*base_eps) < 0.01 and abs(eps_end - 0.1*base_eps) < 0.01:
    print("\n✓ Epsilon scheduling test PASSED!")
else:
    print("\n✗ Epsilon scheduling test FAILED!")
    
print("\n" + "="*60)
print("Testing without schedule (backward compatibility)...")
loss_fn_static = SinkhornPOTLoss(epsilon_mode="offdiag_mean")

eps_values = []
for epoch in range(5):
    loss_fn_static.set_epoch(epoch)  # Should have no effect
    eps_values.append(loss_fn_static.compute_epsilon(C).item())

if all(abs(e - eps_values[0]) < 1e-6 for e in eps_values):
    print("✓ Static epsilon (no schedule) works correctly!")
else:
    print("✗ Static epsilon test failed!")
    
print("\nAll tests completed!")
