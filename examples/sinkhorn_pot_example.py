"""
Example: Using SinkhornPOTLoss for Cost-Aware Classification
=============================================================

This example demonstrates how to use the SinkhornPOTLoss class, which
leverages the Python Optimal Transport (POT) library for efficient and
numerically stable Sinkhorn optimal transport computations.

Author
------
Warith Harchaoui <wharchaoui@nexton-group.com>

Reference: https://pythonot.github.io/
"""

import torch
import torch.nn as nn
from cost_aware_losses import SinkhornPOTLoss


def create_semantic_cost_matrix(num_classes: int, seed: int = 42) -> torch.Tensor:
    """
    Create a simple semantic cost matrix based on class distances.
    
    In practice, you might use word embeddings (Word2Vec, GloVe) or
    other semantic similarity measures.
    
    Parameters
    ----------
    num_classes : int
        Number of classes.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    torch.Tensor
        Cost matrix of shape (num_classes, num_classes).
    """
    torch.manual_seed(seed)
    
    # Random semantic embeddings (in practice, use real embeddings)
    embeddings = torch.randn(num_classes, 10)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    
    # Compute pairwise distances
    C = torch.cdist(embeddings, embeddings, p=2)
    
    # Ensure diagonal is zero (no cost for correct prediction)
    C.fill_diagonal_(0.0)
    
    return C


class SimpleClassifier(nn.Module):
    """Simple 2-layer classifier for demonstration."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    """Main demonstration of SinkhornPOTLoss."""
    
    # Configuration
    input_dim = 20
    num_classes = 5
    batch_size = 16
    num_epochs = 5
    
    print("=" * 60)
    print("SinkhornPOTLoss Example")
    print("=" * 60)
    
    # Create model and loss function
    model = SimpleClassifier(input_dim, num_classes)
    
    # Create cost matrix (e.g., based on semantic similarity)
    C = create_semantic_cost_matrix(num_classes)
    print(f"\nCost matrix shape: {C.shape}")
    print(f"Cost matrix:\n{C.numpy()}")
    
    # Initialize SinkhornPOTLoss
    # - max_iter controls Sinkhorn convergence (higher = more accurate)
    # - epsilon_mode determines how regularization is computed
    # - label_smoothing adds stability
    loss_fn = SinkhornPOTLoss(
        max_iter=100,
        epsilon_mode="offdiag_mean",  # Adaptive ε based on cost matrix
        epsilon_scale=0.1,             # Scale factor for ε
        label_smoothing=1e-3,
        stopThr=1e-9,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"\nTraining classifier with {num_epochs} epochs...")
    print("-" * 60)
    
    # Training loop
    for epoch in range(num_epochs):
        # Generate synthetic data
        X = torch.randn(batch_size, input_dim)
        y = torch.randint(0, num_classes, (batch_size,))
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(X)
        
        # Compute cost-aware loss
        loss = loss_fn(logits, y, C=C)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss.item():.6f}")
    
    print("-" * 60)
    print("\n✓ Training completed successfully!")
    
    # Demonstrate inference
    print("\n" + "=" * 60)
    print("Inference Example")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        # Generate test sample
        X_test = torch.randn(1, input_dim)
        logits_test = model(X_test)
        
        # Get predictions
        probs = torch.softmax(logits_test, dim=1)
        pred_class = probs.argmax(dim=1).item()
        
        print(f"\nPredicted class: {pred_class}")
        print(f"Class probabilities: {probs.squeeze().numpy()}")
    
    print("\n" + "=" * 60)
    print("Key Benefits of SinkhornPOTLoss:")
    print("=" * 60)
    print("1. Uses POT's optimized Sinkhorn solver (battle-tested)")
    print("2. Naturally handles misclassification costs via cost matrix C")
    print("3. Provides smooth gradients via envelope theorem")
    print("4. Adaptive regularization based on cost matrix statistics")
    print("5. Numerical stability from POT's implementation")
    print("=" * 60)


if __name__ == "__main__":
    main()
