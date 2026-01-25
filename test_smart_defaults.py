"""
Test smart architecture defaults computation.

Author
------
Warith Harchaoui <wharchaoui@nexton-group.com>
"""

# Test the heuristics with IEEE-CIS dataset dimensions
def compute_smart_architecture_defaults(input_dim: int, n_train: int, n_classes: int = 2):
    """Compute smart architecture defaults using classic ML heuristics."""
    layer1 = max(int(input_dim * 0.67), n_classes * 2)
    layer2 = max(int(input_dim * 0.33), n_classes * 2)
    
    def round_to_nice(x: int) -> int:
        if x >= 256:
            return ((x + 31) // 32) * 32
        elif x >= 64:
            return ((x + 15) // 16) * 16
        else:
            return ((x + 7) // 8) * 8
    
    layer1 = round_to_nice(layer1)
    layer2 = round_to_nice(layer2)
    
    if layer2 >= layer1:
        layer2 = max(layer1 // 2, n_classes * 2)
        layer2 = round_to_nice(layer2)
    
    total_params = input_dim * layer1 + layer1 * layer2 + layer2 * n_classes
    capacity_ratio = total_params / n_train
    
    if capacity_ratio > 2.0:
        scale_factor = (2.0 / capacity_ratio) ** 0.5
        layer1 = round_to_nice(int(layer1 * scale_factor))
        layer2 = round_to_nice(int(layer2 * scale_factor))
    
    samples_per_feature = n_train / input_dim
    if samples_per_feature < 10:
        dropout_rate = 0.5
    elif samples_per_feature < 50:
        dropout_rate = 0.3
    elif samples_per_feature < 200:
        dropout_rate = 0.2
    else:
        dropout_rate = 0.1
    
    return (layer1, layer2), dropout_rate

# Test with IEEE-CIS-like dimensions
print("Smart Architecture Defaults Test\n" + "="*50)

test_cases = [
    ("IEEE-CIS (typical)", 544, 413378),
    ("Small dataset", 100, 5000),
    ("Large features", 2000, 100000),
    ("Very small", 50, 1000),
]

for name, input_dim, n_train in test_cases:
    hidden_dims, dropout = compute_smart_architecture_defaults(input_dim, n_train)
    params = input_dim * hidden_dims[0] + hidden_dims[0] * hidden_dims[1] + hidden_dims[1] * 2
    capacity_ratio = params / n_train
    samples_per_feat = n_train / input_dim
    
    print(f"\n{name}:")
    print(f"  Input: {input_dim} features, {n_train:,} samples")
    print(f"  Samples/feature: {samples_per_feat:.1f}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Dropout: {dropout:.2f}")
    print(f"  Est. params: {params:,} (ratio: {capacity_ratio:.2f})")

print("\n" + "="*50)
print("✓ All test cases computed successfully!")
