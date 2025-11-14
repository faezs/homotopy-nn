"""
Example: Attention Head Redundancy Detection

Uses conversion rates to find redundant attention heads.
"""

import numpy as np
from src.interpretability import ResourceNetwork, attention_redundancy, NeuralResource

def create_transformer_resources(n_layers=12, n_heads=12, d_model=768):
    """
    Create mock transformer resources for demonstration.

    Args:
        n_layers: Number of transformer layers
        n_heads: Number of attention heads per layer
        d_model: Model dimension

    Returns:
        List of NeuralResource objects representing attention heads
    """
    resources = []

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            # Simulate attention patterns
            # Some heads will be similar (redundant)
            if head_idx % 3 == 0:
                # "Position heads" - similar patterns
                pattern = np.eye(64) + 0.1 * np.random.randn(64, 64)
            elif head_idx % 3 == 1:
                # "Content heads" - similar patterns
                pattern = np.ones((64, 64)) / 64 + 0.1 * np.random.randn(64, 64)
            else:
                # "Mixed heads" - diverse patterns
                pattern = np.random.randn(64, 64)

            # Normalize to probabilities
            pattern = np.abs(pattern)
            pattern = pattern / pattern.sum(axis=-1, keepdims=True)

            resource = NeuralResource(
                name=f"L{layer_idx}H{head_idx}",
                activations=pattern.flatten(),
                weights=np.random.randn(d_model // n_heads, d_model // n_heads),
                metadata={
                    'layer': layer_idx,
                    'head': head_idx,
                    'type': ['position', 'content', 'mixed'][head_idx % 3]
                }
            )
            resources.append(resource)

    return resources


def main():
    print("=" * 70)
    print("Example: Attention Head Redundancy Detection")
    print("=" * 70)
    print()

    # Create transformer model
    print("Creating transformer with 12 layers, 12 heads per layer...")
    resources = create_transformer_resources(n_layers=12, n_heads=12)
    model = ResourceNetwork(resources)
    print(f"Total attention heads: {len(resources)}\n")

    # Compute redundancy matrix for first layer
    print("Computing redundancy matrix for Layer 0...")
    layer_0_heads = resources[:12]  # First 12 heads (layer 0)
    layer_0_network = ResourceNetwork(layer_0_heads)

    redundancy_matrix = attention_redundancy(layer_0_network, threshold=0.7)

    print("\nRedundancy matrix (ρ_{i→j}):")
    print("Head", end="  ")
    for j in range(12):
        print(f"{j:5d}", end=" ")
    print()
    print("-" * 80)

    for i in range(12):
        print(f"{i:4d}", end="  ")
        for j in range(12):
            if i == j:
                print("  -  ", end=" ")
            else:
                rate = redundancy_matrix[i, j]
                print(f"{rate:5.2f}", end=" ")
        print()

    print("\n" + "=" * 70)
    print("Finding highly redundant pairs (ρ ≥ 0.75)...")
    print("=" * 70)

    redundant_pairs = []
    for i in range(12):
        for j in range(i + 1, 12):
            rate_ij = redundancy_matrix[i, j]
            rate_ji = redundancy_matrix[j, i]
            avg_rate = (rate_ij + rate_ji) / 2

            if avg_rate >= 0.75:
                redundant_pairs.append((i, j, avg_rate))

    redundant_pairs.sort(key=lambda x: x[2], reverse=True)

    if redundant_pairs:
        print("\nRedundant pairs found:")
        for i, j, rate in redundant_pairs:
            head_i = layer_0_heads[i]
            head_j = layer_0_heads[j]
            print(f"  Head {i} ({head_i.metadata['type']}) ↔ " +
                  f"Head {j} ({head_j.metadata['type']}): ρ = {rate:.3f}")
    else:
        print("\nNo highly redundant pairs found (threshold = 0.75)")

    print("\n" + "=" * 70)
    print("Pruning Strategy")
    print("=" * 70)

    if redundant_pairs:
        to_prune = set()
        for i, j, rate in redundant_pairs:
            if i not in to_prune and j not in to_prune:
                # Keep head i, prune head j (arbitrary choice)
                to_prune.add(j)

        print(f"\nSuggested heads to prune: {sorted(to_prune)}")
        print(f"Remaining heads: {12 - len(to_prune)} / 12")
        print(f"Compression ratio: {(12 - len(to_prune)) / 12:.1%}")
    else:
        print("\nNo pruning recommended - all heads are distinct")

    print("\n" + "=" * 70)
    print("Cross-Layer Analysis")
    print("=" * 70)

    # Check redundancy across layers
    print("\nChecking if heads repeat across layers...")
    cross_layer_redundancy = []

    for layer_i in range(0, 12, 3):  # Check every 3rd layer
        for layer_j in range(layer_i + 3, 12, 3):
            for head_idx in range(12):
                idx_i = layer_i * 12 + head_idx
                idx_j = layer_j * 12 + head_idx

                rate = model.conversion_rate(idx_i, idx_j)
                if rate >= 0.8:
                    cross_layer_redundancy.append((
                        layer_i, layer_j, head_idx, rate
                    ))

    if cross_layer_redundancy:
        print("\nHeads that repeat across layers:")
        for l_i, l_j, h, rate in cross_layer_redundancy[:5]:
            print(f"  Layer {l_i} Head {h} ↔ Layer {l_j} Head {h}: " +
                  f"ρ = {rate:.3f}")
    else:
        print("\nNo significant cross-layer redundancy found")

    print("\n" + "=" * 70)
    print("Interpretation")
    print("=" * 70)
    print("""
Resource-theoretic view of redundancy:

- High ρ_{A→B} means: "Resource A contains everything in resource B"
- Symmetric high rates (ρ_{A→B} ≈ ρ_{B→A} ≈ 1) means: "A and B are equivalent"
- Pruning strategy: Keep one, remove the other
- Conservation: Total information preserved (Kirchhoff's law)

This approach is principled because:
1. Based on categorical structure (not ad-hoc heuristics)
2. Respects composition (summing functors)
3. Obeys conservation laws
4. Generalizes to any resource type
""")


if __name__ == "__main__":
    main()
