# Network Diagrams

Generates SVG diagrams for neural network architectures using Haskell's [diagrams](https://diagrams.github.io/) library.

## Building

```bash
# Install dependencies and build
cabal build

# Generate all diagrams
cabal run network-diagrams
# or
make
```

This will generate SVG files in `../images/`:
- `mlp.svg` - Simple chain MLP
- `convergent.svg` - Basic convergent network
- `convergent_fork.svg` - Convergent with fork construction
- `convergent_poset.svg` - Resulting poset X
- `complex.svg` - Complex multi-path network

## Diagrams Generated

1. **SimpleMLP**: Chain network `x₀ → h₁ → h₂ → y`
   - No convergence, total order poset

2. **ConvergentNetwork**: ResNet-like architecture
   - Two input branches converging at hidden layer
   - Shows fork construction (A★, A)
   - Diamond poset structure

3. **ComplexNetwork**: Multi-path with multiple convergence
   - Multiple forks at different layers
   - Tree forest poset structure

## Semiring Homomorphism Examples

The diagrams illustrate how neural architectures form a categorical algebra:

- **Sequential composition** (∘): Chain edges end-to-end
- **Parallel composition** (⊗): Multiple independent branches
- **Identity**: Single vertex (pass-through)

Any semiring homomorphism φ: Networks → ℝ gives evaluation semantics:
- `φ(G₁ ∘ G₂) = φ(G₁) * φ(G₂)` (sequential composition multiplies)
- `φ(G₁ ⊗ G₂) = φ(G₁) + φ(G₂)` (parallel composition adds)
- `φ(id) = 1` (identity is neutral)

Examples:
- Parameter count
- FLOPs
- Memory usage
- Information flow (mutual information)
