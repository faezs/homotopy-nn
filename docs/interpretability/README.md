# Neural Interpretability via Resource Functors

A categorical approach to neural network interpretability using resource theory.

## Overview

This library applies the resource-theoretic framework from the homotopy-nn formalization to practical interpretability tasks. Based on the formalization in `src/Neural/Resources/`.

## Key Concepts

### Resource Theory for Interpretability

A **resource theory** is a symmetric monoidal category (R, ◦, ⊗, I) where:
- Objects: Neural resources (activations, weights, information)
- Morphisms: Resource conversions (transformations that preserve structure)
- ⊗: Parallel composition (independent resources)
- ◦: Sequential composition (resource transformation)

### Convertibility and Rates

For resources A, B, the **conversion rate** ρ_{A→B} measures:

```
ρ_{A→B} = sup { m/n | n·A ⪰ m·B }
```

This tells us: "How many units of B can we extract from n units of A?"

**Interpretability applications**:
- Attention head redundancy: ρ_{head_i → head_j}
- Layer importance: ρ_{layer_k → output}
- Information flow: ρ_{input → hidden_i}

### S-Measuring (Section 3.2.2)

A **measuring homomorphism** M: R → (ℝ, +, ≥, 0) assigns real-valued quantities:
- Preserves tensor: M(A ⊗ B) = M(A) + M(B)
- Preserves order: A ⪰ B ⟹ M(A) ≥ M(B)

**Theorem 5.6**: ρ_{A→B} · M(B) ≤ M(A)

**Examples**:
1. **Energy measure**: Power consumption
2. **Entropy measure**: H(p) = -Σ p_i log(p_i)
3. **Parameter count**: Total trainable parameters
4. **FLOPs**: Computational cost

## Library Structure

```
interpretability/
├── src/
│   ├── resource.py          # Core resource theory
│   ├── convertibility.py    # Conversion rates
│   ├── measuring.py          # Measuring homomorphisms
│   ├── summing_functor.py   # Network summing functors
│   └── interpretability.py  # High-level API
├── examples/
│   ├── attention_redundancy.py
│   ├── layer_importance.py
│   └── information_flow.py
├── tests/
│   └── test_resource.py
├── README.md
└── setup.py
```

## Installation

```bash
cd docs/interpretability
pip install -e .
```

## Quick Start

```python
from interpretability import ResourceNetwork, entropy_measure, conversion_rate

# Load a transformer model
net = ResourceNetwork.from_pretrained("gpt2")

# Measure entropy at each layer
for layer in net.layers:
    H = entropy_measure(layer.activations)
    print(f"Layer {layer.name}: H = {H:.3f}")

# Compute conversion rates between attention heads
heads = net.attention_heads(layer_idx=0)
for i, head_i in enumerate(heads):
    for j, head_j in enumerate(heads):
        if i != j:
            rate = conversion_rate(head_i, head_j)
            print(f"ρ(head_{i} → head_{j}) = {rate:.3f}")

# Find redundant heads (high convertibility)
redundant = net.find_redundant_resources(threshold=0.9)
print(f"Redundant heads: {redundant}")
```

## Core Principles

### 1. Compositionality

Networks compose via summing functors Σ_C(G):
- Objects: Subsets S ⊆ Vertices(G)
- Morphisms: S ⊆ T with compatible structure

**Interpretability**: Understand parts by understanding composition.

### 2. Conservation Laws

Kirchhoff's law (Proposition 2.10):
```
Σ_{edges into v} flow = Σ_{edges out of v} flow
```

**Interpretability**: Track information flow via conservation.

### 3. Optimal Resource Assignment

Adjunction β ⊣ ρ (Section 3.3):
- Left adjoint β: Optimal constructor
- Right adjoint ρ: Resource extractor

**Interpretability**: Find minimal resources needed for task.

## Mathematical Background

Based on:
- **Marcolli & Manin (2020)**: Sections 3.2-3.3 (Resource theory)
- **Belfiore & Bennequin (2022)**: Section 1 (Network summing functors)
- **Fritz (2017)**: Resource theories as symmetric monoidal categories

## Examples

### Attention Head Redundancy

```python
# Identify redundant attention heads via conversion rates
from interpretability import attention_redundancy

model = load_model("bert-base")
redundancy_matrix = attention_redundancy(model, threshold=0.85)

# Prune redundant heads
pruned_model = model.prune_heads(redundancy_matrix)
```

### Layer Importance via Information Flow

```python
# Measure information flow from input to output
from interpretability import information_flow, mutual_information_measure

flow = information_flow(
    model,
    measure=mutual_information_measure,
    source="input",
    target="output"
)

# Layers with low flow are less important
print(f"Layer importance: {flow}")
```

### Minimal Circuit Discovery

```python
# Find minimal subnetwork for a task (optimal resource assignment)
from interpretability import minimal_circuit

circuit = minimal_circuit(
    model,
    task_examples=task_data,
    optimality_criterion="minimal_energy"
)

print(f"Minimal circuit uses {len(circuit.nodes)} neurons")
```

## API Reference

See `src/interpretability.py` for full API documentation.

## Citation

```bibtex
@software{homotopy_nn_interpretability,
  title     = {Neural Interpretability via Resource Functors},
  author    = {Shakil, Faez},
  year      = {2025},
  url       = {https://github.com/faezs/homotopy-nn},
  note      = {Categorical interpretability library based on resource theory}
}
```

## License

MIT
