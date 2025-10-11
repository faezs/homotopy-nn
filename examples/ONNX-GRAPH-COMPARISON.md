# Visual Comparison: Paper's Γ vs ONNX

## Example: ResNet Block with Skip Connection

### 1. High-Level Architecture (What you write in PyTorch)

```python
class ResBlock(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual  # Skip connection!
        out = F.relu(out)
        return out
```

### 2. Paper's Oriented Graph Γ (Coarse-Grained)

```
Architecture Level (Sections 1.2-1.3):

    Input (L₀)
       |
  ResBlock (L₁)  ← This is ONE vertex
       |
   Output (L₂)


Γ = { vertices: {L₀, L₁, L₂}
    , edges: {L₀→L₁, L₁→L₂}
    }
```

### 3. Paper's Fork Construction (Categorical Refinement)

```
Fork Construction (Section 1.3):

When L₁ has internal convergence, add fork vertices:

         Input
         /    \
       /        \
    Conv1    Identity
      |          |
    ReLU         |
      |          |
    Conv2        |
      |          |
      └─→ A★ ←───┘   ← Fork-star (convergence point)
           |
           A         ← Fork-tang (intermediate)
           |
        Output       ← Handle (result)


In Category C:
  ForkVertex = original Input
             | fork-star A★ conv
             | fork-tang A conv

  Edges: original x → fork-star a conv  (tips to star)
         fork-star a conv → fork-tang a conv  (star to tang)
         fork-tang a conv → original a  (tang to handle)
```

### 4. ONNX Graph (Fine-Grained Operations)

```
ONNX Computational Graph:

Input (tensor)
  |
  ├─────────────────┐
  |                 |
Conv(W1, b1)     Identity
  |                 |
ReLU              |
  |                 |
Conv(W2, b2)      |
  |                 |
  └─────> Add <─────┘  ← Multi-input node (convergence)
           |
         ReLU
           |
        Output


As ONNX GraphProto:

graph {
  node {
    name: "conv1"
    op_type: "Conv"
    input: ["input", "W1", "b1"]
    output: ["conv1_out"]
  }
  node {
    name: "relu1"
    op_type: "Relu"
    input: ["conv1_out"]
    output: ["relu1_out"]
  }
  node {
    name: "conv2"
    op_type: "Conv"
    input: ["relu1_out", "W2", "b2"]
    output: ["conv2_out"]
  }
  node {
    name: "add"
    op_type: "Add"
    input: ["conv2_out", "input"]  ← TWO INPUTS!
    output: ["add_out"]
  }
  node {
    name: "relu2"
    op_type: "Relu"
    input: ["add_out"]
    output: ["output"]
  }
}
```

## Side-by-Side Comparison

| Aspect | Paper's Γ | Fork Construction | ONNX |
|--------|-----------|-------------------|------|
| **Granularity** | Coarse (layers) | Medium (categorical) | Fine (operations) |
| **Vertices** | L₀, L₁, L₂ | + A★, A vertices | Conv, ReLU, Add nodes |
| **Convergence** | Implicit | Explicit forks | Multi-input nodes |
| **Purpose** | Architecture | Category theory | Execution |
| **Acyclic?** | Yes | Yes | Yes (DAG) |

## How They Connect

```
PyTorch/TensorFlow Code
         ↓ (define architecture)
    Oriented Graph Γ (paper)
         ↓ (add fork construction)
    Category C = C(Γ_fork)
         ↓ (compile/lower)
    ONNX GraphProto
         ↓ (optimize)
    Executable Graph
         ↓ (execute)
    Forward Pass (X^w functor)
         ↓ (differentiate)
    Backward Pass (natural transformation)
         ↓ (update)
    New Weights (flow in W functor)
```

## The Three Functors in ONNX Terms

### Activity Functor X^w (for fixed weights)

**Paper:**
```agda
X^w : Functor C₀(Γ) Set
X^w(Lₖ) = Xₖ  -- Set of possible activities
X^w(e : Lₖ → Lₖ₊₁) = λ x → σ(Wₖ·x + bₖ)  -- Weighted transformation
```

**ONNX:**
```python
# Forward pass through ONNX graph
session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": x})

# This IS the activity functor!
# Each node output = X^w applied to that layer
```

### Weight Functor W = Π

**Paper:**
```agda
W : Functor C₀(Γ) Set
W(Lₖ) = Πₖ = ∏_{i≥k} Wᵢ  -- Product of all weights ≥ k
W(e : Lₖ → Lₖ₊₁) = π  -- Forget Wₖ (projection)
```

**ONNX:**
```python
# All model parameters
graph.initializer = [
    TensorProto("W1", ...),  # ∈ Π₀
    TensorProto("W2", ...),  # ∈ Π₁
    ...
]

# As you move through layers, you "forget" earlier weights
# This is the forgetful functor W
```

### Total Functor X

**Paper (Equation 1.1):**
```agda
X : Functor C₀(Γ) Set
X(Lₖ) = Xₖ × Πₖ  -- Activity-weight pairs
X(e)(xₖ, wₖ) = (X^w_{k+1,k}(xₖ), πₖ₊₁(wₖ))  -- Apply and forget
```

**ONNX:**
```python
# Full state during forward pass
state = {
    "activities": {node: output for node, output in zip(nodes, activations)},
    "weights": {param: value for param, value in initializers},
}

# This is a point in the total functor X!
```

## Backpropagation as Natural Transformation

**Paper (Theorem 1.1):**
```agda
backprop : NatTrans W W
backprop(Lₖ)(wₖ) = wₖ - η·∇L(wₖ)  -- Gradient update
```

**ONNX + Optimizer:**
```python
# Compute gradients (this is the natural transformation!)
grads = compute_gradients(loss, parameters)

# Update weights (flow along the transformation)
for param, grad in zip(parameters, grads):
    param -= learning_rate * grad

# This IS backprop as a natural transformation W → W!
```

## Practical Example: From Agda to ONNX

### Step 1: Define in Agda (Architecture.agda)

```agda
-- ResNet block with fork construction
resnet-block : Fork-Category
resnet-block = record
  { ForkVertex = original input
               | fork-star add-point conv
               | fork-tang add-point conv
               | original output
  ; _≤ᶠ_ = ...  -- Ordering relation
  }
```

### Step 2: Compile to ONNX (hypothetical)

```python
# Generate ONNX from Agda definition
def compile_to_onnx(agda_graph: ForkCategory) -> onnx.ModelProto:
    nodes = []

    for vertex in agda_graph.vertices:
        if is_fork_star(vertex):
            # Fork-star becomes multi-input operation
            node = make_node('Add',
                            inputs=get_tines(vertex),
                            outputs=[vertex.name])
            nodes.append(node)
        else:
            # Regular vertex becomes operation
            node = make_node(get_op_type(vertex), ...)
            nodes.append(node)

    return make_graph(nodes, ...)
```

### Step 3: Execute

```python
# Load compiled ONNX
session = ort.InferenceSession("compiled_model.onnx")

# Run inference (following arrows in the graph!)
result = session.run(None, {"input": data})
```

## Key Insights

### 1. ONNX is MORE Fine-Grained

```
Paper's Γ:     [Input] → [ResBlock] → [Output]
                        (1 vertex)

ONNX:          [Input] → [Conv] → [ReLU] → [Conv] → [Add] → [ReLU] → [Output]
                         (6 nodes)
```

### 2. Fork Construction Handles Convergence

```
Paper:         x → A★ ← y     (explicit fork vertices)
                ↓
                A
                ↓

ONNX:          Add(x, y)      (implicit, just multi-input)
```

### 3. Both Are Oriented Graphs!

- **Acyclic**: Both enforce no cycles (feedforward)
- **Directed**: Information flows one way
- **Graph structure**: Vertices + Edges
- **Different purposes**: Semantics (paper) vs Execution (ONNX)

## Conclusion

**Yes, ONNX is an oriented graph!**

More precisely:
- ONNX is a **computational oriented graph** (fine-grained operations)
- Paper's Γ is an **architectural oriented graph** (coarse-grained layers)
- Fork construction bridges them (handles convergence categorically)
- Both map to the same category-theoretic framework

The categorical approach **subsumes ONNX** and gives it:
- Mathematical semantics (what does it mean?)
- Compositional structure (how to build complex networks?)
- Proof theory (can we verify properties?)
- Optimization theory (what's the best architecture?)

**Next step**: Build a compiler Agda → ONNX to make this connection executable!
