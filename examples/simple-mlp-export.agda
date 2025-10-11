{-# OPTIONS --no-import-sorts #-}

{-|
# Simple MLP Export Example

Demonstrates exporting a simple 2-layer feedforward network:
- Input layer (784 neurons, MNIST-style)
- Hidden layer (256 neurons, ReLU activation)
- Output layer (10 neurons, Softmax for classification)

Graph structure:
```
Vertices: 0 (input), 1 (hidden), 2 (output)
Edges: 0 (input→hidden), 1 (hidden→output)
```

ONNX operations:
- Vertex 0: Input (placeholder)
- Vertex 1: MatMul + ReLU
- Vertex 2: MatMul + Softmax
-}

module examples.simple-mlp-export where

open import 1Lab.Prelude
open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin; fzero; fsuc; Fin-cases)
open import Data.List.Base
open import Data.String.Base using (String)

open import Neural.Base using (DirectedGraph; vertices; edges; source; target)
open import Neural.Compile.ONNX
open import Neural.Compile.ONNX.Export

--------------------------------------------------------------------------------
-- Simple MLP Graph
--------------------------------------------------------------------------------

{-|
Define the graph structure for a simple MLP:
- 3 vertices: input, hidden, output
- 2 edges: input→hidden, hidden→output
-}
postulate simple-mlp-graph : DirectedGraph

postulate
  simple-mlp-vertices : vertices simple-mlp-graph ≡ 3
  simple-mlp-edges : edges simple-mlp-graph ≡ 2

{-|
Annotate the graph with ONNX operation types and tensor shapes.
-}
simple-mlp-annotations : GraphAnnotations simple-mlp-graph
simple-mlp-annotations = record
  { vertex-op-type = λ v →
      -- Use Fin-cases to pattern match on vertex index
      Fin-cases "Input"
        (Fin-cases "Gemm"  -- General Matrix Multiply (includes bias)
          (Fin-cases "Softmax" (λ _ → "Unknown"))
        ) v

  ; edge-shape = λ e →
      -- Edge 0: input→hidden (784 → 256)
      -- Edge 1: hidden→output (256 → 10)
      Fin-cases (784 ∷ 256 ∷ [])
        (Fin-cases (256 ∷ 10 ∷ []) (λ _ → []))
        e

  ; edge-elem-type = λ _ → FLOAT  -- All float32 tensors

  ; vertex-attributes = λ v →
      -- Vertex 0 (Input): no attributes
      -- Vertex 1 (Gemm): ReLU activation via transB, alpha, beta
      -- Vertex 2 (Softmax): axis = 1
      Fin-cases []
        (Fin-cases
          (record { name = "alpha" ; value = attr-float 1.0 } ∷
           record { name = "beta" ; value = attr-float 1.0 } ∷
           record { name = "transB" ; value = attr-int 0 } ∷ [])
          (Fin-cases
            (record { name = "axis" ; value = attr-int 1 } ∷ [])
            (λ _ → [])
          )
        ) v

  ; graph-inputs = fzero ∷ []  -- Vertex 0 is input

  ; graph-outputs = fsuc (fsuc fzero) ∷ []  -- Vertex 2 is output

  ; model-name = "simple-mlp"

  ; model-doc = "Simple 2-layer MLP for MNIST classification: 784→256→10"

  ; producer = "homotopy-nn"
  }

--------------------------------------------------------------------------------
-- Export to ONNX
--------------------------------------------------------------------------------

{-|
Export the graph to ONNX ModelProto.
This produces an in-memory ONNX model that can be serialized to JSON.
-}
simple-mlp-onnx : ModelProto
simple-mlp-onnx = export-to-onnx simple-mlp-graph simple-mlp-annotations

{-|
Verify the export is valid (has inputs and outputs).
-}
simple-mlp-is-valid : is-valid-for-export simple-mlp-graph simple-mlp-annotations ≡ true
simple-mlp-is-valid = refl

{-|
Count convergence points (multi-input nodes).
This network is a simple chain, so should have 0 convergence points.
-}
simple-mlp-convergence-count : count-multi-input-nodes simple-mlp-graph ≡ 0
simple-mlp-convergence-count = {!!}  -- TODO: Prove using graph structure

--------------------------------------------------------------------------------
-- Documentation
--------------------------------------------------------------------------------

{-|
## Usage Instructions

### Step 1: Type-check this file
```bash
agda --library-file=./libraries examples/simple-mlp-export.agda
```

### Step 2: Serialize to JSON
(TODO: Implement JSON serialization in Agda or use reflection)

For now, manually create JSON representation following this structure:
```json
{
  "ir-version": 9,
  "opset-import": [{"domain": "", "version": 17}],
  "producer-name": "homotopy-nn",
  "producer-version": "1.0.0",
  "domain": "neural.homotopy",
  "model-version": 1,
  "doc": "Simple 2-layer MLP for MNIST classification: 784→256→10",
  "graph": {
    "name": "simple-mlp",
    "nodes": [
      {
        "op-type": "Gemm",
        "inputs": ["edge_0_0→1", "hidden_weight", "hidden_bias"],
        "outputs": ["hidden_output"],
        "attributes": [
          {"name": "alpha", "value": {"attr-float": 1.0}},
          {"name": "beta", "value": {"attr-float": 1.0}}
        ],
        "name": "node_1",
        "domain": ""
      },
      {
        "op-type": "Relu",
        "inputs": ["hidden_output"],
        "outputs": ["hidden_relu"],
        "attributes": [],
        "name": "relu_1",
        "domain": ""
      },
      {
        "op-type": "Gemm",
        "inputs": ["hidden_relu", "output_weight", "output_bias"],
        "outputs": ["edge_1_1→2"],
        "attributes": [
          {"name": "alpha", "value": {"attr-float": 1.0}},
          {"name": "beta", "value": {"attr-float": 1.0}}
        ],
        "name": "node_2_gemm",
        "domain": ""
      },
      {
        "op-type": "Softmax",
        "inputs": ["edge_1_1→2"],
        "outputs": ["final_output"],
        "attributes": [{"name": "axis", "value": {"attr-int": 1}}],
        "name": "node_2",
        "domain": ""
      }
    ],
    "inputs": [
      {
        "name": "node_0_input",
        "type": {
          "tensor-type": {
            "elem-type": "FLOAT",
            "shape": [{"dim-value": 784}, {"dim-value": 256}]
          }
        },
        "doc": "Input from vertex 0"
      }
    ],
    "outputs": [
      {
        "name": "node_2_output",
        "type": {
          "tensor-type": {
            "elem-type": "FLOAT",
            "shape": [{"dim-value": 256}, {"dim-value": 10}]
          }
        },
        "doc": "Output from vertex 2"
      }
    ],
    "initializers": []
  }
}
```

### Step 3: Convert to ONNX protobuf
```bash
python tools/onnx_bridge.py \
  --input examples/simple-mlp.json \
  --output examples/simple-mlp.onnx
```

### Step 4: Execute with ONNX Runtime
```bash
python tools/onnx_bridge.py \
  --input examples/simple-mlp.json \
  --execute
```

### Step 5 (Optional): Visualize with Netron
```bash
# Install Netron: pip install netron
netron examples/simple-mlp.onnx
```

## Connection to Paper

This example demonstrates **Section 1.1** of Belfiore & Bennequin (2022):
- Oriented graph Γ with vertices L₀, L₁, L₂ (layers)
- Edges e₀: L₀ → L₁, e₁: L₁ → L₂
- Classical architecture (no convergence, simple chain)

The ONNX export shows how oriented graphs map to executable computation:
- DirectedGraph (categorical) ↔ ONNX DAG (computational)
- Vertices ↔ Operations
- Edges ↔ Tensor flow

## Next Steps

1. **Add ReLU between layers**: Currently only Gemm+Softmax
2. **Include weight initializers**: Export actual trained weights
3. **Multi-input example**: ResNet block with skip connections
4. **Fork construction**: Map fork-star vertices to ONNX Add/Concat nodes
5. **Roundtrip**: Import ONNX → DirectedGraph → verify isomorphism
-}
