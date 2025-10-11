{-# OPTIONS --no-import-sorts #-}

{-|
# Export Simple MLP to ONNX

Connects the concrete oriented graph and dynamical objects to ONNX export.

Workflow:
1. Oriented graph G: ·⇉· → FinSets (SimpleMLP.Graph)
2. Dynamical objects (X^w, W, X) (SimpleMLP.Dynamics)
3. ONNX annotations (this module)
4. Export to ONNX ModelProto
5. Serialize and execute

This demonstrates the complete mapping from Section 1.1-1.2 to executable ONNX.
-}

module examples.SimpleMLP.Export where

open import 1Lab.Prelude
open import Data.Nat.Base using (Nat)
open import Data.Fin.Base using (Fin; fzero; fsuc; Fin-cases)
open import Data.List.Base
open import Data.String.Base using (String)

open import Neural.Compile.ONNX
open import Neural.Compile.ONNX.Export

open import examples.SimpleMLP.Graph
open import examples.SimpleMLP.Dynamics

--------------------------------------------------------------------------------
-- ONNX Annotations for Simple MLP
--------------------------------------------------------------------------------

{-|
Map the categorical structure to ONNX operations.

Vertices → ONNX operations:
- Vertex 0 (input): Input placeholder
- Vertex 1 (hidden): Gemm + ReLU
- Vertex 2 (output): Gemm + Softmax

Edges → ONNX tensors:
- Edge 0: input→hidden weights (784×256)
- Edge 1: hidden→output weights (256×10)
-}

mlp-annotations : GraphAnnotations simple-mlp-graph
mlp-annotations = record
  { vertex-op-type = λ v →
      Fin-cases "Input"
        (Fin-cases "Gemm"
          (Fin-cases "Softmax" (λ _ → "Unknown"))
        ) v

  ; edge-shape = λ e →
      -- Edge shapes correspond to weight matrix dimensions
      Fin-cases (input-dim ∷ hidden-dim ∷ [])
        (Fin-cases (hidden-dim ∷ output-dim ∷ []) (λ _ → []))
        e

  ; edge-elem-type = λ _ → FLOAT

  ; vertex-attributes = λ v →
      Fin-cases []  -- Input: no attributes
        (Fin-cases
          -- Hidden layer (Gemm): standard matrix multiplication
          ( record { name = "alpha" ; value = attr-float 1.0 } ∷
            record { name = "beta" ; value = attr-float 1.0 } ∷
            record { name = "transB" ; value = attr-int 0 } ∷ [] )
          (Fin-cases
            -- Output layer (Softmax): axis=1
            (record { name = "axis" ; value = attr-int 1 } ∷ [])
            (λ _ → [])
          )
        ) v

  ; graph-inputs = fzero ∷ []  -- Vertex 0

  ; graph-outputs = fsuc (fsuc fzero) ∷ []  -- Vertex 2

  ; model-name = "simple-mlp-concrete"

  ; model-doc =
      "Simple MLP with concrete oriented graph and dynamical objects. " <>
      "Architecture: Input(784) → Hidden(256,ReLU) → Output(10,Softmax). " <>
      "Implements Section 1.1-1.2 of Belfiore & Bennequin (2022)."

  ; producer = "homotopy-nn-agda"
  }
  where
    open import Data.String.Base using (_<>_)

--------------------------------------------------------------------------------
-- Export to ONNX
--------------------------------------------------------------------------------

{-|
Export the concrete graph to ONNX ModelProto.
-}

mlp-onnx-model : ModelProto
mlp-onnx-model = export-to-onnx simple-mlp-graph mlp-annotations

{-|
Verify the export is valid.
-}

mlp-export-valid : is-valid-for-export simple-mlp-graph mlp-annotations ≡ true
mlp-export-valid = refl

{-|
Count convergence points (should be 0 for this chain architecture).
-}

mlp-convergence-points : count-multi-input-nodes simple-mlp-graph ≡ 0
mlp-convergence-points = {!!}  -- TODO: Prove

--------------------------------------------------------------------------------
-- Connection to Dynamics
--------------------------------------------------------------------------------

{-|
The ONNX model encodes the dynamics defined in SimpleMLP.Dynamics:

Vertex transformations:
- Input (v0): X^w(v0) = ℝ^784
- Hidden (v1): X^w(v1) = ℝ^256
- Output (v2): X^w(v2) = ℝ^10

Edge transformations:
- Edge e0: X^w(e0) : ℝ^784 → ℝ^256
  ONNX: Gemm(input, W₀, b₀) where W₀ ∈ ℝ^(784×256)
  Agda: matrix-vector-mult (weights e0) : Vec 784 → Vec 256

- Edge e1: X^w(e1) : ℝ^256 → ℝ^10
  ONNX: Gemm(hidden, W₁, b₁) where W₁ ∈ ℝ^(256×10)
  Agda: matrix-vector-mult (weights e1) : Vec 256 → Vec 10

Activations:
- After e0: ReLU : ℝ^256 → ℝ^256
  Agda: activation-fn (fsuc fzero) = map relu

- After e1: Softmax : ℝ^10 → ℝ^10
  Agda: activation-fn (fsuc (fsuc fzero)) = softmax
-}

-- Dimensions match between Agda and ONNX
dimensions-match :
  ( vertex-state-space fzero ≡ input-dim ) ×
  ( vertex-state-space (fsuc fzero) ≡ hidden-dim ) ×
  ( vertex-state-space (fsuc (fsuc fzero)) ≡ output-dim )
dimensions-match = refl , refl , refl

-- Edge weight dimensions match
edge-dims-match :
  ( edge-weight-space fzero ≡ (input-dim , hidden-dim) ) ×
  ( edge-weight-space (fsuc fzero) ≡ (hidden-dim , output-dim) )
edge-dims-match = refl , refl

--------------------------------------------------------------------------------
-- Serialization Instructions
--------------------------------------------------------------------------------

{-|
## To serialize and execute:

### Step 1: Generate JSON (manual for now)

The `mlp-onnx-model` value contains the complete ONNX structure.
Export to JSON following the schema in examples/simple-mlp.json

### Step 2: Convert to ONNX protobuf

```bash
nix develop .# --command python3 tools/onnx_bridge.py \
  --input examples/simple-mlp-concrete.json \
  --output examples/simple-mlp-concrete.onnx
```

### Step 3: Execute

```bash
nix develop .# --command python3 tools/onnx_bridge.py \
  --input examples/simple-mlp-concrete.json \
  --execute
```

### Step 4: Load weights from trained model

To use actual trained weights instead of zeros:

1. Train model in PyTorch/TensorFlow
2. Export weights to JSON/protobuf
3. Load into Agda's `example-weights` in SimpleMLP.Dynamics
4. Re-export to ONNX

## Mathematical Correspondence

**Oriented Graph (Section 1.1)**:
- G: ·⇉· → FinSets
- vertices(G) = 3
- edges(G) = 2
- source, target functions defined concretely

**Dynamical Objects (Section 1.2)**:
- X^w: G → FinSets (activities)
- W: G → FinSets (weights)
- X: Total dynamics

**ONNX Mapping**:
- G.vertices → ONNX nodes
- G.edges → ONNX tensor names
- X^w transitions → ONNX operations (Gemm, ReLU, Softmax)
- W weight spaces → ONNX initializers (TensorProto)

**Backpropagation (Section 1.4)**:
- Natural transformations W → W
- Gradients flow backwards through graph
- ONNX: training mode (not implemented yet)
-}
