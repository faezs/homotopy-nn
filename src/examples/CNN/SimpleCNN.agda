{-# OPTIONS --no-import-sorts #-}

{-|
# Simple CNN with Translation Group Symmetry

A concrete LeNet-style convolutional neural network demonstrating:
1. Oriented graph structure (Section 1.1)
2. Weight functors as dynamical objects (Section 1.2)
3. ℤ² translation group action (Section 2.1)
4. ONNX export using the categorical framework

## Architecture

```
Input (28×28×1)
   ↓
Conv1 (5×5 kernel, 20 filters) + ReLU
   ↓
MaxPool (2×2)
   ↓
Conv2 (5×5 kernel, 50 filters) + ReLU
   ↓
MaxPool (2×2)
   ↓
Flatten
   ↓
Dense (500 neurons) + ReLU
   ↓
Dense (10 neurons) + Softmax
```

Graph structure: 11 vertices, 10 edges (simple chain - no convergence)

## Translation Equivariance

Conv layers are ℤ²-equivariant:
  Conv(T_a(x)) = T_a(Conv(x))  for all a ∈ ℤ²

This is realized through weight sharing - the same kernel is applied at all
spatial positions, respecting the translation group action.
-}

module examples.CNN.SimpleCNN where

open import 1Lab.Prelude
open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin; fzero; fsuc; Fin-cases; weaken)
open import Data.List.Base
open import Data.String.Base using (String)
open import Data.Float.Base using (Float)

open import Cat.Base using (Functor)
open Functor

open import Neural.Base using (DirectedGraph; vertices; edges; source; target)
open import Neural.Compile.ONNX
open import Neural.Compile.ONNX.Export
open import Neural.Compile.ONNX.Serialize
open import Neural.Compile.EinsumDSL

open import examples.CNN.TranslationGroup using (ℤ²)

--------------------------------------------------------------------------------
-- Network Specification Using Shape-Based DSL
--------------------------------------------------------------------------------

{-|
**NEW APPROACH**: Specify the network using only tensor shapes.
All operation parameters (kernel sizes, strides, etc.) are INFERRED from geometry!

The philosophy: **The shape IS the spec**.
-}

simple-cnn-spec : SequentialSpec
simple-cnn-spec = sequential "simple-cnn"
  ( space (1 ∷ 28 ∷ 28 ∷ 1 ∷ [])              -- Input: batch=1, 28×28, 1 channel
  ∷ transform (1 ∷ 24 ∷ 24 ∷ 20 ∷ []) "Conv"  -- Conv: 28→24 implies kernel=5
  ∷ transform (1 ∷ 24 ∷ 24 ∷ 20 ∷ []) "Relu"  -- Elementwise (shape unchanged)
  ∷ transform (1 ∷ 12 ∷ 12 ∷ 20 ∷ []) "MaxPool"  -- Pool: 24→12 implies stride=2
  ∷ transform (1 ∷ 8 ∷ 8 ∷ 50 ∷ []) "Conv"    -- Conv: 12→8 implies kernel=5
  ∷ transform (1 ∷ 8 ∷ 8 ∷ 50 ∷ []) "Relu"
  ∷ transform (1 ∷ 4 ∷ 4 ∷ 50 ∷ []) "MaxPool"    -- Pool: 8→4 implies stride=2
  ∷ transform (1 ∷ 800 ∷ []) "Flatten"        -- Reshape 4×4×50 = 800
  ∷ transform (1 ∷ 500 ∷ []) "Gemm"           -- Dense 800→500
  ∷ transform (1 ∷ 500 ∷ []) "Relu"
  ∷ transform (1 ∷ 10 ∷ []) "Gemm"            -- Dense 500→10 (output)
  ∷ []
  )

{-|
Generate the DirectedGraph automatically from the shape specification.
No manual functor construction needed!
-}
simple-cnn-graph-dsl : DirectedGraph
simple-cnn-graph-dsl = compile-to-graph simple-cnn-spec

{-|
Verify convergence count (should still be 0 for sequential network).
-}
simple-cnn-convergence-dsl : count-multi-input-nodes simple-cnn-graph-dsl ≡ 0
simple-cnn-convergence-dsl = sequential-convergence-count simple-cnn-spec

--------------------------------------------------------------------------------
-- Weight Functors (Dynamical Objects)
--------------------------------------------------------------------------------

{-|
Real numbers for weight values.
-}
postulate
  ℝ : Type
  ℝ-is-set : is-set ℝ

{-|
Weight tensor for convolutional layer.
For a Conv layer with:
- c_in input channels
- c_out output channels
- k_h × k_w kernel size

Weight shape: [c_out, c_in, k_h, k_w]
-}
record ConvWeights (c-in c-out k-h k-w : Nat) : Type where
  field
    kernel : Fin c-out → Fin c-in → Fin k-h → Fin k-w → ℝ
    bias   : Fin c-out → ℝ

{-|
Weight tensor for dense (fully-connected) layer.
Shape: [n-out, n-in]
-}
record DenseWeights (n-in n-out : Nat) : Type where
  field
    weight : Fin n-out → Fin n-in → ℝ
    bias   : Fin n-out → ℝ

--------------------------------------------------------------------------------
-- Concrete Weight Values (Initialized)
--------------------------------------------------------------------------------

{-|
Conv1 weights: 1 input channel → 20 output channels, 5×5 kernel
-}
postulate
  conv1-weights : ConvWeights 1 20 5 5

{-|
Conv2 weights: 20 input channels → 50 output channels, 5×5 kernel
-}
postulate
  conv2-weights : ConvWeights 20 50 5 5

{-|
Dense1 weights: 800 inputs (50 × 4 × 4) → 500 outputs
-}
postulate
  dense1-weights : DenseWeights 800 500

{-|
Dense2 weights: 500 inputs → 10 outputs (digits)
-}
postulate
  dense2-weights : DenseWeights 500 10

--------------------------------------------------------------------------------
-- Oriented Graph Structure
--------------------------------------------------------------------------------

{-|
The CNN as an oriented graph with 11 vertices:
- Vertex 0: Input (28×28×1)
- Vertex 1: Conv1 (24×24×20, valid padding)
- Vertex 2: ReLU1
- Vertex 3: MaxPool1 (12×12×20)
- Vertex 4: Conv2 (8×8×50, valid padding)
- Vertex 5: ReLU2
- Vertex 6: MaxPool2 (4×4×50)
- Vertex 7: Flatten (800)
- Vertex 8: Dense1 (500)
- Vertex 9: ReLU3
- Vertex 10: Dense2 + Softmax (10)

Edges: 0→1, 1→2, 2→3, 3→4, 4→5, 5→6, 6→7, 7→8, 8→9, 9→10

This is a CONCRETE functor ·⇉· → FinSets:
- F₀(false) = 10 (edges)
- F₀(true) = 11 (vertices)
- F₁(source) = weaken : Fin 10 → Fin 11  (edge i has source vertex i)
- F₁(target) = fsuc : Fin 10 → Fin 11    (edge i has target vertex i+1)
-}
simple-cnn-graph : DirectedGraph
simple-cnn-graph .F₀ false = 10  -- 10 edges
simple-cnn-graph .F₀ true = 11   -- 11 vertices
simple-cnn-graph .F₁ {false} {true} true = weaken   -- source: edge i → vertex i
simple-cnn-graph .F₁ {false} {true} false = fsuc    -- target: edge i → vertex i+1
simple-cnn-graph .F₁ {false} {false} tt = λ e → e   -- id on edges
simple-cnn-graph .F₁ {true} {true} tt = λ v → v     -- id on vertices
simple-cnn-graph .F-id {false} = refl
simple-cnn-graph .F-id {true} = refl
simple-cnn-graph .F-∘ {false} {false} {false} tt tt = refl
simple-cnn-graph .F-∘ {false} {false} {true} true tt = refl   -- true ∘ id_false = true
simple-cnn-graph .F-∘ {false} {false} {true} false tt = refl  -- false ∘ id_false = false
simple-cnn-graph .F-∘ {false} {true} {true} tt true = refl    -- id_true ∘ true = true
simple-cnn-graph .F-∘ {false} {true} {true} tt false = refl   -- id_true ∘ false = false
simple-cnn-graph .F-∘ {true} {true} {true} tt tt = refl

--------------------------------------------------------------------------------
-- ONNX Annotations
--------------------------------------------------------------------------------

{-|
Annotate the CNN graph with ONNX operation types, tensor shapes, and attributes.
-}
simple-cnn-annotations : GraphAnnotations simple-cnn-graph
simple-cnn-annotations = record
  { vertex-op-type = λ v →
      -- Map vertex index to operation type
      Fin-cases {P = λ _ → String} "Input"
        (Fin-cases "Conv"
          (Fin-cases "Relu"
            (Fin-cases "MaxPool"
              (Fin-cases "Conv"
                (Fin-cases "Relu"
                  (Fin-cases "MaxPool"
                    (Fin-cases "Flatten"
                      (Fin-cases "Gemm"
                        (Fin-cases "Relu"
                          (Fin-cases "Softmax" (λ _ → "Unknown"))
                        ))
                    ))
                ))
            ))
        ) v  -- No subst needed - vertices simple-cnn-graph is definitionally 11

  ; edge-shape = λ e →
      -- Tensor shapes flowing through edges
      Fin-cases {P = λ _ → List Nat} (1 ∷ 1 ∷ 28 ∷ 28 ∷ [])      -- 0: Input → Conv1 [batch, 1, 28, 28]
        (Fin-cases (1 ∷ 20 ∷ 24 ∷ 24 ∷ [])   -- 1: Conv1 → ReLU1 [batch, 20, 24, 24]
          (Fin-cases (1 ∷ 20 ∷ 24 ∷ 24 ∷ []) -- 2: ReLU1 → MaxPool1
            (Fin-cases (1 ∷ 20 ∷ 12 ∷ 12 ∷ []) -- 3: MaxPool1 → Conv2
              (Fin-cases (1 ∷ 50 ∷ 8 ∷ 8 ∷ []) -- 4: Conv2 → ReLU2
                (Fin-cases (1 ∷ 50 ∷ 8 ∷ 8 ∷ []) -- 5: ReLU2 → MaxPool2
                  (Fin-cases (1 ∷ 50 ∷ 4 ∷ 4 ∷ []) -- 6: MaxPool2 → Flatten
                    (Fin-cases (1 ∷ 800 ∷ [])        -- 7: Flatten → Dense1
                      (Fin-cases (1 ∷ 500 ∷ [])      -- 8: Dense1 → ReLU3
                        (Fin-cases (1 ∷ 10 ∷ [])     -- 9: ReLU3 → Softmax
                          (λ _ → [])
                        ))
                    ))
                ))
            ))
        ) e  -- No subst needed - edges simple-cnn-graph is definitionally 10

  ; edge-elem-type = λ _ → FLOAT  -- All tensors are float32

  ; vertex-attributes = λ v →
      Fin-cases {P = λ _ → List AttributeProto} []  -- Vertex 0: Input - no attributes
        (Fin-cases  -- Vertex 1: Conv1 - kernel_shape, strides, pads
          (record { name = "kernel_shape" ; value = attr-ints (5 ∷ 5 ∷ []) } ∷
           record { name = "strides" ; value = attr-ints (1 ∷ 1 ∷ []) } ∷
           record { name = "pads" ; value = attr-ints (0 ∷ 0 ∷ 0 ∷ 0 ∷ []) } ∷ [])
          (Fin-cases []  -- Vertex 2: ReLU1 - no attributes
            (Fin-cases  -- Vertex 3: MaxPool1 - kernel_shape, strides
              (record { name = "kernel_shape" ; value = attr-ints (2 ∷ 2 ∷ []) } ∷
               record { name = "strides" ; value = attr-ints (2 ∷ 2 ∷ []) } ∷ [])
              (Fin-cases  -- Vertex 4: Conv2 - kernel_shape, strides, pads
                (record { name = "kernel_shape" ; value = attr-ints (5 ∷ 5 ∷ []) } ∷
                 record { name = "strides" ; value = attr-ints (1 ∷ 1 ∷ []) } ∷
                 record { name = "pads" ; value = attr-ints (0 ∷ 0 ∷ 0 ∷ 0 ∷ []) } ∷ [])
                (Fin-cases []  -- Vertex 5: ReLU2 - no attributes
                  (Fin-cases  -- Vertex 6: MaxPool2 - kernel_shape, strides
                    (record { name = "kernel_shape" ; value = attr-ints (2 ∷ 2 ∷ []) } ∷
                     record { name = "strides" ; value = attr-ints (2 ∷ 2 ∷ []) } ∷ [])
                    (Fin-cases  -- Vertex 7: Flatten - axis
                      (record { name = "axis" ; value = attr-int 1 } ∷ [])
                      (Fin-cases  -- Vertex 8: Gemm (Dense1) - alpha, beta
                        (record { name = "alpha" ; value = attr-float 1.0 } ∷
                         record { name = "beta" ; value = attr-float 1.0 } ∷ [])
                        (Fin-cases []  -- Vertex 9: ReLU3 - no attributes
                          (Fin-cases  -- Vertex 10: Softmax - axis
                            (record { name = "axis" ; value = attr-int 1 } ∷ [])
                            (λ _ → [])
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        ) v  -- No subst needed - vertices simple-cnn-graph is definitionally 11

  ; graph-inputs = fzero ∷ []  -- Vertex 0 is input

  ; graph-outputs = (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc fzero)))))))))) ∷ []  -- Vertex 10 is output

  ; model-name = "simple-cnn"

  ; model-doc = "LeNet-style CNN with ℤ² translation equivariance: 28×28×1 → Conv(5×5,20) → MaxPool → Conv(5×5,50) → MaxPool → Dense(500) → Dense(10)"

  ; producer = "homotopy-nn"
  }

--------------------------------------------------------------------------------
-- ONNX Export
--------------------------------------------------------------------------------

{-|
Export the CNN to ONNX ModelProto.
This produces an in-memory ONNX model ready for serialization.
-}
simple-cnn-onnx : ModelProto
simple-cnn-onnx = export-to-onnx simple-cnn-graph simple-cnn-annotations

{-|
Serialize the CNN to JSON format.
This is the key function that generates examples/simple-cnn.json!
-}
simple-cnn-json : String
simple-cnn-json = model-to-json simple-cnn-onnx

{-|
Verify the export is valid.
-}
simple-cnn-is-valid : is-valid-for-export simple-cnn-graph simple-cnn-annotations ≡ true
simple-cnn-is-valid = refl

{-|
Count convergence points (should be 0 - simple chain).

Since `simple-cnn-graph` is now concretely defined, Agda computes this automatically!
The chain structure means each vertex has at most one incoming edge.
-}
simple-cnn-convergence-count : count-multi-input-nodes simple-cnn-graph ≡ 0
simple-cnn-convergence-count = refl  -- Computed automatically via normalization!

--------------------------------------------------------------------------------
-- Translation Equivariance Properties
--------------------------------------------------------------------------------

{-|
## Connection to ℤ² Translation Group (Section 2.1)

The Conv and MaxPool operations in this network are **ℤ²-equivariant**:

**Theorem**: For any translation a ∈ ℤ² and input feature map F:
  Conv_W(T_a(F)) = T_a(Conv_W(F))

where T_a is the translation operator from examples.CNN.FeatureMaps.

**Proof Sketch**:
1. Convolution applies the same kernel W at all spatial positions
2. Translation shifts all positions uniformly by a
3. Since W is position-independent (weight sharing), the convolution
   commutes with translation

This is formalized by:
- Conv weights W: independent of position (kernel shared across spatial grid)
- Translation action: shifts spatial coordinates by a ∈ ℤ²
- Equivariance: output at position x+a depends only on input around x+a

**ONNX Representation**:
- kernel_shape=[5,5]: Same kernel applied everywhere (weight sharing)
- strides=[1,1]: Unit stride preserves grid structure
- pads=[0,0,0,0]: Valid padding (no spatial symmetry breaking at boundaries)

**MaxPool Equivariance**:
Similarly, MaxPool is ℤ²-equivariant:
  MaxPool(T_a(F)) = T_a(MaxPool(F))

with stride=2, downsampling the grid uniformly.

**Breaking Equivariance**:
- Flatten (vertex 7): Destroys spatial structure, breaks ℤ² action
- Dense layers (vertices 8, 10): No spatial structure, no ℤ² action

This CNN has two regimes:
1. **Convolutional (0-6)**: ℤ²-equivariant feature extraction
2. **Classification (7-10)**: Non-equivariant decision making
-}

-- Formal proofs would require:
-- 1. FeatureMap type (Grid → ℝ^c)
-- 2. Translation action T_a : FeatureMap → FeatureMap
-- 3. Convolution operator Conv_W : FeatureMap → FeatureMap
-- 4. Equivariance: ∀ a F. Conv_W(T_a(F)) ≡ T_a(Conv_W(F))
-- See examples.CNN.FeatureMaps for the full formalization

--------------------------------------------------------------------------------
-- Documentation
--------------------------------------------------------------------------------

{-|
## Usage Instructions

### Step 1: Type-check this file
```bash
agda --library-file=./libraries src/examples/CNN/SimpleCNN.agda
```

### Step 2: Serialize to JSON
Create `examples/simple-cnn.json` manually or via reflection:
- Use the structure from `simple-cnn-onnx : ModelProto`
- Follow the same format as examples/simple-mlp.json
- Include Conv, MaxPool, Gemm, Relu, Softmax, Flatten operations

### Step 3: Convert to ONNX protobuf
```bash
python tools/onnx_bridge.py \
  --input examples/simple-cnn.json \
  --output examples/simple-cnn.onnx
```

### Step 4: Visualize with Netron
```bash
netron examples/simple-cnn.onnx
```

## Connection to Paper

**Section 1.1 - Oriented Graphs**:
- simple-cnn-graph is an oriented graph Γ
- 11 vertices = layers/operations
- 10 edges = tensor flow connections
- Acyclic (classical architecture, no recurrence)

**Section 1.2 - Dynamical Objects**:
- conv1-weights, conv2-weights, dense1-weights, dense2-weights
- These are the "weight functors" - dynamical objects on the graph
- Each edge e has associated weight tensor W_e

**Section 2.1 - Group Actions and Stacks**:
- Conv layers form ℤ²-sets (feature maps with translation action)
- Weight sharing realizes fibred action (Equation 2.1)
- Stack structure: contravariant functor F: CNN^op → ℤ²-Sets

**Section 1.3 - Fork Construction** (Future work):
- This CNN has no convergence (no multi-input nodes)
- ResNet variant would have fork-star vertices at skip connection adds
- ONNX Add nodes would represent fork-star construction
-}
