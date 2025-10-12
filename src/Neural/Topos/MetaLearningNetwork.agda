{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Concrete Meta-Learning Network

This module defines a CONCRETE meta-learning network with:
1. **Structure**: DirectedGraph defining the architecture
2. **Dynamics**: ActivityFunctor and WeightFunctor
3. **ONNX Export**: GraphAnnotations for compilation

Unlike MetaLearning.agda (abstract specification), this is an EXECUTABLE
network that can be exported to ONNX and trained.

## Architecture

```
Task Input (k examples) → Task Encoder → Task Embedding
                                            ↓
Support Examples ────────────────→ ┌─────────────┐
                                   │   Adapted    │
Query Input     ─────────────────→ │   Site       │ → Prediction
                                   │  (Sheaf Net) │
                                   └─────────────┘
                                         ↑
                                   Base Coverage
                                   (Meta-learned)
```

## Components

1. **Task Encoder**: Maps k-shot examples → embedding θ ∈ ℝ⁶⁴
2. **Adaptation Network**: Maps embedding θ → coverage adjustments Δ
3. **Sheaf Network**: Applies adapted coverage to query inputs
-}

module Neural.Topos.MetaLearningNetwork where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Data.Nat.Base using (Nat; zero; suc; _+_; _*_)
open import Data.Fin.Base using (Fin; fzero; fsuc)
open import Data.List.Base
open import Data.String.Base using (String)

open import Neural.Base using (DirectedGraph; vertices; edges; source; target)
open import Neural.Dynamics.Chain using (ChainGraph; ActivityFunctor; WeightFunctor)
-- ONNX export will be done via separate serialization module
-- open import Neural.Compile.ONNX using (GraphAnnotations; TensorShape; OperationType; ModelProto)
-- open import Neural.Compile.ONNX.Export using (export-to-onnx)

--------------------------------------------------------------------------------
-- Real Numbers (postulated)
--------------------------------------------------------------------------------

postulate
  ℝ : Type
  ℝ-is-set : is-set ℝ
  _+ᵣ_ : ℝ → ℝ → ℝ
  _*ᵣ_ : ℝ → ℝ → ℝ
  zeroᵣ : ℝ

-- Vector spaces
Vec : Nat → Type
Vec n = Fin n → ℝ

Matrix : Nat → Nat → Type
Matrix m n = Fin m → Fin n → ℝ

-- Activation functions
postulate
  relu : ℝ → ℝ
  tanh : ℝ → ℝ
  attention : {n : Nat} → Vec n → Vec n → Vec n → Vec n

--------------------------------------------------------------------------------
-- Architecture Dimensions
--------------------------------------------------------------------------------

{-|
Meta-learning network dimensions:

**Task Encoder**:
- Input: k pairs of (input grid, output grid)
- Example encoder: grid → 32-dim vector
- Attention layer: aggregate k examples → 64-dim embedding

**Adaptation Network**:
- Input: 64-dim task embedding
- Hidden: 128-dim
- Output: coverage adjustments (20 × 5 × 20 = 2000-dim)

**Sheaf Network**:
- Base coverage: 20 objects, 5 covers per object
- Input: query state (variable size)
- Output: prediction (variable size)
-}

-- Task encoder dimensions
example-dim : Nat
example-dim = 32

embedding-dim : Nat
embedding-dim = 64

-- Adaptation network dimensions
hidden-dim : Nat
hidden-dim = 128

-- Sheaf network dimensions
num-objects : Nat
num-objects = 20

num-covers : Nat
num-covers = 5

coverage-dim : Nat
coverage-dim = num-objects * num-covers * num-objects  -- 2000

--------------------------------------------------------------------------------
-- Meta-Learning Graph Structure
--------------------------------------------------------------------------------

{-|
The meta-learning network as a directed graph.

Vertices (11 total):
- 0: Task input (k examples)
- 1: Example encoder 1
- 2: Example encoder 2
- ...
- k: Example encoder k
- k+1: Attention aggregation
- k+2: Task embedding
- k+3: Adaptation network hidden
- k+4: Coverage adjustments
- k+5: Base coverage (constant)
- k+6: Adapted coverage (sum)
- k+7: Query input
- k+8: Sheaf network hidden
- k+9: Prediction output
-}

-- For 3-shot learning
k-shot : Nat
k-shot = 3

total-vertices : Nat
total-vertices = 11

meta-learning-graph : DirectedGraph
meta-learning-graph = record
  { vertices = total-vertices
  ; edges = 12
  ; source = src
  ; target = tgt
  }
  where
    src : Fin 12 → Fin total-vertices
    src fzero = fzero                    -- 0: input → encoder1
    src (fsuc fzero) = fzero             -- 1: input → encoder2
    src (fsuc (fsuc fzero)) = fzero      -- 2: input → encoder3
    src (fsuc (fsuc (fsuc fzero))) = (fsuc (fsuc (fsuc fzero)))  -- 3: encoders → attention
    src _ = fzero  -- TODO: fill in remaining edges

    tgt : Fin 12 → Fin total-vertices
    tgt fzero = (fsuc fzero)             -- 0: input → encoder1
    tgt (fsuc fzero) = (fsuc (fsuc fzero))  -- 1: input → encoder2
    tgt (fsuc (fsuc fzero)) = (fsuc (fsuc (fsuc fzero)))  -- 2: input → encoder3
    tgt (fsuc (fsuc (fsuc fzero))) = (fsuc (fsuc (fsuc (fsuc fzero))))  -- 3: encoders → attention
    tgt _ = fzero  -- TODO: fill in remaining edges

--------------------------------------------------------------------------------
-- Weight Spaces
--------------------------------------------------------------------------------

{-|
Weight spaces for each edge in the meta-learning graph.
-}

record MetaLearningWeights : Type where
  field
    -- Task encoder weights
    encoder-weights : (i : Fin k-shot) → Matrix 900 example-dim  -- Assuming 30×30 max grid

    -- Attention weights
    query-weights : Matrix example-dim embedding-dim
    key-weights : Matrix example-dim embedding-dim
    value-weights : Matrix example-dim embedding-dim

    -- Adaptation network weights
    adapt-w1 : Matrix embedding-dim hidden-dim
    adapt-w2 : Matrix hidden-dim coverage-dim

    -- Base coverage (meta-learned parameter)
    base-coverage : Matrix (num-objects * num-covers) num-objects

    -- Sheaf network weights
    sheaf-w1 : Matrix coverage-dim hidden-dim
    sheaf-w2 : Matrix hidden-dim 900  -- Max output size

--------------------------------------------------------------------------------
-- Activity Functor (Dynamics)
--------------------------------------------------------------------------------

{-|
Activities functor X^w for the meta-learning network.

For fixed weights w, this describes:
- State space at each vertex
- Transition function for each edge
-}

module MetaLearningDynamics (w : MetaLearningWeights) where
  open MetaLearningWeights w

  -- State space at each vertex
  vertex-state : Fin total-vertices → Nat
  vertex-state fzero = 900 * k-shot * 2  -- Task input: k pairs of grids
  vertex-state (fsuc fzero) = example-dim  -- Encoder 1 output
  vertex-state (fsuc (fsuc fzero)) = example-dim  -- Encoder 2 output
  vertex-state (fsuc (fsuc (fsuc fzero))) = example-dim  -- Encoder 3 output
  vertex-state (fsuc (fsuc (fsuc (fsuc fzero)))) = embedding-dim  -- Task embedding
  vertex-state (fsuc (fsuc (fsuc (fsuc (fsuc fzero))))) = hidden-dim  -- Adaptation hidden
  vertex-state (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc fzero)))))) = coverage-dim  -- Adjusted coverage
  vertex-state (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc fzero))))))) = coverage-dim  -- Base coverage
  vertex-state (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc fzero)))))))) = coverage-dim  -- Adapted coverage
  vertex-state (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc fzero))))))))) = 900  -- Query input
  vertex-state (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc (fsuc fzero)))))))))) = 900  -- Prediction

  -- Transition functions (forward propagation)
  postulate
    matrix-mult : {m n : Nat} → Matrix m n → Vec m → Vec n
    vector-add : {n : Nat} → Vec n → Vec n → Vec n
    apply-relu : {n : Nat} → Vec n → Vec n
    apply-attention : {n : Nat} → (k : Nat) → Vec (k * n) → Vec n

  -- Forward pass through network
  encode-example : Vec 1800 → Vec example-dim  -- Encode one (input, output) pair
  encode-example x = apply-relu (matrix-mult (encoder-weights fzero) x)

  aggregate-examples : Vec (k-shot * example-dim) → Vec embedding-dim
  aggregate-examples encodings = apply-attention k-shot encodings

  adapt-coverage : Vec embedding-dim → Vec coverage-dim
  adapt-coverage θ =
    let h = apply-relu (matrix-mult adapt-w1 θ)
    in apply-relu (matrix-mult adapt-w2 h)

  combine-coverage : Vec coverage-dim → Vec coverage-dim → Vec coverage-dim
  combine-coverage base-cov adjustments = vector-add base-cov adjustments

  sheaf-predict : Vec coverage-dim → Vec 900 → Vec 900
  sheaf-predict adapted-cov query =
    let h = apply-relu (matrix-mult sheaf-w1 adapted-cov)
    in matrix-mult sheaf-w2 h

--------------------------------------------------------------------------------
-- ONNX Export Annotations
--------------------------------------------------------------------------------

{-|
Annotations to export the meta-learning network to ONNX.
-}

postulate
  show-nat : Nat → String
  _<>ₛ_ : String → String → String

-- ONNX export: Will be implemented via JSON serialization
-- See examples/SimpleMLP/Export.agda for pattern

{-|
## Usage

To compile this network:

1. **In Agda**: This module defines the structure and dynamics
   ```agda
   open MetaLearningDynamics initial-weights
   ```

2. **Extract to JSON**: Use Agda reflection to serialize
   ```bash
   agda --compile Neural.Topos.MetaLearningNetwork
   ```

3. **Compile to JAX**: Use neural_compiler
   ```python
   from neural_compiler import compile_architecture
   model = compile_architecture("meta_learning_network.json")
   ```

4. **Train with meta-learning**: Use meta_learner.py training loop
   ```python
   meta_learner = MetaToposLearner.load_from_agda(model)
   meta_learner.meta_train(training_tasks, ...)
   ```

5. **Export to ONNX**: Use onnx_export.py
   ```python
   export_and_test_meta_learner(meta_learner, "trained_models/")
   ```
-}
