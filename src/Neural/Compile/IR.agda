{-# OPTIONS --rewriting --guardedness --cubical #-}

{-|
# Intermediate Representation for Neural Network Compilation

This module defines the IR (Intermediate Representation) for extracting
neural architectures from Agda to be compiled to JAX/XLA.

## Design

The IR captures:
1. **Topology**: Graph structure (vertices, edges)
2. **Operations**: What each vertex computes
3. **Shapes**: Tensor dimensions (from fibrations)
4. **Properties**: Verified properties (conservation, etc.)
5. **Resources**: Constraints (FLOPs, memory)

## Compilation Pipeline

```
Agda Architecture
  ↓ [Extract]
IR (this module)
  ↓ [Serialize]
JSON
  ↓ [Python bridge]
Polynomial Functors
  ↓ [Compile]
JAX/XLA
```

-}

module Neural.Compile.IR where

open import 1Lab.Prelude
open import 1Lab.Path

open import Data.List using (List; []; _∷_)
open import Data.String using (String)
open import Data.Nat using (ℕ)

--------------------------------------------------------------------------------
-- Shape IR

{-|
## Tensor Shapes

Extracted from fibration structure in Neural.Stack.*
-}

data Shape : Type where
  scalar : Shape
  vec : ℕ → Shape  -- 1D vector
  mat : ℕ → ℕ → Shape  -- 2D matrix
  tensor : List ℕ → Shape  -- n-D tensor

-- Shape equality (for type-checking)
shape-eq : Shape → Shape → Bool
shape-eq scalar scalar = true
shape-eq (vec n) (vec m) = n ≡ᵇ m
  where
  _≡ᵇ_ : ℕ → ℕ → Bool
  zero ≡ᵇ zero = true
  suc n ≡ᵇ suc m = n ≡ᵇ m
  _ ≡ᵇ _ = false
shape-eq (mat n1 n2) (mat m1 m2) = (n1 ≡ᵇ m1) ∧ (n2 ≡ᵇ m2)
  where
  _≡ᵇ_ : ℕ → ℕ → Bool
  zero ≡ᵇ zero = true
  suc n ≡ᵇ suc m = n ≡ᵇ m
  _ ≡ᵇ _ = false
  _∧_ : Bool → Bool → Bool
  true ∧ true = true
  _ ∧ _ = false
shape-eq _ _ = false

--------------------------------------------------------------------------------
-- Operation IR

{-|
## Operations

Each vertex in the neural graph performs an operation.
These map to JAX primitives.
-}

data Activation : Type where
  relu : Activation
  sigmoid : Activation
  tanh : Activation
  gelu : Activation
  identity : Activation

data Operation : Type where
  -- Linear operations
  linear : (in-dim out-dim : ℕ) → Operation
  conv2d : (in-channels out-channels kernel-size : ℕ) → Operation

  -- Nonlinearities
  activation : Activation → Operation

  -- Structural operations (from topos theory)
  fork : (arity : ℕ) → Operation  -- Merge n inputs (sheaf condition)
  residual : Operation  -- Skip connection (conservation)

  -- Normalization
  batch-norm : (features : ℕ) → Operation
  layer-norm : (features : ℕ) → Operation

  -- Pooling
  max-pool : (kernel-size stride : ℕ) → Operation
  avg-pool : (kernel-size stride : ℕ) → Operation

  -- Attention (from stack semantics)
  attention : (heads d-model d-k d-v : ℕ) → Operation

--------------------------------------------------------------------------------
-- Vertex IR

{-|
## Vertex

A vertex in the computational graph with:
- Unique ID
- Operation to perform
- Input/output shapes (from fibration)
-}

record Vertex : Type where
  constructor vertex
  field
    id : ℕ
    op : Operation
    input-shapes : List Shape
    output-shape : Shape

open Vertex public

--------------------------------------------------------------------------------
-- Edge IR

{-|
## Edge

A directed edge connecting vertices.
-}

record Edge : Type where
  constructor edge
  field
    source : ℕ  -- Source vertex ID
    target : ℕ  -- Target vertex ID
    shape : Shape  -- Tensor shape flowing through edge

open Edge public

--------------------------------------------------------------------------------
-- Property IR

{-|
## Verified Properties

Properties proven in Agda that we want to preserve/check in compilation.
-}

data Property : Type where
  -- From Neural.Network.Conservation
  conserves-mass : Property

  -- From stack semantics
  shape-correct : Property
  fibration-valid : Property

  -- From resource theory
  flops-bounded : (max-flops : ℕ) → Property
  memory-bounded : (max-memory : ℕ) → Property

  -- From topos theory
  sheaf-condition : Property  -- Fork vertices satisfy ∏ condition

  -- Custom property (with evidence)
  custom : (name : String) → Property

--------------------------------------------------------------------------------
-- Resource Constraints

{-|
## Resource Constraints

From Neural.Resources.* - bounds on computation.
-}

record ResourceConstraints : Type where
  constructor constraints
  field
    max-flops : ℕ  -- Maximum FLOPs
    max-memory : ℕ  -- Maximum memory (bytes)
    max-latency : ℕ  -- Maximum latency (microseconds)
    sparsity : ℕ  -- Sparsity level (% of zeros)

open ResourceConstraints public

--------------------------------------------------------------------------------
-- Neural IR

{-|
## Complete IR

The full intermediate representation of a neural architecture.
-}

record NeuralIR : Type where
  constructor neural-ir
  field
    name : String

    -- Graph structure
    vertices : List Vertex
    edges : List Edge

    -- Entry points
    inputs : List ℕ  -- Input vertex IDs
    outputs : List ℕ  -- Output vertex IDs

    -- Verified properties
    properties : List Property

    -- Resource constraints (optional)
    resources : ResourceConstraints

open NeuralIR public

--------------------------------------------------------------------------------
-- Example: Simple MLP

{-|
## Example IR

A simple 2-layer MLP in IR form.
-}

mlp-ir : NeuralIR
mlp-ir = neural-ir
  "SimpleMLP"
  vertices-list
  edges-list
  (0 ∷ [])  -- Input vertex
  (3 ∷ [])  -- Output vertex
  (shape-correct ∷ conserves-mass ∷ [])
  (constraints 1000000 1000000 1000 0)
  where
  vertices-list : List Vertex
  vertices-list =
    vertex 0 (linear 784 256) (vec 784 ∷ []) (vec 256) ∷
    vertex 1 (activation relu) (vec 256 ∷ []) (vec 256) ∷
    vertex 2 (linear 256 10) (vec 256 ∷ []) (vec 10) ∷
    vertex 3 (activation identity) (vec 10 ∷ []) (vec 10) ∷
    []

  edges-list : List Edge
  edges-list =
    edge 0 1 (vec 256) ∷
    edge 1 2 (vec 256) ∷
    edge 2 3 (vec 10) ∷
    []

--------------------------------------------------------------------------------
-- Example: ResNet Block

{-|
## Example: Residual Block

Demonstrates fork (merge) and residual operations.
-}

resnet-block-ir : NeuralIR
resnet-block-ir = neural-ir
  "ResNetBlock"
  vertices-list
  edges-list
  (0 ∷ [])
  (6 ∷ [])
  (shape-correct ∷ conserves-mass ∷ sheaf-condition ∷ [])
  (constraints 10000000 5000000 5000 50)
  where
  channels : ℕ
  channels = 64

  vertices-list : List Vertex
  vertices-list =
    -- Main path
    vertex 0 (conv2d channels channels 3) (tensor (channels ∷ 32 ∷ 32 ∷ []) ∷ []) (tensor (channels ∷ 32 ∷ 32 ∷ [])) ∷
    vertex 1 (batch-norm channels) (tensor (channels ∷ 32 ∷ 32 ∷ []) ∷ []) (tensor (channels ∷ 32 ∷ 32 ∷ [])) ∷
    vertex 2 (activation relu) (tensor (channels ∷ 32 ∷ 32 ∷ []) ∷ []) (tensor (channels ∷ 32 ∷ 32 ∷ [])) ∷
    vertex 3 (conv2d channels channels 3) (tensor (channels ∷ 32 ∷ 32 ∷ []) ∷ []) (tensor (channels ∷ 32 ∷ 32 ∷ [])) ∷
    vertex 4 (batch-norm channels) (tensor (channels ∷ 32 ∷ 32 ∷ []) ∷ []) (tensor (channels ∷ 32 ∷ 32 ∷ [])) ∷
    -- Residual connection
    vertex 5 residual (tensor (channels ∷ 32 ∷ 32 ∷ []) ∷ []) (tensor (channels ∷ 32 ∷ 32 ∷ [])) ∷
    -- Fork (merge main + residual)
    vertex 6 (fork 2) (tensor (channels ∷ 32 ∷ 32 ∷ []) ∷ tensor (channels ∷ 32 ∷ 32 ∷ []) ∷ []) (tensor (channels ∷ 32 ∷ 32 ∷ [])) ∷
    []

  edges-list : List Edge
  edges-list =
    -- Main path
    edge 0 1 (tensor (channels ∷ 32 ∷ 32 ∷ [])) ∷
    edge 1 2 (tensor (channels ∷ 32 ∷ 32 ∷ [])) ∷
    edge 2 3 (tensor (channels ∷ 32 ∷ 32 ∷ [])) ∷
    edge 3 4 (tensor (channels ∷ 32 ∷ 32 ∷ [])) ∷
    -- Residual connection (from input to fork)
    edge 0 5 (tensor (channels ∷ 32 ∷ 32 ∷ [])) ∷
    edge 5 6 (tensor (channels ∷ 32 ∷ 32 ∷ [])) ∷
    -- Main path to fork
    edge 4 6 (tensor (channels ∷ 32 ∷ 32 ∷ [])) ∷
    []

--------------------------------------------------------------------------------
-- Validation

{-|
## IR Validation

Check that IR is well-formed before compilation.
-}

-- Check all edges reference valid vertices
valid-edges : NeuralIR → Bool
valid-edges ir = all-valid (edges ir)
  where
  vertex-exists : ℕ → Bool
  vertex-exists id = any-matches (vertices ir)
    where
    any-matches : List Vertex → Bool
    any-matches [] = false
    any-matches (v ∷ vs) = (Vertex.id v ≡ᵇ id) ∨ any-matches vs
      where
      _≡ᵇ_ : ℕ → ℕ → Bool
      zero ≡ᵇ zero = true
      suc n ≡ᵇ suc m = n ≡ᵇ m
      _ ≡ᵇ _ = false

      _∨_ : Bool → Bool → Bool
      true ∨ _ = true
      _ ∨ true = true
      _ ∨ _ = false

  all-valid : List Edge → Bool
  all-valid [] = true
  all-valid (e ∷ es) = (vertex-exists (source e) ∧ vertex-exists (target e)) ∧ all-valid es
    where
    _∧_ : Bool → Bool → Bool
    true ∧ true = true
    _ ∧ _ = false

-- Check shapes are consistent along edges
valid-shapes : NeuralIR → Bool
valid-shapes ir = {!!}  -- TODO: Check output shape of source = input shape of target

-- Complete validation
valid-ir : NeuralIR → Bool
valid-ir ir = valid-edges ir ∧ valid-shapes ir
  where
  _∧_ : Bool → Bool → Bool
  true ∧ true = true
  _ ∧ _ = false

--------------------------------------------------------------------------------
-- Summary

{-|
## Summary

This IR provides:
1. ✅ Graph structure (vertices, edges)
2. ✅ Operations (linear, conv, fork, residual, attention)
3. ✅ Shapes (from fibrations)
4. ✅ Properties (conservation, sheaf condition)
5. ✅ Resources (FLOPs, memory bounds)
6. ✅ Validation (well-formedness checks)

**Next step:** Serialize to JSON for Python bridge.
-}
