{-# OPTIONS --no-import-sorts #-}

{-|
# Simple MLP Dynamical Objects (Section 1.2)

Implementation of the three functors from Section 1.2:
1. **X^w**: Activities functor (parametrized by weights w)
2. **W**: Weights functor
3. **X**: Total dynamics functor (combines activities and weights)

These capture the computational dynamics of the neural network.
-}

module examples.SimpleMLP.Dynamics where

open import 1Lab.Prelude
open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin; fzero; fsuc)
open import Data.List.Base

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Instances.FinSets using (FinSets)

open import examples.SimpleMLP.Graph

--------------------------------------------------------------------------------
-- State Spaces and Weight Spaces
--------------------------------------------------------------------------------

{-|
We'll use finite-dimensional real vector spaces.
For simplicity, represent as dimension counts.
-}

-- Postulate real numbers and operations
postulate
  ℝ : Type
  _+ᵣ_ : ℝ → ℝ → ℝ
  _*ᵣ_ : ℝ → ℝ → ℝ
  relu : ℝ → ℝ
  softmax : List ℝ → List ℝ

-- Vector spaces (represented by dimension)
Vec : Nat → Type
Vec n = Fin n → ℝ

-- Matrix spaces
Matrix : Nat → Nat → Type
Matrix m n = Fin m → Fin n → ℝ

--------------------------------------------------------------------------------
-- Layer Dimensions
--------------------------------------------------------------------------------

{-|
Simple MLP dimensions:
- Vertex 0 (input): 784 dimensions (28×28 MNIST)
- Vertex 1 (hidden): 256 dimensions
- Vertex 2 (output): 10 dimensions (digit classes)
-}

input-dim : Nat
input-dim = 784

hidden-dim : Nat
hidden-dim = 256

output-dim : Nat
output-dim = 10

-- State space for each vertex
vertex-state-space : Fin 3 → Nat
vertex-state-space fzero = input-dim           -- vertex 0
vertex-state-space (fsuc fzero) = hidden-dim   -- vertex 1
vertex-state-space (fsuc (fsuc fzero)) = output-dim  -- vertex 2

--------------------------------------------------------------------------------
-- Weight Spaces (Functor W)
--------------------------------------------------------------------------------

{-|
Weight functor W: simple-mlp-graph → FinSets

For each edge e, W(e) is the space of connection weights.
For edge from layer of size m to layer of size n:
  W(e) = ℝ^(m×n) (weight matrix)
-}

-- Weight space dimensions for each edge
edge-weight-space : Fin 2 → (Nat × Nat)
edge-weight-space fzero = (input-dim , hidden-dim)   -- edge 0: 784 × 256
edge-weight-space (fsuc fzero) = (hidden-dim , output-dim)  -- edge 1: 256 × 10

-- Weight functor W
W : Functor simple-mlp-graph FinSets
W .Functor.F₀ v = vertex-state-space v  -- Objects: state spaces
W .Functor.F₁ {v} {v'} e =
  -- For morphisms, we don't have direct edge→vertex maps in this setup
  -- This needs refinement based on the actual edge
  λ _ → fzero  -- Placeholder
W .Functor.F-id = refl
W .Functor.F-∘ f g = refl

--------------------------------------------------------------------------------
-- Activities Functor (X^w for fixed weights w)
--------------------------------------------------------------------------------

{-|
Activities functor X^w: simple-mlp-graph → FinSets

For each vertex v, X^w(v) is the space of neural activities.
For each edge e: v → v', X^w(e) is the transition function
  X^w(e) : X^w(v) → X^w(v')

This depends on:
- The edge's weight matrix w_e ∈ W(e)
- The activation function
-}

-- Activation functions for each layer
activation-fn : Fin 3 → (List ℝ → List ℝ)
activation-fn fzero = λ x → x            -- input: identity
activation-fn (fsuc fzero) = map relu    -- hidden: ReLU
activation-fn (fsuc (fsuc fzero)) = softmax  -- output: Softmax

-- Layer transformation
-- Given weight matrix W ∈ ℝ^(m×n) and input x ∈ ℝ^m, compute Wx ∈ ℝ^n
postulate
  matrix-vector-mult : {m n : Nat} → Matrix m n → Vec m → Vec n

-- Parametrized activities functor
module ActivitiesFunctor (weights : (e : Fin 2) → let (m , n) = edge-weight-space e in Matrix m n) where

  {-|
  X^w: For each vertex, the state space
       For each edge, the transition function
  -}

  X^w : Functor simple-mlp-graph FinSets
  X^w .Functor.F₀ v = vertex-state-space v

  -- Edge transitions
  X^w .Functor.F₁ {false} {true} true = λ e →
    -- Source function: doesn't transform state
    λ x → fzero  -- Placeholder

  X^w .Functor.F₁ {false} {true} false = λ e →
    -- Target function: apply weight and activation
    let (m , n) = edge-weight-space e
        w = weights e
    in λ (x : Fin m) → fzero  -- Placeholder: should be (activation ∘ (w *) ) x

  X^w .Functor.F₁ {false} {false} tt = λ e → e
  X^w .Functor.F₁ {true} {true} tt = λ v → v
  X^w .Functor.F-id {false} = refl
  X^w .Functor.F-id {true} = refl
  X^w .Functor.F-∘ {false} {false} {false} f g = refl
  X^w .Functor.F-∘ {false} {false} {true} f g = refl
  X^w .Functor.F-∘ {false} {true} {true} f g = refl
  X^w .Functor.F-∘ {true} {true} {true} f g = refl

--------------------------------------------------------------------------------
-- Total Dynamics Functor (X)
--------------------------------------------------------------------------------

{-|
Total dynamics X combines activities and weights.

From the paper: "The data structure is a morphism of oriented graphs
  (X^w, W, X): G → FinSets"

Where:
- X^w(v) = activity space at vertex v
- W(e) = weight space for edge e
- X = X^w × W (roughly speaking)
-}

-- Product functor combining activities and weights
module TotalDynamics (weights : (e : Fin 2) → let (m , n) = edge-weight-space e in Matrix m n) where

  open ActivitiesFunctor weights

  {-|
  Total state space: activities at each vertex + weights on each edge
  -}

  -- Total state at a vertex: just the activity
  vertex-total-state : Fin 3 → Type
  vertex-total-state v = Vec (vertex-state-space v)

  -- Total state for an edge: activity + weight
  edge-total-state : Fin 2 → Type
  edge-total-state e =
    let (m , n) = edge-weight-space e
        src = mlp-source e
        tgt = mlp-target e
    in Vec (vertex-state-space src) × Matrix m n × Vec (vertex-state-space tgt)

  {-|
  Forward propagation: compute output activities given input and weights
  -}

  postulate
    forward-prop : (e : Fin 2) → edge-total-state e → Vec (vertex-state-space (mlp-target e))

  {-|
  Backward propagation: compute gradients for learning

  From Section 1.4: Backpropagation as natural transformations
  -}

  postulate
    backward-prop : (e : Fin 2) →
                   Vec (vertex-state-space (mlp-target e)) →  -- Gradient from output
                   edge-total-state e →                        -- Current state
                   Matrix (let (m , n) = edge-weight-space e in m)
                         (let (m , n) = edge-weight-space e in n)  -- Weight gradient

--------------------------------------------------------------------------------
-- Concrete Example Instantiation
--------------------------------------------------------------------------------

{-|
Example: Random initialization of weights
-}

postulate
  random-matrix : (m n : Nat) → Matrix m n

example-weights : (e : Fin 2) → let (m , n) = edge-weight-space e in Matrix m n
example-weights fzero = random-matrix input-dim hidden-dim
example-weights (fsuc fzero) = random-matrix hidden-dim output-dim

-- Instantiate functors with example weights
module Example where
  open ActivitiesFunctor example-weights public
  open TotalDynamics example-weights public

--------------------------------------------------------------------------------
-- Connection to ONNX Export
--------------------------------------------------------------------------------

{-|
The dynamics defined here correspond to ONNX operations:

Vertex 0 → Vertex 1 (edge 0):
  ONNX: Gemm(input, hidden_weight, hidden_bias) → ReLU
  Agda: matrix-vector-mult (weights e0) input |> map relu

Vertex 1 → Vertex 2 (edge 1):
  ONNX: Gemm(hidden, output_weight, output_bias) → Softmax
  Agda: matrix-vector-mult (weights e1) hidden |> softmax

The ONNX initializers (TensorProto) contain the weight matrices.
The ONNX operators (NodeProto) implement the transition functions.
-}

-- Weight matrix dimensions match ONNX initializers
weight-dims-match : (e : Fin 2) →
  edge-weight-space e ≡
  (vertex-state-space (mlp-source e) , vertex-state-space (mlp-target e))
weight-dims-match fzero = refl
weight-dims-match (fsuc fzero) = refl
