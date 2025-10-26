{-# OPTIONS --no-import-sorts #-}
{-|
# Implementation Sketch: gradient-descent-step and train-epoch

This file shows HOW the postulates would be implemented if Network
were restructured as an explicit data type.

**NOTE**: This is NOT a working module. It's a sketch showing what
would be needed.

**Status**: For documentation purposes only.
-}

module IMPLEMENTATION_SKETCH where

open import 1Lab.Prelude
open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin)
open import Data.List.Base using (List; []; _∷_; foldl)
open import Data.Vec.Base using (Vec)

-- Assume we have these from Backpropagation.agda
postulate
  ℝ : Type
  _+ℝ_ : ℝ → ℝ → ℝ
  _-ℝ_ : ℝ → ℝ → ℝ
  _·ℝ_ : ℝ → ℝ → ℝ

-- Layer definition (from Backpropagation.agda)
record Layer (n m : Nat) : Type where
  field
    weight : Fin m → Fin n → ℝ
    bias : Fin m → ℝ
    activation : ℝ → ℝ

-- Gradients for a single layer (from Backpropagation.agda)
record LayerGradients (n m : Nat) : Type where
  field
    weight-grad : Fin m → Fin n → ℝ
    bias-grad : Fin m → ℝ
    input-grad : Vec ℝ n

--------------------------------------------------------------------------------
-- § 1: Structured Network Definition

{-|
## NetworkStructure: Explicit Layer Composition

Instead of Network = Vec ℝ n → Vec ℝ m (function type),
we define an explicit structure that we can inspect and modify.
-}

data NetworkStructure : Nat → Nat → Type where
  -- Single layer network
  single : ∀ {n m} → Layer n m → NetworkStructure n m

  -- Composed network: net2 ∘ net1
  compose : ∀ {n m k} →
    NetworkStructure m k →  -- Second network (output layer)
    NetworkStructure n m →  -- First network (input layer)
    NetworkStructure n k    -- Composed network

{-|
## NetworkGradients: Recursive Structure

Now we can define gradients recursively, matching the network structure.
-}

NetworkGradients : ∀ {n m} → NetworkStructure n m → Type
NetworkGradients (single layer) = LayerGradients _ _
NetworkGradients (compose net2 net1) =
  NetworkGradients net2 × NetworkGradients net1

{-|
## Evaluation Function

Convert NetworkStructure back to function for forward pass.
-}

postulate
  apply-layer : ∀ {n m} → Layer n m → Vec ℝ n → Vec ℝ m

eval-network : ∀ {n m} → NetworkStructure n m → Vec ℝ n → Vec ℝ m
eval-network (single layer) x = apply-layer layer x
eval-network (compose net2 net1) x =
  let hidden = eval-network net1 x
  in eval-network net2 hidden

--------------------------------------------------------------------------------
-- § 2: Update Single Layer

{-|
## update-layer: Apply Gradients to One Layer

Given a layer and its gradients, produce updated layer with:
  W' = W - η·∇W
  b' = b - η·∇b
-}

update-layer : ∀ {n m} →
  Layer n m →             -- Original layer
  ℝ →                     -- Learning rate η
  LayerGradients n m →    -- Gradients
  Layer n m               -- Updated layer
update-layer layer η grads = record
  { weight = λ i j →
      Layer.weight layer i j -ℝ (η ·ℝ LayerGradients.weight-grad grads i j)
  ; bias = λ i →
      Layer.bias layer i -ℝ (η ·ℝ LayerGradients.bias-grad grads i)
  ; activation = Layer.activation layer  -- Unchanged
  }

--------------------------------------------------------------------------------
-- § 3: Gradient Descent Step (IMPLEMENTATION)

{-|
## gradient-descent-step: Recursive Parameter Update

**This is the actual implementation** of the postulate in Backpropagation.agda.

Pattern match on network structure:
- Single layer: Update that one layer
- Composed: Recursively update both sub-networks

**Key insight**: We can access the structure and parameters!
-}

gradient-descent-step : ∀ {n m} →
  NetworkStructure n m →           -- Network
  ℝ →                               -- Learning rate
  NetworkGradients (net) →          -- Gradients (dependent on net!)
  NetworkStructure n m              -- Updated network
gradient-descent-step (single layer) η grads =
  single (update-layer layer η grads)

gradient-descent-step (compose net2 net1) η (grads2 , grads1) =
  let net2' = gradient-descent-step net2 η grads2
      net1' = gradient-descent-step net1 η grads1
  in compose net2' net1'

--------------------------------------------------------------------------------
-- § 4: Backward Pass

{-|
## network-backward: Compute Gradients

This would also need to be implemented to compute gradients recursively.
-}

postulate
  layer-backward : ∀ {n m} →
    Layer n m →
    Vec ℝ n →           -- Input
    Vec ℝ m →           -- Output gradient
    LayerGradients n m

  cross-entropy : ∀ {n} → Vec ℝ n → Vec ℝ n → ℝ

  -- Initial output gradient from loss function
  loss-gradient : ∀ {n} →
    Vec ℝ n →  -- Predicted
    Vec ℝ n →  -- True
    Vec ℝ n    -- ∂L/∂output

network-backward : ∀ {n m} →
  NetworkStructure n m →
  Vec ℝ n →              -- Input
  Vec ℝ m →              -- True output
  NetworkGradients _     -- Gradients

network-backward (single layer) x y_true =
  let output = apply-layer layer x
      output-grad = loss-gradient output y_true
  in layer-backward layer x output-grad

network-backward (compose net2 net1) x y_true =
  let -- Forward pass to get intermediate activations
      hidden = eval-network net1 x
      output = eval-network net2 hidden

      -- Backward pass through net2
      output-grad = loss-gradient output y_true
      grads2 = network-backward net2 hidden y_true

      -- Backward pass through net1 (using gradient from net2)
      -- Need to extract input-grad from last layer of grads2
      -- (This is a simplification - real implementation more complex)
      postulate hidden-grad : Vec ℝ _
      grads1 = network-backward net1 x hidden-grad  -- Simplified

  in (grads2 , grads1)

--------------------------------------------------------------------------------
-- § 5: Training Epoch (IMPLEMENTATION)

{-|
## train-epoch: Fold Over Dataset

**This is the actual implementation** of the postulate in Backpropagation.agda.

For each (input, target) pair:
1. Compute gradients via backprop
2. Update network via gradient descent
3. Continue with updated network

Implemented as left fold over dataset.
-}

Dataset : Nat → Nat → Type
Dataset n m = List (Vec ℝ n × Vec ℝ m)

-- Single training step
train-step : ∀ {n m} →
  NetworkStructure n m →
  ℝ →                              -- Learning rate
  (Vec ℝ n × Vec ℝ m) →            -- One training example
  NetworkStructure n m
train-step net η (x , y_true) =
  let grads = network-backward net x y_true
  in gradient-descent-step net η grads

-- Training epoch: fold train-step over dataset
train-epoch : ∀ {n m} →
  NetworkStructure n m →
  Dataset n m →
  ℝ →
  NetworkStructure n m
train-epoch net dataset η =
  foldl (λ net' example → train-step net' η example) net dataset

{-|
Alternative explicit recursive definition:
-}

train-epoch-recursive : ∀ {n m} →
  NetworkStructure n m →
  Dataset n m →
  ℝ →
  NetworkStructure n m
train-epoch-recursive net [] η = net  -- Empty dataset, no change
train-epoch-recursive net ((x , y_true) ∷ rest) η =
  let grads = network-backward net x y_true
      net' = gradient-descent-step net η grads
  in train-epoch-recursive net' rest η

--------------------------------------------------------------------------------
-- § 6: Example Usage

{-|
## Concrete Example

Show how this would work with a 2-layer network.
-}

postulate
  -- Example layers
  layer1 : Layer 2 3  -- Input ℝ² → Hidden ℝ³
  layer2 : Layer 3 1  -- Hidden ℝ³ → Output ℝ¹

  -- Example dataset
  training-data : Dataset 2 1

  -- Learning rate
  η : ℝ

-- Construct network
example-net : NetworkStructure 2 1
example-net = compose (single layer2) (single layer1)

-- Train for one epoch
trained-net : NetworkStructure 2 1
trained-net = train-epoch example-net training-data η

-- Evaluate trained network
postulate input : Vec ℝ 2
output : Vec ℝ 1
output = eval-network trained-net input

--------------------------------------------------------------------------------
-- § 7: Why This Isn't Done

{-|
## Cost-Benefit Analysis

**Benefits** of structured approach:
- Can implement gradient-descent-step and train-epoch
- Can inspect network structure
- Can serialize/deserialize networks
- More like practical deep learning frameworks

**Costs** of structured approach:
- Much more complex type definitions
- Need to manage dimension proofs (n matches m between layers)
- Lose mathematical elegance (network as morphism)
- Doesn't match geometric interpretation (point on manifold)
- Breaks existing proofs and examples
- Requires rewriting ~500 lines of code
- Still can't compile to efficient code (that's Python's job)

**Verdict**: Costs outweigh benefits for a *theoretical* framework.

**Current approach** (postulates):
- ✅ Clean mathematical specification
- ✅ Matches geometric/categorical intuition
- ✅ Avoids implementation complexity
- ✅ Documents what the operations *mean*
- ✅ External tools handle actual execution

**Structured approach** (this file):
- ❌ Complex dependent types
- ❌ Loses mathematical elegance
- ❌ Large refactoring required
- ❌ Still not executable efficiently
- ✅ Shows implementation is *possible* in principle
-}

--------------------------------------------------------------------------------
-- § 8: Connection to Current Code

{-|
## Bridging the Gap

Current Backpropagation.agda has:
```agda
Network : Nat → Nat → Type
Network n m = Vec ℝ n → Vec ℝ m
```

This module shows that if we instead had:
```agda
NetworkStructure : Nat → Nat → Type
NetworkStructure n m = ... (explicit structure)
```

Then we could:
1. Define NetworkGradients concretely
2. Implement gradient-descent-step
3. Implement train-epoch
4. Implement network-backward

But we choose NOT to because:
- Mathematical specification > Executable implementation
- External tools (PyTorch/JAX) handle practice
- This code establishes theory
-}
