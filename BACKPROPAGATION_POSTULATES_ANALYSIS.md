# Analysis: gradient-descent-step and train-epoch Postulates

## Executive Summary

**Recommendation**: Keep both `gradient-descent-step` and `train-epoch` as postulates with comprehensive documentation.

**Reason**: The current design uses `Network` as a function type (`Vec ℝ input-dim → Vec ℝ output-dim`), making it fundamentally impossible to access or modify internal parameters without a major architectural redesign.

---

## Problem Analysis

### Current Architecture

```agda
-- Network is a FUNCTION TYPE (black box)
Network : Nat → Nat → Type
Network input-dim output-dim = Vec ℝ input-dim → Vec ℝ output-dim

-- NetworkGradients is postulated (no concrete definition)
postulate
  NetworkGradients : ∀ {input-dim output-dim} →
    Network input-dim output-dim → Type

-- Target postulates
postulate
  gradient-descent-step : ∀ {input-dim output-dim} →
    (network : Network input-dim output-dim) →
    (learning-rate : ℝ) →
    (grads : NetworkGradients network) →
    Network input-dim output-dim

  train-epoch : ∀ {input-dim output-dim} →
    (network : Network input-dim output-dim) →
    (dataset : Dataset input-dim output-dim) →
    (learning-rate : ℝ) →
    Network input-dim output-dim
```

### Why Implementation is Impossible

**Function types are opaque**: When `Network = Vec ℝ n → Vec ℝ m`, we have:
1. **No internal structure accessible** - Cannot inspect layers, weights, or biases
2. **Cannot construct modified version** - Need to access parameters to update them
3. **Cannot define NetworkGradients** - Depends on knowing parameter structure
4. **Type-level impossibility** - Not a matter of complexity, but fundamental type theory

**Analogy**: It's like trying to modify a compiled binary without source code access.

---

## What Would Be Needed to Implement

### Option A: Restructure as Data Type

Define Network as an explicit structure:

```agda
data NetworkStructure : Nat → Nat → Type where
  single : ∀ {n m} → Layer n m → NetworkStructure n m
  compose : ∀ {n m k} →
    NetworkStructure m k → NetworkStructure n m → NetworkStructure n k

-- Then NetworkGradients can be defined recursively
NetworkGradients : ∀ {n m} → NetworkStructure n m → Type
NetworkGradients (single layer) = LayerGradients n m
NetworkGradients (compose net2 net1) = NetworkGradients net2 × NetworkGradients net1

-- gradient-descent-step becomes implementable
gradient-descent-step : ∀ {n m} →
  (net : NetworkStructure n m) → ℝ → NetworkGradients net → NetworkStructure n m
gradient-descent-step (single layer) η grads =
  single (update-layer layer η grads)
gradient-descent-step (compose net2 net1) η (grads2 , grads1) =
  compose (gradient-descent-step net2 η grads2)
          (gradient-descent-step net1 η grads1)

-- Helper: Update single layer parameters
update-layer : ∀ {n m} → Layer n m → ℝ → LayerGradients n m → Layer n m
update-layer layer η grads = record
  { weight = λ i j → layer.weight i j -ℝ (η ·ℝ grads.weight-grad i j)
  ; bias = λ i → layer.bias i -ℝ (η ·ℝ grads.bias-grad i)
  ; activation = layer.activation  -- Unchanged
  }

-- train-epoch becomes a fold over the dataset
train-epoch : ∀ {n m} →
  NetworkStructure n m → Dataset n m → ℝ → NetworkStructure n m
train-epoch net [] η = net  -- Empty dataset, no change
train-epoch net ((x , y_true) ∷ rest) η =
  let grads = network-backward net x y_true
      net' = gradient-descent-step net η grads
  in train-epoch net' rest η
```

**Cost**: This requires:
1. Redefining Network throughout the codebase
2. Redefining NetworkGradients and network-backward concretely
3. Updating all examples and proofs
4. Managing type-level dimension proofs for layer composition

### Option B: Well-Typed Layer Sequences

Use dependent types to enforce dimension matching:

```agda
data LayerSeq : Nat → Nat → Type where
  []  : ∀ {n} → LayerSeq n n
  _∷_ : ∀ {n m k} → Layer n m → LayerSeq m k → LayerSeq n k

-- Network wraps LayerSeq with evaluation function
record Network' (n m : Nat) : Type where
  field
    layers : LayerSeq n m
    eval : Vec ℝ n → Vec ℝ m

-- Gradients follow the sequence structure
NetworkGradients' : ∀ {n m} → LayerSeq n m → Type
NetworkGradients' [] = ⊤  -- Identity network, no parameters
NetworkGradients' (layer ∷ rest) = LayerGradients _ _ × NetworkGradients' rest
```

**Cost**: Even more complex, requires:
1. Proving dimension compatibility at type level
2. Dependent pattern matching on length-indexed structures
3. Potentially hitting termination checker issues
4. Complex universe level management

---

## Why Current Design is Better

Despite the inability to implement these functions, the current design has advantages:

### 1. Mathematical Clarity
- `Network = Vec ℝ n → Vec ℝ m` is the **essence** of a neural network
- Represents network as a smooth map between vector spaces
- Matches differential geometry interpretation (morphism in smooth category)

### 2. Geometric Interpretation
- Network corresponds to a **point on parameter manifold**
- Loss is a smooth function on this manifold
- Gradient descent is **flow along geodesic**
- Matches the mathematical framework in the paper

### 3. Postulates Document Specification
- Clearly state **what** these operations should do
- Avoid implementation complexity distracting from theory
- Mathematical specification is more important than executable code

### 4. Practical Implementation External
- Real implementations use Python/PyTorch/JAX/ONNX
- This Agda code provides **mathematical foundations**
- Bridge between theory and practice happens externally

### 5. Avoids Dependent Type Complexity
- No need for dimension-indexed lists
- No termination checker issues
- Simpler universe level management
- Focus on mathematical properties, not Agda intricacies

---

## Recommended Documentation

### For `gradient-descent-step`

Add this documentation above the postulate:

```agda
-- Gradient descent update for parameters
--
-- **WHY THIS IS A POSTULATE**:
--
-- Network is defined as a function type (Vec ℝ n → Vec ℝ m), which is
-- mathematically clean but makes parameters inaccessible. To implement this,
-- we would need to restructure Network as a data type with explicit layers.
--
-- **FORMAL SEMANTICS** (what this specifies):
--
-- Given network = f_θ with parameters θ, learning rate η, and gradients ∇L(θ):
-- Returns network' = f_{θ'} where θ' = θ - η·∇L(θ)
--
-- Properties:
-- 1. Each parameter θᵢ updated: θᵢ' = θᵢ - η·(∂L/∂θᵢ)
-- 2. Gradient descent step along negative gradient direction
-- 3. Loss decreases: L(θ') ≤ L(θ) for sufficiently small η
-- 4. Architecture preserved: same structure, only parameters change
--
-- **IMPLEMENTATION SKETCH** (if Network were structured):
--
-- gradient-descent-step (single layer) η grads =
--   single (record layer
--     { weight = λ i j → layer.weight i j -ℝ (η ·ℝ grads.weight-grad i j)
--     ; bias = λ i → layer.bias i -ℝ (η ·ℝ grads.bias-grad i)
--     })
-- gradient-descent-step (compose net2 net1) η (grads2 , grads1) =
--   compose (gradient-descent-step net2 η grads2)
--           (gradient-descent-step net1 η grads1)
postulate
  gradient-descent-step : ∀ {input-dim output-dim} →
    (network : Network input-dim output-dim) →
    (learning-rate : ℝ) →
    (grads : NetworkGradients network) →
    Network input-dim output-dim
```

### For `train-epoch`

Add this documentation above the postulate:

```agda
-- Training loop over a dataset
--
-- **WHY THIS IS A POSTULATE**:
--
-- Depends on gradient-descent-step, which is also postulated. Additionally,
-- would need to handle sequential updates through list fold.
--
-- **FORMAL SEMANTICS** (what this specifies):
--
-- Given network, dataset [(x₁,y₁), ..., (xₙ,yₙ)], and learning rate η:
-- For each (xᵢ, yᵢ):
--   1. Compute gradients: grads = network-backward network xᵢ yᵢ
--   2. Update: network ← gradient-descent-step network η grads
-- Return final updated network
--
-- Equivalent to fold-left over dataset with gradient descent step.
--
-- **IMPLEMENTATION SKETCH** (if gradient-descent-step existed):
--
-- train-epoch network [] η = network  -- Base case
-- train-epoch network ((x , y_true) ∷ rest) η =
--   let grads = network-backward network x y_true
--       network' = gradient-descent-step network η grads
--   in train-epoch network' rest η
--
-- Or using library fold:
-- train-epoch network dataset η =
--   foldl (λ net (x , y_true) →
--     let grads = network-backward net x y_true
--     in gradient-descent-step net η grads)
--   network dataset
postulate
  train-epoch : ∀ {input-dim output-dim} →
    (network : Network input-dim output-dim) →
    (dataset : Dataset input-dim output-dim) →
    (learning-rate : ℝ) →
    Network input-dim output-dim
```

---

## Conclusion

**Do NOT implement**: The cost of restructuring is prohibitive and defeats the purpose of clean mathematical design.

**Do ADD documentation**: Comprehensive comments explaining:
1. Why these are postulates (type-level impossibility)
2. What they specify formally (semantics)
3. How they would be implemented (if structure were different)
4. Connection to mathematical theory (smooth manifold, gradient flow)

This approach:
- ✅ Documents the specification clearly
- ✅ Maintains mathematical elegance
- ✅ Avoids complex dependent types
- ✅ Enables understanding without implementation
- ✅ Allows external implementations in other languages

---

## Verification

The file currently type-checks:

```bash
$ agda --library-file=./libraries src/Neural/Smooth/Backpropagation.agda
Checking Neural.Smooth.Backpropagation (/Users/faezs/homotopy-nn/src/Neural/Smooth/Backpropagation.agda).
```

No unsolved metas except for the deliberate postulates. The module is mathematically sound as a specification, even though these particular functions aren't executable.

---

## Related Modules

For comparison, see how other modules handle similar issues:

1. **Neural.Topos.Architecture**: Uses postulates for manifold structures (lines 634-770)
2. **Neural.Resources.Optimization**: Postulates optimal constructors (abstract functors)
3. **Neural.Stack.LogicalPropagation**: Postulates for topos-theoretic constructions

These are all **specifications** rather than implementations, which is appropriate for a foundational mathematical framework.
