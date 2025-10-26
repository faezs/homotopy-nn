-- Temporary file to craft the replacement text
-- This will be inserted into Backpropagation.agda at lines 272-277

{-|
## Why List-Based Network Construction Doesn't Work

**Attempted signature**:
```agda
Network-from-layers : ∀ {input-dim output-dim} →
  (layers : List (Σ[ n ∈ Nat ] Σ[ m ∈ Nat ] Layer n m)) →
  Network input-dim output-dim
```

**Fundamental problem**: This type signature makes **no guarantees** that:
1. Layers can actually compose (layer i's output dim = layer i+1's input dim)
2. The first layer accepts `input-dim` inputs
3. The last layer produces `output-dim` outputs

**Example of invalid input that would type-check**:
```agda
bad-layers : List (Σ Nat (Σ Nat ∘ Layer))
bad-layers = (2 , 5 , layer₁) ∷ (3 , 7 , layer₂) ∷ []
-- Layer 1: ℝ² → ℝ⁵ (outputs 5 dimensions)
-- Layer 2: ℝ³ → ℝ⁷ (expects 3 inputs, but gets 5!)
-- Dimension mismatch! 5 ≠ 3, but List type doesn't catch this!
```

This compiles but fails at runtime - **not type-safe**!

**Why this is impossible to fix**:
- Lists are **homogeneous**: all elements have the same type
- But `Layer n m` has **different types** for different dimensions
- We cannot express "layer i+1's input = layer i's output" in List type
- The connection between consecutive layers requires **dependent types**

**TYPE-SAFE SOLUTION**: Use `LayeredNetwork` (defined in § 4.1 below)

`LayeredNetwork` is an **indexed inductive type** that enforces dimension
compatibility at the type level:

```agda
data LayeredNetwork : Nat → Nat → Type where
  single-layer : ∀ {n m} →
    Layer n m →
    LayeredNetwork n m

  compose-layer : ∀ {n k m} →
    LayeredNetwork n k →  -- First part: n → k
    Layer k m →           -- Last layer: k → m
    LayeredNetwork n m    -- Result: n → m
```

The `compose-layer` constructor **enforces** dimension compatibility:
- If `rest : LayeredNetwork n k` (outputs k-dimensional vectors)
- And `last : Layer k m` (accepts k-dimensional inputs)
- Then they compose to `LayeredNetwork n m`
- The intermediate dimension `k` is **automatically matched** by Agda's unification!

**Example of type-safe construction**:
```agda
-- Build a 2 → 3 → 1 network
layer1 : Layer 2 3    -- ℝ² → ℝ³
layer2 : Layer 3 1    -- ℝ³ → ℝ¹

network : LayeredNetwork 2 1
network = compose-layer (single-layer layer1) layer2
-- Type checker verifies: layer1's output (3) = layer2's input (3) ✓
```

**Attempting dimension mismatch is a type error**:
```agda
bad-network : LayeredNetwork 2 1
bad-network = compose-layer (single-layer layer1) bad-layer
  where
    layer1 : Layer 2 5      -- ℝ² → ℝ⁵
    bad-layer : Layer 3 1   -- ℝ³ → ℝ¹
-- Type error! Cannot unify 5 ≠ 3
-- Agda rejects this at compile time!
```

This demonstrates the **power of dependent types for correctness**.

**See § 4.1** below for the full `LayeredNetwork` definition and its use in
backpropagation. The `eval-network`, `backward-with-grad`, and `network-backward`
functions all operate on `LayeredNetwork` and guarantee type-safe gradient computation.
-}

-- NOTE: The postulate below is intentionally left undefined.
-- It documents a fundamentally flawed approach that cannot be implemented type-safely.
-- Do NOT attempt to implement - use LayeredNetwork (§ 4.1) instead!
postulate
  Network-from-layers : ∀ {input-dim output-dim} →
    (layers : List (Σ[ n ∈ Nat ] Σ[ m ∈ Nat ] Layer n m)) →
    Network input-dim output-dim
