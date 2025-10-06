{-# OPTIONS --no-import-sorts #-}
{-|
# Poset Diagram for FFN Example

Recreating Figure 1.4 from Belfiore & Bennequin (2022).

## The Feedforward Network Γ:

```
Network Architecture (information flows ↓):

    x₀,₁        x₀,₂          (Input layers)
      ↓          ↓
      └──→ h ←──┘             (Hidden layer - CONVERGENCE!)
            ↓
            ↓
        [multiple outputs]
            ↓
    y₁  y₂  y₃  y₄  y₅        (Output layers)
```

## After Fork Construction:

At hidden layer h, we have convergence (two inputs), so we add:
- **A★** (star vertex): join point for x₀,₁ and x₀,₂
- **A** (tang vertex): transmission point to h

```
Fork Structure in Γ_fork:

    x₀,₁  →  A★  ←  x₀,₂      (tips → star)
              ↓
              A                (star → tang)
              ↓
              h                (tang → handle)
              ↓
          [outputs]
              ↓
      y₁  y₂  y₃  y₄  y₅
```

## Poset X (remove A★, reverse arrows):

In the categorical convention, arrows go **opposite** to information flow.

```
Poset X structure:

MAXIMAL ELEMENTS (where arrows come FROM):
    x₀,₁        x₀,₂          (input layers)
       ↘        ↙
          A                    (tang - fork join point)
          ↓
          h                    (handle - hidden layer)
          ↓
      ┌───┼───┬───┬───┐
      ↓   ↓   ↓   ↓   ↓
MINIMAL ELEMENTS (where arrows go TO):
     y₁  y₂  y₃  y₄  y₅        (output layers)
```

## Ordering Relations in Poset X:

```agda
data _≤_ : X-Vertex → X-Vertex → Type where
  -- Each output is below everything
  y₁ ≤ h,  y₂ ≤ h,  y₃ ≤ h,  y₄ ≤ h,  y₅ ≤ h

  -- Hidden below tang
  h ≤ A

  -- Tang below both inputs (this is the KEY fork property!)
  A ≤ x₀,₁
  A ≤ x₀,₂

  -- Plus reflexivity and transitivity
```

## The Tree Structure (Theorem 1.2):

From the paper:
> "The poset X of a DNN is made by a finite number of trees,
> rooted in the maximal points and which are joined in the minimal points."

**Two trees rooted at inputs:**

Tree 1 (rooted at x₀,₁):
```
    x₀,₁
      ↓
      A
      ↓
      h
    ↙ ↓ ↘ ↘ ↘
   y₁ y₂ y₃ y₄ y₅
```

Tree 2 (rooted at x₀,₂):
```
    x₀,₂
      ↓
      A
      ↓
      h
    ↙ ↓ ↘ ↘ ↘
   y₁ y₂ y₃ y₄ y₅
```

These trees **join** at the outputs (y₁, ..., y₅), which are the minimal elements.

The tang **A** is where the two input branches merge!

## Alexandrov Coverage on X:

For each vertex x ∈ X, the covering is the **maximal sieve** = all y with y ≤ x.

- **Cover(y₁)** = {y₁}  (only itself)
- **Cover(h)**  = {y₁, y₂, y₃, y₄, y₅, h}
- **Cover(A)**  = {y₁, y₂, y₃, y₄, y₅, h, A}
- **Cover(x₀,₁)** = {y₁, y₂, y₃, y₄, y₅, h, A, x₀,₁}  (entire tree)
- **Cover(x₀,₂)** = {y₁, y₂, y₃, y₄, y₅, h, A, x₀,₂}  (entire tree)

These form a **topology** where information "flows downward" from inputs to outputs.

## The DNN Topos:

Objects: Sheaves F : X^op → Sets
- F(x₀,₁) = possible input states at layer 1
- F(x₀,₂) = possible input states at layer 2
- F(A) = **joint states** after convergence (product-like)
- F(h) = hidden layer activation states
- F(yᵢ) = output states at layer i

Morphisms: Natural transformations preserving layer structure

The **feed-forward dynamics** X^w is a sheaf where:
- X^w(layer) = activity states for given weights w
- Restrictions follow the network's information flow
-}

module Neural.Topos.PosetDiagram where

open import 1Lab.Prelude

{-|
## Concrete Agda Encoding

Let's encode the 5-output convergent network:
-}

data Vertex : Type where
  input₁ : Vertex  -- x₀,₁
  input₂ : Vertex  -- x₀,₂
  tang   : Vertex  -- A (from fork)
  hidden : Vertex  -- h (convergent layer)
  out₁   : Vertex  -- y₁
  out₂   : Vertex  -- y₂
  out₃   : Vertex  -- y₃
  out₄   : Vertex  -- y₄
  out₅   : Vertex  -- y₅

-- The partial order (arrows go opposite to information flow)
data _≤_ : Vertex → Vertex → Type where
  -- Reflexivity
  ≤-refl : ∀ {x} → x ≤ x

  -- Outputs to hidden
  ≤-out1 : out₁ ≤ hidden
  ≤-out2 : out₂ ≤ hidden
  ≤-out3 : out₃ ≤ hidden
  ≤-out4 : out₄ ≤ hidden
  ≤-out5 : out₅ ≤ hidden

  -- Hidden to tang
  ≤-hid : hidden ≤ tang

  -- Tang to inputs (THE FORK!)
  ≤-tang1 : tang ≤ input₁
  ≤-tang2 : tang ≤ input₂

  -- Transitivity
  ≤-trans : ∀ {x y z} → x ≤ y → y ≤ z → x ≤ z

{-|
## Example Ordering Chains:

From outputs to inputs, we have chains like:

```
out₁ ≤ hidden ≤ tang ≤ input₁
out₁ ≤ hidden ≤ tang ≤ input₂
out₂ ≤ hidden ≤ tang ≤ input₁
out₂ ≤ hidden ≤ tang ≤ input₂
...
```

Note that:
- **Minimal elements**: out₁, out₂, out₃, out₄, out₅
- **Maximal elements**: input₁, input₂, tang
- **Join point**: tang (where the two input branches merge)

This matches Figure 1.4 from the paper!
-}

-- Example: out₁ can reach both inputs via tang
out₁-to-input₁ : out₁ ≤ input₁
out₁-to-input₁ = ≤-trans (≤-trans ≤-out1 ≤-hid) ≤-tang1

out₁-to-input₂ : out₁ ≤ input₂
out₁-to-input₂ = ≤-trans (≤-trans ≤-out1 ≤-hid) ≤-tang2

{-|
## Visualization Summary:

```
ASCII Art Poset Diagram:

        input₁           input₂       ← MAXIMAL (inputs)
           ↘              ↙
             tang (A)                  ← MAXIMAL (fork tang)
                ↓
             hidden (h)
             ↙ | | | ↘
         out₁ out₂ out₃ out₄ out₅    ← MINIMAL (outputs)
```

This is the **diamond with multiple bottom elements** structure
that appears in real feedforward networks with:
- Multiple inputs converging
- Shared hidden layers
- Multiple output tasks

The **Alexandrov coverage** makes this into a site, and
**Sh[X, Alexandrov]** is the DNN topos containing all possible
sheaves (= layer-wise compatible state assignments).
-}
