{-# OPTIONS --rewriting --guardedness --cubical --no-load-primitives #-}

{-|
# Localic Topos: Base Definitions

This module contains the foundational definitions for localic toposes and Ω-sets.

## Contents
- §A.1: Complete Heyting Algebras (Frames/Locales)
- §A.2: Ω-Sets (Fuzzy Sets)
- §A.3: Morphisms in SetΩ (Equations 19-22)
-}

module Neural.Topos.Localic.Base where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path
open import 1Lab.Equiv
open import 1Lab.Type.Sigma

open import Order.Base
open import Order.Frame
open import Order.Heyting
open import Order.Diagram.Meet

--------------------------------------------------------------------------------
-- §A.1: Complete Heyting Algebras (Frames/Locales)

{-|
## Complete Heyting Algebra = Frame + Heyting

A **complete Heyting algebra** Ω is a poset with:
1. Frame structure (arbitrary joins ⋃, finite meets ∩, distributivity)
2. Heyting algebra structure (implication ⇨)

**Names**:
- Heyting algebra: Intuitionistic logic structure
- Frame: Complete lattice with distributive joins
- Locale: Frame viewed as "generalized space"

**Examples**:
1. **Open sets** of topological space X
2. **Alexandrov topology** on poset (lower sets)
3. **Power set** 2^X (classical logic, Boolean)
4. **Truth values** in topos (subobjects of 1)

**DNN Interpretation**:
- Ω = Truth values for network decisions
- top = Fully certain
- bot = Fully uncertain
- a ∩ b = Both conditions hold
- a ∪ b = At least one holds
- a ⇨ b = Implication (if a then b)
-}

record CompleteHeytingAlgebra (o ℓ : Level) : Type (lsuc (o ⊔ ℓ)) where
  no-eta-equality
  field
    poset : Poset o ℓ
    frame : is-frame poset
    heyting : is-heyting-algebra poset

  -- Strategy: Use heyting operations consistently to avoid projection mismatches
  -- The heyting algebra is built on the frame, so it has all operations we need

  open Poset poset public

  -- Open heyting algebra fully for ∩, ⇨, and other operations
  open is-heyting-algebra heyting public

  -- Get frame's ⋃ operation and top (heyting doesn't have these)
  -- Do NOT import ∩ operations from frame - use heyting's instead for consistency
  open is-frame frame public
    using (⋃; ⋃-lubs; ⋃-universal; ⋃-inj; ⋃-apᶠ; ⋃-twice; ⋃-apⁱ; ⋃-distribl; top; has-top; !)

  -- Add meet reasoning combinators from heyting's meet structure
  open import Order.Diagram.Meet.Reasoning (∩-meets) public
    using (∩≤∩; ∩≤l; ∩≤r; ∩-universal; ∩-comm; ∩-assoc)

  -- Convenient aliases matching paper notation
  ⋁ : {I : Type o} → (I → Ob) → Ob
  ⋁ = ⋃

open CompleteHeytingAlgebra public
  hiding (⋃-apᶠ; ⋃-twice; ⋃-apⁱ; ⋃-distribl)

-- Locale = Complete Heyting algebra (same thing, different perspective)
Locale : (o ℓ : Level) → Type (lsuc (o ⊔ ℓ))
Locale o ℓ = CompleteHeytingAlgebra o ℓ

--------------------------------------------------------------------------------
-- §A.2: Ω-Sets (Fuzzy Sets)

{-|
## Definition: Ω-Set

An **Ω-set** (X, δ) is:
- Set X
- Fuzzy equality δ: X×X → Ω
- Symmetric: δ(x,y) = δ(y,x)
- Transitive: δ(x,y) ∧ δ(y,z) ≤ δ(x,z) (**Equation 18**)

**NOT required**:
- Reflexivity δ(x,x) = ⊤ (may be < ⊤!)
- This is the key difference from ordinary equality

**Properties** (from paper):
- δ(x,y) = δ(x,y) ∧ δ(y,x) ≤ δ(x,x)
- δ(x,y) ≤ δ(y,y)

**DNN Interpretation**:
- X = Set of possible outputs at a layer
- δ(x,y) = "Degree to which outputs x and y are equal"
- δ(x,x) < ⊤ = "Partial certainty about output x"
- Transitivity = "Equality is transitive even when fuzzy"

**Example: Progressive decision tree**
- Layer L has outputs {x₁, x₂, x₃}
- δ(x₁, x₂) = 0.8 (very similar)
- δ(x₂, x₃) = 0.7 (quite similar)
- δ(x₁, x₃) ≥ 0.8 ∧ 0.7 = 0.7 (transitivity)
- δ(x₁, x₁) might be 0.9 (not fully certain about x₁)
-}

record Ω-Set {o ℓ : Level} (Ω : CompleteHeytingAlgebra o ℓ) : Type (lsuc o ⊔ ℓ) where
  no-eta-equality
  constructor ω-set
  private module ΩA = CompleteHeytingAlgebra Ω

  field
    -- Underlying set (carrier at level o for small suprema)
    Carrier : Type o

    -- Fuzzy equality δ: X×X → Ω
    δ : Carrier → Carrier → ΩA.Ob

    -- Symmetry: δ(x,y) = δ(y,x)
    δ-sym : ∀ {x y} → δ x y ≡ δ y x

    -- Transitivity (Equation 18): δ(x,y) ∧ δ(y,z) ≤ δ(x,z)
    δ-trans : ∀ {x y z} → (ΩA._≤_) (δ x y ΩA.∩ δ y z) (δ x z)

    -- Reflexivity: δ(x,x) is maximal for x
    δ-refl : ∀ {x} → (ΩA._≤_) ΩA.top (δ x x)

  -- Derived properties (from paper)
  -- First show δ(x,y) ≤ δ(x,y) ∧ δ(y,x) using the greatest property
  δ-meet-bound : ∀ {x y} → (ΩA._≤_) (δ x y) ((ΩA._∩_) (δ x y) (δ y x))
  δ-meet-bound {x} {y} = ΩA.∩-meets (δ x y) (δ y x) .is-meet.greatest (δ x y) ΩA.≤-refl (subst (λ z → (ΩA._≤_) (δ x y) z) (sym δ-sym) ΩA.≤-refl)

  -- Then δ(x,y) ≤ δ(x,x) by transitivity
  δ-self-bound : ∀ {x y} → (ΩA._≤_) (δ x y) (δ x x)
  δ-self-bound {x} {y} = ΩA.≤-trans δ-meet-bound (δ-trans {x} {y} {x})

  δ-other-bound : ∀ {x y} → (ΩA._≤_) (δ x y) (δ y y)
  δ-other-bound {x} {y} = subst (λ z → (ΩA._≤_) z (δ y y)) δ-sym (δ-self-bound {y} {x})

open Ω-Set public

--------------------------------------------------------------------------------
-- §A.3: Morphisms in SetΩ (Equations 19-22)

{-|
## Morphisms of Ω-Sets

A morphism from (X,δ) to (X',δ') is a **fuzzy function**:
  f: X×X' → Ω

satisfying (Equations 19-22):

**Equation 19**: δ(x,y) ∧ f(x,x') ≤ f(y,x')
  "If x ≈ y and x maps to x', then y also maps to x'"

**Equation 20**: f(x,x') ∧ δ'(x',y') ≤ f(x,y')
  "If x maps to x' and x' ≈ y', then x also maps to y'"

**Equation 21**: f(x,x') ∧ f(x,y') ≤ δ'(x',y')
  "If x maps to both x' and y', then x' ≈ y'"
  (Single-valued: x can't map to two different outputs)

**Equation 22**: ⋁_{x'∈X'} f(x,x') = δ(x,x)
  "Total: x maps somewhere with certainty δ(x,x)"

**Generalization**: Boolean case (Ω = 2)
- f becomes characteristic function of graph
- Equations 19-21 ensure f is a function
- Equation 22 ensures totality
-}

-- Morphisms require carriers at Type o for eq-22 to work with frame suprema
record Ω-Set-Morphism {o ℓ : Level} {Ω : CompleteHeytingAlgebra o ℓ}
                       (X : Ω-Set Ω) (Y : Ω-Set Ω)
                       : Type (lsuc o ⊔ ℓ) where
  no-eta-equality
  constructor ω-morphism
  private
    module X = Ω-Set X
    module Y = Ω-Set Y
    module ΩA = CompleteHeytingAlgebra Ω

  field
    -- Fuzzy function f: X×Y → Ω
    f : X.Carrier → Y.Carrier → ΩA.Ob

    -- Equation 19: Respect source fuzzy equality
    eq-19 : ∀ {x y : X.Carrier} {x' : Y.Carrier}
          → (ΩA._≤_) (X.δ x y ΩA.∩ f x x') (f y x')

    -- Equation 20: Respect target fuzzy equality
    eq-20 : ∀ {x : X.Carrier} {x' y' : Y.Carrier}
          → (ΩA._≤_) (f x x' ΩA.∩ Y.δ x' y') (f x y')

    -- Equation 21: Single-valued (deterministic)
    eq-21 : ∀ {x : X.Carrier} {x' y' : Y.Carrier}
          → (ΩA._≤_) (f x x' ΩA.∩ f x y') (Y.δ x' y')

    -- Equation 22: Total (defined everywhere)
    eq-22 : ∀ {x : X.Carrier}
          → ΩA.⋃ (λ (x' : Y.Carrier) → f x x') ≡ X.δ x x

open Ω-Set-Morphism public

-- Automatically derive H-Level instances for records
unquoteDecl H-Level-Ω-Set-Morphism = declare-record-hlevel 2 H-Level-Ω-Set-Morphism (quote Ω-Set-Morphism)
