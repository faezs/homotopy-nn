{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Categorical Threshold Non-linearity (Section 6.2)

This module implements Proposition 6.1 from Manin & Marcolli (2024): the
categorical threshold functor that introduces non-linearity in Hopfield dynamics.

## Overview

The key idea is to use a **monoidal functor ρ : C → R** from a category C to
a resource category R with preordered monoid (R, +, ⪰, 0) to define a threshold:

**Threshold functor (·)₊ : Core(C) → Core(C)**:
  - (C)₊ = C if [ρ(C)] ⪰ 0
  - (C)₊ = 0 otherwise

This generalizes the classical threshold function max{0,·} to arbitrary
symmetric monoidal categories.

## Key Results (Proposition 6.1)

1. The threshold (·)₊ is an endofunctor of Core(C) (invertible morphisms only)
2. It extends to an endofunctor of categories of summing functors
3. It's NOT generally a monoidal functor

## Application

This threshold functor is composed with the Hopfield dynamics to implement
the non-linear activation (·)₊ in the categorical equation (6.5):

  Xₑ(n+1) = Xₑ(n) ⊕ (⊕ₑ' Tₑₑ'(Xₑ'(n)) ⊕ Θₑ)₊
-}

module Neural.Dynamics.Hopfield.Threshold where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Monoidal.Base
open import Cat.Monoidal.Braided
open import Cat.Monoidal.Functor
open import Cat.Instances.Core
open import Cat.Instances.Product

open import Data.Nat.Base using (Nat)
open import Data.Fin.Base using (Fin)

open import Algebra.Monoid
open import Order.Base

open import Neural.Information public
  using (ℝ; _+ℝ_; _*ℝ_; _≤ℝ_; _≥ℝ_; zeroℝ; oneℝ)
open import Neural.Base

private variable
  o ℓ o' ℓ' : Level

{-|
## Preordered Monoid

A **preordered monoid** (M, ⊕, ⪰, unit) consists of:
- Monoid structure (M, ⊕, unit) from 1Lab
- Preorder ⪰ on M

This abstracts the (R, +, ⪰, 0) structure from symmetric monoidal resource categories.

We use 1Lab's `Monoid-on` for the monoid structure.
-}

record PreorderedMonoid (o ℓ : Level) : Type (lsuc (o ⊔ ℓ)) where
  no-eta-equality
  field
    carrier : Type o
    has-monoid : Monoid-on carrier
    _⪰_ : carrier → carrier → Type ℓ

    -- Preorder properties
    ⪰-refl : ∀ {x} → x ⪰ x
    ⪰-trans : ∀ {x y z} → x ⪰ y → y ⪰ z → x ⪰ z
    ⪰-prop : ∀ {x y} → is-prop (x ⪰ y)

  open Monoid-on has-monoid public

open PreorderedMonoid public

{-|
## Extract Preordered Monoid from Symmetric Monoidal Category

Given symmetric monoidal category (C, ⊗, I), we extract:
- Carrier = Ob(C)
- + = ⊗ (tensor product)
- zero = I (unit object)
- ⪰ = reversed morphism existence: A ⪰ B iff ∃ morphism B → A

Note: The preorder is REVERSED because in resource categories, a morphism
B → A means "A is at least as resourceful as B" or "A ⪰ B".
-}

-- Note: This only works properly for thin/preorder categories
-- For general symmetric monoidal categories, we'd need additional structure
postulate
  monoidal→preordered-monoid :
    {C : Precategory o ℓ} →
    (Cᵐ : Monoidal-category C) →
    (Cˢ : Symmetric-monoidal Cᵐ) →
    PreorderedMonoid o ℓ

{-|
## Threshold Structure (Section 6.2)

A **threshold structure** consists of:
1. Category C (with monoidal structure, written additively as ⊕)
2. Resource category R (with monoidal structure)
3. Monoidal functor ρ : C → R
4. Preordered monoid extracted from R

The threshold condition [ρ(C)] ⪰ 0 determines whether C passes the threshold.
-}

record ThresholdStructure (o ℓ o' ℓ' : Level) : Type (lsuc (o ⊔ ℓ ⊔ o' ⊔ ℓ')) where
  no-eta-equality
  field
    -- Category C
    C : Precategory o ℓ
    C-monoidal : Monoidal-category C
    C-symmetric : Symmetric-monoidal C-monoidal

    -- Resource category R
    R : Precategory o' ℓ'
    R-monoidal : Monoidal-category R
    R-symmetric : Symmetric-monoidal R-monoidal

    -- Monoidal functor ρ : C → R
    ρ : Functor C R
    ρ-monoidal : Monoidal-functor-on C-monoidal R-monoidal ρ

  -- Extract preordered monoid from R
  R-preordered : PreorderedMonoid o' ℓ'
  R-preordered = monoidal→preordered-monoid R-monoidal R-symmetric

  -- Threshold predicate: [ρ(C)] ⪰ 0 in R
  -- This is: ∃ morphism from Unit_R to ρ(C)
  threshold-satisfied : C .Precategory.Ob → Type ℓ'
  threshold-satisfied C-obj = R .Precategory.Hom (R-monoidal .Monoidal-category.Unit) (ρ .Functor.F₀ C-obj)

open ThresholdStructure public

{-|
## Threshold Functor (Proposition 6.1)

The **threshold endofunctor** (·)₊ : Core(C) → Core(C) acts as:
- (C)₊ = C if threshold-satisfied C (i.e., [ρ(C)] ⪰ 0)
- (C)₊ = 0 (unit object) otherwise

On morphisms (which are isomorphisms in Core(C)):
- If both domain and codomain satisfy threshold: identity
- Otherwise: map to identity on 0

**Key insight**: This only works on Core(C) because we need isomorphisms,
not general morphisms. The threshold might send non-iso morphisms to zero,
breaking functoriality.
-}

postulate
  threshold-functor :
    {o ℓ o' ℓ' : Level} →
    (ts : ThresholdStructure o ℓ o' ℓ') →
    Functor (Core (ts .C)) (Core (ts .C))

{-|
The threshold functor satisfies:

1. **On objects**: (C)₊ = C if [ρ(C)] ⪰ 0, else 0
2. **On morphisms**: Preserves isomorphisms or maps to id₀
3. **Functoriality**: Preserves identities and composition
-}

postulate
  threshold-on-objects :
    (ts : ThresholdStructure o ℓ o' ℓ') →
    (C-obj : ts .C .Precategory.Ob) →
    (ThresholdStructure.threshold-satisfied ts C-obj) →
    threshold-functor ts .Functor.F₀ C-obj ≡ C-obj

  threshold-on-zero :
    (ts : ThresholdStructure o ℓ o' ℓ') →
    (C-obj : ts .C .Precategory.Ob) →
    ¬ (ThresholdStructure.threshold-satisfied ts C-obj) →
    threshold-functor ts .Functor.F₀ C-obj ≡ (ts .C-monoidal .Monoidal-category.Unit)

{-|
## Extension to Summing Functors (Proposition 6.1)

The threshold functor extends to an endofunctor of categories of summing functors:

  ΣC(X) → ΣC(X)

by applying the threshold pointwise: (Φ)₊(x) = (Φ(x))₊ for all x ∈ X.
-}

postulate
  -- TODO: Define summing functor category properly
  SummingFunctorCat : (C : Precategory o ℓ) → (X : Type) → Precategory _ _

  threshold-summing-functor :
    {ts : ThresholdStructure o ℓ o' ℓ'} →
    (X : Type) →
    Functor (SummingFunctorCat (ts .C) X) (SummingFunctorCat (ts .C) X)

{-|
## Properties and Remarks

**Not monoidal**: The threshold functor (·)₊ is generally NOT a monoidal functor.
That is, (C ⊕ C')₊ ≠ C₊ ⊕ C'₊ in general.

This is expected: the threshold is a non-linear operation, so it doesn't
preserve the monoidal structure.

**Core is essential**: The functor only works on Core(C) because arbitrary
morphisms might be sent to zero, breaking composition. Isomorphisms are
preserved or both sent to id₀.
-}
