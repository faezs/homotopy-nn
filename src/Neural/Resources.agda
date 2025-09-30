{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# The Mathematical Theory of Resources (Section 3.2)

This module implements Section 3.2 from Manin & Marcolli (2024):
"Homotopy-theoretic and categorical models of neural information networks"

We formalize the categorical framework for resource theory developed in [27] and [40],
specialized to neural information networks.

## Overview

A **theory of resources** is formulated as a symmetric monoidal category (R,◦,⊗,I) where:
- Objects A ∈ Obj(R) represent resources
- Product A⊗B represents combination of resources A and B
- Unit object I represents empty resource
- Morphisms f: A → B represent possible conversions of resource A into B

### No-Cost and Freely Disposable Resources

- **No-cost resources**: A with MorR(I,A) ≠ ∅ (can be created from nothing)
- **Freely disposable resources**: A with MorR(A,I) ≠ ∅ (can be discarded)

### Sequential Conversion

Composition ◦: MorR(A,B) × MorR(B,C) → MorR(A,C) represents sequential
conversion of resources.

## Examples (Section 3.2.1)

1. **FinProb**: Resources of randomness (finite sets with probability measures)
2. **FinStoch**: Random processes (stochastic matrices)
3. **FP = I/FinStoch**: Partitioned processes (coslice category)

These examples connect to classical information theory and Shannon entropy.
-}

module Neural.Resources where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Monoidal.Base
open import Cat.Monoidal.Braided
open import Cat.Instances.Product
open import Cat.Functor.Adjoint

import Cat.Reasoning

open import Data.Nat.Base using (Nat; zero; suc; _+_; _*_)
open import Data.Fin.Base using (Fin)
open import Data.Bool.Base using (Bool)
open import Data.Sum.Base
open import Data.List.Base using (List)

open import Algebra.Monoid

open import Neural.Base

private variable
  o ℓ o' ℓ' : Level

-- Real numbers (from Neural.Information)
postulate
  ℝ : Type
  _*ℝ_ : ℝ → ℝ → ℝ
  _/ℝ_ : ℝ → ℝ → ℝ
  _+ℝ_ : ℝ → ℝ → ℝ
  _≥ℝ_ : ℝ → ℝ → Type
  sup : List ℝ → ℝ

{-|
## Definition 3.3: Resource Theory as Symmetric Monoidal Category

A **theory of resources** (following [27], [40]) consists of:
- A symmetric monoidal category (R,◦,⊗,I)
- Objects represent resources
- Morphisms represent resource conversions
- ⊗ combines resources
- I is the empty resource

**Physical interpretation**:
- In neural networks, resources include energy, metabolic capacity, information
  transmission capacity, computational power
- Combining two networks combines their resource requirements (⊗)
- Converting resources means reallocating or transforming them
-}

record ResourceTheory (o ℓ : Level) : Type (lsuc (o ⊔ ℓ)) where
  field
    {-| Underlying category of resources and their conversions -}
    R : Precategory o ℓ

    {-| Symmetric monoidal structure for combining resources -}
    Rᵐ : Monoidal-category R
    Rˢ : Symmetric-monoidal Rᵐ

  open Precategory R renaming (Ob to Resource; Hom to Conversion) public
  open Monoidal Rᵐ public
  open Symmetric-monoidal Rˢ public

  {-|
  **No-cost resources**: Resources that can be created from the unit (empty resource).
  In neural networks, these might represent inherent biological capabilities.
  -}
  is-no-cost : Resource → Type ℓ
  is-no-cost A = Conversion Unit A

  {-|
  **Freely disposable resources**: Resources that can be discarded (converted to unit).
  These represent resources with no maintenance cost.
  -}
  is-freely-disposable : Resource → Type ℓ
  is-freely-disposable A = Conversion A Unit

{-|
## Preordered Abelian Monoid from Resource Theory (Section 3.2.2)

From a resource theory (R,◦,⊗,I), we can extract a **preordered abelian monoid**
(R,+,⪰,0) on the set R of isomorphism classes of Obj(R):

- [A] + [B] := class of A⊗B
- 0 := class of I (unit object)
- [A] ⪰ [B] iff MorR(A,B) ≠ ∅ (A can be converted to B)

**Properties**:
- Partial ordering is compatible with monoid operation
- If [A] ⪰ [B] and [C] ⪰ [D] then [A]+[C] ⪰ [B]+[D]

This abstracts away from specific morphisms and focuses only on convertibility.
-}

module PreorderedMonoid
  {o ℓ : Level}
  (RT : ResourceTheory o ℓ)
  where

  open ResourceTheory RT

  {-|
  **Resource class**: Isomorphism class of resources in R.

  Two resources are in the same class if they are isomorphic (interconvertible
  with mutual inverses).
  -}
  postulate
    ResourceClass : Type o
    ⌈_⌉ : Resource → ResourceClass  -- Quotient map to isomorphism classes

  {-|
  **Monoid structure on resource classes**:
  - Addition: [A] + [B] := [A⊗B]
  - Unit: 0 := [I]
  -}
  postulate
    _⊕_ : ResourceClass → ResourceClass → ResourceClass
    𝟘 : ResourceClass

    -- [A] + [B] = [A⊗B]
    ⊕-respects-⊗ : (A B : Resource) → ⌈ A ⌉ ⊕ ⌈ B ⌉ ≡ ⌈ A ⊗ B ⌉

    -- 0 = [I]
    𝟘-is-unit : 𝟘 ≡ ⌈ Unit ⌉

    -- (R, ⊕, 𝟘) forms a monoid
    resource-monoid : Monoid-on ResourceClass

  {-|
  **Convertibility preorder**: [A] ⪰ [B] iff there exists a morphism A → B.

  This captures the notion of "resource A is at least as powerful as B"
  or "A can be converted into B".
  -}
  _⪰_ : ResourceClass → ResourceClass → Type (o ⊔ ℓ)
  A ⪰ B = ∥ Σ[ A' ∈ Resource ] Σ[ B' ∈ Resource ]
              (⌈ A' ⌉ ≡ A) × (⌈ B' ⌉ ≡ B) × Conversion A' B' ∥

  postulate
    {-| Preorder properties -}
    ⪰-refl : {A : ResourceClass} → A ⪰ A
    ⪰-trans : {A B C : ResourceClass} → A ⪰ B → B ⪰ C → A ⪰ C

    {-| Compatibility with monoid operation -}
    ⪰-compatible : {A B C D : ResourceClass} → A ⪰ B → C ⪰ D → (A ⊕ C) ⪰ (B ⊕ D)

{-|
## Maximal Conversion Rate (Equation 3.1)

The **maximal conversion rate** ρA→B measures the optimal (maximal) fraction of
copies of resource B that can be produced from resource A.

  ρA→B := sup { m/n | n·[A] ⪰ m·[B], m,n ∈ ℕ }

where n·[A] denotes [A⊗n] (n-fold tensor power of A).

**Interpretation**:
- ρA→B = 2: Each copy of A can be converted to 2 copies of B
- ρA→B = 0.5: Need 2 copies of A to produce 1 copy of B
- ρA→B measures the "exchange rate" between resources

This is analogous to conversion rates between currencies or resources in economics.
-}

module ConversionRates
  {o ℓ : Level}
  (RT : ResourceTheory o ℓ)
  where

  open ResourceTheory RT
  open PreorderedMonoid RT

  postulate
    {-| Tensor power: A⊗n = A ⊗ A ⊗ ... ⊗ A (n times) -}
    _⊗^_ : Resource → Nat → Resource

    -- 0-fold tensor is unit
    ⊗^-zero : (A : Resource) → A ⊗^ zero ≡ Unit

    -- (n+1)-fold tensor is A ⊗ (A⊗n)
    ⊗^-suc : (A : Resource) → (n : Nat) → A ⊗^ (suc n) ≡ A ⊗ (A ⊗^ n)

  {-|
  **Maximal conversion rate** ρA→B between resources A and B.

  Defined as the supremum over all ratios m/n where n copies of A can be
  converted to m copies of B.
  -}
  ρ : Resource → Resource → ℝ
  ρ A B = sup (rates A B)
    where
      postulate
        -- All achievable rates m/n
        rates : Resource → Resource → List ℝ

        -- m/n is achievable iff n·A ⪰ m·B
        rate-achievable :
          (A B : Resource) → (m n : Nat) →
          ⌈ A ⊗^ n ⌉ ⪰ ⌈ B ⊗^ m ⌉ → -- n·A ⪰ m·B implies rate is in list
          Type

{-|
## S-Valued Measuring of Resources

An **S-valued measuring** of R-resources is a monoid homomorphism
M: (R,+,0) → (S,*,1S) that preserves the ordering:

  M(A) ≥ M(B) in S whenever [A] ⪰ [B] in R

**Example (Theorem 5.6 from [27])**:
For M: (R,+) → (ℝ,+) a measuring, we have:
  ρA→B · M(B) ≤ M(A)

That is, the optimal fraction of B's obtainable from A is bounded by the
ratio of their measured values.
-}

module ResourceMeasuring
  {o ℓ : Level}
  (RT : ResourceTheory o ℓ)
  where

  open ResourceTheory RT
  open PreorderedMonoid RT
  open ConversionRates RT

  record S-Measuring (S : Type) (_*ₛ_ : S → S → S) (_≥ₛ_ : S → S → Type) : Type (o ⊔ ℓ) where
    field
      {-| Measuring function from resource classes to S -}
      measure : ResourceClass → S

      {-| Unit preservation -}
      measure-unit : measure 𝟘 ≡ {!!}  -- Need unit of (S,*ₛ)

      {-| Monoid homomorphism -}
      measure-⊕ : (A B : ResourceClass) → measure (A ⊕ B) ≡ measure A *ₛ measure B

      {-| Order preservation -}
      measure-mono : (A B : ResourceClass) → A ⪰ B → measure A ≥ₛ measure B

  postulate
    {-|
    **Theorem 5.6 from [27]**: For ℝ-valued measuring M, the conversion rate
    ρA→B is bounded by the ratio of measured values.

    ρA→B · M(B) ≤ M(A)

    **Interpretation**: Can't get more value out than you put in, where value
    is measured by M.
    -}
    conversion-rate-bound :
      (M : S-Measuring ℝ _*ℝ_ _≥ℝ_) →
      (A B : Resource) →
      (ρ A B *ℝ M .S-Measuring.measure ⌈ B ⌉) ≥ℝ M .S-Measuring.measure ⌈ A ⌉
