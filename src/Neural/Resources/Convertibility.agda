{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Convertibility of Resources (Section 3.2.2)

This module provides detailed treatment of resource convertibility and conversion rates
from Section 3.2.2 of Manin & Marcolli (2024).

## Overview

From a symmetric monoidal category (R,◦,⊗,I) of resources, we derive:
1. **Preordered monoid** (R,+,⪰,0) on isomorphism classes
2. **Convertibility relation**: [A] ⪰ [B] iff MorR(A,B) ≠ ∅
3. **Conversion rates**: ρA→B measuring optimal exchange ratios
4. **Measuring homomorphisms**: M: (R,+) → (S,∗) preserving order
5. **Bounds on conversion**: Theorem 5.6 relating rates to measurements

## Key Results

**Theorem 5.6 (from [27])**:
For M: (R,+) → (ℝ,+) a measuring monoid homomorphism:
  ρA→B · M(B) ≤ M(A)

**Interpretation**: The optimal number of B's obtainable from A is bounded
by the ratio of their measured values. You can't get more value out than you put in.

## Applications to Neural Networks

In neural information networks:
- **Resources**: Energy, information capacity, metabolic capacity
- **Convertibility**: Can we trade energy for information capacity?
- **Conversion rates**: How much energy per bit of information?
- **Measuring**: Assign numerical values to resources (entropy, joules, etc.)
- **Bounds**: Physical limits on information-energy tradeoffs
-}

module Neural.Resources.Convertibility where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Monoidal.Base
open import Cat.Monoidal.Braided

import Cat.Reasoning

open import Data.Nat.Base using (Nat; zero; suc; _+_; _*_)
open import Data.Fin.Base using (Fin)
open import Data.Sum.Base
open import Data.List.Base using (List; []; _∷_)

open import Algebra.Monoid

open import Neural.Base
open import Neural.Resources

private variable
  o ℓ : Level

-- Real numbers and operations (imported from Neural.Resources)
open Neural.Resources public using (ℝ; _*ℝ_; _/ℝ_; _+ℝ_; _≥ℝ_; sup)

postulate
  _≤ℝ_ : ℝ → ℝ → Type

{-|
## Convertibility Preorder Properties

The convertibility relation [A] ⪰ [B] forms a preorder (reflexive and transitive)
that is compatible with the monoid operation ⊕.

**Reflexivity**: Every resource can be converted to itself (via identity morphism)
**Transitivity**: If A converts to B and B converts to C, then A converts to C (composition)
**Compatibility**: If A⪰B and C⪰D, then A⊗C ⪰ B⊗D (tensor product of conversions)
-}

module ConvertibilityProperties
  {o ℓ : Level}
  (RT : ResourceTheory o ℓ)
  where

  open ResourceTheory RT
  open PreorderedMonoid RT

  {-|
  **Reflexivity of convertibility**: [A] ⪰ [A] via identity morphism id: A → A
  -}
  postulate
    convertibility-refl : (A : ResourceClass) → A ⪰ A

    -- Witness: For any representative A' of class A, id: A' → A' witnesses A ⪰ A
    convertibility-refl-witness :
      (A' : Resource) →
      ∥ Σ[ B' ∈ Resource ] (⌈ A' ⌉ ≡ ⌈ B' ⌉) × Conversion A' B' ∥

  {-|
  **Transitivity of convertibility**: [A] ⪰ [B] and [B] ⪰ [C] implies [A] ⪰ [C]
  via composition of morphisms.
  -}
  postulate
    convertibility-trans :
      (A B C : ResourceClass) →
      A ⪰ B → B ⪰ C → A ⪰ C

    -- Witness: Composition f ∘ g witnesses transitivity
    convertibility-trans-witness :
      {A' B' C' : Resource} →
      (f : Conversion A' B') →
      (g : Conversion B' C') →
      Conversion A' C'

  {-|
  **Compatibility with monoidal product**: The preorder respects ⊗.

  If A can be converted to B and C can be converted to D, then A⊗C can be
  converted to B⊗D via the tensor product of the two conversions.
  -}
  postulate
    convertibility-compatible :
      (A B C D : ResourceClass) →
      A ⪰ B → C ⪰ D → (A ⊕ C) ⪰ (B ⊕ D)

    -- Witness: f ⊗ g : A⊗C → B⊗D from f: A → B and g: C → D
    convertibility-compatible-witness :
      {A' B' C' D' : Resource} →
      (f : Conversion A' B') →
      (g : Conversion C' D') →
      Conversion (A' ⊗ C') (B' ⊗ D')

{-|
## Tensor Powers and Bulk Conversion

To define conversion rates, we need the notion of "n copies of resource A",
represented as A⊗n (n-fold tensor power).

**Examples**:
- A⊗0 = I (unit/empty resource)
- A⊗1 = A
- A⊗2 = A⊗A
- A⊗3 = A⊗A⊗A

**Conversion rate interpretation**: If A⊗2 ⪰ B⊗5, then 2 copies of A can produce
5 copies of B, giving conversion rate ρA→B ≥ 5/2 = 2.5.
-}

module TensorPowers
  {o ℓ : Level}
  (RT : ResourceTheory o ℓ)
  where

  open ResourceTheory RT
  open PreorderedMonoid RT

  private module R = Cat.Reasoning R

  {-| n-fold tensor power of a resource -}
  postulate
    _^⊗_ : Resource → Nat → Resource

  -- Base case: A⊗0 = I
  postulate
    ^⊗-zero : (A : Resource) → A ^⊗ zero ≡ Unit

  -- Recursive case: A⊗(n+1) = A ⊗ (A⊗n)
  postulate
    ^⊗-suc : (A : Resource) → (n : Nat) → A ^⊗ (suc n) ≡ A ⊗ (A ^⊗ n)

  {-|
  **Tensor power laws**: Standard algebraic properties
  -}
  postulate
    -- A⊗(m+n) ≅ (A⊗m) ⊗ (A⊗n)
    ^⊗-+ : (A : Resource) → (m n : Nat) →
           (A ^⊗ (m + n)) R.≅ ((A ^⊗ m) ⊗ (A ^⊗ n))

    -- A⊗(m*n) ≅ (A⊗m)⊗n
    ^⊗-* : (A : Resource) → (m n : Nat) →
           (A ^⊗ (m * n)) R.≅ ((A ^⊗ m) ^⊗ n)

  {-|
  **Bulk convertibility**: If A⪰B, then A⊗n ⪰ B⊗n for all n.

  Having more copies preserves convertibility.
  -}
  postulate
    bulk-convertibility :
      {A B : Resource} →
      (n : Nat) →
      ⌈ A ⌉ ⪰ ⌈ B ⌉ →
      ⌈ A ^⊗ n ⌉ ⪰ ⌈ B ^⊗ n ⌉

{-|
## Conversion Rates in Detail

The **maximal conversion rate** ρA→B is defined as:

  ρA→B := sup { m/n | n·[A] ⪰ m·[B], m,n ∈ ℕ }

**Example 1**: If A⊗1 ⪰ B⊗2, then ρA→B ≥ 2/1 = 2
  (One unit of A produces at least 2 units of B)

**Example 2**: If A⊗3 ⪰ B⊗5, then ρA→B ≥ 5/3 ≈ 1.67
  (Three units of A produce at least 5 units of B)

**Supremum**: ρA→B takes the maximum over all such achievable ratios.

**Special cases**:
- ρA→A = 1 (identity conversion)
- ρA→B = 0 if A cannot produce any B
- ρA→B = ∞ if A can produce arbitrarily many B's (e.g., if B is freely disposable)
-}

module ConversionRateCalculation
  {o ℓ : Level}
  (RT : ResourceTheory o ℓ)
  where

  open ResourceTheory RT
  open PreorderedMonoid RT
  open TensorPowers RT

  postulate
    -- Real number operations
    nat-to-ℝ : Nat → ℝ
    ∞ : ℝ

  {-|
  **Achievable conversion ratios**: All ratios m/n where n copies of A
  can be converted to m copies of B.
  -}
  achievable-ratio : Resource → Resource → Nat → Nat → Type (o ⊔ ℓ)
  achievable-ratio A B m n = ⌈ A ^⊗ n ⌉ ⪰ ⌈ B ^⊗ m ⌉

  {-|
  **Conversion rate** ρA→B as supremum of achievable ratios.
  -}
  postulate
    ρ : Resource → Resource → ℝ

    -- ρA→B is the supremum of all achievable m/n
    ρ-is-supremum :
      (A B : Resource) →
      (m n : Nat) →
      achievable-ratio A B m n →
      (nat-to-ℝ m /ℝ nat-to-ℝ n) ≤ℝ ρ A B

  {-|
  **Properties of conversion rates**
  -}
  postulate
    -- Identity: ρA→A = 1
    ρ-identity : (A : Resource) → ρ A A ≡ nat-to-ℝ 1

    -- Transitivity: ρA→C ≥ ρA→B * ρB→C
    -- (Converting A→B→C is at least as efficient as direct A→C)
    ρ-transitive :
      (A B C : Resource) →
      (ρ A B *ℝ ρ B C) ≤ℝ ρ A C

    -- Monotonicity: If A⪰A' and B'⪰B, then ρA→B ≥ ρA'→B'
    ρ-monotone :
      {A A' B B' : Resource} →
      ⌈ A ⌉ ⪰ ⌈ A' ⌉ →
      ⌈ B' ⌉ ⪰ ⌈ B ⌉ →
      ρ A' B' ≤ℝ ρ A B

  {-|
  **Examples of conversion rate calculations**
  -}
  module Examples where
    postulate
      -- Example resources
      Energy : Resource      -- Metabolic energy
      Information : Resource -- Information capacity (bits)
      Computation : Resource -- Computational cycles

      -- Example conversions
      energy-to-info : Conversion Energy Information
      info-to-comp : Conversion Information Computation

      -- Example rates
      -- "1 joule of energy produces 10^6 bits of information capacity"
      example-rate-1 :
        achievable-ratio Energy Information (suc (suc (suc (suc (suc (suc zero)))))) (suc zero)

      -- "100 bits of information require 1 computation cycle"
      example-rate-2 :
        achievable-ratio Information Computation (suc zero) (suc (suc zero))

{-|
## Measuring Monoid Homomorphisms

An **S-valued measuring** assigns a numerical value to each resource class
while preserving the monoidal and order structure:

  M: (R,⊕,𝟘,⪰) → (S,∗,1S,≥S)

**Properties**:
1. M(𝟘) = 1S (unit preservation)
2. M(A ⊕ B) = M(A) ∗ M(B) (homomorphism)
3. A ⪰ B implies M(A) ≥S M(B) (order preservation)

**Examples**:
- Shannon entropy: M([probability distribution]) = H(P)
- Energy measurement: M([resource]) = energy in joules
- Information capacity: M([resource]) = bits of information
-}

module MeasuringHomomorphisms
  {o ℓ : Level}
  (RT : ResourceTheory o ℓ)
  where

  open ResourceTheory RT
  open PreorderedMonoid RT
  open TensorPowers RT
  open ConversionRateCalculation RT

  {-|
  **Generic S-valued measuring** for an ordered monoid (S,∗,1S,≥S)
  -}
  record Measuring
    (S : Type)                    -- Target monoid
    (_∗_ : S → S → S)             -- Monoid operation
    (1S : S)                       -- Unit
    (_≥S_ : S → S → Type)         -- Order relation
    : Type (o ⊔ ℓ) where
    field
      {-| Measuring function -}
      measure : ResourceClass → S

      {-| Unit preservation: M(𝟘) = 1S -}
      measure-unit : measure 𝟘 ≡ 1S

      {-| Homomorphism: M(A ⊕ B) = M(A) ∗ M(B) -}
      measure-⊕ : (A B : ResourceClass) →
                   measure (A ⊕ B) ≡ measure A ∗ measure B

      {-| Order preservation: A ⪰ B → M(A) ≥S M(B) -}
      measure-monotone : (A B : ResourceClass) →
                          A ⪰ B →
                          measure A ≥S measure B

  {-|
  **Real-valued measuring**: The most common case, measuring resources
  as real numbers with addition.
  -}
  ℝ-Measuring : Type (o ⊔ ℓ)
  ℝ-Measuring = Measuring ℝ _+ℝ_ (nat-to-ℝ 0) _≥ℝ_

  {-|
  **Theorem 5.6 from [27]**: Conversion Rate Bound

  For an ℝ-valued measuring M, the conversion rate ρA→B satisfies:

    ρA→B · M(B) ≤ M(A)

  **Proof idea**: If you could convert A to more than M(A)/M(B) copies of B,
  the total measured value would increase, violating order preservation.

  **Interpretation**: The exchange rate is bounded by the value ratio.
  You can't create value from nothing.
  -}
  module Theorem5∙6 (M : ℝ-Measuring) where
    open Measuring M

    postulate
      conversion-rate-bound :
        (A B : Resource) →
        (ρ A B *ℝ measure ⌈ B ⌉) ≤ℝ measure ⌈ A ⌉

      {-|
      **Corollary**: If ρA→B · M(B) = M(A), the conversion is "optimal" in the
      sense that it achieves the theoretical maximum.
      -}
      optimal-conversion :
        (A B : Resource) →
        (ρ A B *ℝ measure ⌈ B ⌉ ≡ measure ⌈ A ⌉) →
        -- A can be converted to ρA→B copies of B without loss
        Type (o ⊔ ℓ)

{-|
## Examples: Measuring Information and Energy

Concrete examples of measuring homomorphisms relevant to neural networks.
-}

module ConcreteExamples where
  postulate
    {-| **Shannon entropy measuring**: For probability distributions -}
    EntropyMeasure : Type

    -- H(P ⊗ Q) = H(P) + H(Q) for independent distributions
    entropy-is-additive : EntropyMeasure

    -- H(P) ≥ H(Q) if P can be converted to Q (data processing inequality)
    entropy-decreases : EntropyMeasure

    {-| **Energy measuring**: For metabolic resources -}
    EnergyMeasure : Type

    -- E(A ⊗ B) = E(A) + E(B) for independent systems
    energy-is-additive : EnergyMeasure

    -- E(A) ≥ E(B) if A can be converted to B (thermodynamics)
    energy-decreases : EnergyMeasure

    {-|
    **Landauer's principle**: Converting 1 bit of information to heat requires
    at least k_B T ln(2) joules of energy, where k_B is Boltzmann's constant
    and T is temperature.

    This gives a fundamental bound on information-energy conversion rates.
    -}
    landauer-bound : Type
    landauer-constant : ℝ  -- k_B T ln(2)
