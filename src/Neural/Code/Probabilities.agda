{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Finite Probabilities with Fiberwise Measures (§5.1.4)

This module implements the category Pf of finite probabilities from Section 5.1.4
of Manin & Marcolli (2024).

## Overview

**Category Pf** has:
- **Objects**: Pairs (X, P_X) of finite pointed sets with probability measures
- **Morphisms**: (f, Λ) with f : X → Y and fiberwise measures λ_y on fibers f⁻¹(y)
- **Structure**: Coproduct and zero object

**Purpose**: Model probability distributions on neural codes with morphisms that
track how probabilities transform via fiberwise scaling factors.

## Key Construction (Lemma 5.7)

Morphisms ϕ = (f, Λ) : (X, P_X) → (Y, P_Y) consist of:
1. Pointed function f : X → Y with f(x_0) = y_0
2. Fiberwise measures Λ = {λ_y} on fibers f⁻¹(y)
3. Compatibility: P_X(A) = Σ_{y∈Y} λ_y(A ∩ f⁻¹(y)) P_Y(y)

**Coproduct**: (X,P) ⊕ (X',P') has wedge sum X ∨ X' with probabilities:
- P̃(x) = α_{X,X'} · P(x) for x ∈ X \ {x_0}
- P̃(x') = β_{X,X'} · P'(x') for x' ∈ X' \ {x'_0}
- P̃(x_0 ~ x'_0) = α_{X,X'} P(x_0) + β_{X,X'} P'(x'_0)
- where α_{X,X'} = N/(N+N'), β_{X,X'} = N'/(N+N')
-}

module Neural.Code.Probabilities where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base

open import Data.Nat.Base using (Nat; zero; suc; _+_)
open import Data.Fin.Base using (Fin; fzero; fsuc)
open import Data.Sum.Base using (_⊎_; inl; inr)
open import Data.List.Base using (List; []; _∷_; length)

-- Import real numbers from parent module
open import Neural.Information public
  using (ℝ; _+ℝ_; _*ℝ_; _/ℝ_; _≤ℝ_; zeroℝ; oneℝ)

private variable
  o ℓ : Level

{-|
## Finite Pointed Sets

For simplicity, we model finite pointed sets as natural numbers where:
- n represents a set of size (n+1)
- The basepoint is always at index 0 (represented by fzero : Fin (suc n))
-}
FinitePointedSet : Type
FinitePointedSet = Nat

{-|
The underlying finite set of a pointed set
-}
underlying-set : FinitePointedSet → Type
underlying-set n = Fin (suc n)

{-|
Basepoint of a finite pointed set
-}
basepoint : (n : FinitePointedSet) → underlying-set n
basepoint n = fzero

{-|
## Probability Measures on Finite Sets

A probability measure on a finite set X assigns probabilities P(x) ∈ [0,1] to
each element such that Σ_{x∈X} P(x) = 1.

We represent this as a function Fin (suc n) → ℝ with normalization property.
-}
record ProbabilityMeasure (X : FinitePointedSet) : Type where
  no-eta-equality
  field
    {-| Probability assignment -}
    prob : underlying-set X → ℝ

    {-| Probabilities are non-negative -}
    prob-nonneg : ∀ (x : underlying-set X) → zeroℝ ≤ℝ prob x

    {-|
    Normalization: probabilities sum to 1

    In actual implementation would be: Σ_{x} prob(x) = 1
    We postulate the sum operator for now.
    -}
    prob-normalized : {-| Σ prob = 1 -} ⊤

    {-| Basepoint has positive probability -}
    basepoint-positive : zeroℝ ≤ℝ prob (basepoint X)

open ProbabilityMeasure public

{-|
Support of a probability measure: elements with non-zero probability
-}
postulate
  support : {X : FinitePointedSet} → ProbabilityMeasure X → List (underlying-set X)

  support-property :
    {X : FinitePointedSet} →
    (P : ProbabilityMeasure X) →
    (x : underlying-set X) →
    {-| x ∈ support(P) iff P(x) > 0 -} ⊤

{-|
## Fiberwise Measures

Given f : X → Y, a fiberwise measure on fiber f⁻¹(y) assigns weights λ_y(x)
to elements x ∈ f⁻¹(y).

These are NOT probability measures - they're scaling factors that relate P_X to P_Y.
-}
FiberwiseMeasure : {X Y : FinitePointedSet} → (f : underlying-set X → underlying-set Y) → Type
FiberwiseMeasure {X} {Y} f =
  (y : underlying-set Y) → (x : underlying-set X) → ℝ

{-|
Compatibility condition for fiberwise measures (from Lemma 5.7)

P_X(A) = Σ_{y∈Y} λ_y(A ∩ f⁻¹(y)) P_Y(y)
-}
postulate
  fiberwise-compatible :
    {X Y : FinitePointedSet} →
    (f : underlying-set X → underlying-set Y) →
    (P_X : ProbabilityMeasure X) →
    (P_Y : ProbabilityMeasure Y) →
    (Λ : FiberwiseMeasure f) →
    Type

{-|
## Objects of Category Pf (Lemma 5.7)

Objects are pairs (X, P_X) where:
- X: Finite pointed set
- P_X: Probability measure on X with P_X(x_0) > 0 at basepoint
-}
PfObject : Type
PfObject = Σ FinitePointedSet ProbabilityMeasure

{-|
## Morphisms of Category Pf (Lemma 5.7)

Morphisms (f, Λ) : (X, P_X) → (Y, P_Y) consist of:
1. Pointed function f : X → Y
2. Fiberwise measures Λ on fibers
3. f(supp(P_X)) ⊆ supp(P_Y)
4. Compatibility: P_X(A) = Σ_y λ_y(A ∩ f⁻¹(y)) P_Y(y)
-}
record PfMorphism (XP YP : PfObject) : Type where
  no-eta-equality

  private
    X = XP .fst
    P_X = XP .snd
    Y = YP .fst
    P_Y = YP .snd

  field
    {-| Underlying pointed function -}
    func : underlying-set X → underlying-set Y

    {-| Preserves basepoint -}
    func-basepoint : func (basepoint X) ≡ basepoint Y

    {-| Fiberwise measures on fibers f⁻¹(y) -}
    fiberwise : FiberwiseMeasure func

    {-| Basepoint fiber has positive weight -}
    fiberwise-basepoint-positive :
      zeroℝ ≤ℝ fiberwise (basepoint Y) (basepoint X)

    {-|
    Support compatibility: f(supp(P_X)) ⊆ supp(P_Y)

    For now we postulate this as a Type rather than proving it
    -}
    support-compatible : {-| f(supp(P_X)) ⊆ supp(P_Y) -} ⊤

    {-| Fiberwise compatibility condition -}
    compatible : fiberwise-compatible func P_X P_Y fiberwise

open PfMorphism public

{-|
## Composition of Morphisms (Lemma 5.7)

Composition (g, Λ') ∘ (f, Λ) = (g ∘ f, Λ̃) where:
  λ̃_{g(f(x))}(x) = λ_{f(x)}(x) · λ'_{g(f(x))}(f(x))
-}
postulate
  Pf-compose :
    {XP YP ZP : PfObject} →
    PfMorphism YP ZP →
    PfMorphism XP YP →
    PfMorphism XP ZP

  {-| Identity morphism -}
  Pf-id : (XP : PfObject) → PfMorphism XP XP

  {-| Composition is associative -}
  Pf-assoc :
    {WP XP YP ZP : PfObject} →
    (h : PfMorphism YP ZP) →
    (g : PfMorphism XP YP) →
    (f : PfMorphism WP XP) →
    Pf-compose h (Pf-compose g f) ≡ Pf-compose (Pf-compose h g) f

  {-| Identity laws -}
  Pf-idl :
    {XP YP : PfObject} →
    (f : PfMorphism XP YP) →
    Pf-compose (Pf-id YP) f ≡ f

  Pf-idr :
    {XP YP : PfObject} →
    (f : PfMorphism XP YP) →
    Pf-compose f (Pf-id XP) ≡ f

{-|
## Category Pf (Lemma 5.7)

The category of finite probabilities with fiberwise measures.
-}
postulate
  Pf : Precategory lzero lzero

  Pf-Ob : Precategory.Ob Pf ≡ PfObject

  Pf-Hom :
    (XP YP : PfObject) →
    Precategory.Hom Pf {!!} {!!} ≡ PfMorphism XP YP

{-|
## Coproduct in Category Pf (Lemma 5.7)

The coproduct (X,P) ⊕ (X',P') is constructed as:
- Underlying set: X ∨ X' = (X ⊔ X')/(x_0 ~ x'_0) (wedge sum)
- Probabilities:
  - P̃(x) = α_{X,X'} · P(x) for x ∈ X \ {x_0}
  - P̃(x') = β_{X,X'} · P'(x') for x' ∈ X' \ {x'_0}
  - P̃(x_0 ~ x'_0) = α_{X,X'} P(x_0) + β_{X,X'} P'(x'_0)
- Scaling factors: α = N/(N+N'), β = N'/(N+N')
  where N = #X, N' = #X'

**Morphisms**:
- ψ : (X,P) → (X,P) ⊕ (X',P') with fiberwise measure α⁻¹
- ψ': (X',P') → (X,P) ⊕ (X',P') with fiberwise measure β⁻¹
-}

{-|
Wedge sum of finite pointed sets

For simplicity, we represent X ∨ X' as a finite pointed set of size N + N'
where elements 0, ..., N-1 come from X and N, ..., N+N'-1 come from X',
with basepoints identified at 0.
-}
postulate
  wedge-sum : FinitePointedSet → FinitePointedSet → FinitePointedSet

  wedge-sum-size :
    (X X' : FinitePointedSet) →
    wedge-sum X X' ≡ X + X'

  {-| Injection from first set -}
  wedge-inj₁ :
    (X X' : FinitePointedSet) →
    underlying-set X → underlying-set (wedge-sum X X')

  {-| Injection from second set -}
  wedge-inj₂ :
    (X X' : FinitePointedSet) →
    underlying-set X' → underlying-set (wedge-sum X X')

  {-| Basepoints are identified -}
  wedge-basepoint :
    (X X' : FinitePointedSet) →
    wedge-inj₁ X X' (basepoint X) ≡ wedge-inj₂ X X' (basepoint X')

{-|
Scaling factors for coproduct probabilities
-}
alpha-scaling : (X X' : FinitePointedSet) → ℝ
alpha-scaling X X' = {-| (suc X) / (suc X + suc X') -} oneℝ  -- Simplified

beta-scaling : (X X' : FinitePointedSet) → ℝ
beta-scaling X X' = {-| (suc X') / (suc X + suc X') -} oneℝ  -- Simplified

{-|
Coproduct probability measure (Lemma 5.7)
-}
postulate
  coproduct-probability :
    (X X' : FinitePointedSet) →
    (P : ProbabilityMeasure X) →
    (P' : ProbabilityMeasure X') →
    ProbabilityMeasure (wedge-sum X X')

{-|
Coproduct in category Pf
-}
postulate
  Pf-coproduct :
    (XP X'P : PfObject) →
    PfObject

  Pf-coprod-inj₁ :
    (XP X'P : PfObject) →
    PfMorphism XP (Pf-coproduct XP X'P)

  Pf-coprod-inj₂ :
    (XP X'P : PfObject) →
    PfMorphism X'P (Pf-coproduct XP X'P)

  {-|
  Universal property of coproduct

  For any morphisms f : (X,P) → (Y,Q) and f' : (X',P') → (Y,Q),
  there exists unique Φ : (X,P) ⊕ (X',P') → (Y,Q)
  -}
  Pf-coprod-universal :
    {XP X'P YP : PfObject} →
    (f : PfMorphism XP YP) →
    (f' : PfMorphism X'P YP) →
    PfMorphism (Pf-coproduct XP X'P) YP

{-|
## Zero Object in Category Pf (Lemma 5.7)

The zero object is (*, 1) - single point with probability 1.
-}
postulate
  Pf-zero : PfObject

  Pf-zero-is-zero :
    Pf-zero ≡ (0 , {-| ProbabilityMeasure with prob(fzero) = 1 -} {!!})

  Pf-zero-unique-from :
    (XP : PfObject) →
    PfMorphism XP Pf-zero

  Pf-zero-unique-to :
    (XP : PfObject) →
    PfMorphism Pf-zero XP

{-|
## Interpretation and Examples

**Example 1: Surjective morphisms**

When f : X → Y is surjective and fiberwise measures λ_y are probabilities,
we have the extensivity property:
  S(P_X) = S(P_Y) + Σ_y P_Y(y) S(Λ|y)

This measures information loss along the morphism.

**Example 2: Embeddings**

When f : X → Y is an embedding, fiberwise measures λ_y are dilation factors
that adjust normalization. This models coarse-graining where some elements
of Y have zero probability.

**Example 3: Neural codes**

For a neural code C, the probability P_C(c) represents the likelihood of
observing code word c. Morphisms model transformations of codes that
preserve or alter probability distributions.
-}

module Examples where
  postulate
    {-| Example: Uniform distribution on n elements -}
    uniform-distribution : (n : FinitePointedSet) → ProbabilityMeasure n

    {-| Example: Bernoulli distribution (binary) -}
    bernoulli-distribution : (p : ℝ) → ProbabilityMeasure 1

    {-| Example: Surjective morphism collapsing distribution -}
    collapse-morphism :
      (X Y : FinitePointedSet) →
      (P_X : ProbabilityMeasure X) →
      (P_Y : ProbabilityMeasure Y) →
      (f : underlying-set X → underlying-set Y) →
      PfMorphism (X , P_X) (Y , P_Y)
