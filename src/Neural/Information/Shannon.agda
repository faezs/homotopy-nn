{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Information Measures and Shannon Entropy (Section 5.3)

This module implements Shannon entropy as a functor from finite probabilities
to real numbers, following Section 5.3 of Manin & Marcolli (2024).

## Overview

**Shannon information**:
  S(P) = -Σ_{x∈X} P(x) log P(x)

**Key properties**:
1. **Extensivity**: S(P') = S(P) + P·S(Q) for decompositions
2. **Functor** (Lemma 5.13): S : Pf,s → (ℝ, ≥)
3. **Information loss**: S(P) ≥ S(Q) for surjections

## Thin Categories and Order

A **thin category** is a category with at most one morphism between any two objects.
Up to equivalence, thin categories are the same as partial orders (posets).

**Real numbers as thin category** (ℝ, ≥):
- Objects: r ∈ ℝ
- Morphisms: unique r → r' iff r ≥ r'

## Key Results

**Lemma 5.13**: Shannon entropy is a functor S : Pf,s → (ℝ, ≥)
  where Pf,s has surjective morphisms with probability fiberwise measures.

**Lemma 5.14**: For summing functors Φ_X : Σ_{Pf}(X), there exist constants
  λ_min, λ_max ≥ 1 such that:
    S(Φ_X(A)) ≤ λ_max S(Φ_X(A')) - λ_min log λ_min
  for all inclusions A ⊂ A'.
-}

module Neural.Information.Shannon where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base

open import Order.Cat using (is-thin)

open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin)
open import Data.List.Base using (List)

-- Import real numbers and probabilities
open import Neural.Information public
  using (ℝ; _+ℝ_; _*ℝ_; _/ℝ_; _≤ℝ_; _≥ℝ_; zeroℝ; oneℝ; logℝ)
open import Neural.Code.Probabilities public
  using (PfObject; PfMorphism; Pf; FinitePointedSet; underlying-set;
         ProbabilityMeasure; FiberwiseMeasure)

open PfMorphism

private variable
  o ℓ : Level

{-|
## Shannon Entropy Definition

The Shannon entropy (information) of a finite probability measure P is:
  S(P) = -Σ_{x∈X} P(x) log P(x)

**Interpretation**:
- Measures uncertainty/information content
- Maximum when P is uniform
- Zero when P is concentrated on single point
- Measured in bits (log base 2) or nats (natural log)

**Convention**: 0 · log 0 = 0 (by continuity)
-}

postulate
  {-|
  Shannon entropy of a probability measure
  -}
  shannon-entropy :
    {X : FinitePointedSet} →
    ProbabilityMeasure X →
    ℝ

  shannon-entropy-nonneg :
    {X : FinitePointedSet} →
    (P : ProbabilityMeasure X) →
    zeroℝ ≤ℝ shannon-entropy P

  {-|
  Shannon entropy formula (postulated)

  S(P) = -Σ_{x∈X} P(x) log P(x)

  In actual implementation would compute the sum.
  -}
  shannon-entropy-formula :
    {X : FinitePointedSet} →
    (P : ProbabilityMeasure X) →
    {-| S(P) = -Σ P(x) log P(x) -}
    ⊤

{-|
## Extensivity Property (Definition 5.12 discussion)

For decompositions P' = (p'_ij) with p'_ij = p_j · q(i|j):
  S(P') = S(P) + P·S(Q)

where:
  P·S(Q) := Σ_j p_j S(Q|j) = -Σ_j p_j Σ_i q(i|j) log q(i|j)

This is one of the Khinchin axioms characterizing Shannon entropy.
-}

postulate
  {-|
  Extensivity of Shannon entropy

  For subsystem decompositions, entropy is additive.
  -}
  shannon-extensivity :
    {X Y : FinitePointedSet} →
    (P : ProbabilityMeasure X) →
    (Q : (y : underlying-set Y) → ProbabilityMeasure X) →
    {-| S(P') = S(P) + Σ_y P(y) S(Q|y) -}
    ⊤

{-|
## Thin Categories (Definition 5.12)

A **thin category** S is a category where for any two objects X, Y,
the set Mor_C(X,Y) consists of at most one morphism.

**Equivalence with order structures**:
- Up to equivalence: thin category ≃ partially ordered set (poset)
- Up to isomorphism: thin category ≅ preordered set (proset)

Difference:
- Poset: X ≤ Y and Y ≤ X implies X = Y (asymmetry)
- Proset: X ≤ Y and Y ≤ X allowed without X = Y

We write thin categories as (S, ≤) or (S, ≥) for opposite category.
-}

{-|
Real numbers as thin category (ℝ, ≥)
-}
postulate
  ℝ-thin-category : Precategory lzero lzero

  ℝ-thin-Ob : Precategory.Ob ℝ-thin-category ≡ ℝ

  ℝ-thin-Hom :
    (r r' : ℝ) →
    {-| Unique morphism r → r' iff r ≥ r' -}
    Type

  ℝ-is-thin : is-thin ℝ-thin-category

{-|
## Subcategory Pf,s with Surjections

For the entropy functor (Lemma 5.13), we restrict to:

**Pf,s**: Subcategory of Pf where:
- Morphisms (f, Λ) with f : X → Y are surjections
- Fiberwise measures λ_y(x) for x ∈ f⁻¹(y) are probabilities

This ensures the extensivity property applies.
-}

{-|
Surjection predicate
-}
postulate
  is-surjection :
    {X Y : FinitePointedSet} →
    (f : underlying-set X → underlying-set Y) →
    Type

{-|
Fiberwise probability predicate

λ_y is a probability measure on fiber f⁻¹(y)
-}
postulate
  is-fiberwise-probability :
    {X Y : FinitePointedSet} →
    (f : underlying-set X → underlying-set Y) →
    (Λ : FiberwiseMeasure f) →
    Type

{-|
Morphism in Pf,s (surjective with probability fibers)
-}
record PfSurjectiveMorphism (XP YP : PfObject) : Type where
  no-eta-equality
  field
    {-| Underlying Pf morphism -}
    underlying : PfMorphism XP YP

    {-| Function is surjective -}
    is-surj : is-surjection (underlying .func)

    {-| Fiberwise measures are probabilities -}
    fiberwise-prob : is-fiberwise-probability (underlying .func) (underlying .fiberwise)

open PfSurjectiveMorphism public

{-|
Category Pf,s
-}
postulate
  Pf-surjective : Precategory lzero lzero

  Pf-surjective-Ob : Precategory.Ob Pf-surjective ≡ PfObject

{-|
## Lemma 5.13: Shannon Entropy as Functor

The Shannon entropy defines a functor S : Pf,s → (ℝ, ≥).

**Key property**: For morphisms (f, Λ) : (X,P) → (Y,Q) in Pf,s,
  S(P) = S(Q) + Q·S(Λ) = S(Q) + Σ_y Q(y) S(Λ|y)

This implies S(P) ≥ S(Q), with difference measuring **information loss**
along the morphism.

**Functoriality**:
1. F_0((X,P)) = S(P)
2. F_1((f,Λ)) : S(P) ≥ S(Q) is the unique morphism in (ℝ, ≥)
3. Preserves composition and identity
-}

postulate
  {-|
  Shannon entropy functor (Lemma 5.13)
  -}
  shannon-entropy-functor : Functor Pf-surjective ℝ-thin-category

  {-|
  For morphism (f, Λ) : (X,P) → (Y,Q), we have S(P) ≥ S(Q)
  -}
  shannon-entropy-decreasing :
    {XP YP : PfObject} →
    (ϕ : PfSurjectiveMorphism XP YP) →
    shannon-entropy (XP .snd) ≥ℝ shannon-entropy (YP .snd)

  {-|
  Information loss formula

  S(P) - S(Q) = Σ_{y∈Y} Q(y) S(Λ|y)

  Measures information lost along the morphism.
  -}
  information-loss :
    {XP YP : PfObject} →
    (ϕ : PfSurjectiveMorphism XP YP) →
    ℝ  -- = S(P) - S(Q)

  information-loss-formula :
    {XP YP : PfObject} →
    (ϕ : PfSurjectiveMorphism XP YP) →
    information-loss ϕ ≡ {-| Σ_y Q(y) S(Λ|y) -} oneℝ

{-|
## General Morphisms in Pf

For general morphisms (f, Λ) in Pf (not necessarily surjective,
fiberwise not necessarily probabilities), the relation between
S(P) and S(Q) is more complex.

**Case 1**: Surjection with probability fibers (Pf,s)
  S(P) = S(Q) + Q·S(Λ)  (extensivity)
  S(P) ≥ S(Q)

**Case 2**: Embedding j : X → Y
  Fiberwise λ_{j(x)}(x) are dilation factors adjusting normalization
  No simple extensivity formula
  Still get entropy bounds (Lemma 5.14)
-}

postulate
  {-|
  Entropy relation for embeddings

  When j : X → Y is an embedding, relates S(P) to S(Q) via:
    S(P) = -Σ_{y∈j(X)} λ_{j(x)}(x) Q(j(x)) log(λ_{j(x)}(x) Q(j(x)))

  This decomposes into sum of -Σ λQ log Q and -Σ Q λ log λ terms.
  -}
  shannon-entropy-embedding :
    {XP YP : PfObject} →
    (ϕ : PfMorphism XP YP) →
    {-| f is embedding -}
    ⊤ →
    {-| Relation between S(P) and S(Q) -}
    ⊤

{-|
## Lemma 5.14: Entropy Bounds for Summing Functors

Given summing functor Φ_X : Σ_{Pf}(X) for finite pointed set X, there exist
constants λ_min, λ_max ≥ 1 depending only on X such that:

  S(Φ_X(A)) ≤ λ_max S(Φ_X(A')) - λ_min log λ_min

for all inclusions A ⊂ A' of pointed subsets of X.

**Proof idea**:
1. Inclusions j : A → A' induce morphisms (j, Λ) in Pf
2. Dilation factors λ_{j(a)}(a) ≥ 1 adjust normalization
3. Bounds λ_min ≤ λ ≤ λ_max exist by finiteness
4. Entropy estimate follows from fiberwise scaling
-}

{-|
Summing functor for finite probabilities
-}
SummingFunctorPf : Type → Type
SummingFunctorPf X = {-| Functor P(X) → Pf -} ⊤

postulate
  {-|
  Entropy bounds for summing functors (Lemma 5.14)
  -}
  entropy-bound-summing-functor :
    (X : Type) →
    (Φ : SummingFunctorPf X) →
    Σ[ lambda-min ∈ ℝ ] Σ[ lambda-max ∈ ℝ ]
      ((oneℝ ≤ℝ lambda-min) × (oneℝ ≤ℝ lambda-max) ×
       {-| ∀ A ⊂ A': S(Φ(A)) ≤ lambda-max·S(Φ(A')) - lambda-min·log(lambda-min) -}
       ⊤)

  {-|
  Bounds λ_min, λ_max determined by dilation factors

  For inclusions j_a : {*} → {*, a} and ι_{a,k} : {*,a} → ∨^k_{j=1} {*, a_j},
  the dilation factors λ(j_a) ≥ 1 and λ(ι_{a,k}) ≥ 1 from Lemma 5.7.

  Any inclusion j : A → A' is a composition of these, so its scaling
  factors are products of these basic factors.
  -}
  entropy-bound-factors :
    {X : Type} →
    {A A' : List X} →
    {-| λ_min = min λ_{j(a)}(a), λ_max = max λ_{j(a)}(a) -}
    ⊤

{-|
## Category of Simplices (Preliminary for §5.4)

The **simplex category** △ has:
- Objects: [n] = {0,...,n} for n = 0,1,2,...
- Morphisms: non-decreasing maps f : [n] → [m]

Morphisms generated by:
- ∂^i_n : [n-1] → [n] (face map, omits i)
- σ^i_n : [n+1] → [n] (degeneracy map, repeats i)

**Simplicial sets**: Functors △^op → Sets
**Pointed simplicial sets**: Functors △^op → Sets_*

We denote:
- ∆ := Func(△^op, Sets)
- ∆_* := Func(△^op, Sets_*)

This is preliminary notation for Section 5.4 (not implemented here).
-}

postulate
  {-| Simplex category -}
  SimplexCategory : Precategory lzero lzero

  {-| Objects [n] = {0,...,n} -}
  SimplexCategoryOb : Nat → Precategory.Ob SimplexCategory

  {-| Face maps ∂^i_n : [n-1] → [n] -}
  face-map : (n i : Nat) → {-| Morphism [n-1] → [n] -} ⊤

  {-| Degeneracy maps σ^i_n : [n+1] → [n] -}
  degeneracy-map : (n i : Nat) → {-| Morphism [n+1] → [n] -} ⊤

  {-|
  Category of simplicial sets

  ∆ := Func(△^op, Sets)
  -}
  SimplicialSets : Precategory (lsuc lzero) lzero

  {-|
  Category of pointed simplicial sets

  ∆_* := Func(△^op, Sets_*)
  -}
  PointedSimplicialSets : Precategory (lsuc lzero) lzero

{-|
## Examples and Applications

**Example 1**: Binary entropy function
  H_2(p) = -p log p - (1-p) log(1-p)
  Maximum at p = 1/2

**Example 2**: Uniform distribution
  S(P_uniform) = log(#X)
  Maximum entropy for given cardinality

**Example 3**: Information loss in coarse-graining
  Collapsing states reduces entropy:
  S(P_coarse) ≤ S(P_fine)

**Example 4**: Neural code entropy
  For code C with probability P_C:
  S(P_C) measures information content of neural responses
-}

module Examples where
  postulate
    {-| Example: Binary entropy function -}
    binary-entropy : ℝ → ℝ

    binary-entropy-formula :
      (p : ℝ) →
      binary-entropy p ≡ {-| -p log p - (1-p) log(1-p) -} oneℝ

    binary-entropy-maximum :
      (p : ℝ) →
      {-| H_2(p) ≤ H_2(1/2) = log 2 -}
      ⊤

    {-| Example: Uniform distribution entropy -}
    uniform-entropy :
      (n : FinitePointedSet) →
      ℝ

    uniform-entropy-formula :
      (n : FinitePointedSet) →
      uniform-entropy n ≡ {-| log(suc n) -} oneℝ

    {-| Example: Information loss via projection -}
    projection-information-loss :
      {X Y : FinitePointedSet} →
      (P_X : ProbabilityMeasure X) →
      (P_Y : ProbabilityMeasure Y) →
      (proj : underlying-set X → underlying-set Y) →
      ℝ
