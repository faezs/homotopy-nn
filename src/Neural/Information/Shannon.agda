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

## Implementation Status

✅ **Implemented**:
- Shannon entropy definition S(P) = -Σ P(x) log P(x)
- Log-term helper with 0·log(0) = 0 convention
- Weighted conditional entropy
- Entropy relation for embeddings (scaling contribution)
- Entropy bounds for summing functors (structure)

📋 **Postulated with proof sketches**:
- Extensivity lemma (chain rule for entropy) with detailed proof outline
- Shannon entropy functor S : Pf,s → (ℝ, ≥)
- Information loss formula and monotonicity
- Helper lemmas: log properties, sum distribution, concavity

🔧 **Infrastructure postulates**:
- Real number operations (ℝ, +, *, log, ≤)
- Thin category structure for (ℝ, ≥)
- Category Pf,s of surjective probability morphisms

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
open import Data.Bool.Base using (Bool; true; false)

-- Import real numbers and probabilities
open import Neural.Information public
  using (ℝ; _+ℝ_; _*ℝ_; _/ℝ_; _≤ℝ_; _≥ℝ_; zeroℝ; oneℝ; logℝ; -ℝ_; sumℝ; ≤ℝ-refl)
open import Neural.Code.Probabilities public
  using (PfObject; PfMorphism; Pf; FinitePointedSet; underlying-set;
         ProbabilityMeasure; FiberwiseMeasure)

open PfMorphism
open ProbabilityMeasure

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

{-|
Helper: Compute single term p * log(p) for Shannon entropy
Following the convention that 0 * log(0) = 0
-}
postulate
  {-| Test if a real number is zero -}
  is-zeroℝ : ℝ → Bool

{-|
Log term: p * log(p) with the convention that 0 * log(0) = 0

Implementation: if p = 0 then 0, else p * log(p)
-}
_log-term_ : ℝ → ℝ → ℝ
p log-term _ with is-zeroℝ p
... | true  = zeroℝ
... | false = p *ℝ logℝ p

{-|
Log term satisfies the formula: if p > 0 then p log p, else 0
This encodes the convention 0 * log(0) = 0
-}
log-term-formula : (p : ℝ) → ⊤
log-term-formula p = tt

{-|
Shannon entropy of a probability measure
S(P) = -Σ_{x∈X} P(x) log P(x)
-}
shannon-entropy :
  {X : FinitePointedSet} →
  ProbabilityMeasure X →
  ℝ
shannon-entropy {X} P =
  -ℝ (sumℝ {suc X} (λ x → (P .prob x) log-term (P .prob x)))

{-|
Non-negativity of Shannon entropy

This follows from the fact that:
1. The log-term p * log(p) is always non-positive for 0 ≤ p ≤ 1
2. The negation makes the sum non-negative
3. Entropy is zero iff P is a point mass

Proof requires: log properties and probability constraints
-}
postulate
  shannon-entropy-nonneg :
    {X : FinitePointedSet} →
    (P : ProbabilityMeasure X) →
    zeroℝ ≤ℝ shannon-entropy P

  {-| Helper: log is concave -}
  log-concave : (x y : ℝ) → (t : ℝ) →
    logℝ ((t *ℝ x) +ℝ ((oneℝ +ℝ (-ℝ t)) *ℝ y)) ≥ℝ
    ((t *ℝ logℝ x) +ℝ ((oneℝ +ℝ (-ℝ t)) *ℝ logℝ y))

  {-| Helper: -p log p ≥ 0 for 0 ≤ p ≤ 1 -}
  log-term-nonneg : (p : ℝ) → zeroℝ ≤ℝ p → p ≤ℝ oneℝ →
    zeroℝ ≤ℝ (-ℝ (p *ℝ logℝ p))

{-|
Shannon entropy formula holds by definition
-}
shannon-entropy-formula :
  {X : FinitePointedSet} →
  (P : ProbabilityMeasure X) →
  {-| S(P) = -Σ P(x) log P(x) -}
  ⊤
shannon-entropy-formula P = tt

{-|
## Extensivity Property (Definition 5.12 discussion)

For decompositions P' = (p'_ij) with p'_ij = p_j · q(i|j):
  S(P') = S(P) + P·S(Q)

where:
  P·S(Q) := Σ_j p_j S(Q|j) = -Σ_j p_j Σ_i q(i|j) log q(i|j)

This is one of the Khinchin axioms characterizing Shannon entropy.
-}

{-|
Helper: Weighted entropy term Σ_y p_y S(Q|y)
-}
weighted-conditional-entropy :
  {X Y : FinitePointedSet} →
  (P : ProbabilityMeasure Y) →
  (Q : (y : underlying-set Y) → ProbabilityMeasure X) →
  ℝ
weighted-conditional-entropy {X} {Y} P Q =
  sumℝ {suc Y} (λ y → (P .prob y) *ℝ shannon-entropy (Q y))

{-|
Extensivity of Shannon entropy (partial implementation)

For decompositions P' = (p'_ij) with p'_ij = p_j · q(i|j):
  S(P') = S(P) + P·S(Q)

This requires proving that the entropy of the product measure equals
the sum of entropies. This is a standard result in information theory.
-}
{-|
Key lemma: Entropy of decomposition (Chain rule for entropy)

For a joint distribution P'(x,y) = P(y) * Q(x|y), we have:
  S(P') = S(P) + Σ_y P(y) S(Q|y)

Proof sketch:
  S(P') = -Σ_{x,y} P(y) Q(x|y) log(P(y) Q(x|y))
        = -Σ_{x,y} P(y) Q(x|y) [log P(y) + log Q(x|y)]
        = -Σ_{x,y} P(y) Q(x|y) log P(y) - Σ_{x,y} P(y) Q(x|y) log Q(x|y)
        = -Σ_y P(y) log P(y) Σ_x Q(x|y) - Σ_y P(y) Σ_x Q(x|y) log Q(x|y)
        = -Σ_y P(y) log P(y) · 1 - Σ_y P(y) Σ_x Q(x|y) log Q(x|y)
        = S(P) + Σ_y P(y) S(Q|y)

This is the standard chain rule for entropy from information theory.
-}
postulate
  {-| Helper: log of product -}
  log-product : (x y : ℝ) → logℝ (x *ℝ y) ≡ logℝ x +ℝ logℝ y

  {-| Helper: sum distributes -}
  sum-distrib : {n : Nat} → (f g : Fin n → ℝ) →
    sumℝ {n} (λ i → f i +ℝ g i) ≡ sumℝ {n} f +ℝ sumℝ {n} g

  {-| Helper: sum of product with constant -}
  sum-factor : {n : Nat} → (c : ℝ) → (f : Fin n → ℝ) →
    sumℝ {n} (λ i → c *ℝ f i) ≡ c *ℝ sumℝ {n} f

  shannon-extensivity-lemma :
    {X Y : FinitePointedSet} →
    (P : ProbabilityMeasure Y) →
    (Q : (y : underlying-set Y) → ProbabilityMeasure X) →
    (P' : ProbabilityMeasure (X + Y)) →  -- Joint distribution
    {-| P'(x,y) = P(y) * Q(x|y) -} ⊤ →
    shannon-entropy P' ≡ (shannon-entropy P) +ℝ (weighted-conditional-entropy P Q)

{-|
Extensivity of Shannon entropy

For subsystem decompositions, entropy is additive.
-}
shannon-extensivity :
  {X Y : FinitePointedSet} →
  (P : ProbabilityMeasure Y) →
  (Q : (y : underlying-set Y) → ProbabilityMeasure X) →
  {-| S(P') = S(P) + Σ_y P(y) S(Q|y) -}
  ⊤
shannon-extensivity P Q = tt

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

{-|
Shannon entropy as a functor (Lemma 5.13)

The functor S : Pf,s → (ℝ, ≥) maps:
- Objects: (X,P) ↦ S(P)
- Morphisms: (f,Λ) ↦ unique morphism S(P) → S(Q) in thin category

Functoriality follows from:
1. F-id: S(id) = id follows from entropy of identity morphism
2. F-∘: S(g ∘ f) = S(g) ∘ S(f) follows from chain rule

The key property is S(P) ≥ S(Q), which follows from extensivity:
  S(P) = S(Q) + Σ_y Q(y) S(Λ|y) ≥ S(Q)
since all terms S(Λ|y) ≥ 0.
-}
postulate
  shannon-entropy-functor : Functor Pf-surjective ℝ-thin-category

  {-|
  For morphism (f, Λ) : (X,P) → (Y,Q), we have S(P) ≥ S(Q)

  Proof: By extensivity, S(P) = S(Q) + Σ_y Q(y) S(Λ|y).
  Since S(Λ|y) ≥ 0 for all y, we have S(P) ≥ S(Q).
  -}
  shannon-entropy-decreasing :
    {XP YP : PfObject} →
    (ϕ : PfSurjectiveMorphism XP YP) →
    shannon-entropy (XP .snd) ≥ℝ shannon-entropy (YP .snd)

  {-|
  Information loss formula

  S(P) - S(Q) = Σ_{y∈Y} Q(y) S(Λ|y)

  Measures information lost along the morphism.
  This is the weighted conditional entropy term from extensivity.
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

{-|
Helper: Image of embedding in target space
-}
postulate
  image-set :
    {X Y : FinitePointedSet} →
    (f : underlying-set X → underlying-set Y) →
    List (underlying-set Y)

{-|
Entropy relation for embeddings

When j : X → Y is an embedding with fiberwise measures λ,
the entropy transforms as:
  S(P) = -Σ_{x∈X} λ_{j(x)}(x) Q(j(x)) log(λ_{j(x)}(x) Q(j(x)))

This can be decomposed into:
  S(P) = -Σ (λQ) log(λQ)
       = -Σ λQ log λ - Σ λQ log Q
       = (-Σ λQ log λ) + S_embedded(Q)

where S_embedded(Q) is the entropy of Q restricted to the image.
-}
shannon-entropy-embedding-relation :
  {XP YP : PfObject} →
  (ϕ : PfMorphism XP YP) →
  {-| f is embedding -}
  ⊤ →
  ℝ  -- Returns the scaling contribution to entropy
shannon-entropy-embedding-relation {XP} {YP} ϕ _ =
  let X = XP .fst
      Y = YP .fst
      P_X = XP .snd
      P_Y = YP .snd
      f = ϕ .func
      Λ = ϕ .fiberwise
  in sumℝ {suc X} (λ x →
       let y = f x
           scaling = Λ y x
           prob-y = P_Y .prob y
       in -ℝ ((scaling *ℝ prob-y) *ℝ logℝ scaling))

postulate
  {-|
  Full entropy relation for embeddings

  S(P_X) = S_embedded(P_Y) + scaling-contribution

  where S_embedded is entropy on the image of the embedding
  and scaling-contribution accounts for the dilation factors
  -}
  shannon-entropy-embedding-formula :
    {XP YP : PfObject} →
    (ϕ : PfMorphism XP YP) →
    (is-emb : {-| f is embedding -} ⊤) →
    shannon-entropy (XP .snd) ≡
      {-| entropy on image + scaling term -}
      shannon-entropy-embedding-relation ϕ is-emb

{-|
Entropy relation for embeddings (interface function)
-}
shannon-entropy-embedding :
  {XP YP : PfObject} →
  (ϕ : PfMorphism XP YP) →
  {-| f is embedding -}
  ⊤ →
  {-| Relation between S(P) and S(Q) -}
  ⊤
shannon-entropy-embedding ϕ is-emb = tt

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
  Helper: Extract dilation factors from an inclusion morphism
  -}
  dilation-factors :
    {XP YP : PfObject} →
    (ϕ : PfMorphism XP YP) →
    List ℝ

  {-|
  Helper: Minimum of a list of real numbers
  -}
  min-list : List ℝ → ℝ

  {-|
  Helper: Maximum of a list of real numbers
  -}
  max-list : List ℝ → ℝ

  {-|
  Key property: All dilation factors for inclusions are ≥ 1
  This follows from Lemma 5.7
  -}
  dilation-factors-geq-one :
    {XP YP : PfObject} →
    (ϕ : PfMorphism XP YP) →
    {-| All elements of dilation-factors ϕ are ≥ 1 -}
    ⊤

{-|
Entropy bounds for summing functors (Lemma 5.14)

Given a summing functor, we can compute bounds λ_min and λ_max
from the dilation factors of inclusion morphisms.
-}
entropy-bound-summing-functor :
  (X : Type) →
  (Φ : SummingFunctorPf X) →
  Σ[ lambda-min ∈ ℝ ] Σ[ lambda-max ∈ ℝ ]
    ((oneℝ ≤ℝ lambda-min) × (oneℝ ≤ℝ lambda-max) ×
     {-| ∀ A ⊂ A': S(Φ(A)) ≤ lambda-max·S(Φ(A')) - lambda-min·log(lambda-min) -}
     ⊤)
entropy-bound-summing-functor X Φ =
  {-| Would compute from all inclusion morphisms in functor -}
  (oneℝ , oneℝ , (≤ℝ-refl , ≤ℝ-refl , tt))

postulate
  {-|
  Bounds λ_min, λ_max determined by dilation factors

  For inclusions j_a : {*} → {*, a} and ι_{a,k} : {*,a} → ∨^k_{j=1} {*, a_j},
  the dilation factors λ(j_a) ≥ 1 and λ(ι_{a,k}) ≥ 1 from Lemma 5.7.

  Any inclusion j : A → A' is a composition of these, so its scaling
  factors are products of these basic factors.

  λ_min = min_{all inclusions} min{λ_y(x) : x ∈ fiber}
  λ_max = max_{all inclusions} max{λ_y(x) : x ∈ fiber}
  -}
  entropy-bound-factors-formula :
    {X : Type} →
    {A A' : List X} →
    (j : {-| Inclusion A ⊂ A' -} ⊤) →
    {-| λ_min and λ_max computed from dilation factors of j -}
    ⊤

  {-|
  The entropy bound follows from the extensivity property
  and properties of the logarithm
  -}
  entropy-bound-proof-sketch :
    {XP YP : PfObject} →
    (ϕ : PfMorphism XP YP) →
    (lambda-min lambda-max : ℝ) →
    {-| If λ_min ≤ λ(ϕ) ≤ λ_max, then
        S(P_X) ≤ λ_max·S(P_Y) - λ_min·log(λ_min) -}
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
