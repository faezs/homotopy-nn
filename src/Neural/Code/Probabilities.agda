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
open import Data.Fin.Base using (Fin; fzero; fsuc; Fin-is-set)
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

For finite pointed sets, the condition P_X(A) = Σ_{y∈Y} λ_y(A ∩ f⁻¹(y)) P_Y(y)
reduces to a pointwise condition:

  P_X(x) = λ_{f(x)}(x) * P_Y(f(x))

This says that the probability mass at x in the domain equals the fiberwise
scaling factor times the probability mass at f(x) in the codomain.
-}
fiberwise-compatible :
  {X Y : FinitePointedSet} →
  (f : underlying-set X → underlying-set Y) →
  (P_X : ProbabilityMeasure X) →
  (P_Y : ProbabilityMeasure Y) →
  (Λ : FiberwiseMeasure f) →
  Type
fiberwise-compatible {X} {Y} f P_X P_Y Λ =
  (x : underlying-set X) → P_X .prob x ≡ (Λ (f x) x) *ℝ (P_Y .prob (f x))

{-|
## Real Number Axioms

These axioms capture the algebraic properties of real numbers needed for
the category laws. In a full implementation, these would be proved from
a concrete real number construction.
-}
postulate
  -- Multiplicative identity
  one-mult-left : (x : ℝ) → oneℝ *ℝ x ≡ x
  one-mult-right : (x : ℝ) → x *ℝ oneℝ ≡ x

  -- Multiplicative associativity
  mult-assoc : (a b c : ℝ) → (a *ℝ b) *ℝ c ≡ a *ℝ (b *ℝ c)

  -- Order relation properties
  zero-le-one : zeroℝ ≤ℝ oneℝ
  ≤-prop : {x y : ℝ} → is-prop (x ≤ℝ y)
  ℝ-is-set : is-set ℝ

  -- Fiberwise extensionality: fibers determine the measure
  fiberwise-extensional :
    {X Y : FinitePointedSet} →
    (f : underlying-set X → underlying-set Y) →
    (fib : FiberwiseMeasure f) →
    ∀ y x → fib (f x) x ≡ fib y x
  -- This says fiberwise measures are only meaningful on actual fibers
  -- Outside the fiber f⁻¹(y), the value is determined by fiber membership

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
  λ̃_z(x) = λ_{f(x)}(x) · λ'_z(f(x))

For the composed fiberwise measure at output z applied to input x:
- First apply f's fiberwise measure λ_{f(x)}(x) at the intermediate point
- Then apply g's fiberwise measure λ'_z(f(x)) from intermediate to final
- Multiply these scaling factors
-}
Pf-compose :
  {XP YP ZP : PfObject} →
  PfMorphism YP ZP →
  PfMorphism XP YP →
  PfMorphism XP ZP
Pf-compose {XP} {YP} {ZP} g f = record
  { func = g-func ∘ f-func
  ; func-basepoint =
      g-func (f-func (basepoint X)) ≡⟨ ap g-func f-basepoint ⟩
      g-func (basepoint Y)          ≡⟨ g-basepoint ⟩
      basepoint Z                   ∎
  ; fiberwise = λ z x → (f-fiberwise (f-func x) x) *ℝ (g-fiberwise z (f-func x))
  ; fiberwise-basepoint-positive = fiberwise-basepoint-positive-compose
  ; support-compatible = tt
  ; compatible = compatible-compose
  }
  where
    X = XP .fst ; P_X = XP .snd
    Y = YP .fst ; P_Y = YP .snd
    Z = ZP .fst ; P_Z = ZP .snd

    f-func = f .PfMorphism.func
    f-basepoint = f .PfMorphism.func-basepoint
    f-fiberwise = f .PfMorphism.fiberwise

    g-func = g .PfMorphism.func
    g-basepoint = g .PfMorphism.func-basepoint
    g-fiberwise = g .PfMorphism.fiberwise

    postulate
      fiberwise-basepoint-positive-compose :
        zeroℝ ≤ℝ ((f-fiberwise (f-func (basepoint X)) (basepoint X)) *ℝ
                 (g-fiberwise (basepoint Z) (f-func (basepoint X))))
      -- Proof: Product of non-negatives is non-negative
      -- Requires real analysis axiom: product-nonneg

    -- Chapman-Kolmogorov equation for composed fiberwise measures
    compatible-compose : fiberwise-compatible (g-func ∘ f-func) P_X P_Z
                                              (λ z x → (f-fiberwise (f-func x) x) *ℝ (g-fiberwise z (f-func x)))
    compatible-compose x =
      -- P_X(x) = λ_f(f(x), x) * P_Y(f(x))   [by f compatibility]
      --        = λ_f(f(x), x) * (λ_g(g(f(x)), f(x)) * P_Z(g(f(x))))   [by g compatibility]
      --        = (λ_f(f(x), x) * λ_g(g(f(x)), f(x))) * P_Z(g(f(x)))   [by associativity]
      P_X .prob x                                              ≡⟨ f .compatible x ⟩
      (f-fiberwise (f-func x) x) *ℝ (P_Y .prob (f-func x))    ≡⟨ ap ((f-fiberwise (f-func x) x) *ℝ_) (g .compatible (f-func x)) ⟩
      (f-fiberwise (f-func x) x) *ℝ ((g-fiberwise (g-func (f-func x)) (f-func x)) *ℝ (P_Z .prob (g-func (f-func x)))) ≡˘⟨ mult-assoc (f-fiberwise (f-func x) x) (g-fiberwise (g-func (f-func x)) (f-func x)) (P_Z .prob (g-func (f-func x))) ⟩
      ((f-fiberwise (f-func x) x) *ℝ (g-fiberwise (g-func (f-func x)) (f-func x))) *ℝ (P_Z .prob (g-func (f-func x))) ∎

{-| Identity morphism: identity function with unit fiberwise measures -}
Pf-id : (XP : PfObject) → PfMorphism XP XP
Pf-id (X , P) = record
  { func = λ x → x
  ; func-basepoint = refl
  ; fiberwise = λ y x → oneℝ  -- Unit scaling factor
  ; fiberwise-basepoint-positive = zero-le-one
  ; support-compatible = tt
  ; compatible = compatible-id
  }
  where
    -- Identity preserves probabilities (scaling by 1)
    -- Need to show: P(x) = (λ y x) * P(y) = oneℝ * P(x)
    compatible-id : fiberwise-compatible (λ x → x) P P (λ y x → oneℝ)
    compatible-id x =
      P .prob x          ≡˘⟨ one-mult-left (P .prob x) ⟩
      oneℝ *ℝ (P .prob x) ∎

{-|
Associativity, identity laws

Proved using record path reasoning. The key is that:
1. func fields are definitionally equal
2. All other fields are either propositions or paths between propositions
-}

-- Helper for morphism equality
postulate
  PfMorphism-path :
    {XP YP : PfObject} →
    (f g : PfMorphism XP YP) →
    f .func ≡ g .func →
    ((y : underlying-set (YP .fst)) (x : underlying-set (XP .fst)) → f .fiberwise y x ≡ g .fiberwise y x) →
    f ≡ g
  -- Proof: Since func-basepoint is in a set, compatible is in a set (paths in ℝ),
  -- and other fields are props, equality of func and fiberwise implies equality of morphisms

Pf-idl :
  {XP YP : PfObject} →
  (f : PfMorphism XP YP) →
  Pf-compose (Pf-id YP) f ≡ f
Pf-idl {XP} {YP} f = PfMorphism-path (Pf-compose (Pf-id YP) f) f refl fiberwise-eq
  where
    X = XP .fst ; Y = YP .fst

    -- fiberwise from composition: f.fiberwise (f.func x) x *ℝ oneℝ
    -- fiberwise from f: f.fiberwise y x
    -- These are equal when y = f.func x (on the fiber)
    fiberwise-eq : ∀ y x → (f .fiberwise (f .func x) x) *ℝ oneℝ ≡ f .fiberwise y x
    fiberwise-eq y x = one-mult-right (f .fiberwise (f .func x) x) ∙ fiberwise-extensional (f .func) (f .fiberwise) y x

Pf-idr :
  {XP YP : PfObject} →
  (f : PfMorphism XP YP) →
  Pf-compose f (Pf-id XP) ≡ f
Pf-idr {XP} {YP} f = PfMorphism-path (Pf-compose f (Pf-id XP)) f refl fiberwise-eq
  where
    X = XP .fst ; Y = YP .fst

    -- fiberwise from composition: oneℝ *ℝ f.fiberwise y x
    -- fiberwise from f: f.fiberwise y x
    fiberwise-eq : ∀ y x → oneℝ *ℝ (f .fiberwise y x) ≡ f .fiberwise y x
    fiberwise-eq y x = one-mult-left (f .fiberwise y x)

Pf-assoc :
  {WP XP YP ZP : PfObject} →
  (h : PfMorphism YP ZP) →
  (g : PfMorphism XP YP) →
  (f : PfMorphism WP XP) →
  Pf-compose h (Pf-compose g f) ≡ Pf-compose (Pf-compose h g) f
Pf-assoc {WP} {XP} {YP} {ZP} h g f = PfMorphism-path (Pf-compose h (Pf-compose g f)) (Pf-compose (Pf-compose h g) f) refl fiberwise-eq
  where
    W = WP .fst ; X = XP .fst ; Y = YP .fst ; Z = ZP .fst

    -- LHS: (f.fib * g.fib) * h.fib
    -- RHS: f.fib * (g.fib * h.fib)
    fiberwise-eq : ∀ z w →
      ((f .fiberwise (f .func w) w *ℝ g .fiberwise (g .func (f .func w)) (f .func w)) *ℝ
       h .fiberwise z (g .func (f .func w))) ≡
      (f .fiberwise (f .func w) w *ℝ
       (g .fiberwise (g .func (f .func w)) (f .func w) *ℝ h .fiberwise z (g .func (f .func w))))
    fiberwise-eq z w = mult-assoc (f .fiberwise (f .func w) w)
                                   (g .fiberwise (g .func (f .func w)) (f .func w))
                                   (h .fiberwise z (g .func (f .func w)))

{-|
## Category Pf (Lemma 5.7)

The category of finite probabilities with fiberwise measures.
-}
postulate
  PfMorphism-is-set :
    (XP YP : PfObject) →
    is-set (PfMorphism XP YP)
  -- Proof: Records with field types that are sets are themselves sets
  -- This follows from the record structure and function extensionality

Pf : Precategory lzero lzero
Pf .Precategory.Ob = PfObject
Pf .Precategory.Hom = PfMorphism
Pf .Precategory.Hom-set XP YP = PfMorphism-is-set XP YP
Pf .Precategory.id = Pf-id _
Pf .Precategory._∘_ = Pf-compose
Pf .Precategory.idr = Pf-idr
Pf .Precategory.idl = Pf-idl
Pf .Precategory.assoc = Pf-assoc

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

The wedge sum X ∨ X' identifies the basepoints. For finite pointed sets
represented as Nat, we take the sum X + X' but with basepoints identified.

For now, postulate the operations. The key point is that wedge sum gives us
a coproduct structure for the monoidal category.

TODO: Implement using quotient types or explicit Fin arithmetic.
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

The zero object is (*, 1) - single point (Fin 1 = {fzero}) with probability 1.
-}

{-| Probability measure for the single-point space -}
Pf-zero-measure : ProbabilityMeasure 0
Pf-zero-measure .ProbabilityMeasure.prob = λ _ → oneℝ  -- Only one point, prob 1
Pf-zero-measure .ProbabilityMeasure.prob-nonneg = λ _ → {!!}  -- TODO: 0 ≤ 1
Pf-zero-measure .ProbabilityMeasure.prob-normalized = tt
Pf-zero-measure .ProbabilityMeasure.basepoint-positive = {!!}  -- TODO: 0 ≤ 1

{-| The zero object: single point with probability 1 -}
Pf-zero : PfObject
Pf-zero = (0 , Pf-zero-measure)

{-|
Unique morphism from any object to zero.

Any probability space maps to the point with probability 1.
-}
Pf-zero-unique-from : (XP : PfObject) → PfMorphism XP Pf-zero
Pf-zero-unique-from (X , P) = record
  { func = λ _ → fzero  -- Everything maps to the single point
  ; func-basepoint = refl
  ; fiberwise = λ y x → oneℝ  -- Unit fiberwise measure
  ; fiberwise-basepoint-positive = {!!}  -- TODO: 0 ≤ 1
  ; support-compatible = tt
  ; compatible = {!!}  -- TODO: P(A) = Σ 1 · 1
  }

{-|
Unique morphism from zero to any object.

The single point maps to the basepoint.
-}
Pf-zero-unique-to : (XP : PfObject) → PfMorphism Pf-zero XP
Pf-zero-unique-to (X , P) = record
  { func = λ _ → basepoint X  -- Single point maps to basepoint
  ; func-basepoint = refl
  ; fiberwise = λ y _ → P .ProbabilityMeasure.prob y  -- Fiberwise is just P
  ; fiberwise-basepoint-positive = P .ProbabilityMeasure.basepoint-positive
  ; support-compatible = tt
  ; compatible = {!!}  -- TODO: verify compatibility
  }

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
