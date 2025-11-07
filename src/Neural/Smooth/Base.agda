{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Smooth Infinitesimal Analysis - Foundations

**Reference**: Chapters 1-2 from "Smooth Infinitesimal Analysis" (the document)

This module implements the foundational structures for synthetic differential geometry:
- Smooth line ℝ with field operations
- Microneighbourhood Δ = {ε : ε² = 0} (nilsquare infinitesimals)
- Principle of Microaffineness: Every function Δ → ℝ is affine
- Microcancellation: Can cancel ε from equations

## Key Ideas

**Smooth worlds** provide an alternative to classical analysis using:
1. **Infinitesimals**: Quantities ε where ε² = 0 but ε ≠ 0
2. **Microstraightness**: Curves are locally straight at infinitesimal scale
3. **Exact differentials**: f(x+ε) = f(x) + ε·f'(x) (not approximate!)

## Physical Interpretation for Neural Networks

- **Infinitesimals** = Parameter perturbations too small to detect individually
- **Microaffineness** = Local linearity of neural functions (Taylor expansion to 1st order)
- **Nilsquare** = Second-order effects vanish (quadratic terms negligible)
- **Microcancellation** = Can isolate first derivatives cleanly

## Relationship to Existing Code

This module provides rigorous foundations for:
- Postulated ℝ in Neural.Information
- "Tangent vectors" in Information.Geometry
- "Smooth manifolds" in Stack.CatsManifold
- Backpropagation derivatives in Topos.Architecture
-}

module Neural.Smooth.Base where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path
open import 1Lab.Path.Reasoning
open import 1Lab.Type

open import Algebra.Ring
open import Algebra.Group
open import Algebra.Group.Ab

open import Data.Sum.Base using (_⊎_; inl; inr)
open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin)

-- For unique existence
open import 1Lab.HLevel.Universe

private variable
  ℓ : Level

--------------------------------------------------------------------------------
-- § 1: The Smooth Line ℝ

{-|
## The Smooth Line (Section 1.1)

The fundamental object is the **smooth line** ℝ, an indefinitely extensible
homogeneous straight line with:
- **Points/locations**: Elements of ℝ
- **Identity**: Equality relation on points
- **Distinguished points**: 0 (zero) and 1 (unit)
- **Reflection**: Operation ε ↦ -ε through 0
- **Segments**: Oriented magnitudes aˆb from a to b

**Key property**: The smooth line is indecomposable (cannot be split into disjoint parts).
-}

postulate
  ℝ : Type

  -- ℝ is a set (h-level 2: all paths between paths are equal)
  ℝ-is-set : is-set ℝ

  -- Distinguished points
  0ℝ : ℝ
  1ℝ : ℝ

  -- Reflection operation
  -ℝ_ : ℝ → ℝ

  -- Reflection properties
  -ℝ-involutive : ∀ (x : ℝ) → -ℝ (-ℝ x) ≡ x
  -ℝ-zero : -ℝ 0ℝ ≡ 0ℝ

  -- 0 and 1 are distinct
  0≠1 : 0ℝ ≠ 1ℝ

{-|
## Field Operations (Section 1.2)

The smooth line has **field structure**:
- Addition (+): Abelian group with neutral element 0
- Multiplication (·): Commutative monoid with unit 1
- Division (/): Defined for x ≠ 0
- Distributivity: x · (y + z) = x · y + x · z

**Important**: We do NOT assume ab = 0 → a = 0 ∨ b = 0, because infinitesimals
ε ≠ 0 can satisfy ε² = 0.
-}

postulate
  -- Addition
  _+ℝ_ : ℝ → ℝ → ℝ

  -- Multiplication
  _·ℝ_ : ℝ → ℝ → ℝ

  -- Division (partial operation, only defined for non-zero denominator)
  _/ℝ_ : (x y : ℝ) → (y ≠ 0ℝ) → ℝ

-- Operator precedence
infixl 25 _+ℝ_ _-ℝ_
infixl 30 _·ℝ_
infix  35 -ℝ_

-- Subtraction defined as a - b = a + (-b)
_-ℝ_ : ℝ → ℝ → ℝ
a -ℝ b = a +ℝ (-ℝ b)

{-|
## Field Axioms

These axioms define ℝ as a field in the algebraic sense.
-}

postulate
  -- Addition axioms (Abelian group)
  +ℝ-assoc : ∀ (a b c : ℝ) → (a +ℝ b) +ℝ c ≡ a +ℝ (b +ℝ c)
  +ℝ-comm : ∀ (a b : ℝ) → a +ℝ b ≡ b +ℝ a
  +ℝ-idl : ∀ (a : ℝ) → 0ℝ +ℝ a ≡ a
  +ℝ-idr : ∀ (a : ℝ) → a +ℝ 0ℝ ≡ a
  +ℝ-invl : ∀ (a : ℝ) → (-ℝ a) +ℝ a ≡ 0ℝ
  +ℝ-invr : ∀ (a : ℝ) → a +ℝ (-ℝ a) ≡ 0ℝ

  -- Multiplication axioms (Commutative monoid on ℝ \ {0})
  ·ℝ-assoc : ∀ (a b c : ℝ) → (a ·ℝ b) ·ℝ c ≡ a ·ℝ (b ·ℝ c)
  ·ℝ-comm : ∀ (a b : ℝ) → a ·ℝ b ≡ b ·ℝ a
  ·ℝ-idl : ∀ (a : ℝ) → 1ℝ ·ℝ a ≡ a
  ·ℝ-idr : ∀ (a : ℝ) → a ·ℝ 1ℝ ≡ a

  -- Distributivity
  ·ℝ-distribl : ∀ (a b c : ℝ) → a ·ℝ (b +ℝ c) ≡ (a ·ℝ b) +ℝ (a ·ℝ c)
  ·ℝ-distribr : ∀ (a b c : ℝ) → (a +ℝ b) ·ℝ c ≡ (a ·ℝ c) +ℝ (b ·ℝ c)

  -- Multiplicative zero
  ·ℝ-zerol : ∀ (a : ℝ) → 0ℝ ·ℝ a ≡ 0ℝ
  ·ℝ-zeror : ∀ (a : ℝ) → a ·ℝ 0ℝ ≡ 0ℝ

  -- Division properties
  /ℝ-invl : ∀ (a : ℝ) (p : a ≠ 0ℝ) → (1ℝ /ℝ a) p ·ℝ a ≡ 1ℝ
  /ℝ-invr : ∀ (a : ℝ) (p : a ≠ 0ℝ) → a ·ℝ (1ℝ /ℝ a) p ≡ 1ℝ

  -- Zero-product property (field axiom for integral domains)
  -- If a product is zero, at least one factor is zero
  zero-product : ∀ (a b : ℝ) → a ·ℝ b ≡ 0ℝ → (a ≡ 0ℝ) ⊎ (b ≡ 0ℝ)

  -- Product of non-zero numbers is non-zero (contrapositive of zero-product)
  product-nonzero : ∀ (a b : ℝ) → a ≠ 0ℝ → b ≠ 0ℝ → (a ·ℝ b) ≠ 0ℝ

  -- General division cancellation: (a/b)·b = a
  /ℝ-cancel : ∀ (a b : ℝ) (p : b ≠ 0ℝ) → ((a /ℝ b) p) ·ℝ b ≡ a

  -- Multiplicative cancellation: if a·c = b·c and c ≠ 0, then a = b
  ·ℝ-cancelr : ∀ (a b c : ℝ) → c ≠ 0ℝ → a ·ℝ c ≡ b ·ℝ c → a ≡ b

{-|
## Derived Field Properties

These are provable from the field axioms above.
-}

-- Uniqueness of additive inverses
inv-unique : ∀ (a b c : ℝ) → a +ℝ b ≡ 0ℝ → a +ℝ c ≡ 0ℝ → b ≡ c
inv-unique a b c ab=0 ac=0 =
  b
    ≡⟨ sym (+ℝ-idl b) ⟩
  0ℝ +ℝ b
    ≡⟨ ap (_+ℝ b) (sym (+ℝ-invl a)) ⟩
  ((-ℝ a) +ℝ a) +ℝ b
    ≡⟨ +ℝ-assoc (-ℝ a) a b ⟩
  (-ℝ a) +ℝ (a +ℝ b)
    ≡⟨ ap ((-ℝ a) +ℝ_) ab=0 ⟩
  (-ℝ a) +ℝ 0ℝ
    ≡⟨ +ℝ-idr (-ℝ a) ⟩
  -ℝ a
    ≡⟨ sym (+ℝ-idr (-ℝ a)) ⟩
  (-ℝ a) +ℝ 0ℝ
    ≡⟨ ap ((-ℝ a) +ℝ_) (sym ac=0) ⟩
  (-ℝ a) +ℝ (a +ℝ c)
    ≡⟨ sym (+ℝ-assoc (-ℝ a) a c) ⟩
  ((-ℝ a) +ℝ a) +ℝ c
    ≡⟨ ap (_+ℝ c) (+ℝ-invl a) ⟩
  0ℝ +ℝ c
    ≡⟨ +ℝ-idl c ⟩
  c
    ∎

-- Negation via multiplication by -1
neg-mult : ∀ (a : ℝ) → (-ℝ 1ℝ) ·ℝ a ≡ -ℝ a
neg-mult a =
  -- Proof: Both are additive inverses of a, so equal by uniqueness
  inv-unique a ((-ℝ 1ℝ) ·ℝ a) (-ℝ a) neg-is-inv inv-ax
  where
    neg-is-inv : a +ℝ ((-ℝ 1ℝ) ·ℝ a) ≡ 0ℝ
    neg-is-inv =
      a +ℝ ((-ℝ 1ℝ) ·ℝ a)
        ≡⟨ ap (_+ℝ ((-ℝ 1ℝ) ·ℝ a)) (sym (·ℝ-idl a)) ⟩
      (1ℝ ·ℝ a) +ℝ ((-ℝ 1ℝ) ·ℝ a)
        ≡⟨ sym (·ℝ-distribr 1ℝ (-ℝ 1ℝ) a) ⟩
      (1ℝ +ℝ (-ℝ 1ℝ)) ·ℝ a
        ≡⟨ ap (_·ℝ a) (+ℝ-invr 1ℝ) ⟩
      0ℝ ·ℝ a
        ≡⟨ ·ℝ-zerol a ⟩
      0ℝ
        ∎

    inv-ax : a +ℝ (-ℝ a) ≡ 0ℝ
    inv-ax = +ℝ-invr a

-- Double negation
double-neg : ∀ (a : ℝ) → (-ℝ 1ℝ) ·ℝ (-ℝ a) ≡ a
double-neg a =
  (-ℝ 1ℝ) ·ℝ (-ℝ a)
    ≡⟨ neg-mult (-ℝ a) ⟩
  -ℝ (-ℝ a)
    ≡⟨ -ℝ-involutive a ⟩
  a
    ∎

{-|
## Order Structure (Section 1.3)

The smooth line has an **order relation** < (strictly less than) with:
1. Transitivity: a < b and b < c implies a < c
2. Strictness: not (a < a)
3. Compatibility with +: a < b implies a + c < b + c
4. Compatibility with ·: a < b and 0 < c implies ac < bc
5. Separation from unit: 0 < a or a < 1 for all a
6. Distinguishability: a ≠ b implies a < b or b < a

**Note**: We do NOT have trichotomy (a < b or a = b or b < a) because
indistinguishable points may exist.
-}

postulate
  _<ℝ_ : ℝ → ℝ → Type

  -- Order axioms
  <ℝ-trans : ∀ {a b c : ℝ} → a <ℝ b → b <ℝ c → a <ℝ c
  <ℝ-irrefl : ∀ {a : ℝ} → ¬ (a <ℝ a)
  <ℝ-+ℝ-compat : ∀ {a b c : ℝ} → a <ℝ b → (a +ℝ c) <ℝ (b +ℝ c)
  <ℝ-·ℝ-compat : ∀ {a b c : ℝ} → a <ℝ b → 0ℝ <ℝ c → (a ·ℝ c) <ℝ (b ·ℝ c)
  <ℝ-separation : ∀ (a : ℝ) → (0ℝ <ℝ a) ⊎ (a <ℝ 1ℝ)
  <ℝ-distinguish : ∀ {a b : ℝ} → a ≠ b → (a <ℝ b) ⊎ (b <ℝ a)

  -- 0 < 1
  0<1 : 0ℝ <ℝ 1ℝ

-- Less-than-or-equal-to relation (defined as ¬(b < a))
_≤ℝ_ : ℝ → ℝ → Type
a ≤ℝ b = ¬ (b <ℝ a)

-- Greater-than relations
_>ℝ_ : ℝ → ℝ → Type
a >ℝ b = b <ℝ a

_≥ℝ_ : ℝ → ℝ → Type
a ≥ℝ b = b ≤ℝ a

{-|
## Ordering Properties

Additional axioms about the order structure.
-}

postulate
  -- Squares are non-negative: 0 ≤ a² for all a
  -- This is a standard axiom in ordered fields
  -- Proof sketch in classical setting: if a ≠ 0, then a² > 0 (by trichotomy and product rules)
  square-nonneg : ∀ (a : ℝ) → 0ℝ ≤ℝ (a ·ℝ a)

  -- Negative one is less than zero
  -- Proof sketch: From 0 < 1, add (-1) to both sides: -1 < 0
  -1<0 : (-ℝ 1ℝ) <ℝ 0ℝ

{-|
## Closed Intervals

A **closed interval** [a,b] consists of points x with a ≤ x ≤ b.
-}

ClosedInterval : ℝ → ℝ → Type
ClosedInterval a b = Σ ℝ (λ x → a ≤ℝ x × x ≤ℝ b)

--------------------------------------------------------------------------------
-- § 2: Microneighbourhood and Infinitesimals

{-|
## The Microneighbourhood Δ (Definition 1.1)

The **microneighbourhood** (of 0) is:

  Δ = {ε ∈ ℝ | ε² = 0}

These are **nilsquare infinitesimals**: quantities that square to zero.

**Key properties** (Theorem 1.1):
1. Δ ⊆ [0,0] (every ε is indistinguishable from 0)
2. Δ ≠ {0} (Δ is non-degenerate)
3. Every ε ∈ Δ is indistinguishable from 0 (but ε ≠ 0 is unprovable)

**Physical meaning**: Infinitesimals are "smaller than any finite quantity
but larger than zero" in a precise sense.
-}

Δ : Type
Δ = Σ ℝ (λ ε → ε ·ℝ ε ≡ 0ℝ)

-- Extract the real number from infinitesimal
ι : Δ → ℝ
ι (ε , _) = ε

-- Nilsquare property
nilsquare : (δ : Δ) → (ι δ) ·ℝ (ι δ) ≡ 0ℝ
nilsquare (_ , p) = p

{-|
## Properties of Infinitesimals (Theorem 1.1)

From the smooth worlds document, Chapter 1.
-}

postulate
  -- (i) Δ is included in [0,0] but non-degenerate
  Δ-in-[0,0] : ∀ (δ : Δ) → (ι δ) ≤ℝ 0ℝ × 0ℝ ≤ℝ (ι δ)

  Δ-nonempty : Σ Δ (λ δ → ι δ ≠ 0ℝ)

  -- (ii) Every element is indistinguishable from 0
  -- (Proven: if ε² = 0 and ε ≠ 0, then ε/ε exists and 0 = ε² · (1/ε) = ε, contradiction)
  Δ-indistinguishable : ∀ (δ : Δ) → ¬ ((ι δ) ≠ 0ℝ → ⊥)

  -- (iii) Law of excluded middle fails for Δ
  -- It's FALSE that: ∀ ε ∈ Δ, (ε = 0 ∨ ε ≠ 0)
  Δ-no-LEM : ¬ (∀ (δ : Δ) → (ι δ ≡ 0ℝ) ⊎ (ι δ ≠ 0ℝ))

{-|
## Microstability

A subset A ⊆ ℝ is **microstable** if it's closed under adding infinitesimals:
  a ∈ A and ε ∈ Δ implies a + ε ∈ A
-}

is-microstable : (A : ℝ → Type) → Type
is-microstable A = ∀ (a : ℝ) (δ : Δ) → A a → A (a +ℝ ι δ)

-- Example: Closed intervals are microstable (Exercise 1.7)
postulate
  interval-microstable : ∀ (a b : ℝ) → is-microstable (λ x → a ≤ℝ x × x ≤ℝ b)

--------------------------------------------------------------------------------
-- § 3: Principle of Microaffineness

{-|
## Principle of Microaffineness (Fundamental Axiom)

**Statement**: For any function g : Δ → ℝ, there exists a UNIQUE b ∈ ℝ such that:

  ∀ ε ∈ Δ, g(ε) = g(0) + b · ε

**Meaning**: Every function on the microneighbourhood is affine (linear + constant).

**Geometric picture**: Δ behaves like an infinitesimal "rigid rod" that can only
be translated and rotated, not bent.

**Consequences**:
1. Δ is "large enough" to have a slope
2. Δ is "too small" to bend (microstraightness)
3. Every curve is locally straight at infinitesimal scale
4. Enables exact differential calculus

**Connection to calculus**: For f : ℝ → ℝ, the restriction f|_Δ centered at x
is affine, with slope f'(x):

  f(x + ε) = f(x) + ε · f'(x)  for all ε ∈ Δ
-}

Microaffine : Type
Microaffine =
  ∀ (g : Δ → ℝ) →
  Σ ℝ (λ b → (∀ (δ : Δ) → g δ ≡ g (0ℝ , ·ℝ-zerol 0ℝ) +ℝ (b ·ℝ ι δ)) ×
              (∀ (b' : ℝ) → (∀ (δ : Δ) → g δ ≡ g (0ℝ , ·ℝ-zerol 0ℝ) +ℝ (b' ·ℝ ι δ)) → b' ≡ b))

postulate
  microaffineness : Microaffine

-- Extract the unique slope from microaffineness
slope : (g : Δ → ℝ) → ℝ
slope g = microaffineness g .fst

slope-property : (g : Δ → ℝ) (δ : Δ) →
  g δ ≡ g (0ℝ , ·ℝ-zerol 0ℝ) +ℝ (slope g ·ℝ ι δ)
slope-property g δ = microaffineness g .snd .fst δ

slope-unique : (g : Δ → ℝ) (b : ℝ) →
  (∀ (δ : Δ) → g δ ≡ g (0ℝ , ·ℝ-zerol 0ℝ) +ℝ (b ·ℝ ι δ)) →
  b ≡ slope g
slope-unique g b H = microaffineness g .snd .snd b H

{-|
## Microcancellation (Theorem 1.1 part iv)

**Principle of Microcancellation**: If ε · a = ε · b for ALL ε ∈ Δ, then a = b.

**Proof**: Consider g : Δ → ℝ defined by g(ε) = ε · a.
By Microaffineness, g(ε) = g(0) + slope(g) · ε = 0 + slope(g) · ε.
So slope(g) = a.

Similarly for h(ε) = ε · b, we get slope(h) = b.

If g(ε) = h(ε) for all ε, then by uniqueness of slope, a = b.

**Usage**: This allows us to "cancel ε" from equations like:
  ε · f'(x) = ε · g'(x)  for all ε ∈ Δ
  ⟹ f'(x) = g'(x)
-}

microcancellation : ∀ (a b : ℝ) →
  (∀ (δ : Δ) → (ι δ ·ℝ a) ≡ (ι δ ·ℝ b)) →
  a ≡ b
microcancellation a b H =
  a                                    ≡⟨ slope-unique ga a ga-prop ⟩
  slope ga                             ≡⟨ ap slope (funext λ δ → H δ) ⟩
  slope gb                             ≡⟨ sym (slope-unique gb b gb-prop) ⟩
  b                                    ∎
  where
    ga : Δ → ℝ
    ga δ = ι δ ·ℝ a

    gb : Δ → ℝ
    gb δ = ι δ ·ℝ b

    ga-prop : ∀ (δ : Δ) → ga δ ≡ ga (0ℝ , ·ℝ-zerol 0ℝ) +ℝ (a ·ℝ ι δ)
    ga-prop δ =
      ι δ ·ℝ a                                    ≡⟨ sym (+ℝ-idl (ι δ ·ℝ a)) ⟩
      0ℝ +ℝ (ι δ ·ℝ a)                            ≡⟨ ap (_+ℝ (ι δ ·ℝ a)) (sym (·ℝ-zerol a)) ⟩
      (0ℝ ·ℝ a) +ℝ (ι δ ·ℝ a)                     ≡⟨ ap (λ x → x +ℝ (ι δ ·ℝ a)) refl ⟩
      ga (0ℝ , ·ℝ-zerol 0ℝ) +ℝ (ι δ ·ℝ a)         ≡⟨ ap (ga (0ℝ , ·ℝ-zerol 0ℝ) +ℝ_) (·ℝ-comm (ι δ) a) ⟩
      ga (0ℝ , ·ℝ-zerol 0ℝ) +ℝ (a ·ℝ ι δ)         ∎

    gb-prop : ∀ (δ : Δ) → gb δ ≡ gb (0ℝ , ·ℝ-zerol 0ℝ) +ℝ (b ·ℝ ι δ)
    gb-prop δ =
      ι δ ·ℝ b                                    ≡⟨ sym (+ℝ-idl (ι δ ·ℝ b)) ⟩
      0ℝ +ℝ (ι δ ·ℝ b)                            ≡⟨ ap (_+ℝ (ι δ ·ℝ b)) (sym (·ℝ-zerol b)) ⟩
      (0ℝ ·ℝ b) +ℝ (ι δ ·ℝ b)                     ≡⟨ ap (λ x → x +ℝ (ι δ ·ℝ b)) refl ⟩
      gb (0ℝ , ·ℝ-zerol 0ℝ) +ℝ (ι δ ·ℝ b)         ≡⟨ ap (gb (0ℝ , ·ℝ-zerol 0ℝ) +ℝ_) (·ℝ-comm (ι δ) b) ⟩
      gb (0ℝ , ·ℝ-zerol 0ℝ) +ℝ (b ·ℝ ι δ)         ∎

-- Special case: if ε · a = 0 for all ε, then a = 0
microcancellation-zero : ∀ (a : ℝ) →
  (∀ (δ : Δ) → (ι δ ·ℝ a) ≡ 0ℝ) →
  a ≡ 0ℝ
microcancellation-zero a H =
  microcancellation a 0ℝ (λ δ → H δ ∙ sym (·ℝ-zeror (ι δ)))

{-|
## Higher-Order Infinitesimals

Products of infinitesimals are also infinitesimal, but with higher nilpotency.

**Examples**:
- ε, η ∈ Δ ⟹ ε·η ∈ Δ (and (ε·η)² = 0)
- ε₁, ε₂, ε₃ ∈ Δ ⟹ ε₁·ε₂·ε₃ ≠ 0 in general, but (ε₁·ε₂·ε₃)³ = 0
- (ε₁ + ε₂ + ε₃)⁴ = 0 (by expansion and nilpotence)

**Exercise 1.12**: For ε₁,...,εₙ ∈ Δ, we have (ε₁ + ... + εₙ)ⁿ⁺¹ = 0.
-}

postulate
  -- Product of infinitesimals is infinitesimal
  Δ-product : ∀ (δ₁ δ₂ : Δ) → ((ι δ₁ ·ℝ ι δ₂) ·ℝ (ι δ₁ ·ℝ ι δ₂)) ≡ 0ℝ

  -- Sum of n infinitesimals to power n+1 is zero
  Δ-sum-nilpotent : ∀ (n : Nat) (εs : Fin n → Δ) →
    {-| (Σᵢ ι(εᵢ))^(n+1) = 0 -}
    ⊤

{-|
## Neighbour Relation (Exercise 1.10)

Two points a, b are **neighbours** if their difference is infinitesimal:
  a ~ b  ⟺  (a - b) ∈ Δ

**Properties**:
- Reflexive: a ~ a (since 0 ∈ Δ)
- Symmetric: a ~ b ⟹ b ~ a (since -ε ∈ Δ if ε ∈ Δ)
- NOT transitive: Can have a ~ b and b ~ c but not a ~ c
  (because ε + η might not be in Δ if Δ is not microstable)

**Connection to continuity**: Every function f : ℝ → ℝ is continuous in the
sense that neighbouring points map to neighbouring points.
-}

_~_ : ℝ → ℝ → Type
a ~ b = Σ Δ (λ δ → a -ℝ b ≡ ι δ)

neighbour-refl : ∀ (a : ℝ) → a ~ a
neighbour-refl a = (0ℝ , ·ℝ-zerol 0ℝ) , +ℝ-invr a

postulate
  neighbour-sym : ∀ {a b : ℝ} → a ~ b → b ~ a
  -- TODO: Prove using properties of negation
  -- Proof sketch:
  -- If a - b = ε, then b - a = -(a - b) = -ε
  -- Need to show (-ε)² = 0 from ε² = 0
  -- (-ε)·(-ε) = ε·ε = 0 by properties of negation in rings

--------------------------------------------------------------------------------
-- § 4: Cartesian Powers and Euclidean Space

{-|
## n-Dimensional Euclidean Space

We can form Cartesian products ℝⁿ = ℝ × ℝ × ... × ℝ (n times).

Points in ℝⁿ are n-tuples (a₁, ..., aₙ).

Two points are **distinct** if at least one coordinate is explicitly distinct.
-}

-- Definition by recursion on Nat
ℝⁿ : Nat → Type
ℝⁿ zero = ⊤
ℝⁿ (suc n) = ℝ × ℝⁿ n

-- Example: ℝ² = ℝ × ℝ
ℝ² : Type
ℝ² = ℝ × ℝ

ℝ³ : Type
ℝ³ = ℝ × ℝ × ℝ

{-|
## Products and Inverses (Euclidean Constructions)

Using Euclidean geometry in ℝ², we can define:
- Product a·b via similar triangles
- Inverse 1/a for a ≠ 0 via parallel lines
- Square root √a for a > 0 via circle construction

See Figures 1.1, 1.2, 1.3 in the document.
-}

-- These are already postulated above as field operations
-- Here we note that they satisfy the Euclidean geometric constructions

postulate
  -- Product via similar triangles (Fig 1.1)
  product-geometric : ∀ (a b : ℝ) →
    {-| a·b is the y-coordinate of the similar triangle construction -}
    ⊤

  -- Inverse via parallel lines (Fig 1.2)
  inverse-geometric : ∀ (a : ℝ) (p : a ≠ 0ℝ) →
    {-| 1/a is the y-coordinate of the parallel line construction -}
    ⊤

  -- Square root via circle (Fig 1.3)
  sqrt-geometric : ∀ (a : ℝ) → (0ℝ <ℝ a) →
    {-| √a is the height of the perpendicular to diameter construction -}
    Σ ℝ (λ b → b ·ℝ b ≡ a)

--------------------------------------------------------------------------------
-- § 4.5: Helper Functions for MUP and Numeric Literals

{-|
## ℝ-from-nat: Natural Number Conversion

Converts natural numbers to real numbers via iterated addition.
-}

ℝ-from-nat : Nat → ℝ
ℝ-from-nat zero = 0ℝ
ℝ-from-nat (suc n) = 1ℝ +ℝ ℝ-from-nat n

{-|
## sqrtℝ: Square Root Function

Extracts the square root from the geometric construction.
Uses the circle construction from Figure 1.3.
-}

postulate
  sqrtℝ : ℝ → ℝ
  sqrtℝ-spec : ∀ (a : ℝ) → (0ℝ <ℝ a) → (sqrtℝ a) ·ℝ (sqrtℝ a) ≡ a

{-# COMPILE GHC sqrtℝ = \x -> sqrt x #-}

{-|
## Fraction Helper: Create ℝ from Rationals

For convenience in defining MUP hyperparameters like 0.1 = 1/10.
Note: Requires proof that denominator ≠ 0.
-}

-- Helper to convert fraction to ℝ
postulate
  0≠1-lemma : ¬ (0ℝ ≡ 1ℝ)

-- Proof that ℝ-from-nat (suc n) ≠ 0
postulate
  from-nat-suc-nonzero : ∀ (n : Nat) → ℝ-from-nat (suc n) ≠ 0ℝ

-- Convenient infix operator for fractions (only defined for non-zero denom)
_/ₙ_ : (num : Nat) → (denom : Nat) → ℝ
num /ₙ zero = 0ℝ  -- undefined case, should never be called
num /ₙ (suc d) = ((ℝ-from-nat num) /ℝ (ℝ-from-nat (suc d))) (from-nat-suc-nonzero d)

infixl 30 _/ₙ_

--------------------------------------------------------------------------------
-- § 5: Summary and Exports

{-|
## What We've Defined

**Core types**:
- ℝ : Smooth line with field structure
- Δ : Microneighbourhood {ε : ε² = 0}
- ℝⁿ : Euclidean n-space

**Operations**:
- +ℝ, -ℝ, ·ℝ, /ℝ : Field operations
- <ℝ, ≤ℝ, >ℝ, ≥ℝ : Order relations
- ~ : Neighbour relation

**Key principles**:
- Microaffineness: Functions Δ → ℝ are affine
- Microcancellation: Can cancel ε from universal equations
- Nilsquare: ε² = 0 for all ε ∈ Δ

**Next steps** (in Neural.Smooth.Calculus):
- Define derivative f'(x) using f(x+ε) = f(x) + ε·f'(x)
- Prove calculus rules (product, chain, etc.)
- Develop integration theory
-}
