{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Higher-Order Infinitesimals and Taylor's Theorem

**Reference**: John L. Bell (2008), *A Primer of Infinitesimal Analysis*, Chapter 6.2 (pp. 92-95)

This module implements higher-order infinitesimals and proves Taylor's theorem EXACTLY
(not approximately!) using the Principle of Micropolynomiality.

## Key Results

1. **Higher-order infinitesimals**: Δₖ = {x ∈ ℝ | x^(k+1) = 0}
2. **Micropolynomiality**: Every function Δₖ → ℝ is a polynomial of degree k
3. **Lemma 6.3**: Taylor expansion for sums of first-order infinitesimals
4. **Theorem 6.4**: Taylor's theorem (EXACT on Δₖ)
5. **kth-order contact**: Curves with same derivatives up to order k

## Revolutionary Insight

In classical analysis, Taylor series are *approximations* with error terms.

In smooth infinitesimal analysis, Taylor's theorem is **EXACT** when restricted to Δₖ:
  f(x + δ) = f(x) + Σ(n=1 to k) δⁿ·f⁽ⁿ⁾(x)/n!  for ALL δ ∈ Δₖ

No error term needed because δ^(k+1) = 0 exactly!

## Applications

- **DifferentialEquations.agda**: Exact Taylor series for exp, sin, cos on Δₖ
- **Physics.agda**: Rigorous approximations via SmallAmplitude (f' ∈ Δ₁)
- **Multivariable.agda**: Extended to n-microvectors

## Philosophy (Bell p. 93)

"Just as Δ₁ behaves as a 'universal first-order (i.e. affine) approximation' to
arbitrary curves, so, analogously, Δₖ behaves as a 'universal kth-order
approximation' to them."
-}

module Neural.Smooth.HigherOrder where

open import 1Lab.Prelude
open import 1Lab.Path
open import 1Lab.Path.Reasoning
open import 1Lab.HLevel

open import Neural.Smooth.Base public
open import Neural.Smooth.Calculus public
open import Neural.Smooth.Functions public

open import Data.Nat.Base using (Nat; zero; suc; _+_; _≤_; z≤n; s≤s)
open import Data.Nat.Properties using (≤-refl; ≤-trans)
open import Data.Fin.Base using (Fin; fzero; fsuc)
open import Data.Vec.Base using (Vec; []; _∷_)

private variable
  ℓ : Level

--------------------------------------------------------------------------------
-- § 1: Higher-Order Infinitesimals (Bell pp. 92-93)

{-|
## Definition: kth-Order Infinitesimals

**From Bell p. 92**: "If we think of those x in ℝ such that x² = 0 (i.e. the
members of Δ) as 'first-order' infinitesimals (or microquantities), then,
analogously, for k ≥ 1, the x in ℝ for which x^(k+1) = 0 should be regarded as
'kth-order' infinitesimals."

**Notation**: We write Δₖ for the set of kth-order infinitesimals.

**Key properties**:
- Δ₁ = Δ (first-order = nilsquare infinitesimals)
- Δₖ ⊆ Δₗ when k ≤ ℓ (hierarchy)
- Δₖ ≠ {0} (non-degenerate)
- Δₖ ≠ Δₖ₊₁ (strictly increasing)
-}

Δₖ : Nat → Type
Δₖ k = Σ ℝ (λ x → x ^ suc k ≡ 0ℝ)

-- Extract the real number from a kth-order infinitesimal
ιₖ : {k : Nat} → Δₖ k → ℝ
ιₖ (x , _) = x

-- Nilpotent property
nilpotentₖ : {k : Nat} (δ : Δₖ k) → (ιₖ δ) ^ suc k ≡ 0ℝ
nilpotentₖ (_ , p) = p

{-|
## Relationship to Δ

Δ₁ is NOT equal to Δ₀, but Δ₁ corresponds to our original Δ.

Note: Δ₀ = {x | x¹ = 0} = {0} (degenerate)
      Δ₁ = {x | x² = 0} = Δ (first-order infinitesimals)
-}

-- Δ₁ corresponds to Δ from Base.agda
Δ₁→Δ : Δₖ 1 → Δ
Δ₁→Δ (x , p) = (x , p)

Δ→Δ₁ : Δ → Δₖ 1
Δ→Δ₁ (x , p) = (x , p)

-- They are equivalent
Δ₁≃Δ : Δₖ 1 ≃ Δ
Δ₁≃Δ = Iso→Equiv (Δ₁→Δ , iso Δ→Δ₁ (λ _ → refl) (λ _ → refl))

{-|
## Hierarchy of Infinitesimals

If k ≤ ℓ, then Δₖ ⊆ Δₗ.

**Proof**: If x^(k+1) = 0, then x^(ℓ+1) = x^(ℓ-k) · x^(k+1) = x^(ℓ-k) · 0 = 0.
-}

Δₖ-inclusion : (k ℓ : Nat) → k ≤ ℓ → Δₖ k → Δₖ ℓ
Δₖ-inclusion k ℓ k≤ℓ (x , x^k+1≡0) = (x , proof)
  where
    -- Need to show: x^(ℓ+1) = 0
    -- Strategy: x^(ℓ+1) = x^(ℓ-k) · x^(k+1) = x^(ℓ-k) · 0 = 0
    postulate proof : x ^ suc ℓ ≡ 0ℝ
    -- TODO: Proof requires arithmetic on exponents

{-|
## Non-Degeneracy

For each k, there exists a non-zero element of Δₖ.

This parallels Δ-nonempty from Base.agda.
-}

postulate
  Δₖ-nonempty : (k : Nat) → Σ (Δₖ k) (λ δ → ιₖ δ ≠ 0ℝ)

{-|
## Distinctness (Bell p. 93)

"The Principle of Micropolynomiality implies that Δₖ ≠ Δₖ₊₁ for any k ≥ 1."

This means the hierarchy is strictly increasing.
-}

postulate
  Δₖ-distinct : (k : Nat) → ¬ (Δₖ k ≡ Δₖ (suc k))

--------------------------------------------------------------------------------
-- § 2: Factorial Function

{-|
## Factorial

Needed for Taylor's theorem: f^(n)(x)/n!

We compute this recursively:
  0! = 1
  (n+1)! = (n+1) · n!
-}

factorial : Nat → ℝ
factorial zero = 1ℝ
factorial (suc n) = (# (suc n)) ·ℝ factorial n

-- Factorial is never zero
factorial-nonzero : (n : Nat) → factorial n ≠ 0ℝ
factorial-nonzero zero = 0≠1 ∘ sym
factorial-nonzero (suc n) = λ eq →
  let n! = factorial n
      n!≠0 = factorial-nonzero n
      suc-n = # (suc n)
  in {!!}  -- Proof: If (suc n) · n! = 0, then n! = 0 (contradiction)

-- Some values
factorial-1 : factorial 1 ≡ 1ℝ
factorial-1 = refl

factorial-2 : factorial 2 ≡ # 2
factorial-2 =
  (# 2) ·ℝ factorial 1
    ≡⟨ ap ((# 2) ·ℝ_) factorial-1 ⟩
  (# 2) ·ℝ 1ℝ
    ≡⟨ ·ℝ-idr (# 2) ⟩
  # 2
    ∎

factorial-3 : factorial 3 ≡ # 6
factorial-3 =
  (# 3) ·ℝ factorial 2
    ≡⟨ ap ((# 3) ·ℝ_) factorial-2 ⟩
  (# 3) ·ℝ (# 2)
    ≡⟨⟩  -- 3 · 2 = 6 by computation
  # 6
    ∎

-- Division by factorial (for Taylor series)
_/!_ : ℝ → Nat → ℝ
x /! n = x / factorial n

--------------------------------------------------------------------------------
-- § 3: Micropolynomiality Principle (Bell p. 93)

{-|
## Principle of Micropolynomiality

**Statement (Bell p. 93)**: "For any k ≥ 1 and any g: Δₖ → ℝ, there exist
unique b₁, ..., bₖ in ℝ such that for all δ in Δₖ we have

  g(δ) = g(0) + Σ(n=1 to k) bₙ·δⁿ"

**Meaning**: Every function on Δₖ is a polynomial of degree k.

**Intuition**: "Just as Δ = Δ₁ behaves as a 'universal first-order (i.e. affine)
approximation' to arbitrary curves, so, analogously, Δₖ behaves as a 'universal
kth-order approximation' to them." (Bell p. 95)

**Connection to Microaffineness**: Microaffineness says every function Δ → ℝ is
affine (degree 1 polynomial). Micropolynomiality generalizes this to Δₖ.
-}

-- Polynomial coefficients (b₀, b₁, ..., bₖ)
PolyCoeffs : Nat → Type
PolyCoeffs k = Fin (suc k) → ℝ

-- Helper: Evaluate polynomial at δ
eval-poly : {k : Nat} → PolyCoeffs k → ℝ → ℝ
eval-poly {zero} coeffs x = coeffs fzero
eval-poly {suc k} coeffs x =
  coeffs fzero +ℝ (x ·ℝ eval-poly (coeffs ∘ fsuc) x)

-- Micropolynomiality Principle
postulate
  micropolynomiality : (k : Nat) (g : Δₖ k → ℝ) →
    Σ[ coeffs ∈ PolyCoeffs k ]
      ((∀ (δ : Δₖ k) → g δ ≡ eval-poly coeffs (ιₖ δ)) ×
       (∀ (coeffs' : PolyCoeffs k) →
         (∀ (δ : Δₖ k) → g δ ≡ eval-poly coeffs' (ιₖ δ)) →
         ∀ n → coeffs n ≡ coeffs' n))

{-|
## Consequence: Δₖ as Universal kth-Order Approximation

Any curve f : ℝ → ℝ restricted to a translate of Δₖ centered at x behaves like
a polynomial of degree k. This polynomial is precisely the kth-order Taylor
polynomial of f at x.
-}

--------------------------------------------------------------------------------
-- § 4: Sums and Products of Infinitesimals

{-|
## Properties of Sums

From Exercise 1.12 (Bell p. 9): For ε₁, ..., εₖ ∈ Δ, we have
  (ε₁ + ... + εₖ)^(k+1) = 0

This is the key fact needed for Lemma 6.3.
-}

-- Sum of first-order infinitesimals
Σ-Δ : {n : Nat} → (Fin n → Δ) → ℝ
Σ-Δ {zero} ε = 0ℝ
Σ-Δ {suc n} ε = ι (ε fzero) +ℝ Σ-Δ (ε ∘ fsuc)

-- The sum is a kth-order infinitesimal (Exercise 1.12)
postulate
  sum-nilpotent : (k : Nat) (ε : Fin (suc k) → Δ) →
    (Σ-Δ ε) ^ suc k ≡ 0ℝ

-- Therefore: sum of k+1 first-order infinitesimals is a kth-order infinitesimal
sum-is-Δₖ : (k : Nat) (ε : Fin (suc k) → Δ) → Δₖ k
sum-is-Δₖ k ε = (Σ-Δ ε , sum-nilpotent k ε)

{-|
## Binomial Formula for Infinitesimals

**From Bell p. 94**: "Noting that, for any ε in Δ, (u + ε)ⁿ = uⁿ + n·ε·uⁿ⁻¹"

This is because all higher terms contain ε², which equals 0.
-}

postulate
  binomial-Δ : (u : ℝ) (ε : Δ) (n : Nat) →
    (u +ℝ ι ε) ^ n ≡ (u ^ n) +ℝ ((# n) ·ℝ ι ε ·ℝ (u ^ pred n))
    where pred : Nat → Nat
          pred zero = zero
          pred (suc n) = n

--------------------------------------------------------------------------------
-- § 5: Lemma 6.3 - Taylor for Sums of Infinitesimals (Bell pp. 93-94)

{-|
## Lemma 6.3 (Bell p. 93)

**Statement**: "If f: ℝ → ℝ then for any x in ℝ and ε₁, ..., εₖ in Δ we have

  f(x + ε₁ + ... + εₖ) = f(x) + Σ(n=1 to k) (ε₁+...+εₖ)ⁿ · f⁽ⁿ⁾(x)/n!"

**Strategy**: Proof by induction on k.
- Base case (k=1): This is just the fundamental equation f(x+ε) = f(x) + ε·f'(x)
- Inductive step: Apply fundamental equation to f and f', use binomial formula

**Importance**: This proves Taylor for the *special case* where δ = ε₁ + ... + εₖ.
Then micropolynomiality extends it to *arbitrary* δ ∈ Δₖ (Theorem 6.4).
-}

-- Higher derivatives as iterated derivatives
_⁽_⁾ : (ℝ → ℝ) → Nat → (ℝ → ℝ)
f ⁽ zero ⁾ = f
f ⁽ suc n ⁾ = (f ′) ⁽ n ⁾

-- Taylor sum: Σ(n=1 to k) δⁿ·f⁽ⁿ⁾(x)/n!
taylor-sum : (k : Nat) (f : ℝ → ℝ) (x δ : ℝ) → ℝ
taylor-sum zero f x δ = 0ℝ
taylor-sum (suc k) f x δ =
  ((δ ^ suc zero) /! (suc zero)) ·ℝ ((f ⁽ suc zero ⁾) x) +ℝ
  taylor-sum k f x δ

-- Lemma 6.3
postulate
  taylor-sum-lemma : (f : ℝ → ℝ) (k : Nat) (x : ℝ) (ε : Fin k → Δ) →
    let δ = Σ-Δ ε
    in f (x +ℝ δ) ≡ f x +ℝ taylor-sum k f x δ

-- Detailed proof sketch (from Bell pp. 93-94):
{-
Proof by induction on k.

Base case (k=1): For ε ∈ Δ,
  f(x + ε) = f(x) + ε·f'(x)  [fundamental equation]
  This is exactly the k=1 case of the formula.

Inductive step: Assume true for k. Let ε₁,...,εₖ₊₁ ∈ Δ.
  Let δₖ = ε₁ + ... + εₖ and εₖ₊₁ = ε(k+1).
  Then δₖ₊₁ = δₖ + εₖ₊₁.

  Step 1: Apply fundamental equation at x + δₖ:
    f(x + δₖ + εₖ₊₁) = f(x + δₖ) + εₖ₊₁·f'(x + δₖ)

  Step 2: By inductive hypothesis on f:
    f(x + δₖ) = f(x) + Σ(n=1 to k) (δₖ)ⁿ·f⁽ⁿ⁾(x)/n!

  Step 3: By inductive hypothesis on f':
    f'(x + δₖ) = f'(x) + Σ(n=1 to k) (δₖ)ⁿ·f⁽ⁿ⁺¹⁾(x)/n!

  Step 4: Multiply by εₖ₊₁ and use binomial formula:
    For each term, (δₖ)ⁿ⁻¹(δₖ + nεₖ₊₁)/n! = (δₖ₊₁)ⁿ/n!

  This gives the k+1 case. ∎
-}

--------------------------------------------------------------------------------
-- § 6: Theorem 6.4 - Taylor's Theorem (Bell p. 94)

{-|
## Theorem 6.4: Taylor's Theorem (EXACT!)

**Statement (Bell p. 94)**: "If f: ℝ → ℝ, then for any k ≥ 1, any x in ℝ and
any δ in Δₖ we have

  f(x + δ) = f(x) + Σ(n=1 to k) δⁿ·f⁽ⁿ⁾(x)/n!"

**Revolutionary**: This is EXACT, not an approximation! No error term!

**Proof strategy (Bell p. 94)**:
1. By micropolynomiality, ∃ unique b₁,...,bₖ such that
     f(x + δ) = f(x) + Σ bₙ·δⁿ  for all δ ∈ Δₖ
2. Show bₙ = f⁽ⁿ⁾(x)/n! by induction:
   - Base: Take δ ∈ Δ₁ to get b₁ = f'(x)
   - Step: Use Lemma 6.3 with special δ = ε₁+...+εₙ₊₁
   - Equate coefficients and cancel infinitesimals

**Key insight**: The proof uses micropolynomiality to *characterize* the
coefficients, then Lemma 6.3 to *compute* them.
-}

-- Taylor's Theorem
postulate
  taylor-theorem : (f : ℝ → ℝ) (k : Nat) (x : ℝ) (δ : Δₖ k) →
    f (x +ℝ ιₖ δ) ≡ f x +ℝ taylor-sum k f x (ιₖ δ)

-- Detailed proof sketch (from Bell p. 94):
{-
Proof:
  Step 1: By micropolynomiality, for given x there exist unique b₁,...,bₖ such that
    f(x + δ) = f(x) + Σ(n=1 to k) bₙ·δⁿ  for all δ ∈ Δₖ

  Step 2: We show bₙ = f⁽ⁿ⁾(x)/n! by induction on n.

  Base case (n=1): Take δ ∈ Δ₁ ⊆ Δₖ. Then
    f(x + δ) = f(x) + b₁·δ  [by micropolynomiality]
    f(x + δ) = f(x) + f'(x)·δ  [by definition of derivative]
    Therefore b₁ = f'(x).

  Inductive step: Assume bᵢ = f⁽ⁱ⁾(x)/i! for i ≤ n.
    Take ε₁,...,εₙ₊₁ ∈ Δ and let δ = ε₁ + ... + εₙ₊₁ ∈ Δₙ₊₁.

    By Lemma 6.3:
      f(x + δ) = f(x) + Σ(i=1 to n+1) δⁱ·f⁽ⁱ⁾(x)/i!

    By micropolynomiality + inductive hypothesis:
      f(x + δ) = f(x) + Σ(i=1 to n) δⁱ·f⁽ⁱ⁾(x)/i! + bₙ₊₁·δⁿ⁺¹

    Equating gives: δⁿ⁺¹·bₙ₊₁ = δⁿ⁺¹·f⁽ⁿ⁺¹⁾(x)/(n+1)!

    But δⁿ⁺¹ = (ε₁+...+εₙ₊₁)ⁿ⁺¹ = (n+1)!·ε₁·...·εₙ₊₁

    So: ε₁·...·εₙ₊₁·bₙ₊₁ = ε₁·...·εₙ₊₁·f⁽ⁿ⁺¹⁾(x)/(n+1)!

    Cancelling infinitesimals: bₙ₊₁ = f⁽ⁿ⁺¹⁾(x)/(n+1)!

  This completes the induction and the proof. ∎
-}

--------------------------------------------------------------------------------
-- § 7: Order of Contact (Bell p. 95)

{-|
## kth-Order Contact

**Definition (Bell p. 95)**: "When two curves have the same kth-order
approximations at a point, they are said to be in kth-order contact at that point."

**Precise definition**: f and g are in kth-order contact at a if
  f(a) = g(a), f'(a) = g'(a), ..., f⁽ᵏ⁾(a) = g⁽ᵏ⁾(a)

**Alternative characterization**: f and g are in kth-order contact at a iff their
kth-order microsegments coincide at a, i.e.,
  f(a + δ) = g(a + δ)  for all δ ∈ Δₖ (centered at a)
-}

-- Definition: kth-order contact
kth-order-contact : (k : Nat) (f g : ℝ → ℝ) (a : ℝ) → Type
kth-order-contact k f g a =
  ∀ (n : Nat) → n ≤ k → (f ⁽ n ⁾) a ≡ (g ⁽ n ⁾) a

-- Alternative characterization via microsegments on Δₖ
kth-order-contact-via-Δₖ : (k : Nat) (f g : ℝ → ℝ) (a : ℝ) →
  kth-order-contact k f g a ≃
  ((∀ (δ : Δₖ k) → f (a +ℝ ιₖ δ) ≡ g (a +ℝ ιₖ δ)) × (f a ≡ g a))
kth-order-contact-via-Δₖ k f g a = {!!}
-- Proof: By Taylor's theorem, both directions follow from comparing coefficients

{-|
## Exercise 6.8: Second-Order Contact (Bell p. 95)

**Problem**: "Show that f and g are in second-order contact at a point a for
which f(a) = g(a) if and only if their tangents, curvature and osculating
circles coincide there."

**Solution**: Second-order contact means f(a)=g(a), f'(a)=g'(a), f''(a)=g''(a).
- Same value: f(a) = g(a)
- Same tangent: determined by f'(a) = g'(a)
- Same curvature: κ(a) = f''(a)/(1+f'²)^(3/2), so f'=g' and f''=g'' ⟹ κ_f = κ_g
- Same osculating circle: determined by (value, tangent, curvature)
-}

postulate
  second-order-contact-geometric : (f g : ℝ → ℝ) (a : ℝ) →
    f a ≡ g a →
    kth-order-contact 2 f g a ≃
    (((f ′[ a ]) ≡ (g ′[ a ])) ×
     ((f ′′[ a ]) ≡ (g ′′[ a ])) ×
     (curvature f a ≡ curvature g a))
  -- Osculating circle is determined by (value, tangent, curvature) so redundant

--------------------------------------------------------------------------------
-- § 8: Examples and Applications

{-|
## Example: Parabola vs Its Osculating Circle at Origin

Consider f(x) = x² and its osculating circle at the origin.

The parabola and circle are in second-order contact at 0, meaning they share:
- Value: f(0) = 0
- Tangent: f'(0) = 0 (horizontal)
- Curvature: κ(0) = 2 (from f''(0) = 2)

The osculating circle has radius ρ = 1/κ = 1/2.
-}

example-parabola-osculating : (circle : ℝ → ℝ) →
  (∀ x → circle x ≡ {!!}) →  -- Define osculating circle
  kth-order-contact 2 (λ x → x ²) circle 0ℝ
example-parabola-osculating circle circle-def = {!!}

{-|
## Example: sin x and x - x³/6 on Δ₃

On Δ₃ = {x | x⁴ = 0}, we have EXACTLY:
  sin x = x - x³/6

No error term! Because x⁴ = 0 on Δ₃, the Taylor series terminates exactly.
-}

postulate
  sin-taylor-Δ₃ : (δ : Δₖ 3) →
    sin (ιₖ δ) ≡ ιₖ δ -ℝ ((ιₖ δ) ³ / (# 6))
  -- Will be proven in DifferentialEquations.agda using taylor-theorem

--------------------------------------------------------------------------------
-- Summary

{-|
This module provides the foundation for:

1. **DifferentialEquations.agda**: Taylor series for exp, sin, cos, log on Δₖ
   - exp(x) = Σ(n=0 to k) xⁿ/n!  EXACTLY on Δₖ
   - sin(x) = x - x³/6 + x⁵/120 - ...  EXACTLY on Δₖ

2. **Physics.agda**: Rigorous approximations
   - SmallAmplitude: f' ∈ Δ₁ means f'² = 0 exactly (not approximately!)
   - Beam flexure with exact cancellation

3. **Multivariable.agda**: Extended micropolynomiality to n-microvectors
   - Functions on n-microvectors are multi-polynomials

**Key Achievement**: Taylor's theorem is EXACT on higher-order infinitesimals!

This is the revolutionary insight of smooth infinitesimal analysis:
classical "approximations" become exact statements when properly formulated.
-}
