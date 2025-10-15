{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Definite Integral and Integration Theory

**Reference**: John L. Bell (2008), *A Primer of Infinitesimal Analysis*, Chapter 6.1 (pp. 89-92)

This module implements integration theory in smooth infinitesimal analysis based on
the **Integration Principle**, which asserts that every function has a unique
antiderivative.

## Revolutionary Approach

**Classical analysis**: Prove existence of antiderivatives using Riemann sums/limits

**Smooth infinitesimal analysis**:
1. **Postulate Integration Principle**: Every f has unique F with F' = f, F(0) = 0
2. **Define** definite integral: ∫[a,b] f = F(b) - F(a)
3. **Derive** all properties from this foundation

## Key Results

1. **Integration Principle** (postulate): Antiderivatives exist uniquely
2. **Hadamard's Lemma**: f(y) - f(x) = (y-x)·∫₀¹ f'(x+t(y-x))dt
3. **Fundamental Theorem**: ∫[a,b] f' = f(b) - f(a)
4. **Properties**: Linearity, by parts, substitution, Fubini

## Applications

- **Geometry.agda**: Areas, arc lengths, volumes (Chapter 3)
- **Physics.agda**: Moments, centers of mass, work (Chapter 4)
- **Multivariable.agda**: Multiple integrals, Fubini's theorem

## Philosophy

In smooth infinitesimal analysis, we replace the classical limit-based definition
of integration with the axiomatic Integration Principle. This makes integration
theory constructive and exact.

The definite integral represents "exact accumulation" rather than "limit of Riemann sums".
-}

module Neural.Smooth.Integration where

open import 1Lab.Prelude
open import 1Lab.Path
open import 1Lab.Path.Reasoning
open import 1Lab.HLevel

open import Neural.Smooth.Base public
open import Neural.Smooth.Calculus public
open import Neural.Smooth.Functions public  -- For ^-1, #_, and other utilities
open import Neural.Smooth.DifferentialEquations public  -- For exp, sin, cos properties

open import Data.Nat.Base using (Nat; zero; suc)

private variable
  ℓ : Level

-- Note: ^-1-invl, ^-1-invr are now defined in Functions.agda
-- Note: double-neg is now proven in Base.agda
-- Note: _/_, _^_, #_ are imported from Functions.agda

--------------------------------------------------------------------------------
-- § 1: The Integration Principle (Bell pp. 89-90)

{-|
## Integration Principle

**Statement (Bell p. 89)**: "For any f: [0,1] → ℝ there is a unique g: [0,1] → ℝ
such that g' = f and g(0) = 0."

**Intuition**: This principle asserts that for any f, there is a definite function g
such that, for any x in [0,1], g(x) is the "area under the curve y = f(x) from 0 to x".

**Key difference from classical analysis**: In classical analysis, we must *prove*
existence of antiderivatives using Riemann sums and limits. In smooth infinitesimal
analysis, existence is *postulated* as a fundamental principle.

**Notation**: We write ∫₀ˣ f or ∫[0,x] f for g(x).
-}

-- Closed interval type
Interval : ℝ → ℝ → Type
Interval a b = Σ ℝ (λ x → (a ≤ℝ x) × (x ≤ℝ b))

-- Antiderivative with normalization
record Antiderivative (a b : ℝ) (f : ℝ → ℝ) : Type where
  field
    F : ℝ → ℝ
    F-derivative : ∀ x → (a ≤ℝ x) → (x ≤ℝ b) → F ′[ x ] ≡ f x
    F-initial : F a ≡ 0ℝ

-- Integration Principle (Bell p. 89)
postulate
  integration-principle : (a b : ℝ) (f : ℝ → ℝ) →
    Antiderivative a b f

{-|
## Uniqueness

The uniqueness follows from the constancy principle: if F and G both have
derivative f, then (F - G)' = 0, so F - G is constant. By the initial condition
F(0) = G(0) = 0, we have F = G.
-}

antiderivative-unique : (a b : ℝ) (f : ℝ → ℝ)
  (F G : Antiderivative a b f) →
  ∀ x → (a ≤ℝ x) → (x ≤ℝ b) →
  Antiderivative.F F x ≡ Antiderivative.F G x
antiderivative-unique a b f F G x a≤x x≤b =
  let F' = Antiderivative.F F
      G' = Antiderivative.F G
      F-deriv = Antiderivative.F-derivative F
      G-deriv = Antiderivative.F-derivative G
      F-init = Antiderivative.F-initial F
      G-init = Antiderivative.F-initial G

      -- F and G have the same derivative f everywhere in [a,b]
      same-deriv : ∀ y → F' ′[ y ] ≡ G' ′[ y ]
      same-deriv y = {!!}
        -- TODO: Need interval-aware version:
        -- F-deriv y (≤ℝ-refl {a}) (≤ℝ-trans a≤y y≤b) : F' ′[ y ] ≡ f y
        -- G-deriv y (≤ℝ-refl {a}) (≤ℝ-trans a≤y y≤b) : G' ′[ y ] ≡ f y
        -- Then: F' ′[ y ] ≡⟨ F-deriv ... ⟩ f y ≡⟨ sym (G-deriv ...) ⟩ G' ′[ y ] ∎
        -- Issue: Need a≤y and y≤b, but y is arbitrary

      -- By constancy principle, F = G + c for some constant c
      F-eq-G-plus-c : Σ[ c ∈ ℝ ] (∀ y → F' y ≡ G' y +ℝ c)
      F-eq-G-plus-c = same-derivative-constant F' G' same-deriv

      c : ℝ
      c = fst F-eq-G-plus-c

      F=G+c : ∀ y → F' y ≡ G' y +ℝ c
      F=G+c = snd F-eq-G-plus-c

      -- Compute c from initial conditions
      c-is-zero : c ≡ 0ℝ
      c-is-zero =
        c
          ≡⟨ sym (+ℝ-idl c) ⟩
        0ℝ +ℝ c
          ≡⟨ ap (_+ℝ c) (sym G-init) ⟩
        G' a +ℝ c
          ≡⟨ sym (F=G+c a) ⟩
        F' a
          ≡⟨ F-init ⟩
        0ℝ
          ∎

  in -- Use initial conditions: F(a) = 0 and G(a) = 0
     -- So 0 = F(a) = G(a) + c = 0 + c, hence c = 0
     F' x
       ≡⟨ F=G+c x ⟩
     G' x +ℝ c
       ≡⟨ ap (G' x +ℝ_) c-is-zero ⟩
     G' x +ℝ 0ℝ
       ≡⟨ +ℝ-idr (G' x) ⟩
     G' x
       ∎

--------------------------------------------------------------------------------
-- § 2: Definite Integral

{-|
## Definition of Definite Integral

For a ≤ b, we define:
  ∫[a,b] f = F(b) - F(a)

where F is the antiderivative of f with F(a) = 0.

**Note**: For the interval [0,1], we have ∫[0,x] f = F(x) where F is from
the Integration Principle.
-}

-- Definite integral from a to b
∫[_,_] : (a b : ℝ) → (f : ℝ → ℝ) → ℝ
∫[ a , b ] f =
  let anti = integration-principle a b f
      F = Antiderivative.F anti
  in F b -ℝ F a

-- Note: We write ∫[ a , b ] f directly (no syntax sugar needed)

-- Special case: integral from 0 to x
∫₀ : (x : ℝ) → (f : ℝ → ℝ) → ℝ
∫₀ x f = ∫[ 0ℝ , x ] f

--------------------------------------------------------------------------------
-- § 3: Hadamard's Lemma (Bell p. 90)

{-|
## Lemma 6.1: Hadamard's Lemma

**Statement (Bell p. 90)**: "For f : [a,b] → ℝ and x, y in [a,b] we have
  f(y) - f(x) = (y - x)·∫₀¹ f'(x + t(y-x))dt"

**Proof**: For any x,y in [a,b], define h: [0,1] → [a,b] by h(t) = x + t(y-x).
Since h' = y - x, we have
  f(y) - f(x) = f(h(1)) - f(h(0))
              = ∫₀¹ (f ∘ h)'(t) dt
              = ∫₀¹ (f' ∘ h)(t)·h'(t) dt  [chain rule]
              = ∫₀¹ f'(x + t(y-x))·(y-x) dt
              = (y-x)·∫₀¹ f'(x + t(y-x)) dt ∎

**Significance**: This shows that the derivative can be "integrated back" to
recover the difference in function values. It's a constructive version of the
Mean Value Theorem.
-}

postulate
  hadamard : (a b : ℝ) (f : ℝ → ℝ) (x y : ℝ) →
    (a ≤ℝ x) → (x ≤ℝ b) → (a ≤ℝ y) → (y ≤ℝ b) →
    f y -ℝ f x ≡ (y -ℝ x) ·ℝ ∫[ 0ℝ , 1ℝ ] (λ t → f ′[ x +ℝ t ·ℝ (y -ℝ x) ])

--------------------------------------------------------------------------------
-- § 4: Fundamental Theorem of Calculus

{-|
## Theorem 6.2: Fundamental Theorem

**Statement (Bell p. 90)**: "For any f : [a,b] → ℝ there is a unique g: [a,b] → ℝ
such that g' = f and g(a) = 0."

This is just the Integration Principle restated for arbitrary intervals.

**Consequence**: If F is any antiderivative of f, then
  ∫[a,b] f = F(b) - F(a)
-}

fundamental-theorem : (a b : ℝ) (f F : ℝ → ℝ) →
  (∀ x → (a ≤ℝ x) → (x ≤ℝ b) → F ′[ x ] ≡ f x) →
  ∫[ a , b ] f ≡ F b -ℝ F a
fundamental-theorem a b f F F-deriv = {!!}
  -- Proof: Let G be the canonical antiderivative from integration-principle
  -- Then (F - G)' = 0, so F - G is constant
  -- ∫[a,b] f = G(b) - G(a) = (F(b) - c) - (F(a) - c) = F(b) - F(a)

{-|
## Derivative of the Integral

**Corollary**: If F(x) = ∫[a,x] f, then F'(x) = f(x).

This is immediate from the Integration Principle.
-}

integral-derivative : (a b : ℝ) (f : ℝ → ℝ) (x : ℝ) →
  (a ≤ℝ x) → (x ≤ℝ b) →
  let F = λ y → ∫[ a , y ] f
  in F ′[ x ] ≡ f x
integral-derivative a b f x a≤x x≤b = {!!}
  -- Proof: This is exactly the Integration Principle

--------------------------------------------------------------------------------
-- § 5: Properties of Integration (Exercises 6.3-6.7)

{-|
## Linearity

The integral is linear:
  ∫[a,b] (f + g) = ∫[a,b] f + ∫[a,b] g
  ∫[a,b] (c·f) = c·∫[a,b] f
-}

-- Exercise 6.3(a): Additivity
∫-add : (a b : ℝ) (f g : ℝ → ℝ) →
  ∫[ a , b ] (λ x → f x +ℝ g x) ≡ ∫[ a , b ] f +ℝ ∫[ a , b ] g
∫-add a b f g =
  -- Strategy: Expand definitions and use properties of antiderivatives
  let anti-f = integration-principle a b f
      anti-g = integration-principle a b g
      anti-f+g = integration-principle a b (λ x → f x +ℝ g x)
      F_f = Antiderivative.F anti-f
      F_g = Antiderivative.F anti-g
      F_fg = Antiderivative.F anti-f+g
  in ∫[ a , b ] (λ x → f x +ℝ g x)
       ≡⟨⟩  -- Definition
     F_fg b -ℝ F_fg a
       ≡⟨ {!!} ⟩  -- TODO: F_fg = F_f + F_g (both are antiderivatives of f+g)
     (F_f b +ℝ F_g b) -ℝ (F_f a +ℝ F_g a)
       ≡⟨ {!!} ⟩  -- TODO: Algebra: (a+b)-(c+d) = (a-c)+(b-d)
     (F_f b -ℝ F_f a) +ℝ (F_g b -ℝ F_g a)
       ≡⟨⟩  -- Definition
     ∫[ a , b ] f +ℝ ∫[ a , b ] g
       ∎

-- Exercise 6.3(b): Scalar multiplication
∫-scalar : (a b : ℝ) (c : ℝ) (f : ℝ → ℝ) →
  ∫[ a , b ] (λ x → c ·ℝ f x) ≡ c ·ℝ ∫[ a , b ] f
∫-scalar a b c f =
  -- Strategy: Use scalar-rule to show (c·F)' = c·f
  -- Both F_cf and c·F_f are antiderivatives of c·f
  let anti-f = integration-principle a b f
      anti-cf = integration-principle a b (λ x → c ·ℝ f x)
      F_f = Antiderivative.F anti-f
      F_cf = Antiderivative.F anti-cf
  in ∫[ a , b ] (λ x → c ·ℝ f x)
       ≡⟨⟩  -- Definition
     F_cf b -ℝ F_cf a
       ≡⟨ {!!} ⟩  -- TODO: F_cf = c · F_f (both are antiderivatives of c·f with same boundary)
     (c ·ℝ F_f b) -ℝ (c ·ℝ F_f a)
       ≡⟨ {!!} ⟩  -- TODO: Algebra: c·b - c·a = c·(b-a)
     c ·ℝ (F_f b -ℝ F_f a)
       ≡⟨⟩  -- Definition
     c ·ℝ ∫[ a , b ] f
       ∎

{-|
## Integration by Parts (Exercise 6.3(d))

**Statement**: ∫[a,b] f'·g = f(b)·g(b) - f(a)·g(a) - ∫[a,b] f·g'

**Proof**: (f·g)' = f'·g + f·g' by product rule
So ∫[a,b] (f·g)' = f(b)·g(b) - f(a)·g(a) by fundamental theorem
But ∫[a,b] (f·g)' = ∫[a,b] f'·g + ∫[a,b] f·g' by linearity
Rearranging gives the result. ∎
-}

∫-by-parts : (a b : ℝ) (f g : ℝ → ℝ) →
  ∫[ a , b ] (λ x → (f ′[ x ]) ·ℝ g x) ≡
  (f b ·ℝ g b) -ℝ (f a ·ℝ g a) -ℝ ∫[ a , b ] (λ x → f x ·ℝ (g ′[ x ]))
∫-by-parts a b f g = {!!}

{-|
## Change of Variables (Exercise 6.4)

**Statement**: If h: [a,b] → [c,d] with h(a) = c and h(b) = d, then
  ∫[c,d] f = ∫[a,b] f(h(t))·h'(t)dt

**Proof**: Let F(x) = ∫[c,x] f. Then
  G(u) := F(h(u)) has derivative G'(u) = F'(h(u))·h'(u) = f(h(u))·h'(u)
So ∫[a,b] f(h(t))·h'(t)dt = G(b) - G(a) = F(h(b)) - F(h(a)) = F(d) - F(c) = ∫[c,d] f ∎
-}

∫-substitution : (a b c d : ℝ) (f h : ℝ → ℝ) →
  h a ≡ c → h b ≡ d →
  ∫[ c , d ] f ≡ ∫[ a , b ] (λ t → f (h t) ·ℝ (h ′[ t ]))
∫-substitution a b c d f h ha=c hb=d = {!!}

{-|
## Fubini's Theorem (Exercise 6.5)

**Statement**: For f : ℝ × ℝ → ℝ,
  ∫[a,b] (∫[c,d] f(x,y)dy)dx = ∫[c,d] (∫[a,b] f(x,y)dx)dy

**Idea**: Define F(x,y) = ∫∫ f(u,v)dudv. Then Fₓᵧ = f(x,y) and the theorem
follows from symmetry of mixed partials.
-}

postulate
  fubini : (a b c d : ℝ) (f : ℝ × ℝ → ℝ) →
    ∫[ a , b ] (λ x → ∫[ c , d ] (λ y → f (x , y))) ≡
    ∫[ c , d ] (λ y → ∫[ a , b ] (λ x → f (x , y)))

--------------------------------------------------------------------------------
-- § 6: Standard Antiderivatives

{-|
## Antiderivatives of Basic Functions

We can now state (and will prove in context) antiderivatives for basic functions.
-}

-- Antiderivative relationship
is-antiderivative : (F f : ℝ → ℝ) → Type
is-antiderivative F f = ∀ x → F ′[ x ] ≡ f x

-- Power rule for integration
∫-power : (n : Nat) →
  is-antiderivative
    (λ x → x ^ℝ suc n ·ℝ ((# (suc n)) ^-1))
    (λ x → x ^ℝ n)
∫-power n x =
  -- Goal: show (λ y → y^(suc n)/(# (suc n)))' x = x^n
  --
  -- Strategy:
  -- 1. (λ y → y^(suc n)/(# (suc n)))' = (λ y → y^(suc n) · (# (suc n))^(-1))'
  -- 2. = (# (suc n))^(-1) · (λ y → y^(suc n))'  [scalar-rule]
  -- 3. = (# (suc n))^(-1) · ((suc n) · x^n)  [power-rule]
  -- 4. = ((suc n) · (# (suc n))^(-1)) · x^n  [associativity]
  -- 5. = 1 · x^n  [inverse property]
  -- 6. = x^n  ✓
  microcancellation _ _ λ δ →
    (ι δ ·ℝ ((λ y → y ^ℝ suc n ·ℝ ((# (suc n)) ^-1)) ′[ x ]))
      -- First rewrite (f·c) to (c·f) using commutativity inside the derivative
      ≡⟨ ap (ι δ ·ℝ_) (ap (λ h → h ′[ x ]) (funext λ y → ·ℝ-comm (y ^ℝ suc n) ((# (suc n)) ^-1))) ⟩
    (ι δ ·ℝ ((λ y → ((# (suc n)) ^-1) ·ℝ (y ^ℝ suc n)) ′[ x ]))
      -- Now apply scalar-rule: (c·f)' = c·f'
      ≡⟨ ap (ι δ ·ℝ_) (scalar-rule ((# (suc n)) ^-1) (λ y → y ^ℝ suc n) x) ⟩
    (ι δ ·ℝ (((# (suc n)) ^-1) ·ℝ ((λ y → y ^ℝ suc n) ′[ x ])))
      ≡⟨ ap (λ d → ι δ ·ℝ (((# (suc n)) ^-1) ·ℝ d)) (power-rule (suc n) x) ⟩
    (ι δ ·ℝ (((# (suc n)) ^-1) ·ℝ (natToℝ (suc n) ·ℝ (x ^ℝ n))))
      ≡⟨ ap (ι δ ·ℝ_) (sym (·ℝ-assoc ((# (suc n)) ^-1) (natToℝ (suc n)) (x ^ℝ n))) ⟩
    (ι δ ·ℝ ((((# (suc n)) ^-1) ·ℝ natToℝ (suc n)) ·ℝ (x ^ℝ n)))
      ≡⟨⟩  -- # = natToℝ by definition
    (ι δ ·ℝ ((((# (suc n)) ^-1) ·ℝ (# (suc n))) ·ℝ (x ^ℝ n)))
      ≡⟨ ap (λ z → ι δ ·ℝ (z ·ℝ (x ^ℝ n))) (^-1-invl (# (suc n))) ⟩
    (ι δ ·ℝ (1ℝ ·ℝ (x ^ℝ n)))
      ≡⟨ ap (ι δ ·ℝ_) (·ℝ-idl (x ^ℝ n)) ⟩
    (ι δ ·ℝ (x ^ℝ n))
      ∎

-- Exponential
∫-exp : is-antiderivative exp exp
∫-exp x =
  -- Goal: show exp' x = exp x
  -- This is exactly the defining property of exp from DifferentialEquations.agda!
  -- exp is characterized by: exp' = exp and exp(0) = 1
  fst exp-is-exponential x  -- Extract first component: ∀ x → exp' x = exp x

-- Trigonometric functions
∫-sin : is-antiderivative (λ x → -ℝ (cos x)) sin
∫-sin x =
  -- Goal: show (-cos)' x = sin x
  -- From DifferentialEquations.agda: cos' x = -sin x
  -- Strategy: -cos = (-1) · cos, so (-cos)' = (-1) · cos' = (-1) · (-sin) = sin
  ((λ y → -ℝ (cos y)) ′[ x ])
    ≡⟨ ap (λ h → h ′[ x ]) (funext λ y → sym (neg-mult (cos y))) ⟩
  ((λ y → (-ℝ 1ℝ) ·ℝ cos y) ′[ x ])
    ≡⟨ scalar-rule (-ℝ 1ℝ) cos x ⟩
  ((-ℝ 1ℝ) ·ℝ (cos ′[ x ]))
    ≡⟨ ap ((-ℝ 1ℝ) ·ℝ_) (cos-derivative x) ⟩
  ((-ℝ 1ℝ) ·ℝ (-ℝ (sin x)))
    ≡⟨ double-neg (sin x) ⟩
  sin x
    ∎

∫-cos : is-antiderivative sin cos
∫-cos x =
  -- Goal: show sin' x = cos x
  -- This is exactly sin-derivative from DifferentialEquations.agda! ✓
  sin-derivative x

-- Reciprocal (logarithm)
postulate
  ∫-reciprocal : (x₊ : ℝ₊) →
    is-antiderivative (λ y → log (y , {!!})) (λ y → y ^-1)
  -- Proof: log' = 1/x from DifferentialEquations.agda

{-|
## Computing Definite Integrals

Using these antiderivatives, we can compute definite integrals:

∫[0,1] x dx = [x²/2]₀¹ = 1/2

∫[0,π] sin(x)dx = [-cos(x)]₀π = -cos(π) + cos(0) = 1 + 1 = 2
-}

-- Helper lemmas for example
private
  1²-is-1 : 1ℝ ^ 2 ≡ 1ℝ
  1²-is-1 =
    1ℝ ^ 2
      ≡⟨⟩  -- 1ℝ ^ℝ suc 1 = 1ℝ · (1ℝ ^ 1)
    1ℝ ·ℝ (1ℝ ^ 1)
      ≡⟨⟩  -- 1ℝ ^ 1 = 1ℝ · (1ℝ ^ 0)
    1ℝ ·ℝ (1ℝ ·ℝ (1ℝ ^ 0))
      ≡⟨⟩  -- 1ℝ ^ 0 = 1ℝ
    1ℝ ·ℝ (1ℝ ·ℝ 1ℝ)
      ≡⟨ ap (1ℝ ·ℝ_) (·ℝ-idl 1ℝ) ⟩
    1ℝ ·ℝ 1ℝ
      ≡⟨ ·ℝ-idl 1ℝ ⟩
    1ℝ
      ∎

  0²-is-0 : 0ℝ ^ 2 ≡ 0ℝ
  0²-is-0 =
    0ℝ ^ 2
      ≡⟨⟩  -- 0ℝ ^ℝ suc 1 = 0ℝ · (0ℝ ^ 1)
    0ℝ ·ℝ (0ℝ ^ 1)
      ≡⟨⟩  -- 0ℝ ^ 1 = 0ℝ · (0ℝ ^ 0)
    0ℝ ·ℝ (0ℝ ·ℝ (0ℝ ^ 0))
      ≡⟨⟩  -- 0ℝ ^ 0 = 1ℝ
    0ℝ ·ℝ (0ℝ ·ℝ 1ℝ)
      ≡⟨ ap (0ℝ ·ℝ_) (·ℝ-idr 0ℝ) ⟩
    0ℝ ·ℝ 0ℝ
      ≡⟨ ·ℝ-zerol 0ℝ ⟩
    0ℝ
      ∎

  1²/2-is-1/2 : (1ℝ ^ 2) / (# 2) ≡ 1/2
  1²/2-is-1/2 =
    (1ℝ ^ 2) / (# 2)
      ≡⟨ ap (_/ (# 2)) 1²-is-1 ⟩
    1ℝ / (# 2)
      ≡⟨⟩  -- 1ℝ / x = 1ℝ · x^-1 by definition
    1ℝ ·ℝ ((# 2) ^-1)
      ≡⟨ sym (ap (_·ℝ ((# 2) ^-1)) (+ℝ-idr 1ℝ)) ⟩
    (1ℝ +ℝ 0ℝ) ·ℝ ((# 2) ^-1)
      ≡⟨⟩  -- natToℝ 0 = 0ℝ
    (1ℝ +ℝ natToℝ 0) ·ℝ ((# 2) ^-1)
      ≡⟨⟩  -- natToℝ 1 = 1ℝ +ℝ natToℝ 0 by definition
    (# 1) ·ℝ ((# 2) ^-1)
      ≡⟨⟩  -- 1/2 definition
    1/2
      ∎

  0²/2-is-0 : (0ℝ ^ 2) / (# 2) ≡ 0ℝ
  0²/2-is-0 =
    (0ℝ ^ 2) / (# 2)
      ≡⟨ ap (_/ (# 2)) 0²-is-0 ⟩
    0ℝ / (# 2)
      ≡⟨⟩  -- 0 / a = 0 · a^-1
    0ℝ ·ℝ ((# 2) ^-1)
      ≡⟨ ·ℝ-zerol ((# 2) ^-1) ⟩
    0ℝ
      ∎

example-∫-x : ∫[ 0ℝ , 1ℝ ] (λ x → x) ≡ 1/2
example-∫-x =
  -- By ∫-power with n=0, we have F(x) = x^1/1 = x²/2 is antiderivative of x^0 = x
  -- Wait, that's wrong. Let me reconsider.
  -- For n=0: ∫ x^0 dx = ∫ 1 dx = x (not x²/2)
  -- For n=1: ∫ x^1 dx = x²/2
  -- So we want ∫-power with n=1.
  --
  -- ∫[0,1] x = ∫[0,1] x^1
  --          = [x^2/2]₀¹  [by ∫-power with n=1]
  --          = (1^2/2) - (0^2/2)
  --          = 1/2 - 0
  --          = 1/2  ✓
  let F = λ x → (x ^ 2) / (# 2)
      F-is-antideriv = ∫-power 1  -- F'[x] = x^1 for all x
      -- Use fundamental theorem to relate ∫[0,1] to F(1) - F(0)
      F-deriv-in-range : ∀ x → (0ℝ ≤ℝ x) → (x ≤ℝ 1ℝ) → F ′[ x ] ≡ (λ y → y ^ 1) x
      F-deriv-in-range x _ _ = F-is-antideriv x
  in ∫[ 0ℝ , 1ℝ ] (λ x → x)
       ≡⟨ ap (∫[ 0ℝ , 1ℝ ]) (funext λ x → sym (^ℝ-1 x)) ⟩
     ∫[ 0ℝ , 1ℝ ] (λ x → x ^ 1)
       ≡⟨ fundamental-theorem 0ℝ 1ℝ (λ x → x ^ 1) F F-deriv-in-range ⟩
     F 1ℝ -ℝ F 0ℝ
       ≡⟨⟩  -- F(x) = x²/2, so F(1) = 1²/2, F(0) = 0²/2
     ((1ℝ ^ 2) / (# 2)) -ℝ ((0ℝ ^ 2) / (# 2))
       ≡⟨ ap₂ _-ℝ_ 1²/2-is-1/2 0²/2-is-0 ⟩
     1/2 -ℝ 0ℝ
       ≡⟨⟩  -- x -ℝ y = x +ℝ (-ℝ y)
     1/2 +ℝ (-ℝ 0ℝ)
       ≡⟨ ap (1/2 +ℝ_) -ℝ-zero ⟩
     1/2 +ℝ 0ℝ
       ≡⟨ +ℝ-idr 1/2 ⟩
     1/2
       ∎

--------------------------------------------------------------------------------
-- § 7: Connection to Geometry

{-|
## Areas Under Curves

The definite integral represents the **exact area** under a curve.

For f : [a,b] → ℝ with f ≥ 0, the area A under y = f(x) from x = a to x = b is:
  A = ∫[a,b] f

**Derivation**: Consider a vertical strip at position x with width ε ∈ Δ.
The area of this strip is f(x)·ε (a microarea).
The total area is the "sum" of all these microareas, which is ∫[a,b] f.
-}

-- Area under a curve
area-under-curve : (a b : ℝ) (f : ℝ → ℝ) → ℝ
area-under-curve a b f = ∫[ a , b ] f

{-|
## Arc Length

The arc length of a curve y = f(x) from x = a to x = b is:
  s = ∫[a,b] √(1 + f'(x)²) dx

This comes from the microstraightness principle: a microsegment has length
ds = √(dx² + dy²) = √(1 + (dy/dx)²)·dx
-}

-- Arc length of a curve
arc-length : (a b : ℝ) (f : ℝ → ℝ) → ℝ
arc-length a b f = ∫[ a , b ] (λ x → ((1ℝ +ℝ ((f ′[ x ]) ²)) ^1/2))

{-|
## Volume of Revolution

The volume obtained by rotating y = f(x) from x = a to x = b about the x-axis is:
  V = ∫[a,b] π·f(x)² dx

**Derivation**: A thin disk at position x with thickness ε has volume π·f(x)²·ε.
The total volume is ∫[a,b] π·f² dx.
-}

-- Volume of revolution about x-axis
volume-of-revolution : (a b : ℝ) (f : ℝ → ℝ) → ℝ
volume-of-revolution a b f = ∫[ a , b ] (λ x → π ·ℝ ((f x) ²))

--------------------------------------------------------------------------------
-- Summary

{-|
This module provides:

1. **Integration Principle**: Foundation for integration theory
2. **Definite integral**: ∫[a,b] f defined via antiderivatives
3. **Hadamard's Lemma**: Constructive mean value theorem
4. **Fundamental Theorem**: ∫[a,b] f' = f(b) - f(a)
5. **Properties**: Linearity, by parts, substitution, Fubini
6. **Standard antiderivatives**: For powers, exp, sin, cos, 1/x

**Applications**:
- Geometry.agda uses this for areas, arc lengths, volumes
- Physics.agda uses this for moments, centers of mass, work
- Multivariable.agda extends to multiple integrals

**Next**: Physics.agda will implement all of Bell Chapter 4 using these tools!
-}
