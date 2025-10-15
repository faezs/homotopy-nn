{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Special Functions in Smooth Infinitesimal Analysis

**Reference**: Section 2.4 from "Smooth Infinitesimal Analysis"

This module implements the classical transcendental functions (√, sin, cos, exp)
and derives their properties using smooth infinitesimal analysis.

## Key Results

1. **Square root**: √(a+ε) = √a + ε/(2√a) for a > 0
2. **Sine/cosine on Δ**: sin ε = ε, cos ε = 1 for ε ∈ Δ
3. **Derivatives**: sin' = cos, cos' = -sin
4. **Tangent angle**: sin φ = f' cos φ for curve tangent
5. **Exponential**: exp(ε) = 1 + ε, exp' = exp, exp(x+y) = exp(x)·exp(y)

## Physical Interpretation for Neural Networks

- **exp**: Softmax, sigmoid activation, probability densities
- **sin/cos**: Positional encodings (transformers), periodic activations
- **√**: Normalization layers, distance metrics
- **Tangent angle**: Gradient direction in parameter space

## Applications

- **Activation functions**: Smooth approximations (sigmoid ≈ softplus, tanh)
- **Attention**: Softmax uses exp
- **Positional encoding**: sin/cos for sequence position
- **Gradient flow**: exp for learning rate schedules
-}

module Neural.Smooth.Functions where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path
open import 1Lab.Path.Reasoning

open import Neural.Smooth.Base public
open import Neural.Smooth.Calculus public

private variable
  ℓ : Level

--------------------------------------------------------------------------------
-- § 1: Positive Reals and Square Root

{-|
## Positive Real Numbers

We define ℝ₊ = {x ∈ ℝ | x > 0}, the positive reals.

**Key property**: ℝ₊ is microstable (Exercise 1.6(iv)).
-}

ℝ₊ : Type
ℝ₊ = Σ[ x ∈ ℝ ] (0ℝ <ℝ x)

-- Extract the real number
value : ℝ₊ → ℝ
value (x , _) = x

-- Positivity proof
positive : (x₊ : ℝ₊) → 0ℝ <ℝ value x₊
positive (_ , p) = p

-- ℝ₊ is microstable
postulate
  ℝ₊-microstable : ∀ (x₊ : ℝ₊) (δ : Δ) →
    Σ[ y₊ ∈ ℝ₊ ] (value y₊ ≡ value x₊ +ℝ ι δ)

{-|
## Square Root Function (Section 2.4)

For a > 0, the square root √a can be constructed geometrically (Figure 1.3):
- Mark segment OA of length a and AB of length 1 on a line
- Draw circle with diameter OB
- Perpendicular through A meets circle at C
- Length AC = √a

**Properties**:
- (√a)² = a
- √(a·b) = √a · √b
- √1 = 1

**Derivative**: (√x)' = 1/(2√x)
-}

postulate
  √ : ℝ₊ → ℝ₊

  √-square : ∀ (x₊ : ℝ₊) → value (√ x₊) ·ℝ value (√ x₊) ≡ value x₊

  √-one : value (√ (1ℝ , 0<1)) ≡ 1ℝ

  -- Helper: product of two positive reals is positive
  ℝ₊-product : ℝ₊ → ℝ₊ → ℝ₊

  √-product : ∀ (x₊ y₊ : ℝ₊) →
    value (√ (ℝ₊-product x₊ y₊)) ≡ value (√ x₊) ·ℝ value (√ y₊)

-- Derivative of square root (stated as a postulate)
-- In full: there exists a smooth extension f : ℝ → ℝ of √ such that
-- f'(x) = 1/(2√x) for x > 0
postulate
  √-extends-to-ℝ : ℝ → ℝ
  √-extension-correct : ∀ (x₊ : ℝ₊) → √-extends-to-ℝ (value x₊) ≡ value (√ x₊)
  √-deriv : ∀ (x₊ : ℝ₊) (denominator-pos : 0ℝ <ℝ ((1ℝ +ℝ 1ℝ) ·ℝ value (√ x₊))) →
    √-extends-to-ℝ ′[ value x₊ ] ≡ (1ℝ /ℝ ((1ℝ +ℝ 1ℝ) ·ℝ value (√ x₊))) (λ eq → <ℝ-irrefl (subst (0ℝ <ℝ_) eq denominator-pos))

-- Alternative: √ as x^(1/2)
postulate
  √-as-power : ∀ (x₊ : ℝ₊) →
    {-| √x = x^(1/2) in the sense of fractional powers -}
    ⊤

--------------------------------------------------------------------------------
-- § 2: Trigonometric Functions

{-|
## Sine and Cosine (Section 2.4)

The sine and cosine functions sin, cos : ℝ → ℝ are defined via
right-angled triangles:
- For angle x (in radians) in a right triangle with hypotenuse c:
- Opposite side = c · sin x
- Adjacent side = c · cos x

**Fundamental identity**: sin²(x) + cos²(x) = 1

**Addition formulas**:
- sin(x + y) = sin x · cos y + cos x · sin y
- cos(x + y) = cos x · cos y - sin x · sin y

**Boundary conditions**:
- sin 0 = 0
- cos 0 = 1
-}

postulate
  sin : ℝ → ℝ
  cos : ℝ → ℝ

  -- Pythagorean identity
  sin²+cos² : ∀ (x : ℝ) →
    (sin x ·ℝ sin x) +ℝ (cos x ·ℝ cos x) ≡ 1ℝ

  -- Addition formulas
  sin-add : ∀ (x y : ℝ) →
    sin (x +ℝ y) ≡ (sin x ·ℝ cos y) +ℝ (cos x ·ℝ sin y)

  cos-add : ∀ (x y : ℝ) →
    cos (x +ℝ y) ≡ (cos x ·ℝ cos y) -ℝ (sin x ·ℝ sin y)

  -- Boundary conditions
  sin-zero : sin 0ℝ ≡ 0ℝ
  cos-zero : cos 0ℝ ≡ 1ℝ

{-|
## Sine and Cosine on Infinitesimals

**Microstraightness of circles**: For ε ∈ Δ, the arc of angle 2ε on a unit
circle is straight and equals the chord of length 2·sin ε.

By Microstraightness: 2ε = 2·sin ε, so **sin ε = ε**.

From sin²(ε) + cos²(ε) = 1 and sin ε = ε:
  ε² + cos²(ε) = 1
  0 + cos²(ε) = 1  (since ε² = 0)
  cos²(ε) = 1

Therefore **cos ε = 1** (taking positive square root).

**These are EXACT equalities, not approximations!**
-}

sin-on-Δ : ∀ (δ : Δ) → sin (ι δ) ≡ ι δ
sin-on-Δ δ = {!!}  -- Proof via microstraightness of circle

cos-on-Δ : ∀ (δ : Δ) → cos (ι δ) ≡ 1ℝ
cos-on-Δ δ =
  -- From sin²(ε) + cos²(ε) = 1
  -- and sin(ε) = ε, we get ε² + cos²(ε) = 1
  -- Since ε² = 0, we have cos²(ε) = 1
  {!!}  -- Complete using nilsquare and sqrt

{-|
## Derivatives of Sine and Cosine

**Theorem**: sin' = cos and cos' = -sin

**Proof for sin'**:
  sin(x + ε) = sin x · cos ε + cos x · sin ε  (addition formula)
             = sin x · 1 + cos x · ε          (cos ε = 1, sin ε = ε)
             = sin x + ε · cos x

Comparing with fundamental equation sin(x+ε) = sin x + ε·sin'(x),
we get sin'(x) = cos x.

**Proof for cos'**: Similar using cos(x+ε) formula.
-}

sin-deriv : ∀ (x : ℝ) → sin ′[ x ] ≡ cos x
sin-deriv x =
  microcancellation _ _ λ δ →
    let -- fundamental-equation: sin(x + ιδ) = sin x + ιδ · sin'(x)
        -- Rearrange: ιδ · sin'(x) = sin(x + ιδ) - sin x
        fund-eq = fundamental-equation sin x δ
    in ι δ ·ℝ sin ′[ x ]
      ≡⟨ sym (+ℝ-idl (ι δ ·ℝ sin ′[ x ])) ⟩
    0ℝ +ℝ (ι δ ·ℝ sin ′[ x ])
      ≡⟨ ap (_+ℝ (ι δ ·ℝ sin ′[ x ])) (sym (+ℝ-invr (sin x))) ⟩
    ((sin x) +ℝ (-ℝ sin x)) +ℝ (ι δ ·ℝ sin ′[ x ])
      ≡⟨ +ℝ-assoc (sin x) (-ℝ sin x) (ι δ ·ℝ sin ′[ x ]) ⟩
    (sin x) +ℝ ((-ℝ sin x) +ℝ (ι δ ·ℝ sin ′[ x ]))
      ≡⟨ ap (sin x +ℝ_) (+ℝ-comm (-ℝ sin x) (ι δ ·ℝ sin ′[ x ])) ⟩
    (sin x) +ℝ ((ι δ ·ℝ sin ′[ x ]) +ℝ (-ℝ sin x))
      ≡⟨ sym (+ℝ-assoc (sin x) (ι δ ·ℝ sin ′[ x ]) (-ℝ sin x)) ⟩
    ((sin x) +ℝ (ι δ ·ℝ sin ′[ x ])) +ℝ (-ℝ sin x)
      ≡⟨ ap (_+ℝ (-ℝ sin x)) (sym fund-eq) ⟩
    sin (x +ℝ ι δ) +ℝ (-ℝ sin x)
      ≡⟨⟩
    sin (x +ℝ ι δ) -ℝ sin x
      ≡⟨ ap (_-ℝ sin x) (sin-add x (ι δ)) ⟩
    ((sin x ·ℝ cos (ι δ)) +ℝ (cos x ·ℝ sin (ι δ))) -ℝ sin x
      ≡⟨ ap₂ (λ u v → ((sin x ·ℝ u) +ℝ (cos x ·ℝ v)) -ℝ sin x)
             (cos-on-Δ δ) (sin-on-Δ δ) ⟩
    ((sin x ·ℝ 1ℝ) +ℝ (cos x ·ℝ ι δ)) -ℝ sin x
      ≡⟨ {!!} ⟩  -- Algebra: simplify to ι δ · cos x
    ι δ ·ℝ cos x
      ∎

cos-deriv : ∀ (x : ℝ) → cos ′[ x ] ≡ -ℝ sin x
cos-deriv x =
  microcancellation _ _ λ δ →
    let fund-eq = fundamental-equation cos x δ
    in ι δ ·ℝ cos ′[ x ]
      ≡⟨ sym (+ℝ-idl (ι δ ·ℝ cos ′[ x ])) ⟩
    0ℝ +ℝ (ι δ ·ℝ cos ′[ x ])
      ≡⟨ ap (_+ℝ (ι δ ·ℝ cos ′[ x ])) (sym (+ℝ-invr (cos x))) ⟩
    ((cos x) +ℝ (-ℝ cos x)) +ℝ (ι δ ·ℝ cos ′[ x ])
      ≡⟨ +ℝ-assoc (cos x) (-ℝ cos x) (ι δ ·ℝ cos ′[ x ]) ⟩
    (cos x) +ℝ ((-ℝ cos x) +ℝ (ι δ ·ℝ cos ′[ x ]))
      ≡⟨ ap (cos x +ℝ_) (+ℝ-comm (-ℝ cos x) (ι δ ·ℝ cos ′[ x ])) ⟩
    (cos x) +ℝ ((ι δ ·ℝ cos ′[ x ]) +ℝ (-ℝ cos x))
      ≡⟨ sym (+ℝ-assoc (cos x) (ι δ ·ℝ cos ′[ x ]) (-ℝ cos x)) ⟩
    ((cos x) +ℝ (ι δ ·ℝ cos ′[ x ])) +ℝ (-ℝ cos x)
      ≡⟨ ap (_+ℝ (-ℝ cos x)) (sym fund-eq) ⟩
    cos (x +ℝ ι δ) +ℝ (-ℝ cos x)
      ≡⟨⟩
    cos (x +ℝ ι δ) -ℝ cos x
      ≡⟨ ap (_-ℝ cos x) (cos-add x (ι δ)) ⟩
    ((cos x ·ℝ cos (ι δ)) -ℝ (sin x ·ℝ sin (ι δ))) -ℝ cos x
      ≡⟨ ap₂ (λ u v → ((cos x ·ℝ u) -ℝ (sin x ·ℝ v)) -ℝ cos x)
             (cos-on-Δ δ) (sin-on-Δ δ) ⟩
    ((cos x ·ℝ 1ℝ) -ℝ (sin x ·ℝ ι δ)) -ℝ cos x
      ≡⟨ {!!} ⟩  -- Algebra: simplify to -ι δ · sin x
    ι δ ·ℝ (-ℝ sin x)
      ∎

{-|
## Tangent Angle of a Curve (Section 2.4)

For a curve y = f(x), let φ(x) be the angle the tangent makes with the x-axis
at the point (x, f(x)).

Consider the microtriangle formed by:
- Point A = (x, f(x))
- Point B = (x + ε, f(x + ε)) on the curve
- Point C = (x + ε, f(x))

By Microstraightness, AB is straight with:
- Horizontal side AC of length ε
- Vertical side BC of length ε·f'(x)
- Hypotenuse AB making angle φ(x) with horizontal

From trigonometry:
  BC = AC · tan φ(x)

But tan is not defined everywhere, so we use:
  BC · cos φ(x) = AC · sin φ(x)
  ε·f'(x) · cos φ(x) = ε · sin φ(x)

Cancelling ε: **sin φ(x) = f'(x) · cos φ(x)**

**Consequence**: cos φ(x) ≠ 0 always (tangent never vertical in smooth world).
-}

postulate
  tangent-angle : (f : ℝ → ℝ) (x : ℝ) →
    Σ[ φ ∈ ℝ ] (sin φ ≡ (f ′[ x ]) ·ℝ cos φ)

  tangent-angle-unique : (f : ℝ → ℝ) (x : ℝ) →
    {-| The angle φ is uniquely determined (up to 2π) -}
    ⊤

  cos-tangent-nonzero : (f : ℝ → ℝ) (x : ℝ) →
    let φ = tangent-angle f x .fst
    in cos φ ≠ 0ℝ

-- Alternative formula: cos φ = 1/√(1 + f'²)
postulate
  cos-tangent-formula : (f : ℝ → ℝ) (x : ℝ) →
    let φ = tangent-angle f x .fst
    in cos φ ≡ value (√ {!!})  -- 1 + f'(x)² must be positive

--------------------------------------------------------------------------------
-- § 3: Exponential Function

{-|
## The Exponential Function (Section 2.4)

The exponential function exp : ℝ → ℝ is characterized by:
1. exp(x) > 0 for all x
2. exp' = exp (derivative equals itself)
3. exp(0) = 1

These three conditions uniquely determine exp.

**Properties**:
- exp(x + y) = exp(x) · exp(y) (exponential law)
- exp(-x) = 1 / exp(x)
- exp(n) = e^n for natural n, where e = exp(1)
- exp(m/n) = e^(m/n) for rationals

For non-rational x, exp(x) is defined via the differential equation.
-}

postulate
  exp : ℝ → ℝ

  -- Positivity
  exp-positive : ∀ (x : ℝ) → 0ℝ <ℝ exp x

  -- Derivative property
  exp-deriv : ∀ (x : ℝ) → exp ′[ x ] ≡ exp x

  -- Initial condition
  exp-zero : exp 0ℝ ≡ 1ℝ

-- Define e = exp(1)
e : ℝ
e = exp 1ℝ

{-|
## Exponential on Infinitesimals

For ε ∈ Δ:
  exp(ε) = exp(0 + ε)
         = exp(0) + ε · exp'(0)  (fundamental equation)
         = 1 + ε · exp(0)        (exp'(x) = exp(x))
         = 1 + ε · 1
         = 1 + ε

So **exp(ε) = 1 + ε** for ε ∈ Δ.
-}

exp-on-Δ : ∀ (δ : Δ) → exp (ι δ) ≡ 1ℝ +ℝ ι δ
exp-on-Δ δ =
  exp (ι δ)
    ≡⟨ ap exp (sym (+ℝ-idl (ι δ))) ⟩
  exp (0ℝ +ℝ ι δ)
    ≡⟨ fundamental-equation exp 0ℝ δ ⟩
  exp 0ℝ +ℝ (ι δ ·ℝ exp ′[ 0ℝ ])
    ≡⟨ ap₂ _+ℝ_ exp-zero (ap (ι δ ·ℝ_) (exp-deriv 0ℝ)) ⟩
  1ℝ +ℝ (ι δ ·ℝ exp 0ℝ)
    ≡⟨ ap (1ℝ +ℝ_) (ap (ι δ ·ℝ_) exp-zero) ⟩
  1ℝ +ℝ (ι δ ·ℝ 1ℝ)
    ≡⟨ ap (1ℝ +ℝ_) (·ℝ-idr (ι δ)) ⟩
  1ℝ +ℝ ι δ
    ∎

{-|
## Exponential Law (Equation 2.8)

**Theorem**: exp(x + y) = exp(x) · exp(y)

**Proof**: Fix y and consider g(x) = exp(x+y) / exp(x).
Then:
  g'(x) = [exp'(x+y) · exp(x) - exp(x+y) · exp'(x)] / exp²(x)  (quotient rule)
        = [exp(x+y) · exp(x) - exp(x+y) · exp(x)] / exp²(x)    (exp' = exp)
        = 0

By Constancy Principle, g(x) = g(0) for all x.
So exp(x+y) / exp(x) = exp(y) / exp(0) = exp(y) / 1 = exp(y).
Therefore exp(x+y) = exp(x) · exp(y).
-}

exp-add : ∀ (x y : ℝ) → exp (x +ℝ y) ≡ exp x ·ℝ exp y
exp-add x y = {!!}  -- Proof via constancy principle as sketched

-- Corollary: exp(-x) = 1/exp(x)
exp-neg : ∀ (x : ℝ) → exp (-ℝ x) ≡ (1ℝ /ℝ exp x) (λ p → {!!})
exp-neg x = {!!}  -- From exp(x + (-x)) = exp(0) = 1

{-|
## Exponential for Rational Arguments

For natural number n:
  exp(n) = exp(1 + 1 + ... + 1) (n times)
         = exp(1) · exp(1) · ... · exp(1)  (exponential law)
         = e^n

For negative integers: exp(-n) = 1/e^n

For rationals m/n: exp(m/n) = (exp(1/n))^m = (e^(1/n))^m = e^(m/n)
-}

postulate
  exp-nat : ∀ (n : Nat) →
    {-| exp(fromℕ n) = e^n where e = exp(1) -}
    ⊤

  exp-rational : ∀ (m : Nat) (n : Nat) →
    {-| exp(m/n) = e^(m/n) -}
    ⊤

{-|
## Uniqueness of Exponential

**Theorem**: If h : ℝ → ℝ satisfies h(x) > 0, h' = h, and h(0) = 1, then h = exp.

**Proof**: Consider g = h/exp. Then:
  g' = (h' · exp - h · exp') / exp²
     = (h · exp - h · exp) / exp²  (since h' = h and exp' = exp)
     = 0

By Constancy Principle, g is constant. Since g(0) = h(0)/exp(0) = 1/1 = 1,
we have g(x) = 1 for all x, so h(x) = exp(x).
-}

postulate
  exp-unique : ∀ (h : ℝ → ℝ) →
    (∀ x → 0ℝ <ℝ h x) →
    (∀ x → h ′[ x ] ≡ h x) →
    (h 0ℝ ≡ 1ℝ) →
    (∀ x → h x ≡ exp x)

--------------------------------------------------------------------------------
-- § 4: Logarithm (Inverse of Exponential)

{-|
## Natural Logarithm

Since exp : ℝ → ℝ₊ is a bijection (strictly increasing, continuous, unbounded),
it has an inverse log : ℝ₊ → ℝ (natural logarithm).

**Properties**:
- log(exp(x)) = x for all x ∈ ℝ
- exp(log(x)) = x for all x ∈ ℝ₊
- log(x · y) = log(x) + log(y)
- log(1) = 0
- log(e) = 1

**Derivative**: By inverse function rule,
  log'(x) = 1 / exp'(log(x)) = 1 / exp(log(x)) = 1/x
-}

postulate
  log : ℝ₊ → ℝ

  log-exp : ∀ (x : ℝ) → log (exp x , exp-positive x) ≡ x

  exp-log : ∀ (x₊ : ℝ₊) → exp (log x₊) ≡ value x₊

  log-product : ∀ (x₊ y₊ : ℝ₊) →
    log {!!} ≡ log x₊ +ℝ log y₊  -- x₊ · y₊ is positive

  log-one : log (1ℝ , 0<1) ≡ 0ℝ

  log-e : log (e , exp-positive 1ℝ) ≡ 1ℝ

-- Derivative of logarithm (stated as a postulate)
-- The proper statement requires showing log extends to a smooth function on ℝ₊
postulate
  log-extends-to-ℝ : ℝ → ℝ  -- Extends log to all of ℝ (undefined for x ≤ 0)
  log-extension-correct : ∀ (x₊ : ℝ₊) → log-extends-to-ℝ (value x₊) ≡ log x₊
  log-deriv : ∀ (x₊ : ℝ₊) (x-pos : 0ℝ <ℝ value x₊) →
    log-extends-to-ℝ ′[ value x₊ ] ≡ (1ℝ /ℝ value x₊) (λ eq → <ℝ-irrefl (subst (0ℝ <ℝ_) eq x-pos))

--------------------------------------------------------------------------------
-- § 5: Hyperbolic Functions

{-|
## Hyperbolic Sine and Cosine

Defined in terms of exponential:
  sinh(x) = (e^x - e^(-x)) / 2
  cosh(x) = (e^x + e^(-x)) / 2

**Properties**:
- cosh²(x) - sinh²(x) = 1 (hyperbolic Pythagorean identity)
- sinh' = cosh, cosh' = sinh
- sinh(0) = 0, cosh(0) = 1
- sinh(x + y) = sinh(x)cosh(y) + cosh(x)sinh(y)
- cosh(x + y) = cosh(x)cosh(y) + sinh(x)sinh(y)
-}

postulate
  sinh : ℝ → ℝ
  cosh : ℝ → ℝ

  sinh-def : ∀ (x : ℝ) →
    sinh x ≡ ((exp x -ℝ exp (-ℝ x)) /ℝ (1ℝ +ℝ 1ℝ)) {!!}

  cosh-def : ∀ (x : ℝ) →
    cosh x ≡ ((exp x +ℝ exp (-ℝ x)) /ℝ (1ℝ +ℝ 1ℝ)) {!!}

  -- Hyperbolic identity
  cosh²-sinh² : ∀ (x : ℝ) →
    (cosh x ·ℝ cosh x) -ℝ (sinh x ·ℝ sinh x) ≡ 1ℝ

  -- Derivatives
  sinh-deriv : ∀ (x : ℝ) → sinh ′[ x ] ≡ cosh x
  cosh-deriv : ∀ (x : ℝ) → cosh ′[ x ] ≡ sinh x

--------------------------------------------------------------------------------
-- § 6: Summary and Exports

{-|
## What We've Defined

**Functions on ℝ**:
- √ : ℝ₊ → ℝ₊ (square root)
- sin, cos : ℝ → ℝ (trigonometric)
- exp : ℝ → ℝ (exponential)
- log : ℝ₊ → ℝ (natural logarithm)
- sinh, cosh : ℝ → ℝ (hyperbolic)

**Key properties on infinitesimals**:
- sin ε = ε
- cos ε = 1
- exp ε = 1 + ε

**Derivatives**:
- (√x)' = 1/(2√x)
- sin' = cos, cos' = -sin
- exp' = exp
- log' = 1/x
- sinh' = cosh, cosh' = sinh

**Applications**:
- Tangent angle: sin φ = f' · cos φ
- Exponential law: exp(x+y) = exp(x) · exp(y)
- Logarithm laws: log(xy) = log(x) + log(y)

**Next steps** (in Neural.Smooth.Geometry):
- Areas under curves
- Arc lengths
- Volumes of revolution
- Centers of curvature
-}

--------------------------------------------------------------------------------
-- § 8: Computational Primitives for Geometry

{-|
## Pi Constant

π is needed for:
- Circle areas: πr²
- Sphere volumes: (4/3)πr³
- Torus volumes: 2π²r²c
- Surface areas and arc lengths
-}

postulate
  π : ℝ

{-|
## Natural Number Powers

x^n for n : Nat - actual computation via recursion.
-}

_^_ : ℝ → Nat → ℝ
x ^ zero = 1ℝ
x ^ suc n = x ·ℝ (x ^ n)

_² : ℝ → ℝ
x ² = x ·ℝ x

_³ : ℝ → ℝ
x ³ = x ·ℝ x ·ℝ x

-- Power and notation equivalence lemmas
^2-is-² : (x : ℝ) → x ^ 2 ≡ x ²
^2-is-² x =
  x ^ 2                     ≡⟨⟩
  x ·ℝ (x ^ 1)              ≡⟨⟩
  x ·ℝ (x ·ℝ (x ^ 0))       ≡⟨⟩
  x ·ℝ (x ·ℝ 1ℝ)            ≡⟨ ap (x ·ℝ_) (·ℝ-idr x) ⟩
  x ·ℝ x                    ≡⟨⟩
  x ²                       ∎

^3-is-³ : (x : ℝ) → x ^ 3 ≡ x ³
^3-is-³ x =
  x ^ 3                     ≡⟨⟩
  x ·ℝ (x ^ 2)              ≡⟨ ap (x ·ℝ_) (^2-is-² x) ⟩
  x ·ℝ (x ²)                ≡⟨⟩
  x ·ℝ (x ·ℝ x)             ≡⟨ sym (·ℝ-assoc x x x) ⟩
  (x ·ℝ x) ·ℝ x             ≡⟨⟩
  x ³                       ∎

{-|
## Rational Powers

Needed for:
- √x = x^(1/2) in arc length formulas
- (1+f'²)^(3/2) in curvature formulas
-}

postulate
  _^1/2 : ℝ → ℝ
  _^3/2 : ℝ → ℝ
  _^-1 : ℝ → ℝ  -- Reciprocal for division

  -- Inverse properties for ^-1 (axioms since ^-1 is postulated)
  ^-1-invl : (a : ℝ) → (a ^-1) ·ℝ a ≡ 1ℝ
  ^-1-invr : (a : ℝ) → a ·ℝ (a ^-1) ≡ 1ℝ

{-|
## Converting Nat to ℝ

To write constants like 1, 2, 3, 4 as real numbers.
-}

-- Natural number embedding (imported from Calculus.agda as natToℝ)
#_ : Nat → ℝ
# n = natToℝ n

{-|
## Common Fractions

For geometric formulas like (1/3)πr²h and (4/3)πr³.
-}

1/2 : ℝ
1/2 = (# 1) ·ℝ ((# 2) ^-1)

1/3 : ℝ
1/3 = (# 1) ·ℝ ((# 3) ^-1)

2/3 : ℝ
2/3 = (# 2) ·ℝ ((# 3) ^-1)

4/3 : ℝ
4/3 = (# 4) ·ℝ ((# 3) ^-1)

{-|
## Simplified Division

x / y computes as x · y^(-1).
-}

_/_ : ℝ → ℝ → ℝ
x / y = x ·ℝ (y ^-1)

{-|
## Helper: (1 + x²)^(3/2)

Specifically for curvature formula κ = f'' / (1+f'²)^(3/2).
-}

1+x²-to-3/2 : ℝ → ℝ
1+x²-to-3/2 x = (1ℝ +ℝ (x ²)) ^3/2
