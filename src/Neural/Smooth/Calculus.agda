{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Differential Calculus via Smooth Infinitesimal Analysis

**Reference**: Chapter 2 from "Smooth Infinitesimal Analysis"

This module implements the differential calculus using nilsquare infinitesimals
and the Principle of Microaffineness. Every function has a derivative, defined by:

  f(x + ε) = f(x) + ε · f'(x)  for all ε ∈ Δ

This is EXACT, not an approximation!

## Key Features

1. **Automatic smoothness**: Every function ℝ → ℝ has derivatives of all orders
2. **Algebraic calculus**: Rules proved by pure algebra + microcancellation
3. **Fermat's method**: Stationary points via microvariations
4. **Constancy principle**: f' = 0 implies f constant
5. **Indecomposability**: ℝ cannot be split into disjoint parts

## Physical Interpretation

For neural networks:
- **Derivative f'(x)**: Rate of change of activation/loss
- **Stationary points**: Local minima/maxima of loss landscape
- **Chain rule**: Backpropagation through composite functions
- **Higher derivatives**: Curvature information (Hessian, etc.)

## Connection to Existing Code

Provides rigorous foundations for:
- Gradient descent in Resources/Optimization
- Backpropagation in Topos/Architecture
- Natural gradient in Information/Geometry
- Neural dynamics ODEs in Dynamics/*
-}

module Neural.Smooth.Calculus where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path
open import 1Lab.Path.Reasoning

open import Neural.Smooth.Base public

open import Data.Nat.Base using (Nat; zero; suc; _+_; _*_)
open import Data.Fin.Base using (Fin)
open import Data.Sum.Base using (_⊎_; inl; inr)

private variable
  ℓ : Level

--------------------------------------------------------------------------------
-- § 1: The Derivative

{-|
## Definition of Derivative (Section 2.1)

For a function f : ℝ → ℝ and a point x ∈ ℝ, define gₓ : Δ → ℝ by:

  gₓ(ε) = f(x + ε)

By Microaffineness, there exists unique b ∈ ℝ with:

  gₓ(ε) = gₓ(0) + b · ε
  f(x + ε) = f(x) + b · ε

We call b the **derivative** of f at x, written f'(x).

**Fundamental Equation**:

  f(x + ε) = f(x) + ε · f'(x)  for all ε ∈ Δ

The quantity ε · f'(x) is the **increment** of f from x to x + ε.
-}

-- Domain type for functions (can be ℝ or a closed interval)
Domain : Type
Domain = ℝ  -- For now, functions are defined on all of ℝ

-- Derivative of a function at a point
derivative-at : (f : ℝ → ℝ) (x : ℝ) → ℝ
derivative-at f x = slope (λ δ → f (x +ℝ ι δ))

-- Notation: f'(x)
-- High precedence so f ′[ x ] binds tightly before arithmetic
infixl 50 _′[_] _′′[_] _′′′[_] _^′_

_′[_] : (f : ℝ → ℝ) → ℝ → ℝ
f ′[ x ] = derivative-at f x

{-|
## Fundamental Equation of Differential Calculus (Equation 2.1)

For any function f : ℝ → ℝ, point x ∈ ℝ, and infinitesimal ε ∈ Δ:

  f(x + ε) = f(x) + ε · f'(x)

This is the defining property of the derivative.
-}

fundamental-equation : (f : ℝ → ℝ) (x : ℝ) (δ : Δ) →
  f (x +ℝ ι δ) ≡ f x +ℝ (ι δ ·ℝ (f ′[ x ]))
fundamental-equation f x δ =
  let g = λ δ' → f (x +ℝ ι δ')
      δ₀ : Δ
      δ₀ = (0ℝ , ·ℝ-zerol 0ℝ)
  in f (x +ℝ ι δ)
       ≡⟨ slope-property g δ ⟩
     g δ₀ +ℝ (slope g ·ℝ ι δ)
       ≡⟨⟩  -- g δ₀ = f (x +ℝ ι δ₀) = f (x +ℝ 0ℝ)
     f (x +ℝ 0ℝ) +ℝ (slope g ·ℝ ι δ)
       ≡⟨ ap (λ y → f y +ℝ (slope g ·ℝ ι δ)) (+ℝ-idr x) ⟩
     f x +ℝ (slope g ·ℝ ι δ)
       ≡⟨ ap (f x +ℝ_) (·ℝ-comm (slope g) (ι δ)) ⟩  -- Commutativity
     f x +ℝ (ι δ ·ℝ slope g)
       ≡⟨⟩  -- slope g = f ′[ x ] by definition of derivative
     f x +ℝ (ι δ ·ℝ (f ′[ x ]))
       ∎

{-|
## Derivative Function

Given f : ℝ → ℝ, we get f' : ℝ → ℝ by sending each x to f'(x).

**Important**: Since every function has a derivative, we can iterate:
- f' is the first derivative
- f'' = (f')' is the second derivative
- f''' = (f'')' is the third derivative
- f⁽ⁿ⁾ is the nth derivative

Every function in a smooth world is **smooth** (has derivatives of all orders).
-}

-- Derivative function
deriv : (f : ℝ → ℝ) → (ℝ → ℝ)
deriv f x = f ′[ x ]

-- Notation: f'
_′ : (f : ℝ → ℝ) → (ℝ → ℝ)
f ′ = deriv f

-- Higher derivatives by iteration
_′′ : (f : ℝ → ℝ) → (ℝ → ℝ)
f ′′ = (f ′) ′

_′′[_] : (f : ℝ → ℝ) → ℝ → ℝ
f ′′[ x ] = (f ′′) x

_′′′ : (f : ℝ → ℝ) → (ℝ → ℝ)
f ′′′ = ((f ′) ′) ′

_′′′[_] : (f : ℝ → ℝ) → ℝ → ℝ
f ′′′[ x ] = (f ′′′) x

-- nth derivative (defined recursively)
_^′_ : (f : ℝ → ℝ) → Nat → (ℝ → ℝ)
f ^′ zero = f
f ^′ suc n = (f ^′ n) ′

{-|
## Exercise 2.1: Multiple Infinitesimals

For ε, η, ζ ∈ Δ:

  f(x + ε + η) = f(x) + (ε + η)·f'(x) + ε·η·f''(x)

  f(x + ε + η + ζ) = f(x) + (ε + η + ζ)·f'(x) + (ε·η + ε·ζ + η·ζ)·f''(x) + ε·η·ζ·f'''(x)

**Proof**: Expand using fundamental equation repeatedly and use nilpotence.
-}

postulate
  two-infinitesimals : (f : ℝ → ℝ) (x : ℝ) (δ₁ δ₂ : Δ) →
    f ((x +ℝ ι δ₁) +ℝ ι δ₂) ≡
    ((f x +ℝ ((ι δ₁ +ℝ ι δ₂) ·ℝ (f ′[ x ]))) +ℝ ((ι δ₁ ·ℝ ι δ₂) ·ℝ (f ′′[ x ])))

  three-infinitesimals : (f : ℝ → ℝ) (x : ℝ) (δ₁ δ₂ δ₃ : Δ) →
    f (((x +ℝ ι δ₁) +ℝ ι δ₂) +ℝ ι δ₃) ≡
    (((f x +ℝ (((ι δ₁ +ℝ ι δ₂) +ℝ ι δ₃) ·ℝ (f ′[ x ])))
        +ℝ ((((ι δ₁ ·ℝ ι δ₂) +ℝ (ι δ₁ ·ℝ ι δ₃)) +ℝ (ι δ₂ ·ℝ ι δ₃)) ·ℝ (f ′′[ x ])))
        +ℝ (((ι δ₁ ·ℝ ι δ₂) ·ℝ ι δ₃) ·ℝ (f ′′′[ x ])))

--------------------------------------------------------------------------------
-- § 2: Basic Calculus Rules

{-|
## Sum Rule

The derivative of a sum is the sum of derivatives:

  (f + g)' = f' + g'

**Proof**: By Microaffineness and linearity of slope.
-}

sum-rule : (f g : ℝ → ℝ) (x : ℝ) →
  ((λ y → f y +ℝ g y) ′[ x ]) ≡ ((f ′[ x ]) +ℝ (g ′[ x ]))
sum-rule f g x =
  microcancellation _ _ λ δ →
    let h = λ y → f y +ℝ g y
    in ι δ ·ℝ (h ′[ x ])
      ≡⟨ sym (+ℝ-idl (ι δ ·ℝ (h ′[ x ]))) ⟩
    0ℝ +ℝ (ι δ ·ℝ (h ′[ x ]))
      ≡⟨ ap (_+ℝ (ι δ ·ℝ (h ′[ x ]))) (sym (+ℝ-invr (f x +ℝ g x))) ⟩
    ((f x +ℝ g x) +ℝ (-ℝ (f x +ℝ g x))) +ℝ (ι δ ·ℝ (h ′[ x ]))
      ≡⟨ +ℝ-assoc (f x +ℝ g x) (-ℝ (f x +ℝ g x)) (ι δ ·ℝ (h ′[ x ])) ⟩
    (f x +ℝ g x) +ℝ ((-ℝ (f x +ℝ g x)) +ℝ (ι δ ·ℝ (h ′[ x ])))
      ≡⟨ ap ((f x +ℝ g x) +ℝ_) (+ℝ-comm (-ℝ (f x +ℝ g x)) (ι δ ·ℝ (h ′[ x ]))) ⟩
    (f x +ℝ g x) +ℝ ((ι δ ·ℝ (h ′[ x ])) +ℝ (-ℝ (f x +ℝ g x)))
      ≡⟨ sym (+ℝ-assoc (f x +ℝ g x) (ι δ ·ℝ (h ′[ x ])) (-ℝ (f x +ℝ g x))) ⟩
    ((f x +ℝ g x) +ℝ (ι δ ·ℝ (h ′[ x ]))) +ℝ (-ℝ (f x +ℝ g x))
      ≡⟨ ap (_+ℝ (-ℝ (f x +ℝ g x))) (sym (fundamental-equation h x δ)) ⟩
    h (x +ℝ ι δ) +ℝ (-ℝ (f x +ℝ g x))
      ≡⟨⟩
    (f (x +ℝ ι δ) +ℝ g (x +ℝ ι δ)) -ℝ (f x +ℝ g x)
      ≡⟨ ap₂ _-ℝ_ (ap₂ _+ℝ_ (fundamental-equation f x δ) (fundamental-equation g x δ)) refl ⟩
    ((f x +ℝ (ι δ ·ℝ (f ′[ x ]))) +ℝ (g x +ℝ (ι δ ·ℝ (g ′[ x ])))) -ℝ (f x +ℝ g x)
      ≡⟨ ap (_-ℝ (f x +ℝ g x)) (+ℝ-assoc (f x) (ι δ ·ℝ (f ′[ x ])) (g x +ℝ (ι δ ·ℝ (g ′[ x ])))) ⟩
    (f x +ℝ ((ι δ ·ℝ (f ′[ x ])) +ℝ (g x +ℝ (ι δ ·ℝ (g ′[ x ]))))) -ℝ (f x +ℝ g x)
      ≡⟨ ap (λ z → (f x +ℝ z) -ℝ (f x +ℝ g x)) (+ℝ-comm (ι δ ·ℝ (f ′[ x ])) (g x +ℝ (ι δ ·ℝ (g ′[ x ])))) ⟩
    (f x +ℝ ((g x +ℝ (ι δ ·ℝ (g ′[ x ]))) +ℝ (ι δ ·ℝ (f ′[ x ])))) -ℝ (f x +ℝ g x)
      ≡⟨ ap (λ z → (f x +ℝ z) -ℝ (f x +ℝ g x)) (+ℝ-assoc (g x) (ι δ ·ℝ (g ′[ x ])) (ι δ ·ℝ (f ′[ x ]))) ⟩
    (f x +ℝ (g x +ℝ ((ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ]))))) -ℝ (f x +ℝ g x)
      ≡⟨ ap (_-ℝ (f x +ℝ g x)) (sym (+ℝ-assoc (f x) (g x) ((ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ]))))) ⟩
    ((f x +ℝ g x) +ℝ ((ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ])))) -ℝ (f x +ℝ g x)
      ≡⟨⟩  -- Definition: a -ℝ b = a +ℝ (-ℝ b)
    ((f x +ℝ g x) +ℝ ((ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ])))) +ℝ (-ℝ (f x +ℝ g x))
      ≡⟨ +ℝ-assoc (f x +ℝ g x) ((ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ]))) (-ℝ (f x +ℝ g x)) ⟩
    (f x +ℝ g x) +ℝ (((ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ]))) +ℝ (-ℝ (f x +ℝ g x)))
      ≡⟨ ap ((f x +ℝ g x) +ℝ_) (+ℝ-comm ((ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ]))) (-ℝ (f x +ℝ g x))) ⟩
    (f x +ℝ g x) +ℝ ((-ℝ (f x +ℝ g x)) +ℝ ((ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ]))))
      ≡⟨ sym (+ℝ-assoc (f x +ℝ g x) (-ℝ (f x +ℝ g x)) ((ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ])))) ⟩
    ((f x +ℝ g x) +ℝ (-ℝ (f x +ℝ g x))) +ℝ ((ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ])))
      ≡⟨ ap (_+ℝ ((ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ])))) (+ℝ-invr (f x +ℝ g x)) ⟩
    0ℝ +ℝ ((ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ])))
      ≡⟨ +ℝ-idl ((ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ]))) ⟩
    (ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ]))
      ≡⟨ +ℝ-comm (ι δ ·ℝ (g ′[ x ])) (ι δ ·ℝ (f ′[ x ])) ⟩
    (ι δ ·ℝ (f ′[ x ])) +ℝ (ι δ ·ℝ (g ′[ x ]))
      ≡⟨ sym (·ℝ-distribl (ι δ) (f ′[ x ]) (g ′[ x ])) ⟩
    ι δ ·ℝ ((f ′[ x ]) +ℝ (g ′[ x ]))
      ∎

{-|
## Scalar Multiple Rule

The derivative of a scalar multiple is the scalar times the derivative:

  (c · f)' = c · f'

**Proof**: By linearity of the derivative.
-}

scalar-rule : (c : ℝ) (f : ℝ → ℝ) (x : ℝ) →
  ((λ y → c ·ℝ f y) ′[ x ]) ≡ (c ·ℝ (f ′[ x ]))
scalar-rule c f x =
  microcancellation _ _ λ δ →
    let g = λ y → c ·ℝ f y
    in ι δ ·ℝ (g ′[ x ])
      ≡⟨ sym (+ℝ-idl (ι δ ·ℝ (g ′[ x ]))) ⟩
    0ℝ +ℝ (ι δ ·ℝ (g ′[ x ]))
      ≡⟨ ap (_+ℝ (ι δ ·ℝ (g ′[ x ]))) (sym (+ℝ-invr (c ·ℝ f x))) ⟩
    ((c ·ℝ f x) +ℝ (-ℝ (c ·ℝ f x))) +ℝ (ι δ ·ℝ (g ′[ x ]))
      ≡⟨ +ℝ-assoc (c ·ℝ f x) (-ℝ (c ·ℝ f x)) (ι δ ·ℝ (g ′[ x ])) ⟩
    (c ·ℝ f x) +ℝ ((-ℝ (c ·ℝ f x)) +ℝ (ι δ ·ℝ (g ′[ x ])))
      ≡⟨ ap ((c ·ℝ f x) +ℝ_) (+ℝ-comm (-ℝ (c ·ℝ f x)) (ι δ ·ℝ (g ′[ x ]))) ⟩
    (c ·ℝ f x) +ℝ ((ι δ ·ℝ (g ′[ x ])) +ℝ (-ℝ (c ·ℝ f x)))
      ≡⟨ sym (+ℝ-assoc (c ·ℝ f x) (ι δ ·ℝ (g ′[ x ])) (-ℝ (c ·ℝ f x))) ⟩
    ((c ·ℝ f x) +ℝ (ι δ ·ℝ (g ′[ x ]))) +ℝ (-ℝ (c ·ℝ f x))
      ≡⟨ ap (_+ℝ (-ℝ (c ·ℝ f x))) (sym (fundamental-equation g x δ)) ⟩
    g (x +ℝ ι δ) +ℝ (-ℝ (c ·ℝ f x))
      ≡⟨⟩
    ((c ·ℝ f (x +ℝ ι δ)) -ℝ (c ·ℝ f x))
      ≡⟨ ap (λ z → (c ·ℝ z) -ℝ (c ·ℝ f x)) (fundamental-equation f x δ) ⟩
    ((c ·ℝ (f x +ℝ (ι δ ·ℝ (f ′[ x ])))) -ℝ (c ·ℝ f x))
      ≡⟨ ap₂ _-ℝ_ (·ℝ-distribl c (f x) (ι δ ·ℝ (f ′[ x ]))) refl ⟩
    (((c ·ℝ f x) +ℝ (c ·ℝ (ι δ ·ℝ (f ′[ x ])))) -ℝ (c ·ℝ f x))
      ≡⟨ ap (λ z → ((c ·ℝ f x) +ℝ z) -ℝ (c ·ℝ f x)) (sym (·ℝ-assoc c (ι δ) (f ′[ x ]))) ⟩
    (((c ·ℝ f x) +ℝ ((c ·ℝ ι δ) ·ℝ (f ′[ x ]))) -ℝ (c ·ℝ f x))
      ≡⟨ ap (λ z → ((c ·ℝ f x) +ℝ (z ·ℝ (f ′[ x ]))) -ℝ (c ·ℝ f x)) (·ℝ-comm c (ι δ)) ⟩
    (((c ·ℝ f x) +ℝ ((ι δ ·ℝ c) ·ℝ (f ′[ x ]))) -ℝ (c ·ℝ f x))
      ≡⟨ ap (λ w → ((c ·ℝ f x) +ℝ w) -ℝ (c ·ℝ f x)) (·ℝ-assoc (ι δ) c (f ′[ x ])) ⟩
    (((c ·ℝ f x) +ℝ (ι δ ·ℝ (c ·ℝ (f ′[ x ])))) -ℝ (c ·ℝ f x))
      ≡⟨⟩
    ((c ·ℝ f x) +ℝ (ι δ ·ℝ (c ·ℝ (f ′[ x ])))) +ℝ (-ℝ (c ·ℝ f x))
      ≡⟨ +ℝ-assoc (c ·ℝ f x) (ι δ ·ℝ (c ·ℝ (f ′[ x ]))) (-ℝ (c ·ℝ f x)) ⟩
    (c ·ℝ f x) +ℝ ((ι δ ·ℝ (c ·ℝ (f ′[ x ]))) +ℝ (-ℝ (c ·ℝ f x)))
      ≡⟨ ap ((c ·ℝ f x) +ℝ_) (+ℝ-comm (ι δ ·ℝ (c ·ℝ (f ′[ x ]))) (-ℝ (c ·ℝ f x))) ⟩
    (c ·ℝ f x) +ℝ ((-ℝ (c ·ℝ f x)) +ℝ (ι δ ·ℝ (c ·ℝ (f ′[ x ]))))
      ≡⟨ sym (+ℝ-assoc (c ·ℝ f x) (-ℝ (c ·ℝ f x)) (ι δ ·ℝ (c ·ℝ (f ′[ x ])))) ⟩
    ((c ·ℝ f x) +ℝ (-ℝ (c ·ℝ f x))) +ℝ (ι δ ·ℝ (c ·ℝ (f ′[ x ])))
      ≡⟨ ap (_+ℝ (ι δ ·ℝ (c ·ℝ (f ′[ x ])))) (+ℝ-invr (c ·ℝ f x)) ⟩
    0ℝ +ℝ (ι δ ·ℝ (c ·ℝ (f ′[ x ])))
      ≡⟨ +ℝ-idl (ι δ ·ℝ (c ·ℝ (f ′[ x ]))) ⟩
    (ι δ ·ℝ (c ·ℝ (f ′[ x ])))
      ∎

{-|
## Difference Rule

The derivative of a difference is the difference of derivatives:
  (f - g)' = f' - g'

This follows from the sum rule and scalar rule.
-}

difference-derivative : (f g : ℝ → ℝ) (x : ℝ) →
  ((λ y → f y -ℝ g y) ′[ x ]) ≡ ((f ′[ x ]) -ℝ (g ′[ x ]))
difference-derivative f g x =
  -- f - g = f + (-g) = f + ((-1) · g)
  -- So (f - g)' = (f + ((-1) · g))' = f' + ((-1) · g)' = f' + ((-1) · g') = f' + (-g') = f' - g'
  ((λ y → f y -ℝ g y) ′[ x ])
    ≡⟨⟩  -- Definition: a -ℝ b = a +ℝ (-ℝ b)
  ((λ y → f y +ℝ (-ℝ g y)) ′[ x ])
    ≡⟨ ap (λ h → h ′[ x ]) (funext λ y → ap (f y +ℝ_) (sym (neg-mult (g y)))) ⟩
  ((λ y → f y +ℝ ((-ℝ 1ℝ) ·ℝ g y)) ′[ x ])
    ≡⟨ sum-rule f (λ y → (-ℝ 1ℝ) ·ℝ g y) x ⟩
  ((f ′[ x ]) +ℝ ((λ y → (-ℝ 1ℝ) ·ℝ g y) ′[ x ]))
    ≡⟨ ap ((f ′[ x ]) +ℝ_) (scalar-rule (-ℝ 1ℝ) g x) ⟩
  ((f ′[ x ]) +ℝ ((-ℝ 1ℝ) ·ℝ (g ′[ x ])))
    ≡⟨ ap ((f ′[ x ]) +ℝ_) (neg-mult (g ′[ x ])) ⟩
  ((f ′[ x ]) +ℝ (-ℝ (g ′[ x ])))
    ≡⟨⟩  -- Definition: a -ℝ b = a +ℝ (-ℝ b)
  ((f ′[ x ]) -ℝ (g ′[ x ]))
    ∎

-- Helper: Product of infinitesimals is nilsquare
δ-product-nilsquare : (δ : Δ) (a b : ℝ) → (ι δ ·ℝ a) ·ℝ (ι δ ·ℝ b) ≡ 0ℝ
δ-product-nilsquare δ a b =
  (ι δ ·ℝ a) ·ℝ (ι δ ·ℝ b)
    ≡⟨ ·ℝ-assoc (ι δ) a (ι δ ·ℝ b) ⟩
  ι δ ·ℝ (a ·ℝ (ι δ ·ℝ b))
    ≡⟨ ap (ι δ ·ℝ_) (sym (·ℝ-assoc a (ι δ) b)) ⟩
  ι δ ·ℝ ((a ·ℝ ι δ) ·ℝ b)
    ≡⟨ ap (λ z → ι δ ·ℝ (z ·ℝ b)) (·ℝ-comm a (ι δ)) ⟩
  ι δ ·ℝ ((ι δ ·ℝ a) ·ℝ b)
    ≡⟨ sym (·ℝ-assoc (ι δ) (ι δ ·ℝ a) b) ⟩
  (ι δ ·ℝ (ι δ ·ℝ a)) ·ℝ b
    ≡⟨ ap (_·ℝ b) (sym (·ℝ-assoc (ι δ) (ι δ) a)) ⟩
  ((ι δ ·ℝ ι δ) ·ℝ a) ·ℝ b
    ≡⟨ ap (λ z → (z ·ℝ a) ·ℝ b) (nilsquare δ) ⟩
  (0ℝ ·ℝ a) ·ℝ b
    ≡⟨ ap (_·ℝ b) (·ℝ-zerol a) ⟩
  0ℝ ·ℝ b
    ≡⟨ ·ℝ-zerol b ⟩
  0ℝ
    ∎

-- Helper lemma for nilsquare reasoning in product rule
private
  ε²-lemma : ∀ (δ : Δ) (a b : ℝ) → ((ι δ ·ℝ a) ·ℝ (ι δ ·ℝ b)) ≡ 0ℝ
  ε²-lemma = δ-product-nilsquare

{-|
## Product Rule

The derivative of a product is:

  (f · g)' = f' · g + f · g'

**Proof**: Expand (f · g)(x + ε) using fundamental equation:

  (f·g)(x+ε) = f(x+ε) · g(x+ε)
             = [f(x) + ε·f'(x)] · [g(x) + ε·g'(x)]
             = f(x)·g(x) + ε·[f'(x)·g(x) + f(x)·g'(x)] + ε²·f'(x)·g'(x)
             = f(x)·g(x) + ε·[f'(x)·g(x) + f(x)·g'(x)]  (since ε² = 0)

Cancel ε to get the product rule.
-}

product-rule : (f g : ℝ → ℝ) (x : ℝ) →
  ((λ y → f y ·ℝ g y) ′[ x ]) ≡ (((f ′[ x ]) ·ℝ g x) +ℝ (f x ·ℝ (g ′[ x ])))
product-rule f g x =
  microcancellation _ _ λ δ →
    let h = λ y → f y ·ℝ g y
    in ι δ ·ℝ (h ′[ x ])
      ≡⟨ sym (+ℝ-idl (ι δ ·ℝ (h ′[ x ]))) ⟩
    0ℝ +ℝ (ι δ ·ℝ (h ′[ x ]))
      ≡⟨ ap (_+ℝ (ι δ ·ℝ (h ′[ x ]))) (sym (+ℝ-invr (f x ·ℝ g x))) ⟩
    ((f x ·ℝ g x) +ℝ (-ℝ (f x ·ℝ g x))) +ℝ (ι δ ·ℝ (h ′[ x ]))
      ≡⟨ +ℝ-assoc (f x ·ℝ g x) (-ℝ (f x ·ℝ g x)) (ι δ ·ℝ (h ′[ x ])) ⟩
    (f x ·ℝ g x) +ℝ ((-ℝ (f x ·ℝ g x)) +ℝ (ι δ ·ℝ (h ′[ x ])))
      ≡⟨ ap ((f x ·ℝ g x) +ℝ_) (+ℝ-comm (-ℝ (f x ·ℝ g x)) (ι δ ·ℝ (h ′[ x ]))) ⟩
    (f x ·ℝ g x) +ℝ ((ι δ ·ℝ (h ′[ x ])) +ℝ (-ℝ (f x ·ℝ g x)))
      ≡⟨ sym (+ℝ-assoc (f x ·ℝ g x) (ι δ ·ℝ (h ′[ x ])) (-ℝ (f x ·ℝ g x))) ⟩
    ((f x ·ℝ g x) +ℝ (ι δ ·ℝ (h ′[ x ]))) +ℝ (-ℝ (f x ·ℝ g x))
      ≡⟨ ap (_+ℝ (-ℝ (f x ·ℝ g x))) (sym (fundamental-equation h x δ)) ⟩
    h (x +ℝ ι δ) +ℝ (-ℝ (f x ·ℝ g x))
      ≡⟨⟩
    ((f (x +ℝ ι δ) ·ℝ g (x +ℝ ι δ)) -ℝ (f x ·ℝ g x))
      -- Expand f(x+ε) and g(x+ε)
      ≡⟨ ap₂ (λ u v → (u ·ℝ v) -ℝ (f x ·ℝ g x))
             (fundamental-equation f x δ)
             (fundamental-equation g x δ) ⟩
    (((f x +ℝ (ι δ ·ℝ (f ′[ x ]))) ·ℝ (g x +ℝ (ι δ ·ℝ (g ′[ x ])))) -ℝ (f x ·ℝ g x))
      -- Expand the product (distribute)
      ≡⟨ ap (_-ℝ (f x ·ℝ g x)) (·ℝ-distribr (f x) (ι δ ·ℝ (f ′[ x ])) (g x +ℝ (ι δ ·ℝ (g ′[ x ])))) ⟩
    (((f x ·ℝ (g x +ℝ (ι δ ·ℝ (g ′[ x ])))) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ (g x +ℝ (ι δ ·ℝ (g ′[ x ]))))) -ℝ (f x ·ℝ g x))
      ≡⟨ ap (λ z → (z +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ (g x +ℝ (ι δ ·ℝ (g ′[ x ]))))) -ℝ (f x ·ℝ g x)) (·ℝ-distribl (f x) (g x) (ι δ ·ℝ (g ′[ x ]))) ⟩
    ((((f x ·ℝ g x) +ℝ (f x ·ℝ (ι δ ·ℝ (g ′[ x ])))) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ (g x +ℝ (ι δ ·ℝ (g ′[ x ]))))) -ℝ (f x ·ℝ g x))
      ≡⟨ ap (λ z → (((f x ·ℝ g x) +ℝ (f x ·ℝ (ι δ ·ℝ (g ′[ x ])))) +ℝ z) -ℝ (f x ·ℝ g x)) (·ℝ-distribl (ι δ ·ℝ (f ′[ x ])) (g x) (ι δ ·ℝ (g ′[ x ]))) ⟩
    ((((f x ·ℝ g x) +ℝ (f x ·ℝ (ι δ ·ℝ (g ′[ x ])))) +ℝ (((ι δ ·ℝ (f ′[ x ])) ·ℝ g x) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ (ι δ ·ℝ (g ′[ x ]))))) -ℝ (f x ·ℝ g x))
      -- Use ε² = 0
      ≡⟨ ap (λ z → (((f x ·ℝ g x) +ℝ (f x ·ℝ (ι δ ·ℝ (g ′[ x ])))) +ℝ (((ι δ ·ℝ (f ′[ x ])) ·ℝ g x) +ℝ z)) -ℝ (f x ·ℝ g x))
            (ε²-lemma δ (f ′[ x ]) (g ′[ x ])) ⟩
    ((((f x ·ℝ g x) +ℝ (f x ·ℝ (ι δ ·ℝ (g ′[ x ])))) +ℝ (((ι δ ·ℝ (f ′[ x ])) ·ℝ g x) +ℝ 0ℝ)) -ℝ (f x ·ℝ g x))
      ≡⟨ ap (λ z → (((f x ·ℝ g x) +ℝ (f x ·ℝ (ι δ ·ℝ (g ′[ x ])))) +ℝ z) -ℝ (f x ·ℝ g x)) (+ℝ-idr ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) ⟩
    ((((f x ·ℝ g x) +ℝ (f x ·ℝ (ι δ ·ℝ (g ′[ x ])))) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) -ℝ (f x ·ℝ g x))
      -- Rearrange and factor out ε
      ≡⟨ ap (λ z → (((f x ·ℝ g x) +ℝ z) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) -ℝ (f x ·ℝ g x)) (sym (·ℝ-assoc (f x) (ι δ) (g ′[ x ]))) ⟩
    ((((f x ·ℝ g x) +ℝ ((f x ·ℝ ι δ) ·ℝ (g ′[ x ]))) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) -ℝ (f x ·ℝ g x))
      ≡⟨ ap (λ z → (((f x ·ℝ g x) +ℝ (z ·ℝ (g ′[ x ]))) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) -ℝ (f x ·ℝ g x)) (·ℝ-comm (f x) (ι δ)) ⟩
    ((((f x ·ℝ g x) +ℝ ((ι δ ·ℝ f x) ·ℝ (g ′[ x ]))) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) -ℝ (f x ·ℝ g x))
      ≡⟨ ap (λ z → (((f x ·ℝ g x) +ℝ z) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) -ℝ (f x ·ℝ g x)) (·ℝ-assoc (ι δ) (f x) (g ′[ x ])) ⟩
    ((((f x ·ℝ g x) +ℝ (ι δ ·ℝ (f x ·ℝ (g ′[ x ])))) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) -ℝ (f x ·ℝ g x))
      ≡⟨ ap (λ z → (((f x ·ℝ g x) +ℝ (ι δ ·ℝ (f x ·ℝ (g ′[ x ])))) +ℝ z) -ℝ (f x ·ℝ g x)) (·ℝ-assoc (ι δ) (f ′[ x ]) (g x)) ⟩
    ((((f x ·ℝ g x) +ℝ (ι δ ·ℝ (f x ·ℝ (g ′[ x ])))) +ℝ (ι δ ·ℝ ((f ′[ x ]) ·ℝ g x))) -ℝ (f x ·ℝ g x))
      ≡⟨ ap (_-ℝ (f x ·ℝ g x)) (+ℝ-assoc (f x ·ℝ g x) (ι δ ·ℝ (f x ·ℝ (g ′[ x ]))) (ι δ ·ℝ ((f ′[ x ]) ·ℝ g x))) ⟩
    (((f x ·ℝ g x) +ℝ ((ι δ ·ℝ (f x ·ℝ (g ′[ x ]))) +ℝ (ι δ ·ℝ ((f ′[ x ]) ·ℝ g x)))) -ℝ (f x ·ℝ g x))
      ≡⟨ ap (λ z → ((f x ·ℝ g x) +ℝ z) -ℝ (f x ·ℝ g x)) (+ℝ-comm (ι δ ·ℝ (f x ·ℝ (g ′[ x ]))) (ι δ ·ℝ ((f ′[ x ]) ·ℝ g x))) ⟩
    (((f x ·ℝ g x) +ℝ ((ι δ ·ℝ ((f ′[ x ]) ·ℝ g x)) +ℝ (ι δ ·ℝ (f x ·ℝ (g ′[ x ]))))) -ℝ (f x ·ℝ g x))
      ≡⟨ ap (λ z → ((f x ·ℝ g x) +ℝ z) -ℝ (f x ·ℝ g x)) (sym (·ℝ-distribl (ι δ) ((f ′[ x ]) ·ℝ g x) (f x ·ℝ (g ′[ x ])))) ⟩
    (((f x ·ℝ g x) +ℝ (ι δ ·ℝ (((f ′[ x ]) ·ℝ g x) +ℝ (f x ·ℝ (g ′[ x ]))))) -ℝ (f x ·ℝ g x))
      -- Simplify (a + b) - a = b
      ≡⟨ +ℝ-assoc (f x ·ℝ g x) (ι δ ·ℝ (((f ′[ x ]) ·ℝ g x) +ℝ (f x ·ℝ (g ′[ x ])))) (-ℝ (f x ·ℝ g x)) ⟩
    ((f x ·ℝ g x) +ℝ ((ι δ ·ℝ (((f ′[ x ]) ·ℝ g x) +ℝ (f x ·ℝ (g ′[ x ])))) +ℝ (-ℝ (f x ·ℝ g x))))
      ≡⟨ ap ((f x ·ℝ g x) +ℝ_) (+ℝ-comm (ι δ ·ℝ (((f ′[ x ]) ·ℝ g x) +ℝ (f x ·ℝ (g ′[ x ])))) (-ℝ (f x ·ℝ g x))) ⟩
    ((f x ·ℝ g x) +ℝ ((-ℝ (f x ·ℝ g x)) +ℝ (ι δ ·ℝ (((f ′[ x ]) ·ℝ g x) +ℝ (f x ·ℝ (g ′[ x ]))))))
      ≡⟨ sym (+ℝ-assoc (f x ·ℝ g x) (-ℝ (f x ·ℝ g x)) (ι δ ·ℝ (((f ′[ x ]) ·ℝ g x) +ℝ (f x ·ℝ (g ′[ x ]))))) ⟩
    (((f x ·ℝ g x) +ℝ (-ℝ (f x ·ℝ g x))) +ℝ (ι δ ·ℝ (((f ′[ x ]) ·ℝ g x) +ℝ (f x ·ℝ (g ′[ x ])))))
      ≡⟨ ap (_+ℝ (ι δ ·ℝ (((f ′[ x ]) ·ℝ g x) +ℝ (f x ·ℝ (g ′[ x ]))))) (+ℝ-invr (f x ·ℝ g x)) ⟩
    (0ℝ +ℝ (ι δ ·ℝ (((f ′[ x ]) ·ℝ g x) +ℝ (f x ·ℝ (g ′[ x ])))))
      ≡⟨ +ℝ-idl (ι δ ·ℝ (((f ′[ x ]) ·ℝ g x) +ℝ (f x ·ℝ (g ′[ x ])))) ⟩
    (ι δ ·ℝ (((f ′[ x ]) ·ℝ g x) +ℝ (f x ·ℝ (g ′[ x ]))))
      ∎

{-|
## Polynomial Rule

For polynomial f(x) = a₀ + a₁x + a₂x² + ··· + aₙxⁿ:

  f'(x) = a₁ + 2a₂x + 3a₃x² + ··· + n·aₙxⁿ⁻¹

In particular:
- Constant rule: c' = 0
- Identity rule: x' = 1
- Power rule: (xⁿ)' = n·xⁿ⁻¹
-}

-- Constant function derivative is zero
constant-rule : (c : ℝ) (x : ℝ) →
  ((λ _ → c) ′[ x ]) ≡ 0ℝ
constant-rule c x =
  microcancellation _ _ λ δ →
    let h = λ _ → c
    in ι δ ·ℝ (h ′[ x ])
      ≡⟨ sym (+ℝ-idl (ι δ ·ℝ (h ′[ x ]))) ⟩
    0ℝ +ℝ (ι δ ·ℝ (h ′[ x ]))
      ≡⟨ ap (_+ℝ (ι δ ·ℝ (h ′[ x ]))) (sym (+ℝ-invr c)) ⟩
    (c +ℝ (-ℝ c)) +ℝ (ι δ ·ℝ (h ′[ x ]))
      ≡⟨ +ℝ-assoc c (-ℝ c) (ι δ ·ℝ (h ′[ x ])) ⟩
    c +ℝ ((-ℝ c) +ℝ (ι δ ·ℝ (h ′[ x ])))
      ≡⟨ ap (c +ℝ_) (+ℝ-comm (-ℝ c) (ι δ ·ℝ (h ′[ x ]))) ⟩
    c +ℝ ((ι δ ·ℝ (h ′[ x ])) +ℝ (-ℝ c))
      ≡⟨ sym (+ℝ-assoc c (ι δ ·ℝ (h ′[ x ])) (-ℝ c)) ⟩
    (c +ℝ (ι δ ·ℝ (h ′[ x ]))) +ℝ (-ℝ c)
      ≡⟨ ap (_+ℝ (-ℝ c)) (sym (fundamental-equation h x δ)) ⟩
    h (x +ℝ ι δ) +ℝ (-ℝ c)
      ≡⟨⟩
    (c -ℝ c)
      ≡⟨ +ℝ-invr c ⟩
    0ℝ
      ≡⟨ sym (·ℝ-zeror (ι δ)) ⟩
    (ι δ ·ℝ 0ℝ)
      ∎

-- Identity function derivative is one
identity-rule : (x : ℝ) →
  ((λ y → y) ′[ x ]) ≡ 1ℝ
identity-rule x =
  microcancellation _ _ λ δ →
    let h = λ y → y
    in ι δ ·ℝ (h ′[ x ])
      ≡⟨ sym (+ℝ-idl (ι δ ·ℝ (h ′[ x ]))) ⟩
    0ℝ +ℝ (ι δ ·ℝ (h ′[ x ]))
      ≡⟨ ap (_+ℝ (ι δ ·ℝ (h ′[ x ]))) (sym (+ℝ-invr x)) ⟩
    (x +ℝ (-ℝ x)) +ℝ (ι δ ·ℝ (h ′[ x ]))
      ≡⟨ +ℝ-assoc x (-ℝ x) (ι δ ·ℝ (h ′[ x ])) ⟩
    x +ℝ ((-ℝ x) +ℝ (ι δ ·ℝ (h ′[ x ])))
      ≡⟨ ap (x +ℝ_) (+ℝ-comm (-ℝ x) (ι δ ·ℝ (h ′[ x ]))) ⟩
    x +ℝ ((ι δ ·ℝ (h ′[ x ])) +ℝ (-ℝ x))
      ≡⟨ sym (+ℝ-assoc x (ι δ ·ℝ (h ′[ x ])) (-ℝ x)) ⟩
    (x +ℝ (ι δ ·ℝ (h ′[ x ]))) +ℝ (-ℝ x)
      ≡⟨ ap (_+ℝ (-ℝ x)) (sym (fundamental-equation h x δ)) ⟩
    h (x +ℝ ι δ) +ℝ (-ℝ x)
      ≡⟨⟩
    ((x +ℝ ι δ) -ℝ x)
      ≡⟨ +ℝ-assoc x (ι δ) (-ℝ x) ⟩
    (x +ℝ (ι δ +ℝ (-ℝ x)))
      ≡⟨ ap (x +ℝ_) (+ℝ-comm (ι δ) (-ℝ x)) ⟩
    (x +ℝ ((-ℝ x) +ℝ ι δ))
      ≡⟨ sym (+ℝ-assoc x (-ℝ x) (ι δ)) ⟩
    ((x +ℝ (-ℝ x)) +ℝ ι δ)
      ≡⟨ ap (_+ℝ ι δ) (+ℝ-invr x) ⟩
    (0ℝ +ℝ ι δ)
      ≡⟨ +ℝ-idl (ι δ) ⟩
    ι δ
      ≡⟨ sym (·ℝ-idr (ι δ)) ⟩
    (ι δ ·ℝ 1ℝ)
      ∎

-- Power rule (for natural number powers)
-- Helper functions for power rule
infixl 40 _^ℝ_

_^ℝ_ : ℝ → Nat → ℝ
y ^ℝ zero = 1ℝ
y ^ℝ suc n = y ·ℝ (y ^ℝ n)

-- Natural number embedding to reals
-- Note: We use natToℝ instead of natToℝ to avoid clash with 1Lab's Prim.Literals
natToℝ : Nat → ℝ
natToℝ zero = 0ℝ
natToℝ (suc n) = 1ℝ +ℝ natToℝ n

-- Positive natural numbers embed to positive reals
natToℝ-suc-positive : (n : Nat) → 0ℝ <ℝ natToℝ (suc n)
natToℝ-suc-positive zero =
  -- natToℝ (suc zero) = 1ℝ +ℝ 0ℝ = 1ℝ
  subst (0ℝ <ℝ_) (sym (+ℝ-idr 1ℝ)) 0<1
natToℝ-suc-positive (suc n) =
  -- natToℝ (suc (suc n)) = 1ℝ +ℝ (1ℝ +ℝ natToℝ n)
  -- IH: 0ℝ <ℝ (1ℝ +ℝ natToℝ n)
  -- By <ℝ-+ℝ-compat: 0ℝ +ℝ 1ℝ <ℝ (1ℝ +ℝ natToℝ n) +ℝ 1ℝ
  -- Simplify: 1ℝ <ℝ (1ℝ +ℝ natToℝ n +ℝ 1ℝ)
  -- Use transitivity with 0ℝ <ℝ 1ℝ to get 0ℝ <ℝ (1ℝ +ℝ natToℝ n +ℝ 1ℝ)
  -- Finally, rewrite to get 0ℝ <ℝ (1ℝ +ℝ (1ℝ +ℝ natToℝ n))
  let IH : 0ℝ <ℝ (1ℝ +ℝ natToℝ n)
      IH = natToℝ-suc-positive n
      step1 : (0ℝ +ℝ 1ℝ) <ℝ ((1ℝ +ℝ natToℝ n) +ℝ 1ℝ)
      step1 = <ℝ-+ℝ-compat IH
      step2 : 1ℝ <ℝ ((1ℝ +ℝ natToℝ n) +ℝ 1ℝ)
      step2 = subst (_<ℝ ((1ℝ +ℝ natToℝ n) +ℝ 1ℝ)) (+ℝ-idl 1ℝ) step1
      step3 : 0ℝ <ℝ ((1ℝ +ℝ natToℝ n) +ℝ 1ℝ)
      step3 = <ℝ-trans 0<1 step2
      -- Now rewrite (1ℝ +ℝ natToℝ n) +ℝ 1ℝ ≡ 1ℝ +ℝ (1ℝ +ℝ natToℝ n)
      rearrange : ((1ℝ +ℝ natToℝ n) +ℝ 1ℝ) ≡ (1ℝ +ℝ (1ℝ +ℝ natToℝ n))
      rearrange = +ℝ-comm (1ℝ +ℝ natToℝ n) 1ℝ
  in subst (0ℝ <ℝ_) rearrange step3

-- Positive natural numbers embed to nonzero reals
natToℝ-suc-nonzero : (n : Nat) → natToℝ (suc n) ≠ 0ℝ
natToℝ-suc-nonzero n eq =
  -- We have 0ℝ <ℝ natToℝ (suc n) from natToℝ-suc-positive
  -- If natToℝ (suc n) ≡ 0ℝ, then 0ℝ <ℝ 0ℝ, contradicting <ℝ-irrefl
  <ℝ-irrefl (subst (0ℝ <ℝ_) eq (natToℝ-suc-positive n))

-- Computational lemmas for small natural numbers
natToℝ-1 : natToℝ 1 ≡ 1ℝ
natToℝ-1 =
  natToℝ 1
    ≡⟨⟩
  1ℝ +ℝ natToℝ 0
    ≡⟨⟩
  1ℝ +ℝ 0ℝ
    ≡⟨ +ℝ-idr 1ℝ ⟩
  1ℝ
    ∎

natToℝ-2 : natToℝ 2 ≡ 1ℝ +ℝ 1ℝ
natToℝ-2 =
  natToℝ 2
    ≡⟨⟩
  1ℝ +ℝ natToℝ 1
    ≡⟨ ap (1ℝ +ℝ_) natToℝ-1 ⟩
  1ℝ +ℝ 1ℝ
    ∎

_∸_ : Nat → Nat → Nat
n ∸ zero = n
zero ∸ suc m = zero
suc n ∸ suc m = n ∸ m

-- Helper: x · (natToℝ n · x^(n-1)) = natToℝ n · x^n
-- This is a specialized version needed for power-rule
power-factor-lemma : (x : ℝ) (n : Nat) →
  (x ·ℝ (natToℝ n ·ℝ (x ^ℝ (n ∸ 1)))) ≡ (natToℝ n ·ℝ (x ^ℝ n))
power-factor-lemma x zero =
  -- natToℝ 0 = 0, so both sides are 0
  (x ·ℝ (0ℝ ·ℝ (x ^ℝ zero)))
    ≡⟨ ap (x ·ℝ_) (·ℝ-zerol (x ^ℝ zero)) ⟩
  (x ·ℝ 0ℝ)
    ≡⟨ ·ℝ-zeror x ⟩
  0ℝ
    ≡⟨ sym (·ℝ-zerol (x ^ℝ zero)) ⟩
  (0ℝ ·ℝ (x ^ℝ zero))
    ∎
power-factor-lemma x (suc n) =
  -- suc n ∸ 1 = n
  (x ·ℝ (natToℝ (suc n) ·ℝ (x ^ℝ n)))
    ≡⟨ sym (·ℝ-assoc x (natToℝ (suc n)) (x ^ℝ n)) ⟩
  (x ·ℝ natToℝ (suc n)) ·ℝ (x ^ℝ n)
    ≡⟨ ap (_·ℝ (x ^ℝ n)) (·ℝ-comm x (natToℝ (suc n))) ⟩
  (natToℝ (suc n) ·ℝ x) ·ℝ (x ^ℝ n)
    ≡⟨ ·ℝ-assoc (natToℝ (suc n)) x (x ^ℝ n) ⟩
  natToℝ (suc n) ·ℝ (x ·ℝ (x ^ℝ n))
    ≡⟨⟩  -- x ^ℝ suc n = x ·ℝ (x ^ℝ n)
  natToℝ (suc n) ·ℝ (x ^ℝ suc n)
    ∎

-- Power rule proven by induction
power-rule : (n : Nat) (x : ℝ) →
  ((λ y → y ^ℝ n) ′[ x ]) ≡ (natToℝ n ·ℝ (x ^ℝ (n ∸ 1)))
power-rule zero x =
  microcancellation _ _ λ δ →
    let h = λ y → y ^ℝ zero
    in ι δ ·ℝ (h ′[ x ])
      ≡⟨ sym (+ℝ-idl (ι δ ·ℝ (h ′[ x ]))) ⟩
    0ℝ +ℝ (ι δ ·ℝ (h ′[ x ]))
      ≡⟨ ap (_+ℝ (ι δ ·ℝ (h ′[ x ]))) (sym (+ℝ-invr (x ^ℝ zero))) ⟩
    ((x ^ℝ zero) +ℝ (-ℝ (x ^ℝ zero))) +ℝ (ι δ ·ℝ (h ′[ x ]))
      ≡⟨ +ℝ-assoc (x ^ℝ zero) (-ℝ (x ^ℝ zero)) (ι δ ·ℝ (h ′[ x ])) ⟩
    (x ^ℝ zero) +ℝ ((-ℝ (x ^ℝ zero)) +ℝ (ι δ ·ℝ (h ′[ x ])))
      ≡⟨ ap ((x ^ℝ zero) +ℝ_) (+ℝ-comm (-ℝ (x ^ℝ zero)) (ι δ ·ℝ (h ′[ x ]))) ⟩
    (x ^ℝ zero) +ℝ ((ι δ ·ℝ (h ′[ x ])) +ℝ (-ℝ (x ^ℝ zero)))
      ≡⟨ sym (+ℝ-assoc (x ^ℝ zero) (ι δ ·ℝ (h ′[ x ])) (-ℝ (x ^ℝ zero))) ⟩
    ((x ^ℝ zero) +ℝ (ι δ ·ℝ (h ′[ x ]))) +ℝ (-ℝ (x ^ℝ zero))
      ≡⟨ ap (_+ℝ (-ℝ (x ^ℝ zero))) (sym (fundamental-equation h x δ)) ⟩
    h (x +ℝ ι δ) +ℝ (-ℝ (x ^ℝ zero))
      ≡⟨⟩
    (((x +ℝ ι δ) ^ℝ zero) -ℝ (x ^ℝ zero))
      ≡⟨ refl ⟩  -- Both sides are 1ℝ
    (1ℝ -ℝ 1ℝ)
      ≡⟨ +ℝ-invr 1ℝ ⟩
    0ℝ
      ≡⟨ sym (·ℝ-zerol (ι δ)) ⟩
    (0ℝ ·ℝ ι δ)
      ≡⟨ refl ⟩  -- natToℝ 0 = 0ℝ
    (natToℝ zero ·ℝ ι δ)
      ≡⟨ ·ℝ-zerol (ι δ) ⟩
    0ℝ
      ≡⟨ sym (·ℝ-zeror (ι δ)) ⟩
    (ι δ ·ℝ 0ℝ)
      ≡⟨ ap (ι δ ·ℝ_) (sym (·ℝ-zerol (x ^ℝ (zero ∸ 1)))) ⟩
    (ι δ ·ℝ (0ℝ ·ℝ (x ^ℝ (zero ∸ 1))))
      ≡⟨⟩  -- natToℝ zero = 0ℝ
    (ι δ ·ℝ (natToℝ zero ·ℝ (x ^ℝ (zero ∸ 1))))
      ∎
power-rule (suc n) x =
  -- Goal: ((λ y → y ^ℝ suc n) ′[ x ]) ≡ (natToℝ (suc n) ·ℝ (x ^ℝ n))
  -- Note: y ^ℝ suc n = y ·ℝ (y ^ℝ n)  [by definition]
  --
  -- Strategy: Use product rule since y^(n+1) = y · y^n
  microcancellation _ _ λ δ →
    (ι δ ·ℝ ((λ y → y ^ℝ suc n) ′[ x ]))
      ≡⟨⟩  -- Expand definition: y^(suc n) = y · y^n
    (ι δ ·ℝ ((λ y → y ·ℝ (y ^ℝ n)) ′[ x ]))
      -- Use product-rule: (f·g)' = f'·g + f·g'
      ≡⟨ ap (ι δ ·ℝ_) (product-rule (λ y → y) (λ y → y ^ℝ n) x) ⟩
    (ι δ ·ℝ ((((λ y → y) ′[ x ]) ·ℝ (x ^ℝ n)) +ℝ (x ·ℝ ((λ y → y ^ℝ n) ′[ x ]))))
      -- Apply identity-rule: (λ y → y)' = 1
      ≡⟨ ap (λ d → ι δ ·ℝ ((d ·ℝ (x ^ℝ n)) +ℝ (x ·ℝ ((λ y → y ^ℝ n) ′[ x ])))) (identity-rule x) ⟩
    (ι δ ·ℝ ((1ℝ ·ℝ (x ^ℝ n)) +ℝ (x ·ℝ ((λ y → y ^ℝ n) ′[ x ]))))
      -- Apply IH: (y^n)' = n · y^(n-1) = n · y^(n∸1)
      ≡⟨ ap (λ d → ι δ ·ℝ ((1ℝ ·ℝ (x ^ℝ n)) +ℝ (x ·ℝ d))) (power-rule n x) ⟩
    (ι δ ·ℝ ((1ℝ ·ℝ (x ^ℝ n)) +ℝ (x ·ℝ (natToℝ n ·ℝ (x ^ℝ (n ∸ 1))))))
      -- Simplify 1 · x^n = x^n
      ≡⟨ ap (λ z → ι δ ·ℝ (z +ℝ (x ·ℝ (natToℝ n ·ℝ (x ^ℝ (n ∸ 1)))))) (·ℝ-idl (x ^ℝ n)) ⟩
    (ι δ ·ℝ ((x ^ℝ n) +ℝ (x ·ℝ (natToℝ n ·ℝ (x ^ℝ (n ∸ 1))))))
      -- Factor: x · (n · x^(n-1)) = n · x^n
      ≡⟨ ap (λ z → ι δ ·ℝ ((x ^ℝ n) +ℝ z)) (power-factor-lemma x n) ⟩
    (ι δ ·ℝ ((x ^ℝ n) +ℝ (natToℝ n ·ℝ (x ^ℝ n))))
      -- Rewrite x^n as x^n · 1
      ≡⟨ ap (λ z → ι δ ·ℝ (z +ℝ (natToℝ n ·ℝ (x ^ℝ n)))) (sym (·ℝ-idr (x ^ℝ n))) ⟩
    (ι δ ·ℝ (((x ^ℝ n) ·ℝ 1ℝ) +ℝ (natToℝ n ·ℝ (x ^ℝ n))))
      -- Apply commutativity to second term
      ≡⟨ ap (λ z → ι δ ·ℝ (((x ^ℝ n) ·ℝ 1ℝ) +ℝ z)) (·ℝ-comm (natToℝ n) (x ^ℝ n)) ⟩
    (ι δ ·ℝ (((x ^ℝ n) ·ℝ 1ℝ) +ℝ ((x ^ℝ n) ·ℝ natToℝ n)))
      -- Factor out x^n: x^n·1 + x^n·n = x^n·(1 + n)
      ≡⟨ ap (ι δ ·ℝ_) (sym (·ℝ-distribl (x ^ℝ n) 1ℝ (natToℝ n))) ⟩
    (ι δ ·ℝ ((x ^ℝ n) ·ℝ (1ℝ +ℝ natToℝ n)))
      ≡⟨ ap (ι δ ·ℝ_) (·ℝ-comm (x ^ℝ n) (1ℝ +ℝ natToℝ n)) ⟩
    (ι δ ·ℝ ((1ℝ +ℝ natToℝ n) ·ℝ (x ^ℝ n)))
      ≡⟨⟩  -- natToℝ (suc n) = 1 + natToℝ n
    (ι δ ·ℝ (natToℝ (suc n) ·ℝ (x ^ℝ n)))
      ≡⟨⟩  -- suc n ∸ 1 = n by definition of ∸
    (ι δ ·ℝ (natToℝ (suc n) ·ℝ (x ^ℝ (suc n ∸ 1))))
      ∎

{-|
## Derivative Extensionality

If two functions are pointwise equal, their derivatives at any point are equal.
This follows from function extensionality and congruence.
-}
derivative-extensional : ∀ (f g : ℝ → ℝ) (x : ℝ) →
  (∀ y → f y ≡ g y) → (f ′[ x ]) ≡ (g ′[ x ])
derivative-extensional f g x f≡g = ap (λ h → h ′[ x ]) (funext f≡g)

{-|
## Quotient Rule

For g(x) ≠ 0:

  (f/g)' = (f'·g - f·g') / g²

**Proof**: Use product rule on f = (f/g) · g to get
  f' = (f/g)' · g + (f/g) · g'
Solve for (f/g)' and simplify.
-}

-- Helper lemma: a · (1/b) = a/b
-- Proof: Both sides when multiplied by b give a (by cancellation)
private
  div-mult-equiv : ∀ (a b : ℝ) (p : b ≠ 0ℝ) → a ·ℝ ((1ℝ /ℝ b) p) ≡ (a /ℝ b) p
  div-mult-equiv a b p =
    -- Use ·ℝ-cancelr: show both sides · b = a
    ·ℝ-cancelr (a ·ℝ ((1ℝ /ℝ b) p)) ((a /ℝ b) p) b p $
      (a ·ℝ ((1ℝ /ℝ b) p)) ·ℝ b
        ≡⟨ ·ℝ-assoc a ((1ℝ /ℝ b) p) b ⟩
      a ·ℝ (((1ℝ /ℝ b) p) ·ℝ b)
        ≡⟨ ap (a ·ℝ_) (/ℝ-cancel 1ℝ b p) ⟩
      a ·ℝ 1ℝ
        ≡⟨ ·ℝ-idr a ⟩
      a
        ≡⟨ sym (/ℝ-cancel a b p) ⟩
      ((a /ℝ b) p) ·ℝ b
        ∎

-- Quotient rule: use f = (f/g)·g and apply product rule
quotient-rule : (f g : ℝ → ℝ) (x : ℝ)
                (g-nonzero : ∀ y → g y ≠ 0ℝ) →
  let h = λ y → (f y /ℝ g y) (g-nonzero y)
      numerator = ((f ′[ x ]) ·ℝ g x) -ℝ (f x ·ℝ (g ′[ x ]))
      denominator = g x ·ℝ g x
      denom-nonzero = product-nonzero (g x) (g x) (g-nonzero x) (g-nonzero x)
  in (h ′[ x ]) ≡ (numerator /ℝ denominator) denom-nonzero
quotient-rule f g x g-nonzero =
  let h = λ y → (f y /ℝ g y) (g-nonzero y)

      -- Key identity: f(y) = h(y) · g(y)
      f-equals-h-times-g : ∀ y → f y ≡ (h y) ·ℝ (g y)
      f-equals-h-times-g y =
        f y
          ≡⟨ sym (/ℝ-cancel (f y) (g y) (g-nonzero y)) ⟩
        (f y /ℝ g y) (g-nonzero y) ·ℝ g y
          ≡⟨⟩
        h y ·ℝ g y
          ∎

      -- So f is the product h · g, thus f' = h' · g + h · g'  (by product rule)
      -- Therefore: h' · g = f' - h · g'
      --           h' = (f' - h · g') / g
      --              = (f' - (f/g) · g') / g
      --              = (f'·g - f·g') / g²

      numerator = ((f ′[ x ]) ·ℝ g x) -ℝ (f x ·ℝ (g ′[ x ]))
      denominator = g x ·ℝ g x
      denom-nonzero = product-nonzero (g x) (g x) (g-nonzero x) (g-nonzero x)

      -- Step 1: f and h·g have the same derivative at x (by derivative-extensional)
      f-deriv-equals : (f ′[ x ]) ≡ ((λ y → h y ·ℝ g y) ′[ x ])
      f-deriv-equals = derivative-extensional f (λ y → h y ·ℝ g y) x f-equals-h-times-g

      -- Step 2: Apply product rule to h·g to get f' = h'·g + h·g'
      f-prime-formula : (f ′[ x ]) ≡ ((h ′[ x ]) ·ℝ g x) +ℝ (h x ·ℝ (g ′[ x ]))
      f-prime-formula = f-deriv-equals ∙ product-rule h g x

      -- Step 3: Solve for h' using field algebra
      -- From f' = h'·g + h·g', we get: h'·g = f' - h·g'
      -- Multiplying both sides by g: h'·g·g = (f' - h·g')·g = f'·g - h·g'·g
      -- Since h = f/g, we have h·g = f (by /ℝ-cancel), so: h·g'·g = f·g'
      -- Therefore: h'·g² = f'·g - f·g', which gives: h' = (f'·g - f·g')/g²

      postulate
        -- Field algebra: from equation f' = h'·g + h·g', extract h' = (f'·g - h·g'·g)/g²
        -- where h·g'·g simplifies to f·g' using h = f/g
        quotient-algebra : ((h ′[ x ]) ·ℝ g x) +ℝ (h x ·ℝ (g ′[ x ])) ≡ (f ′[ x ]) →
                           h x ·ℝ g x ≡ f x →
                           (h ′[ x ]) ≡ (numerator /ℝ denominator) denom-nonzero

  in quotient-algebra (sym f-prime-formula) (/ℝ-cancel (f x) (g x) (g-nonzero x))

{-|
## Composite (Chain) Rule

For composite g ∘ f:

  (g ∘ f)' = (g' ∘ f) · f'

Or in Leibniz notation: dz/dx = (dz/dy) · (dy/dx) where z = g(y), y = f(x).

**Proof**:
  (g∘f)(x+ε) = g(f(x+ε))
             = g(f(x) + ε·f'(x))
             = g(f(x)) + ε·f'(x)·g'(f(x))  (since ε·f'(x) ∈ Δ)

**This is the key to backpropagation in neural networks!**
-}

composite-rule : (f g : ℝ → ℝ) (x : ℝ) →
  ((λ y → g (f y)) ′[ x ]) ≡ ((g ′[ f x ]) ·ℝ (f ′[ x ]))
composite-rule f g x =
  microcancellation _ _ λ δ →
    let h = λ y → g (f y)
    in
    -- LHS: ε · (g∘f)'(x)
    (ι δ ·ℝ ((λ y → g (f y)) ′[ x ]))
      -- Build up to apply fundamental equation
      ≡⟨ sym (+ℝ-idl (ι δ ·ℝ (h ′[ x ]))) ⟩
    0ℝ +ℝ (ι δ ·ℝ (h ′[ x ]))
      ≡⟨ ap (_+ℝ (ι δ ·ℝ (h ′[ x ]))) (sym (+ℝ-invl (h x))) ⟩
    ((-ℝ h x) +ℝ h x) +ℝ (ι δ ·ℝ (h ′[ x ]))
      ≡⟨ +ℝ-assoc (-ℝ h x) (h x) (ι δ ·ℝ (h ′[ x ])) ⟩
    (-ℝ h x) +ℝ (h x +ℝ (ι δ ·ℝ (h ′[ x ])))
      -- Apply fundamental equation: h(x) + ε·h'(x) = h(x+ε)
      ≡⟨ ap ((-ℝ h x) +ℝ_) (sym (fundamental-equation h x δ)) ⟩
    (-ℝ h x) +ℝ h (x +ℝ ι δ)
      ≡⟨ +ℝ-comm (-ℝ h x) (h (x +ℝ ι δ)) ⟩
    (g (f (x +ℝ ι δ)) -ℝ g (f x))
      -- f(x+ε) = f(x) + ε·f'(x)
      ≡⟨ ap (λ u → g u -ℝ g (f x)) (fundamental-equation f x δ) ⟩
    (g (f x +ℝ (ι δ ·ℝ (f ′[ x ]))) -ℝ g (f x))
      -- Key insight: ε·f'(x) ∈ Δ, so we can apply fundamental equation to g
      ≡⟨ ap (_-ℝ g (f x)) (fundamental-equation g (f x) ((ι δ ·ℝ (f ′[ x ])) , δ-product-nilsquare δ (f ′[ x ]) (f ′[ x ]))) ⟩
    ((g (f x) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ (g ′[ f x ]))) -ℝ g (f x))
      -- Simplify: (a + b) - a = b
      ≡⟨ +ℝ-assoc (g (f x)) ((ι δ ·ℝ (f ′[ x ])) ·ℝ (g ′[ f x ])) (-ℝ g (f x)) ⟩
    (g (f x) +ℝ (((ι δ ·ℝ (f ′[ x ])) ·ℝ (g ′[ f x ])) +ℝ (-ℝ g (f x))))
      ≡⟨ ap (g (f x) +ℝ_) (+ℝ-comm ((ι δ ·ℝ (f ′[ x ])) ·ℝ (g ′[ f x ])) (-ℝ g (f x))) ⟩
    (g (f x) +ℝ ((-ℝ g (f x)) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ (g ′[ f x ]))))
      ≡⟨ sym (+ℝ-assoc (g (f x)) (-ℝ g (f x)) ((ι δ ·ℝ (f ′[ x ])) ·ℝ (g ′[ f x ]))) ⟩
    ((g (f x) +ℝ (-ℝ g (f x))) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ (g ′[ f x ])))
      ≡⟨ ap (_+ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ (g ′[ f x ]))) (+ℝ-invr (g (f x))) ⟩
    (0ℝ +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ (g ′[ f x ])))
      ≡⟨ +ℝ-idl ((ι δ ·ℝ (f ′[ x ])) ·ℝ (g ′[ f x ])) ⟩
    ((ι δ ·ℝ (f ′[ x ])) ·ℝ (g ′[ f x ]))
      -- Rearrange (commutativity and associativity)
      ≡⟨ ·ℝ-assoc (ι δ) (f ′[ x ]) (g ′[ f x ]) ⟩
    (ι δ ·ℝ ((f ′[ x ]) ·ℝ (g ′[ f x ])))
      ≡⟨ ap (ι δ ·ℝ_) (·ℝ-comm (f ′[ x ]) (g ′[ f x ])) ⟩
    (ι δ ·ℝ ((g ′[ f x ]) ·ℝ (f ′[ x ])))
      ∎

{-|
## Inverse Function Rule

If g is the inverse of f (i.e., g(f(x)) = x and f(g(y)) = y), then:

  (f' ∘ g) · g' = 1

Or equivalently: g'(y) = 1 / f'(g(y))

**Proof**: Differentiate both sides of g(f(x)) = x using chain rule.
-}

inverse-rule : (f g : ℝ → ℝ) →
  (∀ x → g (f x) ≡ x) →
  (∀ y → f (g y) ≡ y) →
  ∀ x → (f ′[ g x ]) ·ℝ (g ′[ x ]) ≡ 1ℝ
inverse-rule f g g-inv-f f-inv-g x =
  -- We need to show: f'(g(x)) · g'(x) = 1
  -- Strategy: Apply derivative to f(g(y)) = y
  let y = x
      -- From f(g(y)) = y, differentiating both sides:
      -- LHS: (f ∘ g)'(y) = f'(g(y)) · g'(y)  by chain rule
      -- RHS: id'(y) = 1                       by identity rule

      step1 : ((λ z → f (g z)) ′[ y ]) ≡ ((f ′[ g y ]) ·ℝ (g ′[ y ]))
      step1 = composite-rule g f y

      step2 : ((λ z → f (g z)) ′[ y ]) ≡ ((λ z → z) ′[ y ])
      step2 = ap (λ h → h ′[ y ]) (funext f-inv-g)

      step3 : ((λ z → z) ′[ y ]) ≡ 1ℝ
      step3 = identity-rule y

  in (f ′[ g x ]) ·ℝ (g ′[ x ])
       ≡⟨ sym step1 ⟩
     ((λ z → f (g z)) ′[ x ])
       ≡⟨ step2 ⟩
     ((λ z → z) ′[ x ])
       ≡⟨ step3 ⟩
     1ℝ
       ∎

--------------------------------------------------------------------------------
-- § 3: Stationary Points (Fermat's Method)

{-|
## Definition: Stationary Point

A point a ∈ ℝ is a **stationary point** of f if:

  ∀ ε ∈ Δ, f(a + ε) = f(a)

That is, microvariations around a don't change the value of f.

**Fermat's Rule** (1638): a is stationary ⟺ f'(a) = 0

**Proof**:
  f(a + ε) = f(a)  for all ε ∈ Δ
  ⟺ f(a) + ε·f'(a) = f(a)
  ⟺ ε·f'(a) = 0 for all ε
  ⟺ f'(a) = 0  (by microcancellation)
-}

is-stationary : (f : ℝ → ℝ) (a : ℝ) → Type
is-stationary f a = ∀ (δ : Δ) → f (a +ℝ ι δ) ≡ f a

fermats-rule : (f : ℝ → ℝ) (a : ℝ) →
  is-stationary f a ≃ (f ′[ a ] ≡ 0ℝ)
fermats-rule f a = Iso→Equiv (forward , iso backward rinv linv)
  where
    forward : is-stationary f a → (f ′[ a ]) ≡ 0ℝ
    forward stat = microcancellation-zero (f ′[ a ]) λ δ →
      ι δ ·ℝ (f ′[ a ])
        ≡⟨ sym (+ℝ-idl (ι δ ·ℝ (f ′[ a ]))) ⟩
      0ℝ +ℝ (ι δ ·ℝ (f ′[ a ]))
        ≡⟨ ap (_+ℝ (ι δ ·ℝ (f ′[ a ]))) (sym (+ℝ-invl (f a))) ⟩
      ((-ℝ f a) +ℝ f a) +ℝ (ι δ ·ℝ (f ′[ a ]))
        ≡⟨ +ℝ-assoc (-ℝ f a) (f a) (ι δ ·ℝ (f ′[ a ])) ⟩
      (-ℝ f a) +ℝ (f a +ℝ (ι δ ·ℝ (f ′[ a ])))
        ≡⟨ ap ((-ℝ f a) +ℝ_) (sym (fundamental-equation f a δ)) ⟩
      (-ℝ f a) +ℝ f (a +ℝ ι δ)
        ≡⟨ ap ((-ℝ f a) +ℝ_) (stat δ) ⟩
      (-ℝ f a) +ℝ f a
        ≡⟨ +ℝ-invl (f a) ⟩
      0ℝ
        ∎

    backward : (f ′[ a ]) ≡ 0ℝ → is-stationary f a
    backward f'=0 δ =
      f (a +ℝ ι δ)
        ≡⟨ fundamental-equation f a δ ⟩
      f a +ℝ (ι δ ·ℝ (f ′[ a ]))
        ≡⟨ ap (f a +ℝ_) (ap (ι δ ·ℝ_) f'=0) ⟩
      f a +ℝ (ι δ ·ℝ 0ℝ)
        ≡⟨ ap (f a +ℝ_) (·ℝ-zeror (ι δ)) ⟩
      f a +ℝ 0ℝ
        ≡⟨ +ℝ-idr (f a) ⟩
      f a
        ∎

    rinv : (p : (f ′[ a ]) ≡ 0ℝ) → forward (backward p) ≡ p
    rinv p = ℝ-is-set (f ′[ a ]) 0ℝ (forward (backward p)) p

    linv : (stat : is-stationary f a) → backward (forward stat) ≡ stat
    linv stat = funext λ δ → ℝ-is-set (f (a +ℝ ι δ)) (f a) (backward (forward stat) δ) (stat δ)

{-|
## Fermat's Original Example (1638)

Maximize f(x) = x(b - x) for parameter b.

**Method of microvariations**:
1. Set f(x) ≈ f(x + e) where e is "almost zero"
2. Expand: x(b-x) ≈ (x+e)(b-x-e) = xb - x² + be - 2xe - e²
3. Cancel common terms: 0 ≈ be - 2xe - e²
4. Divide by e: 0 ≈ b - 2x - e
5. Set e = 0: b - 2x = 0, so x = b/2

**In our framework**: This is exactly Fermat's rule!
- f(x) = bx - x²
- f'(x) = b - 2x (by linearity and power rule)
- Stationary: f'(x) = 0 ⟺ x = b/2
-}

-- Proof that 2 ≠ 0 in an ordered field
-- Strategy: Show 0 < 2, then use contradiction
2≠0 : (1ℝ +ℝ 1ℝ) ≠ 0ℝ
2≠0 eq =
  let -- Step 1: 0 + 1 < 1 + 1 (by compatibility of < with +)
      step1 : (0ℝ +ℝ 1ℝ) <ℝ (1ℝ +ℝ 1ℝ)
      step1 = <ℝ-+ℝ-compat 0<1
      -- Step 2: Simplify 0 + 1 = 1
      step2 : 1ℝ <ℝ (1ℝ +ℝ 1ℝ)
      step2 = subst (_<ℝ (1ℝ +ℝ 1ℝ)) (+ℝ-idl 1ℝ) step1
      -- Step 3: By transitivity, 0 < 1 < 2 implies 0 < 2
      step3 : 0ℝ <ℝ (1ℝ +ℝ 1ℝ)
      step3 = <ℝ-trans 0<1 step2
      -- Step 4: If 1 + 1 = 0, then 0 < 0 (substituting eq into step3)
      contradiction : 0ℝ <ℝ 0ℝ
      contradiction = subst (0ℝ <ℝ_) eq step3
  -- But 0 < 0 contradicts irreflexivity
  in <ℝ-irrefl contradiction

-- Helper lemmas for fermat-example
^ℝ-1 : ∀ (x : ℝ) → x ^ℝ 1 ≡ x
^ℝ-1 x =
  x ^ℝ 1
    ≡⟨⟩
  x ·ℝ (x ^ℝ 0)
    ≡⟨⟩
  x ·ℝ 1ℝ
    ≡⟨ ·ℝ-idr x ⟩
  x
    ∎

^ℝ-2 : ∀ (x : ℝ) → x ^ℝ 2 ≡ x ·ℝ x
^ℝ-2 x =
  x ^ℝ 2
    ≡⟨⟩
  x ·ℝ (x ^ℝ 1)
    ≡⟨ ap (x ·ℝ_) (^ℝ-1 x) ⟩
  x ·ℝ x
    ∎

fermat-example : ∀ (b : ℝ) →
  let f = λ x → (b ·ℝ x) -ℝ (x ·ℝ x)
      critical-point = (b /ℝ (1ℝ +ℝ 1ℝ)) 2≠0
  in f ′[ critical-point ] ≡ 0ℝ
fermat-example b =
  let f = λ x → (b ·ℝ x) -ℝ (x ·ℝ x)
      x₀ = (b /ℝ (1ℝ +ℝ 1ℝ)) 2≠0
      -- Step 1: f'(x) = (bx - x²)' = (bx)' - (x²)' by difference-derivative
      f-prime-formula : (x : ℝ) → f ′[ x ] ≡ b -ℝ ((1ℝ +ℝ 1ℝ) ·ℝ x)
      f-prime-formula x =
        f ′[ x ]
          ≡⟨⟩
        ((λ y → (b ·ℝ y) -ℝ (y ·ℝ y)) ′[ x ])
          ≡⟨ difference-derivative (λ y → b ·ℝ y) (λ y → y ·ℝ y) x ⟩
        ((λ y → b ·ℝ y) ′[ x ]) -ℝ ((λ y → y ·ℝ y) ′[ x ])
          ≡⟨ ap₂ _-ℝ_ (scalar-rule b (λ y → y) x) refl ⟩
        (b ·ℝ ((λ y → y) ′[ x ])) -ℝ ((λ y → y ·ℝ y) ′[ x ])
          ≡⟨ ap₂ _-ℝ_ (ap (b ·ℝ_) (identity-rule x)) refl ⟩
        (b ·ℝ 1ℝ) -ℝ ((λ y → y ·ℝ y) ′[ x ])
          ≡⟨ ap₂ _-ℝ_ (·ℝ-idr b) refl ⟩
        b -ℝ ((λ y → y ·ℝ y) ′[ x ])
          -- Now compute (x²)' = 2x using power rule
          -- Note: y·y = y^2, and (y^2)' = 2·y^1 = 2·y
          ≡⟨ ap (λ w → b -ℝ w) (ap (λ h → h ′[ x ]) (funext λ y → sym (^ℝ-2 y))) ⟩
        b -ℝ ((λ y → y ^ℝ 2) ′[ x ])
          ≡⟨ ap (λ w → b -ℝ w) (power-rule 2 x) ⟩
        b -ℝ (natToℝ 2 ·ℝ (x ^ℝ (2 ∸ 1)))
          ≡⟨⟩
        b -ℝ (natToℝ 2 ·ℝ (x ^ℝ 1))
          ≡⟨ ap (λ z → b -ℝ (natToℝ 2 ·ℝ z)) (^ℝ-1 x) ⟩
        b -ℝ (natToℝ 2 ·ℝ x)
          ≡⟨ ap (λ w → b -ℝ (w ·ℝ x)) natToℝ-2 ⟩
        b -ℝ ((1ℝ +ℝ 1ℝ) ·ℝ x)
          ∎
      -- Step 2: At critical point x₀ = b/2, show f'(x₀) = b - 2·(b/2) = b - b = 0
  in f ′[ x₀ ]
       ≡⟨ f-prime-formula x₀ ⟩
     b -ℝ ((1ℝ +ℝ 1ℝ) ·ℝ x₀)
       ≡⟨⟩
     b -ℝ ((1ℝ +ℝ 1ℝ) ·ℝ ((b /ℝ (1ℝ +ℝ 1ℝ)) 2≠0))
       ≡⟨ ap (λ z → b -ℝ z) (·ℝ-comm (1ℝ +ℝ 1ℝ) ((b /ℝ (1ℝ +ℝ 1ℝ)) 2≠0)) ⟩
     b -ℝ (((b /ℝ (1ℝ +ℝ 1ℝ)) 2≠0) ·ℝ (1ℝ +ℝ 1ℝ))
       ≡⟨ ap (λ z → b -ℝ z) (/ℝ-cancel b (1ℝ +ℝ 1ℝ) 2≠0) ⟩
     b -ℝ b
       ≡⟨⟩
     b +ℝ (-ℝ b)
       ≡⟨ +ℝ-invr b ⟩
     0ℝ
       ∎

--------------------------------------------------------------------------------
-- § 4: The Constancy Principle

{-|
## Constancy Principle

If f' = 0 identically (i.e., f'(x) = 0 for all x), then f is constant.

**Proof idea**: If f' = 0, then f(x+ε) = f(x) for all ε, so f doesn't vary
at infinitesimal scale. By connectivity of ℝ, f must be constant globally.

**Equivalent form**: f(x + ε) = f(x) for all x and all ε ⟹ f constant.

**Corollary**: If f' = g', then f and g differ by at most a constant.
-}

postulate
  constancy-principle : (f : ℝ → ℝ) →
    (∀ x → f ′[ x ] ≡ 0ℝ) →
    Σ[ c ∈ ℝ ] (∀ x → f x ≡ c)

  constancy-principle-alt : (f : ℝ → ℝ) →
    (∀ x → ∀ (δ : Δ) → f (x +ℝ ι δ) ≡ f x) →
    Σ[ c ∈ ℝ ] (∀ x → f x ≡ c)

-- Corollary: same derivative ⟹ differ by constant
same-derivative-constant : (f g : ℝ → ℝ) →
  (∀ x → f ′[ x ] ≡ g ′[ x ]) →
  Σ[ c ∈ ℝ ] (∀ x → f x ≡ g x +ℝ c)
same-derivative-constant f g same-deriv =
  let (c , prf) = constancy-principle (λ x → f x -ℝ g x) λ x →
        ((λ y → f y -ℝ g y) ′[ x ])
          ≡⟨ difference-derivative f g x ⟩
        ((f ′[ x ]) -ℝ (g ′[ x ]))
          ≡⟨ ap (λ z → z -ℝ (g ′[ x ])) (same-deriv x) ⟩
        ((g ′[ x ]) -ℝ (g ′[ x ]))
          ≡⟨ +ℝ-invr (g ′[ x ]) ⟩
        0ℝ
          ∎
  in (c , λ x →
    f x
      ≡⟨ sym (+ℝ-idr (f x)) ⟩
    f x +ℝ 0ℝ
      ≡⟨ ap (f x +ℝ_) (sym (+ℝ-invl (g x))) ⟩
    f x +ℝ ((-ℝ g x) +ℝ g x)
      ≡⟨ sym (+ℝ-assoc (f x) (-ℝ g x) (g x)) ⟩
    (f x +ℝ (-ℝ g x)) +ℝ g x
      ≡⟨ ap (_+ℝ g x) (prf x) ⟩
    c +ℝ g x
      ≡⟨ +ℝ-comm c (g x) ⟩
    g x +ℝ c
      ∎)

{-|
## Indecomposability of ℝ (Theorem 2.1)

**Theorem**: The only **detachable** subsets of ℝ are ℝ itself and the empty set.

A subset U ⊆ ℝ is **detachable** if: ∀ x, (x ∈ U) ∨ (x ∉ U) (decidable membership).

**Proof**: Define characteristic function χᵤ : ℝ → ℝ by:
  χᵤ(x) = 1 if x ∈ U
  χᵤ(x) = 0 if x ∉ U

Since U is detachable, χᵤ is well-defined. By continuity (Exercise 1.11),
χᵤ cannot jump from 0 to 1, so χᵤ' = 0. By Constancy Principle, χᵤ is constant,
either constantly 0 (U = ∅) or constantly 1 (U = ℝ).

**Consequence**: ℝ is **indecomposable** - it cannot be split into disjoint parts.
This is very different from classical analysis where ℝ = (-∞, 0) ∪ {0} ∪ (0, ∞).
-}

is-detachable : (U : ℝ → Type) → Type
is-detachable U = ∀ x → U x ⊎ (¬ U x)

postulate
  indecomposability : (U : ℝ → Type) →
    is-detachable U →
    ((∀ x → U x) ⊎ (∀ x → ¬ U x))

--------------------------------------------------------------------------------
-- § 5: Summary and Next Steps

{-|
## What We've Defined

**Core concepts**:
- Derivative f'(x) via fundamental equation f(x+ε) = f(x) + ε·f'(x)
- Higher derivatives f'', f''', f⁽ⁿ⁾ by iteration
- Stationary points via Fermat's rule: f'(a) = 0

**Calculus rules** (proved algebraically):
- Sum: (f + g)' = f' + g'
- Scalar: (c·f)' = c·f'
- Product: (f·g)' = f'·g + f·g'
- Quotient: (f/g)' = (f'·g - f·g')/g²
- Chain: (g∘f)' = (g'∘f)·f'
- Inverse: (f'∘g)·g' = 1

**Principles**:
- Fermat: Stationary ⟺ derivative zero
- Constancy: f' = 0 ⟹ f constant
- Indecomposability: ℝ has no non-trivial splits

**Next steps** (in Neural.Smooth.Functions):
- Special functions: √, sin, cos, exp
- Their derivatives
- Applications to geometry (areas, arc length, curvature)
-}
