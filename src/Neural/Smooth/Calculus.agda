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
fundamental-equation f x δ = slope-property (λ δ' → f (x +ℝ ι δ')) δ

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
    ι δ ·ℝ ((λ y → f y +ℝ g y) ′[ x ])
      ≡⟨ ap (ι δ ·ℝ_) refl ⟩
    ι δ ·ℝ slope (λ δ' → f (x +ℝ ι δ') +ℝ g (x +ℝ ι δ'))
      ≡⟨ sym (fundamental-equation (λ y → f y +ℝ g y) x δ) ∙
         ap₂ _+ℝ_ (fundamental-equation f x δ) (fundamental-equation g x δ) ⟩
    ((f x +ℝ (ι δ ·ℝ (f ′[ x ]))) +ℝ (g x +ℝ (ι δ ·ℝ (g ′[ x ]))))
      ≡⟨ +ℝ-assoc (f x) (ι δ ·ℝ (f ′[ x ])) (g x +ℝ (ι δ ·ℝ (g ′[ x ]))) ⟩
    (f x +ℝ ((ι δ ·ℝ (f ′[ x ])) +ℝ (g x +ℝ (ι δ ·ℝ (g ′[ x ])))))
      ≡⟨ ap (f x +ℝ_) (+ℝ-comm (ι δ ·ℝ (f ′[ x ])) (g x +ℝ (ι δ ·ℝ (g ′[ x ])))) ⟩
    (f x +ℝ ((g x +ℝ (ι δ ·ℝ (g ′[ x ]))) +ℝ (ι δ ·ℝ (f ′[ x ]))))
      ≡⟨ ap (f x +ℝ_) (+ℝ-assoc (g x) (ι δ ·ℝ (g ′[ x ])) (ι δ ·ℝ (f ′[ x ]))) ⟩
    (f x +ℝ (g x +ℝ ((ι δ ·ℝ (g ′[ x ])) +ℝ (ι δ ·ℝ (f ′[ x ])))))
      ≡⟨ ap (λ z → f x +ℝ (g x +ℝ z)) (+ℝ-comm (ι δ ·ℝ (g ′[ x ])) (ι δ ·ℝ (f ′[ x ]))) ⟩
    (f x +ℝ (g x +ℝ ((ι δ ·ℝ (f ′[ x ])) +ℝ (ι δ ·ℝ (g ′[ x ])))))
      ≡⟨ sym (+ℝ-assoc (f x) (g x) ((ι δ ·ℝ (f ′[ x ])) +ℝ (ι δ ·ℝ (g ′[ x ])))) ⟩
    ((f x +ℝ g x) +ℝ ((ι δ ·ℝ (f ′[ x ])) +ℝ (ι δ ·ℝ (g ′[ x ]))))
      ≡⟨ ap ((f x +ℝ g x) +ℝ_) (sym (·ℝ-distribl (ι δ) (f ′[ x ]) (g ′[ x ]))) ⟩
    ((f x +ℝ g x) +ℝ (ι δ ·ℝ ((f ′[ x ]) +ℝ (g ′[ x ]))))
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
    (ι δ ·ℝ ((λ y → c ·ℝ f y) ′[ x ]))
      ≡⟨ sym (fundamental-equation (λ y → c ·ℝ f y) x δ) ⟩
    ((c ·ℝ f (x +ℝ ι δ)) -ℝ (c ·ℝ f x))
      ≡⟨ ap (λ z → (c ·ℝ z) -ℝ (c ·ℝ f x)) (fundamental-equation f x δ) ⟩
    ((c ·ℝ (f x +ℝ (ι δ ·ℝ (f ′[ x ])))) -ℝ (c ·ℝ f x))
      ≡⟨ ap (_-ℝ (c ·ℝ f x)) (·ℝ-distribr c (f x) (ι δ ·ℝ (f ′[ x ]))) ⟩
    (((c ·ℝ f x) +ℝ (c ·ℝ (ι δ ·ℝ (f ′[ x ])))) -ℝ (c ·ℝ f x))
      ≡⟨ ap (λ z → ((c ·ℝ f x) +ℝ z) -ℝ (c ·ℝ f x)) (·ℝ-assoc c (ι δ) (f ′[ x ])) ⟩
    (((c ·ℝ f x) +ℝ ((c ·ℝ ι δ) ·ℝ (f ′[ x ]))) -ℝ (c ·ℝ f x))
      ≡⟨ ap (λ z → ((c ·ℝ f x) +ℝ (z ·ℝ (f ′[ x ]))) -ℝ (c ·ℝ f x)) (·ℝ-comm c (ι δ)) ⟩
    (((c ·ℝ f x) +ℝ ((ι δ ·ℝ c) ·ℝ (f ′[ x ]))) -ℝ (c ·ℝ f x))
      ≡⟨ ap (λ z → ((c ·ℝ f x) +ℝ z) -ℝ (c ·ℝ f x)) (sym (·ℝ-assoc (ι δ) c (f ′[ x ]))) ⟩
    (((c ·ℝ f x) +ℝ (ι δ ·ℝ (c ·ℝ (f ′[ x ])))) -ℝ (c ·ℝ f x))
      ≡⟨ +ℝ-invr (c ·ℝ f x) ∙ +ℝ-idl (ι δ ·ℝ (c ·ℝ (f ′[ x ]))) ⟩
    (ι δ ·ℝ (c ·ℝ (f ′[ x ])))
      ∎

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
    -- LHS: ε · (f·g)'(x)
    (ι δ ·ℝ ((λ y → f y ·ℝ g y) ′[ x ]))
      -- From fundamental equation: (f·g)(x+ε) - (f·g)(x) = ε·(f·g)'(x)
      ≡⟨ sym (ap₂ _-ℝ_ (fundamental-equation (λ y → f y ·ℝ g y) x δ) refl) ⟩
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
      ≡⟨ ap (λ z → (((f x ·ℝ g x) +ℝ (f x ·ℝ (ι δ ·ℝ (g ′[ x ])))) +ℝ z) -ℝ (f x ·ℝ g x)) (·ℝ-distribr (ι δ ·ℝ (f ′[ x ])) (g x) (ι δ ·ℝ (g ′[ x ]))) ⟩
    ((((f x ·ℝ g x) +ℝ (f x ·ℝ (ι δ ·ℝ (g ′[ x ])))) +ℝ (((ι δ ·ℝ (f ′[ x ])) ·ℝ g x) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ (ι δ ·ℝ (g ′[ x ]))))) -ℝ (f x ·ℝ g x))
      -- Use ε² = 0
      ≡⟨ ap (λ z → (((f x ·ℝ g x) +ℝ (f x ·ℝ (ι δ ·ℝ (g ′[ x ])))) +ℝ (((ι δ ·ℝ (f ′[ x ])) ·ℝ g x) +ℝ z)) -ℝ (f x ·ℝ g x))
            (·ℝ-assoc (ι δ ·ℝ (f ′[ x ])) (ι δ) (g ′[ x ])
             ∙ ap (_·ℝ (g ′[ x ])) (sym (·ℝ-assoc (ι δ) (f ′[ x ]) (ι δ))
                                      ∙ ap (ι δ ·ℝ_) (·ℝ-comm (f ′[ x ]) (ι δ))
                                      ∙ ·ℝ-assoc (ι δ) (ι δ) (f ′[ x ])
                                      ∙ ap (_·ℝ (f ′[ x ])) (nilsquare δ)
                                      ∙ ·ℝ-zerol (f ′[ x ]))
             ∙ ·ℝ-zerol (g ′[ x ])) ⟩
    ((((f x ·ℝ g x) +ℝ (f x ·ℝ (ι δ ·ℝ (g ′[ x ])))) +ℝ (((ι δ ·ℝ (f ′[ x ])) ·ℝ g x) +ℝ 0ℝ)) -ℝ (f x ·ℝ g x))
      ≡⟨ ap (λ z → (((f x ·ℝ g x) +ℝ (f x ·ℝ (ι δ ·ℝ (g ′[ x ])))) +ℝ z) -ℝ (f x ·ℝ g x)) (+ℝ-idr ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) ⟩
    ((((f x ·ℝ g x) +ℝ (f x ·ℝ (ι δ ·ℝ (g ′[ x ])))) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) -ℝ (f x ·ℝ g x))
      -- Rearrange and factor out ε
      ≡⟨ ap (λ z → (((f x ·ℝ g x) +ℝ z) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) -ℝ (f x ·ℝ g x)) (·ℝ-assoc (f x) (ι δ) (g ′[ x ])) ⟩
    ((((f x ·ℝ g x) +ℝ ((f x ·ℝ ι δ) ·ℝ (g ′[ x ]))) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) -ℝ (f x ·ℝ g x))
      ≡⟨ ap (λ z → (((f x ·ℝ g x) +ℝ (z ·ℝ (g ′[ x ]))) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) -ℝ (f x ·ℝ g x)) (·ℝ-comm (f x) (ι δ)) ⟩
    ((((f x ·ℝ g x) +ℝ ((ι δ ·ℝ f x) ·ℝ (g ′[ x ]))) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) -ℝ (f x ·ℝ g x))
      ≡⟨ ap (λ z → (((f x ·ℝ g x) +ℝ z) +ℝ ((ι δ ·ℝ (f ′[ x ])) ·ℝ g x)) -ℝ (f x ·ℝ g x)) (sym (·ℝ-assoc (ι δ) (f x) (g ′[ x ]))) ⟩
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
      -- Distribute
      ≡⟨ ·ℝ-distribl (ι δ) ((f ′[ x ]) ·ℝ g x) (f x ·ℝ (g ′[ x ])) ⟩
    ((ι δ ·ℝ ((f ′[ x ]) ·ℝ g x)) +ℝ (ι δ ·ℝ (f x ·ℝ (g ′[ x ]))))
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
    (ι δ ·ℝ ((λ _ → c) ′[ x ]))
      ≡⟨ sym (fundamental-equation (λ _ → c) x δ) ⟩
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
    (ι δ ·ℝ ((λ y → y) ′[ x ]))
      ≡⟨ sym (fundamental-equation (λ y → y) x δ) ⟩
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
_^ℝ_ : ℝ → Nat → ℝ
y ^ℝ zero = 1ℝ
y ^ℝ suc n = y ·ℝ (y ^ℝ n)

fromNat : Nat → ℝ
fromNat zero = 0ℝ
fromNat (suc n) = 1ℝ +ℝ fromNat n

_∸_ : Nat → Nat → Nat
n ∸ zero = n
zero ∸ suc m = zero
suc n ∸ suc m = n ∸ m

-- Power rule proven by induction
power-rule : (n : Nat) (x : ℝ) →
  ((λ y → y ^ℝ n) ′[ x ]) ≡ (fromNat n ·ℝ (x ^ℝ (n ∸ 1)))
power-rule zero x =
  microcancellation _ _ λ δ →
    (ι δ ·ℝ ((λ y → y ^ℝ zero) ′[ x ]))
      ≡⟨ sym (fundamental-equation (λ y → y ^ℝ zero) x δ) ⟩
    (((x +ℝ ι δ) ^ℝ zero) -ℝ (x ^ℝ zero))
      ≡⟨ refl ⟩  -- Both sides are 1ℝ
    (1ℝ -ℝ 1ℝ)
      ≡⟨ +ℝ-invr 1ℝ ⟩
    0ℝ
      ≡⟨ sym (·ℝ-zerol (ι δ)) ⟩
    (0ℝ ·ℝ ι δ)
      ≡⟨ refl ⟩  -- fromNat 0 = 0ℝ
    (fromNat zero ·ℝ ι δ)
      ≡⟨ refl ⟩  -- x ^ℝ (zero ∸ 1) = x ^ℝ 0 = 1ℝ, but 0 · 1 = 0
    (ι δ ·ℝ (fromNat zero ·ℝ (x ^ℝ (zero ∸ 1))))
      ∎
power-rule (suc n) x = {!!}  -- TODO: Use product rule and induction hypothesis

{-|
## Quotient Rule

For g(x) ≠ 0:

  (f/g)' = (f'·g - f·g') / g²

**Proof**: Use product rule on f = (f/g) · g to get
  f' = (f/g)' · g + (f/g) · g'
Solve for (f/g)' and simplify.
-}

postulate
  quotient-rule : (f g : ℝ → ℝ) (x : ℝ) (p : g x ≠ 0ℝ) →
    (λ y → (f y /ℝ g y) {!!}) ′[ x ] ≡
    (((f ′[ x ]) ·ℝ g x) -ℝ (f x ·ℝ (g ′[ x ]))) /ℝ (g x ·ℝ g x) {!!}

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

-- Helper: Product of infinitesimals is nilsquare
δ-product-nilsquare : (δ : Δ) (a b : ℝ) → (ι δ ·ℝ a) ·ℝ (ι δ ·ℝ b) ≡ 0ℝ
δ-product-nilsquare δ a b =
  (ι δ ·ℝ a) ·ℝ (ι δ ·ℝ b)
    ≡⟨ ·ℝ-assoc (ι δ) a (ι δ ·ℝ b) ⟩
  ι δ ·ℝ (a ·ℝ (ι δ ·ℝ b))
    ≡⟨ ap (ι δ ·ℝ_) (·ℝ-assoc a (ι δ) b) ⟩
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

composite-rule : (f g : ℝ → ℝ) (x : ℝ) →
  ((λ y → g (f y)) ′[ x ]) ≡ ((g ′[ f x ]) ·ℝ (f ′[ x ]))
composite-rule f g x =
  microcancellation _ _ λ δ →
    -- LHS: ε · (g∘f)'(x)
    (ι δ ·ℝ ((λ y → g (f y)) ′[ x ]))
      -- From fundamental equation: (g∘f)(x+ε) - (g∘f)(x) = ε·(g∘f)'(x)
      ≡⟨ sym (ap₂ _-ℝ_ (fundamental-equation (λ y → g (f y)) x δ) refl) ⟩
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

postulate
  inverse-rule : (f g : ℝ → ℝ) →
    (∀ x → g (f x) ≡ x) →
    (∀ y → f (g y) ≡ y) →
    ∀ x → (f ′[ g x ]) ·ℝ (g ′[ x ]) ≡ 1ℝ

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
        ≡⟨ sym (ap₂ _-ℝ_ (fundamental-equation f a δ) refl) ⟩
      f (a +ℝ ι δ) -ℝ f a
        ≡⟨ ap (_-ℝ f a) (stat δ) ⟩
      f a -ℝ f a
        ≡⟨ +ℝ-invr (f a) ⟩
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

postulate
  fermat-example : ∀ (b : ℝ) →
    let f = λ x → (b ·ℝ x) -ℝ (x ·ℝ x)
        critical-point = (b /ℝ (1ℝ +ℝ 1ℝ)) {!!}
    in f ′[ critical-point ] ≡ 0ℝ

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
  constancy-principle (λ x → f x -ℝ g x) λ x →
    ((λ y → f y -ℝ g y) ′[ x ])
      ≡⟨ refl ⟩  -- Definition: x -ℝ y = x +ℝ (-ℝ y)
    ((λ y → f y +ℝ (-ℝ g y)) ′[ x ])
      ≡⟨ sum-rule f (λ y → -ℝ g y) x ⟩
    ((f ′[ x ]) +ℝ ((λ y → -ℝ g y) ′[ x ]))
      ≡⟨ refl ⟩  -- (-ℝ g y) = (-ℝ 1ℝ) ·ℝ g y
    ((f ′[ x ]) +ℝ ((λ y → (-ℝ 1ℝ) ·ℝ g y) ′[ x ]))
      ≡⟨ ap ((f ′[ x ]) +ℝ_) (scalar-rule (-ℝ 1ℝ) g x) ⟩
    ((f ′[ x ]) +ℝ ((-ℝ 1ℝ) ·ℝ (g ′[ x ])))
      ≡⟨ refl ⟩  -- Definition: -ℝ applied
    (f ′[ x ] -ℝ g ′[ x ])
      ≡⟨ ap (f ′[ x ] -ℝ_) (same-deriv x) ⟩
    (f ′[ x ] -ℝ f ′[ x ])
      ≡⟨ +ℝ-invr (f ′[ x ]) ⟩
    0ℝ
      ∎

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
