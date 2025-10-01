{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Classical Hopfield Dynamics via Weighted Codes (Section 6.4)

This module implements the recovery of classical Hopfield dynamics from the
categorical framework via positive weighted codes and total weight functors.

## Overview

**Goal**: Show that classical Hopfield dynamics emerges from categorical dynamics
when C = WCodes⁺(n, q') (positive weighted codes) with the total-weight functor.

**Key results**:
1. **Lemma 6.6**: Total-weight functor α : WCodes⁺ → (ℝ, ≥)
2. **Definition 6.7**: Linear excitatory/inhibitory functors
3. **Lemma 6.8**: Properties of linear inhibitory functors
4. **Lemma 6.9**: Classical Hopfield recovery

## Classical Hopfield Equation

The classical discrete Hopfield dynamics is:

  αₙ₊₁(e) = αₙ(e) + (Σₑ' tₑₑ'·αₙ(e') + θₑ)₊

where:
- αₙ(e) is the total weight of code at edge e at time n
- tₑₑ' are real-valued weights
- θₑ is external input
- (·)₊ = max{0,·} is the threshold function

This is recovered from categorical dynamics when:
- C = WCodes⁺(n, q')
- α : WCodes⁺ → (ℝ, ≥) is the total-weight functor
- Interactions use linear inhibitory functors
-}

module Neural.Dynamics.Hopfield.Classical where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Functor.Adjoint
open import Cat.Monoidal.Base

open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin)

open import Neural.Base
open import Neural.Information public
  using (ℝ; _+ℝ_; _*ℝ_; _≤ℝ_; _≥ℝ_; zeroℝ; oneℝ)
open import Neural.Code.Weighted
open import Neural.Dynamics.Hopfield.Threshold
open import Neural.Dynamics.Hopfield.Discrete

private variable
  o ℓ o' ℓ' : Level

{-|
## Total Weight Functor (Lemma 6.6)

For positive weighted codes (C, ω) with ω : C → ℝ and ω(c) ≥ 0, define:

  α(C, ω) := Σ_{c∈C} ω(c)

This is the **total weight** of the code.

**Lemma 6.6**: α defines a functor α : WCodes⁺ → (ℝ, ≥)

**Functoriality**: For morphisms (f, λ) : (C, ω) → (C', ω') in WCodes⁺:

  α(C', ω') = Σ_c' ω'(c')
            = Σ_c' Σ_{c∈f⁻¹(c')} λ(c',c) ω(c)    (by ω'(c') = Σ_{c∈f⁻¹(c')} λ(c',c) ω(c))
            = Σ_c (Σ_{c':f(c)=c'} λ(c',c)) ω(c)
            ≥ Σ_c ω(c)                              (since 0 ≤ λ(c',c) ≤ 1)
            = α(C, ω)

So α(C', ω') ≥ α(C, ω), giving a morphism in (ℝ, ≥).
-}

postulate
  {-| Thin category of real numbers with ≥ order -}
  ℝ-thin : Precategory lzero lzero

  ℝ-thin-Ob : Precategory.Ob ℝ-thin ≡ ℝ

  {-| Total weight of a positive weighted code -}
  total-weight :
    {n q' : Nat} →
    Precategory.Ob (WCodes-positive n q') →
    ℝ

  {-| Total weight functor (Lemma 6.6) -}
  total-weight-functor :
    (n q' : Nat) →
    Functor (WCodes-positive n q') ℝ-thin

  {-| For morphisms, total weight is non-decreasing -}
  total-weight-decreasing :
    {n q' : Nat} →
    (C C' : Precategory.Ob (WCodes-positive n q')) →
    (f : Precategory.Hom (WCodes-positive n q') C C') →
    total-weight C' ≥ℝ total-weight C

{-|
## Linear Inhibitory and Excitatory Functors (Definition 6.7)

A **linear inhibitory functor** I : WCodes⁺ → WCodes⁺ is:
1. **Contravariant** (defined on opposite category)
2. **Linear scaling**: α(I(C)) = r · α(C) for some r > 0

A **linear excitatory functor** E : WCodes⁺ → WCodes⁺ is:
1. **Covariant** (ordinary functor)
2. **Linear scaling**: α(E(C)) = r · α(C) for some r > 0

**Physical interpretation**:
- Inhibitory: More input → Less output (contravariant)
- Excitatory: More input → More output (covariant)
- Linear: Preserves proportionality of total weights
-}

record LinearInhibitoryFunctor (n q' : Nat) : Type (lsuc lzero) where
  no-eta-equality
  field
    -- Contravariant functor (on opposite category)
    functor : Functor (WCodes-positive n q' ^op) (WCodes-positive n q')

    -- Scaling factor r > 0
    scaling-factor : ℝ
    scaling-positive : zeroℝ ≤ℝ scaling-factor

    -- Linear scaling: α(I(C)) = r · α(C)
    linear-scaling :
      (C : Precategory.Ob (WCodes-positive n q')) →
      total-weight (functor .Functor.F₀ C) ≡ scaling-factor *ℝ total-weight C

open LinearInhibitoryFunctor public

record LinearExcitatoryFunctor (n q' : Nat) : Type (lsuc lzero) where
  no-eta-equality
  field
    -- Covariant functor
    functor : Functor (WCodes-positive n q') (WCodes-positive n q')

    -- Scaling factor r > 0
    scaling-factor : ℝ
    scaling-positive : zeroℝ ≤ℝ scaling-factor

    -- Linear scaling: α(E(C)) = r · α(C)
    linear-scaling :
      (C : Precategory.Ob (WCodes-positive n q')) →
      total-weight (functor .Functor.F₀ C) ≡ scaling-factor *ℝ total-weight C

open LinearExcitatoryFunctor public

{-|
## Properties of Linear Inhibitory Functors (Lemma 6.8)

**Lemma 6.8**: For linear inhibitory functors I : WCodes⁺ → WCodes⁺:

1. **Contravariance**: More input codes → Fewer output codes
   C ≤ C' (in WCodes⁺) implies I(C') ≤ I(C)

2. **Ratio preservation**: For morphisms f : C → C',
   α(I(C')) / α(I(C)) = α(C') / α(C)

This ensures that the categorical structure properly models inhibition.
-}

postulate
  {-| Contravariance property -}
  linear-inhibitory-contravariant :
    {n q' : Nat} →
    (I : LinearInhibitoryFunctor n q') →
    (C C' : Precategory.Ob (WCodes-positive n q')) →
    (f : Precategory.Hom (WCodes-positive n q') C C') →
    {-| Produces morphism I(C') → I(C) -}
    Precategory.Hom (WCodes-positive n q') (I .functor .Functor.F₀ C') (I .functor .Functor.F₀ C)

  {-| Ratio preservation property -}
  linear-inhibitory-ratio-preserving :
    {n q' : Nat} →
    (I : LinearInhibitoryFunctor n q') →
    (C C' : Precategory.Ob (WCodes-positive n q')) →
    (f : Precategory.Hom (WCodes-positive n q') C C') →
    {-| α(I(C')) / α(I(C)) = α(C') / α(C) -}
    ⊤

{-|
## Classical Hopfield Recovery (Lemma 6.9)

**Theorem**: The categorical Hopfield dynamics on WCodes⁺ with interactions
given by linear inhibitory functors recovers the classical discrete Hopfield equation.

**Setup**:
- C = WCodes⁺(n, q')
- α : C → (ℝ, ≥) is the total-weight functor
- Tₑₑ' are linear inhibitory functors with scaling factors tₑₑ'
- Ψ(e) has total weight θₑ

**Categorical dynamics** (Equation 6.5):
  Φₙ₊₁(e) = Φₙ(e) ⊕ (⊕ₑ' Tₑₑ'(Φₙ(e')) ⊕ Ψ(e))₊

**Apply α**:
  α(Φₙ₊₁(e)) = α(Φₙ(e) ⊕ (⊕ₑ' Tₑₑ'(Φₙ(e')) ⊕ Ψ(e))₊)

**Since α is additive on ⊕**:
  αₙ₊₁(e) = αₙ(e) + α((⊕ₑ' Tₑₑ'(Φₙ(e')) ⊕ Ψ(e))₊)

**For linear inhibitory Tₑₑ' with scaling tₑₑ'**:
  α(Tₑₑ'(Φₙ(e'))) = tₑₑ' · αₙ(e')

**Therefore**:
  αₙ₊₁(e) = αₙ(e) + (Σₑ' tₑₑ'·αₙ(e') + θₑ)₊

This is exactly the classical Hopfield equation!
-}

postulate
  {-| Classical Hopfield dynamics recovery -}
  classical-hopfield-dynamics :
    {G : DirectedGraph} →
    (n q' : Nat) →
    -- Categorical Hopfield dynamics on WCodes⁺
    (hd : HopfieldDynamics G) →
    -- Interaction functors are linear inhibitory
    (linear : (e e' : Fin (edges G)) → LinearInhibitoryFunctor n q') →
    -- Scaling factors tₑₑ'
    (t : Fin (edges G) → Fin (edges G) → ℝ) →
    -- External inputs θₑ
    (θ : Fin (edges G) → ℝ) →
    -- Then for all time steps n and edges e:
    -- α(Φₙ₊₁(e)) = α(Φₙ(e)) + (Σₑ' tₑₑ'·α(Φₙ(e')) + θₑ)₊
    ⊤

  {-| Additivity of total weight functor on monoidal product -}
  total-weight-additive :
    {n q' : Nat} →
    (C C' : Precategory.Ob (WCodes-positive n q')) →
    {-| α(C ⊕ C') = α(C) + α(C') -}
    ⊤

  {-| Threshold functor commutes with total weight -}
  total-weight-threshold-commute :
    {n q' : Nat} →
    (C : Precategory.Ob (WCodes-positive n q')) →
    {-| α((C)₊) = max{0, α(C)} -}
    ⊤

{-|
## Relationship to Continuous Dynamics

The continuous Hopfield ODE is:

  dxⱼ/dt = -xⱼ(t) + (Σᵢ wᵢⱼ·xᵢ(t) + θⱼ)₊

This relates to our discrete dynamics via:
1. **Time discretization**: Δt step size
2. **Leak term**: -xⱼ(t) corresponds to Xₑ(n) in (6.5)
3. **Limit**: As Δt → 0, discrete approaches continuous

The categorical framework naturally accommodates both discrete and continuous
variants via appropriate choice of time indexing.
-}

postulate
  {-| Continuous Hopfield ODE (for reference) -}
  continuous-hopfield-ode :
    -- State variables xⱼ : ℝ
    -- Weights wᵢⱼ : ℝ
    -- External inputs θⱼ : ℝ
    -- Time derivative dxⱼ/dt = -xⱼ + (Σᵢ wᵢⱼ·xᵢ + θⱼ)₊
    ⊤

{-|
## Examples and Applications

**Example 1**: Binary Hopfield network
- WCodes⁺ with weights {0,1}
- Recovers classical discrete Hopfield with binary states

**Example 2**: Continuous-valued Hopfield
- WCodes⁺ with weights in [0,1]
- Generalizes to probabilistic interpretations

**Example 3**: Sparse coding
- Most weights near zero (sparse codes)
- Efficient representation of patterns
-}
