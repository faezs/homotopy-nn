{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Categorical Hopfield Dynamics (Section 6)

This module implements the categorical Hopfield dynamics framework from
Section 6 of Manin & Marcolli (2024), including threshold non-linearity,
discrete dynamics, and recovery of classical Hopfield networks.

## Organization

This module re-exports the following submodules:

### §6.2: Neural.Dynamics.Hopfield.Threshold
- Categorical threshold non-linearity (Proposition 6.1)
- Threshold functor (·)₊ : Core(C) → Core(C)
- Extension to summing functors

### §6.3: Neural.Dynamics.Hopfield.Discrete
- Discrete Hopfield dynamics (Definition 6.2, Equation 6.5)
- Conservation preservation (Lemma 6.3)
- Nerve dynamics (Lemma 6.4)

### §6.4: Neural.Dynamics.Hopfield.Classical
- Total weight functor (Lemma 6.6)
- Linear inhibitory/excitatory functors (Definition 6.7)
- Classical Hopfield recovery (Lemma 6.9)

## Key Concepts

**Categorical Hopfield dynamics** operates on network summing functors
Φ : P(E) → C in the conservation equalizer Σᵉᑫ_C(G).

**Dynamics equation** (Equation 6.5):
```
Xₑ(n+1) = Xₑ(n) ⊕ (⊕ₑ'∈E Tₑₑ'(Xₑ'(n)) ⊕ Θₑ)₊
```

where:
- Xₑ(n) = Φₙ(e) is the resource at edge e at time n
- Tₑₑ' is interaction endofunctor from edge e' to edge e
- Θₑ = Ψ(e) is constant external input
- (·)₊ is threshold functor from §6.2

**Classical recovery**: When C = WCodes⁺ with total-weight functor α,
this recovers the classical discrete Hopfield equation:

```
αₙ₊₁(e) = αₙ(e) + (Σₑ' tₑₑ'·αₙ(e') + θₑ)₊
```

## References

Manin, Y. I., & Marcolli, M. (2024). Homotopy-theoretic and categorical
models of neural information networks. *arXiv preprint* arXiv:2406.01228.
-}

module Neural.Dynamics.Hopfield where

-- §6.2: Categorical threshold non-linearity
open import Neural.Dynamics.Hopfield.Threshold public
  using ( PreorderedMonoid
        ; ThresholdStructure
        ; threshold-functor
        ; threshold-on-objects
        ; threshold-on-zero
        )

-- §6.3: Discrete Hopfield dynamics
open import Neural.Dynamics.Hopfield.Discrete public
  using ( PowerSetCat
        ; DoubleSummingFunctor
        ; DoubleSummingWithSums
        ; DoubleSummingEqualizer
        ; ConservationEqualizer
        ; HopfieldDynamics
        ; dynamics-preserves-conservation
        ; nerve-dynamics
        ; classifying-space-dynamics
        )

-- §6.4: Classical Hopfield recovery
open import Neural.Dynamics.Hopfield.Classical public
  using ( total-weight
        ; total-weight-functor
        ; LinearInhibitoryFunctor
        ; LinearExcitatoryFunctor
        ; classical-hopfield-dynamics
        )
