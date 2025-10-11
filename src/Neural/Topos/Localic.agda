{-# OPTIONS --rewriting --guardedness --cubical --no-load-primitives #-}

{-|
# Appendix A: Localic Topos and Fuzzy Identities

This module implements the equivalence between localic toposes and Ω-sets
from Appendix A of Belfiore & Bennequin (2022).

## Paper Reference

> "According to Bell [Bel08], a localic topos, as the one of a DNN, is naturally
> equivalent to the category SetΩ of Ω-sets, i.e. sets equipped with fuzzy
> identities with values in Ω."

> "In our context of DNN, [fuzzy equality] can be understood as the progressive
> decision about the outputs on the trees of layers rooted in a given layer."

## Key Concepts

**Ω-sets (Fuzzy sets)**:
- Complete Heyting algebra Ω (frame, locale)
- Set X with fuzzy equality δ: X×X → Ω
- δ is symmetric and transitive (Equation 18)
- Generalizes characteristic function of diagonal

**DNN Interpretation**:
- Ω: Truth values (progressive decisions)
- δ(x,y): "How equal are outputs x and y?"
- δ(x,x): May be ≠ ⊤ (partial certainty)
- Morphisms: Fuzzy functions (Equations 19-22)

**Main Result** (Proposition A.2):
- SetΩ ≃ Sh(Ω, K) (equivalence of categories)
- Localic toposes are exactly Ω-set categories
- DNNs naturally live in this framework

## Key Equations

- **Equation 18**: δ(x,y) ∧ δ(y,z) ≤ δ(x,z) (transitivity)
- **Equations 19-21**: Morphism axioms
- **Equation 22**: ∨_{x'∈X'} f(x,x') = δ(x,x) (totality)
- **Equation 23**: Composition (f' ∘ f)(x,x") = ∨_{x'} f(x,x') ∧ f'(x',x")
- **Equation 24**: Id_{X,δ} = δ
- **Equation 25**: Internal equality δ_U(α,α') = (α ≍ α')
- **Equations 27-28**: Sheaf conditions

## Module Structure

This module has been refactored into focused submodules:
- **Neural.Topos.Localic.Base**: CompleteHeytingAlgebra, Ω-Set, Ω-Set-Morphism
- **Neural.Topos.Localic.Category**: SetΩ category with composition and identity
- **Neural.Topos.Localic.Internal**: Internal hom Ω-U with propositional resizing
- **Neural.Topos.Localic.Equivalence**: F-functor, G-functor, localic equivalence

## References

- [Bel08] Bell (2008): Toposes and Local Set Theories
- [Lin20] Lindberg (2020): PhD thesis on geometric morphisms
-}

module Neural.Topos.Localic where

-- Re-export all submodules
open import Neural.Topos.Localic.Base public
open import Neural.Topos.Localic.Category public
open import Neural.Topos.Localic.Internal public
open import Neural.Topos.Localic.Equivalence public
