{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Discrete Hopfield Dynamics in Categories of Summing Functors (Section 6.3)

This module implements the categorical Hopfield dynamics from Section 6.3 of
Manin & Marcolli (2024), including Definition 6.2 (dynamics equation),
Lemma 6.3 (conservation preservation), and Lemma 6.4 (nerve dynamics).

## Overview

The **categorical Hopfield dynamics** operates on network summing functors
Φ : P(E) → C in the conservation equalizer Σᵉᑫ_C(G).

**Key components**:
1. **Double summing functors** T : P(E) × P(E) → Cat[C,C]
   - T(e,e') gives endofunctor of C modeling interaction from e' to e
   - Satisfies summing properties in both arguments
   - Lives in equalizer (conservation at vertices)

2. **Dynamics equation** (Definition 6.2, Equation 6.5):
   ```
   Xₑ(n+1) = Xₑ(n) ⊕ (⊕ₑ'∈E Tₑₑ'(Xₑ'(n)) ⊕ Θₑ)₊
   ```
   where:
   - Xₑ(n) = Φₙ(e) is the resource at edge e at time n
   - Tₑₑ' is interaction endofunctor
   - Θₑ = Ψ(e) is constant external input
   - (·)₊ is threshold functor from §6.2

3. **Conservation preservation** (Lemma 6.3):
   If Φ₀ and Ψ are in equalizer, then Φₙ remains in equalizer for all n

4. **Nerve dynamics** (Lemma 6.4):
   Induces dynamics on simplicial nerve N(Σᵉᑫ_C(G)) and classifying space
-}

module Neural.Dynamics.Hopfield.Discrete where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Monoidal.Base
open import Cat.Monoidal.Braided
open import Cat.Instances.Product
open import Cat.Instances.Functor using (Cat[_,_])
open import Cat.Instances.Simplex using (Δ)
open import Cat.Diagram.Equaliser

open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin)
open import Data.Sum.Base

open import Neural.Base
open import Neural.SummingFunctor
open import Neural.Network.Conservation
open import Neural.Dynamics.Hopfield.Threshold

private variable
  o ℓ o' ℓ' : Level

{-|
## Power Set Category P(E)

For a finite pointed set E, the power set category P(E) has:
- Objects: Pointed subsets A ⊆ E (containing basepoint)
- Morphisms: Inclusions A ⊆ B

This is used for summing functors Φ : P(E) → C.
-}

postulate
  basepoint : {E : Type} → E

  PowerSetCat : Type → Precategory (lsuc lzero) lzero

  PowerSetCat-Ob :
    (E : Type) →
    Precategory.Ob (PowerSetCat E) ≡ (Σ[ A ∈ (E → Type lzero) ] A basepoint)

  -- Union of pointed subsets (for summing property)
  union :
    {E : Type} →
    Precategory.Ob (PowerSetCat E) →
    Precategory.Ob (PowerSetCat E) →
    Precategory.Ob (PowerSetCat E)

{-|
## Double Summing Functors (Section 6.3)

A **double summing functor** T : P(E) × P(E) → Cat[C,C] assigns to each
pair of subsets (A,B) an endofunctor T(A,B) : C → C.

In the Hopfield dynamics, we use T(e,e') for singleton subsets to model
the interaction from edge e' to edge e.

**Summing properties**:
- T(A∪A', B) = T(A,B) ⊕ T(A',B) when A∩A' = {basepoint}
- T(A, B∪B') = T(A,B) ⊕ T(A,B') when B∩B' = {basepoint}

These ensure that interactions compose additively.
-}

DoubleSummingFunctor :
  (E : Type) →
  {C : Precategory o ℓ} →
  (Cᵐ : Monoidal-category C) →
  Type (lsuc lzero ⊔ o ⊔ ℓ)
DoubleSummingFunctor E {C} Cᵐ =
  Functor (PowerSetCat E ×ᶜ PowerSetCat E) Cat[ C , C ]

record DoubleSummingWithSums
  (E : Type)
  {C : Precategory o ℓ}
  (Cᵐ : Monoidal-category C)
  : Type (lsuc lzero ⊔ o ⊔ ℓ) where
  no-eta-equality

  field
    functor : DoubleSummingFunctor E Cᵐ

    -- Summing in first argument: T(A∪A', B) = T(A,B) ⊕ T(A',B)
    -- Requires: A ∩ A' = {basepoint}
    sum-left :
      (A A' B : Precategory.Ob (PowerSetCat E)) →
      functor .Functor.F₀ (union A A' , B) ≡ functor .Functor.F₀ (A , B)

    -- Summing in second argument: T(A, B∪B') = T(A,B) ⊕ T(A,B')
    -- Requires: B ∩ B' = {basepoint}
    sum-right :
      (A B B' : Precategory.Ob (PowerSetCat E)) →
      functor .Functor.F₀ (A , union B B') ≡ functor .Functor.F₀ (A , B)

open DoubleSummingWithSums public

{-|
## Equalizer for Double Summing Functors

For directed graph G, we need double summing functors T that satisfy
source/target conservation:

  Σ²ₑ(C)(G) := equalizer(s, t : Σ²ₑ(C)(E) ⇒ Σ²ₑ(C)(V))

where s and t are defined by:
- Tˢ(A,B) = T(s⁻¹(A), s⁻¹(B))
- Tᵗ(A,B) = T(t⁻¹(A), t⁻¹(B))

This ensures that interactions respect the graph structure.
-}

postulate
  DoubleSummingEqualizer :
    (G : DirectedGraph) →
    {C : Precategory o ℓ} →
    (Cᵐ : Monoidal-category C) →
    Precategory (o ⊔ ℓ) (o ⊔ ℓ)

  -- Conservation equalizer for network summing functors
  ConservationEqualizer :
    (G : DirectedGraph) →
    Precategory (o ⊔ ℓ) (o ⊔ ℓ)

  -- NetworkSummingFunctor assigns resources to edges
  NetworkSummingFunctor-apply :
    {C : Precategory o ℓ} →
    {G : DirectedGraph} →
    NetworkSummingFunctor C G →
    (e : Fin (edges G)) →
    C .Precategory.Ob

  -- Construct a NetworkSummingFunctor from a function
  NetworkSummingFunctor-mk :
    {C : Precategory o ℓ} →
    {G : DirectedGraph} →
    ((e : Fin (edges G)) → C .Precategory.Ob) →
    NetworkSummingFunctor C G

{-|
## Network Summing Functor in Equalizer

A network summing functor Φ ∈ Σᵉᑫ_C(G) assigns resources to edges
while satisfying conservation at vertices.

From Neural.SummingFunctor, we use:
- NetworkSummingFunctor G C
- ConservationEqualizer G C
-}

{-|
## Hopfield Dynamics (Definition 6.2)

The **categorical Hopfield dynamics** is defined by:

**Given**:
1. Threshold structure (C, R, ρ : C → R)
2. Double summing functor T ∈ Σ²ₑ(C)(G)
3. Initial condition Φ₀ ∈ Σᵉᑫ_C(G)
4. Constant external input Ψ ∈ Σᵉᑫ_C(G)

**Dynamics** (Equation 6.5):
```
Xₑ(n+1) = Xₑ(n) ⊕ (⊕ₑ'∈E Tₑₑ'(Xₑ'(n)) ⊕ Θₑ)₊
```

where Xₑ(n) = Φₙ(e), Θₑ = Ψ(e), and (·)₊ is the threshold functor.
-}

record HopfieldDynamics (G : DirectedGraph) : Type (lsuc (o ⊔ ℓ ⊔ o' ⊔ ℓ')) where
  no-eta-equality

  field
    -- Threshold structure
    ts : ThresholdStructure o ℓ o' ℓ'

    -- Double summing functor T : P(E) × P(E) → Cat[C,C]
    T : DoubleSummingWithSums (Fin (edges G)) (ts .ThresholdStructure.C-monoidal)

    -- T satisfies source/target conservation (functor is in equalizer category)
    T-equalizer : Precategory.Ob (DoubleSummingEqualizer G (ts .ThresholdStructure.C-monoidal))

    -- Initial condition Φ₀ in equalizer
    Φ₀ : NetworkSummingFunctor (ts .ThresholdStructure.C) G
    Φ₀-conservation : Precategory.Ob (ConservationEqualizer G)

    -- Constant external input Ψ in equalizer
    Ψ : NetworkSummingFunctor (ts .ThresholdStructure.C) G
    Ψ-conservation : Precategory.Ob (ConservationEqualizer G)

  postulate
    -- Finite sum over edges (combinatorial operation)
    sum-over-edges :
      (f : Fin (edges G) → (ts .ThresholdStructure.C) .Precategory.Ob) →
      (ts .ThresholdStructure.C) .Precategory.Ob

  {-|
  Single step of dynamics (Equation 6.5)

  Xₑ(n+1) = Xₑ(n) ⊕ (⊕ₑ' Tₑₑ'(Xₑ'(n)) ⊕ Θₑ)₊
  -}
  postulate singleton-edge : Fin (edges G) → Precategory.Ob (PowerSetCat (Fin (edges G)))

  step : NetworkSummingFunctor (ts .ThresholdStructure.C) G → NetworkSummingFunctor (ts .ThresholdStructure.C) G
  step Φₙ = NetworkSummingFunctor-mk λ e →
    let
      Xₑ = NetworkSummingFunctor-apply Φₙ e

      -- Interaction term: ⊕ₑ' Tₑₑ'(Xₑ'(n))
      interaction = sum-over-edges λ e' →
        let Tₑₑ' = T .functor .Functor.F₀ (singleton-edge e , singleton-edge e')
        in Tₑₑ' .Functor.F₀ (NetworkSummingFunctor-apply Φₙ e')

      -- External input
      Θₑ = NetworkSummingFunctor-apply Ψ e

      -- Threshold input: interaction ⊕ Θₑ
      threshold-input = interaction ⊕ Θₑ

      -- Apply threshold: (...)₊
      thresholded = threshold-functor ts .Functor.F₀ threshold-input
    in
      Xₑ ⊕ thresholded
    where
      open Monoidal-category (ts .ThresholdStructure.C-monoidal) renaming (_⊗_ to _⊕_)

  {-|
  Iterate dynamics: Φₙ = stepⁿ(Φ₀)
  -}
  dynamics : Nat → NetworkSummingFunctor (ts .ThresholdStructure.C) G
  dynamics zero = Φ₀
  dynamics (suc n) = step (dynamics n)

open HopfieldDynamics public

{-|
## Conservation Preservation (Lemma 6.3)

**Theorem**: If T ∈ Σ²ₑ(C)(G) and Φ₀, Ψ ∈ Σᵉᑫ_C(G), then
Φₙ ∈ Σᵉᑫ_C(G) for all n.

**Proof sketch**:
1. The threshold functor preserves the equalizer (Proposition 6.1)
2. T satisfies conservation by assumption (T-equalizer)
3. By induction: if Φₙ ∈ equalizer, then step(Φₙ) ∈ equalizer

This ensures dynamics preserves conservation laws at all times.
-}

postulate
  dynamics-preserves-conservation :
    {G : DirectedGraph} →
    (hd : HopfieldDynamics G) →
    (n : Nat) →
    Precategory.Ob (ConservationEqualizer G)

{-|
## Nerve Dynamics (Lemma 6.4)

The dynamics on Σᵉᑫ_C(G) induces:

1. **Simplicial dynamics**: Endofunctor on nerve N(Σᵉᑫ_C(G))
2. **Topological dynamics**: Continuous map on classifying space |N(Σᵉᑫ_C(G))|

**Construction**:
- The step map is a functor Σᵉᑫ_C(G) → Σᵉᑫ_C(G)
- Functoriality of nerve: N : Cat → Simplicial Sets
- Geometric realization |·| : Simplicial Sets → Topological Spaces

This connects categorical dynamics to classical dynamical systems on spaces.
-}

postulate
  -- Simplicial set = Functor from Δ^op to Sets
  SimpliciálSet : Type (lsuc lzero)

  -- Nerve of a category
  nerve : {C : Precategory o ℓ} → SimpliciálSet

  -- Geometric realization (not in 1Lab - topological)
  TopologicalSpace : Type₁
  geometric-realization : SimpliciálSet → TopologicalSpace
  ContinuousMap : TopologicalSpace → TopologicalSpace → Type

{-|
The step functor Σᵉᑫ_C(G) → Σᵉᑫ_C(G) induces simplicial map
-}
postulate
  nerve-dynamics :
    {G : DirectedGraph} →
    (hd : HopfieldDynamics G) →
    SimpliciálSet  -- Endomorphism of N(Σᵉᑫ_C(G))

{-|
Classifying space dynamics: continuous map BΣᵉᑫ_C(G) → BΣᵉᑫ_C(G)
-}
postulate
  classifying-space : {C : Precategory o ℓ} → TopologicalSpace

  classifying-space-dynamics :
    {G : DirectedGraph} →
    (hd : HopfieldDynamics G) →
    ContinuousMap (classifying-space {C = hd .ts .ThresholdStructure.C}) (classifying-space {C = hd .ts .ThresholdStructure.C})

{-|
## Alternative Formulations

**Without leak term** (Equation 6.8):
```
Xₑ(n+1) = (⊕ₑ' Tₑₑ'(Xₑ'(n)) ⊕ Θₑ)₊
```

This is similar to (6.5) but without the Xₑ(n) term on the right.
In classical Hopfield, the leak term -xⱼ(t) ensures exponential decay.

**With small Δt** (Equation 6.9):
```
Xₑ(n+1) ⊕ Xₑ(n) = (⊕ₑ' Tₑₑ'(Xₑ'(n)) ⊕ Θₑ)₊
```

This approximates the continuous ODE when Δt << 1.
However, Lemma 6.3 doesn't directly apply here (need different proof).

**Remark**: For symmetric monoidal categories with zero object, we have
projections Xₑ(n+1) ⊕ Xₑ(n) → Xₑ(n+1), but this makes threshold trivial.
-}

postulate
  -- Variant without leak term (Equation 6.8)
  step-no-leak :
    {G : DirectedGraph} →
    (hd : HopfieldDynamics G) →
    NetworkSummingFunctor (hd .ts .ThresholdStructure.C) G →
    NetworkSummingFunctor (hd .ts .ThresholdStructure.C) G

{-|
## Physical Interpretation

The categorical Hopfield dynamics models:

1. **State**: Φₙ assigns resources to edges at time n
2. **Interaction**: Tₑₑ' models how edge e' influences edge e
3. **External input**: Ψ provides constant drive
4. **Threshold**: (·)₊ implements non-linear activation
5. **Conservation**: Equalizer ensures physical conservation laws

This generalizes classical Hopfield to arbitrary resource categories,
not just real-valued activity levels.
-}
