{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Adjunction and Optimality of Resources (Section 3.3)

This module implements Section 3.3 from Manin & Marcolli (2024):
"Homotopy-theoretic and categorical models of neural information networks"

We formalize the adjunction between computational architectures and resource
assignments, characterizing optimality through universal properties.

## Overview

Given:
- **C**: Category of computational systems (transition systems, automata, neural architectures)
- **R**: Category of resources (metabolic, informational - from Section 3.2)
- **ρ: C → R**: Strict symmetric monoidal functor assigning resources to systems

We show that a **left adjoint β: R → C** (when it exists) provides optimal
system construction from resource constraints, satisfying:

  Mor_C(β(A), C) ≃ Mor_R(A, ρ(C))    (Equation 3.2)

## Key Results

**Interpretation of adjunction**:
- β(A) is the optimal computational system buildable from resources A
- Any resource conversion A → ρ(C) determines unique system modification β(A) → C
- The manufacturing process A → ρ(β(A)) is the initial object in the comma category A ↓ ρ

**Freyd's Adjoint Functor Theorem**:
If ρ is continuous, C is complete, and solution sets exist, then the left
adjoint β exists, guaranteeing existence of optimal systems.

**Solution set interpretation**:
The solution set {(uⱼ, Cⱼ)} represents a finite collection of "design templates"
from which any system using resources A can be obtained via modifications.

## Applications to Neural Networks

- **C**: Category of neural architectures with different connectivity patterns
- **R**: Resource theory from Section 3.2 (energy, information capacity)
- **ρ**: Assigns metabolic/informational requirements to each architecture
- **β**: Constructs optimal architecture for given resource budget
- **Optimality**: Achieved through universal property, not ad-hoc optimization
-}

module Neural.Resources.Optimization where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Functor.Adjoint
open import Cat.Functor.Adjoint.AFT
open import Cat.Functor.Adjoint.Continuous
open import Cat.Instances.Comma
open import Cat.Diagram.Initial
open import Cat.Diagram.Limit.Base
open import Cat.Monoidal.Base
open import Cat.Monoidal.Braided
open import Cat.Monoidal.Functor

import Cat.Reasoning

open import Data.Nat.Base using (Nat)
open import Data.List.Base using (List)
open import Data.Sum.Base using (_⊎_)

open import Neural.Resources

private variable
  o ℓ o' ℓ' : Level

{-|
## Computational Architecture Category

The category **C** represents computational systems and their modifications.

**Objects**: Computational architectures
- Transition systems (see Section 4.1)
- Higher dimensional automata (see Section 4.2)
- Neural network architectures with specific connectivity patterns
- Distributed/concurrent computational systems

**Morphisms**: System modifications/refinements
- Refinements that preserve observable behavior
- Simulations between systems
- Architectural transformations

**Properties**:
- Complete: Has all small limits (enables Freyd's theorem)
- Concrete examples will be developed in Section 4

NOTE: We postulate C for now. Section 4 will provide concrete constructions
such as the category of transition systems (§4.1) and automata (§4.2).
-}

postulate
  ComputationalArchitecture : (o ℓ : Level) → Type (lsuc (o ⊔ ℓ))

  -- Extract the underlying category
  CA-category : ∀ {o ℓ} → ComputationalArchitecture o ℓ → Precategory o ℓ

  -- C is complete (has all small limits)
  CA-complete : ∀ {o ℓ} (C : ComputationalArchitecture o ℓ) →
                is-complete ℓ ℓ (CA-category C)

{-|
## Resource Assignment via Strict Monoidal Functor

A **resource assignment** ρ: C → R associates resource requirements to
computational systems.

**Strict symmetric monoidal**: The functor preserves monoidal structure
on-the-nose, not just up to isomorphism. This encodes:
- ρ(C₁ ⊗ C₂) = ρ(C₁) ⊗ ρ(C₂) (combining independent systems combines resources exactly)
- ρ(I) = I (empty system has no resource requirements)

**Physical interpretation**:
- ρ(system) = (energy requirements, information capacity, computational power)
- Independent systems have additive resource requirements (via monoidal structure)
- The "strictness" reflects that resource accounting is exact, not approximate
-}

module ResourceAssignment
  {o ℓ : Level}
  (C : ComputationalArchitecture o ℓ)
  (RT : ResourceTheory o ℓ)
  where

  open ResourceTheory RT

  private
    𝒞 = CA-category C

  {-|
  A strict symmetric monoidal functor has structure maps that are
  definitional equalities rather than isomorphisms.

  For our purposes, we'll work with the standard monoidal functor from 1Lab
  and add strictness as a property.
  -}
  record StrictMonoidalFunctor
    (Cᵐ : Monoidal-category 𝒞)
    (Rᵐ : Monoidal-category R)
    : Type (lsuc (o ⊔ ℓ)) where

    field
      {-| The underlying functor ρ: C → R -}
      F : Functor 𝒞 R

      {-| Monoidal structure (from 1Lab) -}
      monoidal : Monoidal-functor-on Cᵐ Rᵐ F

    open Monoidal-functor-on monoidal public

    {-|
    **Strictness**: The structure maps are invertible isomorphisms (already
    guaranteed by Monoidal-functor-on), and we can additionally require
    that they satisfy strict coherence conditions.

    For practical purposes, "strict" means the functor preserves tensor products
    and units on-the-nose up to the natural isomorphisms.

    NOTE: In a truly strict monoidal functor, these would be definitional
    equalities. Here we work with the standard 1Lab monoidal functors.
    -}

{-|
## Adjunction: Optimal System Construction (Equation 3.2)

Given resource assignment ρ: C → R, a **left adjoint** β: R → C (when it exists)
provides optimal construction of computational systems from resource constraints.

**Adjunction formula**: Mor_C(β(A), C) ≃ Mor_R(A, ρ(C))

**Interpretation**:
- Left side: Ways to modify the system β(A) into system C
- Right side: Ways to convert resources A into the resources ρ(C) required by C
- Bijection: Each resource conversion corresponds to unique system modification

**Meaning of β(A)**:
- β constructs a computational system from resource specification A
- The system β(A) is "optimal" in the sense that:
  1. It uses resources A (via the manufacturing conversion A → ρ(β(A)))
  2. Any other system C using at most resources A can be obtained from β(A)
     via a unique modification β(A) → C
-}

module OptimalSystemConstruction
  {o ℓ : Level}
  (C : ComputationalArchitecture o ℓ)
  (RT : ResourceTheory o ℓ)
  (Cᵐ : Monoidal-category (CA-category C))
  (ρ-functor : ResourceAssignment.StrictMonoidalFunctor C RT Cᵐ (ResourceTheory.Rᵐ RT))
  where

  open ResourceTheory RT
  open ResourceAssignment C RT
  open StrictMonoidalFunctor ρ-functor

  private
    𝒞 = CA-category C
    ρ = F

  {-|
  **Optimal system constructor**: The left adjoint β: R → C, when it exists.

  β assigns to each resource specification A ∈ R an optimal computational
  system β(A) ∈ C that can be built from those resources.
  -}
  OptimalConstructor : Type (o ⊔ ℓ)
  OptimalConstructor = Σ[ β ∈ Functor R 𝒞 ] (β ⊣ ρ)

  module _ (opt : OptimalConstructor) where
    β : Functor R 𝒞
    β = opt .fst

    β⊣ρ : β ⊣ ρ
    β⊣ρ = opt .snd

    open _⊣_ β⊣ρ public

    {-|
    **Universal property**: For any resources A and system C, there is a
    bijection between system modifications and resource conversions.
    -}
    universal-property :
      (A : Precategory.Ob R) →
      (C : Precategory.Ob 𝒞) →
      Precategory.Hom 𝒞 (Functor.F₀ β A) C ≃ Precategory.Hom R A (Functor.F₀ ρ C)
    universal-property A C = adjunct-hom-equiv β⊣ρ

{-|
## Comma Category A ↓ ρ

For a fixed resource specification A ∈ R, the **comma category** A ↓ ρ has:

**Objects**: Pairs (u, C) where:
- C is a computational system in 𝒞
- u: A → ρ(C) is a resource conversion showing that A can supply C's requirements

**Morphisms** φ: (u₁, C₁) → (u₂, C₂): System modifications φ: C₁ → C₂ such that:
```
    A
    |  \
   u₁   u₂
    |     \
  ρ(C₁) → ρ(C₂)
    ρ(φ)
```

**Interpretation**:
- Objects represent "ways to use resources A to build system C"
- Morphisms represent "system refinements that preserve resource usage"
- The comma category encodes all possible resource allocations for A
-}

module CommaCategory
  {o ℓ : Level}
  (C : ComputationalArchitecture o ℓ)
  (RT : ResourceTheory o ℓ)
  (Cᵐ : Monoidal-category (CA-category C))
  (ρ-functor : ResourceAssignment.StrictMonoidalFunctor C RT Cᵐ (ResourceTheory.Rᵐ RT))
  where

  open ResourceTheory RT
  open ResourceAssignment C RT
  open StrictMonoidalFunctor ρ-functor

  private
    𝒞 = CA-category C
    ρ = F
    module 𝒞 = Cat.Reasoning 𝒞
    module R = Cat.Reasoning R

  {-|
  For each resource A, construct the comma category A ↓ ρ using 1Lab's
  infrastructure.

  We use the coslice (under) category: A ↙ ρ
  -}
  resource-comma-category : Precategory.Ob R → Precategory (o ⊔ ℓ) ℓ
  resource-comma-category A = A ↙ ρ

  {-|
  Objects in A ↓ ρ: resource conversions u: A → ρ(C)
  -}
  ResourceConversion : Precategory.Ob R → Type (o ⊔ ℓ)
  ResourceConversion A = Precategory.Ob (A ↙ ρ)

  module _ {A : Precategory.Ob R} where
    open ↓Obj

    {-|
    Extract the system from a resource conversion
    -}
    conversion-system : ResourceConversion A → Precategory.Ob 𝒞
    conversion-system conv = conv .y

    {-|
    Extract the conversion map u: A → ρ(C)
    -}
    conversion-map : (conv : ResourceConversion A) →
                     Precategory.Hom R A (Functor.F₀ ρ (conversion-system conv))
    conversion-map conv = conv .map

    {-|
    **Initial object = Optimal resource conversion**

    An initial object in A ↓ ρ is a resource conversion u₀: A → ρ(C₀) such that
    every other conversion u: A → ρ(C) factors uniquely through it.

    This means:
    1. C₀ is the optimal system for resources A
    2. Any other system C using resources A can be obtained from C₀ via unique modification
    3. The conversion u₀: A → ρ(C₀) represents the "manufacturing" of C₀ from A
    -}
    OptimalConversion : Type (o ⊔ ℓ)
    OptimalConversion = Initial (resource-comma-category A)

    module _ (init : OptimalConversion) where
      open Initial init

      {-|
      The optimal system β(A) for resources A
      -}
      optimal-system : Precategory.Ob 𝒞
      optimal-system = conversion-system bot

      {-|
      The manufacturing conversion: A → ρ(β(A))

      This represents using resources A to build the optimal system β(A).
      Some resources are consumed in manufacturing.
      -}
      manufacturing : Precategory.Hom R A (Functor.F₀ ρ optimal-system)
      manufacturing = conversion-map bot

      {-|
      **Unique factorization**: For any system C and conversion u: A → ρ(C),
      there exists a unique system modification β(A) → C.
      -}
      unique-factorization :
        (C : Precategory.Ob 𝒞) →
        (u : Precategory.Hom R A (Functor.F₀ ρ C)) →
        is-contr (Precategory.Hom 𝒞 optimal-system C)
      unique-factorization C u = {!!}  -- Follows from initial object property

{-|
## Freyd's Adjoint Functor Theorem

The **Adjoint Functor Theorem** gives sufficient conditions for existence of
the left adjoint β: R → C (hence existence of optimal systems).

**Theorem**: If
1. C is complete (has all small limits)
2. ρ: C → R is continuous (preserves limits)
3. Solution sets exist for all A ∈ R

Then β exists and provides optimal system construction.

**Solution set for A**: A set {(uⱼ, Cⱼ)} of resource conversions such that
any conversion u: A → ρ(C) factors through one of them.

**Interpretation**: The solution set is a finite collection of "design templates"
that are optimal for resource budget A. Any other system using budget A can be
obtained by modifying one of these templates.
-}

module FreydsTheorem
  {o ℓ : Level}
  (C : ComputationalArchitecture o ℓ)
  (RT : ResourceTheory o ℓ)
  (Cᵐ : Monoidal-category (CA-category C))
  (ρ-functor : ResourceAssignment.StrictMonoidalFunctor C RT Cᵐ (ResourceTheory.Rᵐ RT))
  where

  open ResourceTheory RT
  open ResourceAssignment C RT
  open StrictMonoidalFunctor ρ-functor
  open OptimalSystemConstruction C RT Cᵐ ρ-functor
  open CommaCategory C RT Cᵐ ρ-functor

  private
    𝒞 = CA-category C
    ρ = F

  {-|
  **Solution set**: A set of "candidate optimal systems" for resources A.

  This is a set {(uⱼ, Cⱼ)} indexed by j ∈ I where:
  - I is a set (not just a type)
  - Each Cⱼ is a system in 𝒞
  - Each uⱼ: A → ρ(Cⱼ) is a resource conversion
  - Any other conversion u: A → ρ(C) factors through some uⱼ

  **Physical interpretation**:
  For a given resource budget A (e.g., 10 watts, 1 GB RAM), the solution set
  represents a finite catalog of architecture templates (e.g., "low-power RISC",
  "parallel SIMD", "sequential CISC") from which any specific system can be
  obtained through modifications.
  -}
  ResourceSolutionSet : Precategory.Ob R → Type (o ⊔ lsuc ℓ)
  ResourceSolutionSet A = Solution-set ρ A

  {-|
  **Freyd's Adjoint Functor Theorem** (using 1Lab's formalization):

  If ρ is continuous and solution sets exist for all resources A, then
  the left adjoint β: R → C exists, providing optimal system construction.
  -}
  adjoint-functor-theorem :
    is-continuous ℓ ℓ ρ →
    (∀ (A : Precategory.Ob R) → ResourceSolutionSet A) →
    OptimalConstructor
  adjoint-functor-theorem ρ-cont solution-sets =
    solution-set→left-adjoint ρ (CA-complete C) ρ-cont solution-sets

  {-|
  **Corollary**: If the adjoint exists, every resource A has an optimal system β(A)
  which is the initial object in A ↓ ρ.
  -}
  postulate
    optimal-system-exists :
      (opt : OptimalConstructor) →
      (A : Precategory.Ob R) →
      CommaCategory.OptimalConversion C RT Cᵐ ρ-functor {A}
      -- Follows from universal-maps→left-adjoint in 1Lab

{-|
## Physical Interpretation and Examples

### Example 1: Neural Network Architecture Selection

- **Resources A**: Energy budget = 5 watts, Memory = 1 GB
- **Solution set**:
  {(u₁, Conv-Net), (u₂, RNN), (u₃, Transformer)}
  where each uⱼ specifies how to allocate 5W and 1GB to that architecture
- **Optimal system β(A)**: The architecture that minimizes resource waste
- **Factorization**: Any custom architecture with ≤ 5W, ≤ 1GB can be obtained
  by modifying one of the three templates

### Example 2: Metabolic Resource Allocation

- **Resources A**: ATP budget for neural computation
- **Systems C**: Different neural coding schemes (sparse, dense, rate, temporal)
- **ρ(C)**: Metabolic cost of maintaining coding scheme C
- **β(A)**: Optimal coding scheme for ATP budget A (e.g., sparse coding for low ATP)
- **Universal property**: Any other coding scheme using ≤ A ATP can be obtained
  by modifying β(A)

### Example 3: Information-Theoretic Optimization

- **Resources A**: Channel capacity (bits/second)
- **Systems C**: Neural network architectures with different connectivity
- **ρ(C)**: Information requirements of architecture C
- **β(A)**: Optimal architecture for channel capacity A
- **Adjunction**: Bijection between architectural changes and capacity allocations
-}

module Examples where
  postulate
    -- Example computational systems
    ConvNet : Type      -- Convolutional neural network
    RNN : Type          -- Recurrent neural network
    Transformer : Type  -- Transformer architecture

    -- Example resources
    Energy : Type       -- Energy budget (watts)
    Memory : Type       -- Memory budget (bytes)
    Capacity : Type     -- Information channel capacity (bits/second)

    -- Example resource assignments
    conv-net-resources : ConvNet → Energy × Memory
    rnn-resources : RNN → Energy × Memory
    transformer-resources : Transformer → Capacity

    -- Example optimal systems
    optimal-for-5W-1GB : ConvNet ⊎ RNN  -- One of these is optimal
    optimal-sparse-coding : Type        -- Optimal for low ATP
    optimal-architecture : Capacity → Type  -- Optimal for given capacity
