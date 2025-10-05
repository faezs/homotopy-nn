{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Neural Information Networks and Resources (Section 3.1)

This module implements Section 3.1 from Manin & Marcolli (2024):
"Homotopy-theoretic and categorical models of neural information networks"

We formalize neural codes, firing rates, and the fundamental tension between
metabolic efficiency and information transmission in neural networks.

## Overview

Neural networks face optimization pressures from:
1. **Information transmission**: Coding efficiency, information rate
2. **Metabolic efficiency**: Energy cost per spike, maintenance costs
3. **Computational capacity**: Concurrent/distributed processing resources

These constraints are inversely related: optimizing both metabolic efficiency
and information rate is analogous to the tradeoff between code rate and minimum
distance in error-correcting codes.

## Key Concepts

### Neural Codes (§3.1.1)

- **Binary codes**: On/off patterns representing which neurons fire in time intervals
- **Rate codes**: Information encoded in firing rate (spikes per second)
- **Spike timing codes**: Precise timing of spikes carries information
- **Correlation codes**: Probability of spike and time intervals between spikes

### Coding Capacity (§3.1.2)

Using Poisson process for spike generation with firing rate y spikes/second:
- Spikes are mutually independent at given firing rate
- Maximum coding rate: Rmax = −y log(y∆t)
- Output entropy H divided by time interval ∆t

### Metabolic Efficiency (§3.1.3)

Metabolic efficiency ϵ = I(X,Y)/E where:
- I(X,Y): Mutual information of output Y and input X
- E: Energy cost per unit time (channel maintenance + signal power)
- Signal power depends on axon myelination and diameter

### Connection Weights (§3.1.4)

For K neurons over N time steps with output matrix X = (xk,n):
- Transmission to R cells via weight matrix W = (wr,k)
- Output: yr,n = Σk wr,k xk,n + ηr,n (with noise η)
- Optimization: weights W that maximize mutual information I(X,Y)
-}

module Neural.Information where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Algebra.Ring

open import Cat.Base
open import Cat.Functor.Base

open import Data.Nat.Base using (Nat; zero; suc; _+_)
open import Data.Fin.Base using (Fin)
open import Data.Bool.Base using (Bool; true; false)
open import Data.List.Base using (List; []; _∷_)

open import Neural.Base

private variable
  o ℓ : Level

-- Real numbers (postulated for now - 1Lab doesn't have them built-in)
postulate
  ℝ : Type
  _*ℝ_ : ℝ → ℝ → ℝ
  _/ℝ_ : ℝ → ℝ → ℝ
  _+ℝ_ : ℝ → ℝ → ℝ
  -ℝ_ : ℝ → ℝ
  logℝ : ℝ → ℝ

  {-| Order relation on real numbers -}
  _≤ℝ_ : ℝ → ℝ → Type

{-| ≥ℝ defined in terms of ≤ℝ -}
_≥ℝ_ : ℝ → ℝ → Type
x ≥ℝ y = y ≤ℝ x

postulate
  {-| Constants -}
  zeroℝ : ℝ
  oneℝ : ℝ

  {-| Ring structure on ℝ -}
  ℝ-ring : Ring-on ℝ

  {-| Basic properties -}
  ≤ℝ-refl : {x : ℝ} → x ≤ℝ x
  ≤ℝ-trans : {x y z : ℝ} → x ≤ℝ y → y ≤ℝ z → x ≤ℝ z

{-| Open ring structure to use ring notation -}
open Ring-on ℝ-ring public
  using ()
  renaming ( _+_ to _+ℝ'_
           ; _*_ to _*ℝ'_
           ; 1r to 1ℝ
           ; 0r to 0ℝ
           ; -_ to -ℝ'_
           ; _-_ to _-ℝ'_
           )

{-|
## Neural Codes (Definition 3.1)

A **neural code** for a population of n neurons over N time intervals is
a collection of binary vectors representing firing patterns.

For each neuron i, the code word is a length-N binary string where entry j
is 1 if neuron i fired during time interval j, and 0 otherwise.

**Example**: For 3 neurons over 5 time intervals:
  Neuron 1: [1,0,0,1,1]  (fired at times 0,3,4)
  Neuron 2: [0,1,0,0,1]  (fired at times 1,4)
  Neuron 3: [0,0,0,0,0]  (never fired)

**Interpretation**: The i-th "column" across all code words shows which
neurons fired simultaneously during that time interval.

From the example: at time 4, both neurons 1 and 2 fired simultaneously.

NOTE: This is a coarse-grained representation parameterized by the choice
of time discretization ∆t.
-}

-- Binary neural code: List of firing patterns (one per neuron)
BinaryNeuralCode : Nat → Nat → Type
BinaryNeuralCode n-neurons n-timeSteps = Fin n-neurons → (Fin n-timeSteps → Bool)

{-|
## Firing Rate and Coding Capacity (Section 3.1.2)

The **firing rate** y (spikes per second) parameterizes a Poisson process
generating spikes. All spike trains at that rate are equiprobable.

The **coding capacity** is the information content of a spike train with
n spikes over N time intervals:
  Information ∝ log(# ways to arrange n spikes in N intervals)

**Maximum coding rate**: Rmax = -y log(y∆t)
  where y is firing rate and ∆t is the basic time interval.

This represents the maximum information transmission rate achievable
for a given firing rate.
-}

-- Firing rate: spikes per second (represented as a real number)
FiringRate : Type
FiringRate = ℝ

postulate

  -- Maximum coding rate for a given firing rate and time discretization
  max-coding-rate : FiringRate → (Δt : ℝ) → ℝ

  -- Maximum coding rate formula: Rmax = -y log(y∆t)
  max-coding-rate-formula :
    (y : FiringRate) → (Δt : ℝ) →
    max-coding-rate y Δt ≡ (-ℝ y) *ℝ logℝ (y *ℝ Δt)

{-|
## Metabolic Efficiency (Definition 3.2)

The **metabolic efficiency** ϵ of a transmission channel is the ratio:

  ϵ = I(X,Y) / E

where:
- I(X,Y): Mutual information between input X and output Y
- E: Energy cost per unit time = maintenance cost + signal power
- Signal power: Energy required to generate spikes at given firing rate

**Key constraint**: Energy cost per spike depends on:
1. Whether axon is myelinated (myelinated axons are more efficient)
2. Axon diameter (for unmyelinated axons)

**Consequence**: Metabolic efficiency and information rate are inversely related.
Optimizing both simultaneously is a fundamental tradeoff in neural systems.

**Interdependence of resources**: Assignment of informational resources
(mutual information measurements) governs assignment of metabolic resources
once channel energy costs E are known.
-}

postulate
  -- Mutual information I(X,Y) between random variables
  MutualInformation : (X Y : Type) → Type

  -- Energy cost per unit time
  EnergyCost : Type

  -- Extract numerical values
  information-value : {X Y : Type} → MutualInformation X Y → ℝ
  cost-value : EnergyCost → ℝ

  -- Metabolic efficiency: I(X,Y) / E
  MetabolicEfficiency : (X Y : Type) → MutualInformation X Y → EnergyCost → ℝ

  -- Formula: ϵ = I(X,Y) / E
  metabolic-efficiency-formula :
    (X Y : Type) →
    (I : MutualInformation X Y) →
    (E : EnergyCost) →
    MetabolicEfficiency X Y I E ≡ (information-value I) /ℝ (cost-value E)

{-|
## Connection Weights and Information Optimization (Section 3.1.4)

For a network with:
- K neurons responding to a stimulus
- N discrete time steps
- Output encoded as K×N matrix X = (xk,n)
- Transmission to R next-layer cells

The **weight matrix** W = (wr,k) determines how signals propagate:
  yr,n = Σk wr,k xk,n + ηr,n

where η = (ηr,n) is noise (random variable).

**Optimization problem**: Find weights W that maximize mutual information I(X,Y)
between input and output.

**Interdependence**: Informational resources (mutual information) depend on
underlying resources of weighted codes (the weight matrix W).

This demonstrates how different types of resources assigned to a network
are interdependent.
-}

-- Weight matrix W: R × K matrix of connection strengths
WeightMatrix : Nat → Nat → Type
WeightMatrix R K = Fin R → Fin K → ℝ

-- Output computation: Y = WX + η
postulate
  -- Compute output yr,n from inputs xk,n via weight matrix
  weighted-output :
    {K N R : Nat} →
    WeightMatrix R K →                        -- Weight matrix W
    (Fin K → Fin N → ℝ) →                     -- Input matrix X
    (Fin R → Fin N → ℝ) →                     -- Noise matrix η
    (Fin R → Fin N → ℝ)                       -- Output matrix Y

  -- Helper: sum over finite index set
  sumℝ : {K : Nat} → (Fin K → ℝ) → ℝ

  -- Output formula: yr,n = Σk wr,k xk,n + ηr,n
  weighted-output-formula :
    {K N R : Nat} →
    (W : WeightMatrix R K) →
    (X : Fin K → Fin N → ℝ) →
    (η : Fin R → Fin N → ℝ) →
    (r : Fin R) → (n : Fin N) →
    weighted-output W X η r n ≡
      (sumℝ (λ k → (W r k) *ℝ (X k n))) +ℝ (η r n)

{-|
## Resources and Constraints (Section 3.1.5)

We identify three main types of resources assigned to neural networks:

1. **Energy/Metabolic resources**: Power consumption, spike generation cost
2. **Neural codes**: Binary patterns, firing rates, timing information
3. **Information capacity**: Mutual information, channel capacity

These resources are subject to **constraints**:
- **Intrinsic constraints**: Physical limits on firing rates, energy budgets
- **Relational constraints**: Metabolic efficiency vs information rate tradeoff

**Categorical framework benefits**:
1. Symmetric monoidal categories model resource combination (⊗)
2. Summing functors describe resource assignments to networks
3. Functors between resource categories model resource relationships
4. Universal properties characterize optimality constraints

The rest of Section 3 develops this categorical framework in detail.
-}

-- Types of resources assigned to networks
record NetworkResources : Type₁ where
  field
    {-| Energy and metabolic resources (power, spike costs) -}
    metabolic : Type

    {-| Neural codes (binary, rate, timing, correlation codes) -}
    codes : Type

    {-| Information capacity (mutual information, entropy) -}
    information : Type

-- Constraints on resources
record ResourceConstraints (R : NetworkResources) : Type₁ where
  open NetworkResources R
  field
    {-| Intrinsic constraints on individual resource types -}
    metabolic-bounds : metabolic → Type
    code-constraints : codes → Type
    information-limits : information → Type

    {-| Relational constraints between different resource types -}
    efficiency-tradeoff : metabolic → information → Type

postulate
  -- Helper: construct NetworkResources from a category
  mk-resources : Precategory o ℓ → NetworkResources

  {-|
  **Categorical Resource Assignment**: A resource assignment to a network G
  is a summing functor Φ: P(G) → C where C is a category of resources.

  This will be developed in detail in Neural.Resources (Section 3.2).
  -}
  ResourceAssignment : (G : DirectedGraph) → (C : Precategory o ℓ) → Type (o ⊔ ℓ)

  {-|
  **Optimal Resource Assignment**: An assignment Φ is optimal if it satisfies
  universal properties related to the constraints.

  This will be formalized using adjunctions in Section 3.3.
  -}
  IsOptimalAssignment :
    {G : DirectedGraph} →
    {C : Precategory o ℓ} →
    ResourceAssignment G C →
    ResourceConstraints (mk-resources C) →
    Type (o ⊔ ℓ)
