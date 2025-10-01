{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Information Geometry (Section 8.1)

This module implements information geometry structures for probability
distributions, including divergences, Fisher-Rao metric, and geodesics.

## Overview

**Information geometry** studies probability distributions as points on a
statistical manifold with geometric structure induced by information measures.

**Key structures**:
1. **Divergences**: D(p||q) measures distance between distributions
2. **Fisher-Rao metric**: Riemannian metric on probability distributions
3. **Geodesics**: Shortest paths between distributions
4. **Dual connections**: ∇ and ∇* giving dually flat structure
5. **Exponential/mixture families**: Canonical coordinate systems

## Applications

- Neural network training via natural gradient
- Information bottleneck principle
- Integrated information theory (Section 8.2)
- Variational inference

## References

- Amari, "Information Geometry and Its Applications" [4]
- Ay et al., "Information Geometry" [9]
- Manin & Marcolli (2024), Section 8.1

-}

module Neural.Information.Geometry where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Data.Nat.Base using (Nat; zero; suc)

open import Neural.Information public
  using (ℝ; _+ℝ_; _*ℝ_; _≤ℝ_; _≥ℝ_; zeroℝ; oneℝ)

private variable
  ℓ : Level

{-|
## Probability Distributions

A **probability distribution** over a finite set X is a function p : X → [0,1]
with Σ_x p(x) = 1.

We work with discrete distributions for simplicity.
-}

postulate
  -- Probability distribution over n-element set
  Prob : Nat → Type

  Prob-def :
    (n : Nat) →
    {-| Prob n ≡ { p : Fin n → ℝ | 0 ≤ p(i) ≤ 1 and Σᵢ p(i) = 1 } -}
    ⊤

  -- Evaluation: p(i) is probability of event i
  prob-at : {n : Nat} → Prob n → Nat → ℝ

  -- Normalization: probabilities sum to 1
  prob-normalized :
    {n : Nat} →
    (p : Prob n) →
    {-| Σᵢ prob-at p i = 1 -}
    ⊤

  -- Non-negativity: 0 ≤ p(i) ≤ 1
  prob-nonneg :
    {n : Nat} →
    (p : Prob n) →
    (i : Nat) →
    {-| 0 ≤ prob-at p i ≤ 1 -}
    ⊤

{-|
## Divergences

A **divergence** D(p||q) measures how "far" distribution p is from q.

**Properties**:
1. D(p||q) ≥ 0
2. D(p||q) = 0 iff p = q
3. Generally NOT symmetric: D(p||q) ≠ D(q||p)
4. Generally does NOT satisfy triangle inequality

Common divergences:
- KL divergence: D_KL(p||q) = Σᵢ p(i) log(p(i)/q(i))
- Rényi divergence: D_α(p||q) = (1/(α-1)) log Σᵢ p(i)^α q(i)^(1-α)
- f-divergence: D_f(p||q) = Σᵢ q(i) f(p(i)/q(i))
-}

postulate
  -- Divergence between probability distributions
  Divergence : Type

  Divergence-def :
    {-| Divergence is function D : Prob n × Prob n → ℝ≥0 -}
    ⊤

  -- Evaluation
  divergence :
    {n : Nat} →
    Divergence →
    Prob n → Prob n →
    ℝ

  -- Non-negativity
  divergence-nonneg :
    {n : Nat} →
    (D : Divergence) →
    (p q : Prob n) →
    zeroℝ ≤ℝ divergence D p q

  -- Identity of indiscernibles
  divergence-zero :
    {n : Nat} →
    (D : Divergence) →
    (p q : Prob n) →
    {-| divergence D p q = 0 iff p ≡ q -}
    ⊤

{-|
## Kullback-Leibler Divergence

The **KL divergence** (relative entropy) is:

  D_KL(p||q) = Σᵢ p(i) log(p(i)/q(i))

**Physical meaning**:
- Information gained when true distribution is p but we assumed q
- Expected log-likelihood ratio
- Forward KL: mode-seeking, reverse KL: mean-seeking

**Properties**:
- D_KL(p||q) ≥ 0 (Gibbs inequality)
- D_KL(p||q) = 0 iff p = q
- NOT symmetric: D_KL(p||q) ≠ D_KL(q||p) in general
-}

postulate
  -- KL divergence
  D-KL : Divergence

  D-KL-def :
    {n : Nat} →
    (p q : Prob n) →
    {-| divergence D-KL p q = Σᵢ p(i) * log(p(i) / q(i)) -}
    ⊤

  -- Gibbs inequality
  gibbs-inequality :
    {n : Nat} →
    (p q : Prob n) →
    zeroℝ ≤ℝ divergence D-KL p q

  -- Forward vs reverse KL
  forward-reverse-KL :
    {-| D_KL(p||q) minimizes ∫ p log q (mode-seeking)
        D_KL(q||p) minimizes ∫ q log p (mean-seeking) -}
    ⊤

{-|
## Fisher-Rao Metric

The **Fisher information metric** (Fisher-Rao metric) is a Riemannian metric
on the space of probability distributions:

  g_ij(p) = E_p[∂ᵢ log p · ∂ⱼ log p]

where ∂ᵢ log p is the score function.

**Physical meaning**:
- Measures distinguishability of nearby distributions
- Infinitesimal version of KL divergence
- Unique (up to scaling) Riemannian metric invariant under sufficient statistics

**Properties**:
- Positive definite
- Invariant under reparametrization
- Geodesic distance = statistical distinguishability
-}

postulate
  -- Fisher information matrix
  Fisher : {n : Nat} → Prob n → Type

  Fisher-def :
    {n : Nat} →
    (p : Prob n) →
    {-| Fisher p is matrix g_ij = E_p[(∂ᵢ log p)(∂ⱼ log p)] -}
    ⊤

  -- Fisher-Rao metric
  FisherRao-metric :
    {-| Riemannian metric on probability simplex -}
    ⊤

  -- Positive definiteness
  fisher-positive :
    {n : Nat} →
    (p : Prob n) →
    {-| Fisher p is positive definite -}
    ⊤

  -- Reparametrization invariance
  fisher-invariant :
    {-| Fisher metric is invariant under smooth reparametrizations -}
    ⊤

{-|
## Geodesics

A **geodesic** in the Fisher-Rao geometry is the shortest path between two
probability distributions.

For the **e-connection** (exponential family), geodesics are:
  p_t = p_0 exp(t · log(p_1/p_0)) / Z(t)

For the **m-connection** (mixture family), geodesics are:
  p_t = (1-t) p_0 + t p_1

**Physical meaning**: Optimal interpolation between distributions that minimizes
statistical distinguishability.
-}

postulate
  -- Geodesic between distributions
  geodesic :
    {n : Nat} →
    Prob n → Prob n →
    (t : ℝ) →  -- Parameter t ∈ [0,1]
    Prob n

  geodesic-endpoints :
    {n : Nat} →
    (p q : Prob n) →
    {-| geodesic p q 0 = p and geodesic p q 1 = q -}
    ⊤

  -- e-geodesic (exponential connection)
  e-geodesic :
    {n : Nat} →
    (p q : Prob n) →
    (t : ℝ) →
    Prob n

  e-geodesic-def :
    {n : Nat} →
    (p q : Prob n) →
    (t : ℝ) →
    {-| e-geodesic p q t ∝ p^(1-t) * q^t (geometric mean) -}
    ⊤

  -- m-geodesic (mixture connection)
  m-geodesic :
    {n : Nat} →
    (p q : Prob n) →
    (t : ℝ) →
    Prob n

  m-geodesic-def :
    {n : Nat} →
    (p q : Prob n) →
    (t : ℝ) →
    {-| m-geodesic p q t = (1-t)*p + t*q (arithmetic mean) -}
    ⊤

{-|
## Dual Connections and Dually Flat Structure

The Fisher-Rao geometry admits **dual affine connections** ∇ and ∇*:
- ∇: exponential connection (e-connection)
- ∇*: mixture connection (m-connection)

**Pythagorean theorem**: For dually flat space with geodesics,
  D(p||r) = D(p||q) + D(q||r)
when q is on the ∇-geodesic from p to r.

**Physical meaning**: Information decomposes additively along geodesics.
-}

postulate
  -- Dual connections
  e-connection : {-| Exponential/natural connection ∇ -} ⊤
  m-connection : {-| Mixture/mixture connection ∇* -} ⊤

  -- Dually flat structure
  dually-flat :
    {-| Fisher-Rao geometry is dually flat -}
    ⊤

  -- Pythagorean theorem
  pythagorean-divergence :
    {n : Nat} →
    (p q r : Prob n) →
    {-| If q on e-geodesic from p to r, then
        D_KL(p||r) = D_KL(p||q) + D_KL(q||r) -}
    ⊤

  -- Generalized Pythagorean theorem
  generalized-pythagorean :
    {-| For any f-divergence on dually flat manifold -}
    ⊤

{-|
## Exponential and Mixture Families

An **exponential family** has distributions of the form:
  p_θ(x) = exp(θ · T(x) - A(θ))

where θ are natural parameters, T(x) are sufficient statistics, A(θ) is log-partition.

A **mixture family** is parametrized by expectations η = E_θ[T(x)].

**Legendre duality**: Natural parameters θ and expectation parameters η are
Legendre dual: A*(η) = sup_θ (θ · η - A(θ)).
-}

postulate
  -- Exponential family
  ExponentialFamily : Nat → Type

  ExponentialFamily-def :
    (n : Nat) →
    {-| Exponential family with n-dimensional parameter θ -}
    ⊤

  -- Natural parameters
  natural-param :
    {n : Nat} →
    ExponentialFamily n →
    ℝ  -- θ vector

  -- Log-partition function
  log-partition :
    {n : Nat} →
    (θ : ℝ) →
    ℝ

  log-partition-def :
    {n : Nat} →
    (θ : ℝ) →
    {-| log-partition θ = log Σ_x exp(θ · T(x)) -}
    ⊤

  -- Mixture family (dual parametrization)
  MixtureFamily : Nat → Type

  MixtureFamily-def :
    (n : Nat) →
    {-| Mixture family with expectation parameters η -}
    ⊤

  -- Legendre duality
  legendre-dual :
    {n : Nat} →
    {-| θ and η are Legendre dual via A and A* -}
    ⊤

{-|
## Natural Gradient

The **natural gradient** is the gradient in the Fisher-Rao geometry:

  ∇̃ L = F⁻¹ ∇ L

where F is the Fisher information matrix and L is the loss function.

**Physical meaning**:
- Steepest descent in the intrinsic geometry of probability distributions
- Invariant under reparametrization
- Faster convergence than vanilla gradient descent

**Applications**:
- Neural network training
- Variational inference
- Reinforcement learning (policy gradient methods)
-}

postulate
  -- Natural gradient
  natural-gradient :
    {n : Nat} →
    (loss : Prob n → ℝ) →
    (p : Prob n) →
    {-| Direction of steepest descent in Fisher-Rao geometry -}
    ℝ  -- Tangent vector

  natural-gradient-def :
    {n : Nat} →
    (loss : Prob n → ℝ) →
    (p : Prob n) →
    {-| natural-gradient loss p = (Fisher p)⁻¹ · ∇ loss(p) -}
    ⊤

  -- Reparametrization invariance
  natural-gradient-invariant :
    {-| Natural gradient is invariant under smooth reparametrization -}
    ⊤

  -- Convergence theorem
  natural-gradient-convergence :
    {-| Natural gradient has better convergence than vanilla gradient -}
    ⊤

{-|
## Information Bottleneck

The **information bottleneck** principle finds optimal compression of input X
to representation T that preserves information about output Y:

  min I(X;T) - β I(T;Y)

where β is a tradeoff parameter.

**Physical meaning**: Find minimal sufficient statistics for prediction.

**Applications**:
- Deep learning: intermediate layers as information bottleneck
- Neural coding: efficient representations in sensory systems
-}

postulate
  -- Mutual information (parametrized by joint distribution)
  mutual-information :
    {n m : Nat} →
    ℝ

  mutual-information-def :
    {n m : Nat} →
    {-| I(X;Y) = Σ p(x,y) log(p(x,y)/(p(x)p(y))) for joint distribution p -}
    ⊤

  -- Information bottleneck functional
  IB-functional :
    (β : ℝ) →
    {-| Functional I(X;T) - β I(T;Y) to minimize -}
    ℝ

  -- Optimal compression
  IB-optimal :
    (β : ℝ) →
    {-| Optimal T*(β) solving information bottleneck -}
    ⊤

  -- Phase transitions
  IB-phase-transition :
    {-| As β varies, optimal T* undergoes phase transitions -}
    ⊤

{-|
## Physical Interpretation for Neural Networks

In neural networks:

1. **Parameter space = Statistical manifold**: Network parameters θ define
   distribution p_θ(output|input).

2. **Training = Geodesic flow**: Gradient descent follows geodesics in
   Fisher-Rao geometry.

3. **Natural gradient = Optimal training**: Fisher-Rao steepest descent gives
   fastest convergence.

4. **Layers = Information bottleneck**: Each layer compresses information while
   preserving task-relevant features.

5. **Generalization = Divergence**: KL divergence between train and test
   distributions measures overfitting.

6. **Transfer learning = Geodesic interpolation**: Fine-tuning moves along
   geodesic from pre-trained to task-specific distribution.

**Connection to consciousness** (Section 8.2): The integrated information Φ
can be formulated as divergence from product distribution, making it a
geometric quantity in information geometry.
-}
