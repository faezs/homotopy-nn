{-# OPTIONS --allow-unsolved-metas #-}

{-|
# Section 3.2: Spontaneous Activity and Driven Dynamics

This module implements Section 3.2 from Belfiore & Bennequin (2022), formalizing
spontaneous activity vertices, driven dynamics, and the interaction between
endogenous (computed) and exogenous (spontaneous) neural activity.

## Paper Reference

"We extend the oriented graph framework to include spontaneous activity vertices
v₀ ∈ V that receive no incoming edges but provide constant or time-varying inputs
to the network. These represent external sensory input, bias terms, or background
activity independent of network computation."

"The dynamics decomposes into:
1. Endogenous activity: computed from network inputs via weighted sums
2. Exogenous activity: spontaneous inputs from v₀ vertices
3. Combined dynamics: superposition of both sources"

## DNN Interpretation

**Spontaneous Vertices**: External inputs and bias terms
- Input layer neurons (sensory data)
- Bias units (constant offset)
- Attention queries (external context)
- Noise injection (regularization)

**Applications**:
- Sensory-driven networks (vision, audio, language input)
- Bias terms in every layer
- Attention mechanisms (query from different modality)
- Variational autoencoders (noise input to decoder)
- Conditioning in generative models (class labels, text prompts)

## Key Concepts

1. **Augmented Graph G₀**: Original graph G + spontaneous vertices
2. **Partition**: V₀ (spontaneous) ∪ V₁ (computed)
3. **Dynamics**: h(t) = endogenous + exogenous components
4. **Conditioning**: Fixing spontaneous inputs conditions the dynamics
5. **Cofibration**: Inclusion V₀ ↪ V preserves structure

-}

module Neural.Stack.SpontaneousActivity where

open import 1Lab.Prelude hiding (id; _∘_)
open import 1Lab.Type.Sigma

open import Cat.Prelude
open import Cat.Functor.Base
open import Cat.Diagram.Pushout
open import Cat.Diagram.Coproduct
open import Cat.Instances.Functor

-- Import cat's manifolds from Section 3.1
open import Neural.Stack.CatsManifold

private variable
  o ℓ o' ℓ' : Level
  C D : Precategory o ℓ

--------------------------------------------------------------------------------
-- § 3.2.1: Augmented Graphs with Spontaneous Vertices

{-|
## Definition 3.7: Spontaneous Vertex

> "A spontaneous vertex v₀ ∈ V is a vertex with no incoming edges:
> source⁻¹(v₀) = ∅. It provides exogenous input to the network."

**Properties**:
- No dependencies: computed solely from external source
- Source of information: all paths start here (or at cycles)
- Bias interpretation: v₀ can be constant (time-independent)
- Input interpretation: v₀ can be time-varying (e.g., sensory stream)

**DNN Examples**:
- Input layer: all vertices are spontaneous (external data)
- Bias terms: one spontaneous vertex per layer (constant 1)
- Class conditioning: one spontaneous vertex per class (one-hot)
- Noise injection: spontaneous Gaussian noise vertices
-}

postulate
  DirectedGraph : Type (lsuc lzero)
  Vertices : DirectedGraph → Type
  Edges : DirectedGraph → Type
  source : ∀ (G : DirectedGraph) → Edges G → Vertices G
  target : ∀ (G : DirectedGraph) → Edges G → Vertices G

-- Predicate: vertex v has no incoming edges
is-spontaneous : (G : DirectedGraph) → Vertices G → Type
is-spontaneous G v = (e : Edges G) → target G e ≡ v → ⊥

{-|
## Definition 3.8: Augmented Graph with Spontaneous Partition

> "An augmented graph G₀ = (V₀ ⊎ V₁, E, s, t) consists of:
> - V₀: spontaneous vertices (no incoming edges)
> - V₁: computed vertices (may have incoming edges)
> - E: edges connecting V₀ → V₁ and V₁ → V₁ (but never → V₀)"

**Structure**:
```
V₀ (spontaneous)    V₁ (computed)
   v₀₁ ──────────→ v₁
   v₀₂ ──────────→ v₂ ──→ v₃
                   ↑       ↓
                   └───────┘
```

**Constraints**:
1. All v ∈ V₀ are spontaneous: source⁻¹(v) = ∅
2. Edges from V₀ go only to V₁
3. Edges within V₁ are arbitrary (including cycles)

**DNN Interpretation**:
- V₀ = {input vertices, bias vertices}
- V₁ = {all hidden and output vertices}
- This is the standard neural network structure!
-}

record AugmentedGraph : Type (lsuc lzero) where
  field
    base-graph : DirectedGraph

    -- Partition of vertices
    V₀ : Type  -- Spontaneous vertices
    V₁ : Type  -- Computed vertices
    partition : Vertices base-graph ≃ (V₀ ⊎ V₁)

    -- All V₀ vertices are spontaneous
    V₀-spontaneous : (v₀ : V₀) → is-spontaneous base-graph
                                (Equiv.from partition (inl v₀))

    -- No edges target V₀ (implicit in V₀-spontaneous)
    no-edges-to-V₀ : (e : Edges base-graph)
                   → {v₀ : V₀}
                   → target base-graph e ≡ Equiv.from partition (inl v₀)
                   → ⊥

{-|
## Example 3.6: Feedforward Network with Input Layer

Standard feedforward network:
- V₀ = {x₁, x₂, ..., xₙ} (input features)
- V₁ = {h₁, ..., hₘ, y₁, ..., yₖ} (hidden + output)
- Edges: x_i → h_j, h_j → h_k, h_m → y_l

All inputs are spontaneous (provided externally).
-}

postulate
  example-feedforward : AugmentedGraph
  -- V₀ = inputs, V₁ = hidden ⊎ outputs

{-|
## Example 3.7: Network with Bias Terms

Every layer has a bias vertex:
- V₀ = {x₁, ..., xₙ, b₁, b₂, ..., bₗ} (inputs + biases)
- V₁ = {hidden and output vertices}
- Bias vertices: constant spontaneous input (typically = 1)

This explains why bias terms don't need gradient computation:
they're spontaneous (fixed) inputs!
-}

postulate
  example-with-bias : AugmentedGraph
  -- V₀ = inputs ⊎ bias-vertices

--------------------------------------------------------------------------------
-- § 3.2.2: Dynamics Decomposition

{-|
## Definition 3.9: Endogenous vs Exogenous Activity

> "The state h_v(t) at vertex v ∈ V₁ decomposes as:
>   h_v(t) = h_v^endo(t) + h_v^exo(t)
> where:
> - h_v^endo(t) = Σ_{e: u→v, u∈V₁} w_e · h_u(t)  (from computed vertices)
> - h_v^exo(t) = Σ_{e: v₀→v, v₀∈V₀} w_e · h_{v₀}(t)  (from spontaneous vertices)"

**Interpretation**:
- Endogenous: internal computation (recurrent connections)
- Exogenous: external driving (inputs and biases)
- Total activity: linear superposition

**Continuous-Time Dynamics**:
```
dh_v/dt = -h_v + σ(h_v^endo + h_v^exo)
        = -h_v + σ(Σ_{u∈V₁} w_{u→v} h_u + Σ_{v₀∈V₀} w_{v₀→v} h_{v₀})
```

**DNN Interpretation**: Forward pass decomposition
- h_v^exo: bias term and direct inputs
- h_v^endo: weighted sum of previous layer
- Standard: h_v = σ(W·h_{l-1} + b) = σ(h^endo + h^exo)
-}

module _ (G : AugmentedGraph) where
  open AugmentedGraph G

  postulate
    -- Time type (ℝ or discrete steps)
    Time : Type

    -- State at vertex v and time t
    State : Vertices base-graph → Time → Type

    -- Weight on edge e
    Weight : Edges base-graph → Type

  -- Endogenous activity: sum over incoming edges from V₁
  postulate
    endogenous : (v : V₁) → Time → Type
    endo-def : ∀ v t → endogenous v t
                     ≡ {!!}  -- Σ_{e: u→v, u∈V₁} w_e · h_u(t)

  -- Exogenous activity: sum over incoming edges from V₀
  postulate
    exogenous : (v : V₁) → Time → Type
    exo-def : ∀ v t → exogenous v t
                    ≡ {!!}  -- Σ_{e: v₀→v, v₀∈V₀} w_e · h_{v₀}(t)

  -- Total activity decomposition
  postulate
    activity-decomposition : ∀ (v : V₁) (t : Time)
                           → State (Equiv.from partition (inr v)) t
                             ≡ {!!}  -- endogenous v t + exogenous v t

{-|
## Example 3.8: Single Neuron Decomposition

Consider a single neuron in a hidden layer:
- Inputs from previous layer: h^endo = Σᵢ wᵢ·hᵢ
- Bias term: h^exo = b (constant from bias vertex)
- Total: h = σ(h^endo + h^exo) = σ(Σᵢ wᵢ·hᵢ + b)

This is exactly the standard neuron equation!
The categorical framework makes the bias explicit as a spontaneous vertex.
-}

--------------------------------------------------------------------------------
-- § 3.2.3: Conditioning on Spontaneous Inputs

{-|
## Definition 3.10: Conditioned Dynamics

> "Fixing the spontaneous inputs h_{v₀}(t) = c_{v₀}(t) for all v₀ ∈ V₀
> conditions the dynamics on V₁. This is a restriction of the cat's manifold
> M: C^op → Man to the submanifold determined by the conditioning."

**Mathematical Structure**:
- Unconditioned: M(V) = {h: V → ℝⁿ}  (all possible states)
- Conditioned: M|_c(V) = {h: V → ℝⁿ | h|_{V₀} = c}  (fixed spontaneous states)
- This is the conditioning operation from Section 3.1!

**DNN Interpretation**: Fixing inputs
- Training: condition on data batch x ∈ V₀
- Inference: condition on query input
- Attention: condition on query vectors
- Conditional generation: condition on class labels or text

**Example: Conditional GAN**
- Generator G: noise z → image x
- Conditioned G_c: (noise z, class c) → image x of class c
- The class c is a spontaneous vertex with fixed value
- Different c values give different submanifolds
-}

module _ (G : AugmentedGraph) {d : Nat} (M : Cats-Manifold C d) where
  open AugmentedGraph G

  postulate
    -- Fix spontaneous vertices to specific values
    Conditioning : (V₀ → {!!}) → Type  -- Maps v₀ → value

    -- Conditioned cat's manifold (from Section 3.1)
    conditioned-manifold : Conditioning → Cats-Manifold C d

    -- Property: agrees with conditioning operation from Section 3.1
    conditioning-agrees : ∀ (c : Conditioning) (U : C .Precategory.Ob)
                        → {!!}  -- conditioned-manifold c ≡ condition M ... (from 3.1)

{-|
## Example 3.9: Attention as Conditioning

Multi-head attention can be viewed as:
1. Query Q: spontaneous vertex (from different modality/layer)
2. Key K, Value V: computed vertices (from input sequence)
3. Attention weights: α = softmax(QK^T)
4. Output: O = αV (conditioned on query Q)

Different queries Q condition to different subspaces of the value manifold.
This explains why attention is a geometric morphism (Section 2.4)!
-}

postulate
  example-attention-conditioning : ∀ {C : Precategory o ℓ} {d : Nat}
                                 → (M : Cats-Manifold C d)
                                 → {!!}
  -- Query vertices are spontaneous, conditioning determines attention output

--------------------------------------------------------------------------------
-- § 3.2.4: Cofibrations and Spontaneous Inclusions

{-|
## Definition 3.11: Spontaneous Inclusion as Cofibration

> "The inclusion i: V₀ ↪ V of spontaneous vertices is a cofibration in the
> category of directed graphs. This means it has good extension properties:
> dynamics on V₀ can be extended to V via pushout."

**Categorical Structure**:
```
V₀ ─i─→ V
│       │
f↓      ↓g  (pushout)
│       │
D ───→ W
```

**DNN Interpretation**: Modular network construction
- Start with inputs V₀
- Extend to full network V via cofibration
- Guarantees compositional structure
- Enables modular training (freeze inputs, train network)

**Connection to Section 3.3**:
Cofibrations of theories ↔ extensions of languages
Spontaneous vertices ↔ axioms (given truths)
-}

postulate
  -- Cofibration structure on graph category
  is-cofibration : ∀ {G H : DirectedGraph} → (G → H) → Type

  -- Inclusion of spontaneous vertices
  spontaneous-inclusion : (G : AugmentedGraph)
                        → {!!}  -- V₀ → Vertices base-graph

  -- This inclusion is a cofibration
  spontaneous-is-cofibration : ∀ (G : AugmentedGraph)
                             → is-cofibration (spontaneous-inclusion G)

  -- Pushout property for extending dynamics
  extend-dynamics : ∀ {G : AugmentedGraph} {D : DirectedGraph}
                  → {!!}  -- (V₀ → D) → (Vertices base-graph → pushout)

{-|
## Example 3.10: Transfer Learning as Cofibration Extension

Transfer learning from domain A to domain B:
1. Pretrained network f: V₀^A → V^A (source domain)
2. New inputs: V₀^B (target domain)
3. Extend via cofibration: V₀^B ⊔_{V₀^A} V^A (shared backbone, new inputs)
4. Pushout gives network for domain B using domain A features

The cofibration property guarantees the extension preserves structure.
-}

postulate
  example-transfer-learning : ∀ {G_A G_B : AugmentedGraph} → {!!}
  -- Pushout of spontaneous inclusions gives transfer learning architecture

--------------------------------------------------------------------------------
-- § 3.2.5: Spontaneous Dynamics and Ergodicity

{-|
## Proposition 3.2: Ergodicity with Spontaneous Input

> "Consider a network with recurrent dynamics on V₁ and constant spontaneous
> input from V₀. If the recurrent subnetwork is ergodic (irreducible + aperiodic),
> then the steady-state distribution is uniquely determined by the spontaneous
> input intensities."

**Proof Sketch**:
1. Recurrent network on V₁ has transition matrix P
2. Ergodicity: P has unique stationary distribution π
3. Spontaneous input shifts the distribution: π' = π + δ(V₀ inputs)
4. Uniqueness: π' is the unique fixed point of dynamics

**DNN Interpretation**: Stable attractors
- Recurrent network: settles to attractor
- Spontaneous input: determines which attractor
- Training: shape attractor landscape
- Inference: different inputs → different attractors

**Applications**:
- Hopfield networks: attractors = stored memories
- Reservoir computing: spontaneous input drives fixed recurrent network
- Dynamical systems: input-driven systems with stable limit cycles
-}

postulate
  proposition-3-2 : ∀ (G : AugmentedGraph)
                  → {!!}  -- Ergodicity conditions
                  → {!!}  -- Unique steady state determined by V₀

{-|
## Example 3.11: Reservoir Computing

Reservoir computing explicitly uses spontaneous inputs:
- Reservoir: large random recurrent network (V₁)
- Input: time-varying spontaneous signal (V₀)
- Readout: linear decoder from reservoir states
- Training: only train readout (reservoir is fixed)

This works because:
1. Ergodic reservoir explores high-dimensional state space
2. Spontaneous input drives exploration
3. Rich dynamics separate different input patterns
4. Linear readout sufficient for classification

The categorical framework explains why: the cofibration V₀ ↪ V
guarantees input-driven dynamics have good extension properties.
-}

postulate
  example-reservoir : AugmentedGraph
  -- Large random recurrent V₁, time-varying input V₀

--------------------------------------------------------------------------------
-- § 3.2.6: Noise Injection and Regularization

{-|
## Definition 3.12: Stochastic Spontaneous Vertices

> "A stochastic spontaneous vertex v₀^noise provides random input h_{v₀}(t) ~ p(h)
> drawn from a probability distribution p. This introduces controlled randomness
> into the dynamics."

**Applications**:
- Dropout: Bernoulli spontaneous vertices
- Gaussian noise injection: Gaussian spontaneous vertices
- Adversarial training: spontaneous adversarial perturbations
- Variational inference: spontaneous latent variables

**Cat's Manifold Interpretation**:
- Stochastic vertex → probability distribution on manifold
- Conditioning → fixing noise realization
- Expectation over noise → integral over conditioning submanifold
-}

postulate
  -- Probability distribution on manifold
  ProbDist : ∀ {d} → (Man d) .Precategory.Ob → Type

  -- Stochastic spontaneous vertex
  is-stochastic : (G : AugmentedGraph)
                → AugmentedGraph.V₀ G → Type

  -- Distribution for stochastic vertex
  vertex-distribution : ∀ {G : AugmentedGraph} {v₀ : AugmentedGraph.V₀ G}
                      → is-stochastic G v₀
                      → ProbDist {!!}  -- Distribution on state space

{-|
## Example 3.12: Variational Autoencoder

VAE has two types of spontaneous vertices:
1. Input x: observed data (deterministic spontaneous)
2. Noise ε ~ N(0,I): latent noise (stochastic spontaneous)

Encoder: x → (μ, σ)
Latent: z = μ + σ · ε (ε is spontaneous)
Decoder: z → x̂

The noise vertex ε enables:
- Sampling different z for same x
- Regularization (KL divergence term)
- Generative modeling (vary ε at inference)

Categorically: ε is a stochastic spontaneous vertex, conditioning on ε gives
deterministic decoding, marginalizing over ε gives generative distribution.
-}

postulate
  example-vae : AugmentedGraph
  -- V₀ = {input x, noise ε}, where ε is stochastic

--------------------------------------------------------------------------------
-- § 3.2.7: Time-Varying Spontaneous Inputs

{-|
## Definition 3.13: Temporal Spontaneous Dynamics

> "A temporal spontaneous vertex v₀(t) has time-varying input h_{v₀}(t) that
> evolves according to an external process (e.g., sensory stream, clock signal,
> scheduled annealing)."

**Types of Temporal Patterns**:
1. **Periodic**: h_{v₀}(t + T) = h_{v₀}(t) (oscillatory input)
2. **Decaying**: h_{v₀}(t) → 0 as t → ∞ (annealing)
3. **Growing**: h_{v₀}(t) increases with t (curriculum)
4. **Stochastic process**: h_{v₀}(t) ~ process (Brownian motion, Poisson)

**DNN Applications**:
- Learning rate schedules: spontaneous v₀(t) = η(t)
- Curriculum learning: gradually introduce harder examples
- Annealing: reduce temperature over time
- Temporal attention: time-dependent queries
-}

module _ (G : AugmentedGraph) where
  open AugmentedGraph G

  postulate
    -- Time-dependent spontaneous input
    temporal-spontaneous : (v₀ : V₀) → (Time → {!!})

    -- Periodic spontaneous input
    is-periodic : (v₀ : V₀) → {!!} → Type
    periodicity-condition : ∀ {v₀ T} → is-periodic v₀ T
                          → ∀ t → temporal-spontaneous v₀ (t + T)
                                  ≡ temporal-spontaneous v₀ t

    -- Decaying spontaneous input
    is-decaying : (v₀ : V₀) → Type
    decay-condition : ∀ {v₀} → is-decaying v₀
                    → {!!}  -- lim_{t→∞} temporal-spontaneous v₀ t = 0

{-|
## Example 3.13: Learning Rate as Spontaneous Vertex

Consider the learning rate η as a spontaneous vertex:
- η(t) = η₀ · exp(-λt) (exponential decay)
- η(t) = η₀ / (1 + t/T) (inverse time decay)
- η(t) = η₀ · cos²(πt/T) (cosine annealing)

Update rule: θ(t+1) = θ(t) - η(t) · ∇L(θ(t))

Categorically:
- η is a temporal spontaneous vertex
- Gradient descent dynamics depend on η(t)
- Different schedules = different temporal patterns
- Optimal schedule = optimal spontaneous dynamics
-}

postulate
  example-learning-rate : AugmentedGraph
  -- V₀ contains η(t) as temporal spontaneous vertex

--------------------------------------------------------------------------------
-- § 3.2.7b: Explicit Dynamics Formula and H^0 Connection (Equation 3.5)

module _ (G : AugmentedGraph) where
  open AugmentedGraph G

  {-|
  ## Equation 3.5: Explicit Dynamics with Feed-Forward and Feedback

  > "The explicit dynamics at vertex v ∈ V₁ is given by:
  >   dh_v/dt = -h_v + σ(Σ_{u→v} w_{u,v} h_u)
  >           = -h_v + σ(h_v^{ff} + h_v^{fb})
  > where:
  >   h_v^{ff} = Σ_{u∈V_0∪V_{<v}} w_{u,v} h_u   (feed-forward from V₀ and earlier layers)
  >   h_v^{fb} = Σ_{u∈V_{≥v}} w_{u,v} h_u        (feedback from same/later layers)"

  **Decomposition**:
  The dynamics naturally splits into three components:

  1. **Decay term**: -h_v
     - Represents leakage or forgetting
     - Time constant τ = 1 (can be scaled)
     - Ensures stability when isolated

  2. **Feed-forward term**: h_v^{ff} from V₀ ∪ V_{<v}
     - Spontaneous inputs from V₀ (external/bias)
     - Computed inputs from earlier layers V_{<v}
     - Forms a DAG (directed acyclic graph)
     - Can be computed in topological order

  3. **Feedback term**: h_v^{fb} from V_{≥v}
     - Recurrent connections (same layer)
     - Skip connections (later layers)
     - Forms potential cycles
     - Requires fixed-point iteration or dynamics

  **Mathematical Form** (Equation 3.5):
  ```
  dh_v/dt = -h_v + σ(Σ_{u→v, u∈V₀} w_{u,v} h_u
                   + Σ_{u→v, u∈V_{<v}} w_{u,v} h_u
                   + Σ_{u→v, u∈V_{≥v}} w_{u,v} h_u)
          = -h_v + σ(h_v^{V₀} + h_v^{ff} + h_v^{fb})
  ```

  **DNN Interpretation**:

  **Feed-forward networks** (h_v^{fb} = 0):
  - Only spontaneous and earlier layers contribute
  - Can compute in single forward pass
  - Steady state: h_v = σ(h_v^{V₀} + h_v^{ff})
  - Example: Standard MLPs, CNNs without residual

  **Recurrent networks** (h_v^{fb} ≠ 0):
  - Includes same-layer or later-layer feedback
  - Requires temporal dynamics (RNN) or fixed-point (DEQ)
  - Steady state: h_v = σ(h_v^{V₀} + h_v^{ff} + h_v^{fb})
  - Example: RNNs, LSTMs, Transformers (self-attention), ResNets

  **Residual Networks**:
  h_{l+1} = h_l + f_l(h_l) corresponds to:
  - Feed-forward: f_l(h_l) computed from h_l
  - Feedback: identity skip connection h_l
  - Discretized dynamics with dt = 1
  -}

  postulate
    -- Time derivative of state
    dh/dt : (v : V₁) → Time → Type

    -- Activation function
    σ : Type → Type

    -- Vertices ordered by topological sort
    _<_ : V₁ → V₁ → Type  -- u < v if u is "before" v in some ordering

    -- Feed-forward component from V₀ and earlier layers
    h^{ff} : (v : V₁) → Time → Type
    h^{ff}-def : ∀ v t → h^{ff} v t
                       ≡ {!!}  -- Σ_{u→v, u∈V₀∪V_{<v}} w_{u,v} h_u(t)

    -- Feedback component from same/later layers
    h^{fb} : (v : V₁) → Time → Type
    h^{fb}-def : ∀ v t → h^{fb} v t
                       ≡ {!!}  -- Σ_{u→v, u∈V_{≥v}} w_{u,v} h_u(t)

    -- Equation 3.5: Explicit dynamics formula
    dynamics-formula : ∀ (v : V₁) (t : Time)
                     → dh/dt v t ≡ {!!}  -- -h_v(t) + σ(h^{ff} v t + h^{fb} v t)

  {-|
  ## Connection to H^0 Cohomology (Equation 3.4 from Section 3.1)

  > "The H^0 cohomology H^0(A'_strict; M) computes the output-relevant dynamics,
  >  which for spontaneous vertices corresponds to feed-forward flow:
  >    H^0 = M(P_out) = flow from V₀ → ... → V_out (feed-forward only)
  >  Feedback loops h^{fb} do NOT contribute to H^0."

  **Why This Works**:

  1. **H^0 = degree-0 cohomology** (from Section 3.4):
     - Cochains ψ: Θ_λ → K with δψ = 0
     - Constant functions over connected components
     - These are determined by output propositions P_out

  2. **Feed-forward propagates to output**:
     - V₀ → V₁ → ... → V_out forms DAG
     - Information flows acyclically to output
     - All feed-forward paths contribute to M(P_out)

  3. **Feedback does NOT reach output** (in steady state):
     - Cycles V_i ↔ V_j stay within subnetwork
     - Unless V_i has path to output
     - But then that path is feed-forward from V_i!

  4. **Cohomology captures output-relevant structure**:
     - H^0(A'_strict) = functions constant on fibers over P_out
     - These are exactly feed-forward contributions
     - Feedback creates internal cycles that don't change output

  **Mathematical Statement**:
  Let h^∞_v be the steady-state solution of Equation 3.5.
  Then:
    H^0(A'_strict; M)(V₀) = {h^∞|_{V_out} : depends only on h_v^{ff}, not h_v^{fb}}

  In other words: H^0 cohomology computes the feed-forward flow!

  **DNN Interpretation**:

  **Why feed-forward networks learn well**:
  - All dynamics contribute to H^0 (no feedback to discard)
  - Backpropagation computes ∂H^0/∂w (via Equation 3.3 RKan formula)
  - Gradient descent directly optimizes output-relevant features
  - No wasted capacity on internal feedback loops

  **Why recurrent networks are harder**:
  - Feedback h^{fb} doesn't contribute to H^0 directly
  - Must learn dynamics that stabilize AND produce correct h^{ff}
  - Vanishing gradients (long-term feedback dependencies)
  - Need to balance internal dynamics vs output prediction

  **ResNets as compromise**:
  - Skip connections = controlled feedback
  - Gradients flow through skips (feed-forward path preserved)
  - Residual blocks add refinements to feed-forward flow
  - H^0 includes skip paths (they reach output!)
  -}

  postulate
    -- Steady-state solution
    h^∞ : (v : V₁) → Type
    h^∞-steady-state : ∀ v → dh/dt v {!!} ≡ {!!}  -- 0 at steady state

    -- H^0 cohomology (from Section 3.1 and 3.4)
    H0-spontaneous : (M : Cats-Manifold C d) → Type

    -- H^0 equals feed-forward flow to output
    H0-equals-feedforward : ∀ {d} (M : Cats-Manifold C d)
                          → H0-spontaneous M ≃ {!!}  -- {h^∞|_{V_out} from h^{ff} only}

    -- Feed-forward networks have full H^0 (all dynamics contribute)
    feedforward-full-H0 : (acyclic : ∀ v → h^{fb} v {!!} ≡ {!!})  -- h^{fb} = 0
                        → ∀ {d} (M : Cats-Manifold C d)
                        → H0-spontaneous M ≃ {!!}  -- Full output dynamics

  {-|
  ## Summary: Equation 3.5 and H^0

  **Equation 3.5** gives the explicit dynamics:
  - Decomposition into feed-forward h^{ff} and feedback h^{fb}
  - Standard neuron dynamics dh/dt = -h + σ(input)
  - Connects to all DNN architectures (FF, RNN, ResNet, etc.)

  **H^0 Connection** explains:
  - Why cohomology captures output-relevant information
  - Why feed-forward networks are "easier" to train
  - How backpropagation computes RKan (via chain rule on feed-forward paths)
  - Why skip connections help (maintain feed-forward gradient flow)

  **Unified Framework**:
  - **Section 3.1**: M(P_out) = RKan_ι(X_+) (geometric)
  - **Section 3.2**: M(P_out) = feed-forward flow from V₀ (dynamic)
  - **Section 3.4**: H^0(A'_strict) = degree-0 cohomology (algebraic)

  All three perspectives describe the same mathematical structure:
  *information that flows from inputs to outputs*.
  -}

--------------------------------------------------------------------------------
-- § 3.2.8: Summary and Connections

{-|
## Summary: Spontaneous Activity Framework

We have formalized:
1. **Augmented graphs G₀**: Networks with spontaneous vertices V₀
2. **Dynamics decomposition**: h = h^endo + h^exo
3. **Conditioning**: Fixing spontaneous inputs restricts dynamics
4. **Cofibrations**: V₀ ↪ V has extension properties
5. **Stochastic vertices**: Probability distributions on spontaneous inputs
6. **Temporal dynamics**: Time-varying spontaneous signals

## Connections to Other Sections

**Section 2 (Stacks)**:
- Spontaneous vertices ↔ base objects in fibration
- Exogenous input ↔ pull back from base
- Conditioning ↔ restriction to fiber

**Section 3.1 (Cat's Manifolds)**:
- Spontaneous inputs ↔ boundary conditions on manifolds
- Conditioning ↔ submanifold restriction
- Cofibration ↔ manifold with boundary

**Section 3.3 (Languages)**:
- Spontaneous vertices ↔ axioms (given propositions)
- Computed vertices ↔ derived propositions
- Cofibration ↔ theory extension

**Section 3.4 (Homology)**:
- Spontaneous vertices ↔ 0-chains (generators)
- Exogenous flow ↔ boundary operator
- Conditioning ↔ relative homology

## Applications Enabled

1. **Modular Networks**: Cofibration structure enables composition
2. **Input-Driven Dynamics**: Spontaneous forcing of recurrent systems
3. **Regularization**: Stochastic spontaneous vertices
4. **Curriculum Learning**: Temporal spontaneous schedules
5. **Conditional Generation**: Conditioning on class/text spontaneous vertices
6. **Transfer Learning**: Extension via pushout of cofibrations
-}
