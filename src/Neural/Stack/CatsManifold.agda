{-# OPTIONS --allow-unsolved-metas #-}

{-|
# Section 3.1: Cat's Manifolds and Conditioning

This module implements Section 3.1 from Belfiore & Bennequin (2022), formalizing
cat's manifolds (categories of manifolds) and conditioning operations for neural
dynamics.

## Paper Reference

"A cat's manifold over a category C is a functor M: C^op → Man, where Man is
the category of smooth manifolds. This provides a sheaf-theoretic framework for
neural state spaces that vary over network architecture."

"Conditioning operations are modeled as restrictions to submanifolds, implemented
via limits and Kan extensions in the topos of presheaves."

## DNN Interpretation

**Cat's Manifolds**: Neural state spaces that smoothly vary across network layers
- Each layer U has a state manifold M(U)
- Layer transitions induce smooth maps between manifolds
- Constraints and conditioning restrict to submanifolds

**Applications**:
- Continuous-time dynamics (ODEs on manifolds)
- Latent variable models (VAEs, diffusion models)
- Geometric deep learning (manifold-valued features)
- Constrained optimization (submanifold projection)

## Key Concepts

1. **Manifolds Category**: Smooth manifolds with smooth maps
2. **Presheaf of Manifolds**: M: C^op → Man (cat's manifold)
3. **State Spaces**: M(U) is the state space at layer U
4. **Conditioning**: Restriction to submanifolds via pullback
5. **Kan Extensions**: Universal constructions for extending/restricting dynamics

-}

module Neural.Stack.CatsManifold where

open import 1Lab.Prelude hiding (id; _∘_)
open import 1Lab.Type.Sigma

open import Cat.Prelude
open import Cat.Functor.Base
open import Cat.Functor.Kan.Base
open import Cat.Instances.Functor
open import Cat.Diagram.Limit.Base
open import Cat.Diagram.Pullback

private variable
  o ℓ o' ℓ' : Level
  C D E : Precategory o ℓ

--------------------------------------------------------------------------------
-- § 3.1.1: Smooth Manifolds Category

{-|
## Definition: Smooth Manifolds

We postulate the category Man of smooth manifolds and smooth maps.
In a full formalization, this would require:
- Topological spaces with atlas structure
- Smooth compatibility of charts
- Smooth maps between manifolds

**DNN Interpretation**: State spaces for continuous-time neural dynamics
- Manifold = continuous state space (e.g., hidden representations)
- Smooth maps = differentiable transformations
- Submanifolds = constrained states (e.g., unit sphere, simplex)
-}

postulate
  Man : (d : Nat) → Precategory (lsuc lzero) lzero
  -- Man d has d-dimensional smooth manifolds as objects

  -- Euclidean space ℝ^n as a manifold
  ℝⁿ : (n : Nat) → (Man n) .Precategory.Ob

  -- Products of manifolds
  _×ᴹ_ : ∀ {d₁ d₂} → (Man d₁) .Precategory.Ob → (Man d₂) .Precategory.Ob
       → (Man (d₁ + d₂)) .Precategory.Ob

  -- Submanifolds (embeddings)
  is-submanifold : ∀ {d d'} {M : (Man d) .Precategory.Ob} {N : (Man d') .Precategory.Ob}
                 → (Man d') .Precategory.Hom N M → Type

{-|
## Example: Common Neural State Manifolds

**Euclidean space ℝⁿ**: Unrestricted hidden states
- Most common choice for neural networks
- Linear structure allows gradient descent

**Unit sphere Sⁿ⁻¹**: Normalized representations
- Cosine similarity networks
- Hyperspherical prototypes
- Angular metrics

**Probability simplex Δⁿ**: Categorical distributions
- Softmax outputs
- Mixture weights
- Attention distributions

**Grassmannian Gr(k,n)**: k-dimensional subspaces of ℝⁿ
- Subspace representations
- Low-rank approximations
- Principal component analysis
-}

postulate
  -- Unit sphere in ℝⁿ
  Sⁿ⁻¹ : (n : Nat) → (Man n) .Precategory.Ob
  sphere-embedding : ∀ {n} → (Man n) .Precategory.Hom (Sⁿ⁻¹ n) (ℝⁿ n)
  sphere-is-submanifold : ∀ {n} → is-submanifold (sphere-embedding {n})

  -- Probability simplex
  Δⁿ : (n : Nat) → (Man n) .Precategory.Ob
  simplex-embedding : ∀ {n} → (Man n) .Precategory.Hom (Δⁿ n) (ℝⁿ n)
  simplex-is-submanifold : ∀ {n} → is-submanifold (simplex-embedding {n})

--------------------------------------------------------------------------------
-- § 3.1.2: Cat's Manifolds (Presheaves of Manifolds)

{-|
## Definition 3.1: Cat's Manifold

> "A cat's manifold over C is a contravariant functor M: C^op → Man_d for some
> dimension d. This assigns to each object U ∈ C a d-dimensional manifold M(U),
> and to each morphism α: U → U' a smooth map M(α): M(U') → M(U)."

**Components**:
- M₀: C-Ob → Man d .Ob (state space at each layer)
- M₁: C-Hom U U' → Man d .Hom (M₀ U') (M₀ U) (transition maps)
- Functoriality: M(id) = id, M(α ∘ β) = M(β) ∘ M(α)

**DNN Interpretation**:
A neural network with smoothly varying state spaces:
- Each layer U has state manifold M(U)
- Forward connections α induce pullback maps M(α)
- Backpropagation uses pushforward (dual operation)

**Example: Latent Variable Model**
- Encoder layers: ℝⁿ → ℝᵐ → S^(k-1) (to unit sphere)
- Latent space: probability simplex Δᵏ
- Decoder layers: Δᵏ → ℝᵐ → ℝⁿ
-}

Cats-Manifold : (C : Precategory o ℓ) (d : Nat) → Type (o ⊔ ℓ ⊔ lsuc lzero)
Cats-Manifold C d = Functor (C ^op) (Man d)

module _ {C : Precategory o ℓ} {d : Nat} (M : Cats-Manifold C d) where
  open Functor M

  -- State manifold at object U
  State-Space : C .Precategory.Ob → (Man d) .Precategory.Ob
  State-Space = F₀

  -- Transition map for morphism α: U → U'
  Transition-Map : ∀ {U U'} → C .Precategory.Hom U U'
                 → (Man d) .Precategory.Hom (State-Space U') (State-Space U)
  Transition-Map = F₁

{-|
## Example 3.1: Feedforward Network with Normalized Layers

A 3-layer feedforward network with:
- Input layer: ℝ¹⁰⁰
- Hidden layer 1: ℝ⁵⁰ (before normalization) → S⁴⁹ (after batch norm)
- Hidden layer 2: ℝ²⁰ (before normalization) → S¹⁹ (after layer norm)
- Output layer: Δ¹⁰ (probability simplex from softmax)

The cat's manifold tracks the normalized manifolds:
M(input) = ℝ¹⁰⁰, M(hidden1) = S⁴⁹, M(hidden2) = S¹⁹, M(output) = Δ¹⁰
-}

postulate
  -- Example: Layer category for 3-layer network
  example-network : Precategory lzero lzero
  input hidden1 hidden2 output : example-network .Precategory.Ob
  conn1 : example-network .Precategory.Hom input hidden1
  conn2 : example-network .Precategory.Hom hidden1 hidden2
  conn3 : example-network .Precategory.Hom hidden2 output

  -- Cat's manifold assigning state spaces
  example-manifold : Cats-Manifold example-network 100  -- max dimension
  -- M(input) = ℝ¹⁰⁰, M(hidden1) = S⁴⁹, M(hidden2) = S¹⁹, M(output) = Δ¹⁰

--------------------------------------------------------------------------------
-- § 3.1.3: Conditioning via Limits

{-|
## Definition 3.2: Conditioning as Pullback

> "Given a cat's manifold M: C^op → Man and a constraint submanifold N ⊂ M(U),
> conditioning on N is the restriction M|_N obtained by pulling back along the
> embedding i: N ↪ M(U)."

**Construction**: Pullback in presheaf category
```
M|_N(V) -----> M(V)
   |             |
   |             | M(α)
   ↓             ↓
   N ---------> M(U)
      i: N↪M(U)
```

**DNN Interpretation**: Constrained neural dynamics
- Original state space: M(U) (e.g., ℝⁿ)
- Constraint: N ⊂ M(U) (e.g., unit sphere Sⁿ⁻¹)
- Conditioned dynamics: restricted to N

**Applications**:
- Weight normalization: restrict to sphere
- Probability constraints: restrict to simplex
- Energy constraints: restrict to level set
- Attention masking: restrict to valid positions
-}

module _ {C : Precategory o ℓ} {d : Nat} (M : Cats-Manifold C d) where

  {-|
  Conditioning at object U ∈ C by submanifold N ⊂ M(U)
  Returns the restricted cat's manifold M|_N
  -}
  postulate
    condition : (U : C .Precategory.Ob)
              → (N : (Man d) .Precategory.Ob)
              → (i : (Man d) .Precategory.Hom N (State-Space M U))
              → is-submanifold i
              → Cats-Manifold C d

    -- Property: M|_N(U) = N
    condition-at-U : ∀ {U N i h}
                   → State-Space (condition U N i h) U ≡ N

    -- Property: For V ≠ U with no path to U, M|_N(V) = M(V)
    condition-preserves-unrelated : ∀ {U V N i h}
                                  → {!!}  -- No morphisms U ← ... ← V
                                  → State-Space (condition U N i h) V
                                    ≡ State-Space M V

{-|
## Example 3.2: Conditioning Hidden Layer to Unit Sphere

Given a network M with M(hidden) = ℝ⁵⁰, we can condition to the unit sphere:
- N = S⁴⁹ ⊂ ℝ⁵⁰
- M|_{S⁴⁹}(hidden) = S⁴⁹
- Forces all hidden states to lie on sphere
- Corresponds to weight normalization

This affects all downstream layers via pullback.
-}

postulate
  example-condition : Cats-Manifold example-network 100
  -- example-condition = condition example-manifold hidden1 (Sⁿ⁻¹ 49) sphere-embedding sphere-is-submanifold

--------------------------------------------------------------------------------
-- § 3.1.4: Kan Extensions for Dynamics

{-|
## Definition 3.3: Left Kan Extension for Spontaneous Dynamics

> "Given a cat's manifold M: C^op → Man and a functor F: C → D (network
> architecture change), the left Kan extension Lan_F M: D^op → Man extends the
> dynamics to the new architecture."

**Mathematical Structure**:
- Original network: C with dynamics M: C^op → Man
- New network: D with F: C → D (architecture morphism)
- Extended dynamics: Lan_F M: D^op → Man

**Universal Property**:
For any N: D^op → Man with a transformation N ∘ F^op ⇒ M,
there exists a unique transformation N ⇒ Lan_F M factoring through it.

**DNN Interpretation**: Architecture adaptation
- Add spontaneous activity vertices (Section 3.2)
- Add recurrent connections
- Add skip connections (ResNet-style)
- Transfer learning to larger network

**Example: Adding Skip Connections**
- Original: C = linear chain (feedforward)
- Extended: D = C + skip edges (ResNet)
- Lan extension: extends dynamics to skip paths
-}

module _ {C : Precategory o ℓ} {D : Precategory o' ℓ'} {d : Nat}
         (M : Cats-Manifold C d) (F : Functor C D) where

  open Functor

  {-|
  Left Kan extension of cat's manifold along functor F
  Extends dynamics from architecture C to architecture D
  -}
  postulate
    Lan-Manifold : Cats-Manifold D d

    -- Universal natural transformation
    Lan-unit : {!!}  -- M ⇒ Lan-Manifold ∘ F^op

    -- Universal property
    Lan-universal : ∀ (N : Cats-Manifold D d)
                  → {!!}  -- (N ∘ F^op ⇒ M) ≃ (N ⇒ Lan-Manifold)

{-|
## Definition 3.4: Right Kan Extension for Restriction

> "The right Kan extension Ran_F M: D^op → Man restricts dynamics to a
> sub-architecture, preserving all existing structure."

**DNN Interpretation**: Network pruning and restriction
- Remove layers (network compression)
- Freeze layers (transfer learning)
- Extract sub-network (modular analysis)

**Example: Layer Freezing**
- Full network: D with dynamics N: D^op → Man
- Trainable subset: C ⊂ D
- Restricted dynamics: Ran_F N preserves frozen layers
-}

  postulate
    Ran-Manifold : Cats-Manifold D d

    -- Co-universal natural transformation
    Ran-counit : {!!}  -- Ran-Manifold ∘ F^op ⇒ M

    -- Universal property
    Ran-universal : ∀ (N : Cats-Manifold D d)
                  → {!!}  -- (M ⇒ N ∘ F^op) ≃ (Ran-Manifold ⇒ N)

--------------------------------------------------------------------------------
-- § 3.1.5: Limits and Colimits in Cat's Manifolds

{-|
## Proposition 3.1: Limits in Presheaf Category

> "The category [C^op, Man] of cat's manifolds has all limits, computed
> pointwise. Specifically, if {M_i} is a diagram of cat's manifolds, then
> (lim M_i)(U) = lim(M_i(U)) in Man."

**Proof Sketch**:
Presheaf categories inherit (co)limits from the target category.
Since Man has limits (products, equalizers), [C^op, Man] does too.

**DNN Interpretation**: Multi-constraint dynamics
- Multiple constraints on state space
- Intersection of submanifolds
- Shared structure across architectures

**Example: Multi-Constraint Hidden Layer**
Constrain hidden states to:
1. Unit sphere (normalization)
2. Orthogonality constraints (decorrelation)
3. Sparsity constraints (non-negative orthant)

The limit enforces ALL constraints simultaneously.
-}

postulate
  proposition-3-1 : ∀ {C : Precategory o ℓ} {d : Nat}
                  → {!!}  -- Has-limits [C^op, Man d]

{-|
## Example 3.3: Multi-Constraint Optimization

Consider optimizing a hidden layer h ∈ ℝⁿ under multiple constraints:
1. Normalization: h ∈ Sⁿ⁻¹ (unit sphere)
2. Non-negativity: h ∈ ℝⁿ₊ (positive orthant)
3. Sum constraint: ∑hᵢ = 1 (probability simplex)

The limit of these three submanifolds is the probability simplex Δⁿ⁻¹.
This is exactly what softmax produces!

**Categorical Insight**: Softmax is the terminal object (limit) in the category
of normalized, non-negative, sum-to-one constraint manifolds.
-}

postulate
  example-softmax-as-limit : ∀ {n : Nat} → {!!}
  -- Δⁿ = lim (Sⁿ⁻¹, ℝⁿ₊, {∑xᵢ=1})

--------------------------------------------------------------------------------
-- § 3.1.6: Manifold-Valued Features

{-|
## Definition 3.5: Fibered Cat's Manifolds

> "A fibered cat's manifold is a stack F: C^op → Cat together with a natural
> transformation dim: F ⇒ Const(Nat) assigning dimension to each fiber, such
> that F(U)(ξ) is a dim(U,ξ)-dimensional manifold."

**Structure**:
- Stack F: C^op → Cat (from Section 2)
- For each U ∈ C and ξ ∈ F(U): a manifold F(U)(ξ)
- Dimension function dim: F(U)(ξ) → Nat
- Smooth dependence on ξ (fibration structure)

**DNN Interpretation**: Features with manifold structure
- Each feature ξ has an associated manifold F(U)(ξ)
- Example: Image features → local patch manifold
- Example: Graph node features → tangent space at node
- Example: Attention heads → probability simplex per head

**Applications**:
- Geometric deep learning (manifold-valued features)
- Equivariant networks (features on homogeneous spaces)
- Gauge-equivariant networks (vector bundles over base manifold)
-}

record Fibered-Cats-Manifold (C : Precategory o ℓ) (o' ℓ' : Level)
                              : Type (o ⊔ ℓ ⊔ lsuc o' ⊔ lsuc ℓ') where
  field
    -- Base stack
    base-stack : Functor (C ^op) (Cat o' ℓ')

    -- Dimension assignment
    dimension : ∀ (U : C .Precategory.Ob)
              → (base-stack .Functor.F₀ U) .Precategory.Ob → Nat

    -- Manifold at each fiber
    fiber-manifold : ∀ (U : C .Precategory.Ob)
                   → (ξ : (base-stack .Functor.F₀ U) .Precategory.Ob)
                   → (Man (dimension U ξ)) .Precategory.Ob

    -- Transition maps respect fiber structure
    fiber-transition : ∀ {U U' : C .Precategory.Ob} (α : C .Precategory.Hom U U')
                     → (ξ' : (base-stack .Functor.F₀ U') .Precategory.Ob)
                     → {!!}  -- Smooth map between fiber manifolds

{-|
## Example 3.4: Attention with Manifold-Valued Queries/Keys

Multi-head attention where each head has:
- Query space: Tangent space T_q(M) at point q ∈ M
- Key space: Tangent space T_k(M) at point k ∈ M
- Value space: Manifold M itself

This generalizes standard attention from Euclidean to Riemannian manifolds.
Used in geometric transformers and graph neural networks.
-}

postulate
  example-manifold-attention : ∀ {C : Precategory o ℓ} {d : Nat}
                             → Fibered-Cats-Manifold C o ℓ
  -- Query/Key: tangent spaces, Value: base manifold

--------------------------------------------------------------------------------
-- § 3.1.7: Smooth Dynamics and Vector Fields

{-|
## Definition 3.6: Vector Fields on Cat's Manifolds

> "A vector field on a cat's manifold M: C^op → Man assigns to each U ∈ C
> a vector field V(U) on the manifold M(U), compatibly with transition maps."

**Mathematical Structure**:
- For each U: V(U) ∈ Γ(TM(U)) (sections of tangent bundle)
- Compatibility: (M(α))_* V(U') = V(U) for α: U → U'
- Defines continuous-time dynamics: dh/dt = V(U)(h)

**DNN Interpretation**: Continuous-time neural ODEs
- Neural ODE: dh/dt = f(h, t, θ)
- Vector field: V(U)(h) = f_U(h, θ_U)
- Layer transitions: pushforward of vector fields
- Residual connections: V(U) = id + perturbation

**Applications**:
- Neural ODEs (continuous-depth networks)
- Normalizing flows (invertible transformations)
- Hamiltonian neural networks (energy-preserving dynamics)
- Gradient flows (optimization on manifolds)
-}

postulate
  -- Tangent bundle over manifold M
  TM : ∀ {d} → (Man d) .Precategory.Ob → (Man (d + d)) .Precategory.Ob

  -- Vector field = section of tangent bundle
  Vector-Field : ∀ {d} (M : (Man d) .Precategory.Ob) → Type

  -- Pushforward of vector field along smooth map
  pushforward : ∀ {d d'} {M : (Man d) .Precategory.Ob} {N : (Man d') .Precategory.Ob}
              → (Man d) .Precategory.Hom M N
              → Vector-Field M → Vector-Field N

record Cats-Vector-Field {C : Precategory o ℓ} {d : Nat}
                          (M : Cats-Manifold C d) : Type (o ⊔ ℓ ⊔ lsuc lzero) where
  field
    -- Vector field at each object
    field-at : (U : C .Precategory.Ob) → Vector-Field (State-Space M U)

    -- Compatibility with transitions
    field-compatible : ∀ {U U'} (α : C .Precategory.Hom U U')
                     → pushforward (Transition-Map M α) (field-at U')
                       ≡ field-at U

{-|
## Example 3.5: Residual Network as Vector Field

A residual block h_{n+1} = h_n + f(h_n) can be viewed as:
- Euler discretization of ODE: dh/dt = f(h)
- Vector field: V(h) = f(h)
- Continuous limit: infinitely many residual blocks

**Categorical Interpretation**:
- ResNet: discrete-time approximation
- Neural ODE: continuous-time limit
- Cat's vector field: functorial dynamics across layers
-}

postulate
  example-resnet-vector-field : ∀ {C : Precategory o ℓ} {d : Nat}
                              → (M : Cats-Manifold C d)
                              → Cats-Vector-Field M
  -- V(U)(h) = residual function at layer U

--------------------------------------------------------------------------------
-- § 3.1.8: Summary and Connections

{-|
## Summary: Cat's Manifolds Framework

We have formalized:
1. **Smooth manifolds category Man**: State spaces for neural dynamics
2. **Cat's manifolds M: C^op → Man**: Functorial state spaces over network
3. **Conditioning via limits**: Restricting to constraint submanifolds
4. **Kan extensions**: Architecture adaptation (Lan) and restriction (Ran)
5. **Fibered manifolds**: Features with intrinsic geometric structure
6. **Vector fields**: Continuous-time dynamics on manifolds

## Connections to Other Sections

**Section 2 (Stacks and Fibrations)**:
- Cat's manifolds are stacks with smooth structure
- Subobject classifier Ω_F classifies constraint submanifolds
- Geometric functors preserve smooth structure

**Section 3.2 (Spontaneous Activity)**:
- Dynamics with external inputs modeled as vector fields
- Spontaneous vertices add constant vector field components
- Conditioning handles input constraints

**Section 3.3 (Languages)**:
- Deduction systems as dynamics on cat's manifolds
- Proof search as geodesic flow
- Type checking as constraint satisfaction

**Section 3.4 (Semantic Information)**:
- Homology of manifold-valued features
- Persistent homology for topological features
- Information geometry on probability manifolds

## Applications Enabled

1. **Riemannian Neural Networks**: Layers as Riemannian manifolds with metric
2. **Constrained Optimization**: Gradient descent on submanifolds
3. **Normalizing Flows**: Diffeomorphisms between manifolds
4. **Geometric Deep Learning**: Equivariance via manifold symmetries
5. **Neural ODEs**: Continuous-time dynamics with guarantees
6. **Information Geometry**: Natural gradients and Fisher metric
-}
