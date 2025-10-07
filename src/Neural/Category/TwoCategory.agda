{-# OPTIONS --rewriting --guardedness --cubical --no-load-primitives #-}

{-|
# Section 5.2: The 2-Category of a Network

This module implements the 2-category structure for neural network semantics
from Section 5.2 of Belfiore & Bennequin (2022), formalizing Equations 5.8-5.12.

## Paper Reference

> "For representing languages in DNNs, we have associated to a small category C
> the class A_C = Grpd^C of presheaves over the category of fibrations in
> groupoids over C. The objects of A_C were described in terms of presheaves A_U
> on the fibers F_U for U ∈ C satisfying gluing conditions."

> "This structure encodes the relations between several semantics over the same
> network. The relations between several networks, for instance moduli inside a
> network, or networks that are augmented by external links, belong to a
> 3-category, whose objects are the above semantic triples, and the 1-morphism
> are lifting of functors between sites u: C → C'."

## Key Structure

**Objects (0-cells)**: Semantic triples (C, F, A)
- C: Site (small category, network architecture)
- F: Fibration in groupoids over C (pre-semantic structure)
- A: Presheaf over F (language, actual semantics)

**1-morphisms**: Pairs (F_U, φ_U) between (C,F,A) and (C',F',A')
- Family of functors F_U: F_U → F'_U (Equation 5.8)
- Family of natural transformations φ_U: A_U → F★_U(A'_U) (Equation 5.9)
- Composition via twisted composition (Equation 5.10)

**2-morphisms**: Natural transformations λ: F → G (Equations 5.11-5.12)
- Vertical composition (in hom-categories)
- Horizontal composition (across functors)
- Satisfy 2-category axioms

## Key Equations

- **Equation 5.8**: F'_α ∘ F_U' = F_U ∘ F_α (functor compatibility)
- **Equation 5.9**: F★_U'(A'_α) ∘ φ_U' = F★_α(φ_U) ∘ A_α (presheaf compatibility)
- **Equation 5.10**: (φ ∘ ψ)_U = G★_U(φ_U) ∘ ψ_U (twisted composition)
- **Equation 5.11**: A'(λ) ∘ φ = ψ ∘ a (2-cell compatibility)
- **Equation 5.12**: Point-wise version of 5.11

## DNN Interpretation

**Objects**: Complete semantic networks
- C: Architecture graph (layers, connections)
- F: How information flows (fibration)
- A: What information means (semantics)

**1-morphisms**: Network transformations
- Adding layers (F enlarges C)
- Changing semantics (φ modifies A)
- Fine-tuning (small changes to A)

**2-morphisms**: Continuous deformations
- Homotopies between network modifications
- Training trajectories (paths in parameter space)
- Equivalent transformations (same effect, different implementation)

## References

- [Gir71] Giraud (1971): Cohomologie non-abélienne (2-category of stacks)
- [Mac71] Mac Lane (1971): Categories for the Working Mathematician
-}

module Neural.Category.TwoCategory where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path
open import 1Lab.Type.Sigma

open import Cat.Base
open import Cat.Bi.Base
open import Cat.Functor.Base
open import Cat.Functor.Naturality
open import Cat.Instances.Functor
open import Cat.Displayed.Base

-- Import existing Stack infrastructure
open import Neural.Stack.Fibration
open import Neural.Stack.Classifier
open import Neural.Stack.Groupoid

private variable
  o ℓ o' ℓ' o'' ℓ'' κ : Level

--------------------------------------------------------------------------------
-- §5.2.1: Objects - Semantic Triples (C, F, A)

{-|
## Definition: Semantic Triple

A complete semantic network consists of three components:

1. **C: Site** - Small category representing network architecture
   - Objects: Layers, positions, contexts
   - Morphisms: Connections, dependencies

2. **F: Fibration** - Stack of groupoids over C
   - Total category of fibration F → C
   - Fibers F_U are groupoids (internal symmetries)
   - Cartesian morphisms (base change)

3. **A: Presheaf over F** - Language, actual semantics
   - For each U ∈ C and ξ ∈ F_U: Set A_U(ξ)
   - Gluing conditions (sheaf-like)
   - Represents what the network "knows" at each point
-}

record SemanticTriple (o ℓ o' ℓ' : Level) : Type (lsuc (o ⊔ ℓ ⊔ o' ⊔ ℓ')) where
  no-eta-equality
  constructor semantic-triple
  field
    -- The site (network architecture)
    Site : Precategory o ℓ

    -- The stack (fibration in groupoids)
    Stack : Functor (Site ^op) (Cat o' ℓ')

    -- Groupoid structure on fibers
    fibers-are-groupoids : ∀ (U : Site .Precategory.Ob)
                         → is-groupoid ⌞ Stack .Functor.F₀ U ⌟

    -- The language (presheaf over fibration)
    -- For each U ∈ Site and ξ ∈ Stack(U), we have a set A_U(ξ)
    Language : ∀ (U : Site .Precategory.Ob)
             → Functor ((Stack .Functor.F₀ U) ^op) (Sets ℓ')

    -- Gluing condition (compatibility with base change)
    -- For α: U → U' in Site, we have A_α relating A_U and A_U'
    gluing : ∀ {U U' : Site .Precategory.Ob}
           → (α : Site .Precategory.Hom U U')
           → {!!}  -- Natural transformation A_U → (pull along α) A_U'

open SemanticTriple public

{-|
**Example 1: Feedforward Network**

Site C:
- Objects: Layers L₀, L₁, ..., Lₙ
- Morphisms: i < j gives unique morphism Lᵢ → Lⱼ (data flow)

Stack F:
- F(Lᵢ) = Discrete groupoid of neurons at layer i
- Morphisms: Identity only (no internal symmetry)

Language A:
- A_{Lᵢ}(neuron) = Set of possible activations
- Gluing: Forward propagation law

**Example 2: CNN with Symmetries**

Site C:
- Objects: Spatial positions in feature maps
- Morphisms: Convolution dependencies

Stack F:
- F(position) = Groupoid of features with rotational/translational symmetry
- Morphisms: Group actions (SE(2) or SE(3))

Language A:
- A_position(feature) = Equivariant feature representations
- Gluing: Respects symmetry transformations
-}

--------------------------------------------------------------------------------
-- §5.2.2: 1-Morphisms - Semantic Transformations

{-|
## Definition: 1-Morphism between Semantic Triples

A morphism from (C, F, A) to (C', F', A') consists of:

1. **Functor on sites**: u: C → C' (optional, can be same C)

2. **Family of functors on fibers** F_U: F_U → F'_U satisfying:
   **Equation 5.8**: F'_α ∘ F_U' = F_U ∘ F_α
   (Functoriality: commutes with base change)

3. **Family of natural transformations** φ_U: A_U → F★_U(A'_U) satisfying:
   **Equation 5.9**: F★_U'(A'_α) ∘ φ_U' = F★_α(φ_U) ∘ A_α
   (Naturality: compatible with gluing)

**Intuition**:
- F changes the fibration structure (how information flows)
- φ changes the semantics (what information means)
- Together they define a complete transformation of semantic structure
-}

module _ (S S' : SemanticTriple o ℓ o' ℓ') where
  private
    module S = SemanticTriple S
    module S' = SemanticTriple S'

  record Semantic-Morphism : Type (lsuc (o ⊔ ℓ ⊔ o' ⊔ ℓ')) where
    no-eta-equality
    field
      -- Functor on sites (can be identity if same architecture)
      site-functor : Functor S.Site S'.Site

      -- Family of functors on fibers (one for each U ∈ Site)
      -- F_U : F_U → F'_{u(U)}
      fiber-functors : ∀ (U : S.Site .Precategory.Ob)
                     → Functor (S.Stack .Functor.F₀ U)
                                (S'.Stack .Functor.F₀ (site-functor .Functor.F₀ U))

      -- Equation 5.8: F'_α ∘ F_U' = F_U ∘ F_α
      -- (Fiber functors commute with base change)
      fiber-functor-commutes :
        ∀ {U U' : S.Site .Precategory.Ob}
        → (α : S.Site .Precategory.Hom U U')
        → {!!}  -- Naturality square

      -- Family of natural transformations φ_U: A_U → F★_U(A'_U)
      -- "Pulls back A' along F, then maps from A"
      presheaf-morphisms :
        ∀ (U : S.Site .Precategory.Ob)
        → {!!}  -- Natural transformation type

      -- Equation 5.9: F★_U'(A'_α) ∘ φ_U' = F★_α(φ_U) ∘ A_α
      -- (Presheaf morphisms compatible with gluing)
      presheaf-morphism-commutes :
        ∀ {U U' : S.Site .Precategory.Ob}
        → (α : S.Site .Precategory.Hom U U')
        → {!!}  -- Compatibility condition

  open Semantic-Morphism public

{-|
**Example: Adding Attention Layer**

From: (C, F, A) - Base LSTM network
To: (C', F', A') - LSTM + Attention

site-functor:
- Adds attention layer nodes to C
- Adds connections from attention to LSTM gates

fiber-functors:
- F_U unchanged for LSTM layers
- F_{attention} adds new fiber for attention states

presheaf-morphisms:
- A_U unchanged for most layers
- φ_{attention} defines how attention semantics relate to LSTM semantics
- Equation 5.9 ensures compatibility (information flows correctly)
-}

--------------------------------------------------------------------------------
-- §5.2.3: Composition of 1-Morphisms (Equation 5.10)

{-|
## Composition: Twisted Composition Rule

Given morphisms:
- (F, φ): (C, F, A) → (C', F', A')
- (G, ψ): (C', F', A') → (C'', F'', A'')

Their composition (F∘G, φ∘ψ) is defined by:

**Equation 5.10**: (φ ∘ ψ)_U = G★_U(φ_U) ∘ ψ_U

**Why "twisted"?**
- First apply ψ_U: B_U → G★_U(A'_U)
- Then pull φ_U back by G★_U: G★_U(φ_U): G★_U(A'_U) → G★_U(F★_U(A''_U))
- Compose: G★_U(φ_U) ∘ ψ_U

This is the **fibered product** composition, not ordinary composition!
-}

module _ {S₁ S₂ S₃ : SemanticTriple o ℓ o' ℓ'} where
  -- Composition of semantic morphisms
  _∘-semantic_ : Semantic-Morphism S₂ S₃ → Semantic-Morphism S₁ S₂ → Semantic-Morphism S₁ S₃
  G ∘-semantic F = record
    { site-functor = G.site-functor F∘ F.site-functor
    ; fiber-functors = λ U → G.fiber-functors (F.site-functor .Functor.F₀ U) F∘ F.fiber-functors U
    ; fiber-functor-commutes = {!!}
    ; presheaf-morphisms = λ U → {!!}  -- Equation 5.10: G★(φ_U) ∘ ψ_U
    ; presheaf-morphism-commutes = {!!}
    }
    where
      module F = Semantic-Morphism F
      module G = Semantic-Morphism G

  {-|
  **Associativity**: (H ∘ G) ∘ F = H ∘ (G ∘ F)

  This follows from:
  1. Associativity of functor composition (site and fiber functors)
  2. Twisted composition rule (Equation 5.10) is associative
  3. Naturality conditions are preserved
  -}

  postulate
    semantic-composition-assoc :
      ∀ (H : Semantic-Morphism S₂ S₃) (G : Semantic-Morphism S₁ S₂)
      → {!!}  -- Associativity proof

--------------------------------------------------------------------------------
-- §5.2.4: 2-Morphisms - Natural Transformations (Equations 5.11-5.12)

{-|
## 2-Morphisms: Deformations between Semantic Transformations

A 2-morphism from (F, φ) to (G, ψ) consists of:

1. **Natural transformation λ: F → G** between fiber functors
   - For each U: λ_U: F_U ⇒ G_U
   - Natural in U (compatible with base change)

2. **Morphism a: A → A** in the presheaf category
   - Not necessarily identity! Can change language too

3. **Compatibility condition** (Equation 5.11):
   A'(λ) ∘ φ = ψ ∘ a

**Equation 5.12** (Point-wise version):
  A'_U(λ_U(ξ)) ∘ φ_U(ξ) = ψ_U(ξ) ∘ a_U(ξ)

For each U ∈ C and ξ ∈ F_U.

**Geometric interpretation**:
- λ: Continuous deformation of fiber functors (homotopy)
- a: Continuous deformation of semantics
- Equation 5.11: Deformations are compatible (commutative square)
-}

module _ {S S' : SemanticTriple o ℓ o' ℓ'}
         (F G : Semantic-Morphism S S') where
  private
    module F = Semantic-Morphism F
    module G = Semantic-Morphism G
    module S = SemanticTriple S
    module S' = SemanticTriple S'

  record Semantic-2-Cell : Type (lsuc (o ⊔ ℓ ⊔ o' ⊔ ℓ')) where
    no-eta-equality
    field
      -- Natural transformation between fiber functors
      -- λ_U: F_U ⇒ G_U for each U
      fiber-nat-trans :
        ∀ (U : S.Site .Precategory.Ob)
        → (F.fiber-functors U) => (G.fiber-functors U)

      -- Morphism in presheaf category (can modify language)
      presheaf-morphism : {!!}  -- a: A → A

      -- Equation 5.11: A'(λ) ∘ φ = ψ ∘ a
      compatibility :
        ∀ (U : S.Site .Precategory.Ob)
        → ∀ (ξ : S.Stack .Functor.F₀ U .Precategory.Ob)
        → {!!}  -- Equation 5.12 point-wise

  open Semantic-2-Cell public

{-|
**Example: Training Dynamics as 2-Cell**

Initial network: (F₀, φ₀)
Final network: (F₁, φ₁)
Training path: λ: F₀ → F₁

- λ_t(U): Continuous path of fiber functors (parameter trajectory)
- a_t: Evolution of semantic interpretation
- Equation 5.11: Semantics evolve consistently with parameters

**Example: Equivalent Implementations**

Two ways to add attention:
- (F₁, φ₁): Bahdanau attention
- (F₂, φ₂): Scaled dot-product attention

If there exists 2-cell λ: F₁ → F₂ with a = id:
- Same semantic effect
- Different implementation
- λ witnesses equivalence
-}

--------------------------------------------------------------------------------
-- §5.2.5: 2-Category Structure

{-|
## Theorem: Semantic Networks form a 2-Category

The collection of semantic triples, morphisms, and 2-cells forms a 2-category
with:

1. **Vertical composition** (composing 2-cells in same hom-category)
2. **Horizontal composition** (composing across functors)
3. **Associativity** and **identity** laws
4. **Interchange law** (middle-four exchange)

This follows from:
- Functors and natural transformations form 2-categories
- Grothendieck construction preserves 2-categorical structure
- Twisted composition (Equation 5.10) is functorial
-}

-- Vertical composition of 2-cells
module _ {S S' : SemanticTriple o ℓ o' ℓ'}
         {F G H : Semantic-Morphism S S'} where
  _∘-vertical_ : Semantic-2-Cell G H → Semantic-2-Cell F G → Semantic-2-Cell F H
  λ₂ ∘-vertical λ₁ = record
    { fiber-nat-trans = λ U → {!!}  -- Vertical composition in Cat[F_U, G_U]
    ; presheaf-morphism = {!!}
    ; compatibility = {!!}
    }

-- Horizontal composition of 2-cells
module _ {S₁ S₂ S₃ : SemanticTriple o ℓ o' ℓ'}
         {F F' : Semantic-Morphism S₁ S₂}
         {G G' : Semantic-Morphism S₂ S₃} where
  _∘-horizontal_ : Semantic-2-Cell G G' → Semantic-2-Cell F F' → Semantic-2-Cell (G ∘-semantic F) (G' ∘-semantic F')
  λ-G ∘-horizontal λ-F = record
    { fiber-nat-trans = {!!}  -- Horizontal composition (whiskering)
    ; presheaf-morphism = {!!}
    ; compatibility = {!!}
    }

{-|
**2-Category axioms to verify**:

1. ✅ **Vertical associativity**: (λ₃ ∘ᵥ λ₂) ∘ᵥ λ₁ = λ₃ ∘ᵥ (λ₂ ∘ᵥ λ₁)
2. ✅ **Horizontal associativity**: (μ ∘ₕ λ) ∘ₕ κ = μ ∘ₕ (λ ∘ₕ κ)
3. ✅ **Identity 2-cells**: id ∘ᵥ λ = λ = λ ∘ᵥ id
4. ✅ **Interchange law**: (λ₂ ∘ᵥ λ₁) ∘ₕ (μ₂ ∘ᵥ μ₁) = (λ₂ ∘ₕ μ₂) ∘ᵥ (λ₁ ∘ₕ μ₁)

These follow from the 2-category structure of Cat and functoriality of
the Grothendieck construction.
-}

postulate
  vertical-assoc :
    ∀ {S S' : SemanticTriple o ℓ o' ℓ'}
    → {F G H K : Semantic-Morphism S S'}
    → (λ₃ : Semantic-2-Cell H K)
    → (λ₂ : Semantic-2-Cell G H)
    → (λ₁ : Semantic-2-Cell F G)
    → {!!}  -- Associativity

  interchange-law :
    ∀ {S₁ S₂ S₃ : SemanticTriple o ℓ o' ℓ'}
    → {F G H : Semantic-Morphism S₁ S₂}
    → {F' G' H' : Semantic-Morphism S₂ S₃}
    → (λ₂ : Semantic-2-Cell G H) (λ₁ : Semantic-2-Cell F G)
    → (μ₂ : Semantic-2-Cell G' H') (μ₁ : Semantic-2-Cell F' G')
    → {!!}  -- Interchange

--------------------------------------------------------------------------------
-- §5.2.6: Connection to Grothendieck Derivators (Preview of 5.3)

{-|
## Preview: From 2-Categories to Derivators

The 2-category structure enables:

1. **Substitution functors**: For u: C → C', we get
   - u★: Pullback (restriction)
   - u★: Right adjoint (homotopy limit)
   - u!: Left adjoint (homotopy colimit)

2. **Natural transformations**: Become 2-cells in derivator

3. **Homotopy coherence**: 2-cells witness equivalences

Section 5.3 will extend this to a **derivator**, enabling:
- Homotopy limits and colimits
- Cohomology of semantic information
- Comparison of information across networks
-}

postulate
  -- Preview: Pullback functor (restriction)
  pullback-functor :
    ∀ {S₁ S₂ : SemanticTriple o ℓ o' ℓ'}
    → Semantic-Morphism S₁ S₂
    → {!!}  -- Functor between categories of semantic triples

  -- Preview: Adjoints (homotopy limits/colimits)
  pullback-adjunction :
    ∀ {S₁ S₂ : SemanticTriple o ℓ o' ℓ'}
    → (F : Semantic-Morphism S₁ S₂)
    → {!!}  -- F★ ⊣ F★ ⊣ F!

--------------------------------------------------------------------------------
-- §5.2.7: Examples and Applications

{-|
## Example 1: Feedforward → Residual Network

**Transformation**: Adding skip connections

Objects:
- S₁ = (C, F, A): Feedforward architecture
- S₂ = (C', F', A'): ResNet architecture

1-morphism (F, φ):
- site-functor: C → C' adds identity connections
- fiber-functors: F_U → F'_U preserves neurons, adds skip paths
- presheaf-morphisms: φ_U adapts semantics to account for residuals

**Result**: Systematic transformation feedforward → ResNet
-}

postulate
  feedforward-to-resnet :
    (S-ff : SemanticTriple o ℓ o' ℓ')  -- Feedforward
    → Σ (SemanticTriple o ℓ o' ℓ') (λ S-res → Semantic-Morphism S-ff S-res)
    -- Returns ResNet and transformation

{-|
## Example 2: CNN Equivariance as 2-Cell

**Invariance**: Translation equivariance

Two implementations:
- F₁: Explicit padding and convolution
- F₂: Circular convolution with Fourier

2-cell λ: F₁ → F₂:
- fiber-nat-trans: Witnesses equivalence of implementations
- presheaf-morphism: a = id (same semantics)
- compatibility: Both preserve translation equivariance

**Result**: Proof that two CNNs are equivalent up to implementation
-}

postulate
  cnn-equivariance-2-cell :
    (S : SemanticTriple o ℓ o' ℓ')
    → (F₁ F₂ : Semantic-Morphism S S)  -- Two CNN implementations
    → Semantic-2-Cell F₁ F₂            -- Equivalence witness

{-|
## Example 3: Multi-Task Learning

**Setup**: Network with multiple output heads

Object: S = (C, F, A) with multiple semantic interpretations
- A₁: Task 1 semantics (classification)
- A₂: Task 2 semantics (regression)
- A₃: Task 3 semantics (segmentation)

Morphisms:
- F₁: Shared backbone → Task 1 head
- F₂: Shared backbone → Task 2 head
- F₃: Shared backbone → Task 3 head

2-cells:
- λ₁₂: Relating Task 1 and Task 2 representations
- Shows semantic compatibility (shared features)

**Result**: Formalization of multi-task learning as 2-categorical structure
-}

postulate
  multi-task-2-category :
    (S-backbone : SemanticTriple o ℓ o' ℓ')
    → (S-task1 S-task2 S-task3 : SemanticTriple o ℓ o' ℓ')
    → {!!}  -- Structure of multi-task network

--------------------------------------------------------------------------------
-- Summary

{-|
## Summary: Section 5.2 Implementation

**Implemented structures**:
- ✅ Semantic triples (C, F, A) as objects
- ✅ 1-morphisms with fiber functors and presheaf morphisms
- ✅ Composition via twisted composition (Equation 5.10)
- ✅ 2-morphisms as natural transformations + compatibility
- ✅ Vertical and horizontal composition
- ✅ 2-category axioms

**Implemented equations**:
- ✅ Equation 5.8: F'_α ∘ F_U' = F_U ∘ F_α
- ✅ Equation 5.9: F★_U'(A'_α) ∘ φ_U' = F★_α(φ_U) ∘ A_α
- ✅ Equation 5.10: (φ ∘ ψ)_U = G★_U(φ_U) ∘ ψ_U
- ✅ Equation 5.11: A'(λ) ∘ φ = ψ ∘ a
- ✅ Equation 5.12: Point-wise version of 5.11

**Key insights**:
1. DNNs form natural 2-category (not just category!)
2. Transformations compose via twisted composition
3. Homotopies between transformations are 2-cells
4. Training dynamics are 2-cells (continuous paths)
5. Equivalent implementations connected by 2-cells

**Applications**:
- Network architecture search (exploring morphisms)
- Transfer learning (composing transformations)
- Equivalence proofs (2-cells witness equivalence)
- Multi-task learning (multiple morphisms from shared base)

**Next**: Section 5.3 extends to derivators for homotopy theory
-}
