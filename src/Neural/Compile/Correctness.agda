{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Correctness Proofs for Tensor Species Compilation

**Purpose**: Prove that compilation from Fork-Category sheaves to JAX preserves
categorical structure and is sound.

## What We Must Prove

For the compilation to be correct, we need:

1. **Functoriality Preservation**:
   - If F: Fork-Category^op → Set is a functor with F-id and F-∘
   - Then extract-species(F) preserves these laws in JAX
   - Proof: Show that jnp.einsum composition corresponds to F-∘

2. **Sheaf Condition Preservation**:
   - If F(A★) ≅ ∏_{a'→A★} F(a') in Agda
   - Then compiled fork aggregator computes the product
   - Proof: Show learnable monoid implements the product operation

3. **Gradient Correctness**:
   - Gradient of einsum 'ij,jk->ik' is einsum 'ik,jk->ij'
   - Proof: Use categorical duality (permute the feet!)

4. **Completeness**:
   - Every sheaf on Fork-Category can be extracted
   - Proof: Structural induction on Fork-Category objects/morphisms

## Current Status

- [ ] Functoriality preservation proof
- [ ] Sheaf condition preservation proof
- [ ] Gradient correctness proof
- [ ] Completeness proof
- [ ] Soundness: Compiled program behavior matches categorical semantics

Without these proofs, we just have hopeful scaffolding!
-}

module Neural.Compile.Correctness where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path
open import 1Lab.Equiv

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Instances.Sets
open import Cat.Functor.Naturality

open import Neural.Topos.Architecture
  using (OrientedGraph; ForkVertex; Fork-Category)
  renaming (module ForkConstruction to ForkMod)

open import Neural.Compile.TensorSpecies
  using (TensorSpecies; EinsumOp; einsum; LearnableMonoid; ExtractSpecies)

open import Data.Nat.Base using (Nat)
open import Data.List.Base using (List; []; _∷_)
open import Data.String.Base using (String)

private variable
  o ℓ : Level

{-|
## Einsum Algebra

Before proving functoriality, we need to define the algebra of einsum operations:
composition, identity, associativity.
-}

module EinsumAlgebra where

  {-|
  ### Einsum Composition

  The key insight: composing two einsums means:
  1. Contracting shared indices between output of first and input of second
  2. Combining the einsum strings via index matching

  Example:
    e1 = "ij->j"     (I → J)
    e2 = "jk->k"     (J → K)
    compose(e1, e2) = "ij,jk->ik" would be the FUSED version

  But in sequential composition:
    e2 ∘ e1 means: apply e1 first, then e2
    = "ij->j" then "jk->k"

  For proof purposes, we track this as sequential application.
  -}

  -- Compose two einsum operations (sequential composition)
  compose-einsum : EinsumOp → EinsumOp → EinsumOp
  compose-einsum (identity i) e2 = e2
  compose-einsum e1 (identity o) = e1
  compose-einsum (einsum s1 ins1 outs1) (einsum s2 ins2 outs2) =
    -- Check if outputs of e1 match inputs of e2
    -- For now, construct sequential composition marker
    einsum ("(" ++ s1 ++ ") ; (" ++ s2 ++ ")") ins1 outs2
    where open import Data.String.Base using (_++_)
  compose-einsum e1 e2 = einsum "compose" [] []  -- Fallback

  -- Identity einsum (no-op)
  id-einsum : List IndexVar → EinsumOp
  id-einsum [] = identity (idx "?" 0)  -- Degenerate case
  id-einsum (i ∷ []) = identity i
  id-einsum is = einsum "identity" is is

  {-|
  ### Denotational Semantics

  To state correctness, we need to interpret einsums as FUNCTIONS.
  An einsum with inputs I₁,...,Iₙ and outputs O₁,...,Oₘ denotes:

    ⟦einsum s ins outs⟧ : ∏ᵢ ℝ^(dim Iᵢ) → ∏ⱼ ℝ^(dim Oⱼ)

  For proof purposes, we postulate this interpretation.
  -}
  postulate
    ℝ : Type  -- Real numbers
    Tensor : List Nat → Type  -- Tensor with shape
    einsum-denote : EinsumOp → (List Nat → List Nat → Type)  -- Input shapes → Output shapes → Function type

  -- Composition property: Denotation of composition = composition of denotations
  postulate
    einsum-compose-correct :
      ∀ (e1 e2 : EinsumOp) →
      einsum-denote (compose-einsum e1 e2) ≡ {!!}  -- Function composition of denotations

{-|
## Theorem 1: Functoriality Preservation

If F: C^op → Set is a functor, then the extracted tensor species preserves
composition.

Statement:
  For morphisms f: x → y and g: y → z in Fork-Category,
  F₁(f ∘ g) = F₁(g) ∘ F₁(f)

  If we extract these to einsums e_f and e_g, then:
  einsum(e_f ∘ e_g) ≡ einsum(e_g) ∘ einsum(e_f)

This is the KEY correctness property: composition in JAX matches composition
in the category!
-}

module FunctorialityPreservation
  {Γ : OrientedGraph o ℓ}
  (F : Functor (Fork-Category Γ ^op) (Sets o))
  where

  open ForkMod Γ
  open Functor F
  private
    C = Fork-Category Γ
    module C = Precategory C

  -- The extracted tensor species
  open ExtractSpecies Γ
  open EinsumAlgebra

  {-|
  ### Theorem Statement

  For any two composable morphisms in Fork-Category,
  extraction preserves composition in the sense that:

  The einsum extracted from (f ∘ g) is equivalent (up to denotational semantics)
  to composing the einsums extracted from f and g separately.
  -}
  functoriality-preserved :
    ∀ {x y z : C.Ob} (f : C.Hom y z) (g : C.Hom x y) →
    let e_fg = morphism-to-einsum (f C.∘ g)
        e_f = morphism-to-einsum f
        e_g = morphism-to-einsum g
        e_composed = compose-einsum e_g e_f  -- Note: reverse order (g then f)
    in einsum-denote e_fg ≡ einsum-denote e_composed
  functoriality-preserved f g = {!!}

  {-|
  ### Proof Strategy

  1. Pattern match on Fork-Category morphisms (ForkEdge constructors)
  2. For each edge type, show that extracted einsum preserves composition
  3. Use functor laws: F-∘ : F₁(f ∘ g) ≡ F₁(g) ∘ F₁(f)

  Key cases from ForkEdge:
  - **orig-edge**: Standard edges between non-fork vertices
    → Extract as linear transformations (einsum "ij->j")
    → Composition: chain rule

  - **tip-to-star**: Edges to fork aggregation points
    → Extract as identity (routing to aggregator)
    → Composition: identity laws

  - **star-to-tang**: Aggregation application
    → Extract as learnable monoid (special einsum)
    → Composition: monoid associativity

  - **tang-to-handle**: Post-aggregation edges
    → Extract as identity
    → Composition: identity laws

  The proof reduces to showing that einsum composition matches
  the categorical composition in each case.
  -}

  -- Lemma: Identity morphisms extract to identity einsums
  postulate
    id-extracts-to-id : ∀ (x : C.Ob) →
      morphism-to-einsum (C.id {x}) ≡ id-einsum []

  -- Lemma: Composition of orig-edges preserves einsum composition
  postulate
    orig-edge-composition :
      ∀ {x y z : ForkVertex}
        (e1 : OrientedGraph.Edge Γ x y) (e2 : OrientedGraph.Edge Γ y z) →
      let -- TODO: construct ForkEdge instances
          f = {!!} -- orig-edge from e1
          g = {!!} -- orig-edge from e2
      in morphism-to-einsum (f C.∘ g) ≡ compose-einsum (morphism-to-einsum g) (morphism-to-einsum f)

{-|
## Theorem 2: Sheaf Condition Preservation

For fork vertices A★, the sheaf condition F(A★) ≅ ∏ F(incoming) must be
preserved by compilation.

Statement:
  If F satisfies the sheaf condition at A★ in Agda,
  then the compiled learnable monoid computes the product.

Proof obligation:
  learnable-monoid([x₁, x₂, ...]) ≡ x₁ × x₂ × ...

Where × is the categorical product in Sets (tuple construction).
-}

module SheafConditionPreservation
  {Γ : OrientedGraph o ℓ}
  (F : Functor (Fork-Category Γ ^op) (Sets o))
  where

  open ForkMod Γ
  open EinsumAlgebra
  private
    C = Fork-Category Γ
    module C = Precategory C

  {-|
  ### Algebraic Structure of Aggregators

  A learnable monoid is a binary operator ⊕: X × X → X with learned parameters θ:
    x ⊕_θ y = MLP_θ(concat(x, y))

  For correctness, we need it to be a COMMUTATIVE MONOID:
  1. Associativity: (x ⊕ y) ⊕ z ≡ x ⊕ (y ⊕ z)
  2. Commutativity: x ⊕ y ≡ y ⊕ x
  3. Identity: ∃ε. x ⊕ ε ≡ x

  The categorical product in Sets is the unique commutative monoid satisfying
  the universal property!
  -}

  -- Algebraic properties of learned aggregators
  record MonoidProperties (aggregate : List ℝ → ℝ) : Type where
    field
      -- Associativity: tree structure doesn't matter
      associative : ∀ (x y z : ℝ) →
        aggregate (aggregate (x ∷ y ∷ []) ∷ z ∷ []) ≡
        aggregate (x ∷ aggregate (y ∷ z ∷ []) ∷ [])

      -- Commutativity: order doesn't matter
      commutative : ∀ (x y : ℝ) →
        aggregate (x ∷ y ∷ []) ≡ aggregate (y ∷ x ∷ [])

      -- Identity element
      has-identity : Σ[ ε ∈ ℝ ] (∀ (x : ℝ) → aggregate (x ∷ ε ∷ []) ≡ x)

  {-|
  ### Connection to Categorical Product

  In Sets, the categorical product A × B is characterized by:
  - Universal property: ∀ f: X → A, g: X → B, ∃! h: X → A × B
  - Projections: π₁: A × B → A, π₂: A × B → B

  For fork vertices with incoming edges e₁, e₂:
    F(A★) ≅ F(source(e₁)) × F(source(e₂))

  The aggregator implements this product via:
    aggregate([h₁, h₂]) ≈ (h₁, h₂)  (learned to be tuple-like)
  -}

  -- Sheaf condition: F(A★) ≅ ∏ F(incoming)
  postulate
    sheaf-condition :
      ∀ (a : ForkVertex) (conv : is-convergent a) →
      {!!}  -- F(fork-star a conv) ≅ ∏_{e → fork-star} F(source e)

  {-|
  ### Theorem Statement

  If a learnable monoid satisfies the monoid properties (after training),
  then it implements the categorical product from the sheaf condition.
  -}
  theorem-sheaf-preservation :
    ∀ (a : ForkVertex) (conv : is-convergent a)
      (aggregate : List ℝ → ℝ)
      (props : MonoidProperties aggregate) →
    {!!}  -- aggregate implements the product from sheaf-condition

  theorem-sheaf-preservation a conv agg props = {!!}

  {-|
  ### Proof Strategy

  **Step 1**: Show that commutative monoids in Sets are exactly products
    - Use universal property of products
    - Monoid operation gives pairing: (x, y) ↦ x ⊕ y
    - Associativity + commutativity ensure this is well-defined

  **Step 2**: Show that learnable monoids can satisfy these properties
    - Associativity loss: L_assoc = 𝔼[||(x ⊕ y) ⊕ z - x ⊕ (y ⊕ z)||²]
    - Commutativity loss: L_comm = 𝔼[||x ⊕ y - y ⊕ x||²]
    - Training minimizes these → properties hold approximately

  **Step 3**: Connect to sheaf condition
    - F(A★) = aggregate of F(incoming) (by construction in extraction)
    - Sheaf condition says F(A★) ≅ ∏ F(incoming)
    - If aggregate is a commutative monoid, it implements the product ✓

  **Key insight**: The sheaf condition is STRUCTURAL (dimensions match),
  and the learned aggregator makes it FUNCTIONALLY correct!

  Example (Diamond Network):
    - F(input₁) = ℝ¹⁰, F(input₂) = ℝ¹⁰
    - Sheaf condition: F(hidden★) ≅ ℝ¹⁰ × ℝ¹⁰ ≅ ℝ²⁰
    - Aggregator: combine(h₁: ℝ¹⁰, h₂: ℝ¹⁰) → ℝ²⁰
    - Dimension matches (structural) ✓
    - After training: combine ≈ (h₁, h₂) (functional) ✓
  -}

  -- Lemma: Dimensions match structurally
  postulate
    dimensions-match :
      ∀ (a : ForkVertex) (conv : is-convergent a)
        (incoming : List ForkVertex) →
      {!!}  -- dim(F(fork-star)) = sum(dim(F(incoming)))

  -- Lemma: Learned aggregator satisfies monoid properties (after training)
  postulate
    training-ensures-monoid :
      ∀ (monoid : LearnableMonoid)
        (regularization-weight : ℝ) →
      {!!}  -- After training, the aggregator satisfies MonoidProperties

{-|
## Theorem 3: Gradient Correctness

From Dudzik: "The gradient flow through an einsum is an einsum."

Statement:
  For einsum 'ij,jk->ik' with gradient ∂L/∂output,
  the gradient ∂L/∂input is einsum 'ik,jk->ij'

Proof: Categorical duality - transpose the parametric span!
-}

module GradientCorrectness where

  {-|
  ### Parametric Span Representation

  An einsum 'ij,jk->ik' is a span:

      I × J × K
     ↙     ↓     ↘
   I×J    J×K    I×K

  Gradient: permute the feet!

      I × K × J     (swap J and K in apex)
     ↙     ↓     ↘
   I×K    J×K    I×J   (new target!)

  This gives einsum 'ik,jk->ij'
  -}

  -- Einsum duality operation
  postulate
    einsum-dual : EinsumOp → EinsumOp

    -- The dual of an einsum is its gradient
    einsum-dual-is-gradient :
      ∀ (op : EinsumOp) →
      {!!}  -- einsum-dual(op) computes gradient of op

  {-|
  ### Proof Strategy

  1. Represent einsum as parametric span (apex + feet)
  2. Show that permuting feet corresponds to transposition
  3. Use calculus: ∂/∂x (x·y) = y (chain rule becomes foot permutation!)
  4. Verify with concrete examples (matmul, conv, etc.)

  Example: Matrix multiplication
  ```
  Forward: C = A @ B  (einsum 'ij,jk->ik')
  ∂L/∂A = ∂L/∂C @ B^T  (einsum 'ik,jk->ij')  ✓ Feet permuted!
  ∂L/∂B = A^T @ ∂L/∂C  (einsum 'ij,ik->jk')  ✓ Feet permuted!
  ```

  This is a THEOREM, not just an observation!
  -}

{-|
## Theorem 4: Completeness

Every sheaf on Fork-Category can be extracted to a tensor species.

Statement:
  ∀ (F : Functor (Fork-Category^op) Sets),
  ∃ (S : TensorSpecies), extract-species(F) = S

Proof: Structural induction on Fork-Category.
-}

module Completeness
  {Γ : OrientedGraph o ℓ}
  where

  open ForkMod Γ
  private
    C = Fork-Category Γ

  {-|
  ### Enumeration Lemma

  Fork-Category is a finite poset, so we can enumerate all objects and morphisms.
  -}
  postulate
    enumerate-objects : List C.Ob
    enumerate-morphisms : List (Σ[ x ∈ C.Ob ] Σ[ y ∈ C.Ob ] (C.Hom x y))

    finite-fork-category :
      ∀ (x : C.Ob) → x ∈ enumerate-objects

  {-|
  ### Completeness Theorem

  For any functor F, we can extract a complete tensor species.
  -}
  postulate
    extraction-complete :
      ∀ (F : Functor (C ^op) (Sets o)) →
      let S = ExtractSpecies.extract-species Γ F
      in {!!}  -- S contains all objects/morphisms of F

  {-|
  ### Proof Strategy

  1. For each object c ∈ enumerate-objects:
     - Create IndexVar from c
     - Extract dimension from F₀(c)

  2. For each morphism (x, y, f) ∈ enumerate-morphisms:
     - Convert F₁(f) to einsum via pattern matching on fork morphisms

  3. For each fork-star vertex:
     - Create LearnableMonoid from incoming edges

  4. Verify that all functoriality laws are preserved (Theorem 1)

  This is constructive! We literally enumerate the category and extract.
  -}

{-|
## Theorem 5: Soundness

The compiled JAX program has the same behavior as the categorical semantics.

Statement:
  If F: C^op → Set is a sheaf and S = extract-species(F),
  then for any input x and morphism path p in C,

  JAX_compiled(S)(x, p) ≡ F₁(p)(x)

This is the ULTIMATE correctness property: the compiled code computes
the same function as the categorical definition!
-}

module Soundness
  {Γ : OrientedGraph o ℓ}
  (F : Functor (Fork-Category Γ ^op) (Sets o))
  where

  open ForkMod Γ
  private
    C = Fork-Category Γ

  open ExtractSpecies Γ

  {-|
  ### Denotational Semantics

  We need to give semantics to JAX programs in terms of functors.
  -}
  postulate
    -- Interpret JAX einsum as a function
    JAX-einsum-semantics : EinsumOp → (List (Functor.F₀ F _) → Functor.F₀ F _)

    -- The compiled program denotes the functor application
    soundness-theorem :
      ∀ {x y : C.Ob} (f : C.Hom x y) (input : Functor.F₀ F y) →
      let e = morphism-to-einsum f
          jax-result = JAX-einsum-semantics e (input ∷ [])
          agda-result = Functor.F₁ F f input
      in jax-result ≡ agda-result

  {-|
  ### Proof Strategy

  1. Define denotational semantics for JAX operations
     - jnp.einsum('ij,jk->ik', A, B) ≡ matrix multiplication
     - This is compositional!

  2. Show that einsum semantics matches functor application
     - Use Theorem 1 (functoriality preservation)
     - Use Theorem 3 (gradient correctness)

  3. Extend to full programs via composition

  **This proof would be the CROWN JEWEL** - it shows our compilation is
  provably correct, not just hopeful!
  -}

{-|
## What We Need to Complete These Proofs

1. **Formal einsum algebra**: Define composition, duality, etc.
2. **JAX semantics in Agda**: Model JAX operations categorically
3. **Actual extraction implementation**: Fill the {!!} holes in TensorSpecies.agda
4. **Property-based testing**: Test soundness on random inputs

The current code is scaffolding. These proofs make it REAL.
-}
