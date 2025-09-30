{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Grafting Operations and Properad Constraints

This module implements Section 2.3.2 from Manin & Marcolli (2024):
"Homotopy-theoretic and categorical models of neural information networks"

We show how to impose constraints on network summing functors through
**grafting operations** when the target category has properad structure.

## Overview

When the target category C has properad structure (composition along overlapping
inputs/outputs), we can require that network summing functors respect this
compositional structure. This leads to a subcategory Σprop_C(G) ⊂ ΣC(G) where:

1. Resources are assigned to subgraphs based on their inputs/outputs
2. Grafting two subgraphs composes their resource assignments via properad ops
3. The functor is completely determined by its values on single vertices

This is particularly useful when C consists of computational systems (automata,
neural architectures, etc.) where composition has semantic meaning.

## Key Construction (Lemma 2.19):

For acyclic directed graphs G and target category C with properad structure
{C(n,m)}_{n,m∈ℕ}, there is a subcategory Σprop_C(G) of network summing functors
where:

- Φ(G') ∈ C(deg_in(G'), deg_out(G')) for subgraphs G'
- Φ({v}) = Φ(corolla(v)) for single vertices
- Φ(G' ⋆ G'') = Φ(G') ∘_{E(G',G'')} Φ(G'') for composable subgraphs

**Corollary 2.20:** Such functors are completely determined by values on corollas
(single vertices with attached edges).
-}

module Neural.Network.Grafting where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Monoidal.Base
open import Cat.Monoidal.Braided
open import Cat.Instances.Product

import Cat.Reasoning

open import Data.Nat.Base using (Nat; zero; suc; _+_; _-_)
open import Data.Fin.Base using (Fin)
open import Data.Sum.Base
open import Data.List.Base using (List)

open import Neural.Base
open import Neural.SummingFunctor
open import Neural.Network.Conservation

private variable
  o ℓ : Level

{-|
## Definition 2.18: Properads in Cat

A **properad** is a collection P = {P(m,n)}_{m,n∈ℕ} of small categories with
**grafting operations** (composition functors):

  ∘^{i₁,...,iₗ}_{j₁,...,jₗ} : P(m,k) × P(n,r) → P(m+n-ℓ, k+r-ℓ)

for non-empty subsets {i₁,...,iₗ} ⊂ {1,...,k} and {j₁,...,jₗ} ⊂ {1,...,n}.

These satisfy:
- **Associativity**: Nested grafting is independent of order
- **Unity**: Identity 1 ∈ P(1,1) acts as identity for composition
- **Bi-equivariance**: Σₘ × Σₙ acts on P(m,n), compositions are equivariant

**Interpretation for networks:**
- P(m,n) represents "systems with m inputs and n outputs"
- Grafting connects outputs of one system to inputs of another
- Composition is partial - only along specified connections

**Examples:**
1. Deep neural network architectures (see [78] §1.1.1)
2. Automata/computational systems (see §4.1, §4.2)
3. String diagrams / circuit diagrams

NOTE: We assume properads are symmetric (Σₘ × Σₙ actions), otherwise would
need planar structure data. For non-symmetric case, see [105], [68].
-}

record Properad (o ℓ : Level) : Type (lsuc (o ⊔ ℓ)) where
  field
    {-| The family of categories P(m,n) indexed by inputs/outputs -}
    ob : (m n : Nat) → Precategory o ℓ

    {-|
    Grafting operation: compose P ∈ P(m,k) with Q ∈ P(n,r) by connecting
    ℓ outputs of P to ℓ inputs of Q.

    Result is in P(m+n-ℓ, k+r-ℓ) with:
    - Inputs: (m-ℓ from P) + n from Q
    - Outputs: k from P + (r-ℓ from Q)

    TODO: Full implementation requires:
    1. Type-level arithmetic for m+n-ℓ
    2. Indexed families of connection patterns
    3. Associativity and unity proofs
    -}
    graft : ∀ {m k n r ℓ} →
            (connections : List (Fin k × Fin n)) →
            Functor ((ob m k) ×ᶜ (ob n r)) (ob ((m + n) - ℓ) ((k + r) - ℓ))

    {-| Identity element 1 ∈ P(1,1) -}
    unit : (ob 1 1) .Precategory.Ob

    -- TODO: Associativity and unity laws
    -- TODO: Symmetric group actions Σₘ × Σₙ on P(m,n)

{-|
## Categories with Properad Structure

For our application, we need a category C that "contains" a properad structure.
Specifically, C should decompose as a union C = ⋃_{m,n} C(m,n) where each
C(m,n) is a full subcategory, and grafting operations exist.

**Lemma 2.19 conditions:**
1. Objects: Obj(C) = ⋃_{m,n} Obj(C(m,n))
2. Monoidal structure: ⊗ : C(m,k) × C(n,r) → C(m+n, k+r)
3. Properad structure: {C(m,n)}_{m,n} forms a properad in Cat
-}

record HasProperadStructure {C : Precategory o ℓ} (Cᵐ : Monoidal-category C) : Type (lsuc (o ⊔ ℓ)) where
  field
    {-| Full subcategories C(m,n) ⊂ C for systems with m inputs, n outputs -}
    sub : (m n : Nat) → Precategory o ℓ

    {-| Inclusion functors showing C(m,n) are full subcategories -}
    include : ∀ m n → Functor (sub m n) C

    {-| Union property: every object lives in some C(m,n) -}
    decomposition : (X : C .Precategory.Ob) → Σ (Nat × Nat) λ (m , n) → (sub m n) .Precategory.Ob

    {-|
    The monoidal structure ⊗ : C × C → C restricted to C(m,k) × C(n,r)
    lands in C(m+n, k+r).

    This says: tensoring m-input/k-output system with n-input/r-output system
    gives (m+n)-input/(k+r)-output system.
    -}
    monoidal-respects-degrees :
      ∀ {m k n r} →
      (X : (sub m k) .Precategory.Ob) →
      (Y : (sub n r) .Precategory.Ob) →
      (sub (m + n) (k + r)) .Precategory.Ob

    {-| The family {C(m,n)} forms a properad -}
    properad-structure : Properad o ℓ

{-|
## Acyclic Directed Graphs

For the grafting construction, we need directed graphs without cycles.
A **directed cycle** is a sequence of edges e₁,...,eₖ with:
  target(eᵢ) = source(eᵢ₊₁) for i < k, and target(eₖ) = source(e₁)

A graph is **acyclic** if it has no directed cycles.

**Partial order from acyclicity:**
For acyclic G, vertices have a partial order v < w if there exists a directed
path from v to w. We use this to define "composable" subgraphs.
-}

postulate
  -- Predicate: graph G has no directed cycles
  is-acyclic : DirectedGraph → Type

  -- For acyclic graphs, vertices form a partial order
  vertex-order : (G : DirectedGraph) → is-acyclic G → Fin (vertices G) → Fin (vertices G) → Type

{-|
## Open-Ended Graphs (Definition 2.17)

For grafting, we need graphs with **external edges** (inputs/outputs).
An **open-ended directed graph** is a functor G: 2ᵢ/ₒ → F where 2ᵢ/ₒ has:
- Objects: {V, E, Fᵢ, Fₒ} (vertices, edges, input flags, output flags)
- Morphisms: E --fᵢ--> Fᵢ --t--> V <--s-- Fₒ <--fₒ-- E

**Interpretation:**
- Fᵢ(G) = incoming half-edges (flags pointing to vertices)
- Fₒ(G) = outgoing half-edges (flags pointing from vertices)
- fᵢ, fₒ are injective: assign flags to internal edges
- External edges: Eₑₓₜ(G) = (Fᵢ(G) ∖ fᵢ(E)) ⊔ (Fₒ(G) ∖ fₒ(E))

For our purposes:
- deg_in(G') = number of incoming external edges
- deg_out(G') = number of outgoing external edges
-}

postulate
  -- Open-ended directed graph
  OpenGraph : Type

  -- Extract degrees (external inputs/outputs)
  deg-in : OpenGraph → Nat
  deg-out : OpenGraph → Nat

  -- Underlying directed graph (forget external structure)
  underlying-graph : OpenGraph → DirectedGraph

  -- Full convex subgraphs (see paper for definition)
  full-convex-subgraphs : OpenGraph → Type

{-|
## Corollas

The **corolla** C(v) of a vertex v consists of v together with all its attached
half-edges (both incoming and outgoing). This is the "minimal" subgraph around v.

For a vertex v with deg_in(v) incoming edges and deg_out(v) outgoing edges,
C(v) is an open-ended subgraph with:
- 1 vertex: v
- 0 internal edges
- deg_in(v) external inputs
- deg_out(v) external outputs
-}

postulate
  corolla : (G : OpenGraph) → Fin (vertices (underlying-graph G)) → full-convex-subgraphs G

{-|
## Grafting Subgraphs

For two subgraphs G', G'' of G with disjoint vertices, define:
- E(G', G'') = edges in G with source in G' and target in G''
- G' ⋆ G'' = subgraph with V_{G'⋆G''} = V_{G'} ∪ V_{G''} and all edges between/within them

**Composability condition (for acyclic G):**
Say G' < G'' if there are no directed paths from G'' to G'. This ensures
G' ⋆ G'' is well-defined and respects the partial order.
-}

postulate
  -- Edges between two subgraphs
  connecting-edges :
    (G : OpenGraph) →
    (G' G'' : full-convex-subgraphs G) →
    List (Fin (edges (underlying-graph G)))

  -- Grafted subgraph G' ⋆ G''
  _⋆_ :
    {G : OpenGraph} →
    full-convex-subgraphs G →
    full-convex-subgraphs G →
    full-convex-subgraphs G

{-|
## Lemma 2.19: Properad-Constrained Summing Functors

Let C be a symmetric monoidal category with properad structure {C(n,m)}_{n,m∈ℕ}.
Let G be an acyclic directed open-ended graph.

There is a full subcategory **Σprop_C(G) ⊂ ΣC(G)** consisting of summing
functors Φ : P(G) → C satisfying:

1. **Degree-respecting**: For all subgraphs G' ∈ P(G),
     Φ(G') ∈ Obj(C(deg_in(G'), deg_out(G')))

2. **Corolla property**: For any vertex v,
     Φ({v}) = Φ(C(v))
   (value on singleton vertex = value on its corolla)

3. **Grafting property**: For G' < G'' with disjoint vertices,
     Φ(G' ⋆ G'') = Φ(G') ∘_{E(G',G'')} Φ(G'')
   where ∘ is the properad composition along connecting edges E(G',G'')

**Interpretation:**
- Resources assigned to subsystems respect their input/output structure
- Combining subsystems composes their computational capabilities
- The functor encodes how local resources (at vertices) combine globally

**Proof sketch:** Properties (1-3) are closure conditions that define a
subcategory. The grafting property uses associativity of properad composition
to ensure well-definedness regardless of decomposition order.
-}

module Lemma2∙19
  {C : Precategory o ℓ}
  (Cᵐ : Monoidal-category C)
  (Cˢ : Symmetric-monoidal Cᵐ)
  (structure : HasSumsAndZero C)
  (prop : HasProperadStructure Cᵐ)
  (G : OpenGraph)
  (acyclic : is-acyclic (underlying-graph G))
  where

  open HasProperadStructure prop

  postulate
    {-|
    The full subcategory Σprop_C(G) of properad-respecting summing functors.

    Objects: Functors Φ: P(G) → C satisfying conditions (1-3) above
    Morphisms: Invertible natural transformations
    -}
    Σprop : Precategory o ℓ

    {-|
    Inclusion functor Σprop_C(G) ↪ ΣC(G) into all network summing functors.

    This is fully faithful - properad-respecting functors form a full subcategory.
    -}
    properad-inclusion : Functor Σprop (ΣC-network structure (underlying-graph G))

{-|
## Corollary 2.20: Corolla Determinacy

A summing functor Φ ∈ Σprop_C(G) is completely determined by its values on
corollas C(v) for v ∈ V_G.

**Proof sketch:**
- Start with values Φᵥ := Φ(C(v)) for each vertex v
- Choose vertices v₁, v₂ with v₁ < v₂ (using acyclicity order)
- Compute Φ(C(v₁) ⋆ C(v₂)) = Φᵥ₁ ∘_{E(v₁,v₂)} Φᵥ₂ using grafting property
- Inductively build up to full graph G by adding vertices in topological order
- Associativity of properad ensures order of construction doesn't matter
- External edges compose with unit 1 ∈ P(1,1), handled by unity axiom

**Significance:** The functor is entirely determined by "local" data (single
vertices), with global structure coming from properad composition. This is
analogous to how vector bundles are determined by transition functions.
-}

module Corollary2∙20
  {C : Precategory o ℓ}
  (Cᵐ : Monoidal-category C)
  (Cˢ : Symmetric-monoidal Cᵐ)
  (structure : HasSumsAndZero C)
  (prop : HasProperadStructure Cᵐ)
  (G : OpenGraph)
  (acyclic : is-acyclic (underlying-graph G))
  where

  open Lemma2∙19 Cᵐ Cˢ structure prop G acyclic

  postulate
    {-|
    Any Φ ∈ Σprop_C(G) is uniquely determined by its values on corollas.

    Given Φᵥ for all vertices v, there exists unique Φ ∈ Σprop_C(G) extending
    this data, constructed via iterated grafting.
    -}
    corolla-determinacy :
      (corolla-values : (v : Fin (vertices (underlying-graph G))) →
                        C .Precategory.Ob) →
      Σprop .Precategory.Ob
