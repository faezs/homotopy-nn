{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Networks and Conservation Laws

This module implements Section 2.2 from Manin & Marcolli (2024):
"Homotopy-theoretic and categorical models of neural information networks"

We connect directed graphs (networks) to summing functors, showing how to impose
conservation laws at vertices using categorical equalizers and coequalizers.

## Overview

Given a directed graph G representing a network:
- Nodes VG represent neurons
- Edges EG represent connections
- ΣC(VG) parameterizes resource assignments to nodes
- ΣC(EG) parameterizes resource assignments to edges

The source/target maps of G induce functors s,t: ΣC(EG) → ΣC(VG), allowing us
to express compatibility between edge and vertex assignments.

## Two approaches to conservation:

1. **Equalizer (Proposition 2.10)**: Select summing functors on edges satisfying
   Kirchhoff's law Σ{e: s(e)=v} Φ(e) = Σ{e: t(e)=v} Φ(e)

2. **Coequalizer (Proposition 2.12)**: Modify the target category to a quotient
   where conservation laws hold automatically
-}

module Neural.Network.Conservation where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Diagram.Equaliser
open import Cat.Diagram.Coequaliser
open import Cat.Diagram.Coproduct
open import Cat.Monoidal.Base
open import Cat.Monoidal.Braided

import Cat.Reasoning

open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin)
open import Data.Bool.Base using (Bool)
open import Data.Sum.Base

open import Neural.Base
open import Neural.SummingFunctor

private variable
  o ℓ : Level

{-|
## 2.2 Networks and Summing Functors

We begin by showing how the combinatorial structure of a directed graph
induces functors between categories of summing functors.
-}

{-|
### Lemma 2.9: Source and Target Functors

The source and target maps s,t: E → V of a directed graph G determine functors
between the categories of summing functors:

  s,t: ΣC(EG) → ΣC(VG)

These functors transform edge-based resource assignments into vertex-based ones
by taking preimages:

  Φˢ_V(A) := Φ_E(s⁻¹(A)) = ⊕{e ∈ E: s(e) ∈ A} Φ_E(e)
  Φᵗ_V(A) := Φ_E(t⁻¹(A)) = ⊕{e ∈ E: t(e) ∈ A} Φ_E(e)

In network terms:
- Φˢ_V assigns to each vertex v the sum of resources on its outgoing edges
- Φᵗ_V assigns to each vertex v the sum of resources on its incoming edges

**Proof sketch from paper:**
The summing property is preserved because for disjoint A ∩ A' = {*} in P(VG):
  Φˢ_V(A ∪ A') = Φ_E(s⁻¹(A ∪ A'))
                = Φ_E(s⁻¹(A) ∪ s⁻¹(A'))
                = Φ_E(s⁻¹(A)) ⊕ Φ_E(s⁻¹(A'))  (summing property of Φ_E)
                = Φˢ_V(A) ⊕ Φˢ_V(A')

The construction works because s⁻¹ preserves disjoint unions.

TODO: Full implementation requires:
1. Category P(X) of pointed subsets with inclusions (from SummingFunctor)
2. Indexed coproducts in C to express ⊕{e: s(e) ∈ A}
3. Proof that preimage preserves the summing property
-}

module Lemma2∙9
  {C : Precategory o ℓ}
  (structure : HasSumsAndZero C)
  (G : DirectedGraph) where

  open HasSumsAndZero structure

  -- Number of vertices and edges
  n-vertices : Nat
  n-vertices = vertices G

  n-edges : Nat
  n-edges = edges G

  open Lemma2∙3 structure

  postulate
    {-|
    The source functor s: ΣC(EG) → ΣC(VG) induced by the source map.

    For a summing functor Φ_E on edges, produces Φˢ_V on vertices where:
      Φˢ_V(A) = ⊕{e: source(e) ∈ A} Φ_E(e)
    -}
    source-functor : Functor (ΣC[ n-edges ]) (ΣC[ n-vertices ])

    {-|
    The target functor t: ΣC(EG) → ΣC(VG) induced by the target map.

    For a summing functor Φ_E on edges, produces Φᵗ_V on vertices where:
      Φᵗ_V(A) = ⊕{e: target(e) ∈ A} Φ_E(e)
    -}
    target-functor : Functor (ΣC[ n-edges ]) (ΣC[ n-vertices ])

{-|
## 2.3 General Framework: Network Summing Functors (Definition 2.14)

Before specializing to conservation laws, we introduce the general notion of
network summing functors. This provides a framework where specific constraints
(conservation, grafting, etc.) appear as subcategories.

A **subgraph** of G is another directed graph G': 2 → F with a natural
transformation α: G' → G (inclusions αV: VG' → VG and αE: EG' → EG).

The category **P(G)** has:
- Objects: subgraphs G' → G
- Morphisms: inclusions between subgraphs

A **network summing functor** is Φ: P(G) → C satisfying:
1. Φ(∅) = 0 (maps empty graph to zero object)
2. Φ(G' ⊔ G'') = Φ(G') ⊕ Φ(G'') for disjoint subgraphs

The category **ΣC(G)** consists of network summing functors with invertible
natural transformations as morphisms.

**Interpretation:** This generalizes summing functors from subsets of a set X
to subgraphs of a network G. Instead of assigning resources to individual
vertices/edges, we assign resources to entire subnetworks in a compositional way.

**Remark 2.15:** For pointed graphs G* with basepoint component {v*, e*}, the
categories ΣC(G) and ΣC(G*) are equivalent since the basepoint is mapped to 0.
We use the same notation ΣC(G) for both.

NOTE: The categories Σeq_C(G) (conservation via equalizer) and Σcoeq_C(G)
(conservation via coequalizer) from §2.2 are subcategories of ΣC(G),
corresponding to different types of constraints at vertices.

TODO: Full implementation requires:
1. Category of subgraphs P(G) for a given directed graph G
2. Formalization of "disjoint" subgraphs (non-overlapping vertices and edges)
3. Construction of ΣC(G) as a category with appropriate morphisms
-}

postulate
  -- Category of subgraphs of G
  Subgraphs : DirectedGraph → Precategory lzero lzero

  -- Network summing functors Φ: P(G) → C
  NetworkSummingFunctor : (C : Precategory o ℓ) → DirectedGraph → Type (o ⊔ ℓ)

  -- Category of network summing functors with invertible natural transformations
  ΣC-network : {C : Precategory o ℓ} → HasSumsAndZero C → DirectedGraph → Precategory o ℓ

{-|
## 2.2.1 Conservation Laws via Equalizer (Proposition 2.10)

The **equalizer** Σeq_C(G) of the two functors s,t: ΣC(EG) ⇒ ΣC(VG) consists
of summing functors Φ_E on edges that satisfy **Kirchhoff's conservation law**:

  Σ{e: s(e)=v} Φ_E(e) = Σ{e: t(e)=v} Φ_E(e)    for all vertices v

In categorical terms, this is the equalizer:

  Σeq_C(G) --ι--> ΣC(EG) ==s==> ΣC(VG)
                          ==t==>

with universal property: for any A and q: A → ΣC(EG) with s∘q = t∘q,
there exists unique u: A → Σeq_C(G) with ι∘u = q.

**Physical interpretation:** In electrical networks, this is Kirchhoff's current
law - total current flowing into a vertex equals total current flowing out.

**Mathematical content:** The equalizer selects those edge-based resource
assignments where the sum of resources on incoming edges equals the sum on
outgoing edges at each vertex.

**Proof from paper:** A summing functor Φ_E is in the equalizer iff for all
pointed subsets A ∈ P(VG):

  Φ_E(s⁻¹(A)) = Φ_E(t⁻¹(A))

By Lemma 2.3, summing functors are determined by their values on singletons,
so this reduces to the pointwise condition (2.3) at each vertex v.

TODO: Implement using Cat.Diagram.Equaliser from 1Lab.
-}

module Proposition2∙10
  {C : Precategory o ℓ}
  (structure : HasSumsAndZero C)
  (G : DirectedGraph) where

  open Lemma2∙9 structure G
  open Lemma2∙3 structure

  postulate
    {-|
    The equalizer category Σeq_C(G) consisting of summing functors on edges
    that satisfy conservation at vertices.

    Objects: Φ_E: P(EG) → C with Φˢ_V = Φᵗ_V
    Morphisms: Invertible natural transformations
    -}
    Σeq : Precategory o ℓ

    {-|
    Inclusion functor ι: Σeq_C(G) → ΣC(EG) into edge summing functors.
    -}
    equalizer-inclusion : Functor Σeq (ΣC[ n-edges ])

    {-|
    The equalizer property: s ∘ ι = t ∘ ι

    This says that for Φ_E in the equalizer, applying source or target
    functors gives the same vertex-based assignment.
    -}
    equalizer-property :
      source-functor F∘ equalizer-inclusion ≡ target-functor F∘ equalizer-inclusion

{-|
## 2.2.2 Conservation Laws via Coequalizer (Proposition 2.12 & Definition 2.13)

The **coequalizer** construction takes a dual approach: instead of selecting
specific summing functors (as equalizer does), it modifies the target category
to a quotient where conservation laws hold automatically.

Given edge-based assignment Φ_E: P(EG) → C, the coequalizer is:

  C --ρG--> Ccoeq_G(Φ_E) = coeq(Φˢ_V, Φᵗ_V)

where Ccoeq_G(Φ_E) is the quotient category C/~_{G,Φ_E} identifying resources
that differ only by flow conservation at vertices.

**Key equation in the quotient:**

  ρG(Φ_E(s⁻¹(A))) = ρG(Φ_E(t⁻¹(A)))    for all A ∈ P(VG)

**Interpretation:** Resources at incoming edges of a vertex are "identified with"
resources at outgoing edges in the quotient. The category Ccoeq_G is the
"optimal" target for imposing conservation - any other category with this
property factors through it uniquely.

**Construction from paper:** By [12], coequalizers in Cat are quotients by
generalized congruences. The coequalizer C/~_{G,Φ_E} is generated by relations:
- Objects: Φ_E(s⁻¹(A)) ~ Φ_E(t⁻¹(A)) for all A ∈ P(VG)
- Morphisms: corresponding inclusions are identified

**Quotient category implementation note:**
The quotient category construction C/~ exists in the cubical Agda library as
Cubical.Categories.Constructions.Quotient. It takes a category C and an
equivalence relation on morphisms, producing a new category where morphisms
are equivalence classes. This could be adapted using 1Lab's infrastructure,
but requires significant work. For now we postulate the construction.

**Definition 2.13:** If the multiple coequalizer Ccoeq_G admits a symmetric
monoidal structure, we can define:

  Σcoeq_C(G) := ΣCcoeq_G(EG)

This is the category of summing functors valued in the quotient category,
where conservation laws hold by construction.

**Advantage over equalizer:** Can impose conservation by modifying only the
target category, without restricting the collection of summing functors.
-}

module Proposition2∙12
  {C : Precategory o ℓ}
  (structure : HasSumsAndZero C)
  (G : DirectedGraph) where

  open Lemma2∙9 structure G
  open Lemma2∙3 structure

  postulate
    {-|
    The coequalizer category Ccoeq_G(Φ_E) for a specific edge-based summing
    functor Φ_E, obtained by quotienting C by the conservation relations.

    This is the quotient C/~_{G,Φ_E} where ~ is the generalized congruence
    generated by Φ_E(s⁻¹(A)) ~ Φ_E(t⁻¹(A)) for all A ∈ P(VG).

    NOTE: Could be implemented using quotient category construction similar to
    Cubical.Categories.Constructions.Quotient adapted to 1Lab.
    -}
    Ccoeq : (Φ_E : SummingFunctorData n-edges) → Precategory o ℓ

    {-|
    Quotient functor ρG: C → Ccoeq_G(Φ_E) that identifies resources related
    by conservation laws at vertices.

    This is the coequalizer of Φˢ_V, Φᵗ_V: P(VG) ⇒ C.
    -}
    coequalizer-quotient :
      (Φ_E : SummingFunctorData n-edges) →
      Functor C (Ccoeq Φ_E)

    {-|
    Universal property of the coequalizer: ρG is optimal for imposing
    conservation. Any functor ρ: C → R with ρ ∘ Φˢ_V = ρ ∘ Φᵗ_V factors
    uniquely through ρG.
    -}
    coequalizer-universal :
      (Φ_E : SummingFunctorData n-edges) →
      {R : Precategory o ℓ} →
      (ρ : Functor C R) →
      -- TODO: Express ρ ∘ Φˢ_V = ρ ∘ Φᵗ_V condition
      Functor (Ccoeq Φ_E) R

    {-|
    The multiple coequalizer Ccoeq_G over ALL edge-based summing functors,
    quotienting C so that conservation holds for every Φ_E ∈ ΣC(EG).

    This is the quotient by the generalized congruence generated by all
    relations Φ_E(s⁻¹(A)) ~ Φ_E(t⁻¹(A)) as Φ_E varies over ΣC(EG).
    -}
    Ccoeq-global : Precategory o ℓ

    {-|
    Global quotient functor ρG: C → Ccoeq_G imposing conservation for all
    possible edge assignments simultaneously.
    -}
    coequalizer-global-quotient : Functor C Ccoeq-global

{-|
### Definition 2.13: Category of Summing Functors with Coequalizer Constraints

If the multiple coequalizer Ccoeq_G admits a symmetric monoidal structure
(which is generally true - quotients preserve monoidal structure), we can define:

  Σcoeq_C(G) := ΣCcoeq_G(EG)

This is the category of summing functors on edges, but valued in the quotient
category where conservation laws (2.5) hold automatically.

**Advantage:** Imposes conservation by modifying the target category rather than
restricting to a subcategory of functors (as in the equalizer approach).

TODO: Requires proof that quotient preserves symmetric monoidal structure.
-}

-- Category of summing functors with coequalizer-imposed conservation
postulate
  Σcoeq :
    {C : Precategory o ℓ} →
    (structure : HasSumsAndZero C) →
    (G : DirectedGraph) →
    Precategory o ℓ

{-|
# Examples and Tests

Concrete examples demonstrating the conservation law constructions.
-}

module Examples where
  open import Cat.Instances.FinSets
  open import Data.Fin.Base using (fzero; fsuc)

  {-|
  ## Example 1: Simple Linear Graph

  A graph with 2 vertices and 1 edge: v₀ --e₀--> v₁

  This is the simplest non-trivial directed graph.
  -}

  example-linear-graph : DirectedGraph
  example-linear-graph .Functor.F₀ true = 2   -- 2 vertices: v₀, v₁
  example-linear-graph .Functor.F₀ false = 1  -- 1 edge: e₀
  example-linear-graph .Functor.F₁ {false} {true} true = λ _ → fzero   -- source(e₀) = v₀
  example-linear-graph .Functor.F₁ {false} {true} false = λ _ → fsuc fzero -- target(e₀) = v₁
  example-linear-graph .Functor.F₁ {false} {false} tt = Precategory.id FinSets  -- identity on edges
  example-linear-graph .Functor.F₁ {true} {true} tt = Precategory.id FinSets    -- identity on vertices
  example-linear-graph .Functor.F-id {false} = refl
  example-linear-graph .Functor.F-id {true} = refl
  example-linear-graph .Functor.F-∘ = {!!}

  _ : vertices example-linear-graph ≡ 2
  _ = refl

  _ : edges example-linear-graph ≡ 1
  _ = refl

  _ : source example-linear-graph fzero ≡ fzero
  _ = refl

  _ : target example-linear-graph fzero ≡ fsuc fzero
  _ = refl

  {-|
  ## Example 2: Conservation Law Check

  For the linear graph v₀ → v₁, a summing functor Φ on edges satisfies
  Kirchhoff's law if:
  - Resources flowing out of v₀ = Φ(e₀)
  - Resources flowing into v₁ = Φ(e₀)
  - Resources are conserved: inflow = outflow at each vertex

  At v₀: outflow = Φ(e₀), inflow = 0 (source vertex)
  At v₁: outflow = 0 (sink vertex), inflow = Φ(e₀)

  For conservation at interior vertices (none in this simple graph),
  we'd need: Σ{e: source(e)=v} Φ(e) = Σ{e: target(e)=v} Φ(e)
  -}

  -- Using FinSets as our target category (has sums and zero)
  example-category : HasSumsAndZero FinSets
  example-category .HasSumsAndZero.has-zero = record { ∅ = zero
                                                     ; has-is-zero = record { has-is-initial = λ x → contr (λ ()) λ x₁  → {!!}
                                                                            ; has-is-terminal = λ x → {!!} } }  -- FinSets has zero object (empty set)
  example-category .HasSumsAndZero.has-binary-coproducts = λ A B → record { coapex = A ; ι₁ = λ z → z ; ι₂ = {!!} ; has-is-coproduct = {!!} }  -- FinSets has coproducts (disjoint union)

  {-|
  ## Example 3: Source and Target Functors on Linear Graph

  For the linear graph with 1 edge, the source and target functors map:
  - Edge-based assignments Φ_E: P(E) → C
  - To vertex-based assignments Φ_V: P(V) → C

  Φˢ_V({v₀}) = Φ_E({e₀})  (edges sourced from v₀)
  Φˢ_V({v₁}) = Φ_E(∅) = 0  (no edges sourced from v₁)

  Φᵗ_V({v₀}) = Φ_E(∅) = 0  (no edges targeting v₀)
  Φᵗ_V({v₁}) = Φ_E({e₀})  (edges targeting v₁)

  The equalizer Σeq selects those Φ_E where Φˢ_V = Φᵗ_V, which for this
  graph means Φ_E({e₀}) = 0 at both v₀ and v₁ - only the trivial assignment!
  -}

  module LinearGraphConservation where
    open Lemma2∙9 example-category example-linear-graph
    open Proposition2∙10 example-category example-linear-graph
    open Lemma2∙3 example-category

    {-|
    Test: The source and target functors exist and have the correct types.
    They map from ΣC(edges=1) to ΣC(vertices=2).
    -}
    _ : Functor (ΣC[ 1 ]) (ΣC[ 2 ])
    _ = source-functor

    _ : Functor (ΣC[ 1 ]) (ΣC[ 2 ])
    _ = target-functor

    {-|
    Test: The equalizer category exists and includes into edge summing functors.
    -}
    _ : Functor Σeq (ΣC[ 1 ])
    _ = equalizer-inclusion

  {-|
  ## Example 4: Triangle Graph with Conservation

  A graph with 3 vertices and 2 edges forming a path: v₀ → v₁ → v₂

  This demonstrates conservation at an interior vertex (v₁).
  -}

  triangle-graph : DirectedGraph
  triangle-graph .Functor.F₀ true = 3   -- 3 vertices
  triangle-graph .Functor.F₀ false = 2  -- 2 edges
  triangle-graph .Functor.F₁ {false} {true} true =
    Data.Fin.Base.Fin-cases fzero (λ _ → fsuc fzero)  -- source: e₀↦v₀, e₁↦v₁
  triangle-graph .Functor.F₁ {false} {true} false =
    Data.Fin.Base.Fin-cases (fsuc fzero) (λ _ → fsuc (fsuc fzero))  -- target: e₀↦v₁, e₁↦v₂
  triangle-graph .Functor.F₁ {false} {false} _ = Precategory.id FinSets
  triangle-graph .Functor.F₁ {true} {true} _ = Precategory.id FinSets
  triangle-graph .Functor.F-id {false} = refl
  triangle-graph .Functor.F-id {true} = refl
  triangle-graph .Functor.F-∘ {true} {true} {true} f g = {!!}
  triangle-graph .Functor.F-∘ {false} {true} {true} f g = {!!}
  triangle-graph .Functor.F-∘ {false} {false} {true} f g = {!!}
  triangle-graph .Functor.F-∘ {false} {false} {false} f g = {!!}

  {-|
  For the triangle graph, conservation at v₁ (the interior vertex) requires:
    Φ(e₀) = Φ(e₁)

  i.e., the resource on incoming edge e₀ equals the resource on outgoing edge e₁.

  At v₀ (source): no incoming edges, one outgoing (e₀)
  At v₁ (interior): one incoming (e₀), one outgoing (e₁) → conservation: Φ(e₀) = Φ(e₁)
  At v₂ (sink): one incoming (e₁), no outgoing edges
  -}

  module TriangleConservation where
    open Lemma2∙9 example-category triangle-graph
    open Proposition2∙10 example-category triangle-graph
    open Lemma2∙3 example-category

    _ : Functor (ΣC[ 2 ]) (ΣC[ 3 ])
    _ = source-functor

    _ : Functor (ΣC[ 2 ]) (ΣC[ 3 ])
    _ = target-functor

    {-|
    The equalizer for the triangle graph selects edge-assignments Φ_E where
    the sum of outgoing resources equals sum of incoming resources at each vertex.

    This is a non-trivial constraint: Φ(e₀) must equal Φ(e₁) to satisfy
    conservation at the interior vertex v₁.
    -}
    _ : Functor Σeq (ΣC[ 2 ])
    _ = equalizer-inclusion
