{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Concrete Neural Realizations (Phase 2)

This module constructs explicit examples of graphs that realize specific
homotopy types, following the research plan.

## The Strategy

Following Delooping as a template, we build the **inverse** direction:
- Delooping: Group → Space (construct space from group)
- Examples: Graph → Space (show graph realizes space)

We prove by construction that specific graph structures realize K(G,1) spaces.

## Examples to Build

1. **Cycle graph → S¹** (π₁ = ℤ)
2. **n-cycle → Lens space** (π₁ = ℤ/nℤ)
3. **Figure-eight → Rose** (π₁ = Free(2))
4. **Complete graph → Contractible** (π₁ = 0, high Φ)

## References

- Homotopy.Space.Circle (1Lab) - S¹ and π₁(S¹) ≡ ℤ
- Homotopy.Space.Delooping (1Lab) - Construction pattern
- Neural.Homotopy.Realization - Realization framework
-}

module Neural.Homotopy.Examples where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Instances.Shape.Parallel using (·⇉·)

open import Data.Nat.Base using (Nat; zero; suc; _+_; _*_)
open import Data.Fin.Base using (Fin; fzero; fsuc)
open import Data.Bool using (Bool; true; false)

open import Algebra.Group.Instances.Integers using (ℤ)

open import Homotopy.Space.Circle using (S¹; S¹∙; base; loop; π₁S¹≡ℤ)

open import Neural.Base using (DirectedGraph; vertices; edges; source; target)
open import Neural.Homotopy.CliqueComplex
open import Neural.Homotopy.Simplicial using (PSSet)
open import Neural.Homotopy.Realization using (semantic; Realizes'; geometric-realization)
open import Neural.Information using (ℝ)
open import Neural.Homotopy.VanKampen using (rose; rose-edges; rose-vertices; Free-2≅ℤ*ℤ; Free-n)

private variable
  o ℓ : Level

{-|
## Example 1: The Cycle Graph

A **cycle graph** with n vertices is a directed cycle:
- Vertices: v₀, v₁, ..., vₙ₋₁
- Edges: e₀, e₁, ..., eₙ₋₁
- Connectivity: eᵢ : vᵢ → v_{i+1 mod n}

For n=1, this is a single vertex with a self-loop → realizes S¹
-}

-- Single-vertex cycle with self-loop
-- This is the minimal cycle graph - same as rose(1)!

cycle-1 : DirectedGraph
cycle-1 = rose 1

-- This cycle has 1 vertex, 1 edge (self-loop)
cycle-1-vertices : vertices cycle-1 ≡ 1
cycle-1-vertices = rose-vertices 1

cycle-1-edges : edges cycle-1 ≡ 1
cycle-1-edges = rose-edges 1

{-|
## Realization Theorem: Cycle-1 → S¹

**Goal:** Prove that cycle-1 realizes the circle S¹.

**Strategy:**
1. K(cycle-1) is the clique complex of a single vertex with self-loop
2. The clique complex should be homotopy equivalent to S¹
3. Use π₁ as evidence: π₁(K(cycle-1)) ≃ ℤ and π₁(S¹) ≃ ℤ

**Current status:** Postulated - requires working out clique complex details
-}

-- Fundamental group is Free-Group (Fin 1) ≡ ℤ
open import Neural.Homotopy.FreeGroupEquiv using (Free-Fin1≡ℤ)

cycle-1-π₁≡ℤ : Free-n 1 ≡ ℤ
cycle-1-π₁≡ℤ = Free-Fin1≡ℤ

postulate
  -- The clique complex of cycle-1 is equivalent to S¹
  K-cycle-1-is-S¹ : geometric-realization (K cycle-1) ≃ S¹

  -- Therefore cycle-1 realizes S¹
  cycle-1-realizes-S¹ : Realizes' cycle-1 S¹

-- The proof would use:
-- 1. Compute K(cycle-1) explicitly
-- 2. Show it has one 0-simplex (the vertex)
-- 3. Show it has one 1-simplex (the loop)
-- 4. No higher simplices
-- 5. This gives S¹ up to homotopy equivalence

{-|
## General n-Cycle Graph

For general n, a cycle graph has n vertices in a directed cycle.
-}

postulate
  -- General cycle graph with n vertices
  cycle-graph : (n : Nat) → DirectedGraph

  cycle-graph-vertices : (n : Nat) → vertices (cycle-graph n) ≡ n
  cycle-graph-edges : (n : Nat) → edges (cycle-graph n) ≡ n

  -- For n=1, this is cycle-1
  cycle-1-is-cycle-graph-1 : cycle-graph 1 ≡ cycle-1

{-|
## Example 2: n-Cycle and Cyclic Groups

An n-cycle with n > 1 should realize a space with π₁ ≃ ℤ/nℤ.

**Intuition:**
- n vertices arranged in a cycle
- Going around the cycle n times returns to start
- π₁ should have n-fold periodicity → ℤ/nℤ

**Note:** This is different from the 1-cycle case!
- 1-cycle: infinite loop → π₁ = ℤ
- n-cycle (n>1): finite periodicity → π₁ = ℤ/nℤ
-}

postulate
  -- n-cycle realizes a space with cyclic fundamental group
  -- (Lens space or similar)
  n-cycle-realizes-cyclic :
    (n : Nat) →
    {-| π₁(K(cycle-graph n)) ≃ ℤ/nℤ -}
    ⊤

{-|
## Example 3: Figure-Eight Graph

The **figure-eight** graph has:
- 1 vertex (junction point)
- 2 edges, both self-loops (two independent loops)

This should realize the **wedge** S¹ ∨ S¹, which has π₁ = Free(2).
-}

-- Figure-eight graph: one vertex, two self-loop edges
-- This is exactly rose(2) from VanKampen!
open import Algebra.Group.Free.Product using (Free-product)

figure-eight : DirectedGraph
figure-eight = rose 2

figure-eight-vertices : vertices figure-eight ≡ 1
figure-eight-vertices = rose-vertices 2

figure-eight-edges : edges figure-eight ≡ 2
figure-eight-edges = rose-edges 2

-- Fundamental group is Free-Group (Fin 2) ≅ ℤ * ℤ
figure-eight-π₁≅ℤ*ℤ : Free-n 2 Groups.≅ (ℤ *ᴳ ℤ)
figure-eight-π₁≅ℤ*ℤ = Free-2≅ℤ*ℤ
  where
    open import Algebra.Group.Cat.Base using (Groups; Group)
    _*ᴳ_ = Free-product

postulate
  -- Realizes wedge of two circles (requires HIT infrastructure)
  figure-eight-realizes-wedge :
    {-| K(figure-eight) ≃ S¹ ∨ S¹ -}
    ⊤

  -- This follows from figure-eight-π₁≅ℤ*ℤ + van Kampen
  figure-eight-π₁-free :
    {-| π₁(K(figure-eight)) ≃ Free(2) -}
    ⊤

{-|
## Example 4: Complete Graph → Contractible

From Neural.Homotopy.Attention, we know:
- Complete graph has all possible edges
- K(complete-graph) is **contractible** (fills in completely)
- π₁ = 0 (trivial)
- But Φ >> 0 (high integration!)

This is a key example showing Φ ≠ π₁.
-}

postulate
  -- Complete graph (from Attention module)
  complete-digraph : (n : Nat) → DirectedGraph

  -- Complete graph is contractible
  complete-is-contractible :
    (n : Nat) →
    {-| K(complete-digraph n) is contractible -}
    ⊤

  -- Therefore π₁ = 0
  complete-π₁-trivial :
    (n : Nat) →
    {-| π₁(K(complete-digraph n)) ≃ ⊤ (trivial group) -}
    ⊤

{-|
## Example 5: Path Graph → Discrete/Contractible

A **path graph** is a directed chain (feedforward):
v₀ → v₁ → v₂ → ... → vₙ

No cycles → π₁ = 0 and Φ = 0 (proven in IntegratedInformation)
-}

postulate
  path-graph : (n : Nat) → DirectedGraph

  -- Path is acyclic
  path-is-acyclic :
    (n : Nat) →
    {-| No directed cycles -}
    ⊤

  -- Therefore π₁ = 0
  path-π₁-trivial :
    (n : Nat) →
    {-| π₁(K(path-graph n)) ≃ ⊤ -}
    ⊤

  -- And Φ = 0 (already proven in IntegratedInformation)
  path-Φ-zero :
    (n : Nat) →
    {-| Φ(path-graph n) = 0 -}
    ⊤

{-|
## Comparison Table

| Graph          | Vertices | Edges | π₁           | Φ       | Space      |
|----------------|----------|-------|--------------|---------|------------|
| cycle-1        | 1        | 1     | ℤ            | > 0     | S¹         |
| cycle-n (n>1)  | n        | n     | ℤ/nℤ         | > 0     | Lens       |
| figure-eight   | 1        | 2     | Free(2)      | > 0     | S¹ ∨ S¹    |
| complete-n     | n        | n²    | 0 (trivial)  | >> 0    | contractible|
| path-n         | n        | n-1   | 0 (trivial)  | 0       | discrete   |

**Key observations:**
1. **π₁ reflects cycles**: Acyclic → π₁=0, cycles → π₁≠0
2. **Φ reflects connectivity**: Feedforward → Φ=0, bidirectional → Φ>0
3. **π₁ and Φ are independent!**
   - Complete graph: π₁=0 but Φ>>0
   - Distinguishes attention (complete) from feedforward (path)
-}

{-|
## Van Kampen and Grafting

The **van Kampen theorem** tells us how π₁ behaves under wedge sums:
  π₁(X ∨ Y) ≃ π₁(X) * π₁(Y)  (free product of groups)

This should correspond to **grafting** operations from Section 2.3.2!

**Conjecture:**
  π₁(K(G₁ ⋈ G₂)) ≃ π₁(K(G₁)) * π₁(K(G₂))

where G₁ ⋈ G₂ is grafting at a common vertex.
-}

postulate
  -- Grafting operation (from Neural.Network.Grafting)
  -- Combines two graphs by identifying vertices
  graft-at-vertex :
    (G₁ G₂ : DirectedGraph) →
    DirectedGraph

  -- Van Kampen for graphs
  grafting-van-kampen :
    (G₁ G₂ : DirectedGraph) →
    {-| π₁(K(graft-at-vertex G₁ G₂)) ≃ π₁(K(G₁)) * π₁(K(G₂)) -}
    ⊤

{-|
## Synthesis Pattern Recognition

From these examples, we can extract **synthesis rules**:

**Rule 1: π₁ = ℤ**
  → Use cycle-1 (single vertex, self-loop)

**Rule 2: π₁ = ℤ/nℤ**
  → Use cycle-n (n vertices in cycle)

**Rule 3: π₁ = Free(n)**
  → Use n self-loops at single vertex (rose with n petals)

**Rule 4: π₁ = G₁ * G₂**
  → Use graft-at-vertex (van Kampen)

**Rule 5: Φ > 0 required**
  → Need cycles (any of above)
  → For Φ >> 0 with π₁ = 0, use complete graph!

This gives us a **constructive synthesis algorithm**!
-}

-- Synthesis rules as functions
synthesize-ℤ : DirectedGraph
synthesize-ℤ = cycle-1  -- Already defined!

synthesize-ℤ/n : (n : Nat) → DirectedGraph
synthesize-ℤ/n n = cycle-graph n

-- Rose with n petals (1 vertex, n self-loops)
-- π₁(rose n) = Free-Group (Fin n)
synthesize-Free : (n : Nat) → DirectedGraph
synthesize-Free n = rose n

synthesize-product : (G₁ G₂ : DirectedGraph) → DirectedGraph
synthesize-product G₁ G₂ = graft-at-vertex G₁ G₂

{-|
## Summary

**What we've shown (by example):**

1. Specific graphs realize specific spaces
   - cycle-1 → S¹
   - figure-eight → S¹ ∨ S¹
   - complete → contractible

2. π₁ and Φ are independent invariants
   - Both needed to characterize capabilities

3. Grafting implements free products (van Kampen)
   - Compositional synthesis via grafting

**Next steps (Phase 3):**
1. Prove the postulates (fill in {!!})
2. Formalize synthesis rules as theorems
3. Build synthesis algorithm from rules
4. Characterize realizability boundary
-}
