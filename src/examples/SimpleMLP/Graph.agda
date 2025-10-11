{-# OPTIONS --no-import-sorts #-}

{-|
# Simple MLP Oriented Graph (Section 1.1)

Concrete implementation of DirectedGraph for simple MLP.
This is the actual functor G: ·⇉· → FinSets.

Graph structure:
- 3 vertices: V = {0, 1, 2} (input, hidden, output)
- 2 edges: E = {0, 1}
- source: {0 ↦ 0, 1 ↦ 1}  (edge 0 from vertex 0, edge 1 from vertex 1)
- target: {0 ↦ 1, 1 ↦ 2}  (edge 0 to vertex 1, edge 1 to vertex 2)

Topology: 0 --e0--> 1 --e1--> 2
-}

module examples.SimpleMLP.Graph where

open import 1Lab.Prelude
open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin; fzero; fsuc; Fin-cases; Fin-elim)
open import Data.Bool.Base using (Bool; true; false)

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Instances.Shape.Parallel using (·⇉·)
open import Cat.Instances.FinSets using (FinSets)

open import Neural.Base using (DirectedGraph)

--------------------------------------------------------------------------------
-- Simple MLP Graph Structure
--------------------------------------------------------------------------------

{-|
Define the graph as a functor G: ·⇉· → FinSets.

Recall ·⇉· has:
- Objects: false (edges), true (vertices)
- Morphisms: false→true (source, target), identities
-}

simple-mlp-graph : DirectedGraph
simple-mlp-graph .Functor.F₀ false = 2   -- 2 edges
simple-mlp-graph .Functor.F₀ true  = 3   -- 3 vertices

-- Source function: edges → vertices
-- edge 0 → vertex 0 (input)
-- edge 1 → vertex 1 (hidden)
simple-mlp-graph .Functor.F₁ {false} {true} true =
  Fin-cases fzero (λ _ → fsuc fzero)

-- Target function: edges → vertices
-- edge 0 → vertex 1 (hidden)
-- edge 1 → vertex 2 (output)
simple-mlp-graph .Functor.F₁ {false} {true} false =
  Fin-cases (fsuc fzero) (λ _ → fsuc (fsuc fzero))

-- Identity on edges
simple-mlp-graph .Functor.F₁ {false} {false} tt = λ e → e

-- Identity on vertices
simple-mlp-graph .Functor.F₁ {true} {true} tt = λ v → v

-- Functor laws: F(id) = id
simple-mlp-graph .Functor.F-id {false} = refl
simple-mlp-graph .Functor.F-id {true} = refl

-- Functor laws: F(f ∘ g) = F(f) ∘ F(g)
-- TODO: Complete these proofs properly
simple-mlp-graph .Functor.F-∘ {false} {false} {false} f g = {!!}
simple-mlp-graph .Functor.F-∘ {false} {false} {true} f g = {!!}
simple-mlp-graph .Functor.F-∘ {false} {true} {true} f g = refl
simple-mlp-graph .Functor.F-∘ {true} {true} {true} f g = refl

--------------------------------------------------------------------------------
-- Verification
--------------------------------------------------------------------------------

{-|
Verify the graph structure matches our specification.
-}

-- 3 vertices
_ : simple-mlp-graph .Functor.F₀ true ≡ 3
_ = refl

-- 2 edges
_ : simple-mlp-graph .Functor.F₀ false ≡ 2
_ = refl

-- Source of edge 0 is vertex 0
_ : simple-mlp-graph .Functor.F₁ {false} {true} true fzero ≡ fzero
_ = refl

-- Target of edge 0 is vertex 1
_ : simple-mlp-graph .Functor.F₁ {false} {true} false fzero ≡ fsuc fzero
_ = refl

-- Source of edge 1 is vertex 1
_ : simple-mlp-graph .Functor.F₁ {false} {true} true (fsuc fzero) ≡ fsuc fzero
_ = refl

-- Target of edge 1 is vertex 2
_ : simple-mlp-graph .Functor.F₁ {false} {true} false (fsuc fzero) ≡ fsuc (fsuc fzero)
_ = refl

--------------------------------------------------------------------------------
-- Export for use in other modules
--------------------------------------------------------------------------------

open import Neural.Base using (vertices; edges; source; target)

-- Convenient accessors
mlp-vertices : Nat
mlp-vertices = vertices simple-mlp-graph

mlp-edges : Nat
mlp-edges = edges simple-mlp-graph

mlp-source : Fin mlp-edges → Fin mlp-vertices
mlp-source = source simple-mlp-graph

mlp-target : Fin mlp-edges → Fin mlp-vertices
mlp-target = target simple-mlp-graph

-- Explicit proofs
mlp-vertices-is-3 : mlp-vertices ≡ 3
mlp-vertices-is-3 = refl

mlp-edges-is-2 : mlp-edges ≡ 2
mlp-edges-is-2 = refl
