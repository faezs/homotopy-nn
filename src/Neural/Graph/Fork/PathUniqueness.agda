{-|
# Path Uniqueness via Forest Structure

This module proves that paths in the fork graph Γ̄ are unique by using
the general forest structure theorem.

## Key Result

Γ̄ is oriented → Γ̄ is a forest → paths are unique
-}

module Neural.Graph.Fork.PathUniqueness where

open import Neural.Graph.Base
open import Neural.Graph.Oriented
open import Neural.Graph.Path
open import Neural.Graph.Forest
open import Neural.Graph.Fork.Fork
open import Neural.Graph.Fork.Surgery
open import Neural.Graph.Fork.Orientation

open import 1Lab.Prelude

open import Data.Dec.Base

private variable
  o ℓ : Level

module PathUniquenessProof
  (G : Graph o ℓ)
  (G-oriented : is-oriented G)
  (node-eq? : ∀ (x y : Graph.Node G) → Dec (x ≡ y))
  where

  open Graph G
  open ForkConstruction G G-oriented node-eq? using (ForkVertex; ForkEdge; ForkVertex-discrete; ForkVertex-is-set)
  open OrientationProofs G G-oriented node-eq?

  module Γ̄-PathUniqueness where
    open EdgePath Γ̄

    -- Main theorem: paths are unique
    path-unique : ∀ {v w} (p q : EdgePath v w) → p ≡ q
    path-unique {v} {w} p q = oriented-graph-path-unique Γ̄ Γ̄-oriented ForkVertex-discrete p q

    -- Categorical formulation: paths form propositions
    path-is-prop : ∀ {v w} → is-prop (EdgePath v w)
    path-is-prop {v} {w} = oriented-category-is-thin Γ̄ Γ̄-oriented ForkVertex-discrete {v} {w}

  -- Re-export for convenience
  Γ̄-path-is-prop : ∀ {v w} → is-prop (Path-in Γ̄ v w)
  Γ̄-path-is-prop = Γ̄-PathUniqueness.path-unique
