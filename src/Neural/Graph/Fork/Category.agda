{-|
# Fork Categorical Structure

This module provides the complete categorical structure for fork graphs,
re-exporting all components from the modular Fork/* hierarchy.

## Module Organization

- **Fork/Fork.agda**: Core datatypes (ForkVertex, ForkEdge, ForkPath HIT)
- **Fork/Convergence.agda**: Convergence detection for vertex surgery
- **Fork/Surgery.agda**: Γ̄ graph construction from oriented graph Γ
- **Fork/Orientation.agda**: Proof that Γ̄ is oriented (classical, no-loops, acyclic)
- **Fork/PathUniqueness.agda**: Path uniqueness via forest structure
- **Fork/Poset.agda**: Reduced poset X (non-star vertices)

## Paper Reference

Belfiore & Bennequin (2022), Section 1.3: Fork construction for convergent layers
-}

module Neural.Graph.Fork.Category where

open import Neural.Graph.Base
open import Neural.Graph.Oriented
open import Neural.Graph.Path
open import Neural.Graph.Forest
open import Neural.Graph.Fork.Fork
open import Neural.Graph.Fork.Convergence
open import Neural.Graph.Fork.Surgery
open import Neural.Graph.Fork.Orientation
open import Neural.Graph.Fork.PathUniqueness
open import Neural.Graph.Fork.Poset

open import 1Lab.Prelude

open import Data.Dec.Base
open import Data.List

private variable
  o ℓ : Level

{-|
## Main Module Interface

This module wraps all fork construction components with a unified parameterized interface.
-}

module ForkCategoricalStructure
  (G : Graph o ℓ)
  (G-oriented : is-oriented G)
  (nodes : List (Graph.Node G))
  (nodes-complete : ∀ (n : Graph.Node G) → n ∈ nodes)
  (edge? : ∀ (x y : Graph.Node G) → Dec (Graph.Edge G x y))
  (node-eq? : ∀ (x y : Graph.Node G) → Dec (x ≡ y))
  where

  open Graph G

  -- Re-export core fork datatypes
  open ForkConstruction G G-oriented node-eq? public

  -- Re-export convergence detection
  open ConvergenceDetection G G-oriented nodes nodes-complete edge? node-eq? public

  -- Re-export surgery (Γ̄ construction)
  open ForkSurgery G G-oriented node-eq? public

  -- Re-export orientation proofs (hide Γ̄ since we already have it from Surgery)
  open OrientationProofs G G-oriented node-eq? public hiding (Γ̄; Γ̄-hom)

  -- Re-export path uniqueness
  open PathUniquenessProof G G-oriented node-eq? public

  -- Re-export reduced poset structure
  open ForkPosetDefs G G-oriented nodes nodes-complete edge? node-eq? public

{-|
## Simplified Interface (3 parameters)

For modules that don't need convergence detection, provide a simpler interface.
-}

module ForkCategoricalSimple
  (G : Graph o ℓ)
  (G-oriented : is-oriented G)
  (node-eq? : ∀ (x y : Graph.Node G) → Dec (x ≡ y))
  where

  open Graph G

  -- Core datatypes
  open ForkConstruction G G-oriented node-eq? public

  -- Γ̄ construction
  open ForkSurgery G G-oriented node-eq? public

  -- Orientation proofs (hide Γ̄ since we already have it from Surgery)
  open OrientationProofs G G-oriented node-eq? public hiding (Γ̄; Γ̄-hom)

  -- Path uniqueness
  open PathUniquenessProof G G-oriented node-eq? public
