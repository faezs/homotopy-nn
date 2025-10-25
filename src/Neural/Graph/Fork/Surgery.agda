{-|
# Fork Surgery - Graph Construction

This module constructs the augmented fork graph Γ̄ from an oriented graph Γ.

## From the Paper (Belfiore & Bennequin 2022, Section 1.3)

> "At each layer a where more than one layer sends information...
> we perform a surgery: between a and a' introduce two new objects A★ and A,
> with arrows a' → A★, A★ → A, a → A, forming a fork."
-}

module Neural.Graph.Fork.Surgery where

open import Neural.Graph.Base
open import Neural.Graph.Oriented
open import Neural.Graph.Fork.Fork
open import Neural.Graph.Fork.Convergence

open import 1Lab.Prelude

open import Data.Dec.Base
open import Data.List

private variable
  o ℓ o' ℓ' : Level

module ForkSurgery
  (G : Graph o ℓ)
  (G-oriented : is-oriented G)
  (node-eq? : ∀ (x y : Graph.Node G) → Dec (x ≡ y))
  where

  open Graph G
  open ForkConstruction G G-oriented node-eq?

  {-|
  ## Γ̄ as 1Lab Graph
  
  Package ForkVertex and ForkEdge as a proper 1Lab `Graph`.
  
  **This gives us:**
  - Graph homomorphisms (Graph-hom)
  - Category structure (Graphs o ℓ)
  - Limits and exponentials (if needed)
  - Universal properties
  -}
  
  Γ̄ : Graph o (o ⊔ ℓ)
  Γ̄ .Graph.Node = ForkVertex
  Γ̄ .Graph.Edge = ForkEdge
  Γ̄ .Graph.Node-set = ForkVertex-is-set
  Γ̄ .Graph.Edge-set = ForkEdge-is-set
  
  {-|
  ## Graph Homomorphisms
  
  Graph homomorphisms from/to Γ̄ are just 1Lab's `Graph-hom`.
  -}
  
  Γ̄-hom : (H : Graph o' ℓ') → Type (o ⊔ ℓ ⊔ o' ⊔ ℓ')
  Γ̄-hom H = Graph-hom Γ̄ H

