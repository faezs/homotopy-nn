{-|
# Graph Foundations

This module re-exports 1Lab's graph infrastructure for use in our neural network formalization.

**Key insight**: We use 1Lab's `Graph` (directed multigraphs) as the foundation,
then define oriented graphs as a full subcategory via predicates.

## What we get from 1Lab:

- `Graph o ℓ`: Directed multigraphs with Node and Edge
- `Graph-hom`: Graph homomorphisms (node + edge maps)
- `Graphs o ℓ`: The univalent category of graphs
- Products `_×ᴳ_`, terminal `⊤ᴳ`, pullbacks `_⊓ᴳ_`
- Exponentials `Graphs[_,_]`, cartesian closure

## Why this approach?

1. **Reuse infrastructure**: All 1Lab constructions automatically available
2. **Avoid K axiom**: 1Lab's Graph is proven correct
3. **Full subcategory**: OrientedGraphs inherit all structure
4. **Clean separation**: Graph theory vs topos theory

-}

module Neural.Graph.Base where

open import 1Lab.Prelude
open import 1Lab.HLevel

-- Core graph definitions
open import Cat.Instances.Graphs public

-- Graph limits (products, terminal, pullbacks)
open import Cat.Instances.Graphs.Limits public

-- Graph exponentials (function graphs)
open import Cat.Instances.Graphs.Exponentials public
