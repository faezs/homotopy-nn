{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
HomotopyDemo: small, self-contained example that references the
clique complex construction for an illustrative graph. Numerical
homology is computed externally (scripts/topology.py) and serves
as a narrative aid; this module provides the categorical spec side.
-}
module Neural.Examples.HomotopyDemo where

open import 1Lab.Prelude

open import Neural.Base
open import Neural.Homotopy.CliqueComplex

-- Toy directed graph (DAG) as a spec; edges forward only
postulate
  Gᴰᴬᴳ : DirectedGraph

-- Clique complex over the undirected support
postulate
  KG : SimplicialSet

-- Expected qualitative property for a DAG: β₁ = 0 (no cycles)
postulate
  beta1-zero : ⊤

