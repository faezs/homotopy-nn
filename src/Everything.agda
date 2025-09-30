{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Homotopy-Theoretic and Categorical Models of Neural Information Networks

This module imports all formalized components from Manin & Marcolli (2024).

## Organization

### Section 2: Network Summing Functors
- Neural.Base: Directed graphs as functors G: ·⇉· → FinSets
- Neural.Code: Neural codes and simplicial complexes
- Neural.SummingFunctor: Summing functors Φ: P(X) → C and ΣC(X) category
- Neural.Network.Conservation: Conservation laws at vertices (Lemma 2.9, Prop 2.10)
- Neural.Network.Grafting: Properad-constrained functors (Lemma 2.19, Corollary 2.20)

### Section 3: Neural Information and Resources
- Neural.Information: Neural codes, firing rates, metabolic efficiency
- Neural.Resources: Resource theory as symmetric monoidal category
- Neural.Resources.Convertibility: Conversion rates and measuring homomorphisms

## Type Checking

To type-check the entire project:
```bash
agda --library-file=./libraries src/Everything.agda
```
-}

module Everything where

-- Section 2: Network Summing Functors
import Neural.Base
import Neural.Code
import Neural.SummingFunctor
import Neural.Network.Conservation
import Neural.Network.Grafting

-- Section 3: Neural Information and Resources
import Neural.Information
import Neural.Resources
import Neural.Resources.Convertibility
import Neural.Resources.Optimization
