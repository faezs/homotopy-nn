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

### Section 5: Codes, Probabilities, and Information
- Neural.Code.Probabilities: Category Pf with fiberwise measures (Lemma 5.7)
- Neural.Code.Weighted: Weighted codes for linear neuron model (Lemma 5.10)
- Neural.Information.Shannon: Shannon entropy functor (Lemma 5.13)

### Section 6: Categorical Hopfield Dynamics
- Neural.Dynamics.Hopfield.Threshold: Categorical threshold non-linearity (Proposition 6.1)
- Neural.Dynamics.Hopfield.Discrete: Discrete dynamics (Definition 6.2, Lemmas 6.3-6.4)
- Neural.Dynamics.Hopfield.Classical: Classical Hopfield recovery (Lemmas 6.6-6.9)

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

-- Section 4: Networks with Computational Structures
import Neural.Computational.TransitionSystems

-- Section 5: Codes, Probabilities, and Information
import Neural.Code.Probabilities
import Neural.Code.Weighted
import Neural.Information.Shannon

-- Section 6: Categorical Hopfield Dynamics
import Neural.Dynamics.Hopfield
import Neural.Dynamics.Hopfield.Threshold
import Neural.Dynamics.Hopfield.Discrete
import Neural.Dynamics.Hopfield.Classical
