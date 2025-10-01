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

### Section 7: Homotopy-Theoretic Models
- Neural.Homotopy.Simplicial: Pointed simplicial sets, smash product, suspension, spheres
- Neural.Homotopy.GammaSpaces: Segal's Gamma-spaces, connective spectra, infinite loop spaces
- Neural.Homotopy.CliqueComplex: Clique complexes K(G), nerve construction, Betti numbers
- Neural.Homotopy.GammaNetworks: Gamma networks, composition, stable homotopy invariants
- Neural.Homotopy: Main re-export module for Section 7

### Section 8: Information Geometry and Integrated Information
- Neural.Information.Geometry: Divergences, Fisher-Rao metric, geodesics, natural gradient (§8.1)
- Neural.Information.Cohomology: Cohomological information, integrated information Φ (§8.2-8.4)
- Neural.Information.RandomGraphs: Graph information structures, Gamma networks from probability (§8.6)
- Neural.Dynamics.IntegratedInformation: Φ for Hopfield networks, feedforward Lemma 8.1 (§8.2-8.5)
- Neural.Homotopy.UnifiedCohomology: Generalized cohomology, spectra, unified framework (§8.7)

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

-- Section 7: Homotopy-Theoretic Models
import Neural.Homotopy
import Neural.Homotopy.Simplicial
import Neural.Homotopy.GammaSpaces
import Neural.Homotopy.CliqueComplex
import Neural.Homotopy.GammaNetworks

-- Section 8: Information Geometry and Integrated Information
import Neural.Information.Geometry
import Neural.Information.Cohomology
import Neural.Information.RandomGraphs
import Neural.Dynamics.IntegratedInformation
import Neural.Homotopy.UnifiedCohomology

-- Examples and Demonstrations
import Neural.Examples.ConsciousnessDemo
