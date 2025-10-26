# Combinatorial Species Implementation

## Overview

We've implemented combinatorial species in Agda using the 1Lab library for homotopy type theory. This provides a systematic, compositional way to construct directed graphs for neural networks.

## What Are Combinatorial Species?

A **combinatorial species** (Joyal, 1981) is a functor `F: FinSets → Sets` that represents "structures on finite sets". For a species F:
- `F[n]` = the set of all F-structures on an n-element set
- Morphisms transport structures along bijections (relabeling)

## Implementation Status

### ✅ Completed (as of 2025-10-16)

1. **Module**: `Neural.Combinatorial.Species`

2. **Basic Type**:
   ```agda
   Species : Type₁
   Species = Functor FinSets (Sets lzero)
   ```

3. **Fundamental Species**:
   - **ZeroSpecies**: No structures on any set (`F[n] = ⊥`)
   - **OneSpecies**: One structure on empty set only (`F[0] = ⊤, F[n>0] = ⊥`)
   - **XSpecies**: One structure on singleton sets (`F[1] = ⊤, F[n≠1] = ⊥`)

4. **Species Operations**:
   - **Sum** (`_⊕_`): `(F ⊕ G)[n] = F[n] ⊎ G[n]` - Either an F-structure OR a G-structure
   - **Product** (`_⊗_`): Partition into two parts (TODO)
   - **Composition** (`_∘ₛ_`): Hierarchical assembly (TODO)
   - **Derivative**: Pointed structures (TODO)

### 🚧 In Progress

- **Product species**: `(F ⊗ G)[n] = Σ(k ≤ n) (F[k] × G[n-k])`
- **Composition species**: Partition n into blocks, G-structure on each block, F-structure on blocks
- **Derivative**: `F'[n] = F[n+1]` with distinguished element
- **DirectedEdgeSpecies**: Structures on ordered pairs
- **DirectedGraphSpecies record**: Bundle vertex/edge species with source/target maps

### Key Technical Challenges Resolved

1. **H-Level preservation**: Import `Data.Sum.Properties` for automatic `H-Level-⊎` instance
2. **Empty type elimination**: Use `λ ()` for functions from `⊥`
3. **Impossible morphisms**: Non-bijective arrows (e.g., `Fin 0 → Fin 1`) require holes marked as impossible

## Connection to Neural Networks

### Why Species for Graphs?

1. **Compositional**: Build complex graphs from simple pieces
2. **Functorial**: Automatic handling of relabeling/isomorphisms
3. **Type-safe**: Impossible configurations ruled out by types
4. **Generative**: Species operations give graph construction recipes

### Example: Building a Neural Network Graph

```agda
-- Vertex species: label each neuron with activation function type
NeuronSpecies : Species

-- Edge species: weighted connections
EdgeSpecies : Species

-- Complete network as directed graph species
NetworkSpecies : DirectedGraphSpecies
NetworkSpecies .V = NeuronSpecies
NetworkSpecies .E = EdgeSpecies
NetworkSpecies .source = {- natural transformation -}
NetworkSpecies .target = {- natural transformation -}
```

### Future: Graph Combinators

Once complete, we can define:
- **Cycle species**: `C_k[n] = circular arrangements of n elements in k cycles`
- **Tree species**: `T[n] = rooted trees with n labeled vertices`
- **DAG species**: `D[n] = acyclic directed graphs on n vertices`
- **Feedforward species**: Layered networks with no cycles

These combinators will let us systematically generate neural architectures.

## Next Steps

1. ✅ Basic species (Zero, One, X)
2. ✅ Sum operation
3. 🚧 Product, composition, derivative operations
4. 🚧 DirectedGraphSpecies record
5. ⏳ Equivalence proof: `DirectedGraphSpecies ≃ DirectedGraph` (from `Neural.Base`)
6. ⏳ Example species: cycles, trees, DAGs
7. ⏳ Neural architecture generators

## References

- Joyal, A. (1981). "Une théorie combinatoire des séries formelles"
- Yorgey, B. (2014). "Combinatorial Species and Labelled Structures" (PhD thesis)
- 1Lab documentation: [Data.Sum.Properties](https://1lab.dev/Data.Sum.Properties.html)

## Files

- `/src/Neural/Combinatorial/Species.agda` - Main implementation
- `/src/Neural/Base.agda` - Original DirectedGraph definition
- `CLAUDE.md` - Project workflow documentation
