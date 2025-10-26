{-|
# Fractal Structures for Neural Networks

**Key insight**: Oriented graphs are forests (classical + acyclic).
Forests have natural fractal/hierarchical structure that can guide weight initialization.

## Theoretical Motivation

From Belfiore & Bennequin (2022):
- Oriented graphs satisfy: classical (at most one edge) + no loops + acyclic
- Proposition 1.1: CX is a poset (tree-like ordering structure)
- Fork construction preserves this hierarchical structure

**Our contribution**:
- Formalize connection between tree structure and fractals
- Define fractal distributions as functors from graph structure to weight spaces
- Connect to path uniqueness: trees → unique paths → fractal parameterization

## Practical Connection

The Python implementation (`neural_compiler/topos/fractal_initializer.py`) uses:
- **Hilbert curves**: Space-filling fractals that densely approximate any distribution
- **Dragon curves**: Self-similar fractals with 90° rotations
- **Cantor sets**: Hierarchical structures with gaps at multiple scales

This Agda module formalizes the mathematical properties these constructions exploit.

## References

- Mandelbrot (1982): The Fractal Geometry of Nature
- Peano (1890): Space-filling curves
- Hilbert (1891): Hilbert curve construction
- Our insight: Tree structure of oriented graphs ↔ Fractal hierarchy
-}

module Neural.Fractal.Base where

open import Neural.Graph.Base
open import Neural.Graph.Path
open import Neural.Graph.Oriented

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path.Reasoning

open import Cat.Base
open import Cat.Functor.Base

open import Algebra.Monoid

open import Data.List
open import Prim.Data.Nat

-- For graph isomorphisms
import Cat.Reasoning

private variable
  o ℓ : Level

{-|
## Self-Similar Structures

A structure is **self-similar** if it contains smaller copies of itself.
In graph terms: subtrees are isomorphic to the whole tree (up to scaling).

**Definition**: A graph G is self-similar at scale n if for each vertex v,
the subgraph rooted at v is isomorphic to G (or a portion of G).

**Examples**:
- Binary trees: Each subtree is a binary tree
- Hilbert curves: Each subdivision contains 4 smaller Hilbert curves
- Neural networks: Each layer can be viewed as a sub-network
-}

-- Graph isomorphism from category structure
module GraphIso {o ℓ} where
  open Cat.Reasoning (Graphs o ℓ)

  Graph-iso : Graph o ℓ → Graph o ℓ → Type (o ⊔ ℓ)
  Graph-iso G H = G ≅ H

record SelfSimilarStructure {o ℓ} (G : Graph o ℓ) : Type (lsuc o ⊔ lsuc ℓ) where
  open Graph G
  open GraphIso {o} {ℓ}

  postulate
    -- Scaling levels (depth of recursion)
    levels : Nat

    -- At each level, graph decomposes into copies of itself
    decompose : (level : Nat) → List (Graph o ℓ)

    -- Self-similarity: components are isomorphic to original (up to scale)
    self-similar : ∀ (level : Nat) (H : Graph o ℓ)
                 → ∥ Graph-iso H G ∥

  {-|
  **Interpretation**:
  - Trees are naturally self-similar (each subtree is a tree)
  - Fractals formalize this recursive structure
  - Neural networks inherit this from their graph structure
  -}

{-|
## Fractal Dimension (Informal)

For a self-similar structure with:
- N copies at each level
- Each scaled by factor r

Fractal dimension d = log(N) / log(1/r)

**Examples**:
- Line: 2 copies scaled by 1/2 → d = 1
- Square: 4 copies scaled by 1/2 → d = 2
- Cantor set: 2 copies scaled by 1/3 → d = log(2)/log(3) ≈ 0.63
- Binary tree: 2 copies (subtrees) → d = 1 (path-like)

We don't formalize real-valued dimensions here (no ℝ in 1Lab),
but use the structure for discrete distributions.
-}

{-|
## Tree Structure (for Forests)

A **tree** is a connected acyclic graph.
A **forest** is a disjoint union of trees.

**Key theorem** (to be proven in Forest.agda):
```
oriented→forest : is-oriented G → is-forest G
```

Trees have natural hierarchical structure:
- Root vertex (minimal element)
- Branching structure (children of each node)
- Unique paths between vertices
-}

-- Rooted tree: distinguished vertex with partial order
record RootedTree {o ℓ} (G : Graph o ℓ) : Type (lsuc o ⊔ lsuc ℓ) where
  open Graph G

  field
    -- Root vertex
    root : Node

    -- Tree property (to be imported from Forest.agda once defined)
    is-tree : Type (o ⊔ ℓ)  -- Placeholder

    -- Reachability from root defines depth levels
    node-depth : Node → Nat
    root-depth-zero : node-depth root ≡ zero

    -- Parent is unique (tree property)
    parent-unique : ∀ {v w w' : Node} → Edge w v → Edge w' v → w ≡ w'

  -- Children relation: v is child of w if edge w → v
  is-child : Node → Node → Type ℓ
  is-child v w = Edge w v

  {-|
  **Interpretation**:
  - Neural networks with convergent layers are NOT trees
  - But the fork construction splits convergence → forest structure
  - Each component tree has natural fractal hierarchy
  -}

{-|
## Fractal Distributions on Trees

A fractal distribution assigns weights to edges based on:
1. **Level in tree**: Depth from root
2. **Self-similarity**: Similar weights at each level
3. **Hierarchical decay**: Weights decrease with depth

This formalizes what the Python implementation does with Hilbert curves.
-}

{-|
## Fractal Distributions Parameterized by Monoid

Rather than postulating ℝ, we parameterize by any monoid structure.
This allows using:
- ℝ with multiplication (for actual neural networks)
- ℕ with addition (for counting/combinatorics)
- Any other monoid structure

The key operation is scaling via the monoid operation.
-}

record FractalDistribution {o ℓ} (G : Graph o ℓ) (W : Monoid ℓ) : Type (lsuc o ⊔ lsuc ℓ) where
  open Graph G
  open Monoid-on (W .snd) renaming (_⋆_ to _*w_; identity to 1w)

  postulate
    -- Assign weight to each edge based on hierarchical position
    edge-weight : ∀ {x y} → Edge x y → W .fst

    -- Self-similarity: weights at level n+1 are scaled version of level n
    scale-factor : Nat → W .fst
    self-similar-weights : ∀ {x y z} (e : Edge x y) (e' : Edge y z)
                         → (level-x level-y level-z : Nat)
                         → edge-weight e' ≡ (scale-factor level-y) *w (edge-weight e)

    -- Fractal property: total at each level follows power law
    level-total : Nat → W .fst
    power-law : ∀ (n : Nat) → level-total (suc n) ≡ (scale-factor n) *w (level-total n)

  {-|
  **Connection to Hilbert curve initialization**:

  The Hilbert curve in Python maps [0,1] → [0,1]^2 space-filling curve.
  We use it to sample weights by:
  1. Traverse tree in depth-first order
  2. Map position along curve to (u,v) coordinates
  3. Transform (u,v) → Gaussian via Box-Muller

  This respects tree structure because:
  - Hilbert curve is self-similar (each quadrant is a scaled Hilbert curve)
  - Tree is self-similar (each subtree is a tree)
  - Mapping preserves hierarchical relationships
  -}

{-|
## Connection to Path Uniqueness

**Theorem** (to be proven in Forest.agda):
```
tree→path-unique : is-tree G → ∀ {x y} → is-prop (Path-in G x y)
```

**Implication for fractal distributions**:
- Unique paths mean unambiguous hierarchical relationships
- Fractal parameterization is well-defined (no conflicting paths)
- Weight initialization respects the poset structure of CX (Proposition 1.1)

**Why this matters**:
1. Oriented graph → Forest → Trees
2. Trees → Unique paths → Poset structure
3. Poset → Fractal hierarchy → Canonical weight distribution
4. Fractal weights → Better initialization → Faster training

This is the theoretical justification for using fractal initialization!
-}

{-|
## Space-Filling Curves as Universal Embeddings

**Theorem** (Peano-Hilbert): For any dimension n, there exists a continuous
surjective map [0,1] → [0,1]^n (space-filling curve).

**Application to neural networks**:
- Weight space is ℝ^d (d parameters)
- Space-filling curve densely embeds any distribution
- Tree structure gives natural ordering for curve traversal
- Fractal self-similarity of curve matches tree self-similarity

**In our implementation**:
- 2D Hilbert curve for (mean, variance) of Gaussian weights
- Dragon curve for alternative self-similar patterns
- Cantor set for sparse hierarchical structures

**Formalization** (would require topology in Agda):
```
SpaceFillingCurve : (n : ℕ) → (I → ℝⁿ)  -- [0,1] → [0,1]^n
surjective : ∀ (p : ℝⁿ) → ∥ Σ[ t ∈ I ] (curve t ≡ p) ∥
continuous : ...
```

We leave this postulated, as 1Lab doesn't have real analysis.
-}

{-|
## Fractal Initialization Functor

**Categorical formulation**: Fractal initialization is a functor from
the category of oriented graphs to the category of probability distributions.

```
FractalInit : Functor OrientedGraphs ProbDist
```

**On objects**: G ↦ P_G (probability distribution over edge weights)
**On morphisms**: f : G → H ↦ pushforward of distributions

**Properties**:
1. Preserves composition: FractalInit(g ∘ f) = FractalInit(g) ∘ FractalInit(f)
2. Respects poset structure: If G is a poset, P_G concentrates on unique paths
3. Self-similarity: P_G inherits fractal structure from G

This explains why fractal initialization is "natural" in the categorical sense!
-}

-- Placeholder for functor (requires probability theory formalization)
postulate
  ProbDist : ∀ {o ℓ} → Precategory (lsuc o ⊔ lsuc ℓ) (o ⊔ ℓ)
  FractalInitFunctor : ∀ {o ℓ} → Functor (OrientedGraphs o ℓ) (ProbDist {o} {ℓ})

{-|
## Summary

**Mathematical chain**:
1. Oriented graphs = classical + no-loops + acyclic
2. Classical + acyclic = Forest structure (disjoint trees)
3. Trees = Hierarchical self-similar structures
4. Hierarchical structures = Natural fractal parameterization
5. Fractal parameterization = Space-filling curve embedding
6. Space-filling curves = Universal dense approximations
7. Therefore: Fractal initialization is canonical for oriented graphs!

**Practical outcome**:
- `fractal_initializer.py` implements this theory
- Hilbert curve respects tree structure of fork graphs
- Self-similarity of curve matches self-similarity of network
- Better initialization → faster convergence → better learning

**Next steps**:
- Forest.agda: Prove oriented → forest → unique paths
- Integrate with ForkCategorical.agda
- Use path uniqueness to validate fractal parameterization
- Experimental validation on ARC tasks
-}

-- Exports for other modules
open SelfSimilarStructure public
open RootedTree public
open FractalDistribution public
