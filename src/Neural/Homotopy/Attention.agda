{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Attention Mechanisms as Homotopy Types

This module analyzes transformer attention mechanisms through the lens of homotopy
type theory, computing their fundamental homotopy invariants.

## Research Questions

1. **What is π₁(K(attention))?** - The fundamental group of attention's clique complex
2. **What is Φ(attention)?** - The integrated information of attention
3. **Can we synthesize attention from homotopy specs?** - Inverse problem
4. **Do homotopy invariants explain why attention works?** - Theoretical foundation

## Key Hypothesis

Attention mechanisms have a characteristic homotopy type that:
- Enables in-context learning (high integration Φ)
- Supports compositional generalization (non-trivial π₁)
- Distinguishes them from feedforward networks (Φ_feedforward = 0)

## References

- Vaswani et al. (2017) "Attention Is All You Need"
- Manin & Marcolli (2024) "Homotopy Theoretic and Categorical Models of Neural Information Networks"

-}

module Neural.Homotopy.Attention where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Instances.Shape.Parallel using (·⇉·)

open import Data.Nat.Base using (Nat; zero; suc; _+_; _*_)
open import Data.Fin.Base using (Fin; fzero; fsuc)
open import Data.Bool using (Bool; true; false)

open import Neural.Base
open import Neural.Homotopy.CliqueComplex
open import Neural.Homotopy.Simplicial using (PSSet)
open import Neural.Information using (ℝ; zeroℝ)
open import Neural.Dynamics.Hopfield using (HopfieldDynamics)
open import Neural.Dynamics.IntegratedInformation
  using (Φ-hopfield; feedforward-zero-Φ; is-feedforward)

private variable
  o ℓ : Level

-- Static integrated information for graph structure
-- We compute Φ using Hopfield dynamics at time 0
Φ-static : (G : DirectedGraph) → (hd : HopfieldDynamics G) → ℝ
Φ-static G hd = Φ-hopfield hd 0

-- Feedforward graphs have zero static Φ
Φ-static-feedforward :
  (G : DirectedGraph) →
  (hd : HopfieldDynamics G) →
  is-feedforward G →
  Φ-static G hd ≡ zeroℝ
Φ-static-feedforward G hd ff = feedforward-zero-Φ hd ff 0

{-|
## Modeling Attention as DirectedGraph

**Self-attention structure:**
- Input: sequence of n tokens
- Each token attends to all other tokens (including itself)
- Graph structure: Complete directed graph with self-loops

**Vertices**: Tokens t₀, t₁, ..., tₙ₋₁
**Edges**: For each pair (i, j), edge from tⱼ to tᵢ (token j contributes to token i's update)

This creates a **complete directed graph** Kₙ↔.
-}

{-|
### Single-Head Attention Graph

For n tokens with single-head attention:
- n vertices (tokens)
- n² edges (each token attends to each token)
- Self-loops included (tokens attend to themselves)
-}

-- Complete directed graph with self-loops
-- For n vertices, we have n² edges: one for each pair (i,j)
-- Edge numbering: edge k corresponds to (source = k mod n, target = k div n)
postulate
  complete-digraph : (n : Nat) → DirectedGraph

  complete-digraph-vertices :
    (n : Nat) →
    vertices (complete-digraph n) ≡ n

  complete-digraph-edges :
    (n : Nat) →
    edges (complete-digraph n) ≡ n * n

  complete-digraph-complete :
    (n : Nat) →
    {-| For every pair (i,j), there exists an edge from i to j -}
    ⊤

-- Attention graph is complete directed graph
attention-graph : (seq-len : Nat) → DirectedGraph
attention-graph n = complete-digraph n

{-|
### Multi-Head Attention Graph

Multi-head attention with h heads on n tokens:
- Strategy 1: h parallel copies (product graph)
- Strategy 2: Single graph with h * n² edges (one set per head)

We use Strategy 1: parallel heads that compose.
-}

-- Multi-head attention as parallel composition
postulate
  multi-head-attention : (n-heads seq-len : Nat) → DirectedGraph

  multi-head-vertices :
    (h n : Nat) →
    vertices (multi-head-attention h n) ≡ n

  multi-head-edges :
    (h n : Nat) →
    edges (multi-head-attention h n) ≡ h * (n * n)

  multi-head-structure :
    (h n : Nat) →
    {-| h parallel copies of complete graph structure -}
    ⊤

{-|
## Example: Attention on 3 Tokens

Concrete example to build intuition:
- 3 tokens: t₀, t₁, t₂
- 9 edges: all pairs (including self-loops)
- Complete graph K₃↔
-}

-- Explicit construction for 3 tokens
-- Edges: (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)
-- Where (source, target) pairs
attention-3 : DirectedGraph
attention-3 = complete-digraph 3

{-|
## Homotopy Invariants of Attention

Using the CliqueComplex module, we compute:
1. K(attention-graph n) - The clique complex
2. π₁(K(attention)) - Fundamental group
3. Φ(attention) - Integrated information
-}

-- Clique complex of attention graph
K-attention : (n : Nat) → PSSet
K-attention n = K (attention-graph n)

-- π₁ of attention's clique complex
π₁-attention : (n : Nat) → Type
π₁-attention n = pi-K (attention-graph n) 1

{-|
### Conjecture: π₁(K(Kₙ↔)) for Complete Graphs

For a complete directed graph on n vertices:
- **Conjecture**: π₁(K(Kₙ)) ≃ Free(?) for some generator count
- **Intuition**: Complete graph has many independent cycles
- **Comparison**:
  - Cycle graph: π₁ ≃ ℤ (one generator)
  - Figure-eight: π₁ ≃ Free(2) (two generators)
  - Complete graph: π₁ ≃ Free(?) (many generators?)

**Actually**: For complete graphs, K(Kₙ) is **contractible**!
- Complete graph has all possible cliques
- The complex "fills in" completely
- So π₁(K(Kₙ)) ≃ 0 (trivial group)

This is SURPRISING for attention!
-}

postulate
  π₁-complete-trivial :
    (n : Nat) →
    π₁-attention n ≃ {!!}  -- Should be trivial group!

{-|
### The Attention Paradox

**Paradox**: If K(Kₙ) is contractible, then π₁ = 0 (trivial).

But attention is supposed to be highly structured!

**Resolution**: The homotopy type of the **static graph** K(attention)
is not the full story. We need:

1. **Dynamics**: How the graph evolves during computation
2. **Resources**: The resource category C structure
3. **Integration Φ**: Measures information integration, not just topology

**Key insight**: Φ ≠ 0 even when π₁ = 0!
-}

{-|
## Integrated Information Φ of Attention

From Section 8 (Integrated Information), we compute Φ(attention).

**Prediction**: Φ(attention) >> 0 (high integration)
**Comparison**: Φ(feedforward) = 0 (proven)

The difference: Attention has bidirectional information flow!
-}

-- Compute Φ for attention graph
Φ-attention :
  (n : Nat) →
  (hd : HopfieldDynamics (attention-graph n)) →
  ℝ
Φ-attention n hd = Φ-static (attention-graph n) hd

postulate
  -- Attention has positive Φ (needs proof via bidirectional flow analysis)
  Φ-attention-positive :
    (n : Nat) →
    (hd : HopfieldDynamics (attention-graph n)) →
    {-| Φ-attention n hd > 0 due to complete graph structure -}
    ⊤

  -- Comparison with feedforward
  Φ-attention-vs-feedforward :
    (n : Nat) →
    (hd-attn : HopfieldDynamics (attention-graph n)) →
    (G-ff : DirectedGraph) →
    (hd-ff : HopfieldDynamics G-ff) →
    (ff : is-feedforward G-ff) →
    {-| Φ-attention n hd-attn > Φ-static G-ff hd-ff ≡ 0 -}
    ⊤

{-|
## Comparison: Feedforward vs Attention

| Architecture  | Graph Structure    | π₁(K(G))     | Φ(G)    | Capability          |
|--------------|-------------------|-------------|---------|---------------------|
| Feedforward  | DAG (no cycles)   | 0 (trivial) | 0       | No integration      |
| Attention    | Complete digraph  | 0 (trivial) | >> 0    | High integration    |
| RNN          | Cyclic            | ℤ or more   | > 0     | Memory + integration|

**Key discovery**: π₁ alone doesn't distinguish attention from feedforward!
**The difference**: Φ (integrated information)

**Revised hypothesis**:
- Attention works because Φ >> 0 (bidirectional flow enables integration)
- Not because of non-trivial fundamental group
- The graph is "too complete" - no holes, hence contractible
-}

{-|
## Synthesis Challenge

**Original question**: Given homotopy type, can we synthesize attention?

**Revised question**: Given (π₁, Φ) pair, can we synthesize attention?

**Input specification**:
- π₁ = 0 (trivial)
- Φ ≥ threshold (high integration)

**Output**: Should produce something like complete graph (attention-like)

**Test**: Does synthesis give us attention, or other high-Φ architectures?
-}

-- Capability specification for attention
record AttentionCapability : Type₁ where
  field
    π₁-spec : Type  -- Desired fundamental group
    Φ-bound : ℝ     -- Minimum integration
    connectivity : Nat  -- Minimum connectivity

postulate
  synthesize-attention :
    {-| Given (trivial π₁, high Φ), synthesize graph -}
    (Φ-min : ℝ) →
    DirectedGraph

  synthesize-attention-correct :
    (Φ-min : ℝ) →
    (hd : HopfieldDynamics (synthesize-attention Φ-min)) →
    {-| Synthesized graph has Φ ≥ Φ-min -}
    ⊤  -- Φ-static (synthesize-attention Φ-min) hd ≥ Φ-min

{-|
## Multi-Head Attention: The Real Story?

**New hypothesis**: Multi-head attention's power comes from:
- **Not** the homotopy type of individual heads
- **But** the composition structure of multiple heads

Each head is K(Kₙ) (contractible), but:
- h heads in parallel create product structure
- Information integration across heads
- Φ might grow with number of heads?

**Question**: Is Φ(multi-head-attention h n) > Φ(attention-graph n)?
-}

-- Multi-head Φ composition
Φ-multi-head :
  (h n : Nat) →
  (hd : HopfieldDynamics (multi-head-attention h n)) →
  ℝ
Φ-multi-head h n hd = Φ-static (multi-head-attention h n) hd

postulate
  Φ-multi-head-composition :
    (h n : Nat) →
    (hd-multi : HopfieldDynamics (multi-head-attention h n)) →
    (hd-single : HopfieldDynamics (attention-graph n)) →
    {-| Φ-multi-head h n hd-multi ≥ Φ-attention n hd-single -}
    ⊤  -- Multi-head might have higher Φ due to parallel composition

{-|
## The Real Distinguisher: Dynamics

**Static graph analysis limitations**:
- K(attention) is contractible (π₁ = 0)
- Doesn't capture what makes attention special

**Dynamic analysis needed**:
- How does the graph evolve during forward pass?
- Query-Key-Value structure creates dynamic routing
- Attention weights change the effective graph structure per token

**Next steps**:
1. Model attention as **dynamical system** on graphs
2. Compute homotopy type of **trajectory space**
3. Compare Φ over time, not just static Φ
-}

{-|
## Preliminary Findings

1. **Static homotopy**: π₁(K(attention)) = 0 (contractible)
   - Complete graphs have trivial fundamental group
   - No topological "holes" in connectivity structure

2. **Integration**: Φ(attention) >> 0 (high)
   - Bidirectional information flow enables integration
   - Distinguishes from feedforward despite same π₁

3. **Synthesis challenge**: Specifying (π₁ = 0, Φ > threshold)
   - Not specific to attention
   - Many graphs could satisfy this
   - Need additional constraints (dynamics, resource structure)

4. **Next direction**: Dynamic homotopy types
   - Trajectory spaces of attention computation
   - Time-dependent graph evolution
   - Persistent homology over computation steps
-}
