# Homotopy Neural Networks - Claude Code Workflow

## Project Overview

This project implements neural information networks using homotopy type theory and category theory in Agda, based on the paper on neural codes and directed graphs. We're using the 1Lab library for cubical Agda development.

## Main Goals

1. **Implement Definition 2.6**: Directed finite graphs as functors G: ·⇉· → FinSets
2. **Implement Definition 2.7 & Lemma 2.8**: Pointed directed graphs and the construction G* from G
3. **Neural Networks**: Connect directed graphs to neural place fields and response patterns
4. **Homotopy Reconstruction**: Use the combinatorial structure to reconstruct topological spaces

## Library Setup

- **Agda**: Using cubical Agda with 1Lab library for HoTT
- **1Lab Location**: `/nix/store/f5w4kylmw0idvbn7bbhn8837h5k3j7lv-1lab-unstable-2025-07-01/`
- **Library File**: `./libraries` points to 1Lab
- **Flags**: `--rewriting --guardedness --cubical --no-load-primitives`

## Key Workflow Discoveries

### ALWAYS LOOK AT 1LAB SOURCE
- **Golden Rule**: LOOK AT the 1lab src. Always look at the 1lab src for any confusion about what it provides out of the box
- **Search patterns**: Use Grep tool to find examples of similar constructions
- **Reference implementations**: Check Cat.Instances.* for category examples

### 1Lab Infrastructure Available

#### Categorical Structures (in `Cat.*`)
- **Monoidal categories**: `Cat.Monoidal.Base` has `Monoidal-category` record
- **Symmetric monoidal**: `Cat.Monoidal.Braided` has `Symmetric-monoidal` record with braiding
- **Product categories**: `Cat.Instances.Product` has `_×ᶜ_` for C × D
- **Slice categories**: `Cat.Instances.Slice` has `/-Obj` for C/c (objects over c)
- **Equalizers/Coequalizers**: `Cat.Diagram.Equaliser` and `Cat.Diagram.Coequaliser`
- **Adjunctions**: `Cat.Functor.Adjoint` for optimal resource characterizations
- **Parallel arrows**: `Cat.Instances.Shape.Parallel` defines ·⇉· category

#### Algebraic Structures (in `Algebra.*`)
- **Monoids**: `Algebra.Monoid` has `Monoid-on` and `is-monoid`
- **Abelian groups**: `Algebra.Group.Ab` has `is-abelian-group` and `Abelian-group-on`
- **Semigroups**: `Algebra.Semigroup` for associative operations

#### Order Theory (in `Order.*`)
- **Posets**: `Order.Base` has `Poset` with `_≤_` relation
  - Reflexivity, transitivity, antisymmetry built-in
  - `Ob-is-set` proven automatically from antisymmetry

#### What's NOT in 1Lab
- **Real numbers**: Not available - we postulated ℝ with operations
- **Coslice categories**: Dual of slice (c\C) - need to define if needed
- **Quotient categories**: C/~ exists in Cubical library but not 1Lab
  - Reference: `Cubical.Categories.Constructions.Quotient`
  - We postulated for now
- **Preordered monoids**: Monoid + preorder (without antisymmetry) - defined ourselves
- **Stochastic matrices**: Not built-in, would need to define for FinStoch example
- **Natural number subtraction** (monus): Available as `_-_` in `Prim.Data.Nat`

### Compilation
- **Type checking**: `agda --library-file=./libraries src/Neural/Base.agda`
- **No compilation needed**: We're doing pure type theory, no executables

### Working with Fin in Cubical Agda
- **Avoid pattern matching**: `with fin-view` causes UnsupportedIndexedMatch warnings
- **Use Fin-cases instead**: `Fin-cases fzero (λ e → ...) : Fin (suc n) → P`
- **Fin-elim for recursion**: When you need full elimination principle
- **Type confusion**: Be very careful about Fin n vs Fin (suc n) in lambda parameters

### Path Reasoning and ∙-assoc
- **Import required**: `open import 1Lab.Path.Reasoning` and `open import 1Lab.Reflection.Marker`
- **Use ap! macro**: `⌜ expression ⌝ ≡⟨ ap! lemma ⟩` for applying lemmas to marked subexpressions
- **Pattern from ConcreteGroups**: For associativity proofs in pointed categories

### Agda JSON Interaction
- **Start mode**: `agda --interaction-json --library-file=./libraries` (no file argument)
- **Load file**: `IOTCM "src/Neural/Base.agda" None Indirect (Cmd_load "src/Neural/Base.agda" [])`
- **Get goals**: Automatically shown after loading, includes precise types for holes
- **Future improvement**: Haskell MCP using Agda library directly instead of JSON protocol

## Current Implementation Status

### ✅ Completed Modules

#### Section 2: Network Summing Functors
- **Neural.Base**: DirectedGraph as `Functor ·⇉· FinSets`
- **Neural.SummingFunctor**: Lemma 2.3, Proposition 2.4, Definition 2.5 (ΣC(X) category)
- **Neural.Network.Conservation**: Section 2.2 - Conservation laws via equalizers and coequalizers
  - Lemma 2.9: Source and target functors
  - Proposition 2.10: Equalizer approach (Kirchhoff's law)
  - Proposition 2.12: Coequalizer/quotient category approach
  - Definition 2.14: Network summing functors ΣC(G)
- **Neural.Network.Grafting**: Section 2.3.2 - Properad-constrained summing functors
  - Definition 2.18: Properads in Cat
  - HasProperadStructure: Categories with properad structure
  - Lemma 2.19: Σprop_C(G) subcategory
  - Corollary 2.20: Corolla determinacy

#### Section 3: Neural Information and Resources
- **Neural.Information**: Section 3.1 - Neural codes, firing rates, metabolic constraints
  - Binary neural codes, rate codes, spike timing codes
  - Firing rate and coding capacity: Rmax = -y log(y∆t)
  - Metabolic efficiency: ϵ = I(X,Y)/E
  - Connection weights and information optimization
  - Resource assignment framework
- **Neural.Resources**: Section 3.2 - Mathematical theory of resources
  - ResourceTheory: Symmetric monoidal category (R,◦,⊗,I)
  - PreorderedMonoid: (R,+,⪰,0) from isomorphism classes
  - ConversionRates: ρA→B = sup{m/n | n·A ⪰ m·B}
  - S-Measuring: Monoid homomorphisms preserving order
  - Theorem 5.6: ρA→B · M(B) ≤ M(A)
- **Neural.Resources.Convertibility**: Section 3.2.2 detailed - Convertibility and rates
  - Convertibility preorder properties (reflexivity, transitivity, compatibility)
  - Tensor powers A⊗n for bulk conversion
  - Conversion rate calculations with examples
  - Measuring homomorphisms (ℝ-valued, entropy, energy)
  - Theorem 5.6 proof outline and concrete examples
- **Neural.Resources.Optimization**: Section 3.3 - Adjunctions and optimality
  - OptimalConstructor: Left adjoint β ⊣ ρ for resource assignment
  - Universal property via adjunct-hom-equiv
  - Freyd's Adjoint Functor Theorem application
  - Solution sets and continuity conditions
  - Examples: Minimal energy, minimal entropy optimization

#### Section 4: Networks with Computational Structures
- **Neural.Computational.TransitionSystems**: Sections 4.1-4.4 - Transition systems as computational resources
  - Definition: Transition systems τ = (S, ι, L, T, S_F)
  - Morphisms: Partial simulations (σ, λ) with simulation property
  - Category C with coproduct (⊔) and product (×)
  - Subcategory C' with single final states
  - Grafting operations for sequential composition
  - Lemma 4.2: Grafting for acyclic graphs via topological ordering
  - Definition 4.3: Grafting for strongly connected graphs via coproduct
  - Proposition 4.4: Faithful functor Υ: Σ_C'(V_G) → Σ^prop_C'(G)
  - **Section 4.4: Distributed computing and neuromodulation**
    - Definition 4.5: Time-delay transitions with labels L = L' × ℤ₊
    - TimeDelayTransitionSystem record with base labels and time delays
    - Definition 4.6: Distributed structures on graphs
      - Partition into N machines with neuromodulator vertices v₀,ᵢ
      - Source/target vertex subsets for neuromodulation
      - Augmented graph G₀ with time-delayed edges
    - Definition 4.7: Category Gdist of distributed graphs
      - Objects: (G,m) with strongly connected induced subgraphs
      - Morphisms preserve machine partitions
    - Proposition 4.8: Modified summing functor for distributed systems
      - Grafting within machines (strongly connected)
      - Grafting between machines via condensation graph (acyclic)

### 🚧 Needs Work
- **Conservation module examples**: Have unsolved metas (`{!!}`) for:
  - `example-category` (HasSumsAndZero FinSets)
  - Functor composition laws for example graphs
- **Grafting module**: No concrete examples yet (pending)
- **Pointed graphs**: Commented out in Neural.Base due to earlier issues

### 📋 TODO
- Add concrete examples to Conservation module (finish example-category)
- Add concrete examples to Grafting module (corollas, grafting operations)
- Implement FinProb, FinStoch, FP examples from Section 3.2.1
- Section 4.4.3: Protocol simplicial complexes for distributed computing
- Section 4.4.3: Embedded graph invariants (fundamental groups)
- Section 5: Weighted codes and information transmission

## Type Theory Notes

### DirectedGraph Structure
```agda
DirectedGraph = Functor ·⇉· FinSets
-- ·⇉· has objects: false (edges), true (vertices)
-- ·⇉· has morphisms: false→true (source, target), identities

vertices : DirectedGraph → Nat
edges : DirectedGraph → Nat
source : (G : DirectedGraph) → Fin (edges G) → Fin (vertices G)
target : (G : DirectedGraph) → Fin (edges G) → Fin (vertices G)
```

### PointedFinSets Category
```agda
-- Objects: Nat (representing Fin (suc n))
-- Morphisms: Σ (Fin (suc m) → Fin (suc n)) (λ f → f fzero ≡ fzero)
-- Basepoint preservation: f fzero ≡ fzero
```

### Lemma 2.8 Construction
Goal: Given G, construct G* by adding:
- One basepoint vertex (at fzero)
- One looping edge (at fzero, from basepoint to itself)
- Shift original vertices/edges by one position

**Key insight**: Should only need ONE fsuc for shifting, not two!

### Section 4.4: Distributed Computing and Time-Delay Systems

**Time-delay automata** (Definition 4.5):
- Labels: L = L' × ℤ₊ (base label, time delay)
- Example: {aⁿbⁿcⁿ} language (non-context-free)
  - Transitions: (a,0), (b,1), (c,2) deposit symbols at different time steps
- Subcategory Cₜ ⊂ C' with time-delayed transitions

**Distributed structures** (Definition 4.6):
- Partition graph G into N machines (vertex subsets Vᵢ)
- Add neuromodulator vertex v₀,ᵢ per machine
- Edges:
  - Incoming to v₀,ᵢ: from any source vertex (any machine)
  - Outgoing from v₀,ᵢ: to target vertices in same machine
- Time delays nₑ ∈ ℤ₊ on new edges (original edges have nₑ = 0)

**Implementation notes**:
- `DistributedStructure` record at Type₁ (contains Type-valued predicates)
- `embed-edge` field for embedding original edges into augmented graph
- `condensation-distributed` always produces acyclic graph
- Universe level: Changed from Type to Type₁ to accommodate predicates

**Category Gdist** (Definition 4.7):
- Objects: (G,m) with induced subgraphs Gᵢ strongly connected
- Morphisms: preserve machine assignments via `preserves-machine-assignment`

**Proposition 4.8**: Two-level grafting
1. Within machines: strongly-connected grafting (Definition 4.3)
2. Between machines: acyclic grafting via condensation (Lemma 4.2)

## Common Pitfalls

1. **Fin indexing confusion**: Remember Fin n has elements 0, 1, ..., n-1
2. **Basepoint preservation**: Always check f fzero ≡ fzero for PointedFinSets morphisms
3. **Functor laws**: F-id and F-∘ need careful path reasoning
4. **UnsupportedIndexedMatch**: Use Fin-cases/Fin-elim instead of pattern matching
5. **Double shifting**: Don't use fsuc (fsuc (...)) when adding single basepoint/edge
6. **Name clashing**: Importing same postulate/definition from multiple modules causes errors
   - Use `open ModuleName using (...)` to import selectively
   - Example: ℝ postulated in multiple modules - import from one and reuse
7. **Level mismatches**: Type o is not ≤ Type ℓ errors
   - Check that preorders/relations have correct universe levels
   - Use (o ⊔ ℓ) for relations involving both objects and morphisms
8. **Operator precedence**: Parsing errors with mixed infix operators
   - Use parentheses liberally: `(A ^⊗ n) R.≅ (B ^⊗ m)`
   - Check precedence levels don't conflict
9. **Unsolved metas with --allow-unsolved-metas**: Type-checks but incomplete
   - Mark TODOs clearly
   - Examples need actual implementations, not just type signatures

## Development Commands

```bash
# Type check
agda --library-file=./libraries src/Neural/Base.agda

# Interactive development
agda --interaction-json --library-file=./libraries

# Start with specific file for holes/goals
echo 'IOTCM "src/Neural/Base.agda" None Indirect (Cmd_load "src/Neural/Base.agda" [])' | agda --interaction-json --library-file=./libraries
```