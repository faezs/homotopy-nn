# Blog Post: Neural Networks Are Functors

## Summary

Created a comprehensive blog post explaining the categorical foundations of the homotopy-nn codebase, written in the style of the README (precise mathematical definitions with clear conceptual explanations).

## Location

- **Main file**: `docs/index.html`
- **Styling**: `docs/css/style.css`
- **Diagrams**: `docs/images/*.svg`
- **Diagram source**: `docs/diagrams/NetworkDiagrams.hs`

## Content Overview

### 1. DirectedGraph = Functor ·⇉· FinSets

Explains the fundamental definition:
```agda
DirectedGraph : Type
DirectedGraph = Functor ·⇉· FinSets
```

The parallel arrows category `·⇉·` maps to:
- `vertices G = G.F₀ true`
- `edges G = G.F₀ false`
- `source G = G.F₁ true`
- `target G = G.F₁ false`

### 2. Network Summing Functors

Covers Σ_C(G) construction:
- Objects: Subsets S ⊆ V(G)
- Morphisms: Subset inclusions respecting graph structure
- Conservation laws via equalizers (Proposition 2.10)

### 3. Semiring Homomorphisms and Evaluation

**Key insight**: Neural networks form a categorical algebra with:
- Sequential composition (∘): Monoidal structure
- Parallel composition (⊗): Tensor structure
- Identity networks: Units

Any semiring homomorphism φ: Networks → ℝ provides evaluation semantics:
- `φ(G₁ ∘ G₂) = φ(G₁) * φ(G₂)`
- `φ(G₁ ⊗ G₂) = φ(G₁) + φ(G₂)`
- `φ(id) = 1`

**Examples**:
- Parameter count
- FLOPs
- Information flow (mutual information)
- Resource consumption (energy, memory)

### 4. Example Architectures

Three concrete examples with diagrams:

1. **SimpleMLP**: Chain `x₀ → h₁ → h₂ → y`
   - No convergence, total order poset
   - Σ_C(G) has objects: ∅, {x₀}, {h₁}, {h₂}, {y}

2. **ConvergentNetwork**: ResNet-like
   - Two branches: `x₀,₁ → h ← x₀,₂`
   - Fork construction: Add A★ (star) and A (tang)
   - Diamond poset: `y ≤ h ≤ A ≤ x₀,₁`, `y ≤ h ≤ A ≤ x₀,₂`

3. **ComplexNetwork**: Multi-path
   - Multiple convergence points → multiple forks
   - Tree forest poset (Theorem 1.2)

### 5. DNN Topos

Explains the Grothendieck topos construction:
```
DNN-Topos = Sh[X, Alexandrov]
```

- X: Fork poset from architecture
- Alexandrov coverage: Maximal sieves
- Sheaf condition at convergent vertices

**Backpropagation** as natural transformations (Theorem 1.1):
```
∂L/∂w_a = ⊕_{γ_a ∈ Ω_a} φ_{γ_a}
```

### 6. HIT Technique (Technical Note)

Explains how to prove path uniqueness in cubical Agda without K axiom:

**Problem**: Pattern matching on indexed types requires K (disabled)

**Solution**: Path constructor in HIT:
```agda
data _≤ˣ_ : X-Vertex → X-Vertex → Type where
  ≤ˣ-refl : ∀ {x} → x ≤ˣ x
  ≤ˣ-edge : ∀ {x y} → Edge x y → x ≤ˣ y
  ≤ˣ-trans : ∀ {x y z} → x ≤ˣ y → y ≤ˣ z → x ≤ˣ z

  -- PATH CONSTRUCTOR: thinness built-in
  ≤ˣ-thin : ∀ {x y} (p q : x ≤ˣ y) → p ≡ q
```

This makes category laws immediate:
```agda
idl f = ≤ˣ-thin (id ∘ f) f
assoc f g h = ≤ˣ-thin ((f ∘ g) ∘ h) (f ∘ (g ∘ h))
```

## Infrastructure

### Haskell Diagrams

Created `docs/diagrams/NetworkDiagrams.hs` using diagrams library to generate:
- `mlp.svg` - Simple MLP chain
- `convergent.svg` - Basic convergent network
- `convergent_fork.svg` - With fork construction
- `convergent_poset.svg` - Resulting poset X
- `complex.svg` - Complex multi-path network

**Build**: `cd docs/diagrams && make`

### GitHub Pages Setup

- `docs/_config.yml`: Configuration
- `docs/README.md`: Deployment instructions
- Placeholder SVG diagrams included for immediate rendering

**Deployment**:
1. Go to repository Settings → Pages
2. Source: Deploy from branch
3. Branch: `claude/nix-develop-setup-01GUiQF2cgjkUAZLphdjCLdA` (or main after merge)
4. Folder: `/docs`

Site will be at: `https://<username>.github.io/homotopy-nn/`

## Style

Follows the README style:
- **Precise mathematical definitions** in code blocks
- **Clear conceptual explanations** (why functors? why category theory?)
- **No fluff**, direct statements
- **Type theory shown explicitly**
- **Explains rationale** (like "The `thin` constructor makes...")

Inspired by Conal Elliott and John Baez:
- Compositional perspective (everything composes)
- Universal properties (categorical reasoning)
- Clear types and signatures
- Motivation before formalism

## Statistics

- **11 files** created
- **~1000 lines** total (HTML + CSS + Haskell + docs)
- **6 sections** in blog post
- **3 network architectures** with diagrams
- **Complete semiring homomorphism** treatment

## Git Commit

```
commit 073295c
Add blog post: Neural Networks Are Functors

Created comprehensive blog post explaining the categorical foundations
of neural networks, following the style of the project README.
```

Pushed to: `claude/nix-develop-setup-01GUiQF2cgjkUAZLphdjCLdA`

## Next Steps

1. **Generate diagrams**: Run `cd docs/diagrams && make` to create proper diagrams
2. **Enable GitHub Pages**: Configure in repository settings
3. **Merge to main**: After review, merge branch to deploy
4. **Optional**: Enhance Haskell diagrams with more details (fork stars, labels)

## References in Post

1. Marcolli & Manin (2020) - Homotopy theoretic and categorical models
2. Belfiore & Bennequin (2022) - Topos and Stacks of Deep Neural Networks
3. Curto & Itskov (2008) - Neural codes and topology
4. Tononi et al. (2016) - Integrated information theory
