{-# OPTIONS --cubical #-}

{-|
# Graph Coproduct (Disjoint Union)

Implements coproduct in the category of graphs, needed for compositional
reasoning in neural network extraction.

## Mathematical Background

**Graphs are presheaves**: Graphs ≃ PSh(·⇇·) where ·⇇· is the parallel arrows category.

**Presheaves have coproducts**: Computed pointwise as `F₀(x) ⊎ G₀(x)` for each object x.

**Direct construction**: Since graphs are defined as records (not functors), we
implement the coproduct directly rather than transporting via the equivalence.

## The Coproduct

For graphs G and H:
- **Nodes**: G.Node ⊎ H.Node (disjoint union)
- **Edges**:
  - `inl x → inl y`: edges within G
  - `inr x → inr y`: edges within H
  - `inl x → inr y`: no cross-component edges (⊥)
  - `inr x → inl y`: no cross-component edges (⊥)

**Universal property**: For any graph Z with morphisms f : G → Z and g : H → Z,
there exists a unique morphism [f,g] : G + H → Z such that:
- [f,g] ∘ inl = f
- [f,g] ∘ inr = g

## Usage in Neural Network Compilation

**Composition `f ⊙ g`**:
```
build-graph (f ⊙ g) = (build-graph g +ᴳ build-graph f) + connection-edges
```

**Fork `f | g`**:
```
build-graph (Fork f g) = build-graph f +ᴳ build-graph g
```

**Join `f + g`**: Similar to Fork but with output merging

-}

module Neural.Compile.GraphCoproduct where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.HLevel.Closure using (hlevel-instance)

open import Cat.Instances.Graphs using (Graph; Graph-hom; Graphs)
open import Cat.Diagram.Coproduct

open import Data.Sum.Base using (_⊎_; inl; inr; elim)
open import Data.Sum.Properties using (⊎-is-hlevel; Discrete-⊎; inl-inj; inr-inj)
open import Data.Dec.Base using (Dec)

open import Neural.Graph.Oriented using (is-oriented)
open import Neural.Graph.Fork.Fork

private variable
  o ℓ : Level

--------------------------------------------------------------------------------
-- § 1: Graph Coproduct Definition
--------------------------------------------------------------------------------

{-|
## Direct Construction

The coproduct G +ᴳ H is the disjoint union with no cross-edges.
-}

_+ᴳ_ : ∀ {o ℓ} → Graph o ℓ → Graph o ℓ → Graph o ℓ
_+ᴳ_ {o} {ℓ} G H .Graph.Node = G .Graph.Node ⊎ H .Graph.Node

_+ᴳ_ {o} {ℓ} G H .Graph.Edge (inl x) (inl y) = G .Graph.Edge x y
_+ᴳ_ {o} {ℓ} G H .Graph.Edge (inr x) (inr y) = H .Graph.Edge x y
_+ᴳ_ {o} {ℓ} G H .Graph.Edge (inl x) (inr y) = Lift ℓ ⊥  -- No cross edges
_+ᴳ_ {o} {ℓ} G H .Graph.Edge (inr x) (inl y) = Lift ℓ ⊥  -- No cross edges

_+ᴳ_ {o} {ℓ} G H .Graph.Node-set = ⊎-is-hlevel 0 ⦃ hlevel-instance (G .Graph.Node-set) ⦄ ⦃ hlevel-instance (H .Graph.Node-set) ⦄

_+ᴳ_ {o} {ℓ} G H .Graph.Edge-set {inl x} {inl y} = G .Graph.Edge-set
_+ᴳ_ {o} {ℓ} G H .Graph.Edge-set {inr x} {inr y} = H .Graph.Edge-set
_+ᴳ_ {o} {ℓ} G H .Graph.Edge-set {inl x} {inr y} = is-prop→is-set (Lift-is-hlevel 1 λ ())
_+ᴳ_ {o} {ℓ} G H .Graph.Edge-set {inr x} {inl y} = is-prop→is-set (Lift-is-hlevel 1 λ ())

infixr 25 _+ᴳ_

--------------------------------------------------------------------------------
-- § 2: Inclusion Morphisms
--------------------------------------------------------------------------------

{-|
## Canonical Inclusions

The coproduct comes with two inclusion morphisms witnessing that G and H
are subgraphs of G +ᴳ H.
-}

inlᴳ : ∀ {G H : Graph o ℓ} → Graph-hom G (G +ᴳ H)
inlᴳ .Graph-hom.node = inl
inlᴳ .Graph-hom.edge e = e  -- Edges preserved

inrᴳ : ∀ {G H : Graph o ℓ} → Graph-hom H (G +ᴳ H)
inrᴳ .Graph-hom.node = inr
inrᴳ .Graph-hom.edge e = e  -- Edges preserved

--------------------------------------------------------------------------------
-- § 3: Universal Property
--------------------------------------------------------------------------------

{-|
## Coproduct Mediating Morphism

Given f : G → Z and g : H → Z, construct [f,g] : G +ᴳ H → Z.

**Case analysis on nodes**:
- `inl x`: Apply f
- `inr y`: Apply g

**Case analysis on edges**:
- `inl x → inl y`: Apply f.edge
- `inr x → inr y`: Apply g.edge
- Cross edges: Impossible (⊥-elim)
-}

-- TODO: Implement mediating morphism [_,_]ᴳ
-- Currently commented out due to scope checking issues
-- [_,_]ᴳ : ∀ {o ℓ} {G H Z : Graph o ℓ} → Graph-hom G Z → Graph-hom H Z → Graph-hom (G +ᴳ H) Z

--------------------------------------------------------------------------------
-- § 4: Convergence Transport
--------------------------------------------------------------------------------

{-|
## Convergence Preservation Under Coproduct Inclusions

The coproduct inclusions inl : G → G +ᴳ H and inr : H → G +ᴳ H preserve
convergence because:
- Edges within each component are preserved
- Sources remain distinct after lifting

This enables compositional reasoning: if a node is convergent in a subnetwork,
it remains convergent in the composite network.
-}

-- Left inclusion preserves convergence
inl-convergent : ∀ {o ℓ} {G H : Graph o ℓ}
               {G-oriented : is-oriented G}
               {GH-oriented : is-oriented (G +ᴳ H)}
               {G-discrete : ∀ (x y : G .Graph.Node) → Dec (x ≡ y)}
               {GH-discrete : ∀ (x y : (G +ᴳ H) .Graph.Node) → Dec (x ≡ y)}
               {v : G .Graph.Node}
               → ForkConstruction.is-convergent G G-oriented G-discrete v
               → ForkConstruction.is-convergent (G +ᴳ H) GH-oriented GH-discrete (inl v)
inl-convergent {G = G} {H = H} {G-oriented = G-oriented} {GH-oriented = GH-oriented}
               {G-discrete = G-discrete} {GH-discrete = GH-discrete} {v = v} conv-G =
  let s₁ = ForkConstruction.is-convergent.source₁ conv-G
      s₂ = ForkConstruction.is-convergent.source₂ conv-G
  in record
  { source₁ = inl s₁
  ; source₂ = inl s₂
  ; distinct = λ eq → ForkConstruction.is-convergent.distinct conv-G (inl-inj eq)
  ; edge₁ = ForkConstruction.is-convergent.edge₁ conv-G  -- Edge preserved in coproduct
  ; edge₂ = ForkConstruction.is-convergent.edge₂ conv-G  -- Edge preserved in coproduct
  }

-- Right inclusion preserves convergence
inr-convergent : ∀ {o ℓ} {G H : Graph o ℓ}
               {H-oriented : is-oriented H}
               {GH-oriented : is-oriented (G +ᴳ H)}
               {H-discrete : ∀ (x y : H .Graph.Node) → Dec (x ≡ y)}
               {GH-discrete : ∀ (x y : (G +ᴳ H) .Graph.Node) → Dec (x ≡ y)}
               {v : H .Graph.Node}
               → ForkConstruction.is-convergent H H-oriented H-discrete v
               → ForkConstruction.is-convergent (G +ᴳ H) GH-oriented GH-discrete (inr v)
inr-convergent {G = G} {H = H} {H-oriented = H-oriented} {GH-oriented = GH-oriented}
               {H-discrete = H-discrete} {GH-discrete = GH-discrete} {v = v} conv-H =
  let t₁ = ForkConstruction.is-convergent.source₁ conv-H
      t₂ = ForkConstruction.is-convergent.source₂ conv-H
  in record
  { source₁ = inr t₁
  ; source₂ = inr t₂
  ; distinct = λ eq → ForkConstruction.is-convergent.distinct conv-H (inr-inj eq)
  ; edge₁ = ForkConstruction.is-convergent.edge₁ conv-H  -- Edge preserved in coproduct
  ; edge₂ = ForkConstruction.is-convergent.edge₂ conv-H  -- Edge preserved in coproduct
  }

-- Inverse: Non-convergence in coproduct implies non-convergence in component
-- (Contrapositive of forward transport)
inl-not-convergent : ∀ {o ℓ} {G H : Graph o ℓ}
                   {G-oriented : is-oriented G}
                   {GH-oriented : is-oriented (G +ᴳ H)}
                   {G-discrete : ∀ (x y : G .Graph.Node) → Dec (x ≡ y)}
                   {GH-discrete : ∀ (x y : (G +ᴳ H) .Graph.Node) → Dec (x ≡ y)}
                   {v : G .Graph.Node}
                   → ¬ ForkConstruction.is-convergent (G +ᴳ H) GH-oriented GH-discrete (inl v)
                   → ¬ ForkConstruction.is-convergent G G-oriented G-discrete v
inl-not-convergent {G = G} {H = H} {G-oriented = G-oriented} {GH-oriented = GH-oriented}
                   {G-discrete = G-discrete} {GH-discrete = GH-discrete} {v = v}
                   not-conv-coproduct conv-g =
  not-conv-coproduct (inl-convergent conv-g)

inr-not-convergent : ∀ {o ℓ} {G H : Graph o ℓ}
                   {H-oriented : is-oriented H}
                   {GH-oriented : is-oriented (G +ᴳ H)}
                   {H-discrete : ∀ (x y : H .Graph.Node) → Dec (x ≡ y)}
                   {GH-discrete : ∀ (x y : (G +ᴳ H) .Graph.Node) → Dec (x ≡ y)}
                   {v : H .Graph.Node}
                   → ¬ ForkConstruction.is-convergent (G +ᴳ H) GH-oriented GH-discrete (inr v)
                   → ¬ ForkConstruction.is-convergent H H-oriented H-discrete v
inr-not-convergent {G = G} {H = H} {H-oriented = H-oriented} {GH-oriented = GH-oriented}
                   {H-discrete = H-discrete} {GH-discrete = GH-discrete} {v = v}
                   not-conv-coproduct conv-h =
  not-conv-coproduct (inr-convergent conv-h)

-- Forward extraction: Convergence in coproduct implies convergence in component
-- This works because _+ᴳ_ has no cross-edges
inl-convergent-inv : ∀ {o ℓ} {G H : Graph o ℓ}
                   {G-oriented : is-oriented G}
                   {GH-oriented : is-oriented (G +ᴳ H)}
                   {G-discrete : ∀ (x y : G .Graph.Node) → Dec (x ≡ y)}
                   {GH-discrete : ∀ (x y : (G +ᴳ H) .Graph.Node) → Dec (x ≡ y)}
                   {v : G .Graph.Node}
                   → ForkConstruction.is-convergent (G +ᴳ H) GH-oriented GH-discrete (inl v)
                   → ForkConstruction.is-convergent G G-oriented G-discrete v
inl-convergent-inv record { source₁ = inl s₁ ; source₂ = inl s₂ ; distinct = dist ; edge₁ = e₁ ; edge₂ = e₂ } =
  record
  { source₁ = s₁
  ; source₂ = s₂
  ; distinct = λ eq → dist (ap inl eq)
  ; edge₁ = e₁  -- Edges preserved: (G +ᴳ H).Edge (inl s₁) (inl v) = G.Edge s₁ v
  ; edge₂ = e₂  -- Edges preserved: (G +ᴳ H).Edge (inl s₂) (inl v) = G.Edge s₂ v
  }
-- Impossible cases: cross-edges
inl-convergent-inv record { source₁ = inl s₁ ; source₂ = inr s₂ ; edge₂ = e₂ } =
  absurd (Lift.lower e₂)  -- Cross-edge is Lift ℓ ⊥
inl-convergent-inv record { source₁ = inr s₁ ; edge₁ = e₁ } =
  absurd (Lift.lower e₁)  -- Cross-edge is Lift ℓ ⊥

inr-convergent-inv : ∀ {o ℓ} {G H : Graph o ℓ}
                   {H-oriented : is-oriented H}
                   {GH-oriented : is-oriented (G +ᴳ H)}
                   {H-discrete : ∀ (x y : H .Graph.Node) → Dec (x ≡ y)}
                   {GH-discrete : ∀ (x y : (G +ᴳ H) .Graph.Node) → Dec (x ≡ y)}
                   {v : H .Graph.Node}
                   → ForkConstruction.is-convergent (G +ᴳ H) GH-oriented GH-discrete (inr v)
                   → ForkConstruction.is-convergent H H-oriented H-discrete v
inr-convergent-inv record { source₁ = inr s₁ ; source₂ = inr s₂ ; distinct = dist ; edge₁ = e₁ ; edge₂ = e₂ } =
  record
  { source₁ = s₁
  ; source₂ = s₂
  ; distinct = λ eq → dist (ap inr eq)
  ; edge₁ = e₁  -- Edges preserved: (G +ᴳ H).Edge (inr s₁) (inr v) = H.Edge s₁ v
  ; edge₂ = e₂  -- Edges preserved: (G +ᴳ H).Edge (inr s₂) (inr v) = H.Edge s₂ v
  }
-- Impossible cases: cross-edges
inr-convergent-inv record { source₁ = inr s₁ ; source₂ = inl s₂ ; edge₂ = e₂ } =
  absurd (Lift.lower e₂)  -- Cross-edge is Lift ℓ ⊥
inr-convergent-inv record { source₁ = inl s₁ ; edge₁ = e₁ } =
  absurd (Lift.lower e₁)  -- Cross-edge is Lift ℓ ⊥

--------------------------------------------------------------------------------
-- § 5: Coproduct Laws (TODO)
--------------------------------------------------------------------------------

{-|
**TODO**: Prove universal property

Should prove:
1. [f,g]ᴳ ∘ inlᴳ ≡ f
2. [f,g]ᴳ ∘ inrᴳ ≡ g
3. Uniqueness: if h ∘ inlᴳ ≡ f and h ∘ inrᴳ ≡ g, then h ≡ [f,g]ᴳ

These are straightforward by path reasoning on Graph-hom records.
-}

-- Graphs-coproducts : ∀ {o ℓ} (G H : Graph o ℓ) → Coproduct (Graphs o ℓ) G H
-- Graphs-coproducts G H = record
--   { coapex = G +ᴳ H
--   ; ι₁ = inlᴳ
--   ; ι₂ = inrᴳ
--   ; has-is-coproduct = {!!}  -- TODO
--   }

--------------------------------------------------------------------------------
-- § 5: Helper: Adding Edges Between Components
--------------------------------------------------------------------------------

{-|
## Connection Edges for Composition

For neural network composition `f ⊙ g`, we need edges from g's outputs to f's inputs.

**Pattern**: Start with coproduct, then add specific cross-edges.
-}

-- Example: Add edges from G-outputs to H-inputs
-- connect-outputs-inputs : (G H : Graph o ℓ)
--                        → (G-outputs : List (G .Node))
--                        → (H-inputs : List (H .Node))
--                        → Graph o ℓ
-- connect-outputs-inputs G H g-outs h-ins = record (G +ᴳ H)
--   { Edge = λ { (inl g-out) (inr h-in) → if (g-out ∈ g-outs) ∧ (h-in ∈ h-ins) then ⊤ else ⊥
--              ; x y → (G +ᴳ H) .Edge x y
--              }
--   }

{-|
**Note**: Actual implementation will be in ForkExtract.agda where we know
the specific structure of NetworkNode = Fin n ⊎ Fin m.
-}
