{-|
# Oriented Graphs as Full Subcategory

**Key idea**: Oriented graphs are graphs satisfying three properties:
1. **Classical**: At most one edge between vertices (is-prop Edge)
2. **No loops**: No self-edges
3. **Acyclic**: Reachability relation is antisymmetric (directed)

We define them as a **full subcategory** of Graphs using 1Lab's `Restrict` construction.

## Benefits

- Automatic inheritance of graph structure (products, limits, exponentials)
- No K axiom issues (working with 1Lab's proven types)
- Clean mathematical structure (full subcategory preserves properties)
- Graph homomorphisms automatically preserve orientation

## From the Paper (Belfiore & Bennequin 2022, Section 1.1)

> "An oriented graph Γ is **directed** when the relation a ≤ b between vertices,
> defined by the existence of an oriented path, is a partial ordering on the set V(Γ).
> A graph is **classical** if there exists at most one edge between two vertices,
> and no loop at one vertex."

-}

module Neural.Graph.Oriented where

open import Neural.Graph.Base
open import Neural.Graph.Path

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.HLevel.Closure

import Cat.Functor.FullSubcategory
open import Cat.Functor.Properties
open import Cat.Prelude
open import Cat.Base

open import Data.Sum.Base

private variable
  o ℓ : Level

{-|
## The Oriented Property

A graph is **oriented** if it satisfies three conditions:
1. Classical (at most one edge between vertices)
2. No loops (no self-edges)
3. Acyclic (path-based reachability is antisymmetric)
-}

is-oriented : Graph o ℓ → Type _
is-oriented G = classical × no-loops × acyclic
  where
    open Graph G
    open EdgePath G

    classical : Type _
    classical = ∀ x y → is-prop (Edge x y)

    no-loops : Type _
    no-loops = ∀ x → ¬ (Edge x x)

    acyclic : Type _
    acyclic = ∀ x y → EdgePath x y → EdgePath y x → x ≡ y

{-|
## Projections

Extract each component of the oriented property.
-}

module _ {G : Graph o ℓ} (oriented : is-oriented G) where
  open Graph G
  open EdgePath G

  is-classical : ∀ {x y} → is-prop (Edge x y)
  is-classical {x} {y} = oriented .fst x y

  has-no-loops : ∀ {x} → ¬ (Edge x x)
  has-no-loops {x} = oriented .snd .fst x

  is-acyclic : ∀ {x y} → EdgePath x y → EdgePath y x → x ≡ y
  is-acyclic {x} {y} = oriented .snd .snd x y

{-|
## Oriented Property is Propositional

This is crucial for defining OrientedGraphs as a full subcategory:
we need `is-oriented G` to be a proposition (at most one proof).

**Proof strategy**:
1. `classical` is a proposition: Π-is-hlevel applied to is-prop (level 1)
2. `no-loops` is a proposition: Π-is-hlevel applied to ¬ (level 1)
3. `acyclic` is a proposition: Π-is-hlevel applied to function to Vertex equality
4. Product of propositions is a proposition: ×-is-hlevel
-}

is-oriented-is-prop : (G : Graph o ℓ) → is-prop (is-oriented G)
is-oriented-is-prop G = ×-is-hlevel 1 classical-is-prop (×-is-hlevel 1 no-loops-is-prop acyclic-is-prop)
  where
    open Graph G
    open EdgePath G

    -- Classical is a proposition
    classical-is-prop : is-prop (∀ x y → is-prop (Edge x y))
    classical-is-prop = Π-is-hlevel 1 λ x → Π-is-hlevel 1 λ y → is-prop-is-prop
      -- Use Π-is-hlevel twice: ∀ x → ∀ y → is-prop (Edge x y)
      -- is-prop is itself a proposition (is-prop-is-prop)

    -- No-loops is a proposition
    no-loops-is-prop : is-prop (∀ x → ¬ (Edge x x))
    no-loops-is-prop = Π-is-hlevel 1 λ x → fun-is-hlevel 1 (hlevel 1)
      -- Use Π-is-hlevel: ∀ x → ¬ (Edge x x)
      -- ¬ A is a proposition (function to ⊥, which has hlevel 1)

    -- Acyclic is a proposition
    acyclic-is-prop : is-prop (∀ x y → EdgePath x y → EdgePath y x → x ≡ y)
    acyclic-is-prop = Π-is-hlevel 1 λ x → Π-is-hlevel 1 λ y →
                      fun-is-hlevel 1 (fun-is-hlevel 1 (Node-set x y))
      -- Use Π-is-hlevel for x, y and fun-is-hlevel for the two EdgePath arguments
      -- Target is Node equality, which is a proposition (Node-set x y)

{-|
## Oriented Graphs as Full Subcategory

Using 1Lab's `Restrict` construction, we define OrientedGraphs as the
full subcategory of Graphs where the predicate `is-oriented` holds.

**Objects**: Σ[ G ∈ Graph ] (is-oriented G)
**Morphisms**: Same as Graph-hom (all edges between objects)

This automatically gives us:
- Category structure (inherited from Graphs)
- Products, limits (if they preserve orientation)
- Exponentials (if they preserve orientation)
- Univalence (from Restrict-is-category)
-}

OrientedGraphs : ∀ o ℓ → Precategory (lsuc o ⊔ lsuc ℓ) (o ⊔ ℓ)
OrientedGraphs o ℓ = Cat.Functor.FullSubcategory.Restrict {C = Graphs o ℓ} is-oriented

{-|
## Oriented Graphs are Univalent

Since Graphs is univalent and `is-oriented` is propositional,
OrientedGraphs inherits univalence.
-}

OrientedGraphs-is-category : ∀ {o ℓ} → is-category (OrientedGraphs o ℓ)
OrientedGraphs-is-category {o} {ℓ} =
  Cat.Functor.FullSubcategory.Restrict-is-category {C = Graphs o ℓ} is-oriented is-oriented-is-prop Graphs-is-category

{-|
## Forgetful Functor

There is a fully faithful functor from OrientedGraphs to Graphs
that forgets the orientation property.
-}

Forget-oriented : ∀ {o ℓ} → Functor (OrientedGraphs o ℓ) (Graphs o ℓ)
Forget-oriented {o} {ℓ} = Cat.Functor.FullSubcategory.Forget-full-subcat {C = Graphs o ℓ} {P = is-oriented}

Forget-oriented-is-ff : ∀ {o ℓ} → is-fully-faithful (Forget-oriented {o} {ℓ})
Forget-oriented-is-ff = id-equiv

{-|
## Derived Notions for Oriented Graphs

For oriented graphs, we define network-specific terminology:
- **Layers**: Vertices represent network layers
- **Connections**: Edges represent direct layer connections
- **Input layers**: Minimal elements (no incoming edges)
- **Output layers**: Maximal elements (no outgoing edges)
- **Convergent layers**: Multiple incoming edges (need fork construction)
-}

module OrientedGraph (G : Graph o ℓ) (oriented : is-oriented G) where
  open Graph G public
  open EdgePath G public

  -- Layers are vertices
  Layer : Type o
  Layer = Node

  -- Connections are edges
  Connection : Layer → Layer → Type ℓ
  Connection = Edge

  -- Reachability (with conventional direction)
  _≤ᴸ_ : Layer → Layer → Type (o ⊔ ℓ)
  _≤ᴸ_ = EdgePath

  -- Input layers: no incoming edges
  is-input : Layer → Type (o ⊔ ℓ)
  is-input x = ∀ y → ¬ (Connection y x)

  -- Output layers: no outgoing edges
  is-output : Layer → Type (o ⊔ ℓ)
  is-output x = ∀ y → ¬ (Connection x y)

  -- Convergent layer: multiple incoming edges
  -- We use propositional truncation to hide the witness
  is-convergent-witness : Layer → Type (o ⊔ ℓ)
  is-convergent-witness a = Σ[ x ∈ Layer ] Σ[ y ∈ Layer ]
                             (¬ (x ≡ y)) × Connection x a × Connection y a

  is-convergent : Layer → Type (o ⊔ ℓ)
  is-convergent a = ∥ is-convergent-witness a ∥

  -- Extract properties from orientation
  classical : ∀ {x y} → is-prop (Connection x y)
  classical = is-classical oriented

  no-loops : ∀ {x} → ¬ (Connection x x)
  no-loops = has-no-loops oriented

  acyclic : ∀ {x y} → EdgePath x y → EdgePath y x → x ≡ y
  acyclic = is-acyclic oriented
