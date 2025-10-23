{-|
# Paths in Graphs

This module re-exports 1Lab's path infrastructure from Cat.Instances.Free.

## Why Re-export?

1Lab already provides a complete implementation of paths in graphs with:
- `Path-in G x y`: Paths from vertex x to y
- `path-is-set`: Proof that paths form a set (using encode-decode method)
- `_++_`: Path concatenation with associativity and identity laws
- `Path-category`: The free category on a graph

We re-export these with names suitable for graph-theoretic reasoning.

-}

module Neural.Graph.Path where

open import Neural.Graph.Base

open import Cat.Instances.Free public
  using ( Path-in
        ; nil
        ; cons
        ; _++_
        ; ++-idr
        ; ++-assoc
        ; path-is-set
        ; Path-category
        )

open import 1Lab.Prelude
open import Data.Nat.Base

private variable
  o ℓ : Level

{-|
## Convenient Aliases for Graph-Theoretic Reasoning

We provide aliases that make more sense in a graph context.
-}

module EdgePath (G : Graph o ℓ) where
  open Graph G

  -- Alias Path-in with clearer name
  EdgePath : Node → Node → Type (o ⊔ ℓ)
  EdgePath = Path-in G

  -- Re-export constructors
  path-nil : ∀ {x} → EdgePath x x
  path-nil = nil

  path-cons : ∀ {x y z} → Edge x y → EdgePath y z → EdgePath x z
  path-cons = cons

  -- Edge paths form a set (from 1Lab - no holes!)
  EdgePath-is-set : ∀ {x y} → is-set (EdgePath x y)
  EdgePath-is-set = path-is-set G

  -- Path concatenation
  path-concat : ∀ {x y z} → EdgePath x y → EdgePath y z → EdgePath x z
  path-concat = _++_

  -- Right identity
  path-concat-idr : ∀ {x y} (p : EdgePath x y) → p ++ nil ≡ p
  path-concat-idr = ++-idr

  -- Associativity
  path-concat-assoc : ∀ {x y z w} (p : EdgePath x y) (q : EdgePath y z) (r : EdgePath z w)
                    → (p ++ q) ++ r ≡ p ++ (q ++ r)
  path-concat-assoc = ++-assoc

  -- Path length
  path-length : ∀ {x y} → EdgePath x y → Nat
  path-length nil = 0
  path-length (cons _ p) = suc (path-length p)

  -- Non-trivial path (has at least one edge)
  is-non-trivial : ∀ {x y} → EdgePath x y → Type
  is-non-trivial nil = ⊥
  is-non-trivial (cons _ _) = ⊤

  {-|
  ## Reachability Relation

  x is reachable from y (written y ≤ᴾ x) if there exists a path from y to x.
  The superscript P distinguishes this from natural number ordering.
  -}
  _≤ᴾ_ : Node → Node → Type (o ⊔ ℓ)
  x ≤ᴾ y = EdgePath x y

  infix 20 _≤ᴾ_

  ≤ᴾ-refl : ∀ {x} → x ≤ᴾ x
  ≤ᴾ-refl = nil

  ≤ᴾ-trans : ∀ {x y z} → x ≤ᴾ y → y ≤ᴾ z → x ≤ᴾ z
  ≤ᴾ-trans = _++_
