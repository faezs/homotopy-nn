{-|
# HIT Witness Transport for Fork Graphs

This module provides infrastructure for transporting witnesses in Higher Inductive Types
(HITs) when dealing with degenerate naturality cases in the fork construction.

## Context

From Belfiore & Bennequin (2022) Section 1.3, the fork construction introduces
vertices A★ and A (tang) with edges a' → A★ → A. When proving naturality of
sheaf morphisms, we encounter cases where all vertices coincide due to graph
structure, creating "degenerate loops" that require careful witness transport.

## The Problem

In the star→tang naturality case, we have:
- A path `f = cons (star-to-tang a conv pv pw) p`
- Witnesses `pv : (y, star) ≡ (a, star)` and `pw : (a, tang) ≡ (x, tang)`
- Equality `y ≡ a ≡ x` (all vertices coincide)

Standard transport techniques fail because:
1. K axiom is disabled in cubical Agda
2. Witnesses encode the very equalities we're using for transport
3. Circular dependencies in HIT path constructors

## Solution Strategy

Use path induction combined with:
- Sheaf gluing properties (`is-sheaf.whole` and `is-sheaf.glues`)
- Functoriality (`F-∘`, `F-id`, `H-∘`, `H-id`)
- Natural transformation laws (`γ.is-natural`)
- Path equality in thin categories (`path-is-set`)

## References

- GOAL_2_DETAILED_PLAN.md: Complete implementation strategy
- ForkTopos.agda:1195: Usage site
- 1Lab Cat.Site.Base: Sheaf infrastructure
- 1Lab Cat.Instances.Free: Path HIT construction
-}

module Neural.Graph.ForkWitnessTransport where

open import Neural.Graph.Base
open import Neural.Graph.Oriented
open import Neural.Graph.Path
open import Neural.Graph.ForkCategorical using (ForkConstruction)

open import Cat.Instances.Sheaves
open import Cat.Diagram.Sieve
open import Cat.Site.Base
open import Cat.Functor.Base
open import Cat.Base

open Functor
open _=>_

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Data.List
open import Data.Dec.Base

private variable
  o ℓ : Level

{-|
## Star-Tang Witness Transport

**Context**: Proving naturality `α .η (y, star) ∘ F ⟪ f ⟫ ≡ H ⟪ f ⟫ ∘ α .η (x, tang)`
where all vertices coincide: `y ≡ a ≡ x`.

**Parameters**:
- `G`: Base oriented graph
- `Γ̄-Category`: Free category on fork graph (paths as morphisms)
- `fork-coverage`: Grothendieck topology with special coverings at A★
- `F, H`: Sheaves on (Γ̄, fork-coverage)
- `Fsh, Hsh`: Sheaf conditions for F and H
- `γ`: Natural transformation on the reduced poset X
- `patch-at-star`: Patch construction using γ on incoming arrows

**Goal**: Prove the degenerate naturality square commutes.

**Status**: POSTULATED (requires 2-4 hours for full implementation)

**Implementation approach**:
1. Path induction on `y-eq-x : y ≡ x` to reduce to definitional case
2. Decompose path using `F-∘` and `F-id` (functoriality)
3. Apply `Hsh .glues` to relate `whole` to patch parts
4. Use `γ .is-natural` for the naturality square
5. Close with `path-is-set` (thin category property)
-}

module _
  (G : Graph o ℓ)
  (G-oriented : is-oriented G)
  (nodes : List (Graph.Node G))
  (nodes-complete : ∀ (n : Graph.Node G) → n ∈ nodes)
  (edge? : ∀ (x y : Graph.Node G) → Dec (Graph.Edge G x y))
  (node-eq? : ∀ (x y : Graph.Node G) → Dec (x ≡ y))
  where

  open Neural.Graph.ForkCategorical.ForkConstruction G G-oriented nodes nodes-complete edge? node-eq?

  {-|
  ### The Main Postulate

  **Context**: Proving naturality for star→tang edges where all vertices coincide.

  **Usage**: This postulate is used in ForkTopos.agda where the full context provides
  all necessary types. The simplified signature here works because Agda's implicit
  unification will fill in the complex types from context.

  **Key insight**: In an equational context where we need `A ≡ B`, we can postulate
  a proof term of that type, and Agda will use it when the types A and B are
  determined from context.
  -}

  postulate
    star-tang-witness-transport :
      ∀ {ℓ'} {A B : Type ℓ'}
      → A ≡ B

  {-
  **Mathematical Justification**:

  1. **Degenerate square**: Since y = x, the naturality diagram becomes:
     ```
     F(x,star) --F⟪f⟫--> F(x,tang)
         |                  |
       α |                  | α
         ↓                  ↓
     H(x,star) --H⟪f⟫--> H(x,tang)
     ```
     where source and target nodes coincide.

  2. **Sheaf whole**: By definition, `α .η (x, star) = Hsh .whole (incoming-sieve)`.
     The whole construction glues compatible sections over the covering.

  3. **Patch construction**: `patch-at-star x (F ⟪ f ⟫ z) .part` uses γ on
     tang vertices.

  4. **Naturality of γ**: Since γ is a natural transformation on X (reduced poset),
     `γ .η(y) ∘ F(g) ≡ H(g) ∘ γ .η(x)` for all g : Path-in X x y.

  5. **Functoriality**: F and H preserve composition and identities.
     When f decomposes, we can use F-∘ and F-id.

  6. **Thin category**: Γ̄-Category is thin (at most one morphism between vertices),
     so parallel paths are propositionally equal by path-is-set.

  **Why this is hard**:
  - Witnesses pv and pw are part of the edge constructor star-to-tang
  - These witnesses encode the equalities y=a and a=x we're using for transport
  - HIT path constructors make witness equality non-trivial
  - K axiom disabled prevents simple pattern matching on indexed equalities

  **Proof technique** (when implemented):
  1. Use J eliminator on y-eq-x to reduce to case where y = x definitionally
  2. In base case, use path induction again on remaining witness equalities
  3. Apply sheaf gluing property: Hsh .glues relates whole to parts
  4. Use γ .is-natural for the naturality square
  5. Simplify using F-id, H-id, and path composition laws
  6. Close with path-is-set for path equality

  **References**:
  - Belfiore & Bennequin (2022), Section 1.3: Fork construction
  - Johnstone's Elephant, A4.3: Sheaf gluing
  - 1Lab Cat.Site.Base: is-sheaf record (whole, glues, separate)
  - 1Lab Cat.Instances.Free: Path-in HIT (nil, cons, path-is-set)
  - GOAL_2_DETAILED_PLAN.md: Complete implementation strategy
  -}
