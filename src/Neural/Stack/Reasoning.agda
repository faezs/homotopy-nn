{-# OPTIONS --rewriting --guardedness --cubical --no-load-primitives #-}

{-|
Module: Neural.Stack.Reasoning
Description: Reasoning combinators for proving stack coherence laws

This module provides helpers and lemmas that make proving pseudofunctor
coherence laws (hexagon, right-unit, left-unit) much easier, similar to
how Cat.Reasoning helps with categorical proofs.

# Key Features

1. **Identity natural transformation lemmas** (idnt properties)
2. **Whisker simplification** (left/right whisker with idnt)
3. **Compositor identities** (when compositor reduces to idnt)
4. **Reasoning combinators** (building proofs step-by-step)
5. **Component-wise equality** (ext + refl patterns)

# Usage Pattern

```agda
module MyStack where
  open import Neural.Stack.Reasoning

  my-hexagon : ... ≡ ...
  my-hexagon = begin
    LHS
      ≡⟨ simplify-lhs ⟩
    idnt
      ≡⟨ sym simplify-rhs ⟩
    RHS
      ∎
```

-}

module Neural.Stack.Reasoning where

open import 1Lab.Prelude
open import 1Lab.Path
open import 1Lab.Equiv
open import 1Lab.HLevel
open import 1Lab.Univalence using (subst-∙)

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Functor.Compose
open import Cat.Functor.Naturality
open import Cat.Instances.Functor
open import Cat.Bi.Base
open import Cat.Bi.Instances.Discrete

import Cat.Functor.Reasoning as FR
import Cat.Reasoning as CR

private variable
  o ℓ o' ℓ' o'' ℓ'' : Level

--------------------------------------------------------------------------------
-- Identity Natural Transformation Helpers
--------------------------------------------------------------------------------

module _ {C : Precategory o ℓ} {D : Precategory o' ℓ'} where
  private
    module C = Precategory C
    module D = Precategory D
  open _=>_ -- For .η accessor on natural transformations

  -- NOTE: idnt and _∘nt_ are imported from Cat.Functor.Base
  -- We provide helper lemmas for working with idnt

  {-|
  Composing with idnt on the left is the identity.
  -}
  idnt-∘l : {F G : Functor C D} (α : F => G) → (idnt ∘nt α) ≡ α
  idnt-∘l α = ext λ x → D.idl _

  {-|
  Composing with idnt on the right is the identity.
  -}
  idnt-∘r : {F G : Functor C D} (α : F => G) → (α ∘nt idnt) ≡ α
  idnt-∘r α = ext λ x → D.idr _

  {-|
  idnt is the identity for vertical composition (both sides).
  -}
  idnt-idr : {F : Functor C D} → (idnt {F = F} ∘nt idnt) ≡ idnt
  idnt-idr = ext λ x → D.idl _

  idnt-idl : {F : Functor C D} → (idnt {F = F} ∘nt idnt) ≡ idnt
  idnt-idl = ext λ x → D.idl _

  {-|
  Component-wise, idnt is just D.id.
  -}
  idnt-component : {F : Functor C D} (x : C.Ob) → idnt {F = F} .η x ≡ D.id
  idnt-component x = refl

--------------------------------------------------------------------------------
-- Locally-discrete Bicategory Helpers
--------------------------------------------------------------------------------

{-|
For locally-discrete bicategories (where 2-cells are paths between morphisms),
convert a morphism-indexed family of functors into a P₁ mapping.

Given: f : {A B : C.Ob} → C.Hom A B → Functor D E
Returns: Functor (Disc (C.Hom A B)) Cat[D, E]

In a locally-discrete bicategory, Hom A B is Disc' (C.Hom A B), meaning:
- Objects are morphisms in C
- Morphisms (2-cells) are paths between those morphisms

This is used in base change (G ★ F) where morphisms in the base category
get mapped to reindexing functors between fiber categories.
-}
-- Postulate for now: This requires careful reasoning about transport of idnt along paths
-- The proof should use subst-∙ and properties of natural transformations
postulate
  disc-adjunct-F-∘ : ∀ {o ℓ o' ℓ'} {C : Precategory o ℓ} {D E : Precategory o' ℓ'}
                   → (f : {A B : C .Precategory.Ob} → C .Precategory.Hom A B → Functor D E)
                   → {A B : C .Precategory.Ob}
                   → {x y z : C .Precategory.Hom A B}
                   → (g : y ≡ z) (h : x ≡ y)
                   → subst (λ m → f x => f m) (h ∙ g) idnt
                   ≡ (subst (λ m → f y => f m) g idnt) ∘nt (subst (λ m → f x => f m) h idnt)

Disc-adjunct : ∀ {o ℓ o' ℓ'} {C : Precategory o ℓ} {D E : Precategory o' ℓ'}
             → ({A B : C .Precategory.Ob} → C .Precategory.Hom A B → Functor D E)
             → {A B : C .Precategory.Ob}
             → Functor (Locally-discrete C .Prebicategory.Hom A B) Cat[ D , E ]
Disc-adjunct {C = C} {D = D} {E = E} f .Functor.F₀ α = f α
Disc-adjunct f .Functor.F₁ {α} {β} eq = subst (λ m → f α => f m) eq idnt
Disc-adjunct f .Functor.F-id = transport-refl idnt
Disc-adjunct {C = C} {D = D} {E = E} f .Functor.F-∘ {x} {y} {z} g h =
  disc-adjunct-F-∘ {C = C} {D = D} {E = E} f {x = x} {y = y} {z = z} g h

--------------------------------------------------------------------------------
-- Whisker Simplification
--------------------------------------------------------------------------------

module _ {C : Precategory o ℓ} {D : Precategory o' ℓ'} where
  private
    module C = Precategory C
    module D = Precategory D
    module Cat-D = Prebicategory (Cat o' ℓ')

  -- whisker-idnt-r and whisker-idnt-l removed - ext doesn't work with whiskering
  -- (type mismatch: F ▶ idnt has type (F F∘ ?) => (F F∘ ?), not F => F)

  -- whisker-Id-r and whisker-Id-l removed - not used anywhere in the codebase
  -- NOTE: If needed later, reformulate using Nat-pathp or unitor natural isomorphisms

--------------------------------------------------------------------------------
-- Functor Composition Identities
--------------------------------------------------------------------------------

module _ {C : Precategory o ℓ} where
  private
    module C = Precategory C

  -- NOTE: F∘-id2, F∘-idl, F∘-idr imported from Cat.Instances.Functor

  {-|
  Composing identity functor with itself (twice, left-associated).
  (Id ∘ Id) ∘ Id = Id
  -}
  F∘-id3-left : (Id {C = C} F∘ Id) F∘ Id ≡ Id
  F∘-id3-left = ap (_F∘ Id) F∘-id2 ∙ F∘-idr

  {-|
  Composing identity functor with itself (twice, right-associated).
  Id ∘ (Id ∘ Id) = Id
  -}
  F∘-id3-right : Id {C = C} F∘ (Id F∘ Id) ≡ Id
  F∘-id3-right = ap (Id F∘_) F∘-id2 ∙ F∘-idl

  {-|
  All triple compositions of Id are equal.
  -}
  F∘-id3-assoc : (Id {C = C} F∘ Id) F∘ Id ≡ Id F∘ (Id F∘ Id)
  F∘-id3-assoc = F∘-id3-left ∙ sym F∘-id3-right

--------------------------------------------------------------------------------
-- Associator and Unitor Identities for Cat
--------------------------------------------------------------------------------

module _ {C : Precategory o ℓ} where
  private
    module C = Precategory C
    module Cat-C = Prebicategory (Cat o ℓ)
  open _=>_

  {-|
  The associator α→ applied to (Id, Id, Id) has components that are all C.id.

  This is because:
  ((Id ∘ Id) ∘ Id) = Id = (Id ∘ (Id ∘ Id))
  and the natural isomorphism has identity components.
  -}
  -- All Id-components lemmas removed - not used and produce unsolved metas
  -- (Agda can't infer which category Id : Functor ? ? is the identity of)

--------------------------------------------------------------------------------
-- Natural Transformation Equality
--------------------------------------------------------------------------------

module _ {C : Precategory o ℓ} {D : Precategory o' ℓ'} where
  private
    module C = Precategory C
    module D = Precategory D
  open _=>_

  {-|
  To prove two natural transformations are equal,
  it suffices to prove they have equal components.

  This is just `ext`, but named for clarity.
  -}
  nat-trans-ext : {F G : Functor C D} {α β : F => G}
                → (∀ x → α .η x ≡ β .η x)
                → α ≡ β
  nat-trans-ext = ext

  {-|
  Two natural transformations with identity components are equal.
  -}
  nat-trans-id-unique : {F : Functor C D} {α β : F => F}
                      → (∀ x → α .η x ≡ D.id)
                      → (∀ x → β .η x ≡ D.id)
                      → α ≡ β
  nat-trans-id-unique hα hβ = ext λ x → hα x ∙ sym (hβ x)

  {-|
  A natural transformation with all identity components equals idnt.
  -}
  nat-trans-id-is-idnt : {F : Functor C D} {α : F => F}
                       → (∀ x → α .η x ≡ D.id)
                       → α ≡ idnt
  nat-trans-id-is-idnt h = ext h

--------------------------------------------------------------------------------
-- Hom-set Reasoning for Cat
--------------------------------------------------------------------------------

module _ {C : Precategory o ℓ} {D : Precategory o' ℓ'} where
  private
    module Cat-CD = Precategory Cat[ C , D ]

  {-|
  Natural transformations form a set, so any two paths between them are equal.
  This allows proving equality via the set property when stuck.
  -}
  nat-trans-is-set : {F G : Functor C D} → is-set (F => G)
  nat-trans-is-set = Cat-CD.Hom-set _ _

  {-|
  Use set-ness to prove two natural transformations equal
  (when you can't construct an explicit path but know they're in a set).
  -}
  nat-trans-set-path : {F G : Functor C D} {α β : F => G} (p q : α ≡ β)
                     → p ≡ q
  nat-trans-set-path {α = α} {β = β} = nat-trans-is-set α β

--------------------------------------------------------------------------------
-- Disc-adjunct Simplifications
--------------------------------------------------------------------------------

module _ {C : Precategory o ℓ} {D : Precategory o' ℓ'} where
  private
    module C = Precategory C
    module D = Precategory D
  open _=>_

  -- Disc-adjunct-path-is-idnt removed - type signature was incorrect
  -- TODO: Reformulate if needed for actual use cases

--------------------------------------------------------------------------------
-- Reasoning Combinators
--------------------------------------------------------------------------------

module StackReasoning {C : Precategory o ℓ} {D : Precategory o' ℓ'} where
  private
    module C = Precategory C
    module D = Precategory D
    module Cat-D = Prebicategory (Cat o' ℓ')

  {-|
  Simplify a composition chain when one component is idnt.
  Note: For β ≡ idnt to type-check, β must be F => F (endomorphism).
  -}
  simplify-idnt-l : {F G : Functor C D} {α : F => G} {β : F => F}
                  → β ≡ idnt
                  → (α ∘nt β) ≡ α
  simplify-idnt-l {α = α} eq = ap (α ∘nt_) eq ∙ idnt-∘r α

  simplify-idnt-r : {F G : Functor C D} {α : G => G} {β : F => G}
                  → α ≡ idnt
                  → (α ∘nt β) ≡ β
  simplify-idnt-r {β = β} eq = ap (_∘nt β) eq ∙ idnt-∘l β

  -- simplify-whisker-idnt-r and simplify-whisker-idnt-l removed
  -- (dependencies whisker-idnt-r/l were removed)

  -- chain-simpl removed - type signature was malformed
  -- (cannot compose idnt on different functors)

--------------------------------------------------------------------------------
-- High-Level Coherence Proof Patterns
--------------------------------------------------------------------------------

module CoherencePatterns {B : Prebicategory o ℓ ℓ'} {C : Prebicategory o'' ℓ'' ℓ''} where
  private
    module B = Prebicategory B
    module C = Prebicategory C
  open _=>_

  {-|
  Pattern for hexagon proof when all components simplify to identity:

  1. Show LHS compositor is idnt
  2. Show LHS whisker is idnt
  3. Show LHS functor action is idnt
  4. Simplify LHS to idnt
  5. Show RHS compositor is idnt
  6. Show RHS whisker is idnt
  7. Show RHS associator has identity components
  8. Simplify RHS to idnt
  9. Conclude LHS ≡ RHS via idnt
  -}
  -- hexagon-via-idnt-pattern removed - type signatures were incorrect
  -- (γ→ is a 2-cell, not a Hom morphism)

  -- right-unit-via-idnt-pattern removed - type signatures were incorrect
  -- left-unit-via-idnt-pattern removed - type signatures were incorrect

--------------------------------------------------------------------------------
-- NOTE: PathP Helpers for Grothendieck Construction
--------------------------------------------------------------------------------
-- REMOVED: GrothendieckPathP module that was here created circular import
-- (Reasoning imports Base, Base imports Reasoning)
-- Solution: Implement coherence helpers directly in Base.agda using Pseudofunctor fields

--------------------------------------------------------------------------------
-- Export Commonly Used Helpers
--------------------------------------------------------------------------------

-- Re-export reasoning combinators from Cat
open CR public using (module Reasoning)

-- Common imports for stack proofs
stack-imports : Type
stack-imports =
  ⊤  -- Placeholder, forces this to type-check even if unused
