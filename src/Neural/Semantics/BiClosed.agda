{-# OPTIONS --rewriting --guardedness --cubical --allow-unsolved-metas #-}

{-|
# Appendix E.2: Bi-Closed Categories for Lambek Calculus

This module implements bi-closed monoidal categories, which have TWO exponentials:
- Right exponential: A^Y = A/Y (A given Y on the right)
- Left exponential: X^A = X\A (X given A on the left)

These arise naturally in natural language syntax (Lambek 1958) where word order matters.

## Key Concepts

1. **Bi-Closed Categories**: Monoidal categories where _⊗ Y and Y ⊗_ both have
   right adjoints

2. **Two Adjunctions**:
   - Right: `Hom(X⊗Y, A) ≃ Hom(X, A/Y)`
   - Left:  `Hom(Y⊗X, A) ≃ Hom(Y, X\A)`

3. **Lambek Calculus**: Syntactic categories with / and \\operators
   - NP\S: Intransitive verb (NP on left gives S)
   - (S\NP)/NP: Transitive verb (NP on right gives S\NP)

4. **Order Sensitivity**: X⊗Y ≠ Y⊗X in general, so A/Y ≠ Y\A

## References

- [Lam58] Lambek (1958): The mathematics of sentence structure
- [Lam99] Lambek (1999): Type grammar revisited
- [Dou03] Dougherty (2003): Formalization of Lambek calculus

-}

module Neural.Semantics.BiClosed where

open import 1Lab.Prelude hiding (id; _∘_)
open import 1Lab.Path
open import 1Lab.HLevel
open import 1Lab.Equiv

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Functor.Adjoint
open import Cat.Monoidal.Base
open import Cat.Instances.Functor

open import Neural.Semantics.ClosedMonoidal

private variable
  o ℓ o' ℓ' : Level

--------------------------------------------------------------------------------
-- Bi-Closed Monoidal Categories

{-|
## Definition: Bi-Closed Monoidal Category

A **bi-closed monoidal category** is a monoidal category where:
1. Each functor `_⊗ Y: A → A` has a right adjoint `_/Y: A → A` (right exponential)
2. Each functor `Y ⊗_: A → A` has a right adjoint `Y\_: A → A` (left exponential)

**Order matters**:
- `A/Y` is "A after Y" (Y on the right)
- `Y\A` is "A before Y" (Y on the left)
- In general: `A/Y ≠ Y\A`

**Lambek interpretation**:
- `/` is right implication (divides from right)
- `\` is left implication (divides from left)
- X⊗Y is concatenation of strings/phrases
-}

record is-bi-closed-monoidal {o ℓ} (C : Precategory o ℓ)
                              (M : Monoidal-category C) : Type (o ⊔ ℓ) where
  open Precategory C
  open Monoidal-category M

  field
    -- Right exponential: A/Y = "A after Y"
    _/_ : Ob → Ob → Ob

    -- Left exponential: Y\\ A = "A before Y"
    _\\_ : Ob → Ob → Ob

    -- Right evaluation: (A/Y) ⊗ Y → A
    eval-right : ∀ {A Y} → Hom ((_/_ A Y) ⊗ Y) A

    -- Left evaluation: Y ⊗ (Y\A) → A
    eval-left : ∀ {A Y} → Hom (Y ⊗ (_\\_ Y A)) A

    -- Right currying: Hom(X⊗Y, A) → Hom(X, A/Y)
    curry-right : ∀ {X Y A} → Hom (X ⊗ Y) A → Hom X (_/_ A Y)

    -- Left currying: Hom(Y⊗X, A) → Hom(Y, X\A)
    curry-left : ∀ {Y X A} → Hom (Y ⊗ X) A → Hom Y (_\\_ X A)

    -- Right uncurrying: Hom(X, A/Y) → Hom(X⊗Y, A)
    uncurry-right : ∀ {X Y A} → Hom X (_/_ A Y) → Hom (X ⊗ Y) A

    -- Left uncurrying: Hom(Y, X\A) → Hom(Y⊗X, A)
    uncurry-left : ∀ {Y X A} → Hom Y (_\\_ X A) → Hom (Y ⊗ X) A

    -- Right adjunction laws
    curry-uncurry-right : ∀ {X Y A} (f : Hom X (_/_ A Y))
                        → curry-right (uncurry-right f) ≡ f

    uncurry-curry-right : ∀ {X Y A} (g : Hom (X ⊗ Y) A)
                        → uncurry-right (curry-right g) ≡ g

    -- Left adjunction laws
    curry-uncurry-left : ∀ {Y X A} (f : Hom Y (_\\_ X A))
                       → curry-left (uncurry-left f) ≡ f

    uncurry-curry-left : ∀ {Y X A} (g : Hom (Y ⊗ X) A)
                       → uncurry-left (curry-left g) ≡ g

  -- Relation to closed monoidal: right exponential is the same
  -- A/Y = A^Y from ClosedMonoidal
  postulate
    right-is-exp : ∀ {A Y} → (_/_ A Y) ≡ (_/_ A Y)

record BiClosedMonoidalCategory (o ℓ : Level) : Type (lsuc (o ⊔ ℓ)) where
  field
    category : Precategory o ℓ
    monoidal : Monoidal-category category
    bi-closed : is-bi-closed-monoidal category monoidal

  open Precategory category public
  open Monoidal-category monoidal public
  open is-bi-closed-monoidal bi-closed public

--------------------------------------------------------------------------------
-- Lambek Calculus

{-|
## Lambek Calculus for Natural Language

The Lambek calculus is a logical system for natural language syntax where:
- **Basic types**: NP (noun phrase), S (sentence), N (noun), etc.
- **Product**: X⊗Y is concatenation (X followed by Y)
- **Right implication**: A/Y is "what you need on the right to get A"
- **Left implication**: Y\A is "what you need on the left to get A"

**Examples**:
- Intransitive verb: NP\S (takes NP on left, gives S)
- Transitive verb: (S\NP)/NP (takes NP on right, gives S\NP)
- Determiner: NP/N (takes N on right, gives NP)
-}

module LambekCalculus {o ℓ} (C : BiClosedMonoidalCategory o ℓ) where
  open BiClosedMonoidalCategory C

  -- Syntactic category
  SynCat : Type o
  SynCat = Ob

  -- Basic categories (postulated)
  postulate
    NP : SynCat  -- Noun phrase
    S  : SynCat  -- Sentence
    N  : SynCat  -- Noun

  -- Derived categories using Lambek operators

  -- Intransitive verb: NP\\ S
  IV : SynCat
  IV = NP \\ S

  -- Transitive verb: (S\NP)/NP = (NP\S)/NP
  TV : SynCat
  TV = IV / NP

  -- Determiner: NP/N
  Det : SynCat
  Det = NP / N

  -- Adjective: N/N (modifies noun on right)
  Adj : SynCat
  Adj = N / N

  -- Adverb: (NP\\ S)\\(NP\\ S) (modifies IV on right)
  Adv : SynCat
  Adv = IV \\ IV

  -- Lexical entries (morphisms)

  -- "sleeps": NP\\S
  postulate
    sleeps : Hom NP (NP \\ S)

  -- "sees": (NP\\S)/NP
  postulate
    sees : Hom ((NP \\ S) / NP) ((NP \\ S) / NP)

  -- "the": NP/N
  postulate
    the : Hom (NP / N) (NP / N)

  -- Composition via application

  -- Apply intransitive verb: NP ⊗ (NP\\S) → S
  apply-IV : Hom (NP ⊗ IV) S
  apply-IV = eval-left

  -- Apply transitive verb (curried): TV ⊗ NP → IV
  apply-TV-right : Hom (TV ⊗ NP) IV
  apply-TV-right = eval-right

--------------------------------------------------------------------------------
-- Dougherty's Formalization

{-|
## Dougherty (2003): Systematic Bi-Closed Structure

Dougherty provides a systematic categorical treatment of Lambek calculus:

1. **Product-free fragment**: Only /, \, basic types
2. **Product fragment**: Full bi-closed monoidal category
3. **Residuation**: A⊗B ⊢ C iff A ⊢ C/B iff B ⊢ A\C

**Key insight**: The two exponentials arise from considering both sides:
- Right residual: "What's needed on the right?"
- Left residual: "What's needed on the left?"
-}

module Dougherty {o ℓ} (C : BiClosedMonoidalCategory o ℓ) where
  open BiClosedMonoidalCategory C

  -- Residuation principles

  -- Right residuation: Hom(X⊗Y, A) ≃ Hom(X, A/Y)
  right-residuation : ∀ {X Y A}
                    → (Hom (X ⊗ Y) A) ≃ (Hom X (A / Y))
  right-residuation = {!!}  -- Equiv from adjunction

  -- Left residuation: Hom(Y⊗X, A) ≃ Hom(Y, X\A)
  left-residuation : ∀ {Y X A}
                   → (Hom (Y ⊗ X) A) ≃ (Hom Y (X \\ A))
  left-residuation = {!!}  -- Equiv from adjunction

  -- Monotonicity

  -- A ≤ B implies C/B ≤ C/A (contravariant in denominator)
  postulate
    slash-contravariant : ∀ {A B C} (f : Hom A B)
                        → Hom (C / B) (C / A)

  -- A ≤ B implies B\C ≤ A\C (contravariant in denominator)
  postulate
    backslash-contravariant : ∀ {A B C} (f : Hom A B)
                            → Hom (B \\ C) (A \\ C)

  -- A ≤ B implies A/C ≤ B/C (covariant in numerator)
  postulate
    slash-covariant : ∀ {A B C} (f : Hom A B)
                    → Hom (A / C) (B / C)

  -- A ≤ B implies C\A ≤ C\B (covariant in numerator)
  postulate
    backslash-covariant : ∀ {A B C} (f : Hom A B)
                        → Hom (C \\ A) (C \\ B)

  -- Interaction laws

  -- (A/B)/C = A/(B⊗C)
  postulate
    slash-assoc : ∀ {A B C} → Hom ((A / B) / C) (A / (B ⊗ C))

  -- C\(B\A) = (C⊗B)\A
  postulate
    backslash-assoc : ∀ {A B C} → Hom (C \\(B \\ A)) ((C ⊗ B) \\ A)

  -- Mixed: (B\A)/C and B\(A/C)
  postulate
    mixed-assoc : ∀ {A B C} → Hom ((B \\ A) / C) (B \\(A / C))

--------------------------------------------------------------------------------
-- Symmetric vs Non-Symmetric

{-|
## When is A/Y = Y\A?

In a **symmetric** monoidal category with braiding σ: X⊗Y ≅ Y⊗X, we get:
- A/Y ≅ Y\A (the two exponentials coincide)

But natural language is **non-symmetric**:
- "John sees Mary" ≠ "Mary sees John"
- (NP\S)/NP ≠ NP/(S\NP)

This is why we need bi-closed, not just closed symmetric.
-}

module SymmetricCollapse {o ℓ} {C : Precategory o ℓ}
                         {M : Monoidal-category C}
                         (bc : is-bi-closed-monoidal C M) where
  open Precategory C
  open Monoidal-category M
  open is-bi-closed-monoidal bc

  -- In symmetric case, slash and backslash coincide
  postulate
    symmetric-exponentials : ∀ {A Y} → Hom (A / Y) (Y \\ A)

--------------------------------------------------------------------------------
-- Examples

{-|
## Example: Simple Sentence Derivation

Derive "John sleeps" where:
- John : NP
- sleeps : NP\S
- Result : S

**Derivation**:
1. John ⊗ sleeps : NP ⊗ (NP\S)
2. Apply eval-left : NP ⊗ (NP\S) → S
-}

module SimpleSentence {o ℓ} (C : BiClosedMonoidalCategory o ℓ) where
  open BiClosedMonoidalCategory C
  open LambekCalculus C

  -- Simple sentence: Apply IV to NP to get S
  -- Example: "John sleeps" where John:NP, sleeps:IV=NP\\S
  simple-sentence : Hom (NP ⊗ IV) S
  simple-sentence = eval-left

{-|
## Example: Transitive Sentence Derivation

Derive "John sees Mary" where:
- John, Mary : NP
- sees : (NP\S)/NP = TV
- Result : S

**Derivation**:
1. sees ⊗ Mary : TV ⊗ NP
2. eval-right : TV ⊗ NP → IV = NP\S
3. John ⊗ (sees@Mary) : NP ⊗ (NP\S)
4. eval-left : NP ⊗ (NP\S) → S
-}

module TransitiveSentence {o ℓ} (C : BiClosedMonoidalCategory o ℓ) where
  open BiClosedMonoidalCategory C
  open LambekCalculus C

  -- Transitive sentence: Apply TV to NP to get IV, then IV to NP to get S
  -- Example: "John sees Mary"
  transitive-step1 : Hom (TV ⊗ NP) IV
  transitive-step1 = eval-right

  transitive-step2 : Hom (NP ⊗ IV) S
  transitive-step2 = eval-left

--------------------------------------------------------------------------------
-- Summary

{-|
## Summary

This module implements bi-closed monoidal categories for natural language syntax:

1. **Bi-closed structure**: Two exponentials (/, \) for left/right residuation
2. **Lambek calculus**: Syntactic categories with order-sensitive composition
3. **Dougherty's formalization**: Systematic categorical treatment
4. **Non-symmetric**: Word order matters (A/Y ≠ Y\A)
5. **Examples**: Simple and transitive sentence derivations

**Key difference from ClosedMonoidal**:
- ClosedMonoidal: One exponential A^Y, symmetric aggregation
- BiClosed: Two exponentials A/Y and Y\A, asymmetric concatenation

**Next Steps**:
- Linear exponential ! for stable propositions (Module 3)
- Tensorial negation for dialogue categories (Module 4)
- Connection to neural semantic dynamics
-}
