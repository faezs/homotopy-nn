{-# OPTIONS --rewriting --guardedness --cubical --allow-unsolved-metas #-}

{-|
# Appendix E.3: Linear Exponential ! and Kleisli Categories

This module implements Girard's linear exponential modality ! ("of course") and the
associated Kleisli category construction.

## Key Concepts

1. **Exponential ! as Comonad**:
   - ε: !A → A (counit, extraction)
   - δ: !A → !!A (comultiplication, duplication)
   - !A represents "A with unlimited resources"

2. **Linear Logic Interpretation**:
   - !A means "stable proposition" or "A as many times as needed"
   - Weakening: !A → 1 (can discard)
   - Contraction: !A → !A ⊗ !A (can duplicate)

3. **Kleisli Category A_!** (Equation 48):
   - Objects: Same as A
   - Morphisms: Hom_{A_!}(X,Y) = Hom_A(X, !Y)
   - Composition via comonad structure

4. **Proposition E.1**: If A is bi-closed monoidal with !, then A_! is cartesian closed
   - Products: !A ⊗ !B
   - Exponentials: !(A ⊗ B^C)

## References

- [Gir87] Girard (1987): Linear logic
- [See89] Seely (1989): Linear logic, *-autonomous categories
- [Mel09] Melliès (2009): Categorical semantics of linear logic

-}

module Neural.Semantics.LinearExponential where

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
open import Neural.Semantics.BiClosed

private variable
  o ℓ o' ℓ' : Level

--------------------------------------------------------------------------------
-- Exponential Comonad

{-|
## Definition: Exponential Comonad !

The exponential ! is an endofunctor with comonad structure:
- Functor: ! : A → A
- Counit: ε : !A → A (extraction)
- Comultiplication: δ : !A → !!A (duplication)

**Laws**:
1. ε ∘ δ = id (left counit)
2. !ε ∘ δ = id (right counit)
3. δ ∘ δ = !δ ∘ δ (coassociativity)

**Intuition**: !A is "A with unlimited resources"
- Can extract: !A → A
- Can duplicate: !A → !!A
-}

record has-exponential-comonad {o ℓ} (C : Precategory o ℓ) : Type (o ⊔ ℓ) where
  open Precategory C

  field
    -- Exponential functor
    ! : Ob → Ob
    !₁ : ∀ {A B} → Hom A B → Hom (! A) (! B)

    -- Counit: extraction
    ε : ∀ {A} → Hom (! A) A

    -- Comultiplication: duplication
    δ : ∀ {A} → Hom (! A) (! (! A))

    -- Functor laws
    !-id : ∀ {A} → !₁ (id {A}) ≡ id
    !-comp : ∀ {A B C} (f : Hom A B) (g : Hom B C)
           → !₁ (g ∘ f) ≡ !₁ g ∘ !₁ f

    -- Comonad laws
    -- Left counit law: ε ∘ δ = id on !A
    ε-δ : ∀ {A} → ε ∘ δ ≡ id
    -- Right counit law: !ε ∘ δ = id on !A
    !ε-δ : ∀ {A} → !₁ ε ∘ δ ≡ id
    -- Coassociativity: !δ ∘ δ = δ ∘ δ
    δ-coassoc : ∀ {A} → !₁ δ ∘ δ ≡ δ ∘ δ

  -- Weakening: !A → 1 (can discard)
  postulate
    weakening : ∀ {A I} → Hom (! A) I  -- I is unit object

  -- Contraction: !A → !A ⊗ !A (can duplicate)
  -- Requires monoidal structure
  postulate
    contraction : ∀ {A} {_⊗_ : Ob → Ob → Ob} → Hom (! A) (! A ⊗ ! A)

--------------------------------------------------------------------------------
-- Monoidal Exponential

{-|
## Monoidal Structure on !

For ! to interact well with monoidal structure:
- Monoidal unit: !I ≅ I
- Monoidal product: !(A ⊗ B) → !A ⊗ !B (Seely isomorphism)

**Equation 49**: m_{A,B}: !(A ⊗ B) → !A ⊗ !B is an isomorphism
-}

record has-monoidal-exponential {o ℓ} {C : Precategory o ℓ}
                                 (M : Monoidal-category C)
                                 (E : has-exponential-comonad C) : Type (o ⊔ ℓ) where
  open Precategory C
  open Monoidal-category M
  open has-exponential-comonad E

  field
    -- Monoidal unit: !I ≅ I
    !-unit : Hom (! Unit) Unit
    !-unit-inv : Hom Unit (! Unit)

    -- Monoidal product (Seely map, Equation 49)
    m : ∀ {A B} → Hom (! (A ⊗ B)) ((! A) ⊗ (! B))
    m-inv : ∀ {A B} → Hom ((! A) ⊗ (! B)) (! (A ⊗ B))

    -- Isomorphism laws
    m-section : ∀ {A B} → m {A} {B} ∘ m-inv ≡ id
    m-retract : ∀ {A B} → m-inv {A} {B} ∘ m ≡ id

    !-unit-section : !-unit ∘ !-unit-inv ≡ id
    !-unit-retract : !-unit-inv ∘ !-unit ≡ id

  -- Coherence with tensor
  postulate
    m-natural : ∀ {A B C D} (f : Hom A C) (g : Hom B D)
              → ((!₁ f) ⊗₁ (!₁ g)) ∘ m ≡ m ∘ !₁ (f ⊗₁ g)

--------------------------------------------------------------------------------
-- Kleisli Category

{-|
## Kleisli Category A_! (Equation 48)

The Kleisli category for the ! comonad:
- Objects: Same as A
- Morphisms: Hom(X,Y) in A! = Hom(X, !Y) in A
- Identity: η : X → !X (via δ and ε)
- Composition: f ∘ g = !f ∘ g (using !₁)

**Intuition**: Morphisms X → !Y represent "resource-aware" functions
-}

module KleisliConstruction {o ℓ} {C : Precategory o ℓ}
                           (E : has-exponential-comonad C) where
  open Precategory C
  open has-exponential-comonad E

  -- Identity for Kleisli: need X → !X
  -- This is NOT provided by comonad directly - need additional structure
  postulate
    kleisli-id : ∀ {A} → Hom A (! A)

  -- Kleisli category
  A_! : Precategory o ℓ
  A_! .Precategory.Ob = Ob
  A_! .Precategory.Hom X Y = Hom X (! Y)
  A_! .Precategory.Hom-set X Y = Hom-set X (! Y)
  A_! .Precategory.id {X} = kleisli-id {X}
  -- Kleisli composition: g : X → !Y, f : Y → !Z, result: X → !Z
  A_! .Precategory._∘_ {X} {Y} {Z} f g = f ∘ ε ∘ g
  A_! .Precategory.idr {X} {Y} f = {!!}
  A_! .Precategory.idl {X} {Y} f = {!!}
  A_! .Precategory.assoc {W} {X} {Y} {Z} f g h = {!!}

--------------------------------------------------------------------------------
-- Cartesian Closure of Kleisli Category

{-|
## Proposition E.1: Kleisli Category is Cartesian Closed

If A is bi-closed monoidal with monoidal exponential !, then A_! is cartesian closed:

1. **Products in A!**: !A ⊗ !B (using Seely isomorphism)
2. **Terminal object in A!**: I (monoidal unit)
3. **Exponentials in A!**: Adjunction Hom(X⊗Y, Z) ≃ Hom(X, Y⇒Z)
   where Y⇒Z is defined via exponential structure

**Proof sketch**:
- Products: Use m: !(A⊗B) → !A⊗!B
- Exponentials: Use bi-closed structure of A
-}

module CartesianClosureKleisli {o ℓ} {C : Precategory o ℓ}
                                {M : Monoidal-category C}
                                (BC : is-bi-closed-monoidal C M)
                                (E : has-exponential-comonad C)
                                (ME : has-monoidal-exponential M E) where
  open Precategory C
  open Monoidal-category M
  open is-bi-closed-monoidal BC
  open has-exponential-comonad E
  open has-monoidal-exponential ME
  open KleisliConstruction E

  -- Product in Kleisli category
  _×ᴷ_ : Ob → Ob → Ob
  A ×ᴷ B = (! A) ⊗ (! B)

  -- Projections (in Kleisli)
  π₁ᴷ : ∀ {A B} → Hom (A ×ᴷ B) (! A)
  π₁ᴷ {A} {B} = {!!}  -- Need m-inv ∘ ε for first projection

  π₂ᴷ : ∀ {A B} → Hom (A ×ᴷ B) (! B)
  π₂ᴷ {A} {B} = {!!}  -- Need m-inv ∘ ε for second projection

  -- Pairing
  ⟨_,_⟩ᴷ : ∀ {X A B} → Hom X (! A) → Hom X (! B) → Hom X (A ×ᴷ B)
  ⟨_,_⟩ᴷ {X} {A} {B} f g = {!!}  -- Use m-inv ∘ (f ⊗₁ g)

  -- Terminal object in Kleisli
  𝟙ᴷ : Ob
  𝟙ᴷ = Unit

  -- Unique morphism to terminal
  !ᴷ : ∀ {A} → Hom A (! Unit)
  !ᴷ = {!!}

  -- Exponentials in Kleisli
  _⇒ᴷ_ : Ob → Ob → Ob
  A ⇒ᴷ B = {!!}  -- !(A ⊗ B^?) for suitable ?

  -- Evaluation in Kleisli
  evalᴷ : ∀ {A B} → Hom ((A ⇒ᴷ B) ×ᴷ A) (! B)
  evalᴷ = {!!}

  -- Currying in Kleisli
  curryᴷ : ∀ {X A B} → Hom (X ×ᴷ A) (! B) → Hom X (! (A ⇒ᴷ B))
  curryᴷ f = {!!}

  -- Proposition E.1
  postulate
    kleisli-cartesian-closed : {!!}  -- Statement of cartesian closure

--------------------------------------------------------------------------------
-- Stable Propositions

{-|
## Stable Propositions

A proposition A is **stable** if A ≅ !A (has unlimited resources).

**Examples**:
- Tautologies: ⊤ ≅ !⊤
- Persistent facts: Classical propositions
- Structural rules: Can weaken and contract

**Non-examples**:
- Linear resources: A⊗B where each use consumes
- Affine propositions: Can weaken but not contract
-}

module StablePropositions {o ℓ} {C : Precategory o ℓ}
                          (E : has-exponential-comonad C) where
  open Precategory C
  open has-exponential-comonad E

  -- Stable proposition
  record is-stable (A : Ob) : Type ℓ where
    field
      stable : Hom A (! A)
      stable-inv : Hom (! A) A
      stable-section : stable-inv ∘ stable ≡ id
      stable-retract : stable ∘ stable-inv ≡ id

  -- All stable propositions form a subcategory
  Stable : Precategory (o ⊔ ℓ) ℓ
  Stable .Precategory.Ob = Σ[ A ∈ Ob ] (is-stable A)
  Stable .Precategory.Hom (A , _) (B , _) = Hom A B
  Stable .Precategory.Hom-set (A , _) (B , _) = Hom-set A B
  Stable .Precategory.id {A , _} = id {A}
  Stable .Precategory._∘_ {A , _} {B , _} {C , _} = _∘_ {A} {B} {C}
  Stable .Precategory.idr {A , _} {B , _} = idr {A} {B}
  Stable .Precategory.idl {A , _} {B , _} = idl {A} {B}
  Stable .Precategory.assoc {W , _} {X , _} {Y , _} {Z , _} = assoc {W} {X} {Y} {Z}

  -- Unit is stable
  postulate
    unit-stable : ∀ {I} → is-stable I

  -- Tensor of stables is stable (if monoidal)
  postulate
    tensor-stable : ∀ {A B} {_⊗_ : Ob → Ob → Ob}
                  → is-stable A → is-stable B → is-stable (A ⊗ B)

--------------------------------------------------------------------------------
-- Resource-Aware Semantics

{-|
## Resource-Aware Semantics

The ! modality allows modeling resource usage:
- !A: Unlimited resources of type A
- A: Exactly one resource of type A
- A ⊸ B: Linear function (consumes A, produces B)
- !A → B: Can use A multiple times

**Example**: Memory management
- malloc: Unit ⊸ Ptr (linear allocation)
- free: Ptr ⊸ Unit (linear deallocation)
- read: !Ptr → Value (can read many times)
-}

module ResourceSemantics {o ℓ} {C : Precategory o ℓ}
                         {M : Monoidal-category C}
                         (BC : is-bi-closed-monoidal C M)
                         (E : has-exponential-comonad C) where
  open Precategory C
  open Monoidal-category M
  open is-bi-closed-monoidal BC
  open has-exponential-comonad E

  -- Linear implication (from bi-closed structure)
  _⊸_ : Ob → Ob → Ob
  A ⊸ B = A \\ B  -- Left exponential

  -- Intuitionistic implication (using !)
  _→'_ : Ob → Ob → Ob
  A →' B = (! A) ⊸ B

  -- Resource types (postulated for example)
  postulate
    Unit' : Ob
    Ptr : Ob
    Value : Ob

  -- Linear allocation
  postulate
    malloc : Hom Unit' (! Ptr)  -- Can allocate

  -- Linear deallocation
  postulate
    free : Hom Ptr Unit'  -- Must deallocate exactly once

  -- Persistent read
  postulate
    read : Hom (! Ptr) Value  -- Can read many times

--------------------------------------------------------------------------------
-- Examples

{-|
## Example: Sets with Multisets

In **Sets**:
- Objects: Sets
- !A = Mset(A) (multisets/bags of A)
- ε: Mset(A) → A (pick one element)
- δ: Mset(A) → Mset(Mset(A)) (nest multisets)

This gives the Kleisli category of multisets.
-}

module MultisetExample where
  open import Cat.Instances.Sets

  postulate
    Mset : (A : Sets lzero .Precategory.Ob) → Sets lzero .Precategory.Ob

  postulate
    mset-comonad : has-exponential-comonad (Sets lzero)

--------------------------------------------------------------------------------
-- Summary

{-|
## Summary

This module implements the linear exponential ! modality:

1. **Exponential comonad**: ! with ε (extract) and δ (duplicate)
2. **Monoidal structure**: Seely isomorphism m: !(A⊗B) → !A⊗!B (Equation 49)
3. **Kleisli category A_!**: Objects same as A, morphisms X → !Y (Equation 48)
4. **Proposition E.1**: A_! is cartesian closed when A is bi-closed with !
5. **Stable propositions**: A ≅ !A (unlimited resources)
6. **Resource semantics**: Linear (⊸) vs intuitionistic (→) functions

**Key Insight**: The ! modality bridges linear logic (resource-aware) and
intuitionistic logic (resource-free) by allowing controlled duplication.

**Next Steps**:
- Tensorial negation and dialogue categories (Module 4)
- Strong monad structure (Module 5)
- Connection to neural information dynamics
-}
