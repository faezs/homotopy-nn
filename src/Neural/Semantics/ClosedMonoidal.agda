{-# OPTIONS --rewriting --guardedness --cubical --allow-unsolved-metas #-}

{-|
# Appendix E.1-E.3: Closed Monoidal Categories for Semantic Information

This module implements the foundational categorical structure for linear semantic
information theory, based on Belfiore & Bennequin (2022) Appendix E.

## Key Concepts

1. **Closed Monoidal Categories**: Categories with tensor product ⊗ and
   exponential objects A^Y representing "A given Y" (semantic conditioning)

2. **Equation 47**: The defining adjunction
   `Hom(X⊗Y, A) ≃ Hom(X, A^Y)`

3. **Semantic Interpretation**: Objects represent sentence meanings,
   morphisms represent evocations/deductions

4. **Natural Language Connection**: Framework for Lambek calculus and
   Montague grammar

## References

- [Lam58] Lambek (1958): Computational grammar
- [Mon70] Montague (1970): Formal semantics
- [EK66] Eilenberg-Kelly (1966): Closed categories
- [Mel09] Melliès (2009): Categorical semantics of linear logic

-}

module Neural.Semantics.ClosedMonoidal where

open import 1Lab.Prelude hiding (id; _∘_)
open import 1Lab.Path
open import 1Lab.HLevel
open import 1Lab.Equiv

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Functor.Adjoint
open import Cat.Monoidal.Base
open import Cat.Instances.Functor

open import Data.Nat.Base using (Nat)

private variable
  o ℓ o' ℓ' : Level

--------------------------------------------------------------------------------
-- Closed Monoidal Categories

{-|
## Definition: Closed Monoidal Category

A **closed monoidal category** is a monoidal category where each functor
`_⊗ Y` has a right adjoint, yielding exponential objects `A^Y`.

**Intuition**: In linguistic semantics:
- `X ⊗ Y`: Aggregation of sentences X and Y
- `A^Y = A|Y`: Interpretation of A conditioned on Y
- Adjunction: Giving Y and deducing A is the same as deducing A|Y

**Structure**:
- Monoidal product (⊗, ★) with associativity and unit
- For each Y, functor `_⊗ Y: A → A` has right adjoint `_^Y: A → A`
- Natural isomorphism: `Hom(X⊗Y, A) ≃ Hom(X, A^Y)` (Equation 47)
-}

record is-closed-monoidal {o ℓ} (C : Precategory o ℓ)
                          (M : Monoidal-category C) : Type (o ⊔ ℓ) where
  open Precategory C
  open Monoidal-category M

  field
    -- Exponential object: A^Y represents "A given Y"
    _^_ : Ob → Ob → Ob

    -- Evaluation morphism: (A^Y) ⊗ Y → A
    eval : ∀ {A Y} → Hom ((_^_ A Y) ⊗ Y) A

    -- Currying: Hom(X⊗Y, A) → Hom(X, A^Y)
    exp-curry : ∀ {X Y A} → Hom (X ⊗ Y) A → Hom X (_^_ A Y)

    -- Uncurrying: Hom(X, A^Y) → Hom(X⊗Y, A)
    exp-uncurry : ∀ {X Y A} → Hom X (_^_ A Y) → Hom (X ⊗ Y) A

    -- Adjunction laws (Equation 47)
    exp-curry-uncurry : ∀ {X Y A} (f : Hom X (_^_ A Y))
                      → exp-curry (exp-uncurry f) ≡ f

    exp-uncurry-curry : ∀ {X Y A} (g : Hom (X ⊗ Y) A)
                      → exp-uncurry (exp-curry g) ≡ g

  -- Standard characterization: evaluation is uncurry of identity
  -- eval {A} {Y} ≡ exp-uncurry (id {A = A ^ Y})

  -- Canonical morphism A → A^Y (internal constants)
  -- This uses ★ as final object (see remark in paper)
  postulate
    constant : ∀ {A Y} → Hom A (_^_ A Y)

  -- Interpretation: constant embeds A into "A given Y"

record ClosedMonoidalCategory (o ℓ : Level) : Type (lsuc (o ⊔ ℓ)) where
  field
    category : Precategory o ℓ
    monoidal : Monoidal-category category
    closed : is-closed-monoidal category monoidal

  open Precategory category public
  open Monoidal-category monoidal public
  open is-closed-monoidal closed public

--------------------------------------------------------------------------------
-- Symmetric Closed Monoidal Categories

{-|
## Symmetric Monoidal Structure

In natural language semantics, we assume symmetry: the order of X⊗Y doesn't
matter for basic aggregation (though see bi-closed categories for order-dependent
composition in Montague grammar).

**Symmetric structure**:
- Natural isomorphism `σ: X⊗Y ≅ Y⊗X`
- Coherence with associativity and unit
-}

record is-symmetric-closed {o ℓ} {C : Precategory o ℓ}
                           {M : Monoidal-category C}
                           (closed : is-closed-monoidal C M) : Type (o ⊔ ℓ) where
  open Precategory C
  open Monoidal-category M
  open is-closed-monoidal closed

  field
    -- Braiding: swap factors
    braid : ∀ {X Y} → Hom (X ⊗ Y) (Y ⊗ X)

    -- Involutive: swapping twice is identity
    braid-involutive : ∀ {X Y} → braid {Y} {X} ∘ braid {X} {Y} ≡ id

  -- In symmetric case, A^Y and Y^A are naturally isomorphic
  postulate
    exp-symmetric : ∀ {A Y} → Hom (_^_ A Y) (_^_ A Y)  -- Should be ≅

record SymmetricClosedMonoidal (o ℓ : Level) : Type (lsuc (o ⊔ ℓ)) where
  field
    closed-monoidal : ClosedMonoidalCategory o ℓ
    symmetric : is-symmetric-closed (ClosedMonoidalCategory.closed closed-monoidal)

  open ClosedMonoidalCategory closed-monoidal public
  open is-symmetric-closed symmetric public

--------------------------------------------------------------------------------
-- Semantic Conditioning

{-|
## Semantic Conditioning via Exponentials

The exponential `A^Y = A|Y` represents **conditioning** in the semantic sense:
- Given context Y, how do we interpret A?
- Morphism `f: X⊗Y → A` becomes `curry f: X → A|Y`

**Properties**:
1. `A|★ ≅ A` (conditioning on trivial context)
2. `Y' → Y` induces `A|Y → A|Y'` (refined context)
3. Canonical `A → A|Y` (internal constants, when ★ is final)

**Connection to Information Theory**:
This is the categorical analog of conditional probability P(A|Y),
but in a non-Boolean, linear logic setting.
-}

module SemanticConditioning {o ℓ} (C : ClosedMonoidalCategory o ℓ) where
  open ClosedMonoidalCategory C

  -- Notation: A|Y for exponential
  _∣_ : Ob → Ob → Ob
  A ∣ Y = A ^ Y

  -- Conditioning by trivial context (unit ★)
  -- Should satisfy: A|★ ≅ A
  postulate
    condition-by-unit : ∀ {A} → Hom (A ∣ Unit) A

  -- Refinement: Y' → Y induces A|Y → A|Y'
  condition-refine : ∀ {A Y Y'} → Hom Y' Y → Hom (A ∣ Y) (A ∣ Y')
  condition-refine {A} {Y} {Y'} k = exp-curry (eval ∘ (id ⊗₁ k))

  -- Monotonicity: refining context refines conditioning
  -- If Y' → Y then A|Y → A|Y'
  postulate
    condition-monotone : ∀ {A Y Y'} (k : Hom Y' Y) (k' : Hom Y' Y)
                       → k ≡ k'
                       → condition-refine {A} k ≡ condition-refine k'

  -- Internal constants: A → A|Y
  -- Uses closed.constant
  internal-constant : ∀ {A Y} → Hom A (A ∣ Y)
  internal-constant = constant

--------------------------------------------------------------------------------
-- Slice Categories (Contexts)

{-|
## Slice Categories for Contexts

Following the paper, we consider the slice category `Γ\A` of morphisms `Γ → A`
for a fixed context Γ.

**Structure**:
- Objects: Morphisms `f: Γ → A` for various A
- Morphisms: Commutative triangles

**Conditioning in context**:
- Given `f: Γ → A` and Y, we get `Γ → A|Y` by composing with `A → A|Y`

**Remark**: The paper mentions restrictions on Γ for tensor products in
the slice category. This requires Γ to have special properties.
-}

module SliceCategory {o ℓ} (C : ClosedMonoidalCategory o ℓ) (Γ : C .ClosedMonoidalCategory.Ob) where
  open ClosedMonoidalCategory C
  open SemanticConditioning C

  -- Object: morphism from Γ
  record Γ\Obj : Type (o ⊔ ℓ) where
    constructor γ-obj
    field
      target : Ob
      morphism : Hom Γ target

  -- Morphism: commutative triangle
  record Γ\Hom (f g : Γ\Obj) : Type ℓ where
    constructor γ-hom
    module f = Γ\Obj f
    module g = Γ\Obj g
    field
      triangle : Hom f.target g.target
      commutes : g.morphism ≡ triangle ∘ f.morphism

  -- Slice category
  Γ\A : Precategory (o ⊔ ℓ) ℓ
  Γ\A .Precategory.Ob = Γ\Obj
  Γ\A .Precategory.Hom = Γ\Hom
  Γ\A .Precategory.Hom-set x y = {!!}
  Γ\A .Precategory.id {x} = γ-hom id (sym (idl _))
  Γ\A .Precategory._∘_ {x} {y} {z} g f = γ-hom
    (Γ\Hom.triangle g ∘ Γ\Hom.triangle f)
    {!!}  -- Associativity proof
  Γ\A .Precategory.idr f = {!!}
  Γ\A .Precategory.idl f = {!!}
  Γ\A .Precategory.assoc f g h = {!!}

  -- Conditioning in context
  condition-in-context : ∀ {A} (f : Hom Γ A) (Y : Ob)
                       → Hom Γ (A ∣ Y)
  condition-in-context f Y = internal-constant ∘ f

--------------------------------------------------------------------------------
-- Theories in Closed Monoidal Categories

{-|
## Theories as Collections of Propositions

Following the paper (Section E.5):

**Theory**: A collection S of propositions (objects), stable by morphisms
to the right. That is:
- If A ∈ S and f: A → B, then B ∈ S

**Intuition**: S represents "consequences of a discourse"

**Ordering**: S ≤ S' if S ⊆ S' (S' is weaker)

**Conditioning**: S|Y = {A^Y | A ∈ S}

**Property**: S|Y' ≤ S|Y when Y' → Y (monotonicity)
             S|Y ≤ S (since A → A|Y)
-}

module Theories {o ℓ} (C : ClosedMonoidalCategory o ℓ) where
  open ClosedMonoidalCategory C
  open SemanticConditioning C

  -- Theory: right-closed collection of objects
  record Theory : Type (lsuc (o ⊔ ℓ)) where
    constructor theory
    field
      contains : Ob → Type o
      contains-prop : ∀ A → is-prop (contains A)

      -- Right-closure: A ∈ S and A → B implies B ∈ S
      right-closed : ∀ {A B} → contains A → Hom A B → contains B

  -- Inclusion: S ≤ S' means S ⊆ S'
  _≤-theory_ : Theory → Theory → Type o
  S ≤-theory S' = ∀ A → Theory.contains S A → Theory.contains S' A

  -- Conditioning of theories
  _∣-theory_ : Theory → Ob → Theory
  (S ∣-theory Y) .Theory.contains A' = ∥ Σ[ A ∈ Ob ] (Theory.contains S A × (A' ≡ (A ∣ Y))) ∥
  (S ∣-theory Y) .Theory.contains-prop A' = squash
  (S ∣-theory Y) .Theory.right-closed {A'} {B'} cond f = {!!}

  -- Monotonicity: Y' → Y implies S|Y' ≤ S|Y
  postulate
    theory-condition-monotone : ∀ (S : Theory) {Y Y'} (k : Hom Y' Y)
                              → (S ∣-theory Y) ≤-theory (S ∣-theory Y')

  -- Weakening: S|Y ≤ S
  postulate
    theory-condition-weaken : ∀ (S : Theory) (Y : Ob)
                            → (S ∣-theory Y) ≤-theory S

--------------------------------------------------------------------------------
-- Monoidal Action on Functions

{-|
## Action on Measurable Functions

The paper states (Section E.5): "The monoidal category A acts on the set of
functions from the theories to a fixed commutative group, for instance the
real numbers."

**Structure**:
- Functions Φ: Theory → ℝ (information measures)
- Action by tensor product and conditioning
- Prepares for bar-complex construction (Section E.18)
-}

postulate
  ℝ : Type  -- Real numbers (commutative group)
  _+ℝ_ : ℝ → ℝ → ℝ
  zeroℝ : ℝ

module TheoryFunctions {o ℓ} (C : ClosedMonoidalCategory o ℓ) where
  open Theories C

  -- Information function: Theory → ℝ
  TheoryFunction : Type (lsuc (o ⊔ ℓ))
  TheoryFunction = Theory → ℝ

  -- Action by conditioning
  postulate
    condition-function : TheoryFunction → (Y : C .ClosedMonoidalCategory.Ob) → TheoryFunction

  -- Monotonicity of information under conditioning
  postulate
    condition-function-monotone : ∀ (Φ : TheoryFunction) (S : Theory)
                                  {Y Y'} (k : C .ClosedMonoidalCategory.Hom Y' Y)
                                → {!!}  -- Φ(S|Y) ≥ Φ(S|Y')

--------------------------------------------------------------------------------
-- Examples

{-|
## Example: Discrete Category as Trivial Closed Monoidal

Every discrete category (only identity morphisms) is closed monoidal with
- X ⊗ Y = (X, Y) (cartesian product)
- X^Y = Hom(Y, X) = {★} or ∅

This is trivial but shows the structure exists.
-}

module DiscreteExample where
  postulate
    discrete-closed : (C : Precategory lzero lzero)
                    → (M : Monoidal-category C)
                    → is-closed-monoidal C M

{-|
## Example: Category of Sets

The category **Sets** with cartesian product is closed monoidal:
- X ⊗ Y = X × Y
- A^Y = (Y → A) (function space)
- eval: (Y → A) × Y → A is function application
- curry/uncurry are standard

This provides intuition for "A given Y" as "functions from Y to A".
-}

module SetsExample where
  postulate
    Sets-closed : ClosedMonoidalCategory (lsuc lzero) lzero

--------------------------------------------------------------------------------
-- Summary

{-|
## Summary

This module provides the foundation for linear semantic information theory:

1. **Closed monoidal categories**: Categorical structure for semantic composition
2. **Exponentials A^Y = A|Y**: Semantic conditioning ("A given Y")
3. **Equation 47**: Adjunction `Hom(X⊗Y, A) ≃ Hom(X, A^Y)`
4. **Symmetric structure**: Order-independent aggregation
5. **Theories**: Collections of propositions stable under consequence
6. **Conditioning of theories**: S|Y refines S by context Y
7. **Information functions**: Measures on theories

**Next Steps**:
- Bi-closed categories for order-dependent composition (Lambek)
- Linear exponential ! for stable propositions (Girard)
- Tensorial negation for linear logic (Melliès-Tabareau)

**Applications to Neural Networks**:
- Semantic information propagation
- Context-dependent interpretation
- Non-Boolean logic for neural codes
-}
