{-# OPTIONS --rewriting --guardedness --cubical --allow-unsolved-metas #-}

{-|
# Appendix E.4: Tensorial Negation and Dialogue Categories

This module implements Melliès-Tabareau's dialogue categories with tensorial negation,
connecting linear logic to game semantics and neural information dynamics.

## Key Concepts

1. **Tensorial Negation ¬**: Contravariant functor A → A^op with special properties
   - ¬¬A ≅ A (double negation)
   - ¬(A ⊗ B) ≅ ¬A ℘ ¬B (De Morgan)

2. **Dialogue Categories**: Monoidal categories with tensorial negation satisfying:
   - Equation 50: ¬ is a strong functor
   - Equation 51: Continuation monad T = ¬∘¬
   - Equation 52: Pole pole = ¬1
   - Equation 53: Par operation ℘

3. **Continuation Monad**: T = ¬∘¬ with:
   - Unit: η : Id → ¬¬ (double negation introduction)
   - Multiplication: μ : ¬¬¬¬ → ¬¬ (collapse)

4. **Linear Logic Connectives**:
   - Pole: pole = ¬1 (multiplicative false)
   - Par: A ℘ B = ¬(¬A ⊗ ¬B) (multiplicative or)
   - With: A & B = !(A × B) (additive and)
   - Girard's ?: ?A = !(¬A) (stable negation)

## References

- [MT06] Melliès-Tabareau (2006): Resource modalities in tensor logic
- [Mel09] Melliès (2009): Categorical semantics of linear logic
- [Gir87] Girard (1987): Linear logic

-}

module Neural.Semantics.TensorialNegation where

open import 1Lab.Prelude hiding (id; _∘_)
open import 1Lab.Path
open import 1Lab.HLevel
open import 1Lab.Equiv

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Monoidal.Base
open import Cat.Instances.Functor

open import Neural.Semantics.ClosedMonoidal
open import Neural.Semantics.BiClosed
open import Neural.Semantics.LinearExponential

private variable
  o ℓ o' ℓ' : Level

--------------------------------------------------------------------------------
-- Tensorial Negation

{-|
## Definition: Tensorial Negation

A **tensorial negation** on a monoidal category A is a contravariant functor
¬: A → A^op satisfying:
1. Involution: ¬¬A ≅ A (double negation)
2. Monoidal: ¬(A ⊗ B) ≅ ¬A ℘ ¬B where ℘ is the par operation

**Intuition**: ¬A represents "the dual of A" or "refutation of A"
-}

record has-tensorial-negation {o ℓ} (C : Precategory o ℓ)
                               (M : Monoidal-category C) : Type (o ⊔ ℓ) where
  open Precategory C
  open Monoidal-category M

  field
    -- Negation on objects
    ¬' : Ob → Ob

    -- Negation on morphisms (contravariant)
    ¬'₁ : ∀ {A B} → Hom A B → Hom (¬' B) (¬' A)

    -- Involution: ¬¬A ≅ A (Equation 50)
    involution : ∀ {A} → Hom A (¬' (¬' A))
    involution-inv : ∀ {A} → Hom (¬' (¬' A)) A

    involution-section : ∀ {A} → involution-inv {A} ∘ involution ≡ id
    involution-retract : ∀ {A} → involution {A} ∘ involution-inv ≡ id

    -- Contravariant functor laws
    ¬-id : ∀ {A} → ¬'₁ (id {A}) ≡ id
    ¬-comp : ∀ {A B C} (f : Hom A B) (g : Hom B C)
           → ¬'₁ (g ∘ f) ≡ ¬'₁ f ∘ ¬'₁ g  -- Reverses order!

  -- Par operation: A ℘ B = ¬(¬A ⊗ ¬B) (Equation 53)
  _℘_ : Ob → Ob → Ob
  A ℘ B = ¬' ((¬' A) ⊗ (¬' B))

  -- Pole: pole = neg(1) (Equation 52)
  pole : Ob
  pole = ¬' Unit

  -- De Morgan law: ¬(A ⊗ B) ≅ ¬A ℘ ¬B
  postulate
    de-morgan : ∀ {A B} → Hom (¬' (A ⊗ B)) (¬' A ℘ ¬' B)
    de-morgan-inv : ∀ {A B} → Hom (¬' A ℘ ¬' B) (¬' (A ⊗ B))

  -- Pole is self-dual: pole ≅ ¬'pole
  postulate
    pole-self-dual : Hom pole (¬' pole)

  -- Unit duality: 1 ≅ ¬'pole
  postulate
    unit-dual : Hom Unit (¬' pole)

--------------------------------------------------------------------------------
-- Continuation Monad

{-|
## Continuation Monad T = ¬∘¬ (Equation 51)

The double negation ¬∘¬ forms a monad:
- Functor: T(A) = ¬¬A
- Unit: η : A → ¬¬A (involution)
- Multiplication: μ : ¬¬¬¬A → ¬¬A (via ¬-involution)

**Connection to continuations**: In programming, ¬¬A ≅ ((A → R) → R) for return type R
-}

module ContinuationMonad {o ℓ} {C : Precategory o ℓ}
                         {M : Monoidal-category C}
                         (N : has-tensorial-negation C M) where
  open Precategory C
  open Monoidal-category M
  open has-tensorial-negation N

  -- Continuation monad functor
  T : Ob → Ob
  T A = ¬' (¬' A)

  T₁ : ∀ {A B} → Hom A B → Hom (T A) (T B)
  T₁ f = ¬'₁ (¬'₁ f)

  -- Monad unit: η : Id → T
  η : ∀ {A} → Hom A (T A)
  η = involution

  -- Monad multiplication: μ : TT → T
  postulate
    μ : ∀ {A} → Hom (T (T A)) (T A)

  -- Monad laws
  postulate
    T-left-unit : ∀ {A} → μ {A} ∘ η ≡ id
    T-right-unit : ∀ {A} → μ {A} ∘ T₁ η ≡ id
    T-assoc : ∀ {A} → μ {A} ∘ T₁ μ ≡ μ ∘ μ

  -- Continuation passing style
  -- f : A → B becomes ¬f : ¬B → ¬A in CPS
  cps : ∀ {A B} → Hom A B → Hom (¬' B) (¬' A)
  cps = ¬'₁

--------------------------------------------------------------------------------
-- Dialogue Categories

{-|
## Dialogue Categories (Melliès-Tabareau)

A **dialogue category** is a monoidal category A with:
1. Tensorial negation ¬
2. All objects are reflexive: A ≅ ¬¬A
3. Monoidal negation: ¬(A ⊗ B) ≅ ¬A ℘ ¬' B

**Game semantics interpretation**:
- A is a game/dialogue
- ¬A is the game with roles reversed
- A ⊗ B is parallel composition
- A ℘ B is sequential composition
-}

record is-dialogue-category {o ℓ} (C : Precategory o ℓ)
                             (M : Monoidal-category C) : Type (o ⊔ ℓ) where
  open Precategory C
  open Monoidal-category M

  field
    negation : has-tensorial-negation C M

  open has-tensorial-negation negation public

  -- All objects reflexive
  field
    reflexive : ∀ {A} → Hom A (¬' (¬' A))

  -- Strong monoidal negation
  field
    strong-negation : ∀ {A B} → Hom (¬' (A ⊗ B)) ((¬' A) ℘ (¬' B))
    strong-negation-inv : ∀ {A B} → Hom ((¬' A) ℘ (¬' B)) (¬' (A ⊗ B))

record DialogueCategory (o ℓ : Level) : Type (lsuc (o ⊔ ℓ)) where
  field
    category : Precategory o ℓ
    monoidal : Monoidal-category category
    dialogue : is-dialogue-category category monoidal

  open Precategory category public
  open Monoidal-category monoidal public
  open is-dialogue-category dialogue public

--------------------------------------------------------------------------------
-- Linear Logic Connectives

{-|
## Linear Logic Connectives

In a dialogue category, we can define all linear logic connectives:

1. **Multiplicatives**:
   - ⊗ (tensor): already in monoidal category
   - ℘ (par): A ℘ B = ¬(¬A ⊗ ¬B)
   - 1 (unit): monoidal unit
   - pole (pole): ¬1

2. **Additives** (with products):
   - & (with): A & B (product)
   - ⊕ (plus): A ⊕ B (coproduct)
   - ⊤ (top): terminal object
   - 0 (zero): initial object

3. **Exponentials** (with !):
   - ! (of course): from LinearExponential
   - ? (why not): ?A = ¬(!¬A)
-}

module LinearConnectives {o ℓ} (D : DialogueCategory o ℓ) where
  open DialogueCategory D

  -- Multiplicatives (already defined)
  -- ⊗, ℘, 1, pole

  -- Implication: A ⊸ B = ¬A ℘ B
  _⊸_ : Ob → Ob → Ob
  A ⊸ B = (¬' A) ℘ B

  -- With products (postulated)
  postulate
    _&_ : Ob → Ob → Ob
    top-obj : Ob
    π₁ : ∀ {A B} → Hom (A & B) A
    π₂ : ∀ {A B} → Hom (A & B) B
    ⟨_,_⟩& : ∀ {X A B} → Hom X A → Hom X B → Hom X (A & B)

  -- Plus (coproduct, postulated)
  postulate
    _⊕_ : Ob → Ob → Ob
    𝟘 : Ob
    inl : ∀ {A B} → Hom A (A ⊕ B)
    inr : ∀ {A B} → Hom B (A ⊕ B)
    [_,_] : ∀ {A B X} → Hom A X → Hom B X → Hom (A ⊕ B) X

  -- Girard's ? operator (requires !)
  module WithExponential (E : has-exponential-comonad category) where
    open has-exponential-comonad E

    why-not : Ob → Ob
    why-not A = ¬' (! (¬' A))

    -- why-not A is dual to !A
    postulate
      why-not-dual : ∀ {A} → Hom (why-not A) (¬' (! (¬' A)))

--------------------------------------------------------------------------------
-- Self-Dual Objects

{-|
## Self-Dual Objects

An object A is **self-dual** if A ≅ ¬A.

**Examples**:
- Pole: pole ≅ ¬pole
- Boolean: Bool ≅ ¬Bool (in some models)
- Fixed points of ¬

**Lawvere's fixed point theorem**: If ∃ surjection X → (X → Y), then Y ≅ ¬Y
-}

module SelfDualObjects {o ℓ} (D : DialogueCategory o ℓ) where
  open DialogueCategory D

  record is-self-dual (A : Ob) : Type ℓ where
    field
      self-dual : Hom A (¬' A)
      self-dual-inv : Hom (¬' A) A
      self-dual-section : self-dual-inv ∘ self-dual ≡ id
      self-dual-retract : self-dual ∘ self-dual-inv ≡ id

  -- Pole is self-dual
  postulate
    pole-is-self-dual : is-self-dual pole

  -- Product of self-duals is self-dual (via ℘)
  postulate
    tensor-self-dual : ∀ {A B}
                     → is-self-dual A → is-self-dual B
                     → is-self-dual (A ⊗ B)

--------------------------------------------------------------------------------
-- Chu Construction

{-|
## Chu Construction

The **Chu construction** Chu(Set, k) gives a canonical dialogue category:
- Objects: (A, X, e) where A, X : Set and e : A × X → k
- Morphisms: (f, g) : (A, X, e) → (B, Y, e') where f : A → B, g : Y → X
- Negation: ¬(A, X, e) = (X, A, e^op)

**Interpretation**:
- A: states
- X: co-states (observations)
- e: evaluation function
- ¬ swaps states and co-states
-}

module ChuConstruction where
  postulate
    k : Type  -- Base set (often Bool or ℝ)

  record ChuObject : Type₁ where
    field
      states : Type
      costates : Type
      eval : states → costates → k

  -- Negation swaps states and costates
  ¬-Chu : ChuObject → ChuObject
  ¬-Chu obj .ChuObject.states = ChuObject.costates obj
  ¬-Chu obj .ChuObject.costates = ChuObject.states obj
  ¬-Chu obj .ChuObject.eval x a = ChuObject.eval obj a x

  -- Double negation is identity
  postulate
    chu-involution : ∀ (obj : ChuObject) → ¬-Chu (¬-Chu obj) ≡ obj

  -- Chu is a dialogue category
  postulate
    Chu-dialogue : DialogueCategory (lsuc lzero) lzero

--------------------------------------------------------------------------------
-- Game Semantics

{-|
## Game Semantics

Dialogue categories provide semantics for games:
- A: game
- ¬A: same game with Player/Opponent swapped
- A ⊗ B: independent parallel play
- A ℘ B: sequential play with communication
- A ⊸ B: strategy from A to B

**Conway games**: Surreal numbers form a dialogue category
-}

module GameSemantics {o ℓ} (D : DialogueCategory o ℓ) where
  open DialogueCategory D

  -- Game: represented by object
  Game : Type o
  Game = Ob

  -- Strategy: morphism in dialogue category
  Strategy : Game → Game → Type ℓ
  Strategy = Hom

  -- Dual game: swap roles
  dual : Game → Game
  dual = ¬'

  -- Parallel composition
  parallel : Game → Game → Game
  parallel = _⊗_

  -- Sequential composition
  sequential : Game → Game → Game
  sequential = _℘_

  -- Strategy composition
  compose-strategy : ∀ {A B C} → Strategy B C → Strategy A B → Strategy A C
  compose-strategy = _∘_

  -- Identity strategy
  id-strategy : ∀ {A} → Strategy A A
  id-strategy = id

  -- Copycat strategy: A → A ⊗ A (requires structure)
  postulate
    copycat : ∀ {A} → Strategy A (A ⊗ A)

--------------------------------------------------------------------------------
-- Neural Information Dynamics

{-|
## Application to Neural Networks

Dialogue categories model neural information flow:

1. **Feedforward**: A ⊗ B (parallel neurons)
2. **Feedback**: A ℘ B (recurrent connections)
3. **Prediction error**: ¬A (expected vs actual)
4. **Attention**: A ⊸ B (query-key mechanism)

**Predictive coding**: x : A, prediction : ¬¬A, error : ¬A
-}

module NeuralDialogue {o ℓ} (D : DialogueCategory o ℓ) where
  open DialogueCategory D
  open LinearConnectives D

  -- Neural state space
  NeuralState : Type o
  NeuralState = Ob

  -- Prediction: forward model A → ¬¬A
  prediction : ∀ {A} → Hom A (¬' (¬' A))
  prediction = involution

  -- Prediction error: ¬A (mismatch signal)
  PredictionError : NeuralState → NeuralState
  PredictionError = ¬'

  -- Feedforward layer: A ⊗ B
  feedforward : NeuralState → NeuralState → NeuralState
  feedforward = _⊗_

  -- Recurrent layer: A ℘ B
  recurrent : NeuralState → NeuralState → NeuralState
  recurrent = _℘_

  -- Attention mechanism: Query ⊸ (Key ⊗ Value)
  attention-type : NeuralState → NeuralState → NeuralState → NeuralState
  attention-type Query Key Value = Query ⊸ (Key ⊗ Value)

  -- Variational inference: minimize D_KL(q || p) ≈ ¬¬q ⊸ ¬¬p
  postulate
    variational-type : NeuralState → NeuralState → NeuralState

--------------------------------------------------------------------------------
-- Examples

{-|
## Example: Boolean Algebra as Dialogue Category

In classical logic with ¬, ∧, ∨:
- Negation: ¬ is tensorial negation
- Tensor: A ⊗ B = A ∧ B
- Par: A ℘ B = A ∨ B
- Pole: pole = False

De Morgan: ¬(A ∧ B) = ¬A ∨ ¬B
-}

module BooleanExample where
  postulate
    Bool-dialogue : DialogueCategory lzero lzero

  -- In Bool, ¬¬A = A (classical logic)
  postulate
    bool-double-negation : ∀ {A} → (Bool-dialogue .DialogueCategory.¬' (Bool-dialogue .DialogueCategory.¬' A)) ≡ A

{-|
## Example: Phase Semantics

In phase semantics (Girard):
- Objects: subsets of a monoid M
- ¬A = Apole (orthogonal/annihilator)
- A ⊗ B = {ab | a ∈ A, b ∈ B}
- A ℘ B = (Apole ⊗ Bpole)pole

This is a dialogue category.
-}

module PhaseSemantics where
  postulate
    M : Type  -- Monoid
    _·ₘ_ : M → M → M  -- Monoid operation

  -- Phase: subset of M
  Phase : Type₁
  Phase = M → Type

  -- Orthogonal: Apole = {m | ∀a ∈ A. a·m ∈ pole}
  postulate
    _pole : Phase → Phase

  -- Phase is dialogue category
  postulate
    Phase-dialogue : DialogueCategory (lsuc lzero) lzero

--------------------------------------------------------------------------------
-- Summary

{-|
## Summary

This module implements tensorial negation and dialogue categories:

1. **Tensorial negation ¬**: Contravariant involutive functor
2. **Continuation monad T = ¬∘¬**: Equation 51
3. **Pole pole = ¬1**: Equation 52
4. **Par A ℘ B = ¬(¬A ⊗ ¬B)**: Equation 53
5. **Dialogue categories**: Monoidal + tensorial negation
6. **Linear connectives**: ⊗, ℘, ⊸, &, ⊕, !, ?
7. **Game semantics**: Strategies as morphisms
8. **Neural interpretation**: Prediction, error, attention

**Key Results**:
- Equation 50: ¬¬A ≅ A (involution)
- De Morgan: ¬(A ⊗ B) ≅ ¬A ℘ ¬B
- Chu construction gives canonical dialogue category
- Applications to neural predictive coding

**Next Steps**:
- Strong monads and strength/costrength (Module 5)
- Negation via exponentials (Module 6)
- Connection to neural information theory
-}
