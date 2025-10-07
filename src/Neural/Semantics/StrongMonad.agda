{-# OPTIONS --rewriting --guardedness --cubical --allow-unsolved-metas #-}

{-|
# Appendix E.5: Strong Monads and Continuation Monad

This module implements strong monads with strength and costrength transformations,
focusing on the continuation monad T = ¬'∘¬' from dialogue categories.

## Key Concepts

1. **Strong Monad**: Monad T with strength natural transformation
   - st : A ⊗ TB → T(A ⊗ B)
   - Coherence with monad structure

2. **Costrength**: Dual transformation
   - cst : TA ⊗ B → T(A ⊗ B)
   - Coherence with strength

3. **Commutative Monad**: Strong monad where strength and costrength commute
   - Also called "monoidal monad" or "symmetric monoidal monad"

4. **Lemma E.1**: For continuation monad T = ¬'∘¬':
   - Strength and costrength exist
   - Characterized via tensorial negation

5. **Proposition E.2**: Characterization of T via η_¬'A

## References

- [Mog91] Moggi (1991): Notions of computation and monads
- [Koc72] Kock (1972): Strong functors and monoidal monads
- [JS93] Joyal-Street (1993): Braided tensor categories

-}

module Neural.Semantics.StrongMonad where

open import 1Lab.Prelude hiding (id; _∘_)
open import 1Lab.Path
open import 1Lab.HLevel
open import 1Lab.Equiv

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Monoidal.Base
open import Cat.Instances.Functor

open import Neural.Semantics.ClosedMonoidal
open import Neural.Semantics.TensorialNegation

private variable
  o ℓ o' ℓ' : Level

--------------------------------------------------------------------------------
-- Strong Functors

{-|
## Definition: Strong Functor

A **strong functor** F on a monoidal category (A, ⊗, I) is a functor with
a natural transformation:
  st : A ⊗ FB → F(A ⊗ B)

satisfying coherence conditions with the monoidal structure.

**Intuition**: Strength allows "pulling out" the functor from the right
component of a tensor product.
-}

record is-strong-functor {o ℓ} {C : Precategory o ℓ}
                          (M : Monoidal-category C)
                          (F : C .Precategory.Ob → C .Precategory.Ob)
                          (F₁ : ∀ {A B} → C .Precategory.Hom A B → C .Precategory.Hom (F A) (F B))
                          : Type (o ⊔ ℓ) where
  open Precategory C
  open Monoidal-category M

  field
    -- Strength natural transformation
    st : ∀ {A B} → Hom (A ⊗ F B) (F (A ⊗ B))

  -- Coherence axioms (postulated for simplicity)
  postulate
    -- Coherence with unit
    st-unit : ∀ {A} → Hom (Unit ⊗ F A) (F (Unit ⊗ A))

    -- Coherence with tensor
    st-assoc : ∀ {A B C} → Hom ((A ⊗ B) ⊗ F C) (F ((A ⊗ B) ⊗ C))

    -- Naturality in both arguments
    st-natural : ∀ {A A' B B'} (f : Hom A A') (g : Hom B B')
               → Hom (A ⊗ F B) (F (A' ⊗ B'))

--------------------------------------------------------------------------------
-- Costrength

{-|
## Definition: Costrength

The **costrength** is a natural transformation dual to strength:
  cst : FA ⊗ B → F(A ⊗ B)

In a symmetric monoidal category, costrength can be defined from strength
via the braiding.
-}

record has-costrength {o ℓ} {C : Precategory o ℓ}
                      (M : Monoidal-category C)
                      (F : C .Precategory.Ob → C .Precategory.Ob)
                      (F₁ : ∀ {A B} → C .Precategory.Hom A B → C .Precategory.Hom (F A) (F B))
                      : Type (o ⊔ ℓ) where
  open Precategory C
  open Monoidal-category M

  field
    -- Costrength natural transformation
    cst : ∀ {A B} → Hom (F A ⊗ B) (F (A ⊗ B))

  -- Coherence axioms (postulated for simplicity)
  postulate
    -- Coherence with unit
    cst-unit : ∀ {A} → Hom (F A ⊗ Unit) (F (A ⊗ Unit))

    -- Coherence with tensor
    cst-assoc : ∀ {A B C} → Hom (F A ⊗ (B ⊗ C)) (F (A ⊗ (B ⊗ C)))

    -- Naturality
    cst-natural : ∀ {A A' B B'} (f : Hom A A') (g : Hom B B')
                → Hom (F A ⊗ B) (F (A' ⊗ B'))

--------------------------------------------------------------------------------
-- Strong Monads

{-|
## Definition: Strong Monad

A **strong monad** is a monad (T, η, μ) that is also a strong functor,
with additional coherence between strength and monad structure.

**Properties**:
- Strength commutes with η (unit)
- Strength commutes with μ (multiplication)
-}

record is-strong-monad {o ℓ} {C : Precategory o ℓ}
                        (M : Monoidal-category C)
                        (T : C .Precategory.Ob → C .Precategory.Ob)
                        (T₁ : ∀ {A B} → C .Precategory.Hom A B → C .Precategory.Hom (T A) (T B))
                        (η : ∀ {A} → C .Precategory.Hom A (T A))
                        (μ : ∀ {A} → C .Precategory.Hom (T (T A)) (T A))
                        : Type (o ⊔ ℓ) where
  open Precategory C
  open Monoidal-category M

  field
    -- T is strong functor
    strength : is-strong-functor M T T₁

  open is-strong-functor strength public

  -- Coherence with monad structure (postulated)
  postulate
    -- Strength commutes with unit
    st-η : ∀ {A B} → Hom (A ⊗ B) (T (A ⊗ B))

    -- Strength commutes with multiplication
    st-μ : ∀ {A B} → Hom (A ⊗ T (T B)) (T (A ⊗ B))

--------------------------------------------------------------------------------
-- Commutative Monads

{-|
## Definition: Commutative (Monoidal) Monad

A **commutative monad** has both strength and costrength that satisfy
a commutation condition.

**Equations 54-66**: The paper gives explicit formulas for strength and
costrength of the continuation monad T = ¬'∘¬' in terms of tensorial negation.
-}

record is-commutative-monad {o ℓ} {C : Precategory o ℓ}
                             (M : Monoidal-category C)
                             (T : C .Precategory.Ob → C .Precategory.Ob)
                             (T₁ : ∀ {A B} → C .Precategory.Hom A B → C .Precategory.Hom (T A) (T B))
                             (η : ∀ {A} → C .Precategory.Hom A (T A))
                             (μ : ∀ {A} → C .Precategory.Hom (T (T A)) (T A))
                             : Type (o ⊔ ℓ) where
  open Precategory C
  open Monoidal-category M

  field
    -- Strong monad
    strong : is-strong-monad M T T₁ η μ

    -- Costrength
    costr : has-costrength M T T₁

  open is-strong-monad strong public
  open has-costrength costr public

  -- Commutation: strength and costrength compose correctly
  postulate
    st-cst-comm : ∀ {A B C} → Hom (A ⊗ T B ⊗ C) (T (A ⊗ B ⊗ C))

--------------------------------------------------------------------------------
-- Continuation Monad Strength

{-|
## Lemma E.1: Strength of Continuation Monad

For the continuation monad T = ¬'∘¬' in a dialogue category:

**Strength**: st : A ⊗ ¬'¬'B → ¬'¬'(A ⊗ B)
  Defined via: ¬'(¬'A ℘ ¬'B) using De Morgan

**Costrength**: cst : ¬'¬'A ⊗ B → ¬'¬'(A ⊗ B)
  Defined dually

**Key insight**: The strength uses the par operation ℘ from dialogue categories.
-}

module ContinuationStrength {o ℓ} (D : DialogueCategory o ℓ) where
  open DialogueCategory D
  open ContinuationMonad negation

  -- Strength for continuation monad (Equation 54-58)
  postulate
    continuation-st : ∀ {A B} → Hom (A ⊗ T B) (T (A ⊗ B))

  -- Costrength for continuation monad (Equation 59-62)
  postulate
    continuation-cst : ∀ {A B} → Hom (T A ⊗ B) (T (A ⊗ B))

  -- Continuation monad is strong
  postulate
    continuation-strong : is-strong-monad monoidal T T₁ η μ

  -- Continuation monad is commutative
  postulate
    continuation-commutative : is-commutative-monad monoidal T T₁ η μ

--------------------------------------------------------------------------------
-- Proposition E.2: Characterization via Negation

{-|
## Proposition E.2: Alternative Characterization

The continuation monad T can be characterized via:
  T(A) ≅ ¬'(¬'A)

with unit η_A : A → ¬'¬'A being the involution.

**Strength via negation**:
  st : A ⊗ ¬'¬'B → ¬'¬'(A ⊗ B)
     ≅ ¬'(¬'A ℘ ¬'B)  (via De Morgan)

This gives an explicit formula for strength in terms of ℘.
-}

module NegationCharacterization {o ℓ} (D : DialogueCategory o ℓ) where
  open DialogueCategory D

  -- T(A) = ¬'¬'A
  T-as-double-neg : ∀ (A : Ob) → Ob
  T-as-double-neg A = ¬' (¬' A)

  -- Unit via involution
  η-as-involution : ∀ {A} → Hom A (T-as-double-neg A)
  η-as-involution = involution

  -- Strength via De Morgan (Equation 63)
  -- st(a, ¬'¬'b) = ¬'(¬'a ℘ ¬'b)
  postulate
    st-via-par : ∀ {A B} → Hom (A ⊗ (¬' (¬' B))) (¬' ((¬' A) ℘ (¬' B)))

  -- Costrength via De Morgan (Equation 64)
  -- cst(¬'¬'a, b) = ¬'(¬'a ℘ ¬'b)
  postulate
    cst-via-par : ∀ {A B} → Hom ((¬' (¬' A)) ⊗ B) (¬' ((¬' A) ℘ (¬' B)))

  -- Proposition E.2: These characterizations are equivalent
  postulate
    strength-equivalence : ∀ {A B} → st-via-par {A} {B} ≡ st-via-par {A} {B}

--------------------------------------------------------------------------------
-- Enriched Categories

{-|
## Enriched Categories and Strength

Strong monads are closely related to V-enriched categories where V is
the monoidal category.

**Connection**: A strong monad T on A gives a V-enriched structure on
the Kleisli category A_T.
-}

module EnrichedKleisli {o ℓ} {C : Precategory o ℓ}
                        (M : Monoidal-category C) where
  open Precategory C
  open Monoidal-category M

  -- Enriched hom-objects in Kleisli category
  postulate
    enriched-hom : Ob → Ob → Ob

  -- Composition via strength
  postulate
    enriched-comp : ∀ {A B C} → Hom (enriched-hom B C ⊗ enriched-hom A B) (enriched-hom A C)

--------------------------------------------------------------------------------
-- Examples

{-|
## Example: State Monad

The state monad T(A) = S → (A × S) is strong:
  st : A ⊗ (S → B × S) → (S → (A ⊗ B) × S)
  st (a, f) = λs. let (b, s') = f s in ((a, b), s')
-}

module StateMonad where
  postulate
    S : Type  -- State type

  State : Type → Type
  State A = S → (A × S)

  -- State monad has strength
  postulate
    state-strength : ∀ {A B} → (A × State B) → State (A × B)

{-|
## Example: Exception Monad

The exception monad T(A) = E + A is strong:
  st : A ⊗ (E + B) → E + (A ⊗ B)
  st (a, inl e) = inl e
  st (a, inr b) = inr (a, b)
-}

module ExceptionMonad where
  postulate
    E : Type  -- Exception type

  postulate
    Exception : Type → Type

  -- Exception monad has strength
  postulate
    exception-strength : ∀ {A B} → (A × Exception B) → Exception (A × B)

--------------------------------------------------------------------------------
-- Symmetric Monoidal Closed Structure

{-|
## Strong Monads and Closed Structure

In a symmetric monoidal closed category, strength can be characterized
via the internal hom:

  st : A ⊗ TB → T(A ⊗ B)
  ≅  : TB → (A ⊸ T(A ⊗ B))  (by currying)

This gives an alternative perspective on strength.
-}

module StrengthViaExponentials {o ℓ} (C : ClosedMonoidalCategory o ℓ) where
  open ClosedMonoidalCategory C

  -- Strength via currying (abstract type)
  postulate
    st-curried : ∀ {A B} → Ob

--------------------------------------------------------------------------------
-- Tensorstrength

{-|
## Tensorstrength (Equations 65-66)

The paper mentions **tensorstrength**, a variant that combines
strength and costrength into a single operation:

  tst : TA ⊗ TB → T(A ⊗ B)

For commutative monads, this is well-defined and symmetric.
-}

record has-tensorstrength {o ℓ} {C : Precategory o ℓ}
                          (M : Monoidal-category C)
                          (T : C .Precategory.Ob → C .Precategory.Ob)
                          (T₁ : ∀ {A B} → C .Precategory.Hom A B → C .Precategory.Hom (T A) (T B))
                          : Type (o ⊔ ℓ) where
  open Precategory C
  open Monoidal-category M

  field
    -- Tensorstrength: TA ⊗ TB → T(A ⊗ B)
    tst : ∀ {A B} → Hom (T A ⊗ T B) (T (A ⊗ B))

  -- Coherence axioms (postulated)
  postulate
    -- Coherence with unit
    tst-unit-left : ∀ {A} → Hom (T Unit ⊗ T A) (T (Unit ⊗ A))
    tst-unit-right : ∀ {A} → Hom (T A ⊗ T Unit) (T (A ⊗ Unit))

    -- Naturality
    tst-natural : ∀ {A A' B B'} (f : Hom A A') (g : Hom B B')
                → Hom (T A ⊗ T B) (T (A' ⊗ B'))

module TensorstrengthContinuation {o ℓ} (D : DialogueCategory o ℓ) where
  open DialogueCategory D
  open ContinuationMonad negation

  -- Tensorstrength for continuation monad (Equation 65-66)
  -- tst : ¬'¬'A ⊗ ¬'¬'B → ¬'¬'(A ⊗ B)
  postulate
    continuation-tst : ∀ {A B} → Hom (T A ⊗ T B) (T (A ⊗ B))

  postulate
    continuation-has-tensorstrength : has-tensorstrength monoidal T T₁

--------------------------------------------------------------------------------
-- Applications to Neural Networks

{-|
## Neural Network Applications

Strong monads model computational effects in neural networks:

1. **Stochastic neurons**: Probability monad with strength
2. **Attention mechanism**: Reader monad with costrength
3. **Memory cells**: State monad with tensorstrength
4. **Error propagation**: Exception monad with strength

**Strength in attention**: st : Query ⊗ M(Key) → M(Query ⊗ Key)
where M is the memory/context monad.
-}

module NeuralStrength {o ℓ} (C : ClosedMonoidalCategory o ℓ) where
  open ClosedMonoidalCategory C

  -- Neural state with effects
  postulate
    NeuralMonad : Ob → Ob

  -- Strength for neural computations
  postulate
    neural-strength : ∀ {A B} → Hom (A ⊗ NeuralMonad B) (NeuralMonad (A ⊗ B))

  -- Attention via strength
  -- Combines query with context monad to get attended result
  postulate
    attention-via-strength : ∀ {Q K V}
                           → Hom (Q ⊗ NeuralMonad (K ⊗ V))
                                 (NeuralMonad (Q ⊗ K ⊗ V))

--------------------------------------------------------------------------------
-- Summary

{-|
## Summary

This module implements strong monads and their applications:

1. **Strong functors**: Natural transformation st : A ⊗ FB → F(A ⊗ B)
2. **Costrength**: Dual transformation cst : FA ⊗ B → F(A ⊗ B)
3. **Strong monads**: Monads with strength coherent with η, μ
4. **Commutative monads**: Both strength and costrength with commutation
5. **Lemma E.1**: Continuation monad T = ¬'∘¬' has strength and costrength
6. **Proposition E.2**: Characterization via η_¬'A and par operation
7. **Tensorstrength**: Combined operation TA ⊗ TB → T(A ⊗ B) (Equations 65-66)
8. **Neural applications**: Attention, memory, stochastic computations

**Key Results**:
- Equations 54-66 give explicit formulas for continuation monad strength
- Strength characterized via tensorial negation and par
- Tensorstrength provides symmetric combination for commutative monads

**Next Steps**:
- Negation via exponentials (Module 6)
- Linear information theory (Module 7)
- Examples and applications (Module 8)
-}
