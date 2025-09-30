{-# OPTIONS --no-import-sorts #-}
{-|
# Summing Functors on Networks

This module implements Section 2.1 from Manin & Marcolli (2024):
"Homotopy-theoretic and categorical models of neural information networks"

We construct "moduli spaces" (categories) parameterizing all possible assignments
of resources to a network and its subsystems. These categories of summing functors
provide our configuration space attached to a network.

## Overview

A summing functor Φ_X : P(X) → C assigns resources (objects in C) to subsystems
(subsets of X) in a way that respects disjoint unions:

  Φ_X(A ∪ A') = Φ_X(A) ⊕ Φ_X(A')    when A ∩ A' = {*}

The category ΣC(X) of summing functors can be viewed as an "action groupoid"
or "moduli stack" of all consistent resource assignments up to equivalence.
-}

module Neural.SummingFunctor where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Diagram.Zero
open import Cat.Diagram.Coproduct
open import Cat.Monoidal.Base
open import Cat.Monoidal.Braided

import Cat.Reasoning

open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin; fzero; fsuc; Discrete-Fin)
open import Data.Bool.Base using (Bool; true; false; if_then_else_; true≠false)
open import Data.Dec.Base using (Dec; yes; no; _≡?_)
open import Data.Sum.Base

private variable
  o ℓ : Level

{-|
## 2.1.1 Categories with Sums and Zero Objects

Following Definition 2.1 of the paper, we first establish what it means for a
category to have the structure needed for summing functors.

A category C is suitable for defining summing functors if it has:
1. A zero object (both initial and terminal)
2. Binary coproducts (categorical sums)
-}

module _ (C : Precategory o ℓ) where
  open Cat.Reasoning C

  {-|
  A category has sums and zero if it has:
  - A zero object ∅ that is both initial and terminal
  - A coproduct A ⊕ B for any pair of objects A, B
  -}
  record HasSumsAndZero : Type (o ⊔ ℓ) where
    field
      has-zero : Zero C
      has-binary-coproducts : ∀ (A B : C .Precategory.Ob) → Coproduct C A B

    open Zero has-zero public

    -- Convenient notation for coproducts
    _⊕_ : C .Precategory.Ob → C .Precategory.Ob → C .Precategory.Ob
    A ⊕ B = Coproduct.coapex (has-binary-coproducts A B)

    infixr 30 _⊕_

{-|
## 2.1.2 The Category P(X) of Pointed Subsets

For a finite pointed set X (with basepoint *), we construct the category P(X)
whose objects are pointed subsets A ⊆ X (meaning * ∈ A) and morphisms are
inclusions.

This category serves as the domain for summing functors. We think of X\{*} as
representing a system of neurons, and subsets A ⊆ X as representing subsystems.
-}

module _ (n : Nat) where
  open import Data.Fin.Base using (Discrete-Fin)

  {-|
  A pointed subset of Fin (suc n) is represented by a predicate that must
  include the basepoint fzero.

  We encode pointed subsets as functions Fin (suc n) → Bool where:
  - fzero must map to true (the basepoint is always included)
  - Other elements can be true (included) or false (excluded)
  -}
  PointedSubset : Type
  PointedSubset = Σ (Fin (suc n) → Bool) (λ P → P fzero ≡ true)

  {-|
  ### Operations on Pointed Subsets

  These operations would be needed for the full Definition 2.1 of summing functors.
  While we don't use them in Definition 2.5 (which works directly with Fin n → C.Ob),
  they illustrate how the summing property `Φ(A ∪ B) = Φ(A) ⊕ Φ(B)` works concretely.

  TODO: These could be implemented properly, but they require careful handling of
  with-abstractions in cubical Agda to avoid unsolved metas. Since they're not needed
  for the main development (Definition 2.5), we leave them as postulates.
  -}

  postulate
    -- The basepoint-only subset {*}
    basepoint-only : PointedSubset

    -- Singleton subset {i, *} for i ∈ Fin n
    singleton : Fin n → PointedSubset

    -- Union of pointed subsets (pointwise or)
    _∪ₚ_ : PointedSubset → PointedSubset → PointedSubset

    -- Check if two pointed subsets are disjoint (intersect only at basepoint)
    are-disjoint : PointedSubset → PointedSubset → Type

  {-|
  The category P(X) has:
  - Objects: pointed subsets of Fin (suc n)
  - Morphisms: proofs that one subset is included in another
  - Composition: transitivity of inclusion
  - Identity: reflexivity of inclusion
  -}
  P[_] : Precategory lzero lzero
  P[_] .Precategory.Ob = PointedSubset
  P[_] .Precategory.Hom (P₁ , base₁) (P₂ , base₂) =
    (∀ i → P₁ i ≡ true → P₂ i ≡ true)
  P[_] .Precategory.Hom-set _ _ = hlevel 2
  P[_] .Precategory.id {A , _} i p = p
  P[_] .Precategory._∘_ f g i p = f i (g i p)
  P[_] .Precategory.idr _ = refl
  P[_] .Precategory.idl _ = refl
  P[_] .Precategory.assoc _ _ _ = refl

{-|
## 2.1.3 Definition of Summing Functors

Following Definition 2.1 of the paper, a summing functor Φ_X : P(X) → C
satisfies:

1. Φ_X({*}) = 0 (the basepoint maps to the zero object)
2. For disjoint subsets A, A' (meaning A ∩ A' = {*}):
   Φ_X(A ∪ A') = Φ_X(A) ⊕ Φ_X(A')

The key insight is that this additivity property expresses that resource
assignments are additive on independent subsystems.
-}

{-|
### Definition 2.1 Sketch: How to Construct a Functor from Data

This is a **conceptual sketch** - we don't implement it fully.

Given data `φ : Fin n → C.Ob`, we would construct a functor `F : P(X) → C`:

**F₀ (Object part)**:
For a pointed subset A ⊆ X, we need `F₀ A : C.Ob`.
- If A = {*}, return ∅ (zero object)
- Otherwise, take indexed coproduct: `⊕_{i ∈ A \ {*}} φ(i)`

**F₁ (Morphism part)**:
For an inclusion `j : A ↪ B` in P(X), we need `F₁ j : F₀ A → F₀ B`.
- Use the universal property of coproducts
- The inclusion induces canonical morphisms `φ(i) → ⊕_{j ∈ B} φ(j)` for i ∈ A
- These compose to give `(⊕_{i ∈ A} φ(i)) → (⊕_{j ∈ B} φ(j))`

**Functoriality proofs**:
- F-id: identity inclusion gives identity by coproduct uniqueness
- F-∘: composition of inclusions corresponds to coproduct transitivity

**Summing property**:
For disjoint A, B with A ∩ B = {*}:
```
F₀(A ∪ B) = ⊕_{i ∈ (A ∪ B) \ {*}} φ(i)
          = ⊕_{i ∈ A \ {*}} φ(i)  ⊕  ⊕_{j ∈ B \ {*}} φ(j)
          = F₀ A ⊕ F₀ B
```
This follows from the associativity/commutativity of indexed coproducts.

### Why We Don't Implement This Fully

1. **Lemma 2.3 shows equivalence**: The category of summing functors
   is equivalent to just `Fin n → C.Ob`, so we can work with the data directly.

2. **For symmetric monoidal categories**: The coherence theorem ensures
   the summing property holds automatically, so we don't need explicit proofs.

3. **Indexed coproducts are tedious**: While 1Lab provides
   `is-indexed-coproduct`, iterating coproducts over finite predicates
   requires significant boilerplate.

4. **The paper itself skips this**: Definition 2.5 goes straight to
   `ΣC(X) := Ĉⁿ` without constructing functors explicitly.

We'll continue with Lemma 2.3's simplified characterization below.
-}

{-|
## 2.1.4 Lemma 2.3: Simplified Description

**Lemma 2.3** (Paper): A summing functor Φ_X : P(X) → C is completely
determined by its values on singleton sets A_x = {x, *} for x ∈ X \ {*}.

Moreover, the category ΣC(X) of summing functors (with invertible natural
transformations as morphisms) is equivalent to Ĉⁿ, where:
- Ĉ is the groupoid core of C (objects of C, only invertible morphisms)
- n = #X - 1 (number of elements excluding basepoint)

This simplification is crucial: instead of specifying Φ on all 2ⁿ pointed
subsets, we only need to specify n objects Φ(x) for the non-basepoint elements.
-}

module Lemma2∙3 {C : Precategory o ℓ} (structure : HasSumsAndZero C) (n : Nat) where
  open HasSumsAndZero structure
  open Cat.Reasoning C

  {-|
  ### Part 1: Summing functors determined by singletons

  A summing functor need only specify where each non-basepoint element maps.
  We represent this as a function from Fin n to objects of C.
  -}

  SummingFunctorData : Type o
  SummingFunctorData = Fin n → C .Precategory.Ob

  {-|
  ### The Groupoid Core Ĉ

  The groupoid core of C has the same objects as C but only invertible
  morphisms. This ensures the category ΣC(X) has interesting topology.
  -}

  Core : Precategory o ℓ
  Core .Precategory.Ob = C .Precategory.Ob
  Core .Precategory.Hom A B = Σ (C .Precategory.Hom A B) is-invertible
  Core .Precategory.Hom-set A B =
    Σ-is-hlevel 2 (C .Precategory.Hom-set A B) λ _ → is-prop→is-set is-invertible-is-prop
  Core .Precategory.id = C .Precategory.id , id-invertible
  Core .Precategory._∘_ (f , f-inv) (g , g-inv) =
    (C .Precategory._∘_ f g) , invertible-∘ f-inv g-inv
  Core .Precategory.idr (f , f-inv) =
    Σ-pathp (C .Precategory.idr f)
      (is-prop→pathp (λ i → is-invertible-is-prop) _ _)
  Core .Precategory.idl (f , f-inv) =
    Σ-pathp (C .Precategory.idl f)
      (is-prop→pathp (λ i → is-invertible-is-prop) _ _)
  Core .Precategory.assoc (f , _) (g , _) (h , _) =
    Σ-pathp (C .Precategory.assoc f g h)
      (is-prop→pathp (λ i → is-invertible-is-prop) _ _)

  {-|
  ### Part 2: Category ΣC(X) of Summing Functors

  Following Definition 2.2, we define the category of summing functors.
  By Lemma 2.3, this is equivalent to Coreⁿ (n-fold product of Core).

  Objects: n-tuples of objects from C
  Morphisms: n-tuples of invertible morphisms from C
  -}

  ΣC[_] : Precategory o ℓ
  ΣC[_] .Precategory.Ob = Fin n → Core .Precategory.Ob
  ΣC[_] .Precategory.Hom Φ Ψ = ∀ (i : Fin n) → Core .Precategory.Hom (Φ i) (Ψ i)
  ΣC[_] .Precategory.Hom-set Φ Ψ =
    Π-is-hlevel 2 λ i → Core .Precategory.Hom-set (Φ i) (Ψ i)
  ΣC[_] .Precategory.id i = Core .Precategory.id
  ΣC[_] .Precategory._∘_ η θ i = Core .Precategory._∘_ (η i) (θ i)
  ΣC[_] .Precategory.idr η = funext λ i → Core .Precategory.idr (η i)
  ΣC[_] .Precategory.idl η = funext λ i → Core .Precategory.idl (η i)
  ΣC[_] .Precategory.assoc η θ ζ = funext λ i → Core .Precategory.assoc (η i) (θ i) (ζ i)

{-|
## 2.1.5 Corollary 2.4: Commutative Monoidal Categories

For **commutative monoidal categories** (strictly associative and commutative),
the characterization of summing functors simplifies even further.

A commutative monoidal category has all associators, braiding, and unitors as
identities, so ordering and bracketing don't matter.
-}

module Corollary2∙4
  {C : Precategory o ℓ}
  (Cᵐ : Monoidal-category C)
  -- (comm : is-commutative Cᵐ)  -- TODO: Need commutative property
  (n : Nat) where

  open Monoidal-category Cᵐ
  open Cat.Reasoning C

  {-|
  ### The No-Cost Subcategory

  In the monoidal setting, summing functors take values in the "no-cost
  resources" subcategory: objects C with a morphism Unit → C.

  This is necessary because inclusions {*} → A in P(X) induce morphisms
  Unit → Φ(A), and Unit is not required to be initial.
  -}

  has-no-cost : C .Precategory.Ob → Type ℓ
  has-no-cost X = C .Precategory.Hom Unit X

  {-|
  ### Summing Functor Data in Commutative Monoidal Setting

  By Corollary 2.4, a summing functor in a commutative monoidal category
  is determined by:
  1. A collection of objects {Φ(x)} in the no-cost subcategory
  2. The morphisms φ_x : Unit → Φ(x) witnessing no-cost
  -}

  record CommutativeSummingFunctorData : Type (o ⊔ ℓ) where
    field
      objects : Fin n → C .Precategory.Ob
      no-cost-morphisms : ∀ (i : Fin n) → has-no-cost (objects i)

{-|
## 2.1.6 Definition 2.5: Symmetric Monoidal Summing Functors

This is the **general definition** that works for arbitrary symmetric monoidal
categories, not just those with zero objects and coproducts.

Following Definition 2.5 from the paper, we directly define ΣC(X) as the
n-fold product of the groupoid core Ĉ, without needing to verify the summing
property explicitly.

This is the version we'll use going forward, as it:
- Works for general symmetric monoidal categories
- Enables non-linear Hopfield dynamics (§6)
- Includes categories like deep neural networks
-}

module Definition2∙5
  {C : Precategory o ℓ}
  (Cᵐ : Monoidal-category C)
  (Cˢ : Symmetric-monoidal Cᵐ)
  (n : Nat) where

  open Monoidal-category Cᵐ
  open Symmetric-monoidal Cˢ
  open Cat.Reasoning C

  {-|
  ### The Groupoid Core Ĉ

  Same as in Lemma 2.3, but now C is a symmetric monoidal category
  rather than requiring zero object and coproducts.
  -}

  Core : Precategory o ℓ
  Core .Precategory.Ob = C .Precategory.Ob
  Core .Precategory.Hom A B = Σ (C .Precategory.Hom A B) is-invertible
  Core .Precategory.Hom-set A B =
    Σ-is-hlevel 2 (C .Precategory.Hom-set A B) λ _ → is-prop→is-set is-invertible-is-prop
  Core .Precategory.id = C .Precategory.id , id-invertible
  Core .Precategory._∘_ (f , f-inv) (g , g-inv) =
    (C .Precategory._∘_ f g) , invertible-∘ f-inv g-inv
  Core .Precategory.idr (f , f-inv) =
    Σ-pathp (C .Precategory.idr f)
      (is-prop→pathp (λ i → is-invertible-is-prop) _ _)
  Core .Precategory.idl (f , f-inv) =
    Σ-pathp (C .Precategory.idl f)
      (is-prop→pathp (λ i → is-invertible-is-prop) _ _)
  Core .Precategory.assoc (f , _) (g , _) (h , _) =
    Σ-pathp (C .Precategory.assoc f g h)
      (is-prop→pathp (λ i → is-invertible-is-prop) _ _)

  {-|
  ### Category ΣC(X) of Summing Functors (Definition 2.5)

  For a finite pointed set X with #X = n+1 (n non-basepoint elements),
  the category ΣC(X) is defined as Ĉⁿ:

  - Objects: n-tuples {Φ_X(x)}_{x ∈ X\{*}} of objects from C
  - Morphisms: n-tuples of invertible morphisms

  This is the **primary definition** for symmetric monoidal categories.
  The restriction to invertible morphisms ensures:
  1. Non-trivial topology of the nerve
  2. Interpretation as equivalence classes of resource assignments
  3. Connection to spectra via Gamma-spaces (§7)
  -}

  ΣC[_] : Precategory o ℓ
  ΣC[_] .Precategory.Ob = Fin n → Core .Precategory.Ob
  ΣC[_] .Precategory.Hom Φ Ψ = ∀ (i : Fin n) → Core .Precategory.Hom (Φ i) (Ψ i)
  ΣC[_] .Precategory.Hom-set Φ Ψ =
    Π-is-hlevel 2 λ i → Core .Precategory.Hom-set (Φ i) (Ψ i)
  ΣC[_] .Precategory.id i = Core .Precategory.id
  ΣC[_] .Precategory._∘_ η θ i = Core .Precategory._∘_ (η i) (θ i)
  ΣC[_] .Precategory.idr η = funext λ i → Core .Precategory.idr (η i)
  ΣC[_] .Precategory.idl η = funext λ i → Core .Precategory.idl (η i)
  ΣC[_] .Precategory.assoc η θ ζ = funext λ i → Core .Precategory.assoc (η i) (θ i) (ζ i)

  {-|
  ### Why This Definition Works: The Coherence Theorem

  **Key Question**: Why can we define ΣC(X) as just `Fin n → Core.Ob` without
  explicitly constructing functors P(X) → C?

  **Answer**: The **coherence theorem** for symmetric monoidal categories!

  In a symmetric monoidal category, all diagrams built from:
  - Associators α: (A ⊗ B) ⊗ C → A ⊗ (B ⊗ C)
  - Braiding β: A ⊗ B → B ⊗ A
  - Unitors λ, ρ

  commute automatically. This means:
  1. Different bracketings/orderings of `⊕_{i ∈ A} φ(i)` are canonically isomorphic
  2. The summing property Φ(A ∪ B) ≅ Φ(A) ⊗ Φ(B) holds "for free"
  3. We don't need to prove functoriality - it follows from coherence

  **What we're implicitly doing**:
  - Data `Fin n → C.Ob` extends to all pointed subsets via monoidal product
  - The coherence theorem ensures this extension is functorial
  - Inclusions A ↪ B get morphisms via canonical isomorphisms
  - Everything commutes by Mac Lane's coherence theorem (1963)

  **Why the paper uses this**: It's not laziness - for symmetric monoidal
  categories, the explicit functor construction in Definition 2.1 is
  *redundant*. The data `Fin n → C.Ob` already determines a unique coherent
  summing functor.

  **Reference**: Mac Lane, "Natural associativity and commutativity" (1963)
  -}

  {-|
  ### Interpretation as Configuration Space

  Following the paper's discussion:

  **Moduli space interpretation**: ΣC(X) is a categorical "moduli space"
  parameterizing all possible assignments of resources (objects in C) to
  the subsystems (non-basepoint elements of X), up to equivalence
  (invertible natural transformations).

  **Action groupoid analogy**: Rather than taking a quotient by equivalence
  (which would be Ω/G for a space with group action), we keep all equivalences
  as invertible morphisms. This "resolves" the quotient and behaves better
  for non-free actions.

  **Network interpretation**: For X representing a neural network:
  - Objects Φ ∈ ΣC(X): assignments of resources to each neuron
  - Morphisms η : Φ → Ψ: equivalences between resource assignments
  - The nerve |ΣC(X)|: geometric realization of the configuration space

  **Connection to functors**: Each object Φ : Fin n → C.Ob *is* implicitly
  a summing functor Φ_X : P(X) → C via the monoidal structure. The coherence
  theorem ensures this is well-defined without explicit construction.
  -}

{-|
## 2.1.7 Relationship Between Definitions

The paper provides three characterizations of summing functors:

1. **Definition 2.1** (Zero + Coproducts): Full summing functor property
   Φ(A ∪ A') = Φ(A) ⊕ Φ(A') for disjoint A, A'

2. **Lemma 2.3** (Simplified): Determined by values on singletons
   ΣC(X) ≃ Ĉⁿ for categories with zero + coproducts

3. **Definition 2.5** (Symmetric Monoidal): Direct definition as Ĉⁿ
   Works for general symmetric monoidal categories

**Why Definition 2.5 is preferred**:
- More general (doesn't require zero object)
- Enables non-linear dynamics (§6)
- Simpler to work with
- Includes important examples like deep neural networks

The key insight: for symmetric monoidal categories, the coherence theorem
ensures that the summing property holds automatically up to canonical
isomorphism, so we can define ΣC(X) directly as a product.
-}

{-|
## Notes on Implementation Strategy

Following the paper's development:

1. **Definition 2.1**: Basic summing functor with zero + coproducts ✓ (partial)
2. **Lemma 2.3**: Simplification to Ĉⁿ (TODO)
3. **Definition 2.2**: Category ΣC(X) of summing functors (TODO)
4. **Definition 2.5**: Generalization to symmetric monoidal (TODO)
5. **Corollary 2.4**: Commutative monoidal case (TODO)

The key insight: we use 1Lab's existing infrastructure for:
- Zero objects: `Cat.Diagram.Zero`
- Coproducts: `Cat.Diagram.Coproduct`
- Monoidal categories: `Cat.Monoidal.Base`
- Symmetric monoidal: `Cat.Monoidal.Braided`
-}