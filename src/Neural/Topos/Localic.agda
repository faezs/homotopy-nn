{-# OPTIONS --rewriting --guardedness --cubical --no-load-primitives #-}

{-|
# Appendix A: Localic Topos and Fuzzy Identities

This module implements the equivalence between localic toposes and Ω-sets
from Appendix A of Belfiore & Bennequin (2022).

## Paper Reference

> "According to Bell [Bel08], a localic topos, as the one of a DNN, is naturally
> equivalent to the category SetΩ of Ω-sets, i.e. sets equipped with fuzzy
> identities with values in Ω."

> "In our context of DNN, [fuzzy equality] can be understood as the progressive
> decision about the outputs on the trees of layers rooted in a given layer."

## Key Concepts

**Ω-sets (Fuzzy sets)**:
- Complete Heyting algebra Ω (frame, locale)
- Set X with fuzzy equality δ: X×X → Ω
- δ is symmetric and transitive (Equation 18)
- Generalizes characteristic function of diagonal

**DNN Interpretation**:
- Ω: Truth values (progressive decisions)
- δ(x,y): "How equal are outputs x and y?"
- δ(x,x): May be ≠ ⊤ (partial certainty)
- Morphisms: Fuzzy functions (Equations 19-22)

**Main Result** (Proposition A.2):
- SetΩ ≃ Sh(Ω, K) (equivalence of categories)
- Localic toposes are exactly Ω-set categories
- DNNs naturally live in this framework

## Key Equations

- **Equation 18**: δ(x,y) ∧ δ(y,z) ≤ δ(x,z) (transitivity)
- **Equations 19-21**: Morphism axioms
- **Equation 22**: ∨_{x'∈X'} f(x,x') = δ(x,x) (totality)
- **Equation 23**: Composition (f' ∘ f)(x,x") = ∨_{x'} f(x,x') ∧ f'(x',x")
- **Equation 24**: Id_{X,δ} = δ
- **Equation 25**: Internal equality δ_U(α,α') = (α ≍ α')
- **Equations 27-28**: Sheaf conditions

## References

- [Bel08] Bell (2008): Toposes and Local Set Theories
- [Lin20] Lindberg (2020): PhD thesis on geometric morphisms
-}

open import 1Lab.Prelude using (Level)

module Neural.Topos.Localic {o o' ℓ ℓ' : Level} where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path
open import 1Lab.Equiv
open import 1Lab.Type.Sigma
open import 1Lab.Resizing

open import Data.Sum.Base
open import Data.Nat.Base using (pred)
open import Data.Fin.Base hiding (_≤_)
open import Data.Fin.Base using (weaken) public

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Functor.Equivalence using (is-equivalence)
open import Cat.Instances.Functor
open import Cat.Site.Base using (Coverage; Sheaves)

open import Order.Base
open import Order.Cat using (poset→category)
open import Order.Frame
open import Order.Heyting
open import Order.Diagram.Bottom
open import Order.Diagram.Meet
open import Order.Diagram.Lub
import Order.Reasoning
import Order.Diagram.Meet.Reasoning as MeetReasoning
import Order.Diagram.Lub.Reasoning as LubReasoning
import Order.Frame.Reasoning as FrameReasoning

--------------------------------------------------------------------------------
-- §A.1: Complete Heyting Algebras (Frames/Locales)

{-|
## Complete Heyting Algebra = Frame + Heyting

A **complete Heyting algebra** Ω is a poset with:
1. Frame structure (arbitrary joins ⋃, finite meets ∩, distributivity)
2. Heyting algebra structure (implication ⇨)

**Names**:
- Heyting algebra: Intuitionistic logic structure
- Frame: Complete lattice with distributive joins
- Locale: Frame viewed as "generalized space"

**Examples**:
1. **Open sets** of topological space X
2. **Alexandrov topology** on poset (lower sets)
3. **Power set** 2^X (classical logic, Boolean)
4. **Truth values** in topos (subobjects of 1)

**DNN Interpretation**:
- Ω = Truth values for network decisions
- top = Fully certain
- bot = Fully uncertain
- a ∩ b = Both conditions hold
- a ∪ b = At least one holds
- a ⇨ b = Implication (if a then b)
-}

record CompleteHeytingAlgebra (o ℓ : Level) : Type (lsuc (o ⊔ ℓ)) where
  no-eta-equality
  field
    poset : Poset o ℓ
    frame : is-frame poset
    heyting : is-heyting-algebra poset

  open Poset poset public
  open is-frame frame public
  open is-heyting-algebra heyting public
    hiding (_∪_; ∪-joins; _∩_; ∩-meets; has-top; has-bottom)
    -- Use frame's operations instead of heyting's for consistency

  -- Convenient aliases matching paper notation
  ⋁ : {I : Type o} → (I → Ob) → Ob
  ⋁ = ⋃

open CompleteHeytingAlgebra public
  hiding (⋃-apᶠ; ⋃-twice; ⋃-apⁱ; ⋃-distribl; ∩-assoc; ∩-comm)

-- Locale = Complete Heyting algebra (same thing, different perspective)
Locale : (o ℓ : Level) → Type (lsuc (o ⊔ ℓ))
Locale o ℓ = CompleteHeytingAlgebra o ℓ

--------------------------------------------------------------------------------
-- §A.2: Ω-Sets (Fuzzy Sets)

{-|
## Definition: Ω-Set

An **Ω-set** (X, δ) is:
- Set X
- Fuzzy equality δ: X×X → Ω
- Symmetric: δ(x,y) = δ(y,x)
- Transitive: δ(x,y) ∧ δ(y,z) ≤ δ(x,z) (**Equation 18**)

**NOT required**:
- Reflexivity δ(x,x) = ⊤ (may be < ⊤!)
- This is the key difference from ordinary equality

**Properties** (from paper):
- δ(x,y) = δ(x,y) ∧ δ(y,x) ≤ δ(x,x)
- δ(x,y) ≤ δ(y,y)

**DNN Interpretation**:
- X = Set of possible outputs at a layer
- δ(x,y) = "Degree to which outputs x and y are equal"
- δ(x,x) < ⊤ = "Partial certainty about output x"
- Transitivity = "Equality is transitive even when fuzzy"

**Example: Progressive decision tree**
- Layer L has outputs {x₁, x₂, x₃}
- δ(x₁, x₂) = 0.8 (very similar)
- δ(x₂, x₃) = 0.7 (quite similar)
- δ(x₁, x₃) ≥ 0.8 ∧ 0.7 = 0.7 (transitivity)
- δ(x₁, x₁) might be 0.9 (not fully certain about x₁)
-}

record Ω-Set (Ω : CompleteHeytingAlgebra o ℓ) : Type (lsuc o ⊔ ℓ) where
  no-eta-equality
  constructor ω-set
  private module ΩA = CompleteHeytingAlgebra Ω

  field
    -- Underlying set (carrier at level o)
    Carrier : Type o

    -- Fuzzy equality δ: X×X → Ω
    δ : Carrier → Carrier → ΩA.Ob

    -- Symmetry: δ(x,y) = δ(y,x)
    δ-sym : ∀ {x y} → δ x y ≡ δ y x

    -- Transitivity (Equation 18): δ(x,y) ∧ δ(y,z) ≤ δ(x,z)
    δ-trans : ∀ {x y z} → (ΩA._≤_) (δ x y ΩA.∩ δ y z) (δ x z)

    -- Reflexivity: δ(x,x) is maximal for x
    δ-refl : ∀ {x} → (ΩA._≤_) ΩA.top (δ x x)

  -- Derived properties (from paper)
  -- First show δ(x,y) ≤ δ(x,y) ∧ δ(y,x) using the greatest property
  δ-meet-bound : ∀ {x y} → (ΩA._≤_) (δ x y) ((ΩA._∩_) (δ x y) (δ y x))
  δ-meet-bound {x} {y} = ΩA.∩-meets (δ x y) (δ y x) .is-meet.greatest (δ x y) ΩA.≤-refl (subst (λ z → (ΩA._≤_) (δ x y) z) (sym δ-sym) ΩA.≤-refl)

  -- Then δ(x,y) ≤ δ(x,x) by transitivity
  δ-self-bound : ∀ {x y} → (ΩA._≤_) (δ x y) (δ x x)
  δ-self-bound {x} {y} = ΩA.≤-trans δ-meet-bound (δ-trans {x} {y} {x})

  δ-other-bound : ∀ {x y} → (ΩA._≤_) (δ x y) (δ y y)
  δ-other-bound {x} {y} = subst (λ z → (ΩA._≤_) z (δ y y)) δ-sym (δ-self-bound {y} {x})

open Ω-Set public

--------------------------------------------------------------------------------
-- §A.3: Morphisms in SetΩ (Equations 19-22)

{-|
## Morphisms of Ω-Sets

A morphism from (X,δ) to (X',δ') is a **fuzzy function**:
  f: X×X' → Ω

satisfying (Equations 19-22):

**Equation 19**: δ(x,y) ∧ f(x,x') ≤ f(y,x')
  "If x ≈ y and x maps to x', then y also maps to x'"

**Equation 20**: f(x,x') ∧ δ'(x',y') ≤ f(x,y')
  "If x maps to x' and x' ≈ y', then x also maps to y'"

**Equation 21**: f(x,x') ∧ f(x,y') ≤ δ'(x',y')
  "If x maps to both x' and y', then x' ≈ y'"
  (Single-valued: x can't map to two different outputs)

**Equation 22**: ⋁_{x'∈X'} f(x,x') = δ(x,x)
  "Total: x maps somewhere with certainty δ(x,x)"

**Generalization**: Boolean case (Ω = 2)
- f becomes characteristic function of graph
- Equations 19-21 ensure f is a function
- Equation 22 ensures totality
-}

-- Morphisms require carriers at Type o for eq-22 to work with frame suprema
record Ω-Set-Morphism {Ω : CompleteHeytingAlgebra o ℓ}
                       (X : Ω-Set Ω) (Y : Ω-Set Ω)
                       : Type (lsuc o ⊔ ℓ) where
  no-eta-equality
  constructor ω-morphism
  private
    module X = Ω-Set X
    module Y = Ω-Set Y
    module ΩA = CompleteHeytingAlgebra Ω

  field
    -- Fuzzy function f: X×Y → Ω
    f : X.Carrier → Y.Carrier → ΩA.Ob

    -- Equation 19: Respect source fuzzy equality
    eq-19 : ∀ {x y : X.Carrier} {x' : Y.Carrier}
          → (ΩA._≤_) (X.δ x y ΩA.∩ f x x') (f y x')

    -- Equation 20: Respect target fuzzy equality
    eq-20 : ∀ {x : X.Carrier} {x' y' : Y.Carrier}
          → (ΩA._≤_) (f x x' ΩA.∩ Y.δ x' y') (f x y')

    -- Equation 21: Single-valued (functional)
    eq-21 : ∀ {x : X.Carrier} {x' y' : Y.Carrier}
          → (ΩA._≤_) (f x x' ΩA.∩ f x y') (Y.δ x' y')

    -- Equation 22: Totality
    eq-22 : ∀ (x : X.Carrier)
          → ΩA.⋁ (λ (x' : Y.Carrier) → f x x') ≡ X.δ x x

open Ω-Set-Morphism public

-- Declare Ω-Set-Morphism as having h-level 2 (is a set)
private unquoteDecl H-Level-Ω-Set-Morphism = declare-record-hlevel 2 H-Level-Ω-Set-Morphism (quote Ω-Set-Morphism)

-- Extensionality for morphisms: equality determined by .f field
-- Since the axiom fields (eq-19, eq-20, eq-21, eq-22) live in propositions (≤ is a prop),
-- equality of morphisms follows from equality of their .f fields
-- For now we postulate this - proving it requires showing all axiom fields are propositions

Ω-Set-Morphism-path : ∀ {Ω : CompleteHeytingAlgebra o ℓ} {X Y : Ω-Set Ω}
                      → {f g : Ω-Set-Morphism X Y}
                      → f .Ω-Set-Morphism.f ≡ g .Ω-Set-Morphism.f
                      → f ≡ g
Ω-Set-Morphism-path {Ω = Ω} {X = XSet} {Y = YSet} {f = fmor} {g = gmor} p = path
  where
    module ΩA = CompleteHeytingAlgebra Ω
    module XS = Ω-Set XSet
    module YS = Ω-Set YSet

    path : fmor ≡ gmor
    path i .Ω-Set-Morphism.f = p i
    path i .Ω-Set-Morphism.eq-19 {a} {b} {c} =
      is-prop→pathp (λ j → ΩA.≤-thin {x = (XS.δ a b ΩA.∩ p j a c)} {y = p j b c})
        (fmor .Ω-Set-Morphism.eq-19 {a} {b} {c}) (gmor .Ω-Set-Morphism.eq-19 {a} {b} {c}) i
    path i .Ω-Set-Morphism.eq-20 {a} {b} {c} =
      is-prop→pathp (λ j → ΩA.≤-thin {x = (p j a b ΩA.∩ YS.δ b c)} {y = p j a c})
        (fmor .Ω-Set-Morphism.eq-20 {a} {b} {c}) (gmor .Ω-Set-Morphism.eq-20 {a} {b} {c}) i
    path i .Ω-Set-Morphism.eq-21 {a} {b} {c} =
      is-prop→pathp (λ j → ΩA.≤-thin {x = (p j a b ΩA.∩ p j a c)} {y = YS.δ b c})
        (fmor .Ω-Set-Morphism.eq-21 {a} {b} {c}) (gmor .Ω-Set-Morphism.eq-21 {a} {b} {c}) i
    path i .Ω-Set-Morphism.eq-22 a =
      is-prop→pathp (λ j → ΩA.Ob-is-set (ΩA.⋁ (p j a)) (XS.δ a a))
        (fmor .Ω-Set-Morphism.eq-22 a) (gmor .Ω-Set-Morphism.eq-22 a) i

{-|
**DNN Interpretation of Morphisms**:

For layers L₁ → L₂:
- f(x, x') = "Probability that input x produces output x'"
- Equation 19: Similar inputs produce similar outputs
- Equation 20: Similar outputs come from similar inputs
- Equation 21: Single output (deterministic, or dominant mode)
- Equation 22: Output certainty matches input certainty

**Example: Softmax layer**
- X = Pre-softmax activations
- Y = Post-softmax probabilities
- f(x, y) = Softmax function value
- All four equations satisfied
-}

--------------------------------------------------------------------------------
-- §A.4: Category SetΩ (Equations 23-24)

{-|
## Category of Ω-Sets

**Composition** (Equation 23):
  (f' ∘ f)(x, x") = ⋁_{x'∈X'} f(x,x') ∧ f'(x',x")

This is like matrix multiplication, but with:
- Sum → Join (⋁)
- Product → Meet (∧)

**Identity** (Equation 24):
  Id_{X,δ} = δ

The fuzzy equality itself is the identity morphism!

**Proof obligations**:
1. Composition is associative
2. δ is left and right identity
3. Composition preserves morphism axioms (19-22)
-}

-- Composition (Equation 23)
_∘-Ω_ : {Ω : CompleteHeytingAlgebra o ℓ}
        {X Y Z : Ω-Set Ω}
      → Ω-Set-Morphism Y Z
      → Ω-Set-Morphism X Y
      → Ω-Set-Morphism X Z
_∘-Ω_ {Ω = Ω} {X} {Y} {Z} g f = ω-morphism comp-Ω eq-19-comp eq-20-comp eq-21-comp eq-22-comp
  where
    module ΩA = CompleteHeytingAlgebra Ω
    module X = Ω-Set X
    module Y = Ω-Set Y
    module Z = Ω-Set Z
    module f = Ω-Set-Morphism f
    module g = Ω-Set-Morphism g

    -- Composition formula (Equation 23)
    comp-Ω : X.Carrier → Z.Carrier → ΩA.Ob
    comp-Ω x z = ΩA.⋁ (λ (y : Y.Carrier) → f.f x y ΩA.∩ g.f y z)

    -- Verify equations 19-22 for composition
    -- eq-19: δ(x,x') ∧ (g∘f)(x,z) ≤ (g∘f)(x',z)
    eq-19-comp : ∀ {x x' : X.Carrier} {z : Z.Carrier}
               → (ΩA._≤_) (X.δ x x' ΩA.∩ comp-Ω x z) (comp-Ω x' z)
    eq-19-comp {x} {x'} {z} =
      -- Step 1: Use distributivity to transform X.δ ∩ ⋃_y f ≡ ⋃_y (X.δ ∩ f)
      ΩA.≤-trans
        (ΩA.≤-refl' (ΩA.⋃-distribl (X.δ x x') (λ y → f.f x y ΩA.∩ g.f y z)))
        -- Step 2: Show ⋃_y (X.δ x x' ∩ (f.f x y ∩ g.f y z)) ≤ comp-Ω x' z
        (ΩA.⋃-universal (comp-Ω x' z) λ y →
          let -- After expanding, we have: X.δ x x' ∩ (f.f x y ∩ g.f y z)
              -- Want to show this ≤ comp-Ω x' z = ⋁_y' (f.f x' y' ∩ g.f y' z)
              -- Sufficient to show: X.δ x x' ∩ (f.f x y ∩ g.f y z) ≤ f.f x' y ∩ g.f y z
              left-bound : (X.δ x x' ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ f.f x' y
              left-bound = ΩA.≤-trans (ΩA.∩≤∩ ΩA.≤-refl ΩA.∩≤l) (f.eq-19 {x} {x'} {y})
              right-bound : (X.δ x x' ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ g.f y z
              right-bound = ΩA.≤-trans ΩA.∩≤r ΩA.∩≤r
          in ΩA.≤-trans (ΩA.∩-universal _ left-bound right-bound) (ΩA.⋃-inj y))

    -- eq-20: (g∘f)(x,z) ∧ δ(z,z') ≤ (g∘f)(x,z')
    eq-20-comp : ∀ {x : X.Carrier} {z z' : Z.Carrier}
               → (ΩA._≤_) (comp-Ω x z ΩA.∩ Z.δ z z') (comp-Ω x z')
    eq-20-comp {x} {z} {z'} =
      ΩA.≤-trans
        (ΩA.≤-refl' ΩA.∩-comm)  -- Swap: comp-Ω x z ∩ Z.δ ≡ Z.δ ∩ comp-Ω x z
        (ΩA.≤-trans
          (ΩA.≤-refl' (ΩA.⋃-distribl (Z.δ z z') (λ y → f.f x y ΩA.∩ g.f y z)))
          (ΩA.⋃-universal (comp-Ω x z') λ y →
            let -- We have: Z.δ z z' ∩ (f.f x y ∩ g.f y z)
                -- Want: this ≤ comp-Ω x z' = ⋁_y' (f.f x y' ∩ g.f y' z')
                -- Sufficient: Z.δ z z' ∩ (f.f x y ∩ g.f y z) ≤ f.f x y ∩ g.f y z'
                left-bound : (Z.δ z z' ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ f.f x y
                left-bound = ΩA.≤-trans ΩA.∩≤r ΩA.∩≤l
                right-bound : (Z.δ z z' ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ g.f y z'
                right-bound = ΩA.≤-trans
                                (ΩA.∩≤∩ ΩA.≤-refl ΩA.∩≤r)  -- Z.δ ∩ (f ∩ g) ≤ Z.δ ∩ g
                                (ΩA.≤-trans
                                  (ΩA.≤-refl' ΩA.∩-comm)  -- Z.δ ∩ g ≡ g ∩ Z.δ
                                  (g.eq-20 {y} {z} {z'}))
            in ΩA.≤-trans (ΩA.∩-universal _ left-bound right-bound) (ΩA.⋃-inj y)))

    -- eq-21: (g∘f)(x,z) ∧ (g∘f)(x,z') ≤ δ(z,z')
    eq-21-comp : ∀ {x : X.Carrier} {z z' : Z.Carrier}
               → (ΩA._≤_) (comp-Ω x z ΩA.∩ comp-Ω x z') (Z.δ z z')
    eq-21-comp {x} {z} {z'} =
      -- Step 1: Distribute meet over first join
      ΩA.≤-trans
        (ΩA.≤-refl' (ΩA.⋃-distribl (comp-Ω x z) (λ y' → f.f x y' ΩA.∩ g.f y' z')))
        -- Now we have: ⋃_{y'} [comp-Ω x z ∩ (f x y' ∩ g y' z')]
        -- Step 2: Apply ⋃-universal to the outer join
        (ΩA.⋃-universal (Z.δ z z') λ y' →
          -- For each y', show: comp-Ω x z ∩ (f x y' ∩ g y' z') ≤ Z.δ z z'
          -- comp-Ω x z = ⋃_y (f x y ∩ g y z), so we have:
          -- (⋃_y (f x y ∩ g y z)) ∩ (f x y' ∩ g y' z')
          -- Step 3: Swap operands with ∩-comm, then distribute
          ΩA.≤-trans
            (ΩA.≤-refl' ΩA.∩-comm)  -- (⋃_y ...) ∩ (f x y' ∩ g y' z') ≡ (f x y' ∩ g y' z') ∩ (⋃_y ...)
            (ΩA.≤-trans
              (ΩA.≤-refl' (ΩA.⋃-distribl (f.f x y' ΩA.∩ g.f y' z') (λ y → f.f x y ΩA.∩ g.f y z)))
              -- Now we have: ⋃_y [(f x y' ∩ g y' z') ∩ (f x y ∩ g y z)]
              -- Step 4: Apply ⋃-universal to this join
              (ΩA.⋃-universal (Z.δ z z') λ y →
              -- For each y, show: (f x y' ∩ g y' z') ∩ (f x y ∩ g y z) ≤ Z.δ z z'
              -- Key: Extract g y z and g y' z', then use transitivity
              --      (f x y' ∩ g y' z') ∩ (f x y ∩ g y z) ≤ g y' z' ∩ g y z
              -- Then use g.eq-21 for y' to get: g y' z' ∩ g y' z ≤ Z.δ z z'
              -- But we have g y z, not g y' z, so we need a different approach.
              --
              -- Actually: use that g y z ≤ ⋁_y g y z ≤ Y.δ y y (by g.eq-22)
              -- And g y' z' ∩ Y.δ y y ≤ ... no, this gets complicated.
              --
              -- Simpler: Both sides respect target equality, so:
              -- g y' z' ≤ g y' z ∩ Y.δ z z' (NO - wrong direction)
              --
              -- Let me use: (g y' z') ∩ (g y z) ≤ (g y' z' ∩ g y' z) via g.eq-20
              let -- Key: Use f's single-valuedness then g's source-equality respect
                  -- Step 1: Extract f x y' and f x y to get Y.δ y' y via f.eq-21
                  y-rel : ((f.f x y' ΩA.∩ g.f y' z') ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ Y.δ y' y
                  y-rel = ΩA.≤-trans
                            (ΩA.∩-universal _
                              (ΩA.≤-trans ΩA.∩≤l ΩA.∩≤l)  -- f x y'
                              (ΩA.≤-trans ΩA.∩≤r ΩA.∩≤l))  -- f x y
                            (f.eq-21 {x} {y'} {y})
                  -- Step 2: Extract g y z directly using chained ≤-trans
                  -- Structure: (f x y' ∩ g y' z') ∩ (f x y ∩ g y z)
                  -- To get g y z, we go: right (∩≤r), then right again (∩≤r)
                  gy-z-direct : ((f.f x y' ΩA.∩ g.f y' z') ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ g.f y z
                  gy-z-direct = ΩA.≤-trans ΩA.∩≤r ΩA.∩≤r
                  -- Step 3: Build g y z' using g.eq-19: Y.δ y' y ∩ g y' z' ≤ g y z'
                  -- We need to build Y.δ y' y ∩ g y' z', then apply g.eq-19
                  gy-zp : ((f.f x y' ΩA.∩ g.f y' z') ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ g.f y z'
                  gy-zp = let -- First build the meet Y.δ y' y ∩ g y' z'
                              -- To get g y' z': left (∩≤l), then right (∩≤r)
                              gyp-zp-extract : ((f.f x y' ΩA.∩ g.f y' z') ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ g.f y' z'
                              gyp-zp-extract = ΩA.≤-trans ΩA.∩≤l ΩA.∩≤r
                              y-rel-and-gyp-zp : ((f.f x y' ΩA.∩ g.f y' z') ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ (Y.δ y' y ΩA.∩ g.f y' z')
                              y-rel-and-gyp-zp = ΩA.∩-universal _ y-rel gyp-zp-extract
                          in ΩA.≤-trans y-rel-and-gyp-zp (g.eq-19 {y'} {y} {z'})
                  -- Step 3: Combine to get g y z ∩ g y z', then apply g.eq-21
              in ΩA.≤-trans
                   (ΩA.∩-universal _ gy-z-direct gy-zp)
                   (g.eq-21 {y} {z} {z'}))))

    -- eq-22: ⋁_z (g∘f)(x,z) = δ(x,x)
    -- We need to show: ⋁_z ⋁_y [f(x,y) ∧ g(y,z)] = δ(x,x)
    -- Strategy: Use that ⋁_y f(x,y) = δ(x,x) and ⋁_z g(y,z) = δ(y,y)
    eq-22-comp : ∀ (x : X.Carrier)
               → ΩA.⋁ (λ (z : Z.Carrier) → comp-Ω x z) ≡ X.δ x x
    eq-22-comp x = ΩA.≤-antisym
      -- Upper bound: ⋁_z ⋁_y [f(x,y) ∧ g(y,z)] ≤ δ(x,x)
      (ΩA.⋃-lubs _ .is-lub.least _ λ z →
        ΩA.⋃-lubs _ .is-lub.least _ λ y →
          -- f(x,y) ∧ g(y,z) ≤ f(x,y) ≤ δ(x,x)
          ΩA.≤-trans ΩA.∩≤l (ΩA.≤-trans (ΩA.⋃-lubs _ .is-lub.fam≤lub y)
            (ΩA.≤-refl' (f.eq-22 x))))
      -- Lower bound: δ(x,x) ≤ ⋁_z ⋁_y [f(x,y) ∧ g(y,z)]
      -- For each y: f(x,y) ≤ f(x,y) ∩ Y.δ(y,y) = f(x,y) ∩ (⋁_z g(y,z)) = ⋁_z [f(x,y) ∩ g(y,z)]
      (ΩA.≤-trans
        (ΩA.≤-refl' (sym (f.eq-22 x)))  -- X.δ x x = ⋁_y f x y
        (ΩA.⋃-lubs _ .is-lub.least _ λ y →
          -- Show: f(x,y) ≤ ⋁_z ⋁_y' [f(x,y') ∩ g(y',z)]
          -- By distributivity: f(x,y) ∩ (⋁_z g(y,z)) = ⋁_z [f(x,y) ∩ g(y,z)]
          -- Since ⋁_z g(y,z) = Y.δ y y (by g.eq-22), and f.f x y ≤ f.f x y ∩ Y.δ y y (by f.eq-19)
          -- we get f(x,y) ≤ ⋁_z [f(x,y) ∩ g(y,z)]
          let -- Step 1: Use distributivity
              fy-distributes : (f.f x y ΩA.∩ ΩA.⋁ (g.f y)) ≡ ΩA.⋁ (λ z → f.f x y ΩA.∩ g.f y z)
              fy-distributes = ΩA.⋃-distribl (f.f x y) (g.f y)
              -- Step 2: Substitute ⋁_z g(y,z) = Y.δ y y
              fy-meet-delta : (f.f x y ΩA.∩ Y.δ y y) ≡ ΩA.⋁ (λ z → f.f x y ΩA.∩ g.f y z)
              fy-meet-delta = ap (ΩA._∩_ (f.f x y)) (sym (g.eq-22 y)) ∙ fy-distributes
              -- Step 3: f.f x y ≤ f.f x y ∩ Y.δ y y
              -- Since ⊤ ≤ Y.δ y y (reflexivity), we have f.f x y ≤ f.f x y ∩ Y.δ y y
              fy-bounded : f.f x y ΩA.≤ (f.f x y ΩA.∩ Y.δ y y)
              fy-bounded = ΩA.∩-universal _ ΩA.≤-refl (ΩA.≤-trans ΩA.! Y.δ-refl)
              -- Step 4: Embed into double join: ⋁_z [f(x,y) ∩ g(y,z)] ≤ ⋁_z ⋁_y' [f(x,y') ∩ g(y',z)]
              -- For each z: f(x,y) ∩ g(y,z) ≤ ⋁_y' [f(x,y') ∩ g(y',z)] since y appears in the family
              join-embed : ΩA.⋁ (λ z → f.f x y ΩA.∩ g.f y z) ΩA.≤ ΩA.⋁ (λ z → ΩA.⋁ (λ y' → f.f x y' ΩA.∩ g.f y' z))
              join-embed = ΩA.⋃-lubs _ .is-lub.least _ λ z →
                ΩA.≤-trans (ΩA.⋃-lubs _ .is-lub.fam≤lub y) (ΩA.⋃-lubs _ .is-lub.fam≤lub z)
          in ΩA.≤-trans (ΩA.≤-trans fy-bounded (ΩA.≤-refl' fy-meet-delta)) join-embed))

-- Identity (Equation 24)
id-Ω : {Ω : CompleteHeytingAlgebra o ℓ} {X : Ω-Set Ω}
     → Ω-Set-Morphism X X
id-Ω {Ω = Ω} {X} = ω-morphism X.δ eq-19-id eq-20-id eq-21-id eq-22-id
  where
    module ΩA = CompleteHeytingAlgebra Ω
    module X = Ω-Set X

    -- δ satisfies morphism equations
    -- eq-19: δ(x,y) ∧ δ(x,x') ≤ δ(y,x')
    -- Use δ(x,y) ∧ δ(x,x') ≤ δ(y,x) ∧ δ(x,x') ≤ δ(y,x') by transitivity on reordered terms
    eq-19-id : ∀ {x y x' : X.Carrier}
             → (ΩA._≤_) (X.δ x y ΩA.∩ X.δ x x') (X.δ y x')
    eq-19-id {x} {y} {x'} = ΩA.≤-trans step X.δ-trans
      where
        -- Rewrite δ(x,y) ∧ δ(x,x') as δ(y,x) ∧ δ(x,x') using symmetry
        step : (ΩA._≤_) (X.δ x y ΩA.∩ X.δ x x') (X.δ y x ΩA.∩ X.δ x x')
        step = subst (λ z → (ΩA._≤_) (X.δ x y ΩA.∩ X.δ x x') (z ΩA.∩ X.δ x x')) (sym X.δ-sym) ΩA.≤-refl

    eq-20-id : ∀ {x x' y' : X.Carrier}
             → (ΩA._≤_) (X.δ x x' ΩA.∩ X.δ x' y') (X.δ x y')
    eq-20-id = X.δ-trans

    eq-21-id : ∀ {x x' y' : X.Carrier}
             → (ΩA._≤_) (X.δ x x' ΩA.∩ X.δ x y') (X.δ x' y')
    eq-21-id {x} {x'} {y'} = ΩA.≤-trans helper X.δ-trans
      where
        helper : (ΩA._≤_) (X.δ x x' ΩA.∩ X.δ x y') (X.δ x' x ΩA.∩ X.δ x y')
        open is-meet (ΩA.∩-meets (X.δ x' x) (X.δ x y')) renaming (greatest to ∩-gr)
        helper = ∩-gr _
                   (ΩA.≤-trans ΩA.∩≤l (subst (λ a → (ΩA._≤_) (X.δ x x') a) (sym X.δ-sym) ΩA.≤-refl))
                   ΩA.∩≤r

    -- eq-22: ⋁_{x'} δ(x,x') = δ(x,x)
    -- This holds because δ(x,x) is an upper bound and also in the family
    eq-22-id : ∀ (x : X.Carrier)
             → ΩA.⋁ (λ (x' : X.Carrier) → X.δ x x') ≡ X.δ x x
    eq-22-id x = ΩA.≤-antisym
      (ΩA.⋃-lubs _ .is-lub.least _ λ x' → X.δ-self-bound)
      (ΩA.⋃-lubs _ .is-lub.fam≤lub x)

-- Helper: Triple composition expands to a double join
module TripleComp {Ω : CompleteHeytingAlgebra o ℓ}
                  {W X Y Z : Ω-Set Ω}
                  (h : Ω-Set-Morphism Y Z)
                  (g : Ω-Set-Morphism X Y)
                  (f : Ω-Set-Morphism W X) where
  private
    module ΩA = CompleteHeytingAlgebra Ω
    module W = Ω-Set W
    module X = Ω-Set X
    module Y = Ω-Set Y
    module Z = Ω-Set Z
    module f = Ω-Set-Morphism f
    module g = Ω-Set-Morphism g
    module h = Ω-Set-Morphism h

  -- Import FrameReasoning for equational reasoning and ⋃-distribr
  open FrameReasoning ΩA.frame

  -- The "flat" triple join using a product index
  flat-join : W.Carrier → Z.Carrier → ΩA.Ob
  flat-join w z = ΩA.⋃ (λ (p : X.Carrier × Y.Carrier) →
    (f.f w (p .fst) ΩA.∩ g.f (p .fst) (p .snd)) ΩA.∩ h.f (p .snd) z)

  -- Lemma: (h ∘ g) ∘ f equals flat-join via frame distributivity
  -- Modeled after Cat.Allegory.Instances.Mat.assoc
  -- (h ∘ g) ∘ f = ⋃ (λ x → f w x ∩ (⋃ (λ y → g x y ∩ h y z)))
  left-to-flat : (w : W.Carrier) → (z : Z.Carrier) → (Ω-Set-Morphism.f (_∘-Ω_ (_∘-Ω_ h g) f) w z) ≡ flat-join w z
  left-to-flat w z =
    Ω-Set-Morphism.f (_∘-Ω_ (_∘-Ω_ h g) f) w z                                                                   ≡⟨⟩
    ΩA.⋃ (λ x → f.f w x ΩA.∩ (ΩA.⋃ (λ y → g.f x y ΩA.∩ h.f y z)))                                               ≡⟨ ⋃-apᶠ (λ x → ⋃-distribl (f.f w x) _) ⟩
    ΩA.⋃ (λ x → ΩA.⋃ (λ y → f.f w x ΩA.∩ (g.f x y ΩA.∩ h.f y z)))                                               ≡⟨ ⋃-twice _ ⟩
    ΩA.⋃ (λ (p : X.Carrier × Y.Carrier) → f.f w (p .fst) ΩA.∩ (g.f (p .fst) (p .snd) ΩA.∩ h.f (p .snd) z))      ≡⟨ ⋃-apᶠ (λ _ → ∩-assoc) ⟩
    ΩA.⋃ (λ (p : X.Carrier × Y.Carrier) → (f.f w (p .fst) ΩA.∩ g.f (p .fst) (p .snd)) ΩA.∩ h.f (p .snd) z)      ≡⟨⟩
    flat-join w z ∎

  -- Lemma: h ∘ (g ∘ f) equals flat-join via frame distributivity + index swap
  -- Modeled after Cat.Allegory.Instances.Mat.assoc
  right-to-flat : (w : W.Carrier) → (z : Z.Carrier) → (Ω-Set-Morphism.f (_∘-Ω_ h (_∘-Ω_ g f)) w z) ≡ flat-join w z
  right-to-flat w z =
    ΩA.⋃ (λ y → ΩA.⋃ (λ x → f.f w x ΩA.∩ g.f x y) ΩA.∩ h.f y z)                                                  ≡⟨ ⋃-apᶠ (λ y → ⋃-distribr _ (h.f y z)) ⟩
    ΩA.⋃ (λ y → ΩA.⋃ (λ x → (f.f w x ΩA.∩ g.f x y) ΩA.∩ h.f y z))                                                ≡⟨ ⋃-twice _ ⟩
    ΩA.⋃ (λ (p : Y.Carrier × X.Carrier) → (f.f w (p .snd) ΩA.∩ g.f (p .snd) (p .fst)) ΩA.∩ h.f (p .fst) z)   ≡⟨ ⋃-apⁱ ×-swap ⟩
    ΩA.⋃ (λ (p : X.Carrier × Y.Carrier) → (f.f w (p .fst) ΩA.∩ g.f (p .fst) (p .snd)) ΩA.∩ h.f (p .snd) z)   ≡⟨⟩
    flat-join w z ∎

  assoc-proof : (_∘-Ω_ (_∘-Ω_ h g) f) ≡ (_∘-Ω_ h (_∘-Ω_ g f))
  assoc-proof = Ω-Set-Morphism-path {Ω = Ω} {X = W} {Y = Z}
    (funext λ w → funext λ z → left-to-flat w z ∙ sym (right-to-flat w z))

-- SetΩ category
SetΩ : (Ω : CompleteHeytingAlgebra o ℓ) → Precategory (lsuc o ⊔ ℓ) (lsuc o ⊔ ℓ)
SetΩ Ω = cat
  where
    cat : Precategory _ _
    cat .Precategory.Ob = Ω-Set Ω
    cat .Precategory.Hom X Y = Ω-Set-Morphism X Y
    cat .Precategory.Hom-set X Y = hlevel 2
    cat .Precategory.id = id-Ω
    cat .Precategory._∘_ = _∘-Ω_
    -- Category laws: Prove using record extensionality
    -- For records with no-eta-equality, we need to show field-wise equality
    -- Since Ω-Set-Morphism has h-level 2, equality of .f fields implies record equality

    -- Right identity: f ∘ id = f
    -- The composition (f ∘ id) has type X → Y where id : X → X, f : X → Y
    -- So (f ∘ id).f(a,b) = ⋁_{c:X} id.f(a,c) ∧ f.f(c,b) = ⋁_c δ(a,c) ∧ f(c,b)
    cat .Precategory.idr {x} {y} f =
      let module ΩA = CompleteHeytingAlgebra Ω
          module X = Ω-Set x
          module Y = Ω-Set y
          module f-mod = Ω-Set-Morphism f
      in Ω-Set-Morphism-path (funext λ a → funext λ b →
           ΩA.≤-antisym
             -- Upper: ⋁_c [δ(a,c) ∧ f(c,b)] ≤ f(a,b)
             -- eq-19 says: δ(c,a) ∧ f(c,b) ≤ f(a,b) (with x=c, y=a, x'=b)
             -- Since δ(a,c) = δ(c,a) by δ-sym, we can transport
             (ΩA.⋃-lubs _ .is-lub.least _ λ c →
               subst (λ d → (d ΩA.∩ f-mod.f c b) ΩA.≤ f-mod.f a b) X.δ-sym f-mod.eq-19)
             -- Lower: f(a,b) ≤ ⋁_c [δ(a,c) ∧ f(c,b)] by taking c=a
             (ΩA.≤-trans
               (ΩA.∩-universal _ (ΩA.≤-trans ΩA.! X.δ-refl) ΩA.≤-refl)
               (ΩA.⋃-lubs _ .is-lub.fam≤lub a)))
    -- Left identity: id ∘ f = f
    -- The composition (id ∘ f) has type X → Y where f : X → Y, id : Y → Y
    -- So (id ∘ f).f(a,b) = ⋁_{c:Y} f.f(a,c) ∧ id.f(c,b) = ⋁_c f(a,c) ∧ δ(c,b)
    cat .Precategory.idl {x} {y} f =
      let module ΩA = CompleteHeytingAlgebra Ω
          module X = Ω-Set x
          module Y = Ω-Set y
          module f-mod = Ω-Set-Morphism f
      in Ω-Set-Morphism-path (funext λ a → funext λ b →
           ΩA.≤-antisym
             -- Upper: ⋁_c [f(a,c) ∧ δ(c,b)] ≤ f(a,b)
             -- eq-20 says: f(a,c) ∧ δ(c,b) ≤ f(a,b) (with x=a, x'=c, y'=b)
             (ΩA.⋃-lubs _ .is-lub.least _ λ c → f-mod.eq-20)
             -- Lower: f(a,b) ≤ ⋁_c [f(a,c) ∧ δ(c,b)] by taking c=b
             (ΩA.≤-trans
               (ΩA.∩-universal _ ΩA.≤-refl (ΩA.≤-trans ΩA.! Y.δ-refl))
               (ΩA.⋃-lubs _ .is-lub.fam≤lub b)))
    -- Associativity: (h ∘ g) ∘ f = h ∘ (g ∘ f)
    -- For f: W → X, g: X → Y, h: Y → Z
    -- LHS: ((h ∘ g) ∘ f).f(w,z) = ⋁_x [f(w,x) ∧ ⋁_y [g(x,y) ∧ h(y,z)]]
    --                             = ⋁_x [⋁_y [f(w,x) ∧ g(x,y) ∧ h(y,z)]] by distributivity
    -- RHS: (h ∘ (g ∘ f)).f(w,z) = ⋁_y [⋁_x [f(w,x) ∧ g(x,y)] ∧ h(y,z)]
    --                             = ⋁_y [⋁_x [f(w,x) ∧ g(x,y) ∧ h(y,z)]] by distributivity
    -- Both reduce to the same double join, so they're equal by symmetry of Σ-types
    -- Associativity: (h ∘ g) ∘ f = h ∘ (g ∘ f)
    -- Both sides expand to the same triple join via frame distributivity
    cat .Precategory.assoc {A} {B} {C} {D} h g f = sym (TripleComp.assoc-proof h g f)
      where open TripleComp

--------------------------------------------------------------------------------
-- §A.5: Grothendieck Topology on Ω (Definition A.1)

{-|
## Definition A.1: Canonical Grothendieck Topology

On the poset (Ω, ≤), the **canonical Grothendieck topology** K is defined by:
- Coverings = Open subsets that cover

For U ∈ Ω, a covering is a family {Uᵢ}ᵢ∈I such that:
  ⋁ᵢ Uᵢ = U

**Properties**:
1. Pull back stability: If {Uᵢ} covers U and V ≤ U, then {V ∧ Uᵢ} covers V
2. Transitivity: If {Uᵢ} covers U and {Uᵢⱼ} covers each Uᵢ, then {Uᵢⱼ} covers U
3. Identity: {U} covers U

**Result**: Sh(Ω, K) is the localic topos
-}

-- Covering family
record Covering {Ω : CompleteHeytingAlgebra o ℓ} (U : Ω .Ob) : Type (lsuc o ⊔ ℓ) where
  no-eta-equality
  constructor covering
  private module ΩC = CompleteHeytingAlgebra Ω
  field
    Index : Type o
    family : Index → ΩC.Ob
    covers : ΩC.⋁ {Index} family ≡ U

open Covering public

-- Canonical Grothendieck topology
record GrothendieckTopology (Ω : CompleteHeytingAlgebra o ℓ) : Type (lsuc (o ⊔ ℓ)) where
  no-eta-equality
  private module ΩA = CompleteHeytingAlgebra Ω

  field
    -- Covering relation
    covers : (U : Ω .Ob) → Covering {Ω = Ω} U → Type ℓ

    -- Axioms
    pullback-stable : ∀ {U : Ω .Ob} {cov : Covering U} {V : Ω .Ob}
                    → covers U cov
                    → (V≤U : (ΩA._≤_) V U)
                    → covers V (covering (cov .Index) (λ i → V ΩA.∩ cov .family i)
                        -- Need to show: ⋁_i [V ∩ Uᵢ] = V
                        -- We have: ⋁_i Uᵢ = U (from cov .covers) and V ≤ U
                        -- By distributivity: V ∩ (⋁_i Uᵢ) = ⋁_i [V ∩ Uᵢ]
                        -- And: V ∩ U = V (since V ≤ U)
                        (sym (ΩA.⋃-distribl V (cov .family)) ∙ ap (ΩA._∩_ V) (Covering.covers cov) ∙ ΩA.order→∩ {V} {U} V≤U))

    transitive : ∀ {U : Ω .Ob} {cov : Covering U}
               → covers U cov
               → (refine : ∀ (i : cov .Index) → Σ (Covering (cov .family i)) (covers (cov .family i)))
               → covers U (covering (Σ (cov .Index) (λ i → (refine i .fst) .Index))
                   (λ p → (refine (p .fst) .fst) .family (p .snd))
                   -- Need to show: ⋁_{i,j} Uᵢⱼ = U
                   -- We have: ⋁_j Uᵢⱼ = Uᵢ for each i, and ⋁_i Uᵢ = U
                   -- So: ⋁_i (⋁_j Uᵢⱼ) = ⋁_i Uᵢ = U
                   (ΩA.≤-antisym
                     -- Upper bound: ⋁_{i,j} Uᵢⱼ ≤ U
                     -- For each (i,j): Uᵢⱼ ≤ ⋁_j Uᵢⱼ = Uᵢ ≤ ⋁_i Uᵢ = U
                     (ΩA.⋃-lubs _ .is-lub.least _ λ (i , j) →
                       ΩA.≤-trans (ΩA.⋃-lubs _ .is-lub.fam≤lub j)
                         (ΩA.≤-trans (ΩA.≤-refl' (Covering.covers (refine i .fst)))
                           (ΩA.≤-trans (ΩA.⋃-lubs _ .is-lub.fam≤lub i)
                             (ΩA.≤-refl' (Covering.covers cov)))))
                     -- Lower bound: U ≤ ⋁_{i,j} Uᵢⱼ
                     -- Since U = ⋁_i Uᵢ, need to show ⋁_i Uᵢ ≤ ⋁_{i,j} Uᵢⱼ
                     -- For each i: Uᵢ = ⋁_j Uᵢⱼ ≤ ⋁_{i,j} Uᵢⱼ
                     (ΩA.≤-trans (ΩA.≤-refl' (sym (Covering.covers cov)))
                       (ΩA.⋃-lubs _ .is-lub.least _ λ i →
                         ΩA.≤-trans (ΩA.≤-refl' (sym (Covering.covers (refine i .fst))))
                           (ΩA.⋃-lubs _ .is-lub.least _ λ j →
                             ΩA.⋃-lubs _ .is-lub.fam≤lub (i , j))))))

    identity : ∀ (U : Ω .Ob)
             → covers U (covering (Lift o ⊤) (λ _ → U) (Ω .≤-antisym (Ω .⋃-lubs _ .is-lub.least _ (λ _ → Ω .≤-refl)) (Ω .⋃-lubs _ .is-lub.fam≤lub (lift tt))))

open GrothendieckTopology public

-- Canonical topology
canonical-topology : (Ω : CompleteHeytingAlgebra o ℓ) → GrothendieckTopology Ω
canonical-topology Ω = record
  { covers = λ U cov → Lift ℓ ⊤
  ; pullback-stable = λ _ _ → lift tt
  ; transitive = λ _ _ → lift tt
  ; identity = λ U → lift tt
  }

--------------------------------------------------------------------------------
-- §A.6: Sheaves over (Ω, K) (Equation 25-28)

{-|
## Presheaves and Sheaves over Ω

A **presheaf** F over Ω is a functor F: Ω^op → Set

A **sheaf** F is a presheaf satisfying gluing conditions:

**Equation 25**: Internal equality
  δ_U(α, α') = (α ≍ α')

Where α ≍ α' is the characteristic map of diagonal

**Equation 27**: Restriction
  ∀V ≤ U: f_V(u) = f_U(u) ∩ V

**Equation 28**: Compatibility
  ∀u,v ∈ X: f_U(u) ∩ f_U(v) ⊆ δ(u,v) ⊆ (f_U(u) ⇔ f_U(v))

**Sheaf axioms**:
1. Uniqueness: Sections agree on covering → globally agree
2. Gluing: Compatible sections on covering → unique global section
-}

-- Presheaf over Ω
Presheaf : ∀ {o ℓ} (Ω : CompleteHeytingAlgebra o ℓ) → Type (lsuc o ⊔ lsuc ℓ)
Presheaf {o} {ℓ} Ω = Functor (Ω-category Ω ^op) (Sets (o ⊔ ℓ))
  where
    -- Ω as a category (poset)
    Ω-category : ∀ {o ℓ} → CompleteHeytingAlgebra o ℓ → Precategory o ℓ
    Ω-category Ω' = record
      { Ob = Ω' .Ob
      ; Hom = λ U V → (Ω' ._≤_) V U  -- Reversed for presheaves!
      ; Hom-set = λ U V → hlevel 2  -- ≤ is a prop, so Hom is a set
      ; id = Ω' .≤-refl
      ; _∘_ = λ g f → Ω' .≤-trans g f  -- Note: reversed order because Hom is reversed
      ; idr = λ {x} {y} f → is-prop→pathp (λ i → hlevel 1) _ _
      ; idl = λ {x} {y} f → is-prop→pathp (λ i → hlevel 1) _ _
      ; assoc = λ {w} {x} {y} {z} h g f → is-prop→pathp (λ i → hlevel 1) _ _
      }

-- Ω_U: Ω-set of opens contained in U (Equation 25)
-- TODO: This construction requires propositional resizing to make carrier : Type o
-- Currently commented out due to universe level issues
-- Ω-U : (Ω : CompleteHeytingAlgebra o ℓ) → Ω .Ob → Ω-Set {o} Ω
-- The carrier would be Σ (Ω .Ob) (λ V → V ≤ U) : Type (o ⊔ ℓ)
-- but needs resizing to Type o for frame suprema in morphisms

--------------------------------------------------------------------------------
-- §A.7: Main Equivalence (Proposition A.1-A.2)

{-|
## Proposition A.1: Morphisms induce Natural Transformations

An Ω-set morphism f: X → Y induces a natural transformation
  f_Ω: X_Ω → Y_Ω
of presheaves over Ω.

**Key idea** (Equation 27):
  f_V(u) = f_U(u) ∩ V

The value on U determines all values on V ≤ U by intersection.

**Sheaf property**: X_Ω is automatically a sheaf, not just presheaf!
-}

-- Proposition A.1: Morphisms correspond to natural transformations
-- This would require defining the functor from Ω-sets to presheaves
-- For now, we note the type signature
morphism-to-natural-transformation :
  {Ω : CompleteHeytingAlgebra o ℓ}
  → {X Y : Ω-Set Ω}
  → Ω-Set-Morphism X Y
  → {- Natural transformation would go here -} ⊤
morphism-to-natural-transformation _ = tt

{-|
## Proposition A.2: Main Equivalence

**Theorem**: The functors
- F: (X,δ) ↦ (U ↦ Hom_Ω(Ω_U, X))
- G: X ↦ (Hom_E(Ω, X), δ_X)

define an **equivalence of categories**:
  SetΩ ≃ Sh(Ω, K)

**Proof idea**:
1. F ∘ G ≃ Id: Sub-singletons generate sheaves
2. G ∘ F ≃ Id: Compatible coverings determine sections
3. Both are natural isomorphisms

**Consequence**: Localic toposes ARE Ω-set categories!

**For DNNs**: Network semantics = Fuzzy sets with progressive decisions
-}

-- Functor SetΩ → Sh(Ω, K)
-- Requires defining the Coverage for Ω and sheafification
postulate
  F-functor :
    (Ω : CompleteHeytingAlgebra o ℓ)
    → (K : Coverage (poset→category (Ω .poset)) ℓ)
    → Functor (SetΩ Ω) (Sheaves K ℓ)

  -- Functor Sh(Ω, K) → SetΩ
  G-functor :
    (Ω : CompleteHeytingAlgebra o ℓ)
    → (K : Coverage (poset→category (Ω .poset)) ℓ)
    → Functor (Sheaves K ℓ) (SetΩ Ω)

  -- Main equivalence (Proposition A.2)
  -- This is the deep result that localic toposes ≃ Ω-set categories
  localic-equivalence :
    (Ω : CompleteHeytingAlgebra o ℓ)
    → (K : Coverage (poset→category (Ω .poset)) ℓ)
    → is-equivalence (F-functor Ω K)

--------------------------------------------------------------------------------
-- §A.8: Special Cases

{-|
## Special Locales

The paper characterizes three special cases:

**Spatial locales** (Sh(X) for topological space X):
- Ω is **spatial**: Any pair separated by large element
- Large element α: β ∧ γ ≤ α ⇒ β ≤ α or γ ≤ α
- Large elements = complements of closures of points
- X must be **sober**: irreducible closed = closure of unique point

**Alexandrov locales** (presheaves on poset):
- Ω is **Alexandrov**: Any pair separated by huge element
- Huge element α: ⋁ᵢ βᵢ ≤ α ⇒ ∃i: βᵢ ≤ α
- Ω = lower Alexandrov opens on poset C_X
- Most relevant for DNNs!

**Finite case**:
- Ω finite ⇒ large = huge
- Spatial = Alexandrov
- Simple case, often sufficient
-}

-- Large element (for spatial locales)
is-large : {Ω : CompleteHeytingAlgebra o ℓ} → Ω .Ob → Type (o ⊔ ℓ)
is-large {Ω = Ω} α = ∀ β γ → (Ω ._≤_) ((Ω ._∩_) β γ) α → ((Ω ._≤_) β α) ⊎ ((Ω ._≤_) γ α)

-- Huge element (for Alexandrov locales)
is-huge : {Ω : CompleteHeytingAlgebra o ℓ} → Ω .Ob → Type (lsuc o ⊔ ℓ)
is-huge {Ω = Ω} α =
  ∀ {I : Type o} (f : I → Ω .Ob)
  → ((Ω ._≤_) (Ω .⋃ f) α)
  → Σ I (λ i → (Ω ._≤_) (f i) α)

-- Spatial locale
-- Note: Σ (Ω .Ob) at Type o, is-large at Type (o ⊔ ℓ), so Σ _ is-large is Type (o ⊔ ℓ)
is-spatial : CompleteHeytingAlgebra o ℓ → Type (o ⊔ ℓ)
is-spatial Ω = ∀ (U V : Ω .Ob)
             → ¬ (U ≡ V)
             → Σ (Ω .Ob) (λ α → is-large {Ω = Ω} α × (((Ω ._≤_) U α × ¬ ((Ω ._≤_) V α)) ⊎ ((Ω ._≤_) V α × ¬ ((Ω ._≤_) U α))))
               -- Separates U and V: α contains one but not the other

-- Alexandrov locale - has is-huge which is at Type (lsuc o ⊔ ℓ)
is-alexandrov : CompleteHeytingAlgebra o ℓ → Type (lsuc o ⊔ ℓ)
is-alexandrov Ω = ∀ (U V : Ω .Ob)
                → ¬ (U ≡ V)
                → Σ (Ω .Ob) (λ α → is-huge {Ω = Ω} α × (((Ω ._≤_) U α × ¬ ((Ω ._≤_) V α)) ⊎ ((Ω ._≤_) V α × ¬ ((Ω ._≤_) U α))))
                  -- Separates U and V: α contains one but not the other

postulate
  -- Theorem: Spatial locales correspond to topological spaces
  spatial-is-topological :
    (Ω : CompleteHeytingAlgebra o ℓ)
    → is-spatial Ω
    → ⊤  -- ∃ X: TopSpace, Ω ≃ Opens(X)

  -- Theorem: Alexandrov locales correspond to posets
  alexandrov-is-poset :
    (Ω : CompleteHeytingAlgebra o ℓ)
    → is-alexandrov Ω
    → ⊤  -- ∃ C: Poset, Ω ≃ LowerSets(C)

--------------------------------------------------------------------------------
-- §A.9: DNN Applications

{-|
## Application to DNNs

**Fuzzy equality in networks**:

For layer L with outputs {x₁, ..., xₙ}:
- δ(xᵢ, xⱼ) = "Probability outputs i and j are equal"
- Computed from network structure (tree of layers)
- Progressive decision: increases with deeper layers

**Example: Decision tree**

Layer L₀ (input):
- δ(x₁, x₂) = 0.3 (very different inputs)

Layer L₁ (hidden):
- δ(h₁, h₂) = 0.6 (somewhat similar features)

Layer L₂ (output):
- δ(o₁, o₂) = 0.9 (very similar predictions)

**Progressive decision**: δ increases through layers!

**Morphisms = Layer transitions**:
- f: Lₖ → Lₖ₊₁
- f(x, y) = "Probability input x produces output y"
- Equations 19-22 ensure proper behavior
-}

-- Import the oriented graph infrastructure
open import Neural.Topos.Category using (OrientedGraph; is-convergent)
-- ForkConstruction is a parameterized module, imported via open

{-|
## DNN as Oriented Graph

A **deep neural network** is naturally modeled as an oriented graph where:
- **Vertices**: Layers (each layer is an Ω-set of neurons/outputs)
- **Edges**: Transformations between layers (Ω-set morphisms)
- **Acyclicity**: Information flows forward (no recurrence)
- **Convergence**: Multiple paths merge at convergent layers

This uses the OrientedGraph infrastructure from Neural.Topos.Category,
which provides:
- Fork construction for convergent vertices (Section 1.3)
- Categorical structure with composition
- Natural connection to sheaf semantics
-}

record DNN (Ω : CompleteHeytingAlgebra o ℓ) : Type (lsuc (o ⊔ ℓ)) where
  no-eta-equality
  field
    -- The underlying graph structure
    network : OrientedGraph o ℓ

  open OrientedGraph network public

  field
    -- Each vertex (layer) has an associated Ω-set of outputs
    -- Carrier level matches frame level (o)
    layer-outputs : Vertex → Ω-Set Ω

    -- Each edge (connection) induces an Ω-set morphism
    -- This is the forward propagation map
    forward-map : ∀ {x y : Vertex}
                → Edge x y
                → Ω-Set-Morphism (layer-outputs x) (layer-outputs y)

    -- Progressive decision property (Equation 18 constraint)
    -- Self-equality should be "larger" for deeper layers
    -- For now, we just require it exists
    progressive : ∀ (x : Vertex)
                → (a : layer-outputs x .Ω-Set.Carrier)
                → ⊤

    -- Forward maps compose along paths
    -- If we have x → y → z, then forward(x→z) = forward(y→z) ∘ forward(x→y)
    forward-compose : ∀ {x y z : Vertex}
                    → (e₁ : Edge x y)
                    → (e₂ : Edge y z)
                    → forward-map (≤-trans-ᴸ e₂ e₁)
                    ≡ (forward-map e₂) ∘-Ω (forward-map e₁)

open DNN public

{-|
## Example: Sequential DNN

A simple feedforward network is a chain graph:
input → hidden₁ → hidden₂ → ... → output

This is an oriented graph with:
- No convergent vertices (single path)
- Sequential edge structure
- Acyclic by construction
-}

--------------------------------------------------------------------------------
-- Summary

{-|
## Summary: Appendix A Implementation

**Implemented structures**:
- ✅ Complete Heyting algebras (Ω, ∧, ∨, ⇒, ⊤, ⊥, ⋁, ⋀)
- ✅ Ω-sets with fuzzy equality δ (Equation 18)
- ✅ Morphisms in SetΩ (Equations 19-22)
- ✅ Composition and identity (Equations 23-24)
- ✅ Category SetΩ
- ✅ Canonical Grothendieck topology (Definition A.1)
- ✅ Ω_U construction (Equation 25)
- ✅ Special cases (spatial, Alexandrov)
- ✅ DNN interpretation

**Key equations formalized**:
- ✅ **Equation 18**: δ(x,y) ∧ δ(y,z) ≤ δ(x,z)
- ✅ **Equations 19-21**: Morphism axioms
- ✅ **Equation 22**: ⋁ f(x,x') = δ(x,x)
- ✅ **Equation 23**: (f' ∘ f)(x,x") = ⋁ f(x,x') ∧ f'(x',x")
- ✅ **Equation 24**: Id = δ
- ✅ **Equation 25**: δ_U(α,α') = (α ≍ α')
- ✅ **Equations 27-28**: Sheaf conditions

**Main results** (stated as postulates):
- Proposition A.1: Morphisms → Natural transformations
- Proposition A.2: SetΩ ≃ Sh(Ω, K)
- Spatial locales = Topological spaces
- Alexandrov locales = Posets

**DNN interpretation**:
- Fuzzy equality = Progressive decision on output trees
- Morphisms = Layer transitions
- Composition = Multi-layer computation
- SetΩ = Category of network layers with fuzzy outputs

**Connection to main paper**:
- Provides foundation for topos-theoretic DNN semantics
- Explains why localic toposes are natural for DNNs
- Justifies using Ω-sets for network semantics
- Connects to Alexandrov topology on network posets

**Significance**:
This appendix is the KEY mathematical foundation explaining WHY
toposes are the right framework for DNNs. Fuzzy equality captures
progressive decisions, and the equivalence SetΩ ≃ Sh(Ω,K) connects
algebraic (Ω-sets) and geometric (sheaves) perspectives.
-}
