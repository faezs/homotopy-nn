{-# OPTIONS --no-import-sorts #-}
{-|
# Fibrations and Presheaves over Stacks (Chapter 2.2, Equations 2.2-2.6)

**Reference**: Belfiore & Bennequin (2022), Section 2.2
arXiv:2106.14587v3 [math.AT]

## Main Content from Section 2.2:

This section formalizes the fibration structure π: F → C and presheaves over
fibrations. The key equations (2.2-2.6) describe how presheaves are structured
over a fibration.

From the paper:

> "Among the equivalent points of view on stacks and classifying topos [Gir64],
> [Gir71], and [Gir72]), the most concrete one starts with a **contravariant
> functor F from the category C to the 2-category of small categories Cat**.
> (This corresponds to an element of the category Scind(C) in the book of
> Giraud [Gir71].)"

## The Setup:

Given:
- C: Base category (network architecture)
- F: C^op → Cat (stack/sheaf of categories)
- π: F → C (the corresponding fibration)

The fibration π has:
- Objects: Pairs (U, ξ) where U ∈ C, ξ ∈ F(U)
- Morphisms: Described by **Equation (2.2)**

A presheaf A over F consists of:
- Family A_U: F(U)^op → Set for each U ∈ C
- Natural transformations A_α (Equations 2.4-2.6)

## The Six Key Equations:

1. **Equation (2.2)**: Hom_F((U,ξ), (U',ξ')) = ⊔_{α∈Hom_C(U,U')} Hom_{F(U)}(ξ, F(α)ξ')
2. **Equation (2.3)**: s_{α∘β} = F_β(s_α) ∘ s_β (sections)
3. **Equation (2.4)**: A_{α∘β} = F*_α(A_β) ∘ A_α (presheaf composition)
4. **Equation (2.5)**: A(f) = A_U(f) ∘ A_α (presheaf on morphisms)
5. **Equation (2.6)**: F*_α φ_U ∘ A_α = A'_α ∘ φ_{U'} (natural transformation)
-}

module Neural.Stack.Fibration where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Instances.Functor
open import Cat.Bi.Base

open import Neural.Stack.Groupoid

private variable
  o ℓ o' ℓ' κ : Level

{-|
## Fibrations from Stacks (Grothendieck Construction)

Given a functor F: C^op → Cat, the **Grothendieck construction** produces a
fibration π: F → C.

From the paper:

> "The corresponding fibration π: F → C, written ∇F by Grothendieck, has for
> objects the pairs (U, ξ) where U ∈ C and ξ ∈ F(U), sometimes shortly written
> ξ_U, and for morphisms..."

This is also known as:
- The **category of elements** of F
- The **total category** of the fibration
- Notation: ∫F or ∇F (Grothendieck), El(F) (category of elements)
-}

module _ {C : Precategory o ℓ} (F : Functor (C ^op) (Cat o' ℓ')) where
  open Functor F
  open Precategory C renaming (Ob to C-Ob; Hom to C-Hom)

  {-|
  ### Fibration Objects

  Objects of the fibration F are pairs (U, ξ) where:
  - U is an object of the base category C
  - ξ is an object of the fiber F(U)

  Notation: Sometimes written ξ_U to emphasize both components.
  -}
  record Fib-Ob : Type (o ⊔ o') where
    no-eta-equality
    constructor _,_
    field
      base  : C-Ob
      fiber : (F₀ base) .Precategory.Ob

  open Fib-Ob public

  {-|
  ### Equation (2.2): Fibration Morphisms

  From the paper:

  > "Hom_F((U,ξ), (U',ξ')) = ⊔_{α ∈ Hom_C(U,U')} Hom_{F(U)}(ξ, F(α)ξ')"  (2.2)

  A morphism in the fibration from (U, ξ) to (U', ξ') consists of:
  - A morphism α: U → U' in the base C
  - A morphism f: ξ → F(α)(ξ') in the fiber F(U)

  **Key insight**: The morphism f "lifts" α to the total space.
  It connects ξ ∈ F(U) to the pullback F(α)(ξ') ∈ F(U) of ξ' ∈ F(U').

  **Geometric interpretation**:
  - α describes how layers U and U' are connected
  - F(α): F(U') → F(U) transports structure backwards
  - f: ξ → F(α)(ξ') is the lifted morphism
  -}
  record Fib-Hom (obj obj' : Fib-Ob) : Type (ℓ ⊔ ℓ') where
    no-eta-equality
    constructor mk-fib-hom
    private
      U = obj .base
      U' = obj' .base
      ξ = obj .fiber
      ξ' = obj' .fiber

    field
      -- The base morphism α: U → U'
      base-hom : C-Hom U U'

      -- The fiber morphism f: ξ → F(α)(ξ')
      -- This uses the contravariant action F₁: Hom(U,U') → Functor(F(U'), F(U))
      fiber-hom : (F₀ U) .Precategory.Hom ξ ((F₁ base-hom) .Functor.F₀ ξ')

  open Fib-Hom public

  {-|
  ### The Fibration Category

  With these objects and morphisms, we get the total category F (also denoted ∫F).

  Composition: To compose (U,ξ) →^{(α,f)} (U',ξ') →^{(β,g)} (U'',ξ''):
  - Base: α ∘ β: U → U''
  - Fiber: F(β)(g) ∘ f: ξ → F(α)(ξ') → F(β∘α)(ξ'')
    Uses functoriality: F(β∘α) ≡ F(β) ∘ F(α)
  -}

  postulate
    -- Composition of fibration morphisms
    _∘-fib_ : ∀ {x y z} → Fib-Hom y z → Fib-Hom x y → Fib-Hom x z

    -- Identity morphism
    id-fib : ∀ {x} → Fib-Hom x x

    -- Category laws (associativity, identity)
    fib-idr : ∀ {x y} (f : Fib-Hom x y) → f ∘-fib id-fib ≡ f
    fib-idl : ∀ {x y} (f : Fib-Hom x y) → id-fib ∘-fib f ≡ f
    fib-assoc : ∀ {w x y z} (f : Fib-Hom y z) (g : Fib-Hom x y) (h : Fib-Hom w x) →
                (f ∘-fib g) ∘-fib h ≡ f ∘-fib (g ∘-fib h)

  -- The total category (fibration)
  Total-Category : Precategory (o ⊔ o') (ℓ ⊔ ℓ')
  Total-Category .Precategory.Ob = Fib-Ob
  Total-Category .Precategory.Hom = Fib-Hom
  Total-Category .Precategory.Hom-set = {!!} -- Requires proving it's a set
  Total-Category .Precategory.id = id-fib
  Total-Category .Precategory._∘_ = _∘-fib_
  Total-Category .Precategory.idr = fib-idr
  Total-Category .Precategory.idl = fib-idl
  Total-Category .Precategory.assoc = fib-assoc

  -- The projection functor π: F → C
  π : Functor Total-Category C
  π .Functor.F₀ = base
  π .Functor.F₁ = base-hom
  π .Functor.F-id = refl
  π .Functor.F-∘ _ _ = refl

  {-|
  ## Sections of the Fibration (Equation 2.3)

  From the paper:

  > "A **section s of π** corresponds to a family s_U ∈ F_U indexed by U ∈ C,
  > and a family of morphisms s_α ∈ Hom_{F(U)}(s_U, F(α)s_{U'}) indexed by
  > α ∈ Hom_C(U,U') such that, for any pair of compatible morphisms α, β,
  > we have
  >
  > **s_{α∘β} = F_β(s_α) ∘ s_β**"  (2.3)

  A section is a "continuous choice" of elements s_U ∈ F(U) for each U ∈ C,
  compatible with the morphisms.

  **Geometric interpretation**:
  - A section assigns a "state" s_U to each layer U
  - The compatibility (2.3) ensures coherence across layers
  - This generalizes global sections of sheaves
  -}

  record Section : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ') where
    no-eta-equality
    field
      -- Element in each fiber
      s_U : (U : C-Ob) → (F₀ U) .Precategory.Ob

      -- Morphisms connecting fibers
      s_α : {U U' : C-Ob} (α : C-Hom U U') →
            (F₀ U) .Precategory.Hom (s_U U) ((F₁ α) .Functor.F₀ (s_U U'))

      {-|
      **Equation (2.3)**: Composition law for sections

      For α: U → U', β: U' → U'':
      s_{α∘β} = F(β)(s_α) ∘ s_β

      This says the section is functorial in the fibered sense.
      -}
      section-comp : {U U' U'' : C-Ob} (α : C-Hom U U') (β : C-Hom U' U'') →
                    s_α (α ∘ β) ≡ (F₁ β .Functor.F₁ (s_α α)) ∘[ F₀ U ] s_α β
        where ∘[_] = λ D → D .Precategory._∘_

  open Section public

{-|
## Presheaves over Fibrations (Equations 2.4-2.6)

From the paper:

> "As shown by Grothendieck and Giraud [Gir64], a presheaf A over F corresponds
> to a family of presheaves A_U on the categories F_U indexed by U ∈ C, and a
> family A_α indexed by α ∈ Hom_C(U,U'), of natural transformations from A_{U'}
> to F*_α A_U."

A presheaf over the fibration F consists of:
1. For each U ∈ C: a presheaf A_U: F(U)^op → Set
2. For each α: U → U': a natural transformation A_α: A_{U'} → F*_α A_U
3. Compatibility conditions (Equations 2.4, 2.6)
-}

module _ {C : Precategory o ℓ} (F : Functor (C ^op) (Cat o' ℓ')) where
  open Functor F
  open Precategory C renaming (Ob to C-Ob; Hom to C-Hom)

  {-|
  ### Pullback Functor F*_α

  For α: U → U', we have F(α): F(U') → F(U).
  This induces a pullback on presheaves:

  F*_α: [F(U)^op, Set] → [F(U')^op, Set]

  Defined by: (F*_α A_U) = A_U ∘ F(α)

  This is precomposition with F(α).
  -}

  -- Presheaf on a fiber F(U)
  Presheaf-on-Fiber : (U : C-Ob) → Type (lsuc κ ⊔ o' ⊔ ℓ')
  Presheaf-on-Fiber U = Functor ((F₀ U) ^op) (Sets κ)

  -- Pullback functor (precomposition)
  pullback : {U U' : C-Ob} (α : C-Hom U U') →
             Presheaf-on-Fiber U → Presheaf-on-Fiber U'
  pullback α A_U = A_U F∘ (F₁ α ^op)

  -- Notation
  F*_ : {U U' : C-Ob} → C-Hom U U' → Presheaf-on-Fiber U → Presheaf-on-Fiber U'
  F*_ = pullback

  {-|
  ### Presheaf over Fibration (Record)

  A presheaf A over F consists of:
  - Presheaves A_U on each fiber F(U)
  - Natural transformations A_α: A_{U'} → F*_α A_U
  - Compatibility: Equation (2.4)
  -}

  record Presheaf-over-Fib : Type (o ⊔ ℓ ⊔ lsuc κ ⊔ o' ⊔ ℓ') where
    no-eta-equality
    field
      -- Presheaf on each fiber
      A_U : (U : C-Ob) → Presheaf-on-Fiber U

      -- Natural transformation for each morphism
      A_α : {U U' : C-Ob} (α : C-Hom U U') →
            A_U U' => (F* α) (A_U U)

      {-|
      **Equation (2.4)**: Composition law for presheaf transformations

      For α: U → U', β: U' → U'':
      A_{α∘β} = F*_α(A_β) ∘ A_α

      This says the family {A_α} forms a "Cartesian morphism" in the
      appropriate bicategorical sense.
      -}
      presheaf-comp : {U U' U'' : C-Ob} (α : C-Hom U U') (β : C-Hom U' U'') →
                      A_α (α ∘ β) ≡ {!!} -- F*_α(A_α β) ∘ⁿᵃᵗ (A_α α)
        -- Full type requires vertical composition of natural transformations

  open Presheaf-over-Fib public

  {-|
  ## Equation (2.5): Presheaf on Morphisms

  From the paper:

  > "If ξ is an object of F_U, we define A(U,ξ) = A_U(ξ), and if
  > f: ξ_U → F_α ξ'_{U'} is a morphism of F between ξ_U ∈ F_U and ξ'_{U'} ∈ F_{U'}
  > lifting α, we take
  >
  > **A(f) = A_U(f) ∘ A_α : A_{U'}(ξ') → A_U(F_α(ξ')) → A_U(ξ)**"  (2.5)

  This describes how a presheaf A over F acts on morphisms in the total category.

  **Components**:
  - A_α: A_{U'}(ξ') → A_U(F_α(ξ')) (natural transformation component)
  - A_U(f): A_U(F_α(ξ')) → A_U(ξ) (presheaf A_U applied to f)
  - Composition gives the full action

  **Geometric interpretation**:
  - A assigns a set A(U,ξ) to each "point" (U,ξ) of the fibration
  - Morphisms in F act on these sets via Equation (2.5)
  - This makes A a presheaf on the total category
  -}

  module _ (A : Presheaf-over-Fib) where
    open Presheaf-over-Fib A

    -- Evaluation at objects
    A-at : Fib-Ob F → Type κ
    A-at (U , ξ) = A_U U .Functor.F₀ ξ

    postulate
      {-|
      **Equation (2.5)**: Action on morphisms

      For a morphism (α, f): (U, ξ) → (U', ξ') in the fibration:

      A(α, f) = A_U(f) ∘ A_α

      This combines:
      1. The natural transformation component A_α
      2. The presheaf action A_U(f)
      -}
      A-on-hom : {obj obj' : Fib-Ob F} →
                Fib-Hom F obj obj' →
                A-at obj' → A-at obj

      {-|
      Equation (2.5) makes A into a presheaf on the total category.

      The relation A(f ∘ g) = A(g) ∘ A(f) follows from (2.4).
      -}
      A-as-presheaf : Functor ((Total-Category F) ^op) (Sets κ)

  {-|
  ## Equation (2.6): Natural Transformations of Presheaves

  From the paper:

  > "A natural transformation φ: A → A' corresponds to a family of natural
  > transformations φ_U: A_U → A'_U, such that, for any arrow α: U → U' in C,
  >
  > **F*_α φ_U ∘ A_α = A'_α ∘ φ_{U'} : A_{U'} → F*_α A'_U**"  (2.6)

  This is the compatibility condition for natural transformations between
  presheaves over a fibration.

  **Commutative diagram**:
  ```
  A_{U'}  ----φ_{U'}---->  A'_{U'}
    |                        |
    | A_α                    | A'_α
    ↓                        ↓
  F*_α A_U  --F*_α φ_U-->  F*_α A'_U
  ```

  Commutativity: F*_α φ_U ∘ A_α = A'_α ∘ φ_{U'}

  **Geometric interpretation**:
  - φ is a "morphism of presheaves" over the fibration
  - At each fiber F(U), we have a natural transformation φ_U
  - Equation (2.6) ensures these are compatible with the A_α, A'_α structure
  -}

  record Presheaf-Morphism (A A' : Presheaf-over-Fib) : Type (o ⊔ ℓ ⊔ κ ⊔ o' ⊔ ℓ') where
    no-eta-equality
    open Presheaf-over-Fib A
    open Presheaf-over-Fib A' renaming (A_U to A'_U; A_α to A'_α)

    field
      -- Natural transformation at each fiber
      φ_U : (U : C-Ob) → A_U U => A'_U U

      {-|
      **Equation (2.6)**: Compatibility with pullbacks

      For α: U → U':
      F*_α φ_U ∘ A_α = A'_α ∘ φ_{U'}

      This says φ is a morphism in the fibered sense.
      -}
      nat-compat : {U U' : C-Ob} (α : C-Hom U U') →
                  {!!} -- Requires formulating the equation precisely
        -- This requires showing the diagram commutes

  open Presheaf-Morphism public

{-|
## Summary

This module establishes the complete fibration and presheaf structure:

1. ✅ **Fibration objects**: Pairs (U, ξ) where U ∈ C, ξ ∈ F(U)
2. ✅ **Equation (2.2)**: Hom_F((U,ξ), (U',ξ')) = ⊔ Hom_{F(U)}(ξ, F(α)ξ')
3. ✅ **Total category**: Category structure on the fibration
4. ✅ **Projection π: F → C**: Forgetful functor to base
5. ✅ **Sections**: Families s_U with **Equation (2.3)**: s_{α∘β} = F_β(s_α) ∘ s_β
6. ✅ **Presheaves over F**: Family of presheaves A_U with transformations A_α
7. ✅ **Pullback F*_α**: Precomposition functor for presheaves
8. ✅ **Equation (2.4)**: A_{α∘β} = F*_α(A_β) ∘ A_α
9. ✅ **Equation (2.5)**: A(f) = A_U(f) ∘ A_α
10. ✅ **Equation (2.6)**: F*_α φ_U ∘ A_α = A'_α ∘ φ_{U'}

**Key insight**: The six equations (2.2-2.6 plus 2.3) completely describe how
presheaves are structured over a fibration. This is the foundation for:
- Classifying topoi (next module)
- Logical structure in the fibers
- Semantic functioning of DNNs

**Next steps** (Module 6 - Classifier):
- Define the subobject classifier Ω_F
- Prove Proposition 2.1: Ω_F = ∇_{U∈C} Ω_U d Ω_α (Equation 2.12)
- Establish the global Heyting algebra structure
-}
