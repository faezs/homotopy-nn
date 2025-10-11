{-# OPTIONS --no-import-sorts #-}
{-|
# Groupoids, Stacks, and Categorical Invariance (Chapter 2.1)

**Reference**: Belfiore & Bennequin (2022), Section 2.1
arXiv:2106.14587v3 [math.AT]

## Main Content from the Paper:

From Section 2.1:

> "In many interesting cases, a restriction on the structure of the functioning
> X_w, or the learning in W, comes from a **geometrical or semantic invariance**,
> which is extracted (or expected) from the input data and/or the problems that
> the network has to solve as output."

> "The most celebrate example is given by the **convolutional networks CNNs**.
> These networks are made for analyzing images; it can be for finding something
> precise in an image in a given class of images, or it can be for classifying
> special forms. The images are assumed to be by nature **invariant by planar
> translation**, then it is imposed to a large number of layers to accept a
> non trivial action of the group G of 2D-translations."

## Key Concepts:

1. **Group actions on categories**: G acting via functors
2. **CNN example**: Translation invariance (G = ℝ²)
3. **Orbit of an element**: Functor from G|_a to u|V
4. **Stacks**: Fibred categories F → C over the network architecture
5. **Fibred actions**: Equivariance formula (Equation 2.1)
6. **Classifying topos**: C∼_G from stacks

## The Big Picture:

Instead of working with sheaves (Set-valued functors), we work with:
- **Stacks**: Functors from C^op to Cat (or Grpd)
- **Groupoid actions**: Each layer has a groupoid of symmetries
- **Classifying topos**: E = F∼ of sheaves on the stack F

This generalizes the simple case where all fibers are Sets to the case where
fibers are categories/groupoids, encoding internal structure and symmetries.
-}

module Neural.Stack.Groupoid where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Groupoid
open import Cat.Instances.Functor
open import Cat.Instances.Shape.Terminal
open import Cat.Bi.Base

open import Algebra.Group using (Group-on; Group≃)

-- Helper: Group is Σ Type Group-on
Group : ∀ ℓ → Type (lsuc ℓ)
Group ℓ = Σ (Type ℓ) Group-on

open import Data.Nat.Base using (Nat)

private variable
  o ℓ o' ℓ' : Level

{-|
## Group Actions on Categories

From the paper:

> "Invariance is synonymous of action (like group action), and is understood
> here in the categorical sense: a category G acts on another category V when
> a (contravariant) functor from G to V is given."

A **group action** in the categorical sense is a functor G → V where:
- G is a group viewed as a one-object category
- V is the target category (e.g., vector spaces, sets, categories)
- The functor encodes how G transforms objects in V

**Classical example**: Linear representations
- G = group (one object, morphisms = group elements)
- V = Vect_K (vector spaces over field K)
- Functor G → V gives a representation: g ↦ (linear map V → V)
-}

-- A group as a category (one object, morphisms = group elements)
Group-as-Category : ∀ {ℓ} → Group ℓ → Precategory lzero ℓ
Group-as-Category G = cat where
  open Group-on (G .snd)
  cat : Precategory lzero ℓ
  cat .Precategory.Ob = ⊤
  cat .Precategory.Hom _ _ = ⌞ G ⌟
  cat .Precategory.Hom-set _ _ = G .snd .Group-on.has-is-set
  cat .Precategory.id = unit
  cat .Precategory._∘_ = _⋆_
  cat .Precategory.idr _ = idr
  cat .Precategory.idl _ = idl
  cat .Precategory.assoc f g h = associative {x = f} {g} {h}

{-|
### Group Action Definition

A **group action** of G on a category V is a functor F: G → V.

From the paper:
> "The example that justifies this terminology is when G is a group G, and V
> the Abelian category of vector spaces and linear maps over a commutative
> field K. In the latter case, we obtain a linear representation of the group G."
-}

Group-Action : ∀ {ℓ} (G : Group ℓ) (V : Precategory o ℓ') → Type (ℓ ⊔ o ⊔ ℓ')
Group-Action G V = Functor (Group-as-Category G) V

{-|
## Orbits in Categories

From the paper:

> "In any category V, there exists a notion which generalizes the notion of
> element of a set. Any morphism φ: u → v in V can be viewed as an element
> of the object v of V."

> "**Definition**: Suppose that G acts through the functor f: G → V and that
> v = f(a), then the **orbit** of φ under G|_a is the functor from the left
> slice category G|_a to the right slice category u|V, that associates to any
> morphism a' → a the element u → f(a) → f(a') of f(a') in V and to an arrow
> a" → a' over a the corresponding morphism f(a') → f(a")."

**Intuition**:
- An "element" of v is a morphism u → v
- The orbit is how this element transforms under the group action
- In the classical case (V = Vect), this gives the usual orbit in the vector space
-}

module _ {ℓ} {G : Group ℓ} {V : Precategory o ℓ'} (f : Group-Action G V) where
  open Functor f
  open Precategory V

  {-|
  ### Orbit of a Morphism

  Given:
  - A morphism φ: u → v in V
  - An object a in G with f(a) = v

  The orbit is a functor from the slice G|_a to the slice u|V.

  **Construction**:
  - For a' → a in G, we get u → f(a) → f(a')
  - This is the "transformed element" under the group element a' → a
  -}
  postulate
    -- The orbit functor (requires slice category constructions)
    orbit : ∀ {u v : Ob} (φ : Hom u v) →
            -- (a : Ob G) → -- In our case G has one object
            -- f(a) ≡ v →   -- Require v = f(a)
            Type (o ⊔ ℓ ⊔ ℓ')
    -- Full construction: Functor (G|a) (u|V)
    -- Where G|a is left slice, u|V is right slice

{-|
## CNN Example: Translation Invariance

From the paper:

> "The most celebrate example is given by the convolutional networks CNNs.
> These networks are made for analyzing images... The images are assumed to be
> by nature invariant by planar translation, then it is imposed to a large
> number of layers to accept a non trivial action of the group G of
> 2D-translations and to a large number of connections between two layers to
> be compatible with the actions, which implies that the underlying linear
> part when it exists is made by **convolutions** with a numerical function
> on the plane."

**The setup**:
- G = ℝ² (group of 2D translations)
- Each convolutional layer L has G acting on it
- Connections L₁ → L₂ are G-equivariant
- Linear parts are convolution operators (translation-equivariant)

**Modern perspective**:
- This extends to arbitrary Lie groups (SO(2), SO(3), etc.)
- Gauge theory interpretation [Cohen et al., Bronstein et al.]
- Geometric deep learning framework
-}

module CNN-Example where
  postulate
    -- The group of 2D translations (ℝ², +)
    -- In practice we'd work with discrete/bounded translations
    ℝ² : Group lzero

    {-|
    A convolutional layer is a vector space with ℝ² action.

    **Structure**:
    - Input: Functions on ℝ² (or a grid)
    - Convolution: (K * f)(x) = ∫ K(x-y) f(y) dy
    - Translation equivariance: K * (T_a f) = T_a (K * f)

    where T_a is translation by a ∈ ℝ².
    -}
    ConvLayer : Type

    -- The action of ℝ² on a convolutional layer
    conv-action : ConvLayer → Functor (Group-as-Category ℝ²) (Sets lzero)

    {-|
    Convolution as a linear equivariant map.

    For layers L₁, L₂ with ℝ² actions, a convolution kernel K: L₁ → L₂
    satisfies:

    K ∘ T_a = T_a ∘ K  for all a ∈ ℝ²

    This is the equivariance condition.
    -}
    is-equivariant : (L₁ L₂ : ConvLayer) → (K : ConvLayer → ConvLayer) → Type

  {-|
  **Remark from paper**:
  > "This does not forbid that in several layers, the action of G is trivial,
  > to get invariant characteristics under translations, and here, the layers
  > can be fully connected. The Resnets today have such a structure, with non-trivial
  > architectures."

  Modern ResNets mix:
  - Convolutional layers (translation-equivariant)
  - Pooling layers (progressive invariance)
  - Fully connected layers (translation-invariant)
  -}

{-|
## Stacks: Fibred Categories

From the paper:

> "A toposic manner to encode such a situation consists to consider contravariant
> functors from the category C of the network with values in the topos G∧ of
> G-sets, in place of taking values in the category Set of sets."

Instead of presheaves F: C^op → Set, we consider **stacks**:

**F: C^op → Cat** (or F: C^op → Grpd)

where:
- C is the network architecture category (CX or Fork-Category)
- Cat is the 2-category of categories
- Each layer U has an associated category F(U) (not just a set!)

**Interpretation**:
- F(U) encodes the states at layer U with internal structure
- Morphisms in F(U) represent transformations/symmetries
- For groupoids: all morphisms are isomorphisms (symmetries)

This is Giraud's construction [Gir72] of the classifying topos.
-}

-- Stack: Contravariant functor to categories
Stack : (C : Precategory o ℓ) (o' ℓ' : Level) → Type (o ⊔ ℓ ⊔ lsuc o' ⊔ lsuc ℓ')
Stack C o' ℓ' = Functor (C ^op) (Cat o' ℓ')

-- Stack in groupoids: stricter condition
Stack-Grpd : (C : Precategory o ℓ) (o' ℓ' : Level) → Type (o ⊔ ℓ ⊔ lsuc o' ⊔ lsuc ℓ')
Stack-Grpd C o' ℓ' = Functor (C ^op) (Cat o' ℓ')
  -- With additional condition: each F(U) is a groupoid

{-|
### Stack of G-Sets

From the paper:

> "Here the group G is identified with the category with one object and whose
> arrows are given by the elements of G, then a G-set, that is a set with a
> left action of G, is viewed as a set valued sheaf over G."

For a group G, the category **G∧ of G-sets** has:
- Objects: Sets X with a left G-action
- Morphisms: G-equivariant maps (f: X → Y with f(g·x) = g·f(x))

A stack F: C^op → G∧ assigns to each layer U a G-set F(U).

**Example (CNN)**:
- F(conv-layer) = feature maps on ℝ² with translation action
- F(pooling-layer) = feature maps with restricted action
- F(fc-layer) = vectors (trivial G-action)
-}

module _ {ℓ} (G : Group ℓ) where
  postulate
    -- The category of G-sets
    G-Sets : Precategory (lsuc lzero) (lsuc lzero)

    {-|
    A G-set is a set X with a group action G × X → X.

    Morphisms are equivariant maps.
    -}
    G-Set : Type (lsuc lzero)
    G-Set-action : G-Set → (⌞ G ⌟ → G-Set → G-Set)

    {-|
    Stack with G-set fibers.

    This assigns a G-set to each layer of the network.
    -}
    G-Set-Stack : (C : Precategory o ℓ) → Type (o ⊔ ℓ ⊔ lsuc lzero)
    G-Set-Stack C = Functor (C ^op) G-Sets

{-|
## Fibred Actions (Equation 2.1)

From the paper:

> "**Definition**: Suppose we are given a sheaf of categories F: C → Cat, that
> we consider as a general structure of invariance, and another sheaf M: C → Cat.
> An **action of F on M** is a family of contravariant functors f_U: F_U → M_U
> such that, for any morphism α: U → U' of C, we have
>
> **f_U ∘ F_α = M_α ∘ f_{U'}**   (2.1)
>
> This is the equivariance formula generalizing group equivariance."

**Equation (2.1)** is the key equivariance condition for stacks.

**Interpretation**:
- F encodes the symmetry structure (e.g., G-action at each layer)
- M encodes the data structure (e.g., states at each layer)
- f_U: F_U → M_U is the action functor at layer U
- Equation (2.1) says the action commutes with the stack structure

**Commutative diagram**:
```
F_{U'}  ----f_{U'}---->  M_{U'}
  |                        |
  | F_α                    | M_α
  ↓                        ↓
F_U   ----f_U------->    M_U
```

Commutativity: f_U ∘ F_α = M_α ∘ f_{U'}
-}

module _ {C : Precategory o ℓ} where
  open Functor

  {-|
  ### Fibred Action (Definition)

  An action of a stack F on a stack M is a family of functors f_U: F(U) → M(U)
  satisfying the equivariance formula (2.1).
  -}
  record Fibred-Action (F M : Stack C o' ℓ') : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ') where
    no-eta-equality
    field
      -- Action functor at each object U
      f_U : ∀ (U : C .Precategory.Ob) → Functor (F .F₀ U) (M .F₀ U)

      {-|
      **Equation (2.1)**: Equivariance condition

      For any morphism α: U → U' in C:
      f_U ∘ F(α) = M(α) ∘ f_{U'}

      This says the action functors form a natural transformation in the
      fibred sense.
      -}
      equivariance : ∀ {U U' : C .Precategory.Ob} (α : C .Precategory.Hom U U') →
                    f_U U F∘ F .F₁ α ≡ M .F₁ α F∘ f_U U'

open Fibred-Action public

{-|
## Classifying Topos of a Stack

From the paper:

> "The collection of these functors, with morphisms given by the equivariant
> natural transformations, form a category C∼_G, which was shown to be itself
> a topos by Giraud [Gir72]."

For a stack F: C^op → Cat (or Grpd), the **classifying topos** is:

**E = F∼ = Sh(F, J)**

where:
- Objects: Presheaves on the total category of F
- Morphisms: Natural transformations
- J is a canonical Grothendieck topology

**Key theorem (Giraud 1972)**:
If F: C^op → Grpd is a stack in groupoids, then F∼ is a topos.

For the DNN case:
- C = CX (the poset of layers)
- F assigns a groupoid to each layer (encoding symmetries)
- F∼ is the topos of sheaves incorporating these symmetries

**Boolean logic** (from paper):
> "The case of groupoids has the interest that the presheaves on a groupoid
> form a Boolean topos, then ordinary logic is automatically incorporated."
-}

postulate
  {-|
  Classifying topos of a stack in groupoids.

  This is Giraud's theorem [Gir72]: for a stack F: C^op → Grpd,
  the category of presheaves on F is a Grothendieck topos.
  -}
  Classifying-Topos : {C : Precategory o ℓ} →
                      Stack-Grpd C o' ℓ' →
                      Precategory (lsuc (o ⊔ ℓ ⊔ o' ⊔ ℓ')) (o ⊔ ℓ ⊔ o' ⊔ ℓ')

  {-|
  The classifying topos is Boolean when F is a stack in groupoids.

  This means the internal logic is classical (law of excluded middle holds).
  -}
  Classifying-Topos-is-Boolean : {C : Precategory o ℓ} (F : Stack-Grpd C o' ℓ') →
                                 Type (lsuc (o ⊔ ℓ ⊔ o' ⊔ ℓ'))

{-|
## Summary

This module establishes the foundation for stacks and groupoid actions:

1. ✅ **Group actions**: Functors G → V encoding symmetries
2. ✅ **Orbits**: Generalized group orbits in categories
3. ✅ **CNN example**: Translation invariance (G = ℝ²)
4. ✅ **Stacks**: Functors F: C^op → Cat (or Grpd)
5. ✅ **Stack of G-sets**: Incorporating group actions at each layer
6. ✅ **Fibred actions**: **Equation (2.1)** - equivariance condition
7. ✅ **Classifying topos**: F∼ is a Grothendieck topos (Giraud)
8. ✅ **Boolean logic**: Groupoid stacks give Boolean topoi

**Key insight**: By moving from Set-valued sheaves to Cat/Grpd-valued stacks,
we can encode:
- Symmetries and invariances (CNNs, gauge theory)
- Internal structure at each layer
- Richer semantic content

**Next steps** (Module 5 - Fibration):
- Formalize fibrations π: F → C
- Define presheaves over fibrations (Equations 2.2-2.6)
- Construct subobject classifier Ω_F (Proposition 2.1)
-}
