{-# OPTIONS --no-import-sorts #-}

{-|
# ℤ² Translation Group (Section 2.1)

The group of 2D translations for convolutional neural networks.

**Reference**: Belfiore & Bennequin (2022), Section 2.1

> "The images are assumed to be by nature invariant by planar translation,
> then it is imposed to a large number of layers to accept a non trivial
> action of the group G of 2D-translations."

## Structure

ℤ² = Int × Int with componentwise addition forms an abelian group:
- Identity: (0, 0)
- Inverse: (-(a,b)) = (-a, -b)
- Operation: (a,b) + (c,d) = (a+c, b+d)
- Associative, commutative

This is the **translation group** that acts on spatial feature maps.
-}

module examples.CNN.TranslationGroup where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Algebra.Group
open import Algebra.Group.Ab
open import Algebra.Group.Instances.Integers

open import Data.Int.Base
open import Data.Int.Properties

private variable
  ℓ : Level

--------------------------------------------------------------------------------
-- ℤ² as a Type
--------------------------------------------------------------------------------

{-|
2D translation vectors as pairs of integers.

Interpretation:
- (Δx, Δy) represents a translation by Δx horizontally and Δy vertically
- (0, 0) is the identity (no translation)
- (-Δx, -Δy) undoes the translation (Δx, Δy)
-}

ℤ² : Type
ℤ² = Int × Int

{-|
ℤ² is a set (h-level 2) because Int is discrete.
-}
ℤ²-is-set : is-set ℤ²
ℤ²-is-set = ×-is-hlevel 2 (Discrete→is-set Discrete-Int)
                            (Discrete→is-set Discrete-Int)

--------------------------------------------------------------------------------
-- Group Structure on ℤ²
--------------------------------------------------------------------------------

{-|
Componentwise addition: the group operation.

(a, b) +² (c, d) = (a + c, b + d)
-}
_+²_ : ℤ² → ℤ² → ℤ²
(a , b) +² (c , d) = (a +ℤ c , b +ℤ d)

infixl 30 _+²_

{-|
Identity element: (0, 0)
-}
ℤ²-unit : ℤ²
ℤ²-unit = (posz , posz)

{-|
Inverse: negate both components.

-(a, b) = (-a, -b)
-}
ℤ²-inv : ℤ² → ℤ²
ℤ²-inv (a , b) = (negℤ a , negℤ b)

{-|
Group laws for ℤ².
-}

ℤ²-idl : (x : ℤ²) → ℤ²-unit +² x ≡ x
ℤ²-idl (a , b) = ap₂ _,_ (+ℤ-zerol a) (+ℤ-zerol b)

ℤ²-idr : (x : ℤ²) → x +² ℤ²-unit ≡ x
ℤ²-idr (a , b) = ap₂ _,_ (+ℤ-zeror a) (+ℤ-zeror b)

ℤ²-assoc : (x y z : ℤ²) → (x +² y) +² z ≡ x +² (y +² z)
ℤ²-assoc (a , b) (c , d) (e , f) =
  ap₂ _,_ (+ℤ-assoc a c e) (+ℤ-assoc b d f)

ℤ²-invl : (x : ℤ²) → ℤ²-inv x +² x ≡ ℤ²-unit
ℤ²-invl (a , b) = ap₂ _,_ (+ℤ-invl a) (+ℤ-invl b)

ℤ²-invr : (x : ℤ²) → x +² ℤ²-inv x ≡ ℤ²-unit
ℤ²-invr (a , b) = ap₂ _,_ (+ℤ-invr a) (+ℤ-invr b)

{-|
ℤ² is an abelian group under componentwise addition.
-}
ℤ²-comm : (x y : ℤ²) → x +² y ≡ y +² x
ℤ²-comm (a , b) (c , d) = ap₂ _,_ (+ℤ-commutative a c) (+ℤ-commutative b d)

--------------------------------------------------------------------------------
-- ℤ² as a Group
--------------------------------------------------------------------------------

{-|
ℤ² with componentwise addition forms a group.

This is the **translation group** for 2D images.
-}
ℤ²-group-on : Group-on ℤ²
ℤ²-group-on = to-group-on mk where
  open make-group
  mk : make-group ℤ²
  mk .group-is-set = ℤ²-is-set
  mk .unit = ℤ²-unit
  mk .mul = _+²_
  mk .inv = ℤ²-inv
  mk .assoc = ℤ²-assoc
  mk .invl = ℤ²-invl
  mk .idl = ℤ²-idl

ℤ² : Group lzero
ℤ² = ℤ² , ℤ²-group-on

{-|
ℤ² is actually abelian.
-}
ℤ²-is-abelian : is-abelian-group ℤ²-group-on
ℤ²-is-abelian .is-abelian-group.has-is-group = ℤ²-group-on .Group-on.has-is-group
ℤ²-is-abelian .is-abelian-group.commutes = ℤ²-comm

ℤ²-ab : Abelian-group lzero
ℤ²-ab = ℤ² , ℤ²-is-abelian

--------------------------------------------------------------------------------
-- Projection Homomorphisms
--------------------------------------------------------------------------------

{-|
First projection π₁: ℤ² → ℤ is a group homomorphism.
-}
π₁-hom : Groups.Hom ℤ² ℤ
π₁-hom .fst (a , _) = a
π₁-hom .snd .is-group-hom.pres-⋆ (a , _) (c , _) = refl

{-|
Second projection π₂: ℤ² → ℤ is a group homomorphism.
-}
π₂-hom : Groups.Hom ℤ² ℤ
π₂-hom .fst (_ , b) = b
π₂-hom .snd .is-group-hom.pres-⋆ (_ , b) (_ , d) = refl

--------------------------------------------------------------------------------
-- Standard Basis Translations
--------------------------------------------------------------------------------

{-|
Unit translation in x-direction: (1, 0)
-}
e₁ : ℤ²
e₁ = (possuc 0 , posz)

{-|
Unit translation in y-direction: (0, 1)
-}
e₂ : ℤ²
e₂ = (posz , possuc 0)

{-|
Any translation can be written as a linear combination of e₁ and e₂.

For (a, b) ∈ ℤ²:
  (a, b) = a·e₁ + b·e₂
-}
ℤ²-basis : (v : ℤ²) → v ≡ {!!}
  -- Would need scalar multiplication Int → ℤ² → ℤ² to state properly
ℤ²-basis = {!!}

--------------------------------------------------------------------------------
-- Connection to CNNs
--------------------------------------------------------------------------------

{-|
## Interpretation for Convolutional Networks

The group ℤ² acts on images and feature maps through translation:

**Discrete Translation**:
For an image I: Grid → ℝ and translation (Δx, Δy) ∈ ℤ²:
  (T_{(Δx,Δy)} I)(x, y) = I(x - Δx, y - Δy)

This shifts the image content by (Δx, Δy) pixels.

**Weight Sharing** (Section 2.1):
Convolutional layers have the property that the same kernel K is applied
at all spatial locations. This means:

  Conv(T_a(I)) = T_a(Conv(I))  for all a ∈ ℤ²

This is **translation equivariance** - the key property of CNNs!

**Group Action on Layers**:
For a convolutional layer L with feature maps F: Grid → ℝ^c:
- ℤ² acts on F by translating the spatial coordinates
- The channels remain unchanged
- Convolution preserves this action (equivariance)

**Next Steps**:
- Define Grid type with bounded coordinates (Fin h × Fin w)
- Define translation action ℤ² × Grid → Grid (with wrapping/padding)
- Define FeatureMap = Grid → ℝ^c
- Define convolution operator with equivariance proof
-}

{-|
Translate a grid position by a vector.

This will be refined when we define the Grid type with proper bounds.
-}
postulate
  translate-position : ℤ² → (Int × Int) → (Int × Int)
  -- translate-position (Δx, Δy) (x, y) = (x + Δx, y + Δy)
  -- But with proper modular arithmetic for bounded grids

{-|
Translation is a group action:
1. Identity translation does nothing
2. Composing translations = adding the vectors
-}
postulate
  translate-id : ∀ (pos : Int × Int) → translate-position ℤ²-unit pos ≡ pos
  translate-compose : ∀ (a b : ℤ²) (pos : Int × Int) →
    translate-position (a +² b) pos ≡
    translate-position a (translate-position b pos)
