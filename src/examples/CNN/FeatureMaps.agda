{-# OPTIONS --no-import-sorts #-}

{-|
# Feature Maps as ℤ²-Sets

Feature maps for convolutional neural networks with translation group action.

A **feature map** assigns a feature vector to each spatial position in a grid.
The translation group ℤ² acts on feature maps by shifting spatial coordinates.

This realizes the **stack of ℤ²-sets** from Section 2.1 of the paper.

## Structure

FeatureMap c h w = Grid h w → ℝ^c

Where:
- c = number of channels (features per position)
- h × w = spatial grid dimensions
- ℤ² acts by: (T_a F)(x) = F(x - a)  (translation in spatial domain)
-}

module examples.CNN.FeatureMaps where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin)

open import examples.CNN.TranslationGroup
open import examples.CNN.SpatialGrid

private variable
  ℓ : Level
  c h w : Nat

--------------------------------------------------------------------------------
-- Feature Vectors
--------------------------------------------------------------------------------

{-|
Real numbers (postulated for now).

In a full implementation, could use:
- Agda's Float primitives
- Exact real arithmetic
- Or stay abstract for theoretical purposes
-}
postulate
  ℝ : Type
  ℝ-is-set : is-set ℝ

  -- Real number operations (for later use)
  _+ᵣ_ : ℝ → ℝ → ℝ
  _*ᵣ_ : ℝ → ℝ → ℝ
  0ᵣ : ℝ
  1ᵣ : ℝ

{-|
c-dimensional feature vector.

A feature vector assigns a real value to each of c channels.

Examples:
- 3 channels: RGB color (R, G, B)
- 64 channels: Learned features in a conv layer
- 512 channels: Deep layer representations
-}
FeatureVector : Nat → Type
FeatureVector c = Fin c → ℝ

{-|
Feature vectors form a set.
-}
FeatureVector-is-set : is-set (FeatureVector c)
FeatureVector-is-set = fun-is-hlevel 2 ℝ-is-set

{-|
Zero feature vector (all components zero).
-}
zero-vec : FeatureVector c
zero-vec _ = 0ᵣ

--------------------------------------------------------------------------------
-- Feature Maps
--------------------------------------------------------------------------------

{-|
A feature map assigns a feature vector to each grid position.

FeatureMap c h w = Grid h w → ℝ^c

**Interpretation**:
- Spatial dimension: h × w grid
- Channel dimension: c features per position
- Total size: c × h × w parameters

**Examples**:
- Input image (MNIST): FeatureMap 1 28 28  (grayscale)
- Input image (CIFAR): FeatureMap 3 32 32  (RGB)
- Conv layer output: FeatureMap 64 28 28  (64 learned features)
-}
FeatureMap : Nat → Nat → Nat → Type
FeatureMap c h w = Grid h w → FeatureVector c

{-|
Feature maps form a set.
-}
FeatureMap-is-set : is-set (FeatureMap c h w)
FeatureMap-is-set = fun-is-hlevel 2 FeatureVector-is-set

{-|
Zero feature map (all zeros everywhere).
-}
zero-map : FeatureMap c h w
zero-map _ = zero-vec

--------------------------------------------------------------------------------
-- ℤ²-Action on Feature Maps
--------------------------------------------------------------------------------

{-|
Translation acts on feature maps by shifting spatial coordinates.

For a translation a ∈ ℤ² and feature map F:
  (T_a F)(x) = F(translate (-a) x)

The **negative sign** is crucial: translating the *image* to the right
means we sample from positions *to the left* in the original image.

**Example** (1D for simplicity):
- Original: F = [a, b, c, d, e]
- Shift right by 2: T_{+2} F = [?, ?, a, b, c]  (shifted right, sampled left)
- This is F(x - 2), not F(x + 2)

**Note**: Boundary handling via translate-wrap (toroidal) or translate-pad (zero).
-}
translate-feature-map : {h w : Nat}
                      → {h≠0 : h ≡ suc zero ∨ Σ Nat (λ k → h ≡ suc (suc k))}
                      → {w≠0 : w ≡ suc zero ∨ Σ Nat (λ k → w ≡ suc (suc k))}
                      → ℤ² → FeatureMap c h w → FeatureMap c h w
translate-feature-map {h≠0 = h≠0} {w≠0 = w≠0} a F =
  λ x → F (translate-wrap {h≠0 = h≠0} {w≠0 = w≠0} (ℤ²-inv a) x)

{-|
Translation of feature maps is a group action.

**Identity**: Translating by (0,0) does nothing.
-}
postulate
  translate-feature-id : {h w : Nat}
                       → {h≠0 : h ≡ suc zero ∨ Σ Nat (λ k → h ≡ suc (suc k))}
                       → {w≠0 : w ≡ suc zero ∨ Σ Nat (λ k → w ≡ suc (suc k))}
                       → (F : FeatureMap c h w)
                       → translate-feature-map {h≠0 = h≠0} {w≠0 = w≠0} ℤ²-unit F ≡ F

{-|
**Composition**: Translating by a then b = translating by (a + b).

T_a (T_b F) = T_{a+b} F
-}
postulate
  translate-feature-compose : {h w : Nat}
                            → {h≠0 : h ≡ suc zero ∨ Σ Nat (λ k → h ≡ suc (suc k))}
                            → {w≠0 : w ≡ suc zero ∨ Σ Nat (λ k → w ≡ suc (suc k))}
                            → (a b : ℤ²) (F : FeatureMap c h w)
                            → translate-feature-map {h≠0 = h≠0} {w≠0 = w≠0} (a +² b) F ≡
                              translate-feature-map a (translate-feature-map b F)

--------------------------------------------------------------------------------
-- ℤ²-Sets (Categorical Perspective)
--------------------------------------------------------------------------------

{-|
A ℤ²-set is a set X with a group action ℤ² × X → X.

For feature maps:
- X = FeatureMap c h w
- Action = translate-feature-map
- Satisfies identity and composition laws

This makes FeatureMap into an object in the category of ℤ²-sets.
-}
record ℤ²-Set (ℓ : Level) : Type (lsuc ℓ) where
  no-eta-equality
  field
    -- Underlying set
    Carrier : Type ℓ
    Carrier-is-set : is-set Carrier

    -- ℤ² action
    action : ℤ² → Carrier → Carrier

    -- Action laws
    action-id : (x : Carrier) → action ℤ²-unit x ≡ x
    action-compose : (a b : ℤ²) (x : Carrier) →
                    action (a +² b) x ≡ action a (action b x)

open ℤ²-Set public

{-|
Feature maps form a ℤ²-set.

This is the key structure for CNNs: feature maps are not just sets,
but sets equipped with a translation symmetry.
-}
postulate
  FeatureMap-ℤ²-Set : {c h w : Nat}
                    → {h≠0 : h ≡ suc zero ∨ Σ Nat (λ k → h ≡ suc (suc k))}
                    → {w≠0 : w ≡ suc zero ∨ Σ Nat (λ k → w ≡ suc (suc k))}
                    → ℤ²-Set lzero
  -- Would implement with:
  -- Carrier = FeatureMap c h w
  -- action = translate-feature-map
  -- action-id = translate-feature-id
  -- action-compose = translate-feature-compose

--------------------------------------------------------------------------------
-- ℤ²-Equivariant Maps
--------------------------------------------------------------------------------

{-|
A map between ℤ²-sets that respects the group action.

f: X → Y is ℤ²-equivariant if:
  f (a · x) = a · f(x)  for all a ∈ ℤ², x ∈ X

**For CNNs**: Convolution is a ℤ²-equivariant map!
-}
record is-ℤ²-equivariant {ℓ ℓ'} (X : ℤ²-Set ℓ) (Y : ℤ²-Set ℓ')
                         (f : X .Carrier → Y .Carrier) : Type (ℓ ⊔ ℓ') where
  no-eta-equality
  field
    equivariance : (a : ℤ²) (x : X .Carrier) →
                  f (X .action a x) ≡ Y .action a (f x)

open is-ℤ²-equivariant public

{-|
ℤ²-equivariant map (morphism in category of ℤ²-sets).
-}
record ℤ²-Hom {ℓ ℓ'} (X : ℤ²-Set ℓ) (Y : ℤ²-Set ℓ') : Type (ℓ ⊔ ℓ') where
  no-eta-equality
  field
    map : X .Carrier → Y .Carrier
    preserves-action : is-ℤ²-equivariant X Y map

open ℤ²-Hom public

--------------------------------------------------------------------------------
-- Examples and Properties
--------------------------------------------------------------------------------

{-|
**Example**: Pointwise operations are ℤ²-equivariant.

Adding a constant to all pixels:
  f(x, y) = F(x, y) + c

is ℤ²-equivariant because translation doesn't affect the constant.
-}
postulate
  add-constant-equivariant : {h w : Nat}
                           → {h≠0 : h ≡ suc zero ∨ Σ Nat (λ k → h ≡ suc (suc k))}
                           → {w≠0 : w ≡ suc zero ∨ Σ Nat (λ k → w ≡ suc (suc k))}
                           → (c : ℝ)
                           → is-ℤ²-equivariant
                               (FeatureMap-ℤ²-Set {c = 1} {h} {w} {h≠0} {w≠0})
                               (FeatureMap-ℤ²-Set {c = 1} {h} {w} {h≠0} {w≠0})
                               (λ F x → λ ch → F x ch +ᵣ c)

{-|
**Example**: Pooling (max, average) is ℤ²-equivariant.

Max-pooling with a k×k window:
  (MaxPool F)(x, y) = max {F(x+i, y+j) | i,j ∈ [-k/2, k/2]}

is ℤ²-equivariant: translating the input shifts the windows consistently.
-}
postulate
  max-pool-equivariant : {h w h' w' : Nat}
                       → {!!}  -- Type signature would be complex

{-|
**Key Property**: Convolution is ℤ²-equivariant.

This will be proven in the ConvLayer module.

Conv_K (T_a F) = T_a (Conv_K F)

This is the **defining property** of convolutional layers!
-}
postulate
  convolution-equivariant : {!!}  -- Will be stated properly in ConvLayer module

--------------------------------------------------------------------------------
-- Connection to Topos Theory (Section 2.1)
--------------------------------------------------------------------------------

{-|
## Stack of ℤ²-Sets

From the paper:

> "A toposic manner to encode such a situation consists to consider contravariant
> functors from the category C of the network with values in the topos G∧ of
> G-sets, in place of taking values in the category Set of sets."

For CNNs:
- G = ℤ² (translation group)
- Each layer U has F(U) = ℤ²-set of feature maps
- Morphisms U → U' are ℤ²-equivariant maps (convolutions, pooling, etc.)

**Stack Structure**:
- F: C^op → ℤ²-Sets (contravariant functor)
- C = category of layers in the network
- F(U) = FeatureMap c_U h_U w_U (with ℤ²-action)

**Equation (2.1) - Fibred Action**:
For morphism α: U → U' and translation a ∈ ℤ²:

  f_U ∘ F_α(T_a x) = T_a ∘ M_α(f_U' x)

This says the convolution structure respects the translation symmetry.

**Next Steps**:
- Define convolution operator
- Prove convolution is ℤ²-equivariant
- Construct the stack F: C^op → ℤ²-Sets
- Verify Equation (2.1) for CNN layers
-}
