{-# OPTIONS --no-import-sorts #-}

{-|
# Spatial Grid for CNNs

Bounded 2D grids with translation group action.

Feature maps in CNNs are defined on bounded rectangular grids (e.g., 28×28 for MNIST).
The translation group ℤ² acts on these grids with appropriate boundary conditions.

## Structure

Grid h w = Fin h × Fin w  (h rows, w columns)

Translation action with two boundary handling strategies:
1. **Wrap-around** (toroidal topology)
2. **Zero-padding** (boundaries map to a special "outside" value)
-}

module examples.CNN.SpatialGrid where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Data.Nat.Base using (Nat; zero; suc; _+_; _*_)
open import Data.Fin.Base using (Fin; fzero; fsuc; lower; from-nat; Fin-elim)
open import Data.Int.Base
open import Data.Int.Properties

open import examples.CNN.TranslationGroup

private variable
  ℓ : Level
  h w : Nat

--------------------------------------------------------------------------------
-- Grid Type
--------------------------------------------------------------------------------

{-|
A bounded 2D grid with h rows and w columns.

Positions are indexed by (Fin h, Fin w):
- (0, 0) is top-left corner
- (h-1, w-1) is bottom-right corner

Examples:
- Grid 28 28 : MNIST images (28×28)
- Grid 32 32 : CIFAR-10 images (32×32)
- Grid 224 224 : ImageNet images (224×224)
-}
Grid : Nat → Nat → Type
Grid h w = Fin h × Fin w

{-|
Grids are sets (discrete).
-}
Grid-is-set : is-set (Grid h w)
Grid-is-set = ×-is-hlevel 2 (Discrete→is-set auto) (Discrete→is-set auto)

--------------------------------------------------------------------------------
-- Integer Modular Arithmetic
--------------------------------------------------------------------------------

{-|
Compute a mod n for integers, returning a Fin n.

For translation with wrap-around boundary conditions.
-}
postulate
  mod-int : Int → (n : Nat) → {n≠0 : n ≡ suc zero ∨ Σ Nat (λ k → n ≡ suc (suc k))} → Fin n

{-|
Addition with wrap-around (modular arithmetic).

(i + Δ) mod n
-}
add-mod : Fin n → Int → {n≠0 : n ≡ suc zero ∨ Σ Nat (λ k → n ≡ suc (suc k))} → Fin n
add-mod {n} i Δ {n≠0} =
  let i-int = pos (lower i)
      sum = i-int +ℤ Δ
  in mod-int sum n {n≠0}

--------------------------------------------------------------------------------
-- Translation Action (Wrap-around)
--------------------------------------------------------------------------------

{-|
Translate a grid position with wrap-around boundary conditions.

For a grid of size h×w and translation (Δx, Δy):
  translate (Δx, Δy) (x, y) = ((x + Δx) mod h, (y + Δy) mod w)

This gives the grid a **toroidal topology** - translating past the edge
wraps around to the opposite edge.

**Example** (5×5 grid):
  translate (2, 3) (4, 3) = ((4+2) mod 5, (3+3) mod 5) = (1, 1)
-}
translate-wrap : {h w : Nat}
               → {h≠0 : h ≡ suc zero ∨ Σ Nat (λ k → h ≡ suc (suc k))}
               → {w≠0 : w ≡ suc zero ∨ Σ Nat (λ k → w ≡ suc (suc k))}
               → ℤ² → Grid h w → Grid h w
translate-wrap {h} {w} {h≠0} {w≠0} (Δx , Δy) (x , y) =
  (add-mod x Δx {h≠0} , add-mod y Δy {w≠0})

{-|
Wrap-around translation is a group action (identity and composition).

TODO: Prove these properties once mod-int is implemented.
-}
postulate
  translate-wrap-id : {h w : Nat}
                    → {h≠0 : h ≡ suc zero ∨ Σ Nat (λ k → h ≡ suc (suc k))}
                    → {w≠0 : w ≡ suc zero ∨ Σ Nat (λ k → w ≡ suc (suc k))}
                    → (pos : Grid h w)
                    → translate-wrap {h≠0 = h≠0} {w≠0 = w≠0} ℤ²-unit pos ≡ pos

  translate-wrap-compose : {h w : Nat}
                         → {h≠0 : h ≡ suc zero ∨ Σ Nat (λ k → h ≡ suc (suc k))}
                         → {w≠0 : w ≡ suc zero ∨ Σ Nat (λ k → w ≡ suc (suc k))}
                         → (a b : ℤ²) (pos : Grid h w)
                         → translate-wrap {h≠0 = h≠0} {w≠0 = w≠0} (a +² b) pos ≡
                           translate-wrap a (translate-wrap b pos)

--------------------------------------------------------------------------------
-- Translation Action (Zero-padding)
--------------------------------------------------------------------------------

{-|
Extended grid: Grid + "outside" marker.

For zero-padding boundary conditions, we need to represent positions
outside the grid.
-}
data Grid⁺ (h w : Nat) : Type where
  inside  : Grid h w → Grid⁺ h w
  outside : Grid⁺ h w

{-|
Check if an integer coordinate is within bounds.
-}
in-bounds? : Int → (n : Nat) → Bool
in-bounds? (pos k) n = k <? n
  where
    _<?_ : Nat → Nat → Bool
    zero <? zero = false
    zero <? suc _ = true
    suc _ <? zero = false
    suc m <? suc n = m <? n
in-bounds? (negsuc _) _ = false

{-|
Convert integer to Fin if in bounds, otherwise Nothing.
-}
int-to-fin : Int → (n : Nat) → Maybe (Fin n)
int-to-fin i n with in-bounds? i n
... | true  = {!!}  -- Would need proper conversion
... | false = nothing

{-|
Translate with zero-padding boundary conditions.

Positions outside the grid boundaries map to "outside".
-}
translate-pad : ℤ² → Grid⁺ h w → Grid⁺ h w
translate-pad {h} {w} (Δx , Δy) (inside (x , y)) =
  let x' = pos (lower x) +ℤ Δx
      y' = pos (lower y) +ℤ Δy
  in case int-to-fin x' h , int-to-fin y' w of λ
    { (just x-new , just y-new) → inside (x-new , y-new)
    ; _ → outside
    }
translate-pad _ outside = outside

{-|
Zero-padding preserves "outside".
-}
translate-pad-outside : (a : ℤ²) → translate-pad a outside ≡ outside
translate-pad-outside _ = refl

--------------------------------------------------------------------------------
-- Neighborhoods and Receptive Fields
--------------------------------------------------------------------------------

{-|
A k×k neighborhood around a grid position.

This defines the **receptive field** for a convolutional kernel.

Example (3×3 neighborhood around (5, 5)):
  {(4,4), (4,5), (4,6),
   (5,4), (5,5), (5,6),
   (6,4), (6,5), (6,6)}
-}
neighborhood : (k : Nat) → Grid h w → List (Grid⁺ h w)
neighborhood {h} {w} k (x , y) = {!!}
  -- Generate all positions (x + i, y + j) for i,j ∈ {-k/2, ..., k/2}
  -- Use translate-pad to handle boundary conditions

{-|
3×3 receptive field (standard for most CNNs).
-}
receptive-3×3 : Grid h w → List (Grid⁺ h w)
receptive-3×3 = neighborhood 3

{-|
5×5 receptive field (larger kernels).
-}
receptive-5×5 : Grid h w → List (Grid⁺ h w)
receptive-5×5 = neighborhood 5

--------------------------------------------------------------------------------
-- Standard Grid Sizes
--------------------------------------------------------------------------------

{-|
MNIST image grid (28×28).
-}
Grid-MNIST : Type
Grid-MNIST = Grid 28 28

{-|
CIFAR-10 image grid (32×32).
-}
Grid-CIFAR : Type
Grid-CIFAR = Grid 32 32

{-|
Proof that MNIST grid size is non-zero.
-}
postulate
  MNIST-h-nonzero : 28 ≡ suc zero ∨ Σ Nat (λ k → 28 ≡ suc (suc k))
  MNIST-w-nonzero : 28 ≡ suc zero ∨ Σ Nat (λ k → 28 ≡ suc (suc k))

{-|
Translation on MNIST grid (wrap-around).
-}
translate-MNIST : ℤ² → Grid-MNIST → Grid-MNIST
translate-MNIST = translate-wrap {h≠0 = MNIST-h-nonzero} {w≠0 = MNIST-w-nonzero}

--------------------------------------------------------------------------------
-- Connection to Topos Theory
--------------------------------------------------------------------------------

{-|
## ℤ²-Action on Grids

The translation group ℤ² acts on Grid h w via translate-wrap (or translate-pad).

This makes Grid h w into a **ℤ²-set** in the categorical sense:
- Objects: Grid h w (set of positions)
- Action: ℤ² × Grid h w → Grid h w (translation)
- Identity: translate ℤ²-unit = id
- Composition: translate (a +² b) = translate a ∘ translate b

**For the topos construction** (Section 2.1):

1. Each convolutional layer has state space F(U) = Grid h w → ℝ^c
2. ℤ² acts on F(U) by translating the grid coordinates
3. Convolution is a ℤ²-equivariant map between layers

This realizes the **stack of ℤ²-sets** from the paper.

**Next Steps**:
- Define feature maps FeatureMap = Grid h w → ℝ^c
- Prove FeatureMap inherits ℤ²-action from Grid
- Define convolution with equivariance proof
-}

{-|
The grid with wrap-around forms a **torsor** for ℤ².

A torsor is a "group without identity" - it's a set with a free and
transitive group action.

**Free**: translate a pos₁ = translate a pos₂ → pos₁ = pos₂
**Transitive**: ∀ pos₁ pos₂, ∃! a such that translate a pos₁ = pos₂

This means the grid has no distinguished origin - all positions are
equivalent under translation.
-}
postulate
  grid-torsor-free : {h w : Nat}
                   → {h≠0 : h ≡ suc zero ∨ Σ Nat (λ k → h ≡ suc (suc k))}
                   → {w≠0 : w ≡ suc zero ∨ Σ Nat (λ k → w ≡ suc (suc k))}
                   → (a : ℤ²) (pos₁ pos₂ : Grid h w)
                   → translate-wrap {h≠0 = h≠0} {w≠0 = w≠0} a pos₁ ≡
                     translate-wrap a pos₂
                   → pos₁ ≡ pos₂

  grid-torsor-transitive : {h w : Nat}
                         → {h≠0 : h ≡ suc zero ∨ Σ Nat (λ k → h ≡ suc (suc k))}
                         → {w≠0 : w ≡ suc zero ∨ Σ Nat (λ k → w ≡ suc (suc k))}
                         → (pos₁ pos₂ : Grid h w)
                         → Σ ℤ² (λ a → translate-wrap {h≠0 = h≠0} {w≠0 = w≠0} a pos₁ ≡ pos₂)
