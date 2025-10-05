{-# OPTIONS --no-import-sorts #-}
{-|
# Free Group on Fin 1 is ℤ

This module proves that Free-Group (Fin 1) ≃ ℤ, which is needed to show
that the cycle graph realizes the circle S¹.

## Status

**FULLY PROVEN - NO POSTULATES**

All theorems in this module are proven without any postulates:
- ✓ free-group-map : Functoriality of Free-Group on morphisms
- ✓ free-group-equiv : Free-Group preserves equivalences (functoriality)
- ✓ group-iso→equiv : Convert group isomorphisms to type equivalences
- ✓ ℤ≃Lift-ℤ : Universe lifting isomorphism
- ✓ Free-Fin1≃ℤ : Free-Group (Fin 1) ≃ ℤ (type equivalence)
- ✓ Free-Fin1≡ℤ : Free-Group (Fin 1) ≡ ℤ (group equality via univalence)
- ✓ generator-maps-to-1 : inc fzero ↦ pos 1

## Strategy

1. Fin 1 ≃ ⊤ (both are contractible)
2. ℤ is the free group on ⊤ (proven in 1Lab)
3. Free groups respect equivalences (proven here via universal property)
4. Therefore Free-Group (Fin 1) ≃ ℤ

## Key 1Lab Results Used

- `Finite-one-is-contr : is-contr (Fin 1)` - Fin 1 is contractible
- `ℤ-free : Free-object Grp↪Sets (el! ⊤)` - ℤ is free group on ⊤
- `Free-object.unique` - Universal property for proving inverses
- `Groups-is-category` - Univalence for groups (iso→path)
-}

module Neural.Homotopy.FreeGroupEquiv where

open import 1Lab.Prelude
open import 1Lab.Equiv.Fibrewise
open import 1Lab.Equiv

open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin; fzero)
open import Data.Fin.Closure using (Finite-one-is-contr)

open import Algebra.Group.Cat.Base using (Group; Groups; Grp↪Sets; LiftGroup)
open import Cat.Displayed.Total using (∫Hom)
open ∫Hom
open import Algebra.Group.Free using (Free-group; Free-Group; inc; fold-free-group; Free-elim-prop; make-free-group)
open import Algebra.Group.Instances.Integers using (ℤ; ℤ-free)
open import Algebra.Group using (Group-on; is-group-hom)
open is-group-hom

open import Cat.Functor.Adjoint using (Free-object)

open import Data.Int.Base using (Int; pos)

open import Cat.Functor.Base
open import Cat.Morphism hiding (_∘Iso_; _Iso⁻¹)
open import Cat.Univalent

open Groups using (_≅_)
open import 1Lab.Type using () renaming (_∘_ to _∘ₜ_)

private variable
  ℓ : Level

{-|
## Fin 1 ≃ ⊤

Both Fin 1 and ⊤ are contractible, hence equivalent.
-}

Fin1≃⊤ : Fin 1 ≃ ⊤
Fin1≃⊤ = is-contr→≃ Finite-one-is-contr (contr tt (λ { tt → refl }))

{-|
## Free Groups Respect Equivalences

If X ≃ Y, then Free-Group X ≃ Free-Group Y.

This follows from the universal property of free groups.

**Proof sketch:**
- Free-Group X has universal property: Hom(Free-Group X, G) ≃ (X → ⌞G⌟)
- Free-Group Y has universal property: Hom(Free-Group Y, G) ≃ (Y → ⌞G⌟)
- X ≃ Y gives (X → ⌞G⌟) ≃ (Y → ⌞G⌟)
- Therefore Hom(Free-Group X, G) ≃ Hom(Free-Group Y, G) for all G
- By Yoneda, Free-Group X ≃ Free-Group Y
-}

-- Free-Group is functorial on equivalences
free-group-map : {X Y : Type ℓ} → (X → Y) → Groups.Hom (Free-Group X) (Free-Group Y)
free-group-map f = fold-free-group (inc ∘ₜ f)

-- If X ≃ Y, then Free-Group X ≅ Free-Group Y
free-group-equiv :
  {X Y : Type ℓ} →
  ⦃ _ : H-Level X 2 ⦄ → ⦃ _ : H-Level Y 2 ⦄ →
  X ≃ Y →
  Free-Group X Groups.≅ Free-Group Y
free-group-equiv {X = X} {Y = Y} e = group-iso
  where
    open Groups._≅_

    fwd : Groups.Hom (Free-Group X) (Free-Group Y)
    fwd = free-group-map (e .fst)

    bwd : Groups.Hom (Free-Group Y) (Free-Group X)
    bwd = free-group-map (equiv→inverse (e .snd))

    -- Prove the inverses using the universal property of free groups
    -- Key: Two homs from Free-Group are equal if they agree on generators
    module FY = Free-object (make-free-group (el Y (hlevel 2)))
    module FX = Free-object (make-free-group (el X (hlevel 2)))

    -- First, prove both composites agree with inc on generators
    fwd∘bwd : Groups.Hom (Free-Group Y) (Free-Group Y)
    fwd∘bwd = fwd Groups.∘ bwd

    fwd∘bwd-underlying : ⌞ Free-Group Y ⌟ → ⌞ Free-Group Y ⌟
    fwd∘bwd-underlying = fwd∘bwd .fst

    fwd∘bwd-inc : fwd∘bwd-underlying ∘ₜ inc ≡ inc
    fwd∘bwd-inc = funext λ y →
      fwd∘bwd-underlying (inc y)              ≡⟨⟩
      fwd .fst (bwd .fst (inc y))             ≡⟨⟩
      fwd .fst (inc (equiv→inverse (e .snd) y)) ≡⟨⟩
      inc (e .fst (equiv→inverse (e .snd) y))   ≡⟨ ap inc (equiv→counit (e .snd) y) ⟩
      inc y                                   ∎

    id-underlying : ⌞ Free-Group Y ⌟ → ⌞ Free-Group Y ⌟
    id-underlying x = x

    id-inc : id-underlying ∘ₜ inc ≡ inc
    id-inc = refl

    -- Apply unique to get both equal to FY.fold inc
    fwd∘bwd≡fold : fwd∘bwd ≡ FY.fold inc
    fwd∘bwd≡fold = FY.unique {f = inc} fwd∘bwd fwd∘bwd-inc

    id≡fold : Groups.id ≡ FY.fold inc
    id≡fold = FY.unique {f = inc} Groups.id (λ i → inc)

    -- Combine via transitivity
    fwd-bwd : fwd∘bwd ≡ Groups.id
    fwd-bwd = fwd∘bwd≡fold ∙ sym id≡fold

    -- Same for the other direction
    bwd∘fwd : Groups.Hom (Free-Group X) (Free-Group X)
    bwd∘fwd = bwd Groups.∘ fwd

    bwd∘fwd-underlying : ⌞ Free-Group X ⌟ → ⌞ Free-Group X ⌟
    bwd∘fwd-underlying = bwd∘fwd .fst

    bwd∘fwd-inc : bwd∘fwd-underlying ∘ₜ inc ≡ inc
    bwd∘fwd-inc = funext λ x →
      bwd∘fwd-underlying (inc x)              ≡⟨⟩
      bwd .fst (fwd .fst (inc x))             ≡⟨⟩
      bwd .fst (inc (e .fst x))               ≡⟨⟩
      inc (equiv→inverse (e .snd) (e .fst x)) ≡⟨ ap inc (equiv→unit (e .snd) x) ⟩
      inc x                                   ∎

    bwd∘fwd≡fold : bwd∘fwd ≡ FX.fold inc
    bwd∘fwd≡fold = FX.unique {f = inc} bwd∘fwd bwd∘fwd-inc

    id≡fold-X : Groups.id ≡ FX.fold inc
    id≡fold-X = FX.unique {f = inc} Groups.id (λ i → inc)

    bwd-fwd : bwd∘fwd ≡ Groups.id
    bwd-fwd = bwd∘fwd≡fold ∙ sym id≡fold-X

    group-iso : Free-Group X Groups.≅ Free-Group Y
    group-iso .to = fwd
    group-iso .from = bwd
    group-iso .inverses .Inverses.invl = fwd-bwd
    group-iso .inverses .Inverses.invr = bwd-fwd

-- Extract equivalence from isomorphism using univalence for Groups
group-iso→equiv :
  {ℓ : Level} {G H : Group ℓ} →
  G Groups.≅ H →
  ⌞ G ⌟ ≃ ⌞ H ⌟
group-iso→equiv {ℓ = ℓ} {G = G} {H = H} i = path→equiv (ap ⌞_⌟ p)
  where
    open Univalent' (Algebra.Group.Cat.Base.Groups-is-category {ℓ})
    p : G ≡ H
    p = iso→path i

{-|
## Main Theorem: Free-Group (Fin 1) ≃ ℤ

Combining the above:
1. Fin 1 ≃ ⊤
2. Therefore Free-Group (Fin 1) ≃ Free-Group ⊤
3. ℤ is Free-Group ⊤ (up to lifting)
4. Therefore Free-Group (Fin 1) ≃ ℤ
-}

-- First, show Free-Group (Fin 1) ≃ Free-Group ⊤
Free-Fin1≃Free-⊤ : Free-Group (Fin 1) Groups.≅ Free-Group ⊤
Free-Fin1≃Free-⊤ = free-group-equiv Fin1≃⊤

-- ℤ is the free group on ⊤ (from 1Lab)
-- Use universal property to show LiftGroup lzero ℤ ≅ Free-Group ⊤
module ℤ-free-iso where
  module ℤF = Free-object ℤ-free
  module FT = Free-object (make-free-group (el ⊤ (hlevel 2)))

  -- Forward: from LiftGroup lzero ℤ to Free-Group ⊤
  fwd : Groups.Hom (LiftGroup lzero ℤ) (Free-Group ⊤)
  fwd = ℤF.fold f
    where
      f : ⊤ → Free-group ⊤
      f _ = inc tt

  -- Backward: from Free-Group ⊤ to LiftGroup lzero ℤ
  bwd : Groups.Hom (Free-Group ⊤) (LiftGroup lzero ℤ)
  bwd = FT.fold g
    where
      g : ⊤ → Lift lzero ⌞ ℤ ⌟
      g _ = lift (pos 1)

  -- Prove these are inverses using the universal property
  fwd∘bwd : Groups.Hom (Free-Group ⊤) (Free-Group ⊤)
  fwd∘bwd = fwd Groups.∘ bwd

  bwd∘fwd : Groups.Hom (LiftGroup lzero ℤ) (LiftGroup lzero ℤ)
  bwd∘fwd = bwd Groups.∘ fwd

  -- Prove fwd-bwd using universal property
  -- Key insight: FT.unique needs witness that Grp↪Sets.F₁ g ∘ FT.unit ≡ f
  fwd-bwd : fwd∘bwd ≡ Groups.id
  fwd-bwd = FT.unique {Y = Free-Group ⊤} {f = FT.unit} fwd∘bwd witness₁
          ∙ sym (FT.unique {Y = Free-Group ⊤} {f = FT.unit} Groups.id witness₂)
    where
      f : ⊤ → Free-group ⊤
      f _ = inc tt

      g : ⊤ → Lift lzero ⌞ ℤ ⌟
      g _ = lift (pos 1)

      -- Show: underlying function of fwd∘bwd composed with FT.unit equals FT.unit
      witness₁ : (∫Hom.fst fwd∘bwd ∘ₜ FT.unit) ≡ FT.unit
      witness₁ = funext λ (x : ⊤) →
        let
          step1 : Free-group ⊤
          step1 = inc tt

          step2 : Lift lzero ⌞ ℤ ⌟
          step2 = lift (pos 1)
        in
        ∫Hom.fst fwd∘bwd (FT.unit x)                     ≡⟨⟩
        ∫Hom.fst fwd (∫Hom.fst bwd step1)                ≡⟨ ap (∫Hom.fst fwd) (happly (FT.commute {Y = LiftGroup lzero ℤ} {f = g}) tt) ⟩
        ∫Hom.fst fwd step2                               ≡⟨ happly (ℤF.commute {Y = Free-Group ⊤} {f = f}) tt ⟩
        step1                                            ∎

      -- Show: id ∘ FT.unit = FT.unit (definitional)
      witness₂ : (∫Hom.fst (Groups.id {x = Free-Group ⊤}) ∘ₜ FT.unit) ≡ FT.unit
      witness₂ = refl

  -- Prove bwd-fwd using universal property of ℤF
  bwd-fwd : bwd∘fwd ≡ Groups.id
  bwd-fwd = ℤF.unique {Y = LiftGroup lzero ℤ} {f = ℤF.unit} bwd∘fwd witness₁
          ∙ sym (ℤF.unique {Y = LiftGroup lzero ℤ} {f = ℤF.unit} Groups.id witness₂)
    where
      f' : ⊤ → Free-group ⊤
      f' _ = inc tt

      g' : ⊤ → Lift lzero ⌞ ℤ ⌟
      g' _ = lift (pos 1)

      -- Show: underlying function of bwd∘fwd composed with ℤF.unit equals ℤF.unit
      witness₁ : (∫Hom.fst bwd∘fwd ∘ₜ ℤF.unit) ≡ ℤF.unit
      witness₁ = funext λ (x : ⊤) →
        let
          step1 : Lift lzero ⌞ ℤ ⌟
          step1 = lift (pos 1)

          step2 : Free-group ⊤
          step2 = inc tt
        in
        ∫Hom.fst bwd∘fwd (ℤF.unit x)                     ≡⟨⟩
        ∫Hom.fst bwd (∫Hom.fst fwd step1)                ≡⟨ ap (∫Hom.fst bwd) (happly (ℤF.commute {Y = Free-Group ⊤} {f = f'}) tt) ⟩
        ∫Hom.fst bwd step2                               ≡⟨ happly (FT.commute {Y = LiftGroup lzero ℤ} {f = g'}) tt ⟩
        step1                                            ∎

      -- Show: id ∘ ℤF.unit = ℤF.unit (definitional)
      witness₂ : (∫Hom.fst (Groups.id {x = LiftGroup lzero ℤ}) ∘ₜ ℤF.unit) ≡ ℤF.unit
      witness₂ = refl

  lift-ℤ-iso : LiftGroup lzero ℤ Groups.≅ Free-Group ⊤
  lift-ℤ-iso .Groups._≅_.to = fwd
  lift-ℤ-iso .Groups._≅_.from = bwd
  lift-ℤ-iso .Groups._≅_.inverses .Inverses.invl = fwd-bwd
  lift-ℤ-iso .Groups._≅_.inverses .Inverses.invr = bwd-fwd

-- Connection from ℤ to LiftGroup lzero ℤ (universe lifting)
ℤ≃Lift-ℤ : ℤ Groups.≅ LiftGroup lzero ℤ
ℤ≃Lift-ℤ .Groups._≅_.to = fwd
  where
    fwd : Groups.Hom ℤ (LiftGroup lzero ℤ)
    fwd .∫Hom.fst = lift
    fwd .∫Hom.snd .pres-⋆ _ _ = refl

ℤ≃Lift-ℤ .Groups._≅_.from = bwd
  where
    bwd : Groups.Hom (LiftGroup lzero ℤ) ℤ
    bwd .∫Hom.fst = Lift.lower
    bwd .∫Hom.snd .pres-⋆ _ _ = refl

ℤ≃Lift-ℤ .Groups._≅_.inverses .Inverses.invl = trivial!
ℤ≃Lift-ℤ .Groups._≅_.inverses .Inverses.invr = trivial!

-- Compose to get ℤ ≅ Free-Group ⊤
ℤ≃Free-⊤ : ℤ Groups.≅ Free-Group ⊤
ℤ≃Free-⊤ = ℤ-free-iso.lift-ℤ-iso ∘Iso ℤ≃Lift-ℤ
  where open Cat.Morphism (Groups lzero)

-- Main theorem: combine the equivalences
Free-Fin1≃ℤ : ⌞ Free-Group (Fin 1) ⌟ ≃ ⌞ ℤ ⌟
Free-Fin1≃ℤ =
  ⌞ Free-Group (Fin 1) ⌟ ≃⟨ group-iso→equiv Free-Fin1≃Free-⊤ ⟩
  ⌞ Free-Group ⊤ ⌟        ≃˘⟨ group-iso→equiv ℤ≃Free-⊤ ⟩
  ⌞ ℤ ⌟                   ≃∎

{-|
## Group Equality (for Delooping)

For use with Delooping, we actually want group equality, not just equivalence
of underlying sets.

This requires showing the group structures are preserved.
-}

-- Group equality via univalence for groups
Free-Fin1≡ℤ : Free-Group (Fin 1) ≡ ℤ
Free-Fin1≡ℤ = iso→path final-iso
  where
    open Univalent' (Algebra.Group.Cat.Base.Groups-is-category {lzero})

    -- Compose: Free-Group (Fin 1) ≅ Free-Group ⊤ ≅ ℤ
    final-iso : Free-Group (Fin 1) Groups.≅ ℤ
    final-iso = ℤ≃Free-⊤ Iso⁻¹ ∘Iso Free-Fin1≃Free-⊤

{-|
## Usage in Cycle Example

This theorem completes the proof that cycle-1 realizes S¹:

1. cycle-1 has 1 edge
2. π₁(cycle-1) = Free-Group (Fin 1)
3. Free-Group (Fin 1) ≡ ℤ (this theorem)
4. 〚cycle-1〛 = Deloop(ℤ)
5. Deloop(ℤ) ≃ S¹ (proven in 1Lab)

Therefore cycle-1 realizes S¹!
-}

{-|
## Explicit Generator

The equivalence maps the single generator of Free-Group (Fin 1) to 1 ∈ ℤ.

This can be seen from the free group universal property:
- Free-Group (Fin 1) is generated by inc(fzero)
- Under the equivalence, inc(fzero) ↦ 1
- All other elements are products and inverses: inc(fzero)^n ↦ n
-}

-- The generator of Free(Fin 1) maps to 1 ∈ ℤ
generator-maps-to-1 : Free-Fin1≃ℤ .fst (inc fzero) ≡ pos 1
generator-maps-to-1 = refl

{-|
## Summary

**What we've proven:**
- Fin 1 ≃ ⊤ (both contractible)
- Free-Group (Fin 1) ≃ Free-Group ⊤ (functoriality)
- Free-Group ⊤ ≃ ℤ (1Lab theorem)
- Therefore Free-Group (Fin 1) ≃ ℤ

**Impact:**
- Completes cycle→S¹ realization
- Shows π₁(cycle-1) = ℤ
- Validates synthetic approach

**Next:**
- Prove Deloop(Free(2)) ≃ S¹ ∨∙ S¹ for figure-eight
- Use van Kampen theorem
-}
