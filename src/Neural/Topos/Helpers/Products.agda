{-# OPTIONS --cubical --rewriting --guardedness --no-load-primitives #-}

-- Helper lemmas for products preserving various properties
-- Phase 1 of sheafification left-exactness proof
--
-- KEY DISCOVERY: 1Lab already has EVERYTHING we need!
-- - Π-is-hlevel: Products of n-types are n-types (for n=0: contractibility)
-- - PSh-terminal: Terminal object in presheaves
-- - PSh-pullbacks: Pullbacks in presheaves (computed pointwise)
--
-- This file just re-exports the relevant lemmas with convenient names

module Neural.Topos.Helpers.Products where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.HLevel.Closure

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Diagram.Terminal
open import Cat.Diagram.Pullback
open import Cat.Instances.Functor
open import Cat.Instances.Presheaf.Limits

private variable
  o ℓ κ : Level
  C : Precategory o ℓ

-- =============================================================================
-- PHASE 1 COMPLETE: All helper lemmas already exist in 1Lab!
-- =============================================================================

-- Task 1.1: Products of contractibles are contractible ✅
Π-is-contr : {I : Type ℓ} {A : I → Type κ}
           → (∀ i → is-contr (A i))
           → is-contr ((i : I) → A i)
Π-is-contr = Π-is-hlevel 0

-- Task 1.2: Presheaves have pullbacks (computed pointwise) ✅
-- Already available as PSh-pullbacks from Cat.Instances.Presheaf.Limits

-- Task 1.3: Presheaves have terminals ✅
-- Already available as PSh-terminal from Cat.Instances.Presheaf.Limits

-- =============================================================================
-- Summary of what 1Lab provides
-- =============================================================================

-- From 1Lab.HLevel.Closure:
--   Π-is-hlevel n : Products preserve h-levels
--   Specifically for n=0: Products of contractibles are contractible

-- From Cat.Instances.Presheaf.Limits:
--   PSh-terminal : Terminal (PSh κ C)
--   PSh-pullbacks : ∀ {X Y Z} (f : X => Z) (g : Y => Z) → Pullback (PSh κ C) f g
--   PSh-products : ∀ (F : I → Functor (C ^op) (Sets κ)) → Product (PSh κ C) F
--
-- Key property: All limits in PSh are computed pointwise!
--   This means: (lim F)(c) = lim (F(-)(c))

-- =============================================================================
-- What remains for Phase 2 (THE HARD PART)
-- =============================================================================

-- Phase 2 requires proving that fork-sheafification's explicit construction
-- (from the paper, lines 572-579) equals 1Lab's HIT definition.
--
-- Specifically:
--   Sheafify(F)(v) = F(v)                    -- at original vertices
--   Sheafify(F)(A) = F(A)                    -- at fork-tang
--   Sheafify(F)(A★) = ∏_{a'→A★} F(a')      -- at fork-star (PRODUCT!)
--
-- This is the deep HIT reasoning that's genuinely hard.
-- But once we have it, the rest follows from Π-is-contr and PSh-pullbacks!

-- =============================================================================
-- Verification: Π-is-contr actually works
-- =============================================================================

private
  -- Simple test: products of units are contractible
  _ : is-contr ((i : ⊤) → ⊤)
  _ = Π-is-contr λ _ → contr tt (λ _ → refl)

  -- More interesting: products of singleton types
  _ : {A : Type} (a : A) → is-contr ((i : ⊤) → Σ A (λ x → x ≡ a))
  _ = λ a → Π-is-contr λ _ → contr (a , refl) λ { (x , p) i → p (~ i) , λ j → p (~ i ∨ j) }

-- =============================================================================
-- PHASE 1 STATUS: ✅ COMPLETE
-- =============================================================================

-- We have everything we need from 1Lab:
-- ✅ Π-is-contr for products of contractibles
-- ✅ PSh-terminal for terminal object in presheaves
-- ✅ PSh-pullbacks for pullbacks in presheaves
-- ✅ Pointwise limit computation
--
-- Total new code: ~0 lines (just re-exports!)
-- Time spent: ~30 minutes (mostly searching 1Lab)
--
-- Ready to proceed to Phase 2: HIT reasoning for fork sheafification
