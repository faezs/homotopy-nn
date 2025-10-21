{-# OPTIONS --no-import-sorts #-}

-- | Equivalence composition laws for proving Fin-peel-id
-- These should be straightforward but aren't in 1Lab explicitly

module Neural.Combinatorial.EquivLaws where

open import 1Lab.Prelude
open import 1Lab.Equiv using (_∙e_; id≃; id-equiv; equiv→unit; equiv→counit; equiv→inverse)
open import 1Lab.Equiv as Equiv
open import 1Lab.Path
open import 1Lab.HLevel

open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin)

-- Prove left identity: id≃ ∙e e ≡ e
-- By definition: (id≃ ∙e e).fst = e.fst ∘ id = e.fst
∙e-idl : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} (e : A ≃ B) → id≃ ∙e e ≡ e
∙e-idl e = Σ-prop-path is-equiv-is-prop refl

-- Prove right identity: e ∙e id≃ ≡ e
-- By definition: (e ∙e id≃).fst = id ∘ e.fst = e.fst
∙e-idr : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} (e : A ≃ B) → e ∙e id≃ ≡ e
∙e-idr e = Σ-prop-path is-equiv-is-prop refl

-- Prove left inverse: e⁻¹ ∙e e ≡ id≃
-- e : A ≃ B, so e.fst : A → B
-- e⁻¹ : B ≃ A, so (e⁻¹).fst : B → A
-- (e⁻¹ ∙e e).fst = e.fst ∘ (e⁻¹).fst : B → B
--                = e.fst ∘ equiv→inverse (e.snd)
--                = λ (y : B) → e.fst (equiv→inverse (e.snd) y)
-- By equiv→counit: e.fst (equiv→inverse (e.snd) y) ≡ y
∙e-invl : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} (e : A ≃ B) → (e Equiv.e⁻¹) ∙e e ≡ id≃
∙e-invl (f , ef) = Σ-prop-path is-equiv-is-prop (funext (equiv→counit ef))

-- Prove right inverse: e ∙e e⁻¹ ≡ id≃
-- e : A ≃ B, so e.fst : A → B
-- e⁻¹ : B ≃ A, so (e⁻¹).fst : B → A
-- (e ∙e e⁻¹).fst = (e⁻¹).fst ∘ e.fst : A → A
--                = equiv→inverse (e.snd) ∘ e.fst
--                = λ (x : A) → equiv→inverse (e.snd) (e.fst x)
-- By equiv→unit: equiv→inverse (e.snd) (e.fst x) ≡ x
∙e-invr : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} (e : A ≃ B) → e ∙e (e Equiv.e⁻¹) ≡ id≃
∙e-invr (f , ef) = Σ-prop-path is-equiv-is-prop (funext (equiv→unit ef))

-- Now prove Maybe-injective preserves identity
open import Data.Maybe.Base using (Maybe; just; nothing; maybe-injective)
open import Data.Maybe.Properties using (Maybe-injective)

-- Prove Maybe-injective (id≃) ≡ id≃
-- Strategy: Use Σ-prop-path to reduce to showing the first components are equal
-- maybe-injective (id≃) x computes as follows:
--   - id (just x) = just x (by with-pattern matching)
--   - Extract x from (just x), giving x
--   - So maybe-injective (id≃) = id
Maybe-injective-id : {A : Type} → Maybe-injective (id , id-equiv {A = Maybe A}) ≡ (id , id-equiv {A = A})
Maybe-injective-id {A} = Σ-prop-path is-equiv-is-prop (funext λ x → refl)

-- Finally, prove Fin-peel preserves identity
-- Fin-peel (id≃) = Maybe-injective (Equiv.inverse Fin-suc ∙e id≃ ∙e Fin-suc)
--                = Maybe-injective (Equiv.inverse Fin-suc ∙e Fin-suc)  (by ∙e-idr)
--                = Maybe-injective id≃  (by ∙e-invr)
--                = id≃  (by Maybe-injective-id)
open import Data.Fin.Properties using (Fin-peel; Fin-suc)

Fin-peel-id : (n : Nat) → Fin-peel (id , id-equiv {A = Fin (suc n)}) ≡ (id , id-equiv {A = Fin n})
Fin-peel-id n =
  Fin-peel (id , id-equiv)
    ≡⟨⟩  -- By definition of Fin-peel
  Maybe-injective (Equiv.inverse Fin-suc ∙e (id , id-equiv) ∙e Fin-suc)
    ≡⟨ ap Maybe-injective (ap (λ e → Equiv.inverse Fin-suc ∙e e) (∙e-idl Fin-suc)) ⟩
  Maybe-injective (Equiv.inverse Fin-suc ∙e Fin-suc)
    ≡⟨ ap Maybe-injective (∙e-invl {A = Fin (suc n)} {B = Maybe (Fin n)} Fin-suc) ⟩
  Maybe-injective (id , id-equiv)
    ≡⟨ Maybe-injective-id {A = Fin n} ⟩
  (id , id-equiv)
    ∎
