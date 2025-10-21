{-# OPTIONS --no-import-sorts #-}

-- Equivalence composition laws for Species module
-- These lemmas are used to prove Fin-injective-id

module Neural.Combinatorial.SpeciesEquivLaws where

open import 1Lab.Prelude
open import 1Lab.Equiv using (_∙e_; equiv→unit; equiv→counit; id-equiv)
open import 1Lab.Equiv as Equiv
open import Data.Fin.Base using (Fin)
open import Data.Fin.Properties using (Fin-injective; Fin-peel; Fin-suc)
open import Data.Maybe.Base using (Maybe)
open import Data.Maybe.Properties using (Maybe-injective)

-- Right identity law for equivalence composition: e ∙e id ≡ e
∙e-idr : {A B : Type} (e : A ≃ B) → e ∙e (id , id-equiv {A = B}) ≡ e
∙e-idr {A} {B} (f , ef) = Σ-path refl (hlevel 1)

-- Left identity law for equivalence composition: id ∙e e ≡ e
∙e-idl : {A B : Type} (e : A ≃ B) → (id , id-equiv {A = A}) ∙e e ≡ e
∙e-idl {A} {B} (f , ef) = Σ-path refl (hlevel 1)

-- Left inverse law: e⁻¹ ∙e e ≡ id
∙e-invl : {A B : Type} (e : A ≃ B) → Equiv.inverse e ∙e e ≡ (id , id-equiv)
∙e-invl {A} {B} e = Σ-path (funext (equiv→unit e)) (hlevel 1)

-- Right inverse law: e ∙e e⁻¹ ≡ id
∙e-invr : {A B : Type} (e : A ≃ B) → e ∙e Equiv.inverse e ≡ (id , id-equiv)
∙e-invr {A} {B} e = Σ-path (funext (equiv→counit e)) (hlevel 1)

-- Prove that Maybe-injective id ≡ id
Maybe-injective-id : {A : Type} → Maybe-injective (id , id-equiv {A = Maybe A}) ≡ (id , id-equiv {A = A})
Maybe-injective-id {A} = Σ-path refl (hlevel 1)

-- Prove that Fin-peel id ≡ id
open import Data.Nat.Base using (Nat)

Fin-peel-id : (n : Nat) → Fin-peel (id , id-equiv {A = Fin (suc n)}) ≡ (id , id-equiv {A = Fin n})
Fin-peel-id n =
  Fin-peel (id , id-equiv)
    ≡⟨⟩  -- By definition of Fin-peel
  Maybe-injective (Equiv.inverse Fin-suc ∙e (id , id-equiv) ∙e Fin-suc)
    ≡⟨ ap Maybe-injective (∙e-idr (Equiv.inverse Fin-suc ∙e (id , id-equiv))) ⟩
  Maybe-injective (Equiv.inverse Fin-suc ∙e (id , id-equiv))
    ≡⟨ ap Maybe-injective (∙e-idl (Equiv.inverse Fin-suc)) ⟩
  Maybe-injective (Equiv.inverse Fin-suc)
    ≡⟨ ap Maybe-injective (∙e-invl Fin-suc) ⟩
  Maybe-injective (id , id-equiv)
    ≡⟨ Maybe-injective-id ⟩
  (id , id-equiv)
    ∎
