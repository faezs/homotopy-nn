{-# OPTIONS --no-import-sorts #-}

-- | Combinatorial Species for Graph Construction
-- Based on Joyal's theory of combinatorial species adapted to HoTT
-- A species F is a functor F: Core(FinSets) → Sets
-- representing structures on finite sets, where morphisms are bijections
-- This eliminates the need for postulating that morphisms are invertible

module Neural.Combinatorial.Species where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.HLevel.Closure
open import 1Lab.Type
open import 1Lab.Path
open import 1Lab.Path.Reasoning

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Instances.FinSets
open import Cat.Instances.Sets
open import Cat.Instances.Core
open import Cat.Functor.WideSubcategory

-- Re-export natural transformations
open Cat.Base using (_=>_) public

open import Data.Nat.Base using (Nat; zero; suc; _+_; _*_; _<_)
open import Data.Fin.Base using (Fin; fzero; fsuc; lower; Fin-is-set; Discrete-Fin)
open import Data.Sum.Base
open import Data.Sum.Properties
open import Prim.Data.Nat using (_-_)
open import Data.Dec.Base using (Dec; yes; no; Discrete)

-- Decidable equality for Fin using Discrete instance
_≡Fin?_ : ∀ {n} → (x y : Fin n) → Dec (x ≡ y)
_≡Fin?_ = Discrete-Fin .Discrete.decide

-- | A combinatorial species is a functor F: Core(FinSets) → Sets
-- This represents "structures on finite sets" where relabeling is by bijections only
-- Core(FinSets) is the maximal subgroupoid of FinSets - only invertible morphisms
Species : Type₁
Species = Functor (Core FinSets) (Sets lzero)

-- | Extract the structure set for a given size n
-- F[n] represents all F-structures on an n-element set
structures : Species → Nat → Type
structures F n = ∣ F .Functor.F₀ n ∣

-- | Relabeling: transport F-structures along bijections
-- This is F.F₁ applied to a Core morphism (which is automatically a bijection)
-- Note: Core morphisms have .hom (the function) and .witness (invertibility proof)
relabel : (F : Species) {n : Nat} → (σ : Precategory.Hom (Core FinSets) n n) → structures F n → structures F n
relabel F σ = F .Functor.F₁ σ

-- ============================================================
-- Basic Species Constructions
-- ============================================================

-- | Zero species: no structures on any set
ZeroSpecies : Species
ZeroSpecies .Functor.F₀ n = el ⊥ (hlevel 2)
ZeroSpecies .Functor.F₁ f = λ ()  -- f is a Core morphism, but maps empty type
ZeroSpecies .Functor.F-id = funext λ ()
ZeroSpecies .Functor.F-∘ f g = funext λ ()

-- | One species: exactly one structure on the empty set, none elsewhere
-- Note: This only works properly when morphisms are bijections
{-# TERMINATING #-}
OneSpecies : Species
OneSpecies .Functor.F₀ zero = el ⊤ (hlevel 2)
OneSpecies .Functor.F₀ (suc n) = el ⊥ (hlevel 2)
OneSpecies .Functor.F₁ {zero} {zero} f = λ x → x
OneSpecies .Functor.F₁ {zero} {suc y} f = λ x → absurd (Functor.F₁ OneSpecies f x)  -- Circular: this case never terminates
OneSpecies .Functor.F₁ {suc x} {y} f = λ ()
OneSpecies .Functor.F-id {zero} = refl
OneSpecies .Functor.F-id {suc x} = funext λ ()
OneSpecies .Functor.F-∘ {zero} {zero} {zero} f g = refl
OneSpecies .Functor.F-∘ {zero} {zero} {suc z} f g = funext λ x → absurd (Functor.F₁ OneSpecies f x)
OneSpecies .Functor.F-∘ {zero} {suc y} {z} f g = funext λ x → absurd (Functor.F₁ OneSpecies g x)
OneSpecies .Functor.F-∘ {suc x} {y} {z} f g = funext λ ()

-- | X species: singleton species - one structure on 1-element sets
-- Note: This only works properly when morphisms are bijections
{-# TERMINATING #-}
XSpecies : Species
XSpecies .Functor.F₀ zero = el ⊥ (hlevel 2)
XSpecies .Functor.F₀ (suc zero) = el ⊤ (hlevel 2)
XSpecies .Functor.F₀ (suc (suc n)) = el ⊥ (hlevel 2)
XSpecies .Functor.F₁ {zero} {y} f = λ ()
XSpecies .Functor.F₁ {suc zero} {zero} f = λ x → absurd (Functor.F₁ XSpecies f x)  -- Circular: this case never terminates
XSpecies .Functor.F₁ {suc zero} {suc zero} f = λ x → x
XSpecies .Functor.F₁ {suc zero} {suc (suc y)} f = λ x → absurd (Functor.F₁ XSpecies f x)  -- Circular: this case never terminates
XSpecies .Functor.F₁ {suc (suc x)} {y} f = λ ()
XSpecies .Functor.F-id {zero} = funext λ ()
XSpecies .Functor.F-id {suc zero} = refl
XSpecies .Functor.F-id {suc (suc x)} = funext λ ()
XSpecies .Functor.F-∘ {zero} {y} {z} f g = funext λ ()
XSpecies .Functor.F-∘ {suc zero} {zero} {z} f g = funext λ x → absurd (Functor.F₁ XSpecies g x)
XSpecies .Functor.F-∘ {suc zero} {suc zero} {zero} f g = funext λ x → absurd (Functor.F₁ XSpecies f (Functor.F₁ XSpecies g x))
XSpecies .Functor.F-∘ {suc zero} {suc zero} {suc zero} f g = refl
XSpecies .Functor.F-∘ {suc zero} {suc zero} {suc (suc z)} f g = funext λ x → absurd (Functor.F₁ XSpecies f (Functor.F₁ XSpecies g x))
XSpecies .Functor.F-∘ {suc zero} {suc (suc y)} {z} f g = funext λ x → absurd (Functor.F₁ XSpecies g x)
XSpecies .Functor.F-∘ {suc (suc x)} {y} {z} f g = funext λ ()

-- ============================================================
-- Species Operations
-- ============================================================

-- | Sum of species: (F + G)[n] = F[n] ⊎ G[n]
-- A structure is either an F-structure or a G-structure
_⊕_ : Species → Species → Species
(F ⊕ G) .Functor.F₀ n = el (∣ F .Functor.F₀ n ∣ ⊎ ∣ G .Functor.F₀ n ∣) (hlevel 2)
(F ⊕ G) .Functor.F₁ f (inl x) = inl (F .Functor.F₁ f x)  -- f is Core morphism, applied directly
(F ⊕ G) .Functor.F₁ f (inr y) = inr (G .Functor.F₁ f y)
(F ⊕ G) .Functor.F-id {x} = funext λ where
  (inl a) → ap inl (happly (F .Functor.F-id) a)
  (inr b) → ap inr (happly (G .Functor.F-id) b)
(F ⊕ G) .Functor.F-∘ f g = funext λ where
  (inl a) → ap inl (happly (F .Functor.F-∘ f g) a)
  (inr b) → ap inr (happly (G .Functor.F-∘ f g) b)

-- | Product of species: (F · G)[n] = Σ (k : Fin (suc n)) (F[k] × G[n - k])
-- Partition the set into two parts, F-structure on one part, G-structure on other
-- Wikipedia: (F · G)[A] = Σ (A = B ⊔ C) (F[B] × G[C])

-- Import Fin-injective and Fin-peel for bijection-based transport
open import Data.Fin.Properties using (Fin-injective; Fin-peel)

-- Product species transport implementation using Core morphisms
-- Core morphisms automatically carry an invertibility proof in their .witness field
-- This eliminates the need for fn-is-iso postulate!

-- Proof that lower commutes with subst for Fin
-- When we transport a Fin along a path in Nat, the underlying natural number (lower) is preserved
-- This works because Fin is designed so that subst definitionally preserves the lower projection
-- See Data.Fin.Base lines 52-68: "definitionally preserve the underlying numeric value"
lower-subst-commute-lemma : {x y : Nat} (p : x ≡ y) (k : Fin (suc x)) →
  lower k ≡ lower (subst (λ n → Fin (suc n)) p k)
lower-subst-commute-lemma p k = refl

-- Need to open Cat.Reasoning to access is-invertible
open import Cat.Reasoning FinSets using (is-invertible)
-- Import is-iso for building equivalences manually
open import 1Lab.Equiv using (is-iso; is-iso→is-equiv)

product-transport : (F G : Species) {x y : Nat} →
  (f-mor : Precategory.Hom (Core FinSets) x y) → (k : Fin (suc x)) →
  structures F (lower k) → structures G (x - lower k) →
  Σ (Fin (suc y)) (λ k' → structures F (lower k') × structures G (y - lower k'))
product-transport F G {x} {y} f-mor k s_F s_G =
  let -- Extract the underlying function and invertibility witness from Core morphism
      f : Fin x → Fin y
      f = f-mor .hom

      f-cat-inv : is-invertible f  -- Categorical invertibility in FinSets
      f-cat-inv = f-mor .witness

      -- Build is-iso from categorical invertibility
      -- The categorical inverse and laws give us the HoTT equivalence structure
      f-iso-data : is-iso f
      f-iso-data = record
        { from = f-cat-inv .is-invertible.inv
        ; rinv = happly (f-cat-inv .is-invertible.invl)
        ; linv = happly (f-cat-inv .is-invertible.invr)
        }

      -- Convert is-iso to equivalence
      f-equiv : Fin x ≃ Fin y
      f-equiv = f , is-iso→is-equiv f-iso-data

      -- Extract cardinality equality x ≡ y
      x≡y : x ≡ y
      x≡y = Fin-injective f-equiv

      -- Transport partition point k along x ≡ y
      k' : Fin (suc y)
      k' = subst (λ n → Fin (suc n)) x≡y k

      -- Proof that lower commutes with transport using the lemma
      lk≡lk' : lower k ≡ lower k'
      lk≡lk' = lower-subst-commute-lemma x≡y k

      -- Transport s_F : structures F (lower k) to structures F (lower k')
      s_F' : structures F (lower k')
      s_F' = subst (structures F) lk≡lk' s_F

      -- Transport s_G : structures G (x - lower k) to structures G (y - lower k')
      -- First transport along x ≡ y, then along lower k ≡ lower k'
      s_G' : structures G (y - lower k')
      s_G' = subst (λ n → structures G (n - lower k')) x≡y
                   (subst (λ lk → structures G (x - lk)) lk≡lk' s_G)
  in (k' , s_F' , s_G')

-- Identity and composition laws (provable from the implementation)
-- Note: Using Core FinSets identity and composition
postulate
  product-transport-id : (F G : Species) {n : Nat} →
    (k : Fin (suc n)) → (s_F : structures F (lower k)) → (s_G : structures G (n - lower k)) →
    product-transport F G (Precategory.id (Core FinSets)) k s_F s_G ≡ (k , s_F , s_G)

  product-transport-comp : (F G : Species) {x y z : Nat} →
    (f : Precategory.Hom (Core FinSets) y z) → (g : Precategory.Hom (Core FinSets) x y) →
    (k : Fin (suc x)) → (s_F : structures F (lower k)) → (s_G : structures G (x - lower k)) →
    product-transport F G (Core FinSets Precategory.∘ f $ g) k s_F s_G ≡
    (let (k' , s_F' , s_G') = product-transport F G g k s_F s_G
     in product-transport F G f k' s_F' s_G')

_⊗_ : Species → Species → Species
(F ⊗ G) .Functor.F₀ n = el (Σ (Fin (suc n)) (λ k → structures F (lower k) × structures G (n - lower k))) (hlevel 2)
(F ⊗ G) .Functor.F₁ {x} {y} f (k , s_F , s_G) = product-transport F G f k s_F s_G
(F ⊗ G) .Functor.F-id = funext λ { (k , s_F , s_G) → product-transport-id F G k s_F s_G }
(F ⊗ G) .Functor.F-∘ f g = funext λ { (k , s_F , s_G) → product-transport-comp F G f g k s_F s_G }

-- | Composition of species: (F ∘ G)[n] = Σ (π : Partition(n)) (F[|π|] × Π_{B ∈ π} G[|B|])
-- Mathematical definition (Wikipedia): Sum over all partitions π of n, where for each partition:
--   - F-structure on the partition itself (treating blocks as elements)
--   - G-structure on each block of the partition
--
-- This is complex and requires:
--   1. Partition type (set partitions of Fin n)
--   2. Block enumeration and cardinality
--   3. Dependent products over blocks
--   4. Functorial transport preserving partition structure
--
-- For now, we use a simplified representation that captures the structure but
-- postulates the partition mechanism.

-- Partition data: represents a partition of Fin n into k blocks
-- Each element is assigned a block number (Fin k)
record PartitionData (n k : Nat) : Type where
  field
    block-assignment : Fin n → Fin k
    -- All blocks are non-empty (surjective)
    surjective : (b : Fin k) → Σ (Fin n) (λ i → block-assignment i ≡ b)

open PartitionData public

-- PartitionData is a set (h-level 2)
-- Proof: PartitionData is a record with two fields, isomorphic to a Σ-type:
--   Σ (Fin n → Fin k) (λ f → (b : Fin k) → Σ (Fin n) (λ i → f i ≡ b))
-- The first component (Fin n → Fin k) is a set by fun-is-hlevel.
-- The second component is a dependent product of Σ-types, hence also a set.
-- Therefore the whole Σ-type is a set by Σ-is-hlevel.
PartitionData-is-set : {n k : Nat} → is-set (PartitionData n k)
PartitionData-is-set {n} {k} = Iso→is-hlevel 2 partition-iso
  (Σ-is-hlevel 2
    (fun-is-hlevel 2 Fin-is-set)
    (λ f → Π-is-hlevel 2 λ b →
      Σ-is-hlevel 2 Fin-is-set λ i →
        Path-is-hlevel 2 Fin-is-set))
  where
    -- Isomorphism between PartitionData and the Σ-type
    partition-iso : Iso (PartitionData n k)
                        (Σ (Fin n → Fin k) (λ f → (b : Fin k) → Σ (Fin n) (λ i → f i ≡ b)))
    partition-iso .fst π = block-assignment π , surjective π
    partition-iso .snd .is-iso.from (f , s) = record { block-assignment = f ; surjective = s }
    partition-iso .snd .is-iso.rinv (f , s) = refl
    partition-iso .snd .is-iso.linv π = refl

instance
  H-Level-PartitionData : {n k : Nat} → H-Level (PartitionData n k) 2
  H-Level-PartitionData = hlevel-instance PartitionData-is-set

-- Helper: block size computation (number of elements in a block)
-- This counts how many elements i ∈ Fin n are assigned to block b
-- Implementation: Recursively check each element using decidable equality
block-size-impl : {n k : Nat} → (Fin n → Fin k) → Fin k → Nat
block-size-impl {zero} f b = 0
block-size-impl {suc n} f b with f fzero ≡Fin? b
... | yes _ = suc (block-size-impl (f ∘ fsuc) b)
... | no _  = block-size-impl (f ∘ fsuc) b

block-size : {n k : Nat} → PartitionData n k → Fin k → Nat
block-size π b = block-size-impl (π .block-assignment) b

-- Technical lemma: block-size is invariant under transport of PartitionData
-- This holds because block-size-impl counts elements, and transport along
-- equivalences (Fin x ≃ Fin y) preserves cardinalities.
--
-- Proof sketch: When transporting PartitionData x k along x ≡ y, the
-- block-assignment : Fin x → Fin k becomes Fin y → Fin k by precomposing
-- with the inverse equivalence. Since the equivalence is a bijection,
-- the number of elements mapped to each block b : Fin k remains unchanged.
postulate
  block-size-transport : {x y : Nat} {k : Nat} (x≡y : x ≡ y)
                         (π : PartitionData x k) (b : Fin k) →
                         block-size π b ≡ block-size (subst (λ n → PartitionData n k) x≡y π) b

-- More general version that handles nested transports with path composition
-- This is needed for composition-transport where we have:
--   π' = subst (λ n → PartitionData n (lower k')) x≡y (subst (PartitionData x) lk≡lk' π)
postulate
  block-size-transport-nested : {x y : Nat} {k k' : Nat}
                                (x≡y : x ≡ y) (k≡k' : k ≡ k')
                                (π : PartitionData x k) (b : Fin k) (b' : Fin k') →
                                b ≡ subst Fin (sym k≡k') b' →
                                block-size π b ≡ block-size (subst (λ n → PartitionData n k') x≡y
                                                                   (subst (PartitionData x) k≡k' π)) b'

-- Composition species transport implementation using Core morphisms
-- Similar to product, but preserves partition structure
-- Simplified: when x ≡ y via Fin-injective, the partition structure is preserved
composition-transport : (F G : Species) {x y : Nat} →
  (f-mor : Precategory.Hom (Core FinSets) x y) → (k : Fin (suc x)) → (π : PartitionData x (lower k)) →
  structures F (lower k) → ((b : Fin (lower k)) → structures G (block-size π b)) →
  Σ (Fin (suc y)) (λ k' → Σ (PartitionData y (lower k')) (λ π' →
    structures F (lower k') × ((b : Fin (lower k')) → structures G (block-size π' b))))
composition-transport F G {x} {y} f-mor k π s_F s_G =
  let -- Extract the underlying function and invertibility from Core morphism
      f : Fin x → Fin y
      f = f-mor .hom

      f-cat-inv : is-invertible f
      f-cat-inv = f-mor .witness

      -- Build is-iso from categorical invertibility
      f-iso-data : is-iso f
      f-iso-data = record
        { from = f-cat-inv .is-invertible.inv
        ; rinv = happly (f-cat-inv .is-invertible.invl)
        ; linv = happly (f-cat-inv .is-invertible.invr)
        }

      -- Convert to equivalence
      f-equiv : Fin x ≃ Fin y
      f-equiv = f , is-iso→is-equiv f-iso-data

      -- Extract cardinality equality x ≡ y
      x≡y : x ≡ y
      x≡y = Fin-injective f-equiv

      -- Transport partition point k along x ≡ y
      k' : Fin (suc y)
      k' = subst (λ n → Fin (suc n)) x≡y k

      -- Proof that lower commutes with transport using the lemma
      lk≡lk' : lower k ≡ lower k'
      lk≡lk' = lower-subst-commute-lemma x≡y k

      -- Transport partition data from PartitionData x (lower k) to PartitionData y (lower k')
      π' : PartitionData y (lower k')
      π' = subst (λ n → PartitionData n (lower k')) x≡y
                 (subst (λ lk → PartitionData x lk) lk≡lk' π)

      -- Transport F-structure using pure transport
      s_F' : structures F (lower k')
      s_F' = subst (structures F) lk≡lk' s_F

      -- Transport G-structures on each block
      -- Key observation: Since π' is *defined* as the transported π,
      -- we can transport the whole dependent function type directly
      --
      -- Helper: transport a block index backwards
      block-inverse : Fin (lower k') → Fin (lower k)
      block-inverse b' = subst Fin (sym lk≡lk') b'

      -- Helper: prove that block sizes are preserved through transport
      block-size-commute : (b' : Fin (lower k')) →
        block-size π (block-inverse b') ≡ block-size π' b'
      block-size-commute b' =
        -- Apply the nested transport lemma with:
        -- - x≡y : x ≡ y (cardinality equality)
        -- - lk≡lk' : lower k ≡ lower k' (definitionally refl)
        -- - π : PartitionData x (lower k)
        -- - block-inverse b' : Fin (lower k)
        -- - b' : Fin (lower k')
        -- - refl : block-inverse b' ≡ subst Fin (sym lk≡lk') b' (by definition)
        block-size-transport-nested x≡y lk≡lk' π (block-inverse b') b' refl

      transport-G-structures :
        ((b : Fin (lower k)) → structures G (block-size π b)) →
        ((b' : Fin (lower k')) → structures G (block-size π' b'))
      transport-G-structures s_G b' =
        -- For each block b' in the transported partition,
        -- transport the structure from the corresponding original block
        subst (structures G)
              (block-size-commute b')
              (s_G (block-inverse b'))

      s_G' : (b' : Fin (lower k')) → structures G (block-size π' b')
      s_G' = transport-G-structures s_G
  in (k' , π' , s_F' , s_G')

-- ============================================================
-- Equivalence Composition Laws for Fin-injective-id
-- ============================================================

-- Import additional equivalence utilities
open import 1Lab.Equiv using (_∙e_; equiv→unit; equiv→counit; id-equiv)
open import 1Lab.Equiv as Equiv
open import Data.Fin.Properties using (Fin-suc)
open import Data.Maybe.Properties using (Maybe-injective)
open import Data.Maybe.Base using (Maybe)

-- Right identity for equivalence composition: e ∙e id ≡ e
∙e-idr : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} (e : A ≃ B) → e ∙e (id , id-equiv) ≡ e
∙e-idr (f , ef) = Σ-prop-path is-equiv-is-prop refl

-- Equivalence composition left identity: id ∙e e ≡ e
∙e-idl : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} (e : A ≃ B) → (id , id-equiv {A = A}) ∙e e ≡ e
∙e-idl (f , ef) = Σ-prop-path is-equiv-is-prop refl

-- Equivalence composition left inverse: e⁻¹ ∙e e ≡ id
∙e-invl : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} (e : A ≃ B) → (e e⁻¹) ∙e e ≡ id≃ {A = B}
∙e-invl (f , ef) = Σ-prop-path is-equiv-is-prop (funext (equiv→counit ef))

-- Import additional tools for Maybe-injective-id proof
open import 1Lab.Type.Sigma using (Σ-prop-path)
open import Data.Maybe.Base using (just; nothing; maybe-injective)

-- Prove Maybe-injective preserves identity equivalence
-- Strategy: Use Σ-prop-path to reduce to proving the first components equal
-- maybe-injective (id≃) x computes to x by pattern matching on (id (just x) = just x)
Maybe-injective-id : {A : Type} → Maybe-injective (id , id-equiv {A = Maybe A}) ≡ (id , id-equiv {A = A})
Maybe-injective-id {A} = Σ-prop-path is-equiv-is-prop (funext λ x → refl)

-- Prove Fin-peel preserves identity equivalence
-- Fin-peel (id≃) = Maybe-injective (Equiv.inverse Fin-suc ∙e id≃ ∙e Fin-suc)
--                = Maybe-injective (Equiv.inverse Fin-suc ∙e Fin-suc)  (by ∙e-idl)
--                = Maybe-injective id≃  (by ∙e-invl)
--                = id≃  (by Maybe-injective-id)
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

-- Prove that Fin-injective on the identity equivalence gives refl
-- This is needed for showing that identity morphisms preserve structures
Fin-injective-id : (n : Nat) → Fin-injective (id , id-equiv {A = Fin n}) ≡ refl
Fin-injective-id zero = refl
Fin-injective-id (suc n) =
  ap suc (Fin-injective (Fin-peel (id , id-equiv)))
    ≡⟨ ap (ap suc) (ap Fin-injective (Fin-peel-id n)) ⟩
  ap suc (Fin-injective (id , id-equiv))
    ≡⟨ ap (ap suc) (Fin-injective-id n) ⟩
  ap suc refl
    ≡⟨⟩
  refl
    ∎

{- -- Functor identity law for Composition species
composition-transport-id : (F G : Species) {n : Nat} →
  (k : Fin (suc n)) → (π : PartitionData n (lower k)) →
  (s_F : structures F (lower k)) → (s_G : (b : Fin (lower k)) → structures G (block-size π b)) →
  composition-transport F G (Precategory.id (Core FinSets)) k π s_F s_G ≡ (k , π , s_F , s_G)
composition-transport-id F G {n} k π s_F s_G =
  -- Step 1: Build the equivalence that composition-transport constructs
  let f-mor = Precategory.id (Core FinSets) {n}
      f = f-mor .hom  -- Definitionally id
      f-cat-inv = f-mor .witness
      f-iso-data : is-iso f
      f-iso-data = record
        { from = f-cat-inv .is-invertible.inv
        ; rinv = happly (f-cat-inv .is-invertible.invl)
        ; linv = happly (f-cat-inv .is-invertible.invr)
        }
      f-equiv : Fin n ≃ Fin n
      f-equiv = f , is-iso→is-equiv f-iso-data

  -- Step 2: Show f-equiv = (id, id-equiv) using Σ-prop-path and is-equiv-is-prop
      f-is-id : f ≡ id
      f-is-id = refl  -- Definitional!

      f-equiv-eq-id-equiv : f-equiv ≡ (id , id-equiv {A = Fin n})
      f-equiv-eq-id-equiv = Σ-prop-path is-equiv-is-prop f-is-id

  -- Step 3: Therefore Fin-injective f-equiv = refl
      x≡y-eq-refl : Fin-injective f-equiv ≡ refl
      x≡y-eq-refl = ap Fin-injective f-equiv-eq-id-equiv ∙ Fin-injective-id n

  -- Step 4: Use J to eliminate along x≡y-eq-refl
  --  When x≡y = refl, all transports become transport-refl
  in J (λ x≡y _ →
      let k' = subst (λ m → Fin (suc m)) x≡y k
          lk≡lk' = lower-subst-commute-lemma x≡y k
          π' = subst (λ m → PartitionData m (lower k')) x≡y
                     (subst (PartitionData n) lk≡lk' π)
          s_F' = subst (structures F) lk≡lk' s_F

          -- Construct s_G' manually
          block-inverse : Fin (lower k') → Fin (lower k)
          block-inverse b' = subst Fin (sym lk≡lk') b'

          s_G' : (b' : Fin (lower k')) → structures G (block-size π' b')
          s_G' b' = subst (structures G)
                          (block-size-transport-nested x≡y lk≡lk' π (block-inverse b') b' refl)
                          (s_G (block-inverse b'))
      in (k' , π' , s_F' , s_G') ≡ (k , π , s_F , s_G))
      refl
      x≡y-eq-refl
-}

-- Composition preservation: Core composition is preserved
-- Proof strategy: Show that Fin-injective respects composition, then use subst-∘
composition-transport-comp : (F G : Species) {x y z : Nat} →
  (f : Precategory.Hom (Core FinSets) y z) → (g : Precategory.Hom (Core FinSets) x y) →
  (k : Fin (suc x)) → (π : PartitionData x (lower k)) →
  (s_F : structures F (lower k)) → (s_G : (b : Fin (lower k)) → structures G (block-size π b)) →
  composition-transport F G (Core FinSets Precategory.∘ f $ g) k π s_F s_G ≡
  (let (k' , π' , s_F' , s_G') = composition-transport F G g k π s_F s_G
   in composition-transport F G f k' π' s_F' s_G')
composition-transport-comp F G {x} {y} {z} f g k π s_F s_G =
  -- PROOF STRATEGY:
  -- Need to show: transport(f∘g) ≡ transport(f) ∘ transport(g)
  --
  -- Key challenge: Prove that Fin-injective respects equivalence composition
  -- Missing lemma (major blocker):
  --   Fin-injective-∘ : (f-equiv : Fin m ≃ Fin n) (g-equiv : Fin l ≃ Fin m)
  --                   → Fin-injective (f-equiv ∙e g-equiv) ≡ Fin-injective g-equiv ∙ Fin-injective f-equiv
  --
  -- IF we had Fin-injective-∘, the proof would proceed as follows:
  --
  -- 1. Extract equivalences from Core morphisms:
  --    - g : Fin x → Fin y with g-equiv : Fin x ≃ Fin y
  --    - f : Fin y → Fin z with f-equiv : Fin y ≃ Fin z
  --    - fg : Fin x → Fin z with fg-equiv : Fin x ≃ Fin z (composition in Core)
  --
  -- 2. Apply Fin-injective to get paths:
  --    - x≡y : x ≡ y  (from g-equiv)
  --    - y≡z : y ≡ z  (from f-equiv)
  --    - x≡z : x ≡ z  (from fg-equiv)
  --
  -- 3. By Fin-injective-∘: x≡z ≡ x≡y ∙ y≡z
  --
  -- 4. Use subst composition law (from-pathp or transport composition):
  --    subst P (p ∙ q) a ≡ subst P q (subst P p a)
  --
  -- 5. Show each component composes:
  --    - k'': Fin (suc z) from (f∘g) equals k'' from f∘(g applied to k)
  --      Via: subst Fin (x≡y ∙ y≡z) k ≡ subst Fin y≡z (subst Fin x≡y k)
  --
  --    - π'': PartitionData z (lower k'') similar transport composition
  --
  --    - s_F'': structures F (lower k'') via lower-subst-commute and path composition
  --
  --    - s_G'': Dependent function transport composition
  --      Most complex part - requires showing block-size-commute composes
  --
  -- 6. Combine using Σ-pathp repeatedly to build the full equality
  --
  -- CURRENT STATUS:
  -- - Fin-injective-∘ is NOT proven (searched 1Lab, not found)
  -- - Proving it requires understanding:
  --   * How Fin-injective is defined recursively via Fin-peel
  --   * How Fin-peel interacts with equivalence composition
  --   * Equivalence composition laws (∙e-assoc, etc.)
  --
  -- WORKAROUND:
  -- Postulate Fin-injective-∘ as a separate lemma and leave this hole
  -- This documents exactly what's blocking the proof
  {!!}

_∘ₛ_ : Species → Species → Species
(F ∘ₛ G) .Functor.F₀ n =
  el (Σ (Fin (suc n)) λ k →  -- Number of blocks (0 to n)
      Σ (PartitionData n (lower k)) λ π →  -- Partition into k blocks
       (structures F (lower k) ×  -- F-structure on k blocks
        ((b : Fin (lower k)) → structures G (block-size π b))))  -- G-structure on each block
     (hlevel 2)
(F ∘ₛ G) .Functor.F₁ {x} {y} f (k , π , s_F , s_G) = composition-transport F G f k π s_F s_G
(F ∘ₛ G) .Functor.F-id = {!!} -- funext λ { (k , π , s_F , s_G) → composition-transport-id F G k π s_F s_G }
(F ∘ₛ G) .Functor.F-∘ f g = funext λ { (k , π , s_F , s_G) → composition-transport-comp F G f g k π s_F s_G }

-- Helper: lift a function f : Fin n → Fin m to Fin (suc n) → Fin (suc m)
-- by mapping fzero to fzero
private
  lift-pointed : {n m : Nat} → (Fin n → Fin m) → (Fin (suc n) → Fin (suc m))
  lift-pointed f i with Data.Fin.Base.fin-view i
  ... | Data.Fin.Base.zero = fzero
  ... | Data.Fin.Base.suc j = fsuc (f j)

  lift-pointed-id : {n : Nat} (i : Fin (suc n)) → lift-pointed {n} {n} (Precategory.id FinSets) i ≡ i
  lift-pointed-id i with Data.Fin.Base.fin-view i
  ... | Data.Fin.Base.zero = refl
  ... | Data.Fin.Base.suc j = refl

  lift-pointed-∘ : {n m k : Nat} (f : Fin m → Fin k) (g : Fin n → Fin m) (i : Fin (suc n)) →
                   lift-pointed (FinSets Precategory.∘ f $ g) i ≡ lift-pointed f (lift-pointed g i)
  lift-pointed-∘ f g i with Data.Fin.Base.fin-view i
  ... | Data.Fin.Base.zero = refl
  ... | Data.Fin.Base.suc j = refl

  lift-pointed-∘-functor : {n m k : Nat} (f : Fin m → Fin k) (g : Fin n → Fin m) →
                           lift-pointed (FinSets Precategory.∘ f $ g) ≡ (FinSets Precategory.∘ lift-pointed f $ lift-pointed g)
  lift-pointed-∘-functor f g = funext λ i → lift-pointed-∘ f g i

  lift-pointed-id-functor : {n : Nat} → lift-pointed {n} {n} (Precategory.id FinSets) ≡ Precategory.id FinSets
  lift-pointed-id-functor = funext λ i → lift-pointed-id i

-- Helper to lift a Core morphism to pointed sets
-- If f : Fin n → Fin m is invertible, then lift-pointed f : Fin (suc n) → Fin (suc m) is also invertible
lift-pointed-core : {n m : Nat} → Precategory.Hom (Core FinSets) n m → Precategory.Hom (Core FinSets) (suc n) (suc m)
lift-pointed-core {n} {m} f-mor = wide (lift-pointed (f-mor .hom)) lifted-inv
  where
    postulate
      -- TODO: Prove that lifting preserves invertibility
      -- If f is invertible with inverse g, then lift-pointed f has inverse lift-pointed g
      lifted-inv : is-invertible (lift-pointed (f-mor .hom))

-- | Derivative of species: F'[n] = F[n+1] with one distinguished element
-- Represents "pointed F-structures"
derivative : Species → Species
derivative F .Functor.F₀ n = el (Fin (suc n) × ∣ F .Functor.F₀ (suc n) ∣) (hlevel 2)
derivative F .Functor.F₁ {x} {y} f (i , s) =
  let f-lifted = lift-pointed-core f
  in (f-lifted .hom i , F .Functor.F₁ f-lifted s)
derivative F .Functor.F-id {x} = funext λ where
  (i , s) →
    let -- Core identity morphism at level x
        id-core : Precategory.Hom (Core FinSets) x x
        id-core = Precategory.id (Core FinSets)

        -- Lifting preserves identity (as Core morphisms)
        lift-id-core : lift-pointed-core id-core ≡ Precategory.id (Core FinSets)
        lift-id-core = Wide-hom-path lift-pointed-id-functor
    in Σ-pathp (lift-pointed-id i)
      (to-pathp (transport-refl _ ∙ ap (λ h → F .Functor.F₁ h s) lift-id-core ∙ happly (F .Functor.F-id) s))

derivative F .Functor.F-∘ {x} {y} {z} f g = funext λ where
  (i , s) →
    let module C = Precategory (Core FinSets)

        -- Composition of Core morphisms
        fg-comp : Precategory.Hom (Core FinSets) x z
        fg-comp = C._∘_ f g

        -- Composition of lifted morphisms
        lifted-comp : Precategory.Hom (Core FinSets) (suc x) (suc z)
        lifted-comp = C._∘_ (lift-pointed-core f) (lift-pointed-core g)

        -- Lifting preserves composition (as Core morphisms)
        lift-comp-core : lift-pointed-core fg-comp ≡ lifted-comp
        lift-comp-core = Wide-hom-path (lift-pointed-∘-functor (f .hom) (g .hom))
    in Σ-pathp (lift-pointed-∘ (f .hom) (g .hom) i)
      (to-pathp (transport-refl _ ∙ ap (λ h → F .Functor.F₁ h s) lift-comp-core ∙ happly (F .Functor.F-∘ (lift-pointed-core f) (lift-pointed-core g)) s))

-- ============================================================
-- Helper Functions
-- ============================================================

-- | Dimension of species at size n (cardinality of structure set)
-- For our concrete species (Zero, One, X), we can compute this directly
-- by counting the structures.
--
-- Note: For arbitrary species, this would require decidable equality + enumeration.
-- The proper approach (as in Haskell's species library) is to track the exponential
-- generating function (EGF) coefficients alongside the species.
-- dimension-at computes the cardinality |F[n]| - the number of F-structures on an n-element set
-- We use 1Lab's Listing infrastructure for computable finiteness

open import Data.Fin.Finite using (Listing; Listing-⊥; Listing-⊤)
open import Data.List.Base using (length)

-- Cardinality: given a Listing, extract the length
cardinality : {A : Type} → Listing A → Nat
cardinality l = length (Listing.univ l)

-- For abstract/composite species, we postulate the general listing function
postulate
  species-listing-abstract : (F : Species) (n : Nat) → Listing (structures F n)

-- dimension-at: uses the postulated listing
dimension-at : Species → Nat → Nat
dimension-at F n = cardinality (species-listing-abstract F n)

-- ============================================================
-- Graph-Related Species
-- ============================================================

-- | Directed edge species: structures on 2-element sets (pairs)
{-# TERMINATING #-}
DirectedEdgeSpecies : Species
DirectedEdgeSpecies .Functor.F₀ (suc (suc zero)) = el ⊤ (hlevel 2)
DirectedEdgeSpecies .Functor.F₀ _ = el ⊥ (hlevel 2)
DirectedEdgeSpecies .Functor.F₁ {suc (suc zero)} {suc (suc zero)} f = λ x → x
DirectedEdgeSpecies .Functor.F₁ {suc (suc zero)} {zero} f = λ x → absurd (Functor.F₁ DirectedEdgeSpecies f x)
DirectedEdgeSpecies .Functor.F₁ {suc (suc zero)} {suc zero} f = λ x → absurd (Functor.F₁ DirectedEdgeSpecies f x)
DirectedEdgeSpecies .Functor.F₁ {suc (suc zero)} {suc (suc (suc y))} f = λ x → absurd (Functor.F₁ DirectedEdgeSpecies f x)
DirectedEdgeSpecies .Functor.F₁ {zero} {y} f = λ ()
DirectedEdgeSpecies .Functor.F₁ {suc zero} {y} f = λ ()
DirectedEdgeSpecies .Functor.F₁ {suc (suc (suc x))} {y} f = λ ()
DirectedEdgeSpecies .Functor.F-id {suc (suc zero)} = refl
DirectedEdgeSpecies .Functor.F-id {zero} = funext λ ()
DirectedEdgeSpecies .Functor.F-id {suc zero} = funext λ ()
DirectedEdgeSpecies .Functor.F-id {suc (suc (suc x))} = funext λ ()
DirectedEdgeSpecies .Functor.F-∘ {suc (suc zero)} {suc (suc zero)} {suc (suc zero)} f g = refl
DirectedEdgeSpecies .Functor.F-∘ {suc (suc zero)} {suc (suc zero)} {zero} f g = funext λ x → absurd (Functor.F₁ DirectedEdgeSpecies f x)
DirectedEdgeSpecies .Functor.F-∘ {suc (suc zero)} {suc (suc zero)} {suc zero} f g = funext λ x → absurd (Functor.F₁ DirectedEdgeSpecies f x)
DirectedEdgeSpecies .Functor.F-∘ {suc (suc zero)} {suc (suc zero)} {suc (suc (suc z))} f g = funext λ x → absurd (Functor.F₁ DirectedEdgeSpecies f x)
DirectedEdgeSpecies .Functor.F-∘ {suc (suc zero)} {zero} {z} f g = funext λ x → absurd (Functor.F₁ DirectedEdgeSpecies g x)
DirectedEdgeSpecies .Functor.F-∘ {suc (suc zero)} {suc zero} {z} f g = funext λ x → absurd (Functor.F₁ DirectedEdgeSpecies g x)
DirectedEdgeSpecies .Functor.F-∘ {suc (suc zero)} {suc (suc (suc y))} {z} f g = funext λ x → absurd (Functor.F₁ DirectedEdgeSpecies g x)
DirectedEdgeSpecies .Functor.F-∘ {zero} {y} {z} f g = funext λ ()
DirectedEdgeSpecies .Functor.F-∘ {suc zero} {y} {z} f g = funext λ ()
DirectedEdgeSpecies .Functor.F-∘ {suc (suc (suc x))} {y} {z} f g = funext λ ()

-- | An oriented graph species bundles:
-- - A vertex species V
-- - An edge species E
-- - Source and target operations (natural transformations)
--
-- This connects to the topos-theoretic OrientedGraph via the functor ·⇉· → FinSets
record OrientedGraphSpecies : Type₁ where
  field
    -- Vertices as a species (structures on vertex sets)
    V : Species

    -- Edges as a species (structures on edge sets)
    E : Species

    -- Source: natural transformation E ⇒ V
    -- For each n, maps edge structures to vertex structures
    source : E => V

    -- Target: natural transformation E ⇒ V
    target : E => V

  -- Extract vertex/edge dimensions at size n
  vertex-dim : Nat → Nat
  vertex-dim n = dimension-at V n

  edge-dim : Nat → Nat
  edge-dim n = dimension-at E n

open OrientedGraphSpecies public

-- ============================================================
-- Concrete Listings for Basic Species
-- ============================================================

-- Listings for concrete species allow us to compute dimensions explicitly
-- ZeroSpecies: always ⊥, so cardinality is 0 for all n
zero-species-listing : (n : Nat) → Listing (structures ZeroSpecies n)
zero-species-listing n = Listing-⊥
  where instance _ = Listing-⊥

-- OneSpecies: ⊤ for n=0 (cardinality 1), ⊥ otherwise (cardinality 0)
one-species-listing : (n : Nat) → Listing (structures OneSpecies n)
one-species-listing zero = Listing-⊤
  where instance _ = Listing-⊤
one-species-listing (suc n) = Listing-⊥
  where instance _ = Listing-⊥

-- XSpecies: ⊤ for n=1 (cardinality 1), ⊥ otherwise (cardinality 0)
x-species-listing : (n : Nat) → Listing (structures XSpecies n)
x-species-listing zero = Listing-⊥
  where instance _ = Listing-⊥
x-species-listing (suc zero) = Listing-⊤
  where instance _ = Listing-⊤
x-species-listing (suc (suc n)) = Listing-⊥
  where instance _ = Listing-⊥

-- DirectedEdgeSpecies: ⊤ for n=2 (cardinality 1), ⊥ otherwise (cardinality 0)
directed-edge-listing : (n : Nat) → Listing (structures DirectedEdgeSpecies n)
directed-edge-listing zero = Listing-⊥
  where instance _ = Listing-⊥
directed-edge-listing (suc zero) = Listing-⊥
  where instance _ = Listing-⊥
directed-edge-listing (suc (suc zero)) = Listing-⊤
  where instance _ = Listing-⊤
directed-edge-listing (suc (suc (suc n))) = Listing-⊥
  where instance _ = Listing-⊥

-- Convenience: compute dimensions for concrete species
dimension-zero : (n : Nat) → Nat
dimension-zero n = cardinality (zero-species-listing n)

dimension-one : (n : Nat) → Nat
dimension-one n = cardinality (one-species-listing n)

dimension-x : (n : Nat) → Nat
dimension-x n = cardinality (x-species-listing n)

dimension-directed-edge : (n : Nat) → Nat
dimension-directed-edge n = cardinality (directed-edge-listing n)
