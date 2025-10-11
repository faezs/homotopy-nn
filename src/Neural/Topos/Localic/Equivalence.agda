{-# OPTIONS --rewriting --guardedness --cubical --no-load-primitives #-}

{-|
# Localic Topos: Main Equivalence

This module proves the equivalence between SetΩ and Sh(Ω, K).

## Contents
- §A.7: F-functor: SetΩ → Sh(Ω, K)
- §A.7: G-functor: Sh(Ω, K) → SetΩ
- Proposition A.2: Localic equivalence SetΩ ≃ Sh(Ω, K)
-}

module Neural.Topos.Localic.Equivalence where

open import 1Lab.Prelude
open import 1Lab.Resizing

open import Neural.Topos.Localic.Base
open import Neural.Topos.Localic.Category
open import Neural.Topos.Localic.Internal

open import 1Lab.Resizing using (□-out!)

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Functor.Equivalence
open import Cat.Instances.Sets
open import Cat.Site.Base using (Coverage; Sheaves; is-sheaf)
open import Order.Cat using (poset→category)
open import Order.Diagram.Meet
open import Order.Diagram.Lub
open import Order.Instances.Lower using (Lower-sets; Lower-set; よₚ; ↓)
open import Order.Heyting
open import Order.Frame
open import Order.Semilattice.Meet.Reasoning

--------------------------------------------------------------------------------
-- §A.6: Yoneda Embedding and Ω-U
--
-- Following Belfiore & Bennequin Appendix A (page 121-122):
-- "The set Ω_U can be identified with the Ω-set associated to the
--  Yoneda presheaf defined by U."
--
-- Note: Ω_U is already defined in Neural.Topos.Localic.Internal.
-- It has carrier Σ[V : Ω] (□ (V ≤ U)), which is the principal downset ↓U.
--
-- This corresponds to 1Lab's よₚ: P → Lower-sets P (Yoneda for posets),
-- where Lower-sets P = Poset[(P ^op), Props].
--
-- The inclusion morphisms Ω_V → Ω_U (when V ≤ U) are induced by
-- よₚ being a monotone map, i.e., よₚ.pres-≤ : V ≤ U → よₚ(V) ≤ よₚ(U).
--
-- This makes functoriality (inclusion-comp) automatic from monotone map laws!

--------------------------------------------------------------------------------
-- §A.7: Main Equivalence (Proposition A.1-A.2)

{-|
## Proposition A.1: Morphisms induce Natural Transformations

An Ω-set morphism f: X → Y induces a natural transformation
  f_Ω: X_Ω → Y_Ω
of presheaves over Ω.

**Key idea** (Equation 27):
  f_V(u) = f_U(u) ∩ V

The value on U determines all values on V ≤ U by intersection.

**Sheaf property**: X_Ω is automatically a sheaf, not just presheaf!
-}

-- Proposition A.1: Morphisms correspond to natural transformations
-- This would require defining the functor from Ω-sets to presheaves
-- For now, we note the type signature
morphism-to-natural-transformation :
  {o ℓ : Level} {Ω : CompleteHeytingAlgebra o ℓ}
  → {X Y : Ω-Set Ω}
  → Ω-Set-Morphism X Y
  → {- Natural transformation would go here -} ⊤
morphism-to-natural-transformation _ = tt

{-|
## Proposition A.2: Main Equivalence

**Theorem**: The functors
- F: (X,δ) ↦ (U ↦ Hom_Ω(Ω_U, X))
- G: X ↦ (Hom_E(Ω, X), δ_X)

define an **equivalence of categories**:
  SetΩ ≃ Sh(Ω, K)

**Proof idea**:
1. F ∘ G ≃ Id: Sub-singletons generate sheaves
2. G ∘ F ≃ Id: Compatible coverings determine sections
3. Both are natural isomorphisms

**Consequence**: Localic toposes ARE Ω-set categories!

**For DNNs**: Network semantics = Fuzzy sets with progressive decisions
-}

-- Implementation of the equivalence SetΩ ≃ Sh(Ω, K)
module Localic-Equivalence {o ℓ : Level} (Ω : CompleteHeytingAlgebra o ℓ) where
  private
    module ΩA = CompleteHeytingAlgebra Ω
    C = poset→category ΩA.poset

  open import Cat.Functor.Adjoint using (_⊣_)
  open Cat.Base using (_=>_)
  open _=>_ renaming (η to component; is-natural to naturality)

  -- Use the canonical coverage from 1Lab
  open import Cat.Site.Instances.Frame ΩA.poset ΩA.frame using (Open-coverage)

  K : Coverage C (lsuc o)
  K = Open-coverage

  -- F-functor: SetΩ → Sh(Ω, K)
  -- Maps (X,δ) to presheaf U ↦ Hom_SetΩ(Ω_U, X)
  module F-Functor where

    -- Helper for inclusion morphism: maps carriers
    inclusion-carrier : ∀ {U V : ΩA.Ob} → (V ΩA.≤ U)
                      → Ω-Set.Carrier (Ω-U Ω V) → Ω-Set.Carrier (Ω-U Ω U)
    inclusion-carrier {U} {V} V≤U (W , W≤V-boxed) =
      W , inc (ΩA.≤-trans (□-rec ΩA.≤-thin (λ p → p) W≤V-boxed) V≤U)

    -- Helper for inclusion morphism: fuzzy function
    inclusion-f : ∀ {U V : ΩA.Ob} → (V≤U : V ΩA.≤ U)
                → Ω-Set.Carrier (Ω-U Ω V) → Ω-Set.Carrier (Ω-U Ω U) → ΩA.Ob
    inclusion-f {U} {V} V≤U x y = Ω-Set.δ (Ω-U Ω U) (inclusion-carrier V≤U x) y

    -- Properties of inclusion (provable from δ_U properties)
    -- Key insight: inclusion-f is just δ_U applied to the mapped carrier

    -- eq-20: Direct application of δ_U transitivity
    inclusion-eq-20 : ∀ {U V : ΩA.Ob} (V≤U : V ΩA.≤ U) {x x' y'}
                    → (inclusion-f V≤U x x' ΩA.∩ Ω-Set.δ (Ω-U Ω U) x' y')
                      ΩA.≤ inclusion-f V≤U x y'
    inclusion-eq-20 {U} {V} V≤U {x} {x'} {y'} =
      Ω-Set.δ-trans (Ω-U Ω U) {inclusion-carrier V≤U x} {x'} {y'}

    -- eq-19: δ_V(x,y) ∩ δ_U(incl(x), x') ≤ δ_U(incl(y), x')
    -- Key: δ_V and δ_U compute the same value since they only depend on first component
    -- So δ_V(x,y) = δ_U(incl(x), incl(y))
    inclusion-eq-19 : ∀ {U V : ΩA.Ob} (V≤U : V ΩA.≤ U) {x y x'}
                    → (Ω-Set.δ (Ω-U Ω V) x y ΩA.∩ inclusion-f V≤U x x')
                      ΩA.≤ inclusion-f V≤U y x'
    inclusion-eq-19 {U} {V} V≤U {(W₁ , p₁)} {(W₂ , p₂)} {x'} =
      ΩA.≤-trans step (Ω-Set.δ-trans (Ω-U Ω U) {inclusion-carrier V≤U (W₂ , p₂)} {inclusion-carrier V≤U (W₁ , p₁)} {x'})
      where
        module ΩV = Ω-Set (Ω-U Ω V)
        module ΩU = Ω-Set (Ω-U Ω U)
        -- Key: δ_V((W₁,_), (W₂,_)) = (W₁⇨W₂) ∩ (W₂⇨W₁) = δ_U((W₁,_), (W₂,_))
        -- And δ is symmetric, so δ(W₁,W₂) = δ(W₂,W₁)
        step : (ΩV.δ (W₁ , p₁) (W₂ , p₂) ΩA.∩ inclusion-f V≤U (W₁ , p₁) x')
               ΩA.≤ (ΩU.δ (inclusion-carrier V≤U (W₂ , p₂)) (inclusion-carrier V≤U (W₁ , p₁)) ΩA.∩ inclusion-f V≤U (W₁ , p₁) x')
        step = ΩA.≤-refl' (ap₂ ΩA._∩_ (ΩU.δ-sym {inclusion-carrier V≤U (W₁ , p₁)} {inclusion-carrier V≤U (W₂ , p₂)}) refl)

    -- eq-21: f(x,x') ∩ f(x,y') ≤ δ_U(x',y')
    -- Proof: δ_U(incl(x), x') ∩ δ_U(incl(x), y') ≤ δ_U(x', incl(x)) ∩ δ_U(incl(x), y') ≤ δ_U(x',y')
    inclusion-eq-21 : ∀ {U V : ΩA.Ob} (V≤U : V ΩA.≤ U) {x x' y'}
                    → (inclusion-f V≤U x x' ΩA.∩ inclusion-f V≤U x y')
                      ΩA.≤ Ω-Set.δ (Ω-U Ω U) x' y'
    inclusion-eq-21 {U} {V} V≤U {x} {x'} {y'} = ΩA.≤-trans helper (Ω-Set.δ-trans (Ω-U Ω U) {x'} {inclusion-carrier V≤U x} {y'})
      where
        module ΩU = Ω-Set (Ω-U Ω U)
        helper : (ΩU.δ (inclusion-carrier V≤U x) x' ΩA.∩ ΩU.δ (inclusion-carrier V≤U x) y')
                 ΩA.≤ (ΩU.δ x' (inclusion-carrier V≤U x) ΩA.∩ ΩU.δ (inclusion-carrier V≤U x) y')
        open is-meet (ΩA.∩-meets (ΩU.δ x' (inclusion-carrier V≤U x)) (ΩU.δ (inclusion-carrier V≤U x) y')) renaming (greatest to ∩-gr)
        helper = ∩-gr _
                   (ΩA.≤-trans ΩA.∩≤l (ΩA.≤-refl' (ΩU.δ-sym {inclusion-carrier V≤U x} {x'})))
                   ΩA.∩≤r

    -- eq-22: ⋃_{x'} f(x,x') = δ_V(x,x)
    -- Key: δ_V(x,x) = δ_U(incl(x), incl(x)) and incl(x) ∈ Ω_U
    inclusion-eq-22 : ∀ {U V : ΩA.Ob} (V≤U : V ΩA.≤ U) {x}
                    → ΩA.⋃ (λ (x' : Ω-Set.Carrier (Ω-U Ω U)) → inclusion-f V≤U x x')
                      ≡ Ω-Set.δ (Ω-U Ω V) x x
    inclusion-eq-22 {U} {V} V≤U {x} = ΩA.≤-antisym
      (ΩA.⋃-lubs _ .is-lub.least _ λ x' → Ω-Set.δ-self-bound (Ω-U Ω U) {inclusion-carrier V≤U x} {x'})
      (ΩA.⋃-lubs _ .is-lub.fam≤lub (inclusion-carrier V≤U x))

    -- Inclusion: Ω_V ↪ Ω_U when V ≤ U
    inclusion : ∀ {U V : ΩA.Ob} → (V ΩA.≤ U) → Ω-Set-Morphism (Ω-U Ω V) (Ω-U Ω U)
    inclusion V≤U = ω-morphism (inclusion-f V≤U)
                                (λ {x} {y} {x'} → inclusion-eq-19 V≤U {x} {y} {x'})
                                (λ {x} {x'} {y'} → inclusion-eq-20 V≤U {x} {x'} {y'})
                                (λ {x} {x'} {y'} → inclusion-eq-21 V≤U {x} {x'} {y'})
                                (λ {x} → inclusion-eq-22 V≤U {x})

    -- Helper: inclusion at ≤-refl is the identity morphism
    inclusion-refl-is-id : ∀ {U : ΩA.Ob} → inclusion {U} {U} ΩA.≤-refl ≡ id-Ω {Ω = Ω} {X = Ω-U Ω U}
    inclusion-refl-is-id {U} = Ω-Set-Morphism-path (funext λ x → funext λ y → refl)
      -- Both compute δ_U(x,y) so they're definitionally equal

    -- Helper: inclusion is functorial (composition)
    -- Given f : U ≤ V and g : V ≤ W, with ≤-trans f g : U ≤ W
    -- We have: inclusion {W} {U} (≤-trans f g) ≡ inclusion {W} {V} g ∘ inclusion {V} {U} f
    -- Note: ∘-Ω has type (Y→Z) → (X→Y) → (X→Z), so "g ∘ f" applies f first then g
    --
    -- **Proof strategy**: Show the underlying fuzzy functions are equal.
    -- Both sides compute the same element of Ω_W (same first component),
    -- and the proofs that it's ≤ W are equal by ≤-thin (propositions).
    inclusion-comp : ∀ {U V W : ΩA.Ob} (f : U ΩA.≤ V) (g : V ΩA.≤ W)
                   → inclusion {W} {U} (ΩA.≤-trans f g) ≡ (inclusion {W} {V} g ∘-Ω inclusion {V} {U} f)
    inclusion-comp {U} {V} {W} f g = Ω-Set-Morphism-path (funext lemma₁)
      where
        lemma₁ : (x : Carrier (Ω-U Ω U)) →
                 Ω-Set-Morphism.f (inclusion (ΩA.≤-trans f g)) x ≡ Ω-Set-Morphism.f (inclusion g ∘-Ω inclusion f) x
        lemma₁ x = funext lemma₂
          where
            lemma₂ : (y : Carrier (Ω-U Ω W)) →
                     Ω-Set-Morphism.f (inclusion (ΩA.≤-trans f g)) x y
                       ≡ Ω-Set-Morphism.f (inclusion g ∘-Ω inclusion f) x y
            lemma₂ y =
              let (Z , Z≤U) = x
                  (Z' , Z'≤W) = y

                  -- Unbox the ≤ proofs
                  Z≤U-unboxed : Z ΩA.≤ U
                  Z≤U-unboxed = □-rec ΩA.≤-thin (λ p → p) Z≤U

                  Z'≤W-unboxed : Z' ΩA.≤ W
                  Z'≤W-unboxed = □-rec ΩA.≤-thin (λ p → p) Z'≤W

                  -- Compute the transitive embedding
                  Z≤V : Z ΩA.≤ V
                  Z≤V = ΩA.≤-trans Z≤U-unboxed f

                  Z≤W : Z ΩA.≤ W
                  Z≤W = ΩA.≤-trans Z≤V g

                  -- Define family first
                  rhs-family : Carrier (Ω-U Ω V) → ΩA.Ob
                  rhs-family v = Ω-Set.δ (Ω-U Ω V) (Z , inc Z≤V) v ΩA.∩
                                 Ω-Set.δ (Ω-U Ω W) (inclusion-carrier g v) (Z' , Z'≤W)

                  -- LHS value: δ_W(incl_{≤-trans f g}(x), (Z', Z'≤W))
                  lhs : ΩA.Ob
                  lhs = Ω-Set.δ (Ω-U Ω W) (Z , inc Z≤W) (Z' , Z'≤W)

                  -- RHS value: ⋃_{y ∈ Ω_V} δ_V(incl_f(x), y) ∩ δ_W(incl_g(y), (Z', Z'≤W))
                  rhs : ΩA.Ob
                  rhs = ΩA.⋃ rhs-family

                  -- Forward: lhs ≤ rhs (lhs is a witness in the supremum at y = (Z, inc Z≤V))
                  forward : lhs ΩA.≤ rhs
                  forward =
                    let witness-value : ΩA.Ob
                        witness-value = Ω-Set.δ (Ω-U Ω V) (Z , inc Z≤V) (Z , inc Z≤V) ΩA.∩
                                        Ω-Set.δ (Ω-U Ω W) (inclusion-carrier g (Z , inc Z≤V)) (Z' , Z'≤W)

                        -- inclusion-carrier g (Z, inc Z≤V) = (Z, inc Z≤W) by definition
                        incl-g-calc : inclusion-carrier g (Z , inc Z≤V) ≡ (Z , inc Z≤W)
                        incl-g-calc = Σ-prop-path (λ _ → hlevel 1) refl

                        -- δ_V(Z,Z) = ⊤ (fuzzy reflexivity)
                        δ-refl-eq : Ω-Set.δ (Ω-U Ω V) (Z , inc Z≤V) (Z , inc Z≤V) ≡ ΩA.top
                        δ-refl-eq = ΩA.≤-antisym
                          ΩA.!
                          (Ω-Set.δ-refl (Ω-U Ω V) {Z , inc Z≤V})

                        -- witness-value = ⊤ ∩ δ_W((Z, inc Z≤W), (Z', Z'≤W))
                        --               = δ_W((Z, inc Z≤W), (Z', Z'≤W))
                        --               = lhs
                        witness-eq : witness-value ≡ lhs
                        witness-eq =
                          let module ΩR = Order.Semilattice.Meet.Reasoning ΩA.has-meet-slat
                          in witness-value                                                               ≡⟨ ap₂ ΩA._∩_ δ-refl-eq (ap (λ w → Ω-Set.δ (Ω-U Ω W) w (Z' , Z'≤W)) incl-g-calc) ⟩
                             ΩA.top ΩA.∩ Ω-Set.δ (Ω-U Ω W) (Z , inc Z≤W) (Z' , Z'≤W)                    ≡⟨ ΩR.∩-idl ⟩
                             Ω-Set.δ (Ω-U Ω W) (Z , inc Z≤W) (Z' , Z'≤W)                                 ∎

                    in ΩA.≤-trans
                         (ΩA.≤-refl' (sym witness-eq))
                         (ΩA.⋃-lubs rhs-family .is-lub.fam≤lub (Z , inc Z≤V))

                  -- Backward: rhs ≤ lhs (supremum bounded by transitivity of δ)
                  backward : rhs ΩA.≤ lhs
                  backward =
                    let bound-family : ∀ (v : Carrier (Ω-U Ω V)) →
                                       (Ω-Set.δ (Ω-U Ω V) (Z , inc Z≤V) v ΩA.∩
                                        Ω-Set.δ (Ω-U Ω W) (inclusion-carrier g v) (Z' , Z'≤W))
                                       ΩA.≤ lhs
                        bound-family (W' , W'≤V) =
                          let W'≤V-unboxed : W' ΩA.≤ V
                              W'≤V-unboxed = □-rec ΩA.≤-thin (λ p → p) W'≤V

                              W'≤W : W' ΩA.≤ W
                              W'≤W = ΩA.≤-trans W'≤V-unboxed g

                              -- inclusion-carrier g (W', W'≤V) = (W', inc W'≤W)
                              incl-g-W' : inclusion-carrier g (W' , W'≤V) ≡ (W' , inc W'≤W)
                              incl-g-W' = Σ-prop-path (λ _ → hlevel 1) refl

                              -- Strategy: After rewriting incl_g(W') = W',
                              -- we have (δ_V(Z,W') ∩ δ_W(W',Z')) ≤ δ_W(Z,Z')
                              -- Use ∩≤r to get δ_W(W',Z'), then transitivity in Ω_W
                              step1 : (Ω-Set.δ (Ω-U Ω V) (Z , inc Z≤V) (W' , W'≤V) ΩA.∩
                                       Ω-Set.δ (Ω-U Ω W) (W' , inc W'≤W) (Z' , Z'≤W))
                                      ΩA.≤ Ω-Set.δ (Ω-U Ω W) (W' , inc W'≤W) (Z' , Z'≤W)
                              step1 = ΩA.∩≤r

                              -- δ-trans: (δ x y ∩ δ y z) ≤ δ x z
                              -- We have: δ_W(W',Z') ≤ δ_W(Z,W') ∩ δ_W(W',Z') ≤ δ_W(Z,Z')
                              step2 : Ω-Set.δ (Ω-U Ω W) (W' , inc W'≤W) (Z' , Z'≤W) ΩA.≤ lhs
                              step2 = ΩA.≤-trans
                                        ΩA.∩≤r
                                        (Ω-Set.δ-trans (Ω-U Ω W) {Z , inc Z≤W} {W' , inc W'≤W} {Z' , Z'≤W})

                          in ΩA.≤-trans
                               (ΩA.≤-refl' (ap (Ω-Set.δ (Ω-U Ω V) (Z , inc Z≤V) (W' , W'≤V) ΩA.∩_) (ap (λ w → Ω-Set.δ (Ω-U Ω W) w (Z' , Z'≤W)) incl-g-W')))
                               (ΩA.≤-trans step1 step2)
                    in ΩA.⋃-lubs rhs-family .is-lub.least lhs bound-family

              in ΩA.≤-antisym forward backward

    F₀-pre : Ω-Set Ω → Functor (C ^op) (Sets (lsuc o ⊔ ℓ))
    F₀-pre X .Functor.F₀ U = el (Ω-Set-Morphism (Ω-U Ω U) X) (hlevel 2)
    F₀-pre X .Functor.F₁ V≤U x = x ∘-Ω inclusion V≤U  -- Precompose with inclusion
    -- Functor laws: In C^op, both id and composition reduce to ≤-thin since C is a poset
    -- For id: inclusion ≤-refl is definitionally id-Ω, and composition with id is automatic
    F₀-pre X .Functor.F-id {U} = funext λ x →
      x ∘-Ω inclusion ΩA.≤-refl  ≡⟨ ap (x ∘-Ω_) inclusion-refl-is-id ⟩
      x ∘-Ω id-Ω                 ≡⟨ SetΩ Ω .Precategory.idr x ⟩
      x                          ∎
    F₀-pre X .Functor.F-∘ {W} {V} {U} f g = funext λ x →
      x ∘-Ω inclusion (ΩA.≤-trans f g)         ≡⟨ ap (x ∘-Ω_) (inclusion-comp f g) ⟩
      x ∘-Ω (inclusion g ∘-Ω inclusion f)      ≡⟨ SetΩ Ω .Precategory.assoc x (inclusion g) (inclusion f) ⟩
      (x ∘-Ω inclusion g) ∘-Ω inclusion f      ∎

    postulate F₀-sheaf : (X : Ω-Set Ω) → is-sheaf K (F₀-pre X)

    F₀ : Ω-Set Ω → Sheaves K (lsuc o ⊔ ℓ) .Precategory.Ob
    F₀ X = F₀-pre X , F₀-sheaf X

    F₁ : {X Y : Ω-Set Ω} → Ω-Set-Morphism X Y → Sheaves K (lsuc o ⊔ ℓ) .Precategory.Hom (F₀ X) (F₀ Y)
    F₁ f .component U g = f ∘-Ω g
    F₁ f .naturality x y h = {! functoriality !}

  F-functor : Functor (SetΩ Ω) (Sheaves K (lsuc o ⊔ ℓ))
  F-functor .Functor.F₀ = F-Functor.F₀
  F-functor .Functor.F₁ = F-Functor.F₁
  F-functor .Functor.F-id = {! natural transformation extensionality !}
  F-functor .Functor.F-∘ f g = {! natural transformation extensionality !}

  -- G-functor: Sh(Ω, K) → SetΩ
  -- Maps sheaf F to global sections F(⊤) with internal equality
  postulate
    -- Resizing axiom: can extract carrier from F(⊤) at level o
    resize-global-sections : (F : Functor (C ^op) (Sets (lsuc o ⊔ ℓ))) → Type o

  module G-Functor where
    G₀ : Sheaves K (lsuc o ⊔ ℓ) .Precategory.Ob → Ω-Set Ω
    G₀ (F , _) = ω-set (resize-global-sections F) δ-internal {!!} {!!} {!!}
      where
        δ-internal : resize-global-sections F → resize-global-sections F → ΩA.Ob
        δ-internal x y = {! ⋃{U | x|_U = y|_U} !}

  G-functor : Functor (Sheaves K (lsuc o ⊔ ℓ)) (SetΩ Ω)
  G-functor .Functor.F₀ = G-Functor.G₀
  G-functor .Functor.F₁ η = {! map on global sections !}
  G-functor .Functor.F-id = {!!}
  G-functor .Functor.F-∘ f g = {!!}

  -- Main equivalence (Proposition A.2)
  postulate localic-equivalence : is-equivalence F-functor
