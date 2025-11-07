{-# OPTIONS --no-import-sorts #-}
{-|
Stack Foundations
Core stack definitions, Grothendieck construction, fibrations
-}
module Neural.Stack.Base where

open import 1Lab.Prelude hiding (_+_)
open import 1Lab.HLevel
open import 1Lab.HLevel.Closure
open import 1Lab.Path
open import 1Lab.Equiv

open import Cat.Base
open import Cat.Reasoning
open import Cat.Functor.Base
open import Cat.Functor.Compose
open import Cat.Functor.Reasoning
open import Cat.Functor.Equivalence
open import Cat.Functor.Naturality
open import Cat.Instances.Functor
open import Cat.Instances.Sets
open import Cat.Instances.Shape.Terminal
open import Cat.Bi.Base
open import Cat.Bi.Instances.Discrete

open import Neural.Stack.Reasoning

private variable
  o ℓ o' ℓ' o'' ℓ'' : Level

--------------------------------------------------------------------------------
-- Part 2: Stacks & Fibrations
--------------------------------------------------------------------------------

-- Stack = Pseudofunctor C^op → Cat
-- Using 1Lab's bicategorical infrastructure for full coherence
Stack : Precategory o ℓ → ∀ o' ℓ' → Type _
Stack {o} {ℓ} C o' ℓ' = Pseudofunctor (Locally-discrete (C ^op)) (Cat o' ℓ')

module _ {C : Precategory o ℓ} where
  private
    module C = Precategory C
  open _=>_ -- For .η accessor on natural transformations

  -- Evaluation at object
  _ʻˢ_ : Stack C o' ℓ' → C.Ob → Precategory o' ℓ'
  F ʻˢ U = Pseudofunctor.₀ F U

  -- Reindexing functor along morphism α : U → V
  -- F(α) : F(V) → F(U) (contravariant because of ^op)
  _⟪ˢ_⟫ : (F : Stack C o' ℓ') → {U V : C.Ob}
        → C.Hom U V → Functor (F ʻˢ V) (F ʻˢ U)
  F ⟪ˢ α ⟫ = Pseudofunctor.₁ F α

  -- Identity coherence: Natural transformation Id => F(id)
  -- In Cat, this witnesses that F(id) is naturally isomorphic to Id
  ⟪ˢ-υ→ : (F : Stack C o' ℓ') {U : C.Ob}
        → Id {C = F ʻˢ U} => (F ⟪ˢ C.id {U} ⟫)
  ⟪ˢ-υ→ F {U} = Pseudofunctor.unitor F {U}

  -- Inverse: F(id) => Id
  ⟪ˢ-υ← : (F : Stack C o' ℓ') {U : C.Ob}
        → (F ⟪ˢ C.id {U} ⟫) => Id {C = F ʻˢ U}
  ⟪ˢ-υ← F {U} = Pseudofunctor.υ← F {U}

  -- Composition coherence: Natural transformation (F α ∘ F β) => F(β∘α)
  -- Note: order reversed because of contravariance (C^op)
  ⟪ˢ-γ→ : (F : Stack C o' ℓ') {U V W : C.Ob}
        → (β : C.Hom V W) (α : C.Hom U V)
        → ((F ⟪ˢ α ⟫) F∘ (F ⟪ˢ β ⟫)) => (F ⟪ˢ (C._∘_ β α) ⟫)
  ⟪ˢ-γ→ F β α = Pseudofunctor.γ→ F α β

  -- Inverse: F(β∘α) => (F α ∘ F β)
  ⟪ˢ-γ← : (F : Stack C o' ℓ') {U V W : C.Ob}
        → (β : C.Hom V W) (α : C.Hom U V)
        → (F ⟪ˢ (C._∘_ β α) ⟫) => ((F ⟪ˢ α ⟫) F∘ (F ⟪ˢ β ⟫))
  ⟪ˢ-γ← F β α = Pseudofunctor.γ← F α β

  -- NOTE: Constant and Codiscrete stacks moved to Neural.Stack.Constant
  -- Import that module when needed (avoiding circular dependency)

  -- TODO: Discrete stack construction (requires initial objects or alternate formulation)

  -- Base change: pullback of stack along functor
  -- Given F : Stack C (pseudofunctor C^op → Cat) and G : Functor D → C
  -- Produce: Stack D (pseudofunctor D^op → Cat) by pre-composing with G^op
  --
  -- TODO: This requires careful handling of:
  -- 1. Disc-adjunct for P₁ (converting morphism families to functors)
  -- 2. Compositor naturality in locally-discrete bicategories
  -- 3. Coherence laws inherited from F via G
  --
  -- The implementation involves composing F : C^op → Cat with G^op : D^op → C^op
  -- In locally-discrete bicategories, morphisms are paths, making this delicate.
  postulate
    _★ : {D : Precategory o'' ℓ''} (G : Functor D C) → Stack C o' ℓ' → Stack D o' ℓ'

  -- Properties of base change (TODO: prove once _★ is implemented)
  postulate
    base-change : {D : Precategory o'' ℓ''} (G : Functor D C)
                → (F : Stack C o' ℓ')
                → ((G ★) F ʻˢ_) ≡ (λ U → F ʻˢ (G .Functor.F₀ U))

    base-change-∘ : {D : Precategory o'' ℓ''} {E : Precategory lzero lzero}
                  → (H : Functor E D) (G : Functor D C)
                  → (F : Stack C o' ℓ')
                  → ((G ★) F ʻˢ_) ≡ (((G F∘ H) ★) F ʻˢ_)


  -- Grothendieck construction coherence helpers
  -- These use Pseudofunctor coherence (right-unit, left-unit, hexagon) to prove category laws
  private
    ∫ˢ-idr-coh : (F : Stack C o' ℓ') {U V : C.Ob}
               → (α : C.Hom U V)
               → (ξ : Precategory.Ob (F ʻˢ U)) (η : Precategory.Ob (F ʻˢ V))
               → (f : Precategory.Hom (F ʻˢ U) ξ ((F ⟪ˢ α ⟫) .Functor.F₀ η))
               → PathP (λ i → Precategory.Hom (F ʻˢ U) ξ (Functor.F₀ (F ⟪ˢ C.idr α i ⟫) η))
                   (Precategory._∘_ (F ʻˢ U) (_=>_.η (⟪ˢ-γ→ F α C.id) η)
                     (Precategory._∘_ (F ʻˢ U) (Functor.F₁ (F ⟪ˢ C.id ⟫) f) (_=>_.η (⟪ˢ-υ→ F) ξ)))
                   f
    ∫ˢ-idr-coh F {U} {V} α ξ η f = to-pathp helper
      where
        open Precategory (F ʻˢ U) renaming (_∘_ to _∘U_; idr to idrU; Hom to HomU)

        -- At i0: α ∘ id, with complex composition in fiber
        -- At i1: α, with just f
        -- Need to show they're equal after transport

        -- The LHS composition: γ→(α,id)(η) ∘ F⟪ˢid⟫.F₁(f) ∘ υ→(ξ)
        lhs : HomU ξ (Functor.F₀ (F ⟪ˢ (C._∘_ α C.id) ⟫) η)
        lhs = _∘U_ (_=>_.η (⟪ˢ-γ→ F α C.id) η)
                   (_∘U_ (Functor.F₁ (F ⟪ˢ C.id ⟫) f) (_=>_.η (⟪ˢ-υ→ F) ξ))

        -- After transport along C.idr α, should equal f
        -- Key: Use naturality of compositor + unitor + functoriality
        helper : subst (λ α' → HomU ξ (Functor.F₀ (F ⟪ˢ α' ⟫) η)) (C.idr α) lhs ≡ f
        helper = {!!}  -- TODO: Use Pseudofunctor.right-unit coherence

    ∫ˢ-idl-coh : (F : Stack C o' ℓ') {U V : C.Ob}
               → (α : C.Hom U V)
               → (ξ : Precategory.Ob (F ʻˢ U)) (η : Precategory.Ob (F ʻˢ V))
               → (f : Precategory.Hom (F ʻˢ U) ξ ((F ⟪ˢ α ⟫) .Functor.F₀ η))
               → PathP (λ i → Precategory.Hom (F ʻˢ U) ξ (Functor.F₀ (F ⟪ˢ C.idl α i ⟫) η))
                   (Precategory._∘_ (F ʻˢ U) (_=>_.η (⟪ˢ-γ→ F C.id α) η)
                     (Precategory._∘_ (F ʻˢ U) (Functor.F₁ (F ⟪ˢ α ⟫) (_=>_.η (⟪ˢ-υ→ F) η)) f))
                   f
    ∫ˢ-idl-coh F {U} {V} α ξ η f = to-pathp helper
      where
        open Precategory (F ʻˢ U) renaming (_∘_ to _∘U_; idl to idlU; Hom to HomU)

        -- The LHS composition: γ→(id,α)(η) ∘ F⟪ˢα⟫.F₁(υ→(η)) ∘ f
        lhs : HomU ξ (Functor.F₀ (F ⟪ˢ (C._∘_ C.id α) ⟫) η)
        lhs = _∘U_ (_=>_.η (⟪ˢ-γ→ F C.id α) η)
                   (_∘U_ (Functor.F₁ (F ⟪ˢ α ⟫) (_=>_.η (⟪ˢ-υ→ F) η)) f)

        -- After transport along C.idl α, should equal f
        helper : subst (λ α' → HomU ξ (Functor.F₀ (F ⟪ˢ α' ⟫) η)) (C.idl α) lhs ≡ f
        helper = {!!}  -- TODO: Use Pseudofunctor.left-unit coherence

    ∫ˢ-assoc-coh : (F : Stack C o' ℓ') {U V W X : C.Ob}
                 → {ξ : Precategory.Ob (F ʻˢ U)} {η : Precategory.Ob (F ʻˢ V)}
                 → {ζ : Precategory.Ob (F ʻˢ W)} {θ : Precategory.Ob (F ʻˢ X)}
                 → (γ : C.Hom W X) (h : Precategory.Hom (F ʻˢ W) ζ ((F ⟪ˢ γ ⟫) .Functor.F₀ θ))
                 → (β : C.Hom V W) (g : Precategory.Hom (F ʻˢ V) η ((F ⟪ˢ β ⟫) .Functor.F₀ ζ))
                 → (α : C.Hom U V) (f : Precategory.Hom (F ʻˢ U) ξ ((F ⟪ˢ α ⟫) .Functor.F₀ η))
                 → PathP (λ i → Precategory.Hom (F ʻˢ U) ξ (Functor.F₀ (F ⟪ˢ C.assoc γ β α i ⟫) θ))
                     (Precategory._∘_ (F ʻˢ U) (_=>_.η (⟪ˢ-γ→ F γ (C._∘_ β α)) θ)
                       (Precategory._∘_ (F ʻˢ U) (Functor.F₁ (F ⟪ˢ C._∘_ β α ⟫) h)
                         (Precategory._∘_ (F ʻˢ U) (_=>_.η (⟪ˢ-γ→ F β α) ζ)
                           (Precategory._∘_ (F ʻˢ U) (Functor.F₁ (F ⟪ˢ α ⟫) g) f))))
                     (Precategory._∘_ (F ʻˢ U) (_=>_.η (⟪ˢ-γ→ F (C._∘_ γ β) α) θ)
                       (Precategory._∘_ (F ʻˢ U) (Functor.F₁ (F ⟪ˢ α ⟫)
                         (Precategory._∘_ (F ʻˢ V) (_=>_.η (⟪ˢ-γ→ F γ β) θ)
                           (Precategory._∘_ (F ʻˢ V) (Functor.F₁ (F ⟪ˢ β ⟫) h) g)))
                       f))
    ∫ˢ-assoc-coh F {U} {V} {W} {X} {ξ} {η} {ζ} {θ} γ h β g α f = {!!}

  -- Grothendieck construction: total category ∫F
  -- Uses Pseudofunctor coherence (compositor, unitor) for category laws
  ∫ˢ : Stack C o' ℓ' → Precategory (o ⊔ o') (ℓ ⊔ ℓ')
  ∫ˢ F = record
    { Ob = Σ[ U ∈ C.Ob ] Precategory.Ob (F ʻˢ U)
    ; Hom = λ (U , ξ) (V , η) → Σ[ α ∈ C.Hom U V ] Precategory.Hom (F ʻˢ U) ξ ((F ⟪ˢ α ⟫) .Functor.F₀ η)
    ; Hom-set = λ (U , ξ) (V , η) → hlevel 2
    ; id = λ {(U , ξ)} → _,_ C.id (_=>_.η (⟪ˢ-υ→ F {U}) ξ)
    ; _∘_ = λ {(U , ξ)} {(V , η)} {(W , ζ)} (β , g) (α , f) →
        _,_ (C._∘_ β α)
            (Precategory._∘_ (F ʻˢ U) (_=>_.η (⟪ˢ-γ→ F β α) ζ)
              (Precategory._∘_ (F ʻˢ U) ((F ⟪ˢ α ⟫) .Functor.F₁ g) f))
    ; idr = λ {(U , ξ)} {(V , η)} (α , f) →
        Σ-pathp (C.idr α) (∫ˢ-idr-coh F α ξ η f)
    ; idl = λ {(U , ξ)} {(V , η)} (α , f) →
        Σ-pathp (C.idl α) (∫ˢ-idl-coh F α ξ η f)
    ; assoc = λ {(U , ξ)} {(V , η)} {(W , ζ)} {(X , θ)} (γ , h) (β , g) (α , f) →
        Σ-pathp (C.assoc γ β α) (∫ˢ-assoc-coh F γ h β g α f)
    }

  -- ∫ commutes with base change (TODO: state properly)
  -- ∫ˢ ((G ★) F) should be equivalent to pullback of ∫ˢ F along G

  -- Vertical morphism: morphism in same fiber
  is-vertical : {F : Stack C o' ℓ'} {U : C.Ob}
              → {ξ η : Precategory.Ob (F ʻˢ U)}
              → Precategory.Hom (∫ˢ F) (U , ξ) (U , η)
              → Type ℓ
  is-vertical {F} {U} {ξ} {η} (α , f) = α ≡ C.id

  -- Horizontal morphism: identity in fiber
  is-horizontal : {F : Stack C o' ℓ'} {x y : Precategory.Ob (∫ˢ F)}
                → Precategory.Hom (∫ˢ F) x y → Type _
  is-horizontal {F = F} {U , ξ} {V , η} (α , f) =
    Σ[ p ∈ ((F ⟪ˢ α ⟫) .Functor.F₀ η ≡ ξ) ]
      (subst (λ ζ → Precategory.Hom (F ʻˢ U) ξ ζ) p f ≡ Precategory.id (F ʻˢ U))

  {- Commented out due to type error - needs ⟪ˢ-id and ⟪ˢ-∘ filled first
  -- Every morphism factors through vertical + horizontal
  factor-vh : {F : Stack C o' ℓ'} {x y : Precategory.Ob (∫ˢ F)}
            → (f : Precategory.Hom (∫ˢ F) x y)
            → Σ[ h ∈ Precategory.Hom (∫ˢ F) x _ ]
              Σ[ v ∈ Precategory.Hom (∫ˢ F) _ y ]
                (is-horizontal h × is-vertical v × (Precategory._∘_ (∫ˢ F) v h ≡ f))
  factor-vh {F = F} {U , ξ} {V , η} (α , f) = {!!}
  -}

  -- Projection functor π : ∫F → C
  πˢ : (F : Stack C o' ℓ') → Functor (∫ˢ F) C
  πˢ F .Functor.F₀ (U , ξ) = U
  πˢ F .Functor.F₁ (α , f) = α
  πˢ F .Functor.F-id = refl
  πˢ F .Functor.F-∘ _ _ = refl

  -- π is split if and only if there exists global section
  π-split : (F : Stack C o' ℓ')
          → (Σ[ s ∈ Functor C (∫ˢ F) ] ((πˢ F) F∘ s ≡ Id))
          → ∀ U → Precategory.Ob (F ʻˢ U)
  π-split F (s , p) U =
    subst (λ V → Precategory.Ob (F ʻˢ V))
          (ap (λ φ → φ .Functor.F₀ U) p)
          (snd (s .Functor.F₀ U))

  -- Fiber over U: subcategory of ∫F
  Fiber : (F : Stack C o' ℓ') (U : C.Ob) → Precategory o' ℓ'
  Fiber F U .Precategory.Ob = Precategory.Ob (F ʻˢ U)
  Fiber F U .Precategory.Hom ξ η = Precategory.Hom (F ʻˢ U) ξ η
  Fiber F U .Precategory.Hom-set ξ η = Precategory.Hom-set (F ʻˢ U) ξ η
  Fiber F U .Precategory.id = Precategory.id (F ʻˢ U)
  Fiber F U .Precategory._∘_ = Precategory._∘_ (F ʻˢ U)
  Fiber F U .Precategory.idr = Precategory.idr (F ʻˢ U)
  Fiber F U .Precategory.idl = Precategory.idl (F ʻˢ U)
  Fiber F U .Precategory.assoc = Precategory.assoc (F ʻˢ U)

  -- Inclusion of fiber into total category
  ι : (F : Stack C o' ℓ') (U : C.Ob) → Functor (Fiber F U) (∫ˢ F)
  ι F U .Functor.F₀ ξ = (U , ξ)
  ι F U .Functor.F₁ {ξ} {η} f =
    C.id ,
    -- We need: ξ → F(id).F₀(η)
    -- We have: f : ξ → η
    -- Apply υ→ : Id => F(id), giving υ→.η(η) : η → F(id).F₀(η)
    Precategory._∘_ (F ʻˢ U) (_=>_.η (⟪ˢ-υ→ F {U}) η) f
  ι F U .Functor.F-id {ξ} =
    -- Goal: ι.F₁ id ≡ id_∫
    -- LHS: (C.id , υ→.η(ξ) ∘ id_fiber)
    -- RHS: (C.id , υ→.η(ξ))
    -- Use idr in fiber to show υ→.η(ξ) ∘ id_fiber ≡ υ→.η(ξ)
    Σ-pathp refl (Precategory.idr (F ʻˢ U) (_=>_.η (⟪ˢ-υ→ F) ξ))
  ι F U .Functor.F-∘ {ξ} {η} {ζ} f g =
    -- Goal: ι.F₁ (g ∘ f) ≡ ι.F₁ g ∘_∫ ι.F₁ f
    -- LHS: (C.id , υ→.η(ζ) ∘ (g ∘ f))
    -- RHS: (C.id ∘ C.id , γ→(id,id)(ζ) ∘ F(id).F₁(υ→.η(ζ) ∘ g) ∘ (υ→.η(η) ∘ f))
    -- TODO: Use compositor coherence and category laws
    {!!}

  -- Reindexing along α gives functor between fibers
  reindex-fiber : (F : Stack C o' ℓ') {U V : C.Ob}
                → C.Hom U V → Functor (Fiber F V) (Fiber F U)
  reindex-fiber F α .Functor.F₀ = (F ⟪ˢ α ⟫) .Functor.F₀
  reindex-fiber F α .Functor.F₁ = (F ⟪ˢ α ⟫) .Functor.F₁
  reindex-fiber F α .Functor.F-id = (F ⟪ˢ α ⟫) .Functor.F-id
  reindex-fiber F α .Functor.F-∘ f g = (F ⟪ˢ α ⟫) .Functor.F-∘ f g

  -- Section: splitting of π
  StackSection : Stack C o' ℓ' → Type _
  StackSection F = Σ[ s ∈ Functor C (∫ˢ F) ] ((πˢ F) F∘ s ≡ Id)

  -- Global element at U: section of fiber inclusion
  GlobalElement : (F : Stack C o' ℓ') (U : C.Ob) → Type _
  GlobalElement F U = Precategory.Ob (F ʻˢ U)

  -- Morphism of global elements
  GlobalMorphism : (F : Stack C o' ℓ') {U V : C.Ob}
                 → C.Hom U V
                 → GlobalElement F V
                 → GlobalElement F U
  GlobalMorphism F α η = (F ⟪ˢ α ⟫) .Functor.F₀ η

  -- Cartesian morphism: universal lift of base morphism
  -- f : ξ → (F α)(η) over α is cartesian if it has the universal lifting property:
  -- For any β : W → U and g over α∘β, there exists unique h over β factoring g through f
  is-cartesian : {F : Stack C o' ℓ'}
               → {U V : C.Ob}
               → (α : C.Hom U V)
               → {ξ : Precategory.Ob (F ʻˢ U)}
               → {η : Precategory.Ob (F ʻˢ V)}
               → Precategory.Hom (F ʻˢ U) ξ ((F ⟪ˢ α ⟫) .Functor.F₀ η)
               → Type _
  is-cartesian {F = F} {U = U} {V = V} α {ξ = ξ} {η = η} f =
    ∀ {W : C.Ob} (β : C.Hom W U) (ζ : Precategory.Ob (F ʻˢ W))
    → (g : Precategory.Hom (F ʻˢ W) ζ ((F ⟪ˢ (C._∘_ α β) ⟫) .Functor.F₀ η))
    → is-contr (Σ[ h ∈ Precategory.Hom (F ʻˢ W) ζ ((F ⟪ˢ β ⟫) .Functor.F₀ ξ) ]
        let FW = F ʻˢ W
            composed = Precategory._∘_ FW ((F ⟪ˢ β ⟫) .Functor.F₁ f) h
            -- composed : ζ → (F β).F₀ ((F α).F₀ η) = ((F α) ∘ (F β)).F₀ η
            -- g : ζ → F(α∘β).F₀ η
            -- Use compositor: γ→ : (F α) ∘ (F β) => F(α∘β)
            -- So γ→.η : ((F α) ∘ (F β)).F₀(η) → F(α∘β).F₀(η)
            transported = Precategory._∘_ FW (_=>_.η (⟪ˢ-γ→ F α β) η) composed
        in transported ≡ g)

  -- Cartesian morphisms compose
  cart-∘ : {F : Stack C o' ℓ'}
         → {U V W : C.Ob}
         → {α : C.Hom U V} {β : C.Hom V W}
         → {ξ : Precategory.Ob (F ʻˢ U)}
         → {η : Precategory.Ob (F ʻˢ V)}
         → {ζ : Precategory.Ob (F ʻˢ W)}
         → (f : Precategory.Hom (F ʻˢ U) ξ ((F ⟪ˢ α ⟫) .Functor.F₀ η))
         → (g : Precategory.Hom (F ʻˢ V) η ((F ⟪ˢ β ⟫) .Functor.F₀ ζ))
         → is-cartesian α f
         → is-cartesian β g
         → is-cartesian (C._∘_ β α)
             -- Composed morphism: ξ → F(β∘α).F₀(ζ)
             -- Composition: γ→(β,α) ∘ F(α).F₁(g) ∘ f
             (Precategory._∘_ (F ʻˢ U) (_=>_.η (⟪ˢ-γ→ F β α) ζ)
               (Precategory._∘_ (F ʻˢ U) ((F ⟪ˢ α ⟫) .Functor.F₁ g) f))
  cart-∘ f g cf cg {X} γ θ h = {!!}

  -- Identity morphism is cartesian
  -- The identity id : ξ → (F id)(ξ) satisfies the universal property trivially
  postulate
    cart-id : (F : Stack C o' ℓ') {U : C.Ob} {ξ : Precategory.Ob (F ʻˢ U)}
            → is-cartesian {F = F} {U = U} {V = U} C.id {ξ = ξ} {η = ξ}
                -- Use υ→: Id => F(id) to get ξ → F(id).F₀(ξ)
                (_=>_.η (⟪ˢ-υ→ F {U}) ξ)

  -- Fibration: all reindexings have cartesian lifts
  is-fibration : Stack C o' ℓ' → Type _
  is-fibration F =
    ∀ {U V} (α : C.Hom U V) (η : Precategory.Ob (F ʻˢ V))
    → Σ[ ξ ∈ Precategory.Ob (F ʻˢ U) ]
      Σ[ f ∈ Precategory.Hom (F ʻˢ U) ξ ((F ⟪ˢ α ⟫) .Functor.F₀ η) ]
        is-cartesian α f

  -- Cleavage: specified choice of cartesian lifts
  record Cleavage (F : Stack C o' ℓ') : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ') where
    field
      lift-obj : {U V : C.Ob} → C.Hom U V → Precategory.Ob (F ʻˢ V)
               → Precategory.Ob (F ʻˢ U)
      lift-mor : {U V : C.Ob} (α : C.Hom U V) (η : Precategory.Ob (F ʻˢ V))
               → Precategory.Hom (F ʻˢ U) (lift-obj α η) ((F ⟪ˢ α ⟫) .Functor.F₀ η)
      is-cart : {U V : C.Ob} (α : C.Hom U V) (η : Precategory.Ob (F ʻˢ V))
              → is-cartesian α (lift-mor α η)

  open Cleavage public

  -- Split fibration: cleavage respects composition strictly
  record SplitFibration (F : Stack C o' ℓ') : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ') where
    field
      cleavage : Cleavage F
      split-id : {U : C.Ob} {η : Precategory.Ob (F ʻˢ U)}
               → cleavage .lift-obj C.id η ≡ η
      split-∘ : {U V W : C.Ob}
              → (β : C.Hom V W) (α : C.Hom U V)
              → (ζ : Precategory.Ob (F ʻˢ W))
              → cleavage .lift-obj (C._∘_ β α) ζ
              ≡ cleavage .lift-obj α (cleavage .lift-obj β ζ)

  -- Every fibration has a cleavage (by choice)
  -- Use propositional truncation because cleavage requires choice
  fibration→cleavage : (F : Stack C o' ℓ') → is-fibration F → ∥ Cleavage F ∥
  fibration→cleavage F fib = inc record
    { lift-obj = λ {U} {V} α η → fib α η .fst
    ; lift-mor = λ {U} {V} α η → fib α η .snd .fst
    ; is-cart = λ {U} {V} α η → fib α η .snd .snd
    }

  -- Cocartesian morphism: dual to cartesian (pushout property)
  -- For contravariant stacks, the cocartesian property states that
  -- f : ξ → (F α)(η) is universal for factorizations
  is-cocartesian : {F : Stack C o' ℓ'}
                 → {U V : C.Ob}
                 → (α : C.Hom U V)
                 → {ξ : Precategory.Ob (F ʻˢ U)}
                 → {η : Precategory.Ob (F ʻˢ V)}
                 → Precategory.Hom (F ʻˢ U) ξ ((F ⟪ˢ α ⟫) .Functor.F₀ η)
                 → Type _
  is-cocartesian {F = F} {U = U} {V = V} α {ξ = ξ} {η = η} f =
    -- Universal factorization property for morphisms after α in the base
    -- For contravariant F: F(α) : F(V) → F(U)
    -- Given β : V → W and g : ξ → F(β∘α).F₀(ζ), find unique h lying over β
    -- Note: ζ is in fiber W, so F(β).F₀(ζ) is in fiber V, and h : η → F(β).F₀(ζ) in fiber V
    ∀ {W : C.Ob} (β : C.Hom V W) (ζ : Precategory.Ob (F ʻˢ W))
    → (g : Precategory.Hom (F ʻˢ U) ξ ((F ⟪ˢ (C._∘_ β α) ⟫) .Functor.F₀ ζ))
    → is-contr (Σ[ h ∈ Precategory.Hom (F ʻˢ V) η ((F ⟪ˢ β ⟫) .Functor.F₀ ζ) ]
        let FU = F ʻˢ U
            -- Compose: ξ --f--> F(α).F₀(η) --F(α).F₁(h)--> F(α).F₀(F(β).F₀(ζ))
            composed = Precategory._∘_ FU ((F ⟪ˢ α ⟫) .Functor.F₁ h) f
            -- composed : ξ → (F α).F₀ ((F β).F₀ ζ) = ((F β) ∘ (F α)).F₀ ζ
            -- g : ξ → F(β∘α).F₀ ζ
            -- Use compositor: γ→ : (F β) ∘ (F α) => F(β∘α)
            transported = Precategory._∘_ FU (_=>_.η (⟪ˢ-γ→ F β α) ζ) composed
        in transported ≡ g)

  -- Cocartesian lift: object with cocartesian morphism
  record Cocartesian-lift (F : Stack C o' ℓ') {U V : C.Ob} (α : C.Hom U V) (η : Precategory.Ob (F ʻˢ V)) : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ') where
    field
      ξ : Precategory.Ob (F ʻˢ U)
      lifting : Precategory.Hom (F ʻˢ U) ξ ((F ⟪ˢ α ⟫) .Functor.F₀ η)
      cocartesian : is-cocartesian α lifting

  -- Opfibration (Cocartesian fibration): all morphisms have cocartesian lifts
  is-opfibration : Stack C o' ℓ' → Type _
  is-opfibration F =
    ∀ {U V} (α : C.Hom U V) (η : Precategory.Ob (F ʻˢ V))
    → Cocartesian-lift F α η

  -- NOTE: Constant-is-split and Codiscrete-is-split moved to Neural.Stack.Constant

--------------------------------------------------------------------------------
