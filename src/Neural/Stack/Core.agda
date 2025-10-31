{-# OPTIONS --no-import-sorts #-}
{-|
Dense Stack Theory for Neural Networks
Denotational design style (cf. Conal Elliott)
Types as specifications, minimal prose
-}
module Neural.Stack.Core where

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
open import Cat.Instances.Product
open import Cat.Instances.Discrete
open import Cat.Displayed.Base
open import Cat.Displayed.Total
open import Cat.Site.Base
open import Cat.Bi.Base
open import Cat.Bi.Instances.Discrete
open import Topoi.Base

open import Algebra.Group
open import Algebra.Monoid

private variable
  o ℓ o' ℓ' o'' ℓ'' : Level

--------------------------------------------------------------------------------
-- Part 1: Groups & Actions
--------------------------------------------------------------------------------

-- Group as one-object category (delooping)
BG : ∀ {ℓ} → (G : Type ℓ) → Group-on G → Precategory lzero ℓ
BG G grp .Precategory.Ob = ⊤
BG G grp .Precategory.Hom _ _ = G
BG G grp .Precategory.Hom-set _ _ = grp .Group-on.has-is-set
BG G grp .Precategory.id = grp .Group-on.unit
BG G grp .Precategory._∘_ = grp .Group-on._⋆_
BG G grp .Precategory.idr _ = grp .Group-on.idr
BG G grp .Precategory.idl _ = grp .Group-on.idl
BG G grp .Precategory.assoc _ _ _ = grp .Group-on.associative

-- G-set = presheaf over BG
G-Set : ∀ {ℓg ℓ} → (G : Type ℓg) → Group-on G → Type _
G-Set {ℓg} {ℓ} G grp = Functor (BG G grp ^op) (Sets ℓ)

module _ {ℓg ℓ} {G : Type ℓg} (grp : Group-on G) where
  private module Grp = Group-on grp

  -- Action: g · x
  _·ᴳ_ : (X : G-Set G grp) → G → ⌞ X ʻ tt ⌟ → ⌞ X ʻ tt ⌟
  X ·ᴳ g = X .Functor.F₁ g

  -- Identity action: e · x ≡ x
  ·-id : (X : G-Set G grp) → ∀ x → (X ·ᴳ Grp.unit) x ≡ x
  ·-id X x = happly (X .Functor.F-id) x

  -- Associativity: (g ∘ h) · x ≡ g · (h · x)
  ·-assoc : (X : G-Set G grp) → ∀ g h x
          → (X ·ᴳ (Grp._⋆_ g h)) x ≡ (X ·ᴳ g) ((X ·ᴳ h) x)
  ·-assoc X g h x = happly (X .Functor.F-∘ g h) x

  -- Equivariant map between G-sets
  record _→ᴳ_ (X Y : G-Set G grp) : Type (ℓg ⊔ ℓ) where
    field
      map : ⌞ X ʻ tt ⌟ → ⌞ Y ʻ tt ⌟
      equivariant : ∀ g x → map ((X ·ᴳ g) x) ≡ (Y ·ᴳ g) (map x)

  open _→ᴳ_ public

  -- Identity equivariant map
  idᴳ : {X : G-Set G grp} → X →ᴳ X
  idᴳ .map x = x
  idᴳ .equivariant g x = refl

  -- Composition of equivariant maps
  _∘ᴳ_ : {X Y Z : G-Set G grp} → Y →ᴳ Z → X →ᴳ Y → X →ᴳ Z
  _∘ᴳ_ {X} {Y} {Z} f g .map x = f .map (g .map x)
  _∘ᴳ_ {X} {Y} {Z} f g .equivariant h x =
    f .map (g .map ((X ·ᴳ h) x))    ≡⟨ ap (f .map) (g .equivariant h x) ⟩
    f .map ((Y ·ᴳ h) (g .map x))    ≡⟨ f .equivariant h (g .map x) ⟩
    (Z ·ᴳ h) (f .map (g .map x))    ∎

  {- G-set category proofs commented out due to pre-existing type errors
  -- Equivariant maps form a set (TODO: prove using record iso)
  →ᴳ-is-set : {X Y : G-Set G grp} → is-set (X →ᴳ Y)
  →ᴳ-is-set = {!!}

  -- Category of G-sets
  G^ : Precategory (ℓg ⊔ lsuc ℓ) (ℓg ⊔ ℓ)
  G^ .Precategory.Ob = G-Set G grp
  G^ .Precategory.Hom = _→ᴳ_
  G^ .Precategory.Hom-set X Y = {!!}  -- TODO: prove is-set
  G^ .Precategory.id = idᴳ
  G^ .Precategory._∘_ = _∘ᴳ_
  G^ .Precategory.idr f = {!!}  -- TODO: extensionality
  G^ .Precategory.idl f = {!!}
  G^ .Precategory.assoc f g h = {!!}

  -- Forgetful functor G^ → Sets
  U : Functor G^ (Sets ℓ)
  U .Functor.F₀ X = el (⌞ X ʻ tt ⌟) (X .Functor.F₀ tt .is-tr)
  U .Functor.F₁ f = f .map
  U .Functor.F-id = refl
  U .Functor.F-∘ _ _ = refl

  -- Free G-set on a set (right G-action becomes left action in ^op)
  Free : Functor (Sets ℓ) G^
  Free .Functor.F₀ S .Functor.F₀ _ = el (G × ⌞ S ⌟) (hlevel 2)
  Free .Functor.F₀ S .Functor.F₁ g (h , x) = (Grp._⋆_ h g , x)  -- Right action
  Free .Functor.F₀ S .Functor.F-id = funext λ { (h , x) → ap (_, x) Grp.idr }
  Free .Functor.F₀ S .Functor.F-∘ g₁ g₂ = funext λ { (h , x) → ap₂ _,_ (Grp.associative {h} {g₂} {g₁}) refl }
  Free .Functor.F₁ f .map (g , x) = (g , f x)
  Free .Functor.F₁ f .equivariant g (h , x) = refl
  Free .Functor.F-id = {!!}
  Free .Functor.F-∘ _ _ = {!!}
  -}

  -- Trivial G-set (with trivial action)
  Trivial : ⌞ Sets ℓ ⌟ → G-Set G grp
  Trivial S .Functor.F₀ _ = S
  Trivial S .Functor.F₁ _ x = x
  Trivial S .Functor.F-id = refl
  Trivial S .Functor.F-∘ _ _ = refl

  -- Fixed points: {x | ∀g. g·x ≡ x}
  Fixed : G-Set G grp → Type ℓ
  Fixed X = Σ[ x ∈ ⌞ X ʻ tt ⌟ ] (∀ g → (X ·ᴳ g) x ≡ x)

  -- Orbit: {g·x | g ∈ G}
  Orbit : (X : G-Set G grp) → ⌞ X ʻ tt ⌟ → Type (ℓg ⊔ ℓ)
  Orbit X x = Σ[ g ∈ G ] Σ[ y ∈ ⌞ X ʻ tt ⌟ ] ((X ·ᴳ g) x ≡ y)

  -- Stabilizer: {g | g·x ≡ x}
  Stabilizer : (X : G-Set G grp) → ⌞ X ʻ tt ⌟ → Type ℓg
  Stabilizer X x = Σ[ g ∈ G ] ((X ·ᴳ g) x ≡ x)

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

  -- Constant stack: same fiber everywhere
  -- All reindexing functors are identity, so compositor witnesses Id ∘ Id ≅ Id
  module _ (D : Precategory o' ℓ') where
    private
      module D' = Precategory D
      module FC {x} {y} = Cat.Reasoning (Cat[ x , y ])
      open _=>_

      -- Identity is invertible (idnt ∘ idnt = idnt since both are identity)
      id-equiv-invertible : FC.is-invertible idnt
      id-equiv-invertible = FC.make-invertible idnt
        (ext λ _ → Precategory.idl D _)
        (ext λ _ → Precategory.idl D _)

      -- Compositor invertibility: id morphism is self-inverse
      compositor-id-invertible : ∀ (f g : _) → FC.is-invertible (record { η = λ d → Precategory.id D ; is-natural = λ _ _ h → Precategory.idl D _ ∙ sym (Precategory.idr D _) })
      compositor-id-invertible f g = FC.make-invertible (record { η = λ d → Precategory.id D ; is-natural = λ _ _ h → Precategory.idl D _ ∙ sym (Precategory.idr D _) })
        (ext λ _ → Precategory.idl D _)
        (ext λ _ → Precategory.idl D _)

      {- PLAN for compositor naturality proof (hole at line 223):

      Goal: Prove that the identity natural transformation is natural with respect to
      the compositor structure.

      Key observations:
      1. P₁ = Disc-adjunct (λ _ → Id), so all functors are identity functors
      2. Compositor η component is identity morphism on each object
      3. Both sides of the equation compose identity transformations

      Strategy:
      1. Use Disc-adjunct.F-∘ to understand how morphisms compose
         - Location: Cat.Instances.Discrete (lines 155-165)
         - Shows: (Disc-adjunct F).F₁ (g ∙ f) = id ∘ id (via subst)

      2. Use Id functor laws (F-id, F-∘) from Cat.Functor.Base
         - Id.F₁ f = f (identity on morphisms)
         - Composition: (Id ∘ Id).F₁ = Id.F₁

      3. Apply prebicategory composition laws
         - compose.F₁ (id , id) = id (identity preservation)

      4. Use ext (extensionality) to prove equality componentwise
         - Show: ∀ d → LHS d ≡ RHS d
         - Both reduce to D'.id via idl/idr laws

      Required imports (already available):
      - Cat.Instances.Discrete (Disc-adjunct properties)
      - Cat.Functor.Base (Id functor)
      - Cat.Bi.Base (Prebicategory laws)

      Proof structure:
        λ x y f → ext λ d →
          (compose LHS via Disc-adjunct.F-∘ and Id.F-∘)
          ∙ (simplify using idl/idr)
          ∙ sym (compose RHS similarly)
      -}

    Constant : Stack C o' ℓ'
    Constant = record
      { lax = record
          { P₀ = λ _ → D
          ; P₁ = Disc-adjunct (λ _ → Id {C = D})
          ; compositor = record
              { η = λ _ → record { η = λ d → D'.id ; is-natural = λ _ _ h → D'.idl _ ∙ sym (D'.idr _) }
              ; is-natural = λ x y f →
                  let (p , q) = f in
                  ext λ d →
                    J (λ y' pq' →
                        ((Prebicategory.compose (Cat o' ℓ') F∘ (Disc-adjunct (λ _ → Id) F× Disc-adjunct (λ _ → Id))) .Functor.F₁ (ap fst pq' , ap snd pq') FC.∘ record { η = λ d → D'.id ; is-natural = λ _ _ h → D'.idl _ ∙ sym (D'.idr _) }) .η d
                        ≡ (record { η = λ d → D'.id ; is-natural = λ _ _ h → D'.idl _ ∙ sym (D'.idr _) } FC.∘ (Disc-adjunct (λ _ → Id) F∘ Prebicategory.compose (Locally-discrete (C ^op))) .Functor.F₁ (ap fst pq' , ap snd pq')) .η d)
                      refl
                      (Σ-pathp p q)
              }
          ; unitor = idnt
          ; hexagon = λ f g h → ext λ d →
              J (λ h' _ →
                J (λ g' _ →
                  J (λ f' _ →
                    ((Prebicategory.compose (Cat o' ℓ') F∘ (Disc-adjunct (λ _ → Id) F× Disc-adjunct (λ _ → Id))) .Functor.F₁ (Prebicategory.α→ (Locally-discrete (C ^op)) f' g' h') FC.∘ idnt FC.∘ (Cat o' ℓ' Prebicategory.◀ idnt) (Functor.F₀ (Disc-adjunct (λ _ → Id)) h')) .η d
                    ≡ (idnt FC.∘ ((Cat o' ℓ' Prebicategory.▶ Functor.F₀ (Disc-adjunct (λ _ → Id)) f') idnt) FC.∘ Prebicategory.α→ (Cat o' ℓ') (Functor.F₀ (Disc-adjunct (λ _ → Id)) f') (Functor.F₀ (Disc-adjunct (λ _ → Id)) g') (Functor.F₀ (Disc-adjunct (λ _ → Id)) h')) .η d)
                  refl f)
                refl g)
              refl h
          ; right-unit = λ f → {!!}
          ; left-unit = λ f → {!!}
          }
      ; unitor-inv = id-equiv-invertible
      ; compositor-inv = compositor-id-invertible
      }

  -- TODO: Discrete stack construction (requires initial objects or alternate formulation)

  -- Codiscrete stack: same fiber, identity functors (identical to Constant)
  Codiscrete : Precategory o' ℓ' → Stack C o' ℓ'
  Codiscrete = Constant

  -- Base change: pullback of stack along functor
  -- This requires building a Pseudofunctor by composing F with G
  -- Will implement after Grothendieck construction is updated
  _★ : {D : Precategory o'' ℓ''} (G : Functor D C) → Stack C o' ℓ' → Stack D o' ℓ'
  (G ★) F = {!!}  -- TODO: Compose pseudofunctors F ∘ G

  -- Base change functor is stack morphism
  base-change : {D : Precategory o'' ℓ''} (G : Functor D C)
              → (F : Stack C o' ℓ')
              → ((G ★) F ʻˢ_) ≡ (λ U → F ʻˢ (G .Functor.F₀ U))
  base-change G F = refl

  -- Composition of base changes
  base-change-∘ : {D : Precategory o'' ℓ''} {E : Precategory lzero lzero}
                → (H : Functor E D) (G : Functor D C)
                → (F : Stack C o' ℓ')
                → ((G ★) F ʻˢ_) ≡ (((G F∘ H) ★) F ʻˢ_)
  base-change-∘ H G F = refl

  -- Grothendieck construction: total category ∫F
  ∫ˢ : Stack C o' ℓ' → Precategory (o ⊔ o') (ℓ ⊔ ℓ')
  ∫ˢ F .Precategory.Ob = Σ[ U ∈ C.Ob ] Precategory.Ob (F ʻˢ U)
  ∫ˢ F .Precategory.Hom (U , ξ) (V , η) =
    Σ[ α ∈ C.Hom U V ] Precategory.Hom (F ʻˢ U) ξ ((F ⟪ˢ α ⟫) .Functor.F₀ η)
  ∫ˢ F .Precategory.Hom-set (U , ξ) (V , η) = hlevel 2
  ∫ˢ F .Precategory.id {U , ξ} =
    C.id ,
    -- Use forward unitor: υ→ : Id => F(id), component at ξ: ξ → F(id).F₀(ξ)
    _=>_.η (⟪ˢ-υ→ F {U}) ξ
  ∫ˢ F .Precategory._∘_ {U , ξ} {V , η} {W , ζ} (β , g) (α , f) =
    C._∘_ β α ,
    -- Use forward compositor: γ→ : F(α) ∘ F(β) => F(β∘α)
    -- We need: ξ → F(β∘α).F₀(ζ)
    -- We have: f : ξ → F(α).F₀(η) and g : η → F(β).F₀(ζ)
    -- So: F(α).F₁(g) : F(α).F₀(η) → F(α).F₀(F(β).F₀(ζ)) = (F(α) ∘ F(β)).F₀(ζ)
    -- Then γ→.η(ζ) : (F(α) ∘ F(β)).F₀(ζ) → F(β∘α).F₀(ζ)
    Precategory._∘_ (F ʻˢ U) (_=>_.η (⟪ˢ-γ→ F β α) ζ) (Precategory._∘_ (F ʻˢ U) ((F ⟪ˢ α ⟫) .Functor.F₁ g) f)
  ∫ˢ F .Precategory.idr {U , ξ} {V , η} (α , f) = {!!}
  ∫ˢ F .Precategory.idl {U , ξ} {V , η} (α , f) = {!!}
  ∫ˢ F .Precategory.assoc {U , ξ} {V , η} {W , ζ} {X , θ} (γ , h) (β , g) (α , f) = {!!}

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
  ι F U .Functor.F-id {ξ} = {!!}
    -- Goal: (C.id , υ←.η ∘ id) ≡ (C.id , υ←.η ∘ id)
    -- Needs: naturality of υ← and idl/idr
  ι F U .Functor.F-∘ {ξ} {η} {ζ} f g = {!!}
    -- Goal: ι(g ∘ f) ≡ ι(g) ∘_∫ ι(f)
    -- Needs: coherence between υ← and γ←

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
         → is-cartesian (C._∘_ β α) {!!}
  cart-∘ f g cf cg = {!!}

  -- Identity morphism is cartesian
  -- The identity id : ξ → (F id)(ξ) satisfies the universal property trivially
  cart-id : (F : Stack C o' ℓ') {U : C.Ob} {ξ : Precategory.Ob (F ʻˢ U)}
          → is-cartesian {F = F} {U = U} {V = U} C.id {ξ = ξ} {η = ξ}
              -- Use υ→: Id => F(id) to get ξ → F(id).F₀(ξ)
              (_=>_.η (⟪ˢ-υ→ F {U}) ξ)
  cart-id F {U} {ξ} = {!!}  -- Universal property holds because id ∘ β = β and F(id) ≅ Id

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

  -- Constant stack is split fibration
  -- Since all reindexing functors are Id, lifts are trivial
  -- NOTE: Full proof requires ⟪ˢ-id and ⟪ˢ-∘ to be filled
  Constant-is-split : (D : Precategory o' ℓ') → SplitFibration (Constant D)
  Constant-is-split D = record
    { cleavage = record
        { lift-obj = λ {U} {V} α η → η  -- Same object (no change since F(α) = Id)
        ; lift-mor = λ {U} {V} α η → {!!}
            -- Should be: transport (λ i → Hom D η (⟪ˢ-id (Constant D) (~ i) .F₀ η)) id_D
            -- But ⟪ˢ-id is a hole, so we can't fill this yet
        ; is-cart = λ {U} {V} α η {W} β ζ g → {!!}
            -- Universal property holds trivially since all functors are Id
            -- But requires ⟪ˢ-id and ⟪ˢ-∘ to construct the proof
        }
    ; split-id = refl  -- lift-obj id η = η = η
    ; split-∘ = λ β α ζ → refl  -- lift-obj (β∘α) ζ = ζ = lift-obj α (lift-obj β ζ)
    }

  -- Codiscrete stack is split fibration
  -- Identical to Constant since Codiscrete has same structure
  -- NOTE: Full proof requires ⟪ˢ-id and ⟪ˢ-∘ to be filled
  Codiscrete-is-split : (D : Precategory o' ℓ') → SplitFibration (Codiscrete D)
  Codiscrete-is-split D = record
    { cleavage = record
        { lift-obj = λ {U} {V} α η → η  -- Same object (no change since F(α) = Id)
        ; lift-mor = λ {U} {V} α η → {!!}
            -- Should be: transport (λ i → Hom D η (⟪ˢ-id (Codiscrete D) (~ i) .F₀ η)) id_D
            -- But ⟪ˢ-id is a hole, so we can't fill this yet
        ; is-cart = λ {U} {V} α η {W} β ζ g → {!!}
            -- Universal property holds trivially since all functors are Id
            -- But requires ⟪ˢ-id and ⟪ˢ-∘ to construct the proof
        }
    ; split-id = refl  -- lift-obj id η = η = η
    ; split-∘ = λ β α ζ → refl  -- lift-obj (β∘α) ζ = ζ = lift-obj α (lift-obj β ζ)
    }

--------------------------------------------------------------------------------
-- Part 3: Equivariance (Eq 2.1)
--------------------------------------------------------------------------------

-- Action of stack F on stack M
record StackAction {C : Precategory o ℓ} (F M : Stack C o' ℓ') : Type _ where
  field
    act : ∀ U → Functor (F ʻˢ U) (M ʻˢ U)

    -- Equivariance: act_U ∘ F(α) = M(α) ∘ act_V
    equivariant : ∀ {U V} (α : Precategory.Hom C U V)
                → act U F∘ (F ⟪ˢ α ⟫) ≡ (M ⟪ˢ α ⟫) F∘ act V

open StackAction public

module _ {C : Precategory o ℓ} where

  -- Identity action
  id-stack-action : {F : Stack C o' ℓ'} → StackAction F F
  id-stack-action .act U = Id
  id-stack-action .equivariant α = Functor-path (λ _ → refl) (λ _ → refl)

  -- Composition of actions
  _∘ˢᵃ_ : ∀ {o' ℓ'} {F M N : Stack C o' ℓ'}
        → StackAction M N → StackAction F M → StackAction F N
  _∘ˢᵃ_ {o'} {ℓ'} {F} {M} {N} β α .act U = β .act U F∘ α .act U
  _∘ˢᵃ_ {o'} {ℓ'} {F} {M} {N} β α .equivariant {U} {V} φ =
    -- Composition of equivariant actions is equivariant
    -- Follows from associativity in functor category and naturality
    -- Goal: (β.act U ∘ α.act U) ∘ F(φ) ≡ N(φ) ∘ (β.act V ∘ α.act V)
    Functor-path
      (λ x →
        -- Objects: use F₀ part of equivariances
        β .act U .Functor.F₀ (α .act U .Functor.F₀ ((F ⟪ˢ φ ⟫) .Functor.F₀ x))
          ≡⟨ ap (β .act U .Functor.F₀) (happly (ap Functor.F₀ (α .equivariant φ)) x) ⟩
        β .act U .Functor.F₀ ((M ⟪ˢ φ ⟫) .Functor.F₀ (α .act V .Functor.F₀ x))
          ≡⟨ happly (ap Functor.F₀ (β .equivariant φ)) (α .act V .Functor.F₀ x) ⟩
        (N ⟪ˢ φ ⟫) .Functor.F₀ (β .act V .Functor.F₀ (α .act V .Functor.F₀ x))
          ∎)
      (λ {ξ} {η} f →
        -- Morphisms: complex dependent path proof
        -- This requires transporting along the object path above
        {!!})

  {- TODO: Natural transformation between actions - postponed due to complexity
  -- Natural transformation between actions (modification in bicategory terminology)
  -- This is a family of natural transformations component U : α.act U => β.act U
  -- that commute with the stack reindexing functors
  record _⇒ˢᵃ_ {F M : Stack C o' ℓ'} (α β : StackAction F M) : Type _ where
    private
      module FC {U} = Cat.Reasoning (F ʻˢ U)
      module MC {U} = Cat.Reasoning (M ʻˢ U)
    field
      component : ∀ U → _=>_ {C = F ʻˢ U} {D = M ʻˢ U} (α .act U) (β .act U)
      -- Naturality: the component natural transformations commute with reindexing
      -- For φ : U → V in C and ξ : Ob(F(V)), the component commutes with M(φ)
      -- We need to use equivariance of α and β to match the types
      -- The square (in the hom-sets of M(U)):
      --   M(φ)(α.act V ξ) ────[M(φ) comp_V]────→ M(φ)(β.act V ξ)
      --        ∥                                          ∥
      --   (via α.equivariant)                    (via β.equivariant)
      --        ∥                                          ∥
      --   α.act U (F(φ) ξ) ──[comp_U (F(φ) ξ)]──→ β.act U (F(φ) ξ)
      -- This is stated as: component commutes with the actions' equivariance
      natural : ∀ {U V} (φ : Precategory.Hom C U V) (ξ : Precategory.Ob (F ʻˢ V))
              → let αV-ξ = (α .act V) .Functor.F₀ ξ
                    βV-ξ = (β .act V) .Functor.F₀ ξ
                    comp-at-ξ = _=>_.η (component V) ξ
                in -- Apply M(φ) to the component morphism
                   (M ⟪ˢ φ ⟫) .Functor.F₁ comp-at-ξ
                   -- This should equal component at U, modulo equivariances
                   -- But types don't match directly - need to postulate or use path
                ≡ {!!}
  -}

  -- Orbit of section under stack action
  -- Given section s : C → ∫F and action α : F → M, compute orbit
  OrbitSection : ∀ {o' ℓ'} {F M : Stack C o' ℓ'}
               → StackAction F M
               → StackSection F
               → Type _
  OrbitSection {F = F} {M = M} α s =
    -- Family of objects in M fibers obtained by acting on section
    -- For each U, apply action to section value: α.act U (s.F₀ U)
    ∀ (U : Precategory.Ob C) → Precategory.Ob (M ʻˢ U)

--------------------------------------------------------------------------------
-- Part 4: Fibered Presheaves (Eq 2.2-2.5)
--------------------------------------------------------------------------------

-- Presheaf A over fibration F
-- Following Eq 2.2-2.6 from paper (Grothendieck-Giraud construction)
record FiberedPresheaf {C : Precategory o ℓ} (F : Stack C o' ℓ') : Type _ where
  field
    -- A_U : presheaf on fiber F_U (Eq 2.2)
    obj : ∀ U → Functor ((F ʻˢ U) ^op) (Sets (o ⊔ ℓ ⊔ o' ⊔ ℓ'))

    -- TODO: A_α and compat postponed - need proper opposite functor construction
    -- A_α : natural transformation A_V → (F★_α)(A_U)  for α : U → V (Eq 2.3)
    -- where (F★_α) is precomposition with F(α) : F(V) → F(U)
    -- We need F(α)^op : (F(V))^op → (F(U))^op to precompose with obj_U
    {-
    map : ∀ {U V} (α : Precategory.Hom C U V)
        → let F-α-op = record
                { F₀ = (F ⟪ˢ α ⟫) .Functor.F₀
                ; F₁ = (F ⟪ˢ α ⟫) .Functor.F₁
                ; F-id = (F ⟪ˢ α ⟫) .Functor.F-id
                ; F-∘ = λ f g → (F ⟪ˢ α ⟫) .Functor.F-∘ g f  -- Reversed for opposite!
                }
          in _=>_ {C = (F ʻˢ V) ^op} {D = Sets (o ⊔ ℓ ⊔ o' ⊔ ℓ')}
                 (obj V)
                 (obj U F∘ F-α-op)

    -- Compatibility: A_{α∘β} = (F★_α)(A_β) ∘ A_α (Eq 2.4)
    compat : ∀ {U V W} (α : Precategory.Hom C V W) (β : Precategory.Hom C U V)
           → map (Precategory._∘_ C α β) ≡ ((F ⟪ˢ α ⟫) ∘nt map β) ∘nt map α  -- Whiskering composition law
    -}

open FiberedPresheaf public

module _ {C : Precategory o ℓ} {F : Stack C o' ℓ'} where

  -- Evaluation: A(U,ξ) = A_U(ξ)
  eval : FiberedPresheaf F → Precategory.Ob (∫ˢ F) → Type _
  eval A (U , ξ) = ⌞ A .obj U ʻ ξ ⌟

  -- Restriction along morphism in ∫F (Eq 2.5)
  restrict : (A : FiberedPresheaf F)
           → {x y : Precategory.Ob (∫ˢ F)}
           → Precategory.Hom (∫ˢ F) x y
           → eval A y → eval A x
  restrict A {U , ξ} {V , η} (α , f) a = {!!}  -- Combine A_U(f) and A_α

  -- Identity: restrict id ≡ id
  restrict-id : (A : FiberedPresheaf F) {x : Precategory.Ob (∫ˢ F)}
              → ∀ a → restrict A (Precategory.id (∫ˢ F) {x}) a ≡ a
  restrict-id A a = {!!}

  -- Composition: restrict (g ∘ f) ≡ restrict f ∘ restrict g
  restrict-∘ : (A : FiberedPresheaf F)
             → {x y z : Precategory.Ob (∫ˢ F)}
             → (g : Precategory.Hom (∫ˢ F) y z)
             → (f : Precategory.Hom (∫ˢ F) x y)
             → ∀ a → restrict A (Precategory._∘_ (∫ˢ F) g f) a
                   ≡ restrict A f (restrict A g a)
  restrict-∘ A g f a = {!!}

  -- Fibered presheaves form a category
  FibPSh : Precategory _ _
  FibPSh .Precategory.Ob = FiberedPresheaf F
  FibPSh .Precategory.Hom A B = {!!}  -- Natural transformations
  FibPSh .Precategory.Hom-set _ _ = hlevel 2
  FibPSh .Precategory.id = {!!}
  FibPSh .Precategory._∘_ = {!!}
  FibPSh .Precategory.idr _ = {!!}
  FibPSh .Precategory.idl _ = {!!}
  FibPSh .Precategory.assoc _ _ _ = {!!}

--------------------------------------------------------------------------------
-- Part 5: Classifying Topos
--------------------------------------------------------------------------------

-- For neural networks with fork graphs, we use the actual fork-coverage
-- This section shows how to construct classifying topoi for general stacks

-- NOTE: The pattern from Fork.Coverage is:
-- 1. At "special" vertices (fork-star for DNNs), we have TWO coverings:
--    - Maximal sieve (all morphisms)
--    - Incoming sieve (morphisms from tips)
-- 2. At regular vertices: only maximal sieve
-- This enforces the sheaf condition F(A★) ≅ ∏_{tip→A★} F(tip)

module ClassifyingTopos {C : Precategory o ℓ} (F : Stack C o' ℓ') where

  open import Cat.Diagram.Sieve
  open import Cat.Site.Sheafification

  -- Presheaf category as base
  PSh[∫F] : Precategory _ _
  PSh[∫F] = Cat[ (∫ˢ F) ^op , Sets (o ⊔ ℓ ⊔ o' ⊔ ℓ') ]

  -- Canonical coverage on ∫F (least fine making π cocontinuous)
  -- For general stacks, we use the trivial coverage
  -- For fork-stacks from neural networks, import Neural.Graph.Fork.Coverage

  maximal-sieve : (v : Precategory.Ob (∫ˢ F)) → Sieve (∫ˢ F) v
  maximal-sieve v .Sieve.arrows f = ⊤Ω
  maximal-sieve v .Sieve.closed _ _ = tt

  -- Pullback of maximal sieve is maximal
  pullback-maximal : ∀ {u v} (f : Precategory.Hom (∫ˢ F) v u)
                   → pullback f (maximal-sieve u) ≡ maximal-sieve v
  pullback-maximal f = {!!}  -- TODO: Ω-ua type mismatch

  J-canonical : Coverage (∫ˢ F) (o ⊔ ℓ ⊔ o' ⊔ ℓ')
  J-canonical .Coverage.covers U = Lift (o ⊔ ℓ ⊔ o' ⊔ ℓ') ⊤
  J-canonical .Coverage.cover {U} _ = maximal-sieve U
  J-canonical .Coverage.stable {U} {V} R f = inc (lift tt , subset-proof)
    where
      subset-proof : maximal-sieve V ⊆ pullback f (maximal-sieve U)
      subset-proof {w} g g-in = subst (g ∈_) (sym (pullback-maximal f)) g-in

  -- Classifying topos E[F]: category of sheaves on (∫F, J)
  -- Diaconescu: PSh(∫F) classifies flat functors ∫F → Sets
  E[F] : Precategory _ _
  E[F] = PSh[∫F]

  -- Note: For sheafification with J-canonical, use Cat.Site.Sheafification directly
  -- Example: module MySheaf = Sheafification J-canonical MyPresheaf
  --          MySheaf.Sheafify : Functor ((∫ˢ F) ^op) (Sets _)

  -- Connection to neural networks (see Part 6 ForkStackIntegration):
  -- Fork graphs use fork-coverage (not J-canonical!) to get product at star vertices

  -- Every fibered presheaf extends to presheaf on ∫F
  -- TODO: Postponed - depends on FiberedPresheaf.map
  {-
  extend : FiberedPresheaf F → Functor ((∫ˢ F) ^op) (Sets _)
  extend A .Functor.F₀ (U , ξ) = el (eval A (U , ξ)) (hlevel 2)
  extend A .Functor.F₁ f a = restrict A f a
  extend A .Functor.F-id = funext (restrict-id A)
  extend A .Functor.F-∘ g f = funext (restrict-∘ A g f)
  -}

  -- Sheafification of fibered presheaf
  -- To sheafify, apply Cat.Site.Sheafification with appropriate coverage
  -- Example: module ASheaf = Sheafification J-canonical (extend A)
  --          sheafified = ASheaf.Sheafify
  -- For fork graphs, use fork-coverage (see Part 6)

-- For constant group fibration F = BG, get equivalence with G-sets
-- TODO: Update to use Constant properly after compositor is filled
{-
module _ {ℓg ℓ} {G : Type ℓg} (grp : Group-on G) where

  -- Constant stack: F(U) = BG for all U
  Constant-BG : {C : Precategory o ℓ} → Stack C lzero ℓg
  Constant-BG {C} = Constant (BG G grp)

  -- Classifying topos for constant BG fibration
  E[BG] : {C : Precategory o ℓ} → Precategory _ _
  E[BG] {C} = ClassifyingTopos.E[F] (Constant-BG {C})
-}

  {- Equivalence requires G^ category (commented out above)
  -- Equivalence: E[BG] ≃ G^ (topos of G-sets)
  -- TODO: Prove that sheaves on ∫(constant BG) are equivalent to G-sets
  classify : {C : Precategory o ℓ} → {!!} -- is-equivalence between E[BG] and G^
  classify = {!!}
  -}

--------------------------------------------------------------------------------
-- Part 6: Integration with Fork Theory
--------------------------------------------------------------------------------

open import Neural.Graph.Base
open import Neural.Graph.Oriented
open import Neural.Graph.Fork.Fork
open import Neural.Graph.Fork.Category
open import Neural.Graph.Fork.Coverage

open import Data.List hiding (_++_)
open import Data.Dec.Base

-- Fork-Stack: Stack arising from fork construction on oriented graph
module ForkStackIntegration
  (G : Graph o ℓ)
  (G-oriented : is-oriented G)
  (nodes : List (Graph.Node G))
  (nodes-complete : ∀ (n : Graph.Node G) → n ∈ nodes)
  (edge? : ∀ (x y : Graph.Node G) → Dec (Graph.Edge G x y))
  (node-eq? : ∀ (x y : Graph.Node G) → Dec (x ≡ y))
  where

  open import Cat.Site.Sheafification

  open ForkCategoricalStructure G G-oriented nodes nodes-complete edge? node-eq?
  open ForkCoverageConstruction G G-oriented nodes nodes-complete edge? node-eq?

  -- The fork category Γ̄-Category is NOT a stack over another category
  -- It IS a base category with fork-coverage
  -- So we use it directly for the classifying topos

  -- Presheaves over fork graph
  PSh[Γ̄] : Precategory _ _
  PSh[Γ̄] = Cat[ Γ̄-Category ^op , Sets (o ⊔ ℓ) ]

  -- Sheaves over fork graph with fork-coverage
  -- This is where weights and activities live!
  Sh[Γ̄,fork] : Precategory _ _
  Sh[Γ̄,fork] = PSh[Γ̄]  -- Same objects, but distinguished by sheaf property

  -- Sheafification using fork-coverage (not trivial J-canonical!)
  module _ (P : Functor (Γ̄-Category ^op) (Sets (o ⊔ ℓ))) where
    module ForkSheaf = Sheafification fork-coverage P
    P⁺ = ForkSheaf.Sheafify
    P⁺-is-sheaf = ForkSheaf.Sheafify-is-sheaf

  -- At fork-star vertices A★, the sheaf condition gives:
  -- P⁺(A★) ≅ ∏_{a'→A★} P⁺(a')  (automatic from fork-coverage + glue constructor)

  -- TODO: Express Weights from Neural.Graph.Fork.Weights as presheaf here
  -- TODO: Show sheafification replaces placeholder at A★ with product
  -- TODO: Connect backpropagation to natural transformations

