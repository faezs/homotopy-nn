{-# OPTIONS --rewriting --guardedness --cubical --no-load-primitives #-}

{-|
# Localic Topos: Category Structure

This module defines the category SetΩ of Ω-sets.

## Contents
- §A.4: Category SetΩ (Equations 23-24)
- Composition and identity
- Associativity proof
-}

module Neural.Topos.Localic.Category where

open import 1Lab.Prelude
open import Cat.Base
open import Cat.Reasoning

open import Neural.Topos.Localic.Base
open import Order.Diagram.Lub
open import Order.Diagram.Meet
import Order.Frame.Reasoning as FrameReasoning

--------------------------------------------------------------------------------
-- §A.4: Category SetΩ (Equations 23-24)

{-|
## Category of Ω-Sets

**Composition** (Equation 23):
  (f' ∘ f)(x, x") = ⋁_{x'∈X'} f(x,x') ∧ f'(x',x")

This is like matrix multiplication, but with:
- Sum → Join (⋁)
- Product → Meet (∧)

**Identity** (Equation 24):
  Id_{X,δ} = δ

The fuzzy equality itself is the identity morphism!

**Proof obligations**:
1. Composition is associative
2. δ is left and right identity
3. Composition preserves morphism axioms (19-22)
-}

-- Path equality for Ω-Set morphisms
-- Since axiom fields live in propositions, equality follows from equality of .f fields
Ω-Set-Morphism-path : ∀ {o ℓ : Level} {Ω : CompleteHeytingAlgebra o ℓ} {X Y : Ω-Set Ω}
                      → {f g : Ω-Set-Morphism X Y}
                      → f .Ω-Set-Morphism.f ≡ g .Ω-Set-Morphism.f
                      → f ≡ g
Ω-Set-Morphism-path {Ω = Ω} {X = XSet} {Y = YSet} {f = fmor} {g = gmor} p = path
  where
    module ΩA = CompleteHeytingAlgebra Ω
    module XS = Ω-Set XSet
    module YS = Ω-Set YSet

    path : fmor ≡ gmor
    path i .Ω-Set-Morphism.f = p i
    path i .Ω-Set-Morphism.eq-19 {a} {b} {c} =
      is-prop→pathp (λ j → ΩA.≤-thin {x = (XS.δ a b ΩA.∩ p j a c)} {y = p j b c})
        (fmor .Ω-Set-Morphism.eq-19 {a} {b} {c}) (gmor .Ω-Set-Morphism.eq-19 {a} {b} {c}) i
    path i .Ω-Set-Morphism.eq-20 {a} {b} {c} =
      is-prop→pathp (λ j → ΩA.≤-thin {x = (p j a b ΩA.∩ YS.δ b c)} {y = p j a c})
        (fmor .Ω-Set-Morphism.eq-20 {a} {b} {c}) (gmor .Ω-Set-Morphism.eq-20 {a} {b} {c}) i
    path i .Ω-Set-Morphism.eq-21 {a} {b} {c} =
      is-prop→pathp (λ j → ΩA.≤-thin {x = (p j a b ΩA.∩ p j a c)} {y = YS.δ b c})
        (fmor .Ω-Set-Morphism.eq-21 {a} {b} {c}) (gmor .Ω-Set-Morphism.eq-21 {a} {b} {c}) i
    path i .Ω-Set-Morphism.eq-22 {a} =
      is-prop→pathp (λ j → ΩA.Ob-is-set (ΩA.⋁ (p j a)) (XS.δ a a))
        (fmor .Ω-Set-Morphism.eq-22 {a}) (gmor .Ω-Set-Morphism.eq-22 {a}) i

-- Composition (Equation 23)
_∘-Ω_ : {o ℓ : Level} {Ω : CompleteHeytingAlgebra o ℓ}
        {X Y Z : Ω-Set Ω}
      → Ω-Set-Morphism Y Z
      → Ω-Set-Morphism X Y
      → Ω-Set-Morphism X Z
_∘-Ω_ {Ω = Ω} {X} {Y} {Z} g f = ω-morphism comp-Ω eq-19-comp eq-20-comp eq-21-comp eq-22-comp
  where
    module ΩA = CompleteHeytingAlgebra Ω
    module X = Ω-Set X
    module Y = Ω-Set Y
    module Z = Ω-Set Z
    module f = Ω-Set-Morphism f
    module g = Ω-Set-Morphism g

    -- Composition formula (Equation 23)
    comp-Ω : X.Carrier → Z.Carrier → ΩA.Ob
    comp-Ω x z = ΩA.⋁ (λ (y : Y.Carrier) → f.f x y ΩA.∩ g.f y z)

    -- Verify equations 19-22 for composition
    -- eq-19: δ(x,x') ∧ (g∘f)(x,z) ≤ (g∘f)(x',z)
    eq-19-comp : ∀ {x x' : X.Carrier} {z : Z.Carrier}
               → (ΩA._≤_) (X.δ x x' ΩA.∩ comp-Ω x z) (comp-Ω x' z)
    eq-19-comp {x} {x'} {z} =
      -- Step 1: Use distributivity to transform X.δ ∩ ⋃_y f ≡ ⋃_y (X.δ ∩ f)
      ΩA.≤-trans
        (ΩA.≤-refl' (ΩA.⋃-distribl (X.δ x x') (λ y → f.f x y ΩA.∩ g.f y z)))
        -- Step 2: Show ⋃_y (X.δ x x' ∩ (f.f x y ∩ g.f y z)) ≤ comp-Ω x' z
        (ΩA.⋃-universal (comp-Ω x' z) λ y →
          let -- After expanding, we have: X.δ x x' ∩ (f.f x y ∩ g.f y z)
              -- Want to show this ≤ comp-Ω x' z = ⋁_y' (f.f x' y' ∩ g.f y' z)
              -- Sufficient to show: X.δ x x' ∩ (f.f x y ∩ g.f y z) ≤ f.f x' y ∩ g.f y z
              left-bound : (X.δ x x' ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ f.f x' y
              left-bound = ΩA.≤-trans (ΩA.∩≤∩ ΩA.≤-refl ΩA.∩≤l) (f.eq-19 {x} {x'} {y})
              right-bound : (X.δ x x' ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ g.f y z
              right-bound = ΩA.≤-trans ΩA.∩≤r ΩA.∩≤r
          in ΩA.≤-trans (ΩA.∩-universal _ left-bound right-bound) (ΩA.⋃-inj y))

    -- eq-20: (g∘f)(x,z) ∧ δ(z,z') ≤ (g∘f)(x,z')
    eq-20-comp : ∀ {x : X.Carrier} {z z' : Z.Carrier}
               → (ΩA._≤_) (comp-Ω x z ΩA.∩ Z.δ z z') (comp-Ω x z')
    eq-20-comp {x} {z} {z'} =
      ΩA.≤-trans
        (ΩA.≤-refl' ΩA.∩-comm)  -- Swap: comp-Ω x z ∩ Z.δ ≡ Z.δ ∩ comp-Ω x z
        (ΩA.≤-trans
          (ΩA.≤-refl' (ΩA.⋃-distribl (Z.δ z z') (λ y → f.f x y ΩA.∩ g.f y z)))
          (ΩA.⋃-universal (comp-Ω x z') λ y →
            let -- We have: Z.δ z z' ∩ (f.f x y ∩ g.f y z)
                -- Want: this ≤ comp-Ω x z' = ⋁_y' (f.f x y' ∩ g.f y' z')
                -- Sufficient: Z.δ z z' ∩ (f.f x y ∩ g.f y z) ≤ f.f x y ∩ g.f y z'
                left-bound : (Z.δ z z' ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ f.f x y
                left-bound = ΩA.≤-trans ΩA.∩≤r ΩA.∩≤l
                right-bound : (Z.δ z z' ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ g.f y z'
                right-bound = ΩA.≤-trans
                                (ΩA.∩≤∩ ΩA.≤-refl ΩA.∩≤r)  -- Z.δ ∩ (f ∩ g) ≤ Z.δ ∩ g
                                (ΩA.≤-trans
                                  (ΩA.≤-refl' ΩA.∩-comm)  -- Z.δ ∩ g ≡ g ∩ Z.δ
                                  (g.eq-20 {y} {z} {z'}))
            in ΩA.≤-trans (ΩA.∩-universal _ left-bound right-bound) (ΩA.⋃-inj y)))

    -- eq-21: (g∘f)(x,z) ∧ (g∘f)(x,z') ≤ δ(z,z')
    eq-21-comp : ∀ {x : X.Carrier} {z z' : Z.Carrier}
               → (ΩA._≤_) (comp-Ω x z ΩA.∩ comp-Ω x z') (Z.δ z z')
    eq-21-comp {x} {z} {z'} =
      -- Step 1: Distribute meet over first join
      ΩA.≤-trans
        (ΩA.≤-refl' (ΩA.⋃-distribl (comp-Ω x z) (λ y' → f.f x y' ΩA.∩ g.f y' z')))
        -- Now we have: ⋃_{y'} [comp-Ω x z ∩ (f x y' ∩ g y' z')]
        -- Step 2: Apply ⋃-universal to the outer join
        (ΩA.⋃-universal (Z.δ z z') λ y' →
          -- For each y', show: comp-Ω x z ∩ (f x y' ∩ g y' z') ≤ Z.δ z z'
          -- comp-Ω x z = ⋃_y (f x y ∩ g y z), so we have:
          -- (⋃_y (f x y ∩ g y z)) ∩ (f x y' ∩ g y' z')
          -- Step 3: Swap operands with ∩-comm, then distribute
          ΩA.≤-trans
            (ΩA.≤-refl' ΩA.∩-comm)  -- (⋃_y ...) ∩ (f x y' ∩ g y' z') ≡ (f x y' ∩ g y' z') ∩ (⋃_y ...)
            (ΩA.≤-trans
              (ΩA.≤-refl' (ΩA.⋃-distribl (f.f x y' ΩA.∩ g.f y' z') (λ y → f.f x y ΩA.∩ g.f y z)))
              -- Now we have: ⋃_y [(f x y' ∩ g y' z') ∩ (f x y ∩ g y z)]
              -- Step 4: Apply ⋃-universal to this join
              (ΩA.⋃-universal (Z.δ z z') λ y →
              -- For each y, show: (f x y' ∩ g y' z') ∩ (f x y ∩ g y z) ≤ Z.δ z z'
              -- Key: Extract g y z and g y' z', then use transitivity
              --      (f x y' ∩ g y' z') ∩ (f x y ∩ g y z) ≤ g y' z' ∩ g y z
              -- Then use g.eq-21 for y' to get: g y' z' ∩ g y' z ≤ Z.δ z z'
              -- But we have g y z, not g y' z, so we need a different approach.
              --
              -- Actually: use that g y z ≤ ⋁_y g y z ≤ Y.δ y y (by g.eq-22)
              -- And g y' z' ∩ Y.δ y y ≤ ... no, this gets complicated.
              --
              -- Simpler: Both sides respect target equality, so:
              -- g y' z' ≤ g y' z ∩ Y.δ z z' (NO - wrong direction)
              --
              -- Let me use: (g y' z') ∩ (g y z) ≤ (g y' z' ∩ g y' z) via g.eq-20
              let -- Key: Use f's single-valuedness then g's source-equality respect
                  -- Step 1: Extract f x y' and f x y to get Y.δ y' y via f.eq-21
                  y-rel : ((f.f x y' ΩA.∩ g.f y' z') ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ Y.δ y' y
                  y-rel = ΩA.≤-trans
                            (ΩA.∩-universal _
                              (ΩA.≤-trans ΩA.∩≤l ΩA.∩≤l)  -- f x y'
                              (ΩA.≤-trans ΩA.∩≤r ΩA.∩≤l))  -- f x y
                            (f.eq-21 {x} {y'} {y})
                  -- Step 2: Extract g y z directly using chained ≤-trans
                  -- Structure: (f x y' ∩ g y' z') ∩ (f x y ∩ g y z)
                  -- To get g y z, we go: right (∩≤r), then right again (∩≤r)
                  gy-z-direct : ((f.f x y' ΩA.∩ g.f y' z') ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ g.f y z
                  gy-z-direct = ΩA.≤-trans ΩA.∩≤r ΩA.∩≤r
                  -- Step 3: Build g y z' using g.eq-19: Y.δ y' y ∩ g y' z' ≤ g y z'
                  -- We need to build Y.δ y' y ∩ g y' z', then apply g.eq-19
                  gy-zp : ((f.f x y' ΩA.∩ g.f y' z') ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ g.f y z'
                  gy-zp = let -- First build the meet Y.δ y' y ∩ g y' z'
                              -- To get g y' z': left (∩≤l), then right (∩≤r)
                              gyp-zp-extract : ((f.f x y' ΩA.∩ g.f y' z') ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ g.f y' z'
                              gyp-zp-extract = ΩA.≤-trans ΩA.∩≤l ΩA.∩≤r
                              y-rel-and-gyp-zp : ((f.f x y' ΩA.∩ g.f y' z') ΩA.∩ (f.f x y ΩA.∩ g.f y z)) ΩA.≤ (Y.δ y' y ΩA.∩ g.f y' z')
                              y-rel-and-gyp-zp = ΩA.∩-universal _ y-rel gyp-zp-extract
                          in ΩA.≤-trans y-rel-and-gyp-zp (g.eq-19 {y'} {y} {z'})
                  -- Step 3: Combine to get g y z ∩ g y z', then apply g.eq-21
              in ΩA.≤-trans
                   (ΩA.∩-universal _ gy-z-direct gy-zp)
                   (g.eq-21 {y} {z} {z'}))))

    -- eq-22: ⋁_z (g∘f)(x,z) = δ(x,x)
    -- We need to show: ⋁_z ⋁_y [f(x,y) ∧ g(y,z)] = δ(x,x)
    -- Strategy: Use that ⋁_y f(x,y) = δ(x,x) and ⋁_z g(y,z) = δ(y,y)
    eq-22-comp : ∀ {x : X.Carrier}
               → ΩA.⋁ (λ (z : Z.Carrier) → comp-Ω x z) ≡ X.δ x x
    eq-22-comp {x} = ΩA.≤-antisym
      -- Upper bound: ⋁_z ⋁_y [f(x,y) ∧ g(y,z)] ≤ δ(x,x)
      (ΩA.⋃-lubs _ .is-lub.least _ λ z →
        ΩA.⋃-lubs _ .is-lub.least _ λ y →
          -- f(x,y) ∧ g(y,z) ≤ f(x,y) ≤ δ(x,x)
          ΩA.≤-trans ΩA.∩≤l (ΩA.≤-trans (ΩA.⋃-lubs _ .is-lub.fam≤lub y)
            (ΩA.≤-refl' (f.eq-22 {x}))))
      -- Lower bound: δ(x,x) ≤ ⋁_z ⋁_y [f(x,y) ∧ g(y,z)]
      -- For each y: f(x,y) ≤ f(x,y) ∩ Y.δ(y,y) = f(x,y) ∩ (⋁_z g(y,z)) = ⋁_z [f(x,y) ∩ g(y,z)]
      (ΩA.≤-trans
        (ΩA.≤-refl' (sym (f.eq-22 {x})))  -- X.δ x x = ⋁_y f x y
        (ΩA.⋃-lubs _ .is-lub.least _ λ y →
          -- Show: f(x,y) ≤ ⋁_z ⋁_y' [f(x,y') ∩ g(y',z)]
          -- By distributivity: f(x,y) ∩ (⋁_z g(y,z)) = ⋁_z [f(x,y) ∩ g(y,z)]
          -- Since ⋁_z g(y,z) = Y.δ y y (by g.eq-22), and f.f x y ≤ f.f x y ∩ Y.δ y y (by f.eq-19)
          -- we get f(x,y) ≤ ⋁_z [f(x,y) ∩ g(y,z)]
          let -- Step 1: Use distributivity
              fy-distributes : (f.f x y ΩA.∩ ΩA.⋁ (g.f y)) ≡ ΩA.⋁ (λ z → f.f x y ΩA.∩ g.f y z)
              fy-distributes = ΩA.⋃-distribl (f.f x y) (g.f y)
              -- Step 2: Substitute ⋁_z g(y,z) = Y.δ y y
              fy-meet-delta : (f.f x y ΩA.∩ Y.δ y y) ≡ ΩA.⋁ (λ z → f.f x y ΩA.∩ g.f y z)
              fy-meet-delta = ap (ΩA._∩_ (f.f x y)) (sym (g.eq-22 {y})) ∙ fy-distributes
              -- Step 3: f.f x y ≤ f.f x y ∩ Y.δ y y
              -- Since ⊤ ≤ Y.δ y y (reflexivity), we have f.f x y ≤ f.f x y ∩ Y.δ y y
              fy-bounded : f.f x y ΩA.≤ (f.f x y ΩA.∩ Y.δ y y)
              fy-bounded = ΩA.∩-universal _ ΩA.≤-refl (ΩA.≤-trans ΩA.! Y.δ-refl)
              -- Step 4: Embed into double join: ⋁_z [f(x,y) ∩ g(y,z)] ≤ ⋁_z ⋁_y' [f(x,y') ∩ g(y',z)]
              -- For each z: f(x,y) ∩ g(y,z) ≤ ⋁_y' [f(x,y') ∩ g(y',z)] since y appears in the family
              join-embed : ΩA.⋁ (λ z → f.f x y ΩA.∩ g.f y z) ΩA.≤ ΩA.⋁ (λ z → ΩA.⋁ (λ y' → f.f x y' ΩA.∩ g.f y' z))
              join-embed = ΩA.⋃-lubs _ .is-lub.least _ λ z →
                ΩA.≤-trans (ΩA.⋃-lubs _ .is-lub.fam≤lub y) (ΩA.⋃-lubs _ .is-lub.fam≤lub z)
          in ΩA.≤-trans (ΩA.≤-trans fy-bounded (ΩA.≤-refl' fy-meet-delta)) join-embed))

-- Identity (Equation 24)
id-Ω : {o ℓ : Level} {Ω : CompleteHeytingAlgebra o ℓ} {X : Ω-Set Ω}
     → Ω-Set-Morphism X X
id-Ω {Ω = Ω} {X} = ω-morphism X.δ eq-19-id eq-20-id eq-21-id eq-22-id
  where
    module ΩA = CompleteHeytingAlgebra Ω
    module X = Ω-Set X

    -- δ satisfies morphism equations
    -- eq-19: δ(x,y) ∧ δ(x,x') ≤ δ(y,x')
    -- Use δ(x,y) ∧ δ(x,x') ≤ δ(y,x) ∧ δ(x,x') ≤ δ(y,x') by transitivity on reordered terms
    eq-19-id : ∀ {x y x' : X.Carrier}
             → (ΩA._≤_) (X.δ x y ΩA.∩ X.δ x x') (X.δ y x')
    eq-19-id {x} {y} {x'} = ΩA.≤-trans step X.δ-trans
      where
        -- Rewrite δ(x,y) ∧ δ(x,x') as δ(y,x) ∧ δ(x,x') using symmetry
        step : (ΩA._≤_) (X.δ x y ΩA.∩ X.δ x x') (X.δ y x ΩA.∩ X.δ x x')
        step = subst (λ z → (ΩA._≤_) (X.δ x y ΩA.∩ X.δ x x') (z ΩA.∩ X.δ x x')) (sym X.δ-sym) ΩA.≤-refl

    eq-20-id : ∀ {x x' y' : X.Carrier}
             → (ΩA._≤_) (X.δ x x' ΩA.∩ X.δ x' y') (X.δ x y')
    eq-20-id = X.δ-trans

    eq-21-id : ∀ {x x' y' : X.Carrier}
             → (ΩA._≤_) (X.δ x x' ΩA.∩ X.δ x y') (X.δ x' y')
    eq-21-id {x} {x'} {y'} = ΩA.≤-trans helper X.δ-trans
      where
        helper : (ΩA._≤_) (X.δ x x' ΩA.∩ X.δ x y') (X.δ x' x ΩA.∩ X.δ x y')
        open is-meet (ΩA.∩-meets (X.δ x' x) (X.δ x y')) renaming (greatest to ∩-gr)
        helper = ∩-gr _
                   (ΩA.≤-trans ΩA.∩≤l (subst (λ a → (ΩA._≤_) (X.δ x x') a) (sym X.δ-sym) ΩA.≤-refl))
                   ΩA.∩≤r

    -- eq-22: ⋁_{x'} δ(x,x') = δ(x,x)
    -- This holds because δ(x,x) is an upper bound and also in the family
    eq-22-id : ∀ {x : X.Carrier}
             → ΩA.⋁ (λ (x' : X.Carrier) → X.δ x x') ≡ X.δ x x
    eq-22-id {x} = ΩA.≤-antisym
      (ΩA.⋃-lubs _ .is-lub.least _ λ x' → X.δ-self-bound)
      (ΩA.⋃-lubs _ .is-lub.fam≤lub x)

-- Helper: Triple composition expands to a double join
module TripleComp {o ℓ : Level} {Ω : CompleteHeytingAlgebra o ℓ}
                  {W X Y Z : Ω-Set Ω}
                  (h : Ω-Set-Morphism Y Z)
                  (g : Ω-Set-Morphism X Y)
                  (f : Ω-Set-Morphism W X) where
  private
    module ΩA = CompleteHeytingAlgebra Ω
    module W = Ω-Set W
    module X = Ω-Set X
    module Y = Ω-Set Y
    module Z = Ω-Set Z
    module f = Ω-Set-Morphism f
    module g = Ω-Set-Morphism g
    module h = Ω-Set-Morphism h

  -- Import FrameReasoning for equational reasoning and ⋃-distribr
  open FrameReasoning ΩA.frame

  -- The "flat" triple join using a product index
  flat-join : W.Carrier → Z.Carrier → ΩA.Ob
  flat-join w z = ΩA.⋃ (λ (p : X.Carrier × Y.Carrier) →
    (f.f w (p .fst) ΩA.∩ g.f (p .fst) (p .snd)) ΩA.∩ h.f (p .snd) z)

  -- Lemma: (h ∘ g) ∘ f equals flat-join via frame distributivity
  -- Modeled after Cat.Allegory.Instances.Mat.assoc
  -- (h ∘ g) ∘ f = ⋃ (λ x → f w x ∩ (⋃ (λ y → g x y ∩ h y z)))
  left-to-flat : (w : W.Carrier) → (z : Z.Carrier) → (Ω-Set-Morphism.f (_∘-Ω_ (_∘-Ω_ h g) f) w z) ≡ flat-join w z
  left-to-flat w z =
    Ω-Set-Morphism.f (_∘-Ω_ (_∘-Ω_ h g) f) w z                                                                   ≡⟨⟩
    ΩA.⋃ (λ x → f.f w x ΩA.∩ (ΩA.⋃ (λ y → g.f x y ΩA.∩ h.f y z)))                                               ≡⟨ ⋃-apᶠ (λ x → ⋃-distribl (f.f w x) _) ⟩
    ΩA.⋃ (λ x → ΩA.⋃ (λ y → f.f w x ΩA.∩ (g.f x y ΩA.∩ h.f y z)))                                               ≡⟨ ⋃-twice _ ⟩
    ΩA.⋃ (λ (p : X.Carrier × Y.Carrier) → f.f w (p .fst) ΩA.∩ (g.f (p .fst) (p .snd) ΩA.∩ h.f (p .snd) z))      ≡⟨ ⋃-apᶠ (λ _ → ΩA.∩-assoc) ⟩
    ΩA.⋃ (λ (p : X.Carrier × Y.Carrier) → (f.f w (p .fst) ΩA.∩ g.f (p .fst) (p .snd)) ΩA.∩ h.f (p .snd) z)      ≡⟨⟩
    flat-join w z ∎

  -- Lemma: h ∘ (g ∘ f) equals flat-join via frame distributivity + index swap
  -- Modeled after Cat.Allegory.Instances.Mat.assoc
  right-to-flat : (w : W.Carrier) → (z : Z.Carrier) → (Ω-Set-Morphism.f (_∘-Ω_ h (_∘-Ω_ g f)) w z) ≡ flat-join w z
  right-to-flat w z =
    ΩA.⋃ (λ y → ΩA.⋃ (λ x → f.f w x ΩA.∩ g.f x y) ΩA.∩ h.f y z)                                                  ≡⟨ ⋃-apᶠ (λ y → ⋃-distribr _ (h.f y z)) ⟩
    ΩA.⋃ (λ y → ΩA.⋃ (λ x → (f.f w x ΩA.∩ g.f x y) ΩA.∩ h.f y z))                                                ≡⟨ ⋃-twice _ ⟩
    ΩA.⋃ (λ (p : Y.Carrier × X.Carrier) → (f.f w (p .snd) ΩA.∩ g.f (p .snd) (p .fst)) ΩA.∩ h.f (p .fst) z)   ≡⟨ ⋃-apⁱ ×-swap ⟩
    ΩA.⋃ (λ (p : X.Carrier × Y.Carrier) → (f.f w (p .fst) ΩA.∩ g.f (p .fst) (p .snd)) ΩA.∩ h.f (p .snd) z)   ≡⟨⟩
    flat-join w z ∎

  assoc-proof : (_∘-Ω_ (_∘-Ω_ h g) f) ≡ (_∘-Ω_ h (_∘-Ω_ g f))
  assoc-proof = Ω-Set-Morphism-path {Ω = Ω} {X = W} {Y = Z}
    (funext λ w → funext λ z → left-to-flat w z ∙ sym (right-to-flat w z))

-- SetΩ category
SetΩ : {o ℓ : Level} (Ω : CompleteHeytingAlgebra o ℓ) → Precategory (lsuc o ⊔ ℓ) (lsuc o ⊔ ℓ)
SetΩ Ω = cat
  where
    cat : Precategory _ _
    cat .Precategory.Ob = Ω-Set Ω
    cat .Precategory.Hom X Y = Ω-Set-Morphism X Y
    cat .Precategory.Hom-set X Y = hlevel 2
    cat .Precategory.id = id-Ω
    cat .Precategory._∘_ = _∘-Ω_
    -- Category laws: Prove using record extensionality
    -- For records with no-eta-equality, we need to show field-wise equality
    -- Since Ω-Set-Morphism has h-level 2, equality of .f fields implies record equality

    -- Right identity: f ∘ id = f
    -- The composition (f ∘ id) has type X → Y where id : X → X, f : X → Y
    -- So (f ∘ id).f(a,b) = ⋁_{c:X} id.f(a,c) ∧ f.f(c,b) = ⋁_c δ(a,c) ∧ f(c,b)
    cat .Precategory.idr {x} {y} f =
      let module ΩA = CompleteHeytingAlgebra Ω
          module X = Ω-Set x
          module Y = Ω-Set y
          module f-mod = Ω-Set-Morphism f
      in Ω-Set-Morphism-path (funext λ a → funext λ b →
           ΩA.≤-antisym
             -- Upper: ⋁_c [δ(a,c) ∧ f(c,b)] ≤ f(a,b)
             -- eq-19 says: δ(c,a) ∧ f(c,b) ≤ f(a,b) (with x=c, y=a, x'=b)
             -- Since δ(a,c) = δ(c,a) by δ-sym, we can transport
             (ΩA.⋃-lubs _ .is-lub.least _ λ c →
               subst (λ d → (d ΩA.∩ f-mod.f c b) ΩA.≤ f-mod.f a b) X.δ-sym f-mod.eq-19)
             -- Lower: f(a,b) ≤ ⋁_c [δ(a,c) ∧ f(c,b)] by taking c=a
             (ΩA.≤-trans
               (ΩA.∩-universal _ (ΩA.≤-trans ΩA.! X.δ-refl) ΩA.≤-refl)
               (ΩA.⋃-lubs _ .is-lub.fam≤lub a)))
    -- Left identity: id ∘ f = f
    -- The composition (id ∘ f) has type X → Y where f : X → Y, id : Y → Y
    -- So (id ∘ f).f(a,b) = ⋁_{c:Y} f.f(a,c) ∧ id.f(c,b) = ⋁_c f(a,c) ∧ δ(c,b)
    cat .Precategory.idl {x} {y} f =
      let module ΩA = CompleteHeytingAlgebra Ω
          module X = Ω-Set x
          module Y = Ω-Set y
          module f-mod = Ω-Set-Morphism f
      in Ω-Set-Morphism-path (funext λ a → funext λ b →
           ΩA.≤-antisym
             -- Upper: ⋁_c [f(a,c) ∧ δ(c,b)] ≤ f(a,b)
             -- eq-20 says: f(a,c) ∧ δ(c,b) ≤ f(a,b) (with x=a, x'=c, y'=b)
             (ΩA.⋃-lubs _ .is-lub.least _ λ c → f-mod.eq-20)
             -- Lower: f(a,b) ≤ ⋁_c [f(a,c) ∧ δ(c,b)] by taking c=b
             (ΩA.≤-trans
               (ΩA.∩-universal _ ΩA.≤-refl (ΩA.≤-trans ΩA.! Y.δ-refl))
               (ΩA.⋃-lubs _ .is-lub.fam≤lub b)))
    -- Associativity: (h ∘ g) ∘ f = h ∘ (g ∘ f)
    -- For f: W → X, g: X → Y, h: Y → Z
    -- LHS: ((h ∘ g) ∘ f).f(w,z) = ⋁_x [f(w,x) ∧ ⋁_y [g(x,y) ∧ h(y,z)]]
    --                             = ⋁_x [⋁_y [f(w,x) ∧ g(x,y) ∧ h(y,z)]] by distributivity
    -- RHS: (h ∘ (g ∘ f)).f(w,z) = ⋁_y [⋁_x [f(w,x) ∧ g(x,y)] ∧ h(y,z)]
    --                             = ⋁_y [⋁_x [f(w,x) ∧ g(x,y) ∧ h(y,z)]] by distributivity
    -- Both reduce to the same double join, so they're equal by symmetry of Σ-types
    -- Associativity: (h ∘ g) ∘ f = h ∘ (g ∘ f)
    -- Both sides expand to the same triple join via frame distributivity
    cat .Precategory.assoc {A} {B} {C} {D} h g f = sym (TripleComp.assoc-proof h g f)
      where open TripleComp

