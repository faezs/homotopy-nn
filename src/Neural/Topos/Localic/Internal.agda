{-# OPTIONS --rewriting --guardedness --cubical --no-load-primitives #-}

{-|
# Localic Topos: Internal Hom Objects

This module defines the internal hom Ω-U for Ω-sets.

## Contents
- Internal hom Ω-U: The Ω-set of opens below U
- Propositional resizing to get carrier at Type o
- Complete proofs of symmetry, transitivity, reflexivity
-}

module Neural.Topos.Localic.Internal where

open import 1Lab.Prelude
open import 1Lab.Resizing

open import Neural.Topos.Localic.Base
open import Order.Diagram.Meet
open import Order.Heyting
open import Order.Frame
open import Order.Diagram.Meet.Reasoning

--------------------------------------------------------------------------------
-- Internal Hom: Ω-U

{-|
## Internal Hom Ω-U

For a locale Ω and element U ∈ Ω, we define the **internal hom** Ω-U as:
- **Carrier**: Opens V with V ≤ U (using □ resizing to stay at Type o)
- **Fuzzy equality**: δ(V,W) = (V⇨W) ∧ (W⇨V) (biimplication)

**Key technical point**:
- Naively, carrier = Σ[V ∈ Ω] (V ≤ U) : Type (o⊔ℓ) since ≤ : Type ℓ
- But Ω-Set.Carrier must be Type o for frame joins to work!
- Solution: Use propositional resizing □ : Type ℓ → Type 0
- Since ≤ is a proposition (≤-thin), □(V ≤ U) is equivalent to (V ≤ U)
- So carrier = Σ[V ∈ Ω] (□(V ≤ U)) : Type o ✓

**Properties proven**:
- δ-U-sym: Symmetry via ∩-comm
- δ-U-trans: Transitivity via Heyting implication chaining (ƛ and ev)
- δ-U-refl: Reflexivity via ⊤ ≤ V⇨V

**DNN Interpretation**:
- Ω-U represents "contexts bounded by certainty U"
- V,W ∈ Ω-U are "sub-contexts"
- δ(V,W) = (V⇨W)∧(W⇨V) measures "logical equivalence of contexts"
- Used for contextual reasoning in neural networks
-}

Ω-U : {o ℓ : Level} (Ω : CompleteHeytingAlgebra o ℓ) → CompleteHeytingAlgebra.Ob Ω → Ω-Set Ω
Ω-U {o} {ℓ} Ω U = ω-set carrier δ-U (λ {x} {y} → δ-U-sym {x} {y}) (λ {x} {y} {z} → δ-U-trans {x} {y} {z}) (λ {x} → δ-U-refl {x})
  where
    module Ω' = CompleteHeytingAlgebra Ω

    -- Carrier: opens V with □(V ≤ U) using propositional resizing
    -- Since ≤ is a proposition (≤-thin) and □ resizes to Type 0,
    -- we get Σ[V : Type o] (Type 0) = Type o ✓
    carrier : Type o
    carrier = Σ[ V ∈ Ω'.Ob ] (□ (Ω'._≤_ V U))

    -- Fuzzy equality: biimplication (V ⇨ W) ∩ (W ⇨ V)
    δ-U : carrier → carrier → Ω'.Ob
    δ-U (V , _) (W , _) = (Ω'._⇨_ V W) Ω'.∩ (Ω'._⇨_ W V)

    -- Symmetry: follows from ∩-comm
    δ-U-sym : ∀ {x y} → δ-U x y ≡ δ-U y x
    δ-U-sym {V , _} {W , _} = Ω'.∩-comm

    -- Transitivity: (V⇨W)∩(W⇨V) ∩ (W⇨Z)∩(Z⇨W) ≤ (V⇨Z)∩(Z⇨V)
    -- Proof: implication chaining via ƛ and ev
    δ-U-trans : ∀ {x y z} → Ω'._≤_ (δ-U x y Ω'.∩ δ-U y z) (δ-U x z)
    δ-U-trans {V , _} {W , _} {Z , _} =
      let
        module H = is-heyting-algebra (Ω .heyting)
        module F = is-frame (Ω .frame)
        module MR = Order.Diagram.Meet.Reasoning (H.∩-meets)

        v⇨w = Ω'._⇨_ V W
        w⇨v = Ω'._⇨_ W V
        w⇨z = Ω'._⇨_ W Z
        z⇨w = Ω'._⇨_ Z W
        v⇨z = Ω'._⇨_ V Z
        z⇨v = Ω'._⇨_ Z V

        bigexpr = (v⇨w Ω'.∩ w⇨v) Ω'.∩ (w⇨z Ω'.∩ z⇨w)

        -- Part 1: bigexpr ≤ v⇨z
        -- Strategy: use ƛ to show (bigexpr H.∩ V) ≤ Z
        part1-body : (bigexpr H.∩ V) Ω'.≤ Z
        part1-body =
          -- Extract components
          let step1 : bigexpr Ω'.≤ v⇨w
              step1 = Ω'.≤-trans (Ω'.∩-meets _ _ .is-meet.meet≤l) (Ω'.∩-meets _ _ .is-meet.meet≤l)

              step2 : bigexpr Ω'.≤ w⇨z
              step2 = Ω'.≤-trans (Ω'.∩-meets _ _ .is-meet.meet≤r) (Ω'.∩-meets _ _ .is-meet.meet≤l)

              -- Apply monotonicity: (bigexpr H.∩ V) ≤ (v⇨w H.∩ V)
              mono1 : (bigexpr H.∩ V) Ω'.≤ (v⇨w H.∩ V)
              mono1 = MR.∩≤∩ step1 Ω'.≤-refl

              -- Apply ev: (v⇨w H.∩ V) ≤ W
              to-W : (v⇨w H.∩ V) Ω'.≤ W
              to-W = H.ev Ω'.≤-refl

              -- Now: (bigexpr H.∩ V) ≤ W
              have-W : (bigexpr H.∩ V) Ω'.≤ W
              have-W = Ω'.≤-trans mono1 to-W

              -- Need: (bigexpr H.∩ V) ≤ (w⇨z H.∩ W) to apply second ev
              -- Build greatest lower bound: (bigexpr H.∩ V) ≤ w⇨z and ≤ W
              to-meet : (bigexpr H.∩ V) Ω'.≤ (w⇨z H.∩ W)
              to-meet = H.∩-meets w⇨z W .is-meet.greatest (bigexpr H.∩ V)
                          (Ω'.≤-trans (H.∩-meets bigexpr V .is-meet.meet≤l) step2)
                          have-W

              -- Apply ev: (w⇨z H.∩ W) ≤ Z
              to-Z : (w⇨z H.∩ W) Ω'.≤ Z
              to-Z = H.ev Ω'.≤-refl

          in Ω'.≤-trans to-meet to-Z

        part1 : bigexpr Ω'.≤ v⇨z
        part1 = H.ƛ part1-body

        -- Part 2: bigexpr ≤ z⇨v (symmetric)
        part2-body : (bigexpr H.∩ Z) Ω'.≤ V
        part2-body =
          let step1 : bigexpr Ω'.≤ z⇨w
              step1 = Ω'.≤-trans (Ω'.∩-meets _ _ .is-meet.meet≤r) (Ω'.∩-meets _ _ .is-meet.meet≤r)

              step2 : bigexpr Ω'.≤ w⇨v
              step2 = Ω'.≤-trans (Ω'.∩-meets _ _ .is-meet.meet≤l) (Ω'.∩-meets _ _ .is-meet.meet≤r)

              mono1 : (bigexpr H.∩ Z) Ω'.≤ (z⇨w H.∩ Z)
              mono1 = MR.∩≤∩ step1 Ω'.≤-refl

              to-W : (z⇨w H.∩ Z) Ω'.≤ W
              to-W = H.ev Ω'.≤-refl

              have-W : (bigexpr H.∩ Z) Ω'.≤ W
              have-W = Ω'.≤-trans mono1 to-W

              to-meet : (bigexpr H.∩ Z) Ω'.≤ (w⇨v H.∩ W)
              to-meet = H.∩-meets w⇨v W .is-meet.greatest (bigexpr H.∩ Z)
                          (Ω'.≤-trans (H.∩-meets bigexpr Z .is-meet.meet≤l) step2)
                          have-W

              to-V : (w⇨v H.∩ W) Ω'.≤ V
              to-V = H.ev Ω'.≤-refl

          in Ω'.≤-trans to-meet to-V

        part2 : bigexpr Ω'.≤ z⇨v
        part2 = H.ƛ part2-body

      in Ω'.∩-meets v⇨z z⇨v .is-meet.greatest _ part1 part2

    -- Reflexivity: ⊤ ≤ (V⇨V)∩(V⇨V)
    -- Need to use Heyting algebra's ∩ and ƛ
    δ-U-refl : ∀ {x} → Ω'._≤_ Ω'.top (δ-U x x)
    δ-U-refl {V , _} =
      let
        module H = is-heyting-algebra (Ω .heyting)

        -- ⊤ H.∩ V ≤ V (using heyting's ∩)
        top-meet-v-le-v : (H._∩_ Ω'.top V) Ω'.≤ V
        top-meet-v-le-v = H.∩-meets Ω'.top V .is-meet.meet≤r

        -- ⊤ ≤ V⇨V using ƛ on ⊤∩V ≤ V
        v-impl-v : Ω'._≤_ Ω'.top (Ω'._⇨_ V V)
        v-impl-v = H.ƛ top-meet-v-le-v

        -- (V⇨V)∩(V⇨V) = V⇨V, so ⊤ ≤ V⇨V implies ⊤ ≤ (V⇨V)∩(V⇨V)
        impl-idempotent : Ω'._≤_ (Ω'._⇨_ V V) ((Ω'._⇨_ V V) Ω'.∩ (Ω'._⇨_ V V))
        impl-idempotent = Ω'.∩-meets (Ω'._⇨_ V V) (Ω'._⇨_ V V) .is-meet.greatest (Ω'._⇨_ V V) Ω'.≤-refl Ω'.≤-refl
      in Ω'.≤-trans v-impl-v impl-idempotent
