-- Proof of quotient-algebra for quotient rule
-- This is a standalone proof to be integrated into Calculus.agda

module QuotientAlgebraProof where

open import Neural.Smooth.Base

-- Helper: Division isolation lemma
-- If a · b = c and b ≠ 0, then a = (c / b)
postulate
  division-isolate : ∀ {a b c : ℝ} (p : b ≠ 0ℝ) → a ·ℝ b ≡ c → a ≡ (c /ℝ b) p

-- Field algebra proof: from equation f' = h'·g + h·g', extract h' = (f'·g - f·g')/g²
quotient-algebra-proof : ∀ {f h g : ℝ → ℝ} {x : ℝ}
  → (numerator : ℝ)
  → (denominator : ℝ)
  → (denom-nonzero : denominator ≠ 0ℝ)
  → ((h ′[ x ]) ·ℝ g x) +ℝ (h x ·ℝ (g ′[ x ])) ≡ (f ′[ x ])
  → h x ·ℝ g x ≡ f x
  → (h ′[ x ]) ≡ (numerator /ℝ denominator) denom-nonzero
quotient-algebra-proof {f} {h} {g} {x} numerator denominator denom-nonzero prod-eq h-g-eq =
  let -- Step 1: Isolate h'·g from product rule equation
      -- From: h'·g + h·g' = f', derive: h'·g = f' - h·g'
      step1 : (h ′[ x ]) ·ℝ g x ≡ (f ′[ x ]) -ℝ (h x ·ℝ (g ′[ x ]))
      step1 =
        (h ′[ x ]) ·ℝ g x
          ≡⟨ sym (+ℝ-idr ((h ′[ x ]) ·ℝ g x)) ⟩
        ((h ′[ x ]) ·ℝ g x) +ℝ 0ℝ
          ≡⟨ ap (((h ′[ x ]) ·ℝ g x) +ℝ_) (sym (+ℝ-invr (h x ·ℝ (g ′[ x ])))) ⟩
        ((h ′[ x ]) ·ℝ g x) +ℝ ((h x ·ℝ (g ′[ x ])) +ℝ (-ℝ (h x ·ℝ (g ′[ x ]))))
          ≡⟨ sym (+ℝ-assoc ((h ′[ x ]) ·ℝ g x) (h x ·ℝ (g ′[ x ])) (-ℝ (h x ·ℝ (g ′[ x ])))) ⟩
        (((h ′[ x ]) ·ℝ g x) +ℝ (h x ·ℝ (g ′[ x ]))) +ℝ (-ℝ (h x ·ℝ (g ′[ x ])))
          ≡⟨ ap (_+ℝ (-ℝ (h x ·ℝ (g ′[ x ])))) prod-eq ⟩
        (f ′[ x ]) +ℝ (-ℝ (h x ·ℝ (g ′[ x ])))
          ∎

      -- Step 2: Multiply both sides by g: h'·g·g = (f' - h·g')·g
      step2 : ((h ′[ x ]) ·ℝ g x) ·ℝ g x ≡ ((f ′[ x ]) -ℝ (h x ·ℝ (g ′[ x ]))) ·ℝ g x
      step2 = ap (_·ℝ g x) step1

      -- Step 3: Distribute g over subtraction: h'·g·g = f'·g - (h·g')·g
      step3 : ((h ′[ x ]) ·ℝ g x) ·ℝ g x ≡ ((f ′[ x ]) ·ℝ g x) -ℝ ((h x ·ℝ (g ′[ x ])) ·ℝ g x)
      step3 =
        step2 ∙
        (((f ′[ x ]) +ℝ (-ℝ (h x ·ℝ (g ′[ x ])))) ·ℝ g x
          ≡⟨ ·ℝ-distribr (f ′[ x ]) (-ℝ (h x ·ℝ (g ′[ x ]))) (g x) ⟩
        ((f ′[ x ]) ·ℝ g x) +ℝ ((-ℝ (h x ·ℝ (g ′[ x ]))) ·ℝ g x)
          ∎)

      -- Step 4: Rearrange (h·g')·g to (h·g)·g' using associativity and commutativity
      step4 : ((h x ·ℝ (g ′[ x ])) ·ℝ g x) ≡ (h x ·ℝ g x) ·ℝ (g ′[ x ])
      step4 =
        (h x ·ℝ (g ′[ x ])) ·ℝ g x
          ≡⟨ ·ℝ-assoc (h x) (g ′[ x ]) (g x) ⟩
        h x ·ℝ ((g ′[ x ]) ·ℝ g x)
          ≡⟨ ap (h x ·ℝ_) (·ℝ-comm (g ′[ x ]) (g x)) ⟩
        h x ·ℝ (g x ·ℝ (g ′[ x ]))
          ≡⟨ sym (·ℝ-assoc (h x) (g x) (g ′[ x ])) ⟩
        (h x ·ℝ g x) ·ℝ (g ′[ x ])
          ∎

      -- Step 5: Substitute step4 into step3: h'·g·g = f'·g - (h·g)·g'
      step5 : ((h ′[ x ]) ·ℝ g x) ·ℝ g x ≡ ((f ′[ x ]) ·ℝ g x) -ℝ ((h x ·ℝ g x) ·ℝ (g ′[ x ]))
      step5 = step3 ∙ ap (λ z → ((f ′[ x ]) ·ℝ g x) -ℝ z) step4

      -- Step 6: Substitute h·g = f (from h-g-eq): h'·g·g = f'·g - f·g'
      step6 : ((h ′[ x ]) ·ℝ g x) ·ℝ g x ≡ ((f ′[ x ]) ·ℝ g x) -ℝ (f x ·ℝ (g ′[ x ]))
      step6 = step5 ∙ ap (λ z → ((f ′[ x ]) ·ℝ g x) -ℝ (z ·ℝ (g ′[ x ]))) h-g-eq

      -- Step 7: Reassociate left side to get h'·g²
      step7 : (h ′[ x ]) ·ℝ (g x ·ℝ g x) ≡ numerator
      step7 = sym (·ℝ-assoc (h ′[ x ]) (g x) (g x)) ∙ step6

      -- Step 8: Apply division isolation to extract h' = numerator/g²
  in division-isolate denom-nonzero step7
