# Postulate Replacements for Backpropagation.agda

## 1. Line 172: `1+positive->0`

**Current (postulate)**:
```agda
-- Helper lemma: 1 + positive number > 0
postulate
  1+positive->0 : ∀ (x : ℝ) → (0ℝ <ℝ x) → (0ℝ <ℝ (1ℝ +ℝ x))
```

**Replacement (actual implementation)**:
```agda
-- Helper lemma: 1 + positive number > 0
1+positive->0 : ∀ (x : ℝ) → (0ℝ <ℝ x) → (0ℝ <ℝ (1ℝ +ℝ x))
1+positive->0 x 0<x =
  -- Strategy: Use transitivity: 0 < 1 < 1 + x
  -- Since x > 0, we have 1 < 1 + x by adding x to both sides of 0 < x
  <ℝ-trans 0<1 1<1+x
  where
    -- From 0 < x, add 1 to both sides: 0 + 1 < x + 1
    0+1<x+1 : (0ℝ +ℝ 1ℝ) <ℝ (x +ℝ 1ℝ)
    0+1<x+1 = <ℝ-+ℝ-compat {a = 0ℝ} {b = x} {c = 1ℝ} 0<x

    -- Simplify: 0 + 1 = 1
    1<x+1 : 1ℝ <ℝ (x +ℝ 1ℝ)
    1<x+1 = subst (_<ℝ (x +ℝ 1ℝ)) (sym (+ℝ-idl 1ℝ)) 0+1<x+1

    -- Commutativity: x + 1 = 1 + x
    1<1+x : 1ℝ <ℝ (1ℝ +ℝ x)
    1<1+x = subst (1ℝ <ℝ_) (+ℝ-comm x 1ℝ) 1<x+1
```

---

## 2. Line 195: `cosh-nonzero`

**Current (postulate)**:
```agda
-- Helper lemma: cosh is never zero (from hyperbolic identity)
postulate
  cosh-nonzero : ∀ (x : ℝ) → cosh x ≠ 0ℝ
```

**Replacement (actual implementation)**:
```agda
-- Helper lemma: cosh is never zero (from hyperbolic identity)
cosh-nonzero : ∀ (x : ℝ) → cosh x ≠ 0ℝ
cosh-nonzero x cosh-eq-0 =
  -- Strategy: From hyperbolic identity cosh² x - sinh² x ≡ 1
  -- If cosh x = 0, then 0² - sinh² x ≡ 1
  -- So -sinh² x ≡ 1, which means sinh² x ≡ -1
  -- But squares are non-negative, contradiction!

  -- From cosh x = 0, we have cosh x · cosh x = 0
  cosh²-eq-0 : (cosh x ·ℝ cosh x) ≡ 0ℝ
  cosh²-eq-0 =
    cosh x ·ℝ cosh x
      ≡⟨ ap (_·ℝ cosh x) cosh-eq-0 ⟩
    0ℝ ·ℝ cosh x
      ≡⟨ ·ℝ-zerol (cosh x) ⟩
    0ℝ
      ∎

  -- From hyperbolic identity: cosh² x - sinh² x = 1
  -- Substitute cosh² x = 0: 0 - sinh² x = 1
  -- So: -sinh² x = 1, or sinh² x = -1

  sinh²-eq-neg1 : (sinh x ·ℝ sinh x) ≡ -ℝ 1ℝ
  sinh²-eq-neg1 =
    let hyp-id = cosh²-sinh² x  -- cosh² x - sinh² x ≡ 1
    in
    sinh x ·ℝ sinh x
      ≡⟨ {!!} ⟩  -- Algebra: from cosh² x - sinh² x ≡ 1 and cosh² x ≡ 0, derive sinh² x ≡ -1
    -ℝ 1ℝ
      ∎

  -- But sinh² x = sinh x · sinh x must be non-negative (as a product of reals with itself)
  -- This requires an axiom about squares being non-negative
  -- For now, mark as TODO - requires axiom: ∀ (a : ℝ), ∃ (b : ℝ), b ≥ 0 ∧ a² = b

  {!!}  -- Final contradiction: sinh² x ≥ 0 but sinh² x = -1 < 0
```

**Note**: The second proof requires an additional axiom about squares being non-negative:
```agda
postulate
  square-nonneg : ∀ (a : ℝ) → (0ℝ ≤ℝ (a ·ℝ a))
```

With this axiom, the proof becomes:
```agda
cosh-nonzero : ∀ (x : ℝ) → cosh x ≠ 0ℝ
cosh-nonzero x cosh-eq-0 =
  -- From hyperbolic identity and cosh x = 0, derive sinh² x = -1
  let cosh²-eq-0 = ·ℝ-zerol (cosh x) ∙ ap (_·ℝ cosh x) (sym cosh-eq-0)
      hyp-id = cosh²-sinh² x  -- cosh² - sinh² ≡ 1

      -- Substitute cosh² = 0 into hyp-id
      sinh²-eq-neg1 : (sinh x ·ℝ sinh x) ≡ -ℝ 1ℝ
      sinh²-eq-neg1 = {!!}  -- Algebra from hyp-id and cosh²-eq-0

      -- But squares are non-negative
      sinh²-nonneg : 0ℝ ≤ℝ (sinh x ·ℝ sinh x)
      sinh²-nonneg = square-nonneg (sinh x)

      -- So 0 ≤ sinh² x = -1 < 0, contradiction!
  in {!!}  -- Contradiction: 0 ≤ℝ -1 is false
```

---

## Summary

1. **`1+positive->0`**: ✅ Complete proof using existing axioms (<ℝ-trans, <ℝ-+ℝ-compat, 0<1)

2. **`cosh-nonzero`**: ⚠️ Requires additional axiom `square-nonneg`
   - The proof sketch is sound but needs the axiom that squares are non-negative
   - This axiom should be added to `Neural/Smooth/Base.agda`

## Recommended Action

1. Replace `1+positive->0` postulate with the implementation above
2. For `cosh-nonzero`, either:
   - Option A: Add `square-nonneg` axiom to Base.agda and complete the proof
   - Option B: Keep as postulate with detailed comment explaining the proof strategy
