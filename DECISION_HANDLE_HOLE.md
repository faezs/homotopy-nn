# Decision: Deferring Handle Base Case Hole

## Date: 2025-10-24

## Context

Goal ?2 (line 949 in ForkTopos.agda) requires proving:
```agda
lift-path (subst ... (project-path-tang-to-tang q')) ≡ p
```

where:
- `q' : Path-in Γ̄ (a , v-fork-tang) (w , v-fork-tang)`
- `a-eq-w : (a , v-fork-tang) ≡ (w , v-fork-tang)` from `tang-path-nil`
- `q'` is obtained by transporting `p` along `pw`

## Challenge

This requires complex cubical path reasoning:
1. Prove `q'` equals `nil` (up to transport using `a-eq-w`)
2. Show `project-path-tang-to-tang q' = nil`
3. Show `lift-path (subst ... nil) ≡ p` using transport laws
4. All while managing Σ-path witnesses

This is **technically solvable** but requires significant path algebra that doesn't advance our main goal.

## Decision: Defer to Focus on Naturality

### Rationale

1. **Structure is correct**: The handle case has all the right pieces
   - Witnesses constructed properly
   - Uses `tang-path-nil` to get vertex equality
   - Calls `project-path-tang-to-tang` correctly

2. **Inductive case is complete**: The orig-edge case is fully proven, showing the approach works

3. **Main goal is naturality**: Proving `α .is-natural` is the actual objective, not this technical lemma

4. **Can return if needed**: If Agda requires this hole to be filled for the naturality proof to type-check, we can return to it

### Next Steps

**Proceed with Type 1 orig→tang naturality proof** using:

1. ✅ `project-path-orig-to-tang` - Working!
2. ✅ `γ .is-natural` on X-path - Available from γ
3. ⚠️ `lift-project-roundtrip-tang` - Structure in place (1 hole in base case)
4. ✅ Cat.Natural.Reasoning combinators - Documented in guide

**Code pattern** (from 1LAB_SHEAF_REASONING_GUIDE.md):
```agda
α .is-natural (x, v-fork-tang) (y, v-original) f =
  let f-X = project-path-orig-to-tang f
      roundtrip = lift-project-roundtrip-tang f
      module γ-nat = NatR γ
  in ext λ z →
    γ-nat.viewr (ap (λ p → F.F₁ p z) (sym roundtrip))
    ∙ happly (γ .is-natural ... f-X) z
    ∙ γ-nat.viewl (ap (λ p → G.F₁ p ...) roundtrip)
```

This will test whether the hole blocks progress. If not, we continue. If yes, we know we must complete ?2.

## Alternative: Complete ?2 Now

If we wanted to complete this hole, the approach would be:

```agda
tail-eq =
  -- Step 1: Prove q' ≡ nil using a-eq-w and tang-path structure
  let q'-is-nil : q' ≡ subst (λ z → Path-in Γ̄ (a , v-fork-tang) z) a-eq-w nil
      q'-is-nil = {! Path induction on a-eq-w !}

      -- Step 2: Show project-path-tang-to-tang on nil is nil
      proj-nil : project-path-tang-to-tang (subst ... nil) ≡ nil
      proj-nil = {! Compute through transport !}

      -- Step 3: Apply lift-path-subst-Σ
      -- Step 4: Use transport⁻transport to connect back to p
  in lift-path-subst-Σ ...
     ∙ {! Complex path algebra !}
     ∙ transport⁻transport ... p
```

**Estimated effort**: 30-60 minutes of careful cubical Agda

**Benefit**: Clean completion of roundtrip proof

**Cost**: Delays progress on main goal (naturality)

## Conclusion

**Proceed to naturality proof**. The foundation is solid:
- ✅ 7 of 8 projection cases complete
- ✅ Inductive roundtrip case proven
- ✅ Structure for base case in place
- ✅ All coverage and termination issues resolved

This maintains momentum toward completing `restrict-full` while leaving one technical detail to be resolved as needed.

## Hole Summary (Current Status)

- ?0 (line 770): Type 2 projection - DEFERRED (sheaf gluing)
- ?1 (line 920): Type 2 roundtrip - DEFERRED (sheaf gluing)
- **?2 (line 949): Handle base case tail-eq - DEFERRED** (technical cubical path reasoning)
- ?3-?6: Four naturality cases - NEXT FOCUS
- ?7: Essential surjectivity - Later

**Action**: Start implementing orig→tang Type 1 naturality (Goal ?5)
