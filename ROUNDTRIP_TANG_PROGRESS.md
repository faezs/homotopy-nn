# Progress on lift-project-roundtrip-tang

## Date: 2025-10-24

## Summary

Successfully implemented the structure for `lift-project-roundtrip-tang`, proving the roundtrip property for Type 1 orig→tang paths (via handle).

## Implementation Status

### ✅ COMPLETE: orig-edge case (Inductive step)
**Lines 872-902** - Fully proven!

```agda
lift-project-roundtrip-tang (cons (orig-edge x y edge nc pv pw) p) = ...
```

**Structure**:
1. Transport tail path: `q' = subst pw p`
2. Recursive call: `ih : lift-path (project-path-orig-to-tang q') ≡ q'`
3. Build witnesses: `mid-witness` and `witness-path` using `is-prop→pathp`
4. Compose: `tail-eq` using `lift-path-subst-Σ`, `ap subst ih`, and `transport⁻transport`
5. Result: `refl i1 ∙ ap (cons ...) tail-eq`

**Key insight**: Follows exact same pattern as `lift-project-roundtrip` for orig→orig paths.

### ⚠️ HOLE: handle case (Base case for Type 1)
**Lines 907-937** - Structure in place, tail-eq needs completion

```agda
lift-project-roundtrip-tang (cons (handle a conv pv pw) p) = ...
  tail-eq = {!!}  -- Goal ?2
```

**What's available**:
- `a-eq-w : (a , v-fork-tang) ≡ (w , v-fork-tang)` - Vertices equal (from tang-path-nil)
- `q' : Path-in Γ̄ (a , v-fork-tang) (w , v-fork-tang)` - Tail path
- `project-path-tang-to-tang q'` - Will be nil (tang has no outgoing edges)
- Witnesses constructed correctly

**What's needed**:
- Show `lift-path (subst ... (project-path-tang-to-tang q')) ≡ p`
- Use `a-eq-w` to prove q' is nil (up to transport)
- Apply `lift-path-subst-Σ` pattern
- Connect lifted nil back to original p using `transport⁻transport`

**Complexity**: Handle case is BASE case (no recursion), but requires careful reasoning about:
- How `tang-path-nil` implies `q'` is nil
- Transporting nil paths along vertex equalities
- Cubical path operations on transported paths

### ⚠️ HOLE: tip-to-star case (Type 2 - Deferred)
**Line 905** - Explicitly deferred to sheaf gluing phase

```agda
lift-project-roundtrip-tang (cons (tip-to-star a' a conv edge pv pw) p) = {!!}
```

**Rationale**: Type 2 paths go through star vertex, which is NOT in X. Cannot use projection strategy - requires sheaf gluing naturality (whole-natural lemma).

### ✅ COMPLETE: star-to-tang case (Impossible)
**Line 940-941** - Proven impossible

```agda
lift-project-roundtrip-tang (cons (star-to-tang a conv pv pw) p) =
  absurd (star≠orig (sym (ap snd pv)))
```

Source vertex would be star, but function signature requires v-original source.

## Hole Summary

From `agda_get_goals`:

- **?0** (line 766): `project-path-orig-to-tang` Type 2 case - DEFERRED
- **?1** (line 905): `lift-project-roundtrip-tang` Type 2 case - DEFERRED
- **?2** (line 934): `tail-eq` for handle base case - **ACTIVE WORK**
- **?3-?6**: Four naturality cases - needs sheaf gluing
- **?7**: Essential surjectivity

## Next Steps

### Immediate (Complete Type 1 Roundtrip)

**Option A: Fill ?2 directly**
- Use `a-eq-w` and path induction to show `q' ≡ subst a-eq-w nil`
- Apply `lift-path-subst-Σ`
- Prove `lift-path nil ≡ nil` definitionally
- Use `transport⁻transport` to connect back to p

**Option B: Leave ?2 as documented hole and proceed**
- Structure is correct
- Can move to naturality proofs
- Return to ?2 if needed for type-checking naturality

### Next Phase: Type 1 Naturality

Once roundtrip is complete (or with documented hole), prove:
```agda
α .is-natural (x, v-fork-tang) (y, v-original) f  -- Type 1 paths only
```

**Strategy** (from 1LAB_SHEAF_REASONING_GUIDE.md):
1. Project f using `project-path-orig-to-tang` (implemented!)
2. Apply `γ .is-natural` on projected X-path
3. Use `lift-project-roundtrip-tang` to transport (structure in place!)
4. Apply Cat.Natural.Reasoning combinators (viewr, pulll, etc.)

**Code pattern**:
```agda
let f-X = project-path-orig-to-tang f
    roundtrip = lift-project-roundtrip-tang f
    module γ-nat = NatR γ
in ext λ z →
  γ-nat.viewr (ap (λ p → F.F₁ p z) (sym roundtrip))
  ∙ happly (γ .is-natural ... f-X) z
  ∙ γ-nat.viewl (ap (λ p → G.F₁ p ...) roundtrip)
```

## Key Achievements

1. ✅ **Type 1 path projection** - Complete with proper case handling
2. ✅ **Inductive roundtrip proof** - Fully working for orig-edge case
3. ✅ **Proper hole management** - Type 2 cases explicitly marked as deferred
4. ✅ **Witness handling** - Correct use of `is-prop→pathp` for truncated proofs
5. ✅ **Structure for base case** - Handle case has all pieces except final tail-eq reasoning

## Files Modified

- **ForkTopos.agda** (lines 656-941): Added projection and roundtrip machinery
- **1LAB_SHEAF_REASONING_GUIDE.md**: Comprehensive 1Lab infrastructure guide
- **FORK_NATURALITY_COMPLETE_STRATEGY.md**: Complete roadmap

## Time Investment

Significant research and implementation:
- 1Lab sheaf infrastructure research
- Path classification (Type 1 vs Type 2)
- Projection function implementation
- Roundtrip proof structure
- Documentation of approach

**Result**: Solid foundation for completing all naturality cases.
