# Complete Strategy for Fork Topos Naturality Proofs

## Date: 2025-10-24

## Executive Summary

After comprehensive research into the Fork* modules and 1Lab infrastructure, we have identified the fundamental challenge:

**NOT ALL PATHS BETWEEN NON-STAR VERTICES STAY IN THE X SUBCATEGORY!**

Specifically, paths can go through star vertices even when both endpoints are non-star. This invalidates the naive projection strategy and requires case-by-case handling.

## The Core Issue

### What We Initially Thought

The NATURALITY_STRATEGY.md initially claimed:
> "Both vertices are in X (non-star) → Can project f to X-path"

This is **WRONG** for some paths!

### The Reality

**Edge Structure** (from ForkCategorical.agda):
- `orig-edge`: Original vertex → Original vertex
- `tip-to-star`: Original vertex → Star vertex (convergent node)
- `star-to-tang`: Star vertex → Tang vertex (same node)
- `handle`: Original vertex → Tang vertex (same node)

**Key Constraints**:
- Star only connects to tang of SAME node (`star-to-tang` requires source.node = target.node)
- Handle connects orig to tang of SAME node
- Tang has NO outgoing edges (`tang-no-outgoing`)

## Path Classification

### Paths from Original to Original (SOLVED ✓)

**Structure**: Only uses `orig-edge` constructor
**Reason**: If path touches star, it must go to tang (by `star-only-to-tang`), and tang has no outgoing edges (by `tang-no-outgoing`), so cannot reach another original vertex.
**Projection**: `project-path-orig` - COMPLETE
**Roundtrip**: `lift-project-roundtrip` - COMPLETE
**Naturality**: PROVEN using path projection + γ.is-natural

### Paths from Tang to Tang (SOLVED ✓)

**Structure**: Only `nil` (tang has no outgoing)
**Naturality**: PROVEN using F-id and G-id

### Paths from Tang to Original (IMPOSSIBLE ✓)

Tang has no outgoing edges, so only nil is possible, which requires tang = original (type mismatch).
**Naturality**: PROVEN via `absurd (tang-no-outgoing e)`

### Paths from Star to Original (IMPOSSIBLE ✓)

Star only goes to tang (same node), tang has no outgoing, cannot reach original.
**Naturality**: PROVEN via star-only-to-tang + tang-path-nil + tang≠orig

### Paths from Original to Tang (**COMPLEX - 2 TYPES**)

This is where the strategy diverges!

#### Type 1: Via Handle (X-paths) ✓ Partially Implemented

**Structure**:
```
(y, v-original) --orig-edges--> (x, v-original) --handle--> (x, v-fork-tang)
```

**Properties**:
- Prefix uses only orig-edges (stays in X)
- Final edge is `handle` (also in X - both endpoints non-star!)
- Can be projected using `project-path-orig-to-tang`

**Implementation Status**: PARTIAL
- Base case recursion works
- Handle edge case works
- **Remaining**: Type 2 case (hole at line 766)

**Naturality Strategy**:
1. Use `project-path-orig-to-tang` to get X-path
2. Apply γ.is-natural on the X-path
3. Use roundtrip property (needs to be proven for orig-to-tang)

#### Type 2: Via Star (NON-X-paths) ⚠️ BLOCKED

**Structure**:
```
(y, v-original) --orig-edges--> (a', v-original) --tip-to-star--> (x, v-fork-star) --star-to-tang--> (x, v-fork-tang)
```

**Properties**:
- Goes through `(x, v-fork-star)` which is NOT in X!
- **CANNOT** be projected to X-Category
- Requires `x` to be convergent

**Why This Exists**: In convergent networks, tips can go to star, then star goes to tang. This is a VALID Γ̄-path!

**Example Network**:
```
v₁ ---→ w★ ---→ w_tang
v₂ ---→ w★
        ↑
        w (original vertex, convergent)
```

Path from `(v₁, v-original)` to `(w, v-fork-tang)`:
- `(v₁, v-original) --tip-to-star--> (w, v-fork-star) --star-to-tang--> (w, v-fork-tang)`

**Naturality Strategy**: NEEDS SHEAF GLUING
- Cannot use γ.is-natural (γ is only defined on X)
- Must use properties of `Gsh .whole` (sheaf gluing at star vertex)
- Similar to orig→star case
- **Research needed**: How does `whole` interact with naturality?

### Paths from Original to Star ⚠️ NEEDS SHEAF GLUING

**Structure**:
```
(y, v-original) --orig-edges--> (a', v-original) --tip-to-star--> (x, v-fork-star)
```

**Properties**:
- Ends at star vertex
- Star vertex NOT in X
- α.η at star uses sheaf gluing: `α .η (v, v-fork-star) = λ x → Gsh .whole (lift false) (patch-at-star x)`

**Naturality Strategy**:
- LHS: `(λ z → Gsh .whole ... (patch-at-star z)) ∘ F.F₁ f`
- RHS: `G.F₁ f ∘ γ .η ((y, v-original), inc tt)`
- Need to show these are equal using sheaf gluing properties

### Paths from Star to Star ⚠️ MAY BE IMPOSSIBLE

**Analysis**: Star only goes to tang (same node), so paths from one star to another star seem impossible unless they're the same star (nil).

**Possible cases**:
1. `nil`: x = y (identity case)
2. Length ≥ 2: Would need star → tang → ??? → star, but tang has no outgoing!

**Expected**: Only nil is possible, making this an identity case like tang→tang.

### Paths from Star to Tang ⚠️ NEEDS SHEAF GLUING

**Structure**:
```
(x, v-fork-star) --star-to-tang--> (x, v-fork-tang)  (single edge, requires x-node = y-node)
```

or potentially longer paths through original vertices?

**Naturality Strategy**: Similar to orig→star, involves sheaf gluing.

## Implementation Roadmap

### Phase 1: Complete Type 1 Paths (In Progress)

1. ✅ Implement `project-path-orig-to-tang` for handle case
2. ⚠️ Fill hole at line 766 for tip-to-star case (Type 2)
   - **Decision**: Keep as explicit hole with documentation
   - **Rationale**: Type 2 requires sheaf gluing, different proof strategy
3. Implement `lift-project-roundtrip-orig-to-tang` (roundtrip for orig-to-tang X-paths)
4. Prove naturality for Type 1 orig→tang paths using projection

### Phase 2: Research Sheaf Gluing (Critical)

**Key Questions**:
1. How does `Gsh .whole` behave under functoriality?
2. Is there a naturality lemma for `whole`?
3. Can we factor out common proof pattern for all sheaf cases?

**Resources to Check**:
- `Cat.Site.Base` - sheaf axioms
- `Cat.Site.Sheafification` - sheafification properties
- 1Lab documentation on sheaves and naturality

**Expected Pattern**: Something like
```agda
whole-natural : ∀ {F G} (η : F => G) (cover : Covering X) (patch : Patch F cover)
              → η .η X (whole F cover patch) ≡ whole G cover (map-patch η patch)
```

### Phase 3: Implement Sheaf Gluing Cases

Once we understand sheaf gluing naturality:

1. **orig→star naturality**
   - LHS involves sheaf gluing at star
   - RHS involves γ at original
   - Connect using sheaf properties

2. **star→tang naturality**
   - Similar to orig→star but reversed

3. **orig→tang Type 2 naturality**
   - Goes through star, combines both patterns

4. **star→star naturality**
   - Prove it's only nil (identity case) OR
   - Handle sheaf gluing if non-nil paths exist

### Phase 4: Complete the Proof

1. Combine all naturality cases
2. Prove `restrict-ess-surj` (essential surjectivity)
3. Conclude `restrict` is an equivalence

## Key Insights from Research

### From ForkCategorical.agda

1. **Edge constraints are strict**: Each edge type has specific source/target requirements
2. **Tang is terminal locally**: No outgoing edges from tang vertices
3. **Star is a junction**: Collects inputs (tips) and outputs to tang only
4. **Handle bypasses fork**: Direct path from original to tang, skipping star

### From ForkPoset.agda

1. **X-edges are Γ̄-edges**: `X .Graph.Edge (v , _) (w , _) = ForkEdge v w`
2. **Syntactic identity**: X-paths ARE Γ̄-paths (with non-star witnesses)
3. **is-non-star is propositional**: Witnesses don't matter for path structure

### From Cat.Site.Base (1Lab)

1. **Sheaf gluing**: `whole : (cover : Covering X) → Patch F cover → F.₀ X`
2. **Patch structure**: Local data + compatibility on overlaps
3. **Naturality of patches**: `map-patch` from 1Lab (line 293)

## Success Metrics

- ✅ **No postulates** for projection functions (achieved for Type 1)
- ✅ **Explicit holes** for Type 2 cases with clear documentation
- ⏸️ **Naturality proofs** using projection OR sheaf gluing (in progress)
- ⏸️ **Complete `restrict-full`** proof (2/7 naturality cases proven)
- ⏸️ **Complete `restrict-ess-surj`** proof (not started)

## Current Status

**Goals**: 6 total
- 1 in `project-path-orig-to-tang` (Type 2 path - needs sheaf gluing research)
- 4 naturality cases (orig→star, star→star, orig→tang, star→tang)
- 1 essential surjectivity

**Next Step**: Research sheaf gluing naturality in 1Lab to unlock all remaining cases.

## Files Created

1. `ORIG_TANG_PATH_ANALYSIS.md` - Path type classification
2. `ORIG_TANG_NATURALITY_STRATEGY.md` - Implementation approaches
3. `FORK_NATURALITY_COMPLETE_STRATEGY.md` - This document (comprehensive guide)

## Updated Files

1. `NATURALITY_STRATEGY.md` - Corrected orig→tang case analysis
2. `ForkTopos.agda` - Added `project-path-orig-to-tang` (partial), `project-path-tang-to-tang`
