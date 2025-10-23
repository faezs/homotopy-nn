# restrict-full Proof Session - October 2024

## Goal

Complete the proof of `restrict-full` in `ForkTopos.agda`, showing that the restriction functor `Φ : DNN-Topos → PSh(X)` is full (every natural transformation on X lifts to Γ̄).

## Progress Summary

### ✅ Completed

1. **Impossible edge case helpers** (lines 609-619)
   - `star≠orig`: v-fork-star ≠ v-original
   - `star≠tang`: v-fork-star ≠ v-fork-tang
   - `tang≠orig`: v-fork-tang ≠ v-original (NEW)

2. **Impossible patch cases eliminated** (lines 688-700)
   - **star→orig case** (line 688): Path from fork-star to original impossible
     - Uses `star-only-to-tang` + `tang-path-nil` + `tang≠orig`
     - Proof: star edges only go to tang, tang has no outgoing, derives contradiction
   - **tang→orig case** (line 689): Path from fork-tang to original impossible
     - Uses `tang-path-nil` + `tang≠orig`
     - Proof: tang has no outgoing edges, so any path must be nil, but that requires tang=orig

### ⏸️ Remaining Holes (4 total)

#### 1. Patch Compatibility: orig→orig case (line 698)

**Goal**: `G.F₁ g (γ .η (...) (F.F₁ f x)) ≡ γ .η (...) (F.F₁ (g ++ f) x)`

**Challenge**:
- Both `V = (v-node, v-original)` and `W = (fst₁, v-original)` are in X
- `g : Path-in Γ̄ (fst₁, v-original) (v-node, v-original)` is a Γ̄-path
- `γ : restrict(F) => restrict(G)` expects X-Category morphisms
- Need to relate Γ̄-paths to X-paths for original vertices

**Strategy** (documented in code):
1. Use functoriality: `F.F₁ (g ++ f) = F.F₁ f ∘ F.F₁ g`
2. Since both vertices are original (in X), `g` corresponds to unique X-morphism
3. Apply naturality of γ on the X-morphism
4. Pattern from 1Lab `map-patch`: `sym (eta .is-natural _ _ _ $ₚ _) ∙ ap (eta .η _) (x .patch f hf g hgf)`

**Key insight**: Need to convert between Γ̄-paths and X-paths using `lift-path` or similar

#### 2. α .is-natural (line 705)

**Goal**: Prove naturality of the constructed transformation α : F => G

**Construction**:
- α .η on original: uses γ directly
- α .η on fork-star: uses `Gsh .whole (lift false) (patch-at-star x)`
- α .η on fork-tang: uses γ directly

**Challenge**: Prove naturality square commutes for all morphisms in Γ̄

#### 3. restrict .F₁ α ≡ γ (line 660)

**Goal**: Prove that restricting the lifted α gives back the original γ

**Should follow** from construction of α on original/tang vertices

#### 4. restrict-ess-surj (separate proof)

**Goal**: Every presheaf P on X extends to a sheaf F on Γ̄

**Construction** (outlined in comments):
- F(v) = P(v, proof) for v non-star
- F(A★) = lim_{tips to A★} P(tip)

## Commits

1. **c831145**: "Complete impossible cases in patch compatibility proof"
   - Added `tang≠orig` helper
   - Eliminated star→orig and tang→orig impossible cases
   - Left orig→orig naturality with TODO comment

## Technical Insights

### Vertex Type Discrimination Pattern

All impossible case proofs follow this pattern:
```agda
1. Edge constructor implies vertex equality (via ap snd of path equality)
2. Apply helper (star≠orig, tang≠orig, star≠tang) to derive ⊥
3. Use absurd to inhabit any type from ⊥
```

Example:
```agda
absurd (tang≠orig (ap snd (tang-path-nil g)))
```

### 1Lab Patterns for Patch Compatibility

From `Cat.Site.Base`:
```agda
map-patch eta x .patch f hf g hgf =
  sym (eta .is-natural _ _ _ $ₚ _) ∙
  ap (eta .η _) (x .patch f hf g hgf)
```

Key: Use naturality of the transformation + functoriality of the patch

### Sheaf Gluing Workflow

From `Cat.Site.Sheafification`:
1. **Generators**: `inc`, `map`, `glue`
2. **Relations**: `map-id`, `map-∘`, `inc-natural`, `sep`, `glues`
3. **Elimination**: `Sheafify-elim-prop` for propositions + locality

## Next Steps

### Immediate

1. **Prove orig→orig patch compatibility**
   - Find correspondence between Γ̄-paths and X-paths for original vertices
   - May need to define or use existing conversion `Γ̄-path-to-X-path`
   - Apply γ naturality on the X-morphism

2. **Prove α .is-natural**
   - Case split on source/target vertex types
   - Use sheaf gluing properties for fork-star cases
   - Use direct naturality of γ for original/tang cases

3. **Prove restrict .F₁ α ≡ γ**
   - Should follow by construction + function extensionality
   - Use Nat-path

4. **Prove restrict-ess-surj**
   - Define extension functor explicitly
   - Prove it's a sheaf
   - Prove restriction gives back original presheaf

### Research Needed

- Understand relationship between Γ̄-Category and X-Category morphisms
- Check if there's a canonical projection or section
- Look for examples in 1Lab of similar restriction/extension pairs

## Files Modified

- `src/Neural/Graph/ForkTopos.agda` (lines 609-705)
  - Added 1 helper (tang≠orig)
  - Eliminated 2 impossible cases
  - Added documentation for remaining naturality proof

## Statistics

- **Holes eliminated**: 2 (star→orig, tang→orig)
- **Holes remaining**: 4 (orig→orig, α naturality, restrict roundtrip, ess-surj)
- **Lines added**: ~20
- **Commits**: 1

## Key References

- **1Lab Cat.Site.Base**: `map-patch` pattern for patch compatibility
- **1Lab Cat.Functor.Compose**: `_▸_` whiskering for natural transformation composition
- **1Lab Cat.Site.Sheafification**: Sheaf gluing with `whole` and `glues`
- **Paper Belfiore & Bennequin 2022**: Section 1.5, Corollary 749 (DNN-Topos ≃ PSh(X))
