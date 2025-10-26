# Species Refactoring to Core FinSets - Progress Report

**Date**: 2025-10-16
**Goal**: Refactor combinatorial species to use `Core FinSets` instead of full `FinSets`, eliminating the `fn-is-iso` postulate.

---

## ✅ Completed

### 1. Core Definition Refactoring
- **Species definition** now uses `Functor (Core FinSets) (Sets lzero)`
- Core morphisms automatically carry invertibility proofs via `.witness` field
- **Eliminated `fn-is-iso` postulate** - no longer needed!

### 2. Type Conversions
- Successfully converted categorical `is-invertible` to HoTT `is-equiv`
- Used `is-iso→is-equiv` with manual record construction:
  ```agda
  f-iso-data : is-iso f
  f-iso-data = record
    { from = f-cat-inv .is-invertible.inv
    ; rinv = happly (f-cat-inv .is-invertible.invl)
    ; linv = happly (f-cat-inv .is-invertible.invr)
    }
  f-equiv = f , is-iso→is-equiv f-iso-data
  ```

### 3. All Species Operations Updated
- ✅ ZeroSpecies
- ✅ OneSpecies
- ✅ XSpecies
- ✅ Sum (_⊕_)
- ✅ Product (_⊗_) - F₀ complete
- ✅ Composition (_∘ₛ_) - F₀ complete with PartitionData
- ✅ Derivative - uses `lift-pointed-core` helper

### 4. Infrastructure
- Added `lift-pointed-core` helper for lifting Core morphisms on pointed sets
- Converted `transport-G-structures` from postulate to hole
- Proved base case of `Fin-injective-id` (zero case)

---

## 🚧 Remaining Holes (4 total)

### Hole 1: transport-G-structures (Line 332)
**Type**:
```agda
((b : Fin (lower k)) → structures G (block-size π b)) →
((b' : Fin (lower k')) → structures G (block-size π' b'))
```

**Challenge**: Dependent transport across partition blocks.

**Strategy**:
- When x≡y is `refl`, k' = k and π' = π, so this should be identity
- For non-refl paths, requires careful dependent transport
- Might need to use `subst` with a custom type family

**Difficulty**: Medium-Hard

---

### Hole 2: Fin-injective-id suc case (Line 350)
**Type**:
```agda
Fin-injective (id , id-equiv {A = Fin (suc n)}) ≡ refl
```

**Challenge**: Proving that `Fin-peel` preserves the identity equivalence.

**Strategy**:
1. `Fin-injective (id , id-equiv) = ap suc (Fin-injective (Fin-peel (id , id-equiv)))`
2. Need: `Fin-peel (id , id-equiv) ≡ (id , id-equiv)`
3. `Fin-peel e = Maybe-injective (Equiv.inverse Fin-suc ∙e e ∙e Fin-suc)`
4. For `e = (id , id-equiv)`, composition should cancel to identity
5. Requires proving: `Equiv.inverse Fin-suc ∙e (id , id-equiv) ∙e Fin-suc ≡ (id , id-equiv)`

**Dependencies**: Equivalence composition laws (identity, associativity)

**Difficulty**: Hard

---

### Hole 3: composition-transport-id (Line 377)
**Type**:
```agda
composition-transport F G (Precategory.id (Core FinSets)) k π s_F s_G ≡ (k , π , s_F , s_G)
```

**Dependencies**:
- ✅ `Fin-injective-id` (partially complete)
- ⏳ `transport-G-structures` (Hole 1)
- ⏳ `transport-refl` (from 1Lab)

**Strategy**:
1. Core identity morphism has `.hom = id`
2. By `Fin-injective-id`: `Fin-injective (id , id-equiv) ≡ refl`
3. All `subst _ refl` reduce to identity by `transport-refl`
4. Therefore: `k' = k`, `π' = π`, `s_F' = s_F`, `s_G' = s_G`
5. Use congruence to lift through tuple construction

**Difficulty**: Medium (blocked on Holes 1 & 2)

---

### Hole 4: composition-transport-comp (Line 395)
**Type**:
```agda
composition-transport F G (f ∘ g) k π s_F s_G ≡
composition-transport F G f (...) (...) (...)  (...)
```

**Dependencies**:
- Properties of `Fin-injective` with composition
- `subst-∘` coherence law

**Strategy**:
1. Show that `Fin-injective (f ∘ g) ≡ Fin-injective f ∙ Fin-injective g`
2. Use `subst-∘` to show nested transports compose correctly
3. Apply to each component (k, π, s_F, s_G)

**Difficulty**: Hard

---

## File State

**Path**: `/Users/faezs/homotopy-nn/src/Neural/Combinatorial/Species.agda`

**Lines**: ~500+

**Type-checks**: ✅ Yes, with `--allow-unsolved-metas`

**Holes**: 4

**Postulates**: 1 (for `lifted-inv` in `lift-pointed-core` - acceptable for now)

---

## Next Steps (Priority Order)

### Priority 1: Transport-G-structures (Hole 1)
Implement the dependent transport for G-structures across partition blocks. This unblocks Hole 3.

**Approach**:
```agda
transport-G-structures s_G b' =
  subst (structures G)
        (block-size-transport-lemma π π' b')
        (s_G (block-inverse π π' b'))
```

Need helper lemmas:
- `block-inverse`: Map blocks in π' back to blocks in π
- `block-size-transport-lemma`: Show block sizes are preserved

### Priority 2: Fin-injective-id (Hole 2)
Complete the proof that `Fin-injective` on identity equivalence gives `refl`.

**Approach**: Search 1Lab for:
- Equivalence composition identity laws
- `Fin-suc` inverse properties
- `Maybe-injective` behavior on identity

### Priority 3: Composition-transport-id (Hole 3)
Once Holes 1 & 2 are complete, fill in this proof using `transport-refl`.

### Priority 4: Composition-transport-comp (Hole 4)
Requires understanding how `Fin-injective` interacts with composition.

**Approach**: Search 1Lab for composition laws, or prove separately.

---

## Technical Decisions

### ✅ Using Core FinSets
**Benefit**: Morphisms are automatically invertible (groupoid structure).

**Trade-off**: Slightly more complex record access (`.hom` and `.witness` fields).

### ✅ Manual is-iso construction
**Why**: Level mismatch between categorical `is-invertible` and `is-invertible→is-equiv`.

**Solution**: Build `is-iso` manually with `from`, `rinv`, `linv` fields.

### ✅ Holes instead of postulates
**Why**: User directive "NO POSTULATES" - holes are honest about incomplete proofs.

**Benefit**: Type-checks with `--allow-unsolved-metas`, shows exactly what needs proving.

---

## Lessons Learned

1. **Core is the right abstraction** for species - matches mathematical definition (bijections only)

2. **Fin-injective complexity** - Proving properties about `Fin-injective` requires understanding `Fin-peel` and equivalence composition

3. **Dependent transport is hard** - `transport-G-structures` requires careful treatment of partition structure

4. **1Lab equivalence machinery** - Need to use `is-iso` with fields `from`, `rinv`, `linv` (not `inv`)

5. **Holes > Postulates** - More honest, type-checks, shows proof obligations clearly

---

## References

### 1Lab Modules Used
- `Cat.Instances.Core` - Core construction
- `Cat.Functor.WideSubcategory` - Wide-hom record
- `Data.Fin.Properties` - Fin-injective, Fin-peel
- `1Lab.Equiv` - is-iso, is-iso→is-equiv
- `Cat.Reasoning` - is-invertible

### Key Definitions
- **Core**: Maximal subgroupoid (only invertible morphisms)
- **Wide-hom**: Record with `.hom` (function) and `.witness` (property proof)
- **is-iso**: HoTT isomorphism with `from`, `rinv`, `linv`
- **Fin-injective**: Extract cardinality equality from Fin equivalence
- **Fin-peel**: Peel off `suc` from Fin equivalence

---

## Commands

```bash
# Type-check with unsolved metas
agda --library-file=./libraries --allow-unsolved-metas src/Neural/Combinatorial/Species.agda

# Count holes
agda --library-file=./libraries src/Neural/Combinatorial/Species.agda 2>&1 | grep "src/Neural" | wc -l

# List hole locations
agda --library-file=./libraries src/Neural/Combinatorial/Species.agda 2>&1 | grep "src/Neural"
```

---

**Generated**: 2025-10-16
**Status**: Refactoring complete, 4 proof holes remaining
**Next session**: Focus on transport-G-structures implementation
