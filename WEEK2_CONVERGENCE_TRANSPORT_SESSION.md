# Week 2 Continuation: Convergence Transport Implementation

**Date**: 2025-10-31
**Session Focus**: Implement convergence transport for compositional networks
**Result**: ⚠️ **Partial Progress** - Infrastructure complete, yes cases work, no cases remain
**Status**: 24/45 goals remaining (53%)

---

## 🎯 Mission

**Goal**: Fill compositional `detect-convergent` holes by implementing convergence transport between subgraphs and coproducts.

**Challenge**: When a node is convergent in subgraph `g`, prove it remains convergent in coproduct `g +ᴳ f`.

---

## 🏗️ Infrastructure Built

### GraphCoproduct.agda: Convergence Transport

**New exports** (lines 153-193):

```agda
-- Left inclusion preserves convergence
inl-convergent : ∀ {o ℓ} {G H : Graph o ℓ}
               {G-oriented : is-oriented G}
               {GH-oriented : is-oriented (G +ᴳ H)}
               {G-discrete : ∀ (x y : G .Graph.Node) → Dec (x ≡ y)}
               {GH-discrete : ∀ (x y : (G +ᴳ H) .Graph.Node) → Dec (x ≡ y)}
               {v : G .Graph.Node}
               → ForkConstruction.is-convergent G G-oriented G-discrete v
               → ForkConstruction.is-convergent (G +ᴳ H) GH-oriented GH-discrete (inl v)

-- Right inclusion preserves convergence
inr-convergent : ∀ {o ℓ} {G H : Graph o ℓ} ...
               → ForkConstruction.is-convergent H H-oriented H-discrete v
               → ForkConstruction.is-convergent (G +ᴳ H) GH-oriented GH-discrete (inr v)
```

**Key technique**: Use `inl-inj` and `inr-inj` from `Data.Sum.Properties` to prove distinctness is preserved:

```agda
distinct = λ eq → ForkConstruction.is-convergent.distinct conv-G (inl-inj eq)
```

---

## 🔧 ForkExtract.agda Integration

### Import Helpers

```agda
open import Neural.Compile.GraphCoproduct using (_+ᴳ_; inl-convergent; inr-convergent)
```

### Compositional detect-convergent Pattern

**Composition** (lines 476-489):

```agda
detect-convergent (f ⊙ g) v =
  elim-coproduct
    (λ v → Dec (∥ is-convergent ... v ∥))
    (λ v-g → 1Lab.Type.case detect-convergent g v-g of λ
      { (yes conv-g) → yes (∥-∥-map inl-convergent conv-g)  -- ✅ Works!
      ; (no not-conv-g) → no {!!}  -- ⚠️ TODO: inverse transport
      })
    (λ v-f → 1Lab.Type.case detect-convergent f v-f of λ
      { (yes conv-f) → yes (∥-∥-map inr-convergent conv-f)  -- ✅ Works!
      ; (no not-conv-f) → no {!!}  -- ⚠️ TODO: inverse transport
      })
    v
```

**Same pattern** applied to `Fork` and `Join` (lines 495-523).

---

## ✅ What Works

### Yes Cases: Forward Transport

**Pattern**: If `v-g` is convergent in `g`, then `inl v-g` is convergent in `g +ᴳ f`

```agda
(yes conv-g) → yes (∥-∥-map inl-convergent conv-g)
```

**Why it works**:
- `inl-convergent` lifts convergence witness from `g` to coproduct
- `∥-∥-map` preserves truncation
- Edges are preserved: `(G +ᴳ H) .Edge (inl x) (inl y) = G .Edge x y`

---

## ⚠️ What Remains: No Cases

### Inverse Transport Challenge

**Problem**: If `inl v-g` is NOT convergent in coproduct, prove `v-g` is not convergent in `g`.

**Needed**: Inverse function

```agda
inl-convergent-inv : ¬ is-convergent (G +ᴳ H) (inl v)
                   → ¬ is-convergent G v
```

**Strategy**: Prove by contraposition - if `v` IS convergent in `G`, then `inl v` IS convergent in coproduct (which contradicts the assumption).

**Hole count**: 6 no-case holes (2 each for Composition, Fork, Join)

---

## 📊 Progress Breakdown

### Holes Status

| Category | Before | After | Change |
|----------|--------|-------|--------|
| Primitive detect-convergent | 6 | 6 | unchanged |
| Compositional detect-convergent | 6 | 6 | ⚠️ replaced with no-cases |
| Extract-tines compositional | 6 | 6 | unchanged |
| Extract-gluing compositional | 6 | 6 | unchanged |
| **Total** | **24** | **24** | **0** |

**Status**: Infrastructure built, but holes not yet reduced.

---

## 🎓 Technical Challenges Solved

### Challenge 1: Module Qualification for ForkConstruction

**Problem**: `ForkConstruction.is-convergent` not in scope

**Solution**: Import qualified module

```agda
open import Neural.Graph.Fork.Fork
```

Then use full path in type signatures.

### Challenge 2: Injectivity Proofs

**Problem**: Need to prove `inl s₁ ≡ inl s₂ → s₁ ≡ s₂` for distinctness

**Solution**: Use 1Lab's `inl-inj` and `inr-inj` from `Data.Sum.Properties`

```agda
distinct = λ eq → ForkConstruction.is-convergent.distinct conv-G (inl-inj eq)
```

### Challenge 3: Ambiguous case_of_

**Problem**: Both `1Lab.Prelude.case_of_` and `1Lab.Type.case_of_` in scope

**Solution**: Use qualified name

```agda
1Lab.Type.case detect-convergent g v-g of λ { ... }
```

---

## 🚧 Next Steps

### Immediate: Fill No Cases (6 holes)

**Option 1**: Prove inverse transport (rigorous)

```agda
inl-not-convergent : ∀ {G H : Graph o ℓ} {v : G .Graph.Node}
                   → ¬ is-convergent (G +ᴳ H) (inl v)
                   → ¬ is-convergent G v
inl-not-convergent not-conv-coproduct conv-g =
  not-conv-coproduct (inl-convergent conv-g)
```

**Option 2**: Use absurdity for now (pragmatic)

```agda
(no not-conv-g) → no λ conv → absurd {!!}
```

### Medium: Primitive detect-convergent (6 holes)

- Dense, Conv1D, MaxPool, AvgPool: Construct `is-convergent` witnesses
- Activation, BatchNorm: Prove no convergence (1-to-1 mappings)

### Long: Extraction (12 holes)

- Extract-tines: 6 holes for compositional cases
- Extract-gluing: 6 holes for compositional cases

---

## 📁 Files Modified

### Created/Modified

**`/Users/faezs/homotopy-nn/src/Neural/Compile/GraphCoproduct.agda`**
- Added imports: `Data.Dec`, `is-oriented`, `ForkConstruction`
- Implemented `inl-convergent` and `inr-convergent` (lines 153-193)
- Uses `inl-inj`/`inr-inj` for distinctness proofs
- **Type-check**: ✅ 0 goals

**`/Users/faezs/homotopy-nn/src/Neural/Compile/ForkExtract.agda`**
- Import convergence helpers (line 52)
- Implemented compositional detect-convergent with yes cases (lines 476-523)
- 6 no-case holes remain
- **Type-check**: ⚠️ 24 goals

---

## 💡 Key Insights

### 1. Forward Transport is Easy

If you can construct the witness in the subgraph, you can lift it to the coproduct:

```agda
inl-convergent : is-convergent G v → is-convergent (G +ᴳ H) (inl v)
```

Edges are preserved by coproduct construction.

### 2. Inverse Transport Needs Contraposition

The no case requires proving the contrapositive:

```
¬ P → ¬ Q   ≡  Q → P
```

So: `¬ conv-coproduct → ¬ conv-g` becomes `conv-g → conv-coproduct`.

### 3. Truncation Commutes with Mapping

`∥-∥-map` from 1Lab allows applying functions inside propositional truncation:

```agda
∥-∥-map : (A → B) → ∥ A ∥ → ∥ B ∥
```

This is essential for lifting convergence witnesses.

---

## 🔮 Vision: Complete Pipeline

```
NeuralNet m n
    ↓ build-graph (compositional with _+ᴳ_)
Graph with NetworkNode vertices
    ↓ detect-convergent (yes cases working!)
Dec (∥ is-convergent v ∥)
    ↓ ForkConstruction
ForkVertex = Node ⊎ fork-stars ⊎ fork-tangs
    ↓ extract-tines (TODO)
    ↓ extract-gluing (TODO)
ForkStructure
    ↓ TritonEmit (Week 3)
Verified GPU code
```

**Week 2 Progress**: Convergence transport infrastructure complete, yes cases working! 🎉

---

## 🏁 Session Summary

**What we built**:
- ✅ Convergence transport helpers in GraphCoproduct
- ✅ Yes cases for compositional detect-convergent
- ✅ Clean pattern using `elim-coproduct` + `1Lab.Type.case_of_`

**What we learned**:
- Forward transport (yes case) is straightforward
- Inverse transport (no case) needs careful proof
- 1Lab provides `inl-inj`/`inr-inj` for sum injectivity
- Module qualification resolves ambiguity

**What's next**:
- Fill 6 no-case holes with contrapositive proofs
- Then tackle primitive convergence detection (6 holes)
- Target: 18/45 holes remaining (60% → 67%)

---

**Session End**: 2025-10-31
**Achievement**: Infrastructure complete, pattern established!
**Blocker**: Need inverse convergence transport for no cases

