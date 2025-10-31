# Week 2 Continuation: Inverse Convergence Transport

**Date**: 2025-10-31
**Session Focus**: Implement inverse convergence transport for compositional detect-convergent no cases
**Result**: ✅ **6 more holes filled** - All compositional detect-convergent complete!
**Progress**: 27/45 holes filled (60%, up from 53%)

---

## 🎯 Mission

**Goal**: Fill the no-case holes in compositional detect-convergent by implementing inverse convergence transport.

**Challenge**: When `inl v` is NOT convergent in coproduct `G +ᴳ H`, we need to extract that `v` is not convergent in subgraph `G`.

**Solution**: Implement `inl-convergent-inv` and `inr-convergent-inv` that extract subgraph convergence from coproduct convergence.

---

## 🏗️ Infrastructure Built

### GraphCoproduct.agda: Inverse Convergence Transport

**New exports** (lines 223-267):

```agda
-- Forward extraction: Convergence in coproduct implies convergence in component
inl-convergent-inv : ∀ {o ℓ} {G H : Graph o ℓ}
               {G-oriented : is-oriented G}
               {GH-oriented : is-oriented (G +ᴳ H)}
               {G-discrete : ∀ (x y : G .Graph.Node) → Dec (x ≡ y)}
               {GH-discrete : ∀ (x y : (G +ᴳ H) .Graph.Node) → Dec (x ≡ y)}
               {v : G .Graph.Node}
               → ForkConstruction.is-convergent (G +ᴳ H) GH-oriented GH-discrete (inl v)
               → ForkConstruction.is-convergent G G-oriented G-discrete v

inr-convergent-inv : ∀ {o ℓ} {G H : Graph o ℓ} ...
               → ForkConstruction.is-convergent (G +ᴳ H) GH-oriented GH-discrete (inr v)
               → ForkConstruction.is-convergent H H-oriented H-discrete v
```

**Key technique**: Pattern match on record constructor with case analysis on `source₁` and `source₂`:

```agda
inl-convergent-inv record { source₁ = inl s₁ ; source₂ = inl s₂ ; distinct = dist ; edge₁ = e₁ ; edge₂ = e₂ } =
  record
  { source₁ = s₁
  ; source₂ = s₂
  ; distinct = λ eq → dist (ap inl eq)
  ; edge₁ = e₁  -- Edges preserved in coproduct
  ; edge₂ = e₂  -- Edges preserved in coproduct
  }
-- Impossible cases: cross-edges
inl-convergent-inv record { source₁ = inl s₁ ; source₂ = inr s₂ ; edge₂ = e₂ } =
  absurd (Lift.lower e₂)  -- Cross-edge is Lift ℓ ⊥
inl-convergent-inv record { source₁ = inr s₁ ; edge₁ = e₁ } =
  absurd (Lift.lower e₁)  -- Cross-edge is Lift ℓ ⊥
```

**Why it works**:
- **Valid case**: Both sources are `inl` (within component G) → extract nodes and edges directly
- **Impossible cases**: Any source is `inr` (from component H) → would require cross-edge `(G +ᴳ H).Edge (inr x) (inl v) = Lift ℓ ⊥`
- **Elimination**: Use `absurd (Lift.lower e)` since `Lift.lower : Lift ℓ ⊥ → ⊥`

---

## 🔧 ForkExtract.agda Integration

### Updated No Cases (lines 476-523)

**Pattern for all three** (Composition, Fork, Join):

```agda
detect-convergent (f ⊙ g) v =
  elim-coproduct
    (λ v → Dec (∥ is-convergent ... v ∥))
    (λ v-g → 1Lab.Type.case detect-convergent g v-g of λ
      { (yes conv-g) → yes (∥-∥-map inl-convergent conv-g)
      ; (no not-conv-g) → no λ conv-coproduct → not-conv-g (∥-∥-map inl-convergent-inv conv-coproduct)
      })
    (λ v-f → 1Lab.Type.case detect-convergent f v-f of λ
      { (yes conv-f) → yes (∥-∥-map inr-convergent conv-f)
      ; (no not-conv-f) → no λ conv-coproduct → not-conv-f (∥-∥-map inr-convergent-inv conv-coproduct)
      })
    v
```

**Key change**:
- **Before**: `∥-∥-map inl-not-convergent` (contrapositive - wrong direction!)
- **After**: `∥-∥-map inl-convergent-inv` (forward extraction - correct!)

**Type flow**:
```
conv-coproduct : ∥ is-convergent (G +ᴳ H) (inl v) ∥
                    ↓ ∥-∥-map inl-convergent-inv
conv-g : ∥ is-convergent G v ∥
                    ↓ not-conv-g : ¬ ∥ is-convergent G v ∥
contradiction : ⊥
```

---

## ✅ Holes Filled This Session (6 total)

| Network Type | Case | Helper Used | Status |
|-------------|------|-------------|--------|
| `f ⊙ g` | `inl v-g` no case | `inl-convergent-inv` | ✅ |
| `f ⊙ g` | `inr v-f` no case | `inr-convergent-inv` | ✅ |
| `Fork f g` | `inl v-f` no case | `inl-convergent-inv` | ✅ |
| `Fork f g` | `inr v-g` no case | `inr-convergent-inv` | ✅ |
| `Join f g` | `inl v-f` no case | `inl-convergent-inv` | ✅ |
| `Join f g` | `inr v-g` no case | `inr-convergent-inv` | ✅ |

**Total this session**: 6 holes filled
**Cumulative**: 27/45 holes (60%)

---

## 📊 Progress Breakdown

### Holes Remaining (18 total)

| Category | Count | Blocker |
|----------|-------|---------|
| detect-convergent (primitives) | 6 | Need `is-convergent` witness construction |
| extract-tines (Composition/Fork/Join) | 6 | Case analysis on inl/inr provenance |
| extract-gluing (Composition/Fork/Join) | 6 | Routing based on node provenance |

---

## 🎓 Technical Challenges Solved

### Challenge 1: Wrong Direction for No Cases

**Initial attempt**: Used `inl-not-convergent` (contrapositive)
```agda
inl-not-convergent : ¬ is-convergent (G +ᴳ H) (inl v)
                   → ¬ is-convergent G v
```

**Problem**: This takes `¬ (coproduct convergent)` as input, but we HAVE `coproduct convergent` in the lambda!

**Fix**: Need forward extraction
```agda
inl-convergent-inv : is-convergent (G +ᴳ H) (inl v)
                   → is-convergent G v
```

Then negate: `λ conv-coproduct → not-conv-g (∥-∥-map inl-convergent-inv conv-coproduct)`

### Challenge 2: With-Clauses in Let

**Initial attempt**: Used `with` pattern matching inside `let` binding
```agda
let s₁' : G .Graph.Node
    s₁' with source₁
    ... | inl x = x
```

**Error**: `Not a valid let binding`

**Fix**: Use pattern matching directly on record constructor
```agda
inl-convergent-inv record { source₁ = inl s₁ ; source₂ = inl s₂ ; ... } = ...
inl-convergent-inv record { source₁ = inr s₁ ; edge₁ = e₁ } = absurd (Lift.lower e₁)
```

### Challenge 3: Eliminating Cross-Edges

**Problem**: Cross-edges in coproduct have type `Lift ℓ ⊥` (empty type lifted to correct universe level)

**Solution**: Use `Lift.lower : Lift ℓ ⊥ → ⊥` to extract the contradiction
```agda
inl-convergent-inv record { source₁ = inr s₁ ; edge₁ = e₁ } =
  absurd (Lift.lower e₁)
```

---

## 💡 Key Insights

### 1. Forward vs Contrapositive

**Contrapositive** (what we initially tried):
```
¬ P → ¬ Q   means   Q → P
```

**Forward extraction** (what we needed):
```
P → Q   directly
```

For no cases, we have `P` (coproduct convergent) and need to derive `Q` (subgraph convergent) to contradict `¬ Q`.

### 2. Pattern Matching on Records is Clean

Rather than using `with` clauses or `let` bindings, direct pattern matching on record constructors is:
- More readable
- Type-checks cleanly
- Handles all cases naturally

### 3. Cross-Edge Elimination is Mechanical

Since `_+ᴳ_` defines cross-edges as `Lift ℓ ⊥`, any function that receives a cross-edge can immediately call `absurd (Lift.lower e)` - this is a purely mechanical step.

### 4. Coproduct Preserves AND Reflects Convergence

**Preserves** (forward): `inl-convergent : is-convergent G v → is-convergent (G +ᴳ H) (inl v)`

**Reflects** (inverse): `inl-convergent-inv : is-convergent (G +ᴳ H) (inl v) → is-convergent G v`

This makes coproduct a **full and faithful** operation on convergence structure!

---

## 📁 Files Modified

### Modified

**`/Users/faezs/homotopy-nn/src/Neural/Compile/GraphCoproduct.agda`**
- Added import: `Data.Sum.Base using (...; elim)` (line 58)
- Implemented `inl-convergent-inv` (lines 225-245)
- Implemented `inr-convergent-inv` (lines 247-267)
- Uses pattern matching on record constructors with case analysis
- **Type-check**: ✅ 0 goals

**`/Users/faezs/homotopy-nn/src/Neural/Compile/ForkExtract.agda`**
- Import inverse helpers (line 52): `inl-convergent-inv; inr-convergent-inv`
- Fixed Composition no cases (lines 483, 487)
- Fixed Fork no cases (lines 502, 506)
- Fixed Join no cases (lines 517, 521)
- **Type-check**: ✅ 18 goals (down from 24)

---

## 🚧 Next Steps

### Immediate: Fill Primitive Convergence (6 holes)

**Dense, Conv1D, MaxPool, AvgPool**: Construct `is-convergent` witnesses
```agda
detect-convergent (Prim (Dense W b)) (inr j) =
  yes (inc record
    { source₁ = inl fzero
    ; source₂ = inl (fsuc fzero)
    ; distinct = {!!}  -- Prove fzero ≠ fsuc fzero
    ; edge₁ = tt  -- Fully connected
    ; edge₂ = tt  -- Fully connected
    })
```

**Activation, BatchNorm**: Prove no convergence (1-to-1 mappings)
```agda
detect-convergent (Prim (Activation f)) v =
  no λ { (inc conv) → {!!} }  -- No node has 2+ sources
```

### Medium: Compositional Extract-Tines (6 holes)

**Pattern**: Case analysis on node provenance, lift tines from subnetworks
```agda
extract-tines (f ⊙ g) star pf =
  elim-coproduct
    (λ v → List ...)
    (λ v-g → map (lift-tine-left) (extract-tines g star' pf'))
    (λ v-f → map (lift-tine-right) (extract-tines f star' pf'))
    (fst star)
```

### Long: Compositional Extract-Gluing (6 holes)

**Pattern**: Route to appropriate subnetwork's gluing
```agda
extract-gluing (f ⊙ g) star pf =
  elim-coproduct
    (λ v → GluingOp)
    (λ v-g → extract-gluing g star' pf')
    (λ v-f → extract-gluing f star' pf')
    (fst star)
```

---

## 🏁 Session Summary

**What we built**:
- ✅ Inverse convergence transport (`inl-convergent-inv`, `inr-convergent-inv`)
- ✅ Pattern matching on record constructors with case analysis
- ✅ Cross-edge elimination using `absurd (Lift.lower e)`
- ✅ 6 holes filled (53% → 60%)

**What we learned**:
- Forward extraction (P → Q) vs contrapositive (¬P → ¬Q)
- Pattern matching on records is cleaner than with/let
- Coproduct fully and faithfully preserves convergence
- Cross-edges are mechanically eliminable

**What's next**:
- Fill primitive detect-convergent (6 holes) → 33/45 (73%)
- Then compositional extract-tines (6 holes) → 39/45 (87%)
- Then compositional extract-gluing (6 holes) → 45/45 (100%)

---

**Session End**: 2025-10-31
**Achievement**: Compositional detect-convergent COMPLETE! 🎉
**Status**: 27/45 holes (60%), on track for 73% by end of Week 2
