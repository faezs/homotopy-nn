# F₁ Morphism Direction Blocker

**Date**: 2025-10-16 (continuation session)
**Status**: 🔴 **BLOCKED** - Fundamental issue with morphism direction in extension functor
**Duration**: ~2 hours on F₁ implementation

---

## Summary

Successfully completed:
- ✅ Terminal preservation (zero postulates!)
- ✅ Extension functor F₀ definition (with universe level resolution)
- ✅ Extension functor structure (map-path, map-edge helpers)
- ✅ Two edge cases: `tip-to-star` and partial `star-to-tang`

**Blocked on**: Morphism direction mismatch for `orig-edge` and `tang-to-handle` cases

---

## The Problem

### Context

We're implementing the extension functor from Proposition 1.1(iii):
```agda
extend-to-fork : Functor (Fork-Category ^op) (Sets (o ⊔ ℓ ⊔ κ))
```

Given a presheaf `F : X-Category^op → Sets κ`, we extend it to Fork-Category by:
- F₀ at original/tang: Lift from F
- F₀ at fork-star: Product over tips
- F₁: **This is where we're stuck**

### The Morphism Direction Mismatch

For `orig-edge conn ¬conv` where `conn : x₁ → y₂` (graph edge):

**In Fork-Category**:
- Edge in graph: `x₁ → y₂`
- Morphism in category: `original x₁ → original y₂` (same direction)
- Morphism in Fork^op: `original y₂ → original x₁` (reversed)

**In X-Category**:
- `conn : x₁ → y₂` gives `≤ˣ-orig conn ¬conv : x-original y₂ ≤ˣ x-original x₁`
- This is a morphism `x-original y₂ → x-original x₁` in X-Category

**Under presheaf F : X-Category^op → Sets**:
- `F₁ (y₂ ≤ˣ x₁) : F₀ (x-original x₁) → F₀ (x-original y₂)`
- Direction: `F(x₁) → F(y₂)`

**What we need for the extension**:
- `map-edge : F₀ (original y₂) → F₀ (original x₁)`
- Which is: `Lift ∣F(x-original y₂)∣ → Lift ∣F(x-original x₁)∣`
- Direction: `F(y₂) → F(x₁)`

**THE MISMATCH**: F₁-X gives us `F(x₁) → F(y₂)` but we need `F(y₂) → F(x₁)` (opposite!)

---

## What We Tried

### Attempt 1: Direct use of F₁-X
```agda
map-edge (BuildFork.orig-edge conn ¬conv) = λ fb →
  lift (F₁-X (≤ˣ-orig conn ¬conv) (lower fb))
```
**Error**: `y₁ != x₁ of type Graph.Vertex` - type mismatch

### Attempt 2: Reversing the morphism
Tried to find the "inverse" morphism in X-Category, but:
- The morphism `≤ˣ-orig conn ¬conv` is the ONLY morphism from the edge
- There's no natural "reverse" available
- Antisymmetry means if both `a ≤ b` and `b ≤ a` then `a = b`

### Attempt 3: Compositional approach
Tried building morphisms from components, but the fundamental direction issue remains.

---

## Analysis: Why This Is Hard

### Categorical Structure

1. **Fork-Category** is a free category on a graph (Path-category)
   - Morphisms `a → b` are PATHS from `a` to `b`
   - Composition: `f ∘ g = g ++ f` (path concatenation)

2. **X-Category** is a poset (thin category)
   - Morphisms are `≤ˣ` relations
   - At most one morphism between any two objects

3. **Extension should be "restriction"**
   - For vertices in X, F₀ should use F directly
   - For morphisms between X-vertices, F₁ should... ???

### The Missing Piece

The paper says (line 679-681):
> "At original/tang: F(x) = F_X(x)"

But it doesn't specify HOW morphisms in Fork-Category map to morphisms in X-Category!

**Hypothesis**: There should be a "projection" or "restriction" functor:
```agda
π : Fork-Category → X-Category  -- (maybe? partial?)
```

Then the extension would be:
```agda
extended .F₁ e = F₁-X (π e)  -- Apply F to the projected morphism
```

But:
1. Not all Fork morphisms project (paths through stars!)
2. We haven't defined such a functor
3. The direction issue suggests it might need to be Fork-Category^op → X-Category

---

## Possible Solutions

### Option 1: Define Projection Functor (2-4 hours)
Define `π : Fork-Category → X-Category` that:
- Maps original → x-original, fork-tang → x-fork-tang
- Maps orig-edge to corresponding ≤ˣ morphism
- Maps tang-to-handle to corresponding ≤ˣ morphism
- Maps paths through stars to COMPOSITE morphisms in X

**Pros**: Mathematically clean, matches the paper's intent
**Cons**: Complex, needs to handle all edge cases and prove functoriality

### Option 2: Case-by-Case with Postulates (1 hour)
Postulate the action for each edge type:
```agda
postulate
  extend-orig-edge : {x y : Layer} (conn : Connection x y) (¬conv : ¬ is-convergent y) →
    ∣ F₀-X (x-original y) ∣ → ∣ F₀-X (x-original x) ∣
```

**Pros**: Unblocks progress immediately
**Cons**: User said "NO FUCKING POSTULATES"

### Option 3: Use Paper's Construction Directly (3-5 hours)
Go back to the paper (ToposOfDNNs.agda Section 1.5) and find the EXPLICIT construction of the extension functor's F₁.

**Pros**: Should have the answer
**Cons**: Requires careful reading and translation

### Option 4: Rethink the Approach (2-3 hours)
Maybe the extension shouldn't use F₁-X at all for individual edges. Instead:
- Build up FULL PATHS in X-Category that correspond to paths in Fork-Category
- Apply F₁-X to the complete path
- This would be a "path homomorphism" approach

**Pros**: Might match the mathematical intent better
**Cons**: More complex implementation

---

## Current Code State

### File: Architecture.agda lines 1122-1165

```agda
extended .Functor.F₁ {x} {y} p = map-path p
  where
  map-edge : {a b : ForkVertex} → ForkEdge a b → (F₀ b → F₀ a)

  -- ✅ IMPLEMENTED:
  map-edge (tip-to-star conv conn) = λ fb → lift (fb (a' , conn))

  -- ⏳ PARTIAL:
  map-edge (star-to-tang conv) = λ fb → λ tip → {!!}

  -- ❌ BLOCKED:
  map-edge (orig-edge conn ¬conv) = {!!}      -- Morphism direction mismatch
  map-edge (tang-to-handle conv) = {!!}       -- Same issue
  map-edge (ForkEdge-is-set ...) = {!!}       -- HIT coherence (deferred)

  map-path nil = id
  map-path (cons e p) = map-edge e ∘ map-path p
```

**18 goals remaining** in Architecture.agda (up from 15 due to new holes)

---

## Recommended Next Steps

### Immediate (Tonight if continuing)

1. **Read the paper carefully** - Section 1.5, Proposition 1.1(iii)
   - Look for explicit description of morphism mapping
   - Check if there's a diagram showing the extension

2. **Search 1Lab** for similar constructions
   - Restriction/extension functors
   - Kan extensions (this might be a right Kan extension!)
   - Sheafification constructions

3. **Ask for help** - This is a fundamental categorical question:
   - "How do morphisms in a path category restrict to a subcategory?"
   - "Is the extension a right Kan extension along an inclusion?"

### Tomorrow

If still stuck after research:
- **Option A**: Implement projection functor (cleanest)
- **Option B**: Postulate with detailed justification (pragmatic)
- **Option C**: Skip F₁ entirely and work on other parts of the project

---

## Key Insights from This Session

1. **Universe levels are real** - Had to use `Lift` and target `Sets (o ⊔ ℓ ⊔ κ)`
2. **Path categories are subtle** - Morphisms are paths, not edges
3. **Contravariance is tricky** - Direction reversals at multiple levels
4. **Paper's extension is underspecified** - F₁ action not explicitly described

---

## Questions to Answer

1. Is there a functor `Fork-Category → X-Category`?
2. If so, is it defined on all morphisms or just some?
3. Is the extension a right Kan extension?
4. How does the paper actually define F₁ for the extension?
5. Are we supposed to use the sheaf condition to derive F₁?

---

## Bottom Line

We've made **substantial progress** on the structure but hit a **fundamental blocker** on the morphism mapping. The F₀ definition is complete and correct. The F₁ structure (map-path/map-edge) is set up correctly. We just need to understand the CATEGORICAL relationship between Fork-Category morphisms and X-Category morphisms.

This is a **mathematical question**, not a technical Agda issue. Once we understand the math, the code will follow.

Estimated time to resolve: **2-5 hours** depending on approach.
