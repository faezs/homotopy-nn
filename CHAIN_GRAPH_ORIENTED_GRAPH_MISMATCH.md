# Chain Graph / OrientedGraph Semantic Mismatch

**Date**: 2025-10-13
**Issue**: Chain graphs do NOT satisfy OrientedGraph axioms
**Status**: ✅ **RESOLVED** - Fixed on 2025-10-14

## Resolution Summary

The issue was a **misunderstanding of the paper's definition**, not a design flaw. Belfiore & Bennequin's Definition 1.1 states:

> "An oriented graph Γ is directed when the relation a≤b between vertices, **defined by the existence of an oriented path**, is a partial ordering."

**Key insight**: The `≤` relation is the **transitive closure** of edges (reachability), NOT the edge relation itself!

### What Was Fixed

1. **Category.agda**: Added `EdgePath` data type and separated `Edge` (direct connections) from `_≤_` (reachability)
2. **Chain.agda**: Implemented `_≤_` as propositional truncation of `EdgePath`, removing impossible postulates
3. **All modules now type-check successfully** ✅

---

# Original Analysis (for historical reference)
**Affected Files**:
- `src/Neural/Dynamics/Chain.agda`
- `src/Neural/Topos/Category.agda` (OrientedGraph definition)
- `src/Neural/Topos/Architecture.agda` (uses OrientedGraph)

---

## The Problem

### What We Tried

We attempted to define chain networks (feedforward layers with only consecutive edges) as an `OrientedGraph`:

```agda
chain-graph : (n : Nat) → OrientedGraph lzero lzero
chain-graph n = record
  { graph = chain-underlying
  ; classical = chain-classical        -- ✓ Provable
  ; no-loops = chain-no-loops          -- ✓ Provable
  ; ≤-refl-ᴸ = λ _ → refl             -- ✓ Trivial
  ; ≤-trans-ᴸ = chain-trans            -- ✗ UNPROVABLE
  ; ≤-antisym-ᴸ = chain-antisym        -- ✗ UNPROVABLE
  }
  where
    Edge i j = Σ[ k ∈ Fin n ] (i ≡ weaken k) × (j ≡ fsuc k)
    -- This creates ONLY edges: 0→1, 1→2, 2→3, ..., (n-1)→n
```

### Why It Fails

**OrientedGraph requires transitivity**: If edges y→z and x→y exist, then edge x→z must exist.

**Chain graphs violate this**:
- Edge 0→1 exists: `k'=0, x=weaken 0=0, y=fsuc 0=1`
- Edge 1→2 exists: `k=1, y=weaken 1=1, z=fsuc 1=2`
- Edge 0→2 does NOT exist!

To have edge 0→2, we'd need `k''` such that:
- `0 = weaken k''` (implies k''=0)
- `2 = fsuc k''` (implies k''=1)

**Contradiction**: k'' cannot be both 0 and 1 simultaneously.

### The Root Cause

**Semantic mismatch**:
- `OrientedGraph` models **reachability relations** (transitive closure of edges)
- `chain-graph` models **direct connectivity** (only immediate neighbors)

**OrientedGraph expects**: If A can reach B, and B can reach C, then A can reach C (transitive).

**Chain graphs provide**: Only direct edges between consecutive layers (not transitive).

---

## Technical Details

### Provable Properties

✅ **classical** (edges are propositions): Successfully proven using `weaken-inj`
```agda
chain-classical : {x y : Fin (suc n)} → is-prop (Edge x y)
-- Proof: Two edges (k, p, q) and (k', p', q') must have k = k'
-- because weaken is injective
```

✅ **no-loops** (no self-edges): Successfully proven using `weaken≠fsuc`
```agda
chain-no-loops : {x : Fin (suc n)} → ¬ (Edge x x)
-- Proof: Edge x→x requires x = weaken k and x = fsuc k
-- But weaken k ≠ fsuc k for all k (proven by induction)
```

### Unprovable Properties

✗ **chain-trans** (transitivity): IMPOSSIBLE to prove
```agda
-- Requires: edges y→z and x→y ⇒ edge x→z exists
-- But composition of consecutive edges does NOT yield a single edge!
-- Example: 0→1 ∘ 1→2 does NOT produce 0→2
```

✗ **chain-antisym** (antisymmetry): IMPOSSIBLE to prove without additional structure
```agda
-- Requires: edges x→y and y→x both exist ⇒ x ≡ y
-- For chains, this is vacuously true (premises never both hold)
-- But proving the contradiction is complex without numerical ordering
```

---

## What We Need Instead

### Option 1: Path Graph (Transitive Closure)

Define a **path graph** that includes all reachable pairs, not just consecutive edges:

```agda
path-graph : (n : Nat) → OrientedGraph lzero lzero
path-graph n = record
  { Edge i j = Σ[ k₁ k₂ : Fin (suc n) ]
                (i ≡ weaken k₁) ×
                (j ≡ fsuc k₂) ×
                (k₁ ≤ k₂)  -- Path exists if k₁ ≤ k₂
  ; ...
  }
```

**Pros**: Satisfies OrientedGraph axioms (transitivity holds)
**Cons**: More complex, not what "chain" intuitively means

### Option 2: Directed Graph (No Transitivity)

Create a weaker abstraction that doesn't require transitivity:

```agda
record DirectedGraph (o ℓ : Level) : Type (lsuc (o ⊔ ℓ)) where
  field
    graph : Graph o ℓ
    classical : ∀ {x y} → is-prop (Edge x y)
    no-loops : ∀ {x} → ¬ Edge x x
    -- NO transitivity requirement
    -- NO antisymmetry requirement
```

**Pros**: Semantically accurate for chains
**Cons**: May need to define path/reachability separately when needed

### Option 3: Separate Direct vs Reachability

Keep both concepts distinct:

```agda
record ChainGraph (n : Nat) : Type where
  field
    direct-edge : (i j : Fin (suc n)) → Type
    -- Only consecutive edges

  reachable : (i j : Fin (suc n)) → Type
  reachable i j = Σ[ path : List (Fin (suc n)) ]
                  (path-connects i j via direct-edge)
```

**Pros**: Clear separation of concerns
**Cons**: More complex type structure

---

## Recommendations

### For Neural.Topos.Architecture

1. **Don't use OrientedGraph for chains** - it's semantically incorrect
2. **Create DirectedGraph** - a weaker structure without transitivity
3. **Define reachability separately** - when you need transitive closure, compute it explicitly
4. **Update fork construction** - base it on DirectedGraph, not OrientedGraph

### For Neural.Dynamics.Chain

1. **Remove OrientedGraph dependency** - use DirectedGraph instead
2. **Keep current edge definition** - it's correct for direct connectivity
3. **Add path/reachability** - when needed for reasoning about information flow
4. **Document the distinction** - make it clear we model direct edges, not all paths

### For Future Work

When implementing backpropagation or information flow:
- Use **direct edges** for weight matrices (only consecutive layers)
- Use **reachability** for gradient flow (can flow through multiple layers)
- Keep the two concepts separate in the type system

---

## Code Locations

### Current (Broken) Implementation
- **File**: `src/Neural/Dynamics/Chain.agda`
- **Lines**: 69-143 (chain-graph definition with postulates)
- **Issue**: Lines 125-129 (chain-trans postulated), 139-143 (chain-antisym postulated)

### OrientedGraph Definition
- **File**: `src/Neural/Topos/Category.agda`
- **Lines**: 55-80 (OrientedGraph record with ≤-trans-ᴸ requirement)

### Uses in Architecture
- **File**: `src/Neural/Topos/Architecture.agda`
- Grep for `OrientedGraph` to find all uses

---

## Action Items

- [ ] Define `DirectedGraph` in `Neural/Topos/Category.agda` (weaker than OrientedGraph)
- [ ] Rewrite `chain-graph` to return `DirectedGraph` instead
- [ ] Update `Neural/Topos/Architecture.agda` to use DirectedGraph for non-convergent cases
- [ ] Define separate `Reachability` or `PathGraph` when transitive closure is needed
- [ ] Update fork construction to work with DirectedGraph base
- [ ] Add proofs that fork graphs (with convergence) DO satisfy OrientedGraph
- [ ] Document the distinction between direct connectivity and reachability

---

## References

### Academic Context

From Belfiore & Bennequin (2022), Section 1:
- Networks are modeled as **directed graphs**
- Poset X captures **reachability** (transitive relation)
- But the actual **network edges** are just the weight matrices (not transitive)

This confirms: we need TWO structures:
1. **Network graph**: Direct edges (weight matrices) - NOT transitive
2. **Poset X**: Reachability relation - IS transitive

We incorrectly conflated these into a single OrientedGraph.

### Related Issues

- SimpleMLP in `Neural/Topos/Examples.agda` also just postulates OrientedGraph (line 58)
- This suggests the issue is widespread, not just in Chain.agda
- The entire architecture may need revision

---

## Conclusion

**The chain-graph definition is fundamentally unsound as an OrientedGraph.**

The fix requires architectural changes:
1. Introduce DirectedGraph (no transitivity)
2. Use DirectedGraph for network topology
3. Define reachability/paths separately when needed
4. Reserve OrientedGraph for structures that truly need transitivity (like fork graphs with all paths)

This is not a proof bug - it's a **semantic type mismatch** in the architecture.
