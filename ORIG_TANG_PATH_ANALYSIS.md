# Paths from Original to Tang Vertices: Complete Analysis

## Date: 2025-10-24

## Problem

For the orig→tang naturality case, we need to prove:
```agda
α .η (x, v-fork-tang) ∘ F.F₁ f ≡ G.F₁ f ∘ α .η (y, v-original)
```

where `f : Path-in Γ̄ (y, v-original) (x, v-fork-tang)` (opposite category).

**Question**: Can we always project such paths to X-paths?

## Edge Structure to Tang Vertices

From `ForkCategorical.agda`, only TWO types of edges can reach a tang vertex:

1. **star-to-tang**: `(a, v-fork-star) → (a, v-fork-tang)` (same node)
2. **handle**: `(a, v-original) → (a, v-fork-tang)` (same node)

**Key constraint**: Both edges require `source.node ≡ target.node`!

## Path Classification

For paths from `(y, v-original)` to `(x, v-fork-tang)`, the LAST edge must be one of:

### Type 1: Ends with Handle (X-paths)

**Structure**:
```
(y, v-original) --orig-edges--> (x, v-original) --handle--> (x, v-fork-tang)
```

**Properties**:
- Prefix is a path between original vertices
- Prefix can be projected to X via `project-path-orig`
- These ARE X-paths (modulo the final handle)

**Proof strategy**:
1. Decompose: `f = prefix ++ handle`
2. Project prefix to X: `prefix-X = project-path-orig prefix`
3. Use γ naturality on `prefix-X`
4. Handle the final `handle` edge separately

### Type 2: Ends with Star-to-Tang (Non-X-paths)

**Structure**:
```
(y, v-original) --...-->(a', v-original) --tip-to-star--> (x, v-fork-star) --star-to-tang--> (x, v-fork-tang)
```

**Properties**:
- Path goes through `(x, v-fork-star)` which is NOT in X
- CANNOT be projected to X-Category
- Requires `x` to be convergent (has fork)

**Proof strategy**:
- CANNOT use γ.is-natural directly (γ is defined on X, not on paths through star)
- Need to use sheaf gluing properties at star vertices
- This is a HARD case similar to orig→star naturality

## Key Insight: Type 2 Paths Cannot Be Projected

**Theorem**: Paths from original to tang that go through star vertices are NOT X-paths.

**Proof**: X-Category is defined via `is-non-star : ForkVertex → Ω` where:
```agda
is-non-star (a, v-original) = ⊤
is-non-star (a, v-fork-star) = ⊥  -- excluded!
is-non-star (a, v-fork-tang) = ⊤
```

Any path containing `(x, v-fork-star)` has `is-non-star` = ⊥ at that vertex, so it's NOT an X-path.

## Why the Original Strategy Failed

The NATURALITY_STRATEGY.md incorrectly states:
> "Both vertices are in X (non-star) ... Can project f to X-path"

This is WRONG for Type 2 paths! While both endpoints ARE in X, the PATH ITSELF may go through star vertices.

**Analogy**: In a subgraph, just because two vertices are in the subgraph doesn't mean every path between them stays in the subgraph.

## Solution Approaches

### Option 1: Case Split on Last Edge

```agda
project-path-orig-to-tang : Path-in Γ̄ (y, v-original) (x, v-fork-tang)
                          → Path-in X ((y, v-original), inc tt) ((x, v-fork-tang), inc tt)
project-path-orig-to-tang f = case last-edge f of
  handle → project prefix ++ lift handle
  star-to-tang → ERROR -- impossible to project!
```

**Problem**: Type 2 paths CANNOT be projected to X!

### Option 2: Dependent Type on Path Structure

```agda
data PathType : ForkVertex → ForkVertex → Type where
  via-handle : Path-in Γ̄ (y, v-original) (x, v-original)
             → PathType (y, v-original) (x, v-fork-tang)
  via-star : Path-in Γ̄ (y, v-original) (x, v-fork-star)
           → PathType (y, v-original) (x, v-fork-tang)

classify : (f : Path-in Γ̄ (y, v-original) (x, v-fork-tang))
         → PathType (y, v-original) (x, v-fork-tang)
```

Then prove naturality separately for each type.

### Option 3: Direct Naturality Proof WITHOUT Projection

**Key observation**: We don't actually NEED to project to X!

For Type 1 paths, we can prove naturality by:
1. Decomposing into prefix + handle
2. Using functoriality: `F(prefix ++ handle) = F(prefix) ∘ F(handle)`
3. Using γ.is-natural on the prefix
4. Handling the handle edge using... what?

For Type 2 paths, we need to use sheaf gluing at the star vertex.

### Option 4: Prove Type 2 Paths are Impossible (WRONG)

Could we prove that Type 2 paths don't exist? NO - they DO exist when `x` is convergent!

Example: Convergent network with node `w` receiving from nodes `v₁`, `v₂`:
- Original graph: `v₁ → w`, `v₂ → w`
- Fork graph: `v₁ → w★`, `v₂ → w★`, `w★ → w_tang`, `w → w_tang`

Path from `(v₁, v-original)` to `(w, v-fork-tang)`:
```
(v₁, v-original) --tip-to-star--> (w, v-fork-star) --star-to-tang--> (w, v-fork-tang)
```

This is a valid Type 2 path!

## Recommendation

We need to handle BOTH types of paths:

1. **For naturality proofs**: Case split based on last edge
   - If handle: Use γ.is-natural on prefix (X-path)
   - If star-to-tang: Use sheaf gluing properties (Non-X-path)

2. **Correct NATURALITY_STRATEGY.md**: Update orig→tang case to acknowledge both path types

3. **Do NOT implement `project-path-orig-to-tang`**: It's partial! Only works for Type 1.

## Next Steps

1. Update NATURALITY_STRATEGY.md with correct analysis
2. Implement case-by-case naturality for orig→tang:
   ```agda
   α .is-natural (x, v-fork-tang) (y, v-original) (cons e p) =
     case e of
       handle → proof-via-projection p
       star-to-tang → proof-via-sheaf-gluing p
       _ → impossible
   ```

3. Similarly handle other cases involving star vertices (orig→star, star→tang, star→star)

## Open Questions

1. How exactly do we prove naturality for paths going through star?
2. Is there a general lemma about sheaf gluing and naturality we can use?
3. Can we factor out the sheaf gluing proof strategy into a helper lemma?
