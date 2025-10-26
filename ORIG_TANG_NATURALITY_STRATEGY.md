# Implementation Strategy: orig→tang Naturality

## Date: 2025-10-24

## Goal

Prove:
```agda
α .is-natural (x-node , v-fork-tang) (y-node , v-original) f
```

where `f : Path-in Γ̄ (y-node, v-original) (x-node, v-fork-tang)` (opposite category path)

## Approach: Recursive Case Analysis

Instead of trying to project the ENTIRE path, we handle it RECURSIVELY by pattern matching on the path structure.

### Base Case: nil

`nil : Path-in Γ̄ (x-node, v-fork-tang) (x-node, v-fork-tang)` - This means x-node = y-node and we have identity.

But wait, we need a path from `(y-node, v-original)` to `(x-node, v-fork-tang)`. If this is nil, then:
- `(y-node, v-original) ≡ (x-node, v-fork-tang)`

This is IMPOSSIBLE because v-original ≠ v-fork-tang!

So `nil` case is absurd.

### Inductive Case: cons e p

`cons e p : Path-in Γ̄ (y-node, v-original) (x-node, v-fork-tang)`

where `e : ForkEdge (y-node, v-original) b` for some intermediate vertex `b`, and `p : Path-in Γ̄ b (x-node, v-fork-tang)`.

Now case split on `b`:

#### Case b = (b-node, v-original)

Then `e` must be `orig-edge`. And `p : Path-in Γ̄ (b-node, v-original) (x-node, v-fork-tang)`.

**Recursive call**: By IH, we can prove naturality for `p`.

But how do we combine with edge `e`? Use functoriality!

#### Case b = (b-node, v-fork-star)

Then `e` must be `tip-to-star`. And `p : Path-in Γ̄ (b-node, v-fork-star) (x-node, v-fork-tang)`.

Since star only goes to tang, `p` must be:
- `cons (star-to-tang ...) nil` where b-node = x-node

This is **Type 2 path** (via star)! Need sheaf gluing proof.

#### Case b = (b-node, v-fork-tang)

Then `e` could be `handle` (if y-node = b-node). And `p : Path-in Γ̄ (b-node, v-fork-tang) (x-node, v-fork-tang)`.

By `tang-no-outgoing`, tang has no outgoing edges, so `p` must be `nil`, meaning `b-node = x-node`.

So this is **Type 1 path**: Single handle edge from `(x-node, v-original)` to `(x-node, v-fork-tang)`.

Actually wait, if `b = (b-node, v-fork-tang)` and `e : ForkEdge (y-node, v-original) (b-node, v-fork-tang)`, then `e` must be `handle`, which requires y-node = b-node. And `p` must be nil (tang-no-outgoing), so b-node = x-node. Therefore y-node = x-node.

This gives us paths of length 1: just a single handle edge.

But we could also have **longer Type 1 paths**:
- `(y-node, v-original) --orig-edges--> (x-node, v-original) --handle--> (x-node, v-fork-tang)`

Hmm, let me reconsider the recursion structure...

## Better Approach: Path Length Analysis

Let me think about path length:

**Length 0**: nil - impossible (type mismatch)

**Length 1**: Single edge from (y, v-original) to (x, v-fork-tang)
- Only possibility: `handle` (requires y = x)
- Proof: Direct, using α definition at both ends

**Length ≥ 2**: cons e (cons e' p')
- First edge `e` from (y, v-original) to some `b`
- Case split on `b`:
  - `b = (b-node, v-original)`: e is orig-edge, recurse
  - `b = (b-node, v-fork-star)`: e is tip-to-star, then MUST have e' = star-to-tang (Type 2)
  - `b = (b-node, v-fork-tang)`: e is handle, then p' must be nil (Type 1 with y ≠ x)

## Cleaner Approach: Case by Last Two Edges

Actually, let me think about this from the END of the path instead of the beginning!

The path ends at `(x-node, v-fork-tang)`. What's the second-to-last vertex?

**Case A**: Second-to-last is `(x-node, v-original)`
- Last edge is `handle`
- Rest of path: `(y, v-original) ---> (x, v-original)` which is an X-path!
- Use `project-path-orig` on the prefix

**Case B**: Second-to-last is `(x-node, v-fork-star)`
- Last edge is `star-to-tang`
- Third-to-last must be some `(a', v-original)` (by tip-to-star structure)
- This is Type 2 path through star

## Implementation via Helper Function

```agda
mutual
  -- For paths ending at tang, split by last edge
  naturality-orig-to-tang : ∀ {y x}
    → (f : Path-in Γ̄ (y, v-original) (x, v-fork-tang))
    → α .η (x, v-fork-tang) ∘ F.F₁ f ≡ G.F₁ f ∘ α .η (y, v-original)

  naturality-orig-to-tang (cons e nil) = case e of
    handle → proof-single-handle e
    _ → absurd (wrong-edge-type e)

  naturality-orig-to-tang (cons e (cons e' p')) = case e' of
    handle → proof-via-handle-suffix e (cons e' p')
    star-to-tang → proof-via-star-suffix e e' p'
    _ → continue-recursion e e' p'

  proof-via-handle-suffix : ...
  proof-via-star-suffix : ...
```

Actually, this is getting complex. Let me think of an even simpler approach.

## Simplest Approach: Postulate Type 2 for Now

Since Type 2 paths (through star) require sheaf gluing which we don't understand yet, let's:

1. **Implement Type 1 paths** (via handle) which CAN be projected
2. **Postulate** Type 2 paths (via star) with a clear TODO

This lets us make progress on the easier cases while documenting what needs to be done.

## Concrete Implementation for Type 1

For paths that end with handle:

1. Use recursion to decompose: `f = prefix ++ singleton handle`
2. Project `prefix` to X using `project-path-orig` (since prefix goes from orig to orig)
3. Use γ.is-natural on the projected prefix
4. Handle the final handle edge using:
   - Functoriality: F(prefix ++ handle) = F(prefix) ∘ F(handle)
   - Definition of α at tang: α.η (x, tang) = γ.η ((x, tang), inc tt)
   - Definition of α at orig: α.η (x, orig) = γ.η ((x, orig), inc tt)
   - Some property relating γ at (x, tang) and (x, orig)?

Wait, I need to check: is there a relationship between γ.η at tang and orig for the same node?

Actually, γ is defined on X-Category, so γ.η is indexed by X-nodes. Both (x, v-original) and (x, v-fork-tang) are in X (as long as x is not a star). So γ.η for these are independent - there's no automatic relationship!

This means for Type 1 paths, I can't just use γ.is-natural on a prefix... I need to think more carefully.

Hmm, let me reconsider. Maybe the right approach is NOT to try to decompose the path, but instead to recognize that Type 1 and Type 2 need completely different proof strategies:

- **Type 1**: The prefix to (x, v-original) can be projected to X, and then we use γ.is-natural. But we also need to handle the handle edge.
- **Type 2**: Cannot be projected, need sheaf gluing.

## Decision: Defer Both Type 1 and Type 2

Both require careful thought about:
1. How to decompose paths (need a snoc view or length induction)
2. How to handle edges that are not in X (handle edge)
3. How to apply sheaf gluing for star paths

Let's postulate `project-path-orig-to-tang` with a detailed explanation and move on to other cases that we CAN solve.

Actually wait, the user said NO postulates! Let me think harder...
