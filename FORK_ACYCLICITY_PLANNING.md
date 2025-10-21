# Fork Graph Acyclicity Proof - Planning Document

## Current Challenge (2025-10-22)

Working on: `ForkCategorical.agda` line ~1036
Goal: Prove `Γ̄-acyclic` (fork graph is acyclic)
Status: Multiple holes remaining in `path-between-different-types-impossible` helper

## Mathematical Insight (CORRECT)

**Key discovery**: Fork structure at node `a` forms a COSPAN:

```
In Γ̄ (the graph):
  (a, orig) ----handle---→ (a, tang)  ←---star-to-tang---- (a, star)
                                 ↑
                           apex (TERMINAL)

In C = Γ̄^op (for presheaves):
  (a, orig) ←--handle^op--- (a, tang)  ---star-to-tang^op→ (a, star)
                                 |
                           apex (INITIAL)
```

**For acyclicity in Γ̄**: Tang is **TERMINAL** at same node (no outgoing edges)
- Any cycle requires paths in BOTH directions
- But tang → anything is IMPOSSIBLE (terminal object)
- Therefore cycles at same node with different types are impossible

**Correctly fixed**: `handle` direction is `a → A` (original → tang), matching paper

## Edge Constructors and Their Target Types

At node `a`, what target types can each constructor produce?

| Constructor | Source Type | Target Type | Notes |
|------------|------------|-------------|-------|
| `orig-edge x y e nc` | `v-original` | `v-original` | Stays in original vertices |
| `tip-to-star a' a'' conv e` | `v-original` | `v-fork-star` | Edge `a' → a''` in G |
| `star-to-tang a conv` | `v-fork-star` | `v-fork-tang` | Mandatory transition |
| `handle a conv` | `v-original` | `v-fork-tang` | The "handle" edge |

**Key observations**:
1. FROM `v-original`: can reach `v-original` (orig-edge) OR `v-fork-star` (tip-to-star) OR `v-fork-tang` (handle)
2. FROM `v-fork-star`: can ONLY reach `v-fork-tang` (star-to-tang)
3. FROM `v-fork-tang`: can reach NOTHING at same node (TERMINAL!)

## Current Proof Strategy

```agda
path-between-different-types-impossible : ∀ (a : Node) (tv tw : VertexType)
                                        → tv ≠ tw
                                        → EdgePath (a , tv) (a , tw) → ⊥
```

**Cases to handle** (9 total, but symmetric):

| From → To | Possible? | Proof Strategy |
|-----------|-----------|----------------|
| orig → orig | No (tv ≠ tw) | Absurd on tv≠tw refl |
| orig → star | **HARD** | Need complex analysis |
| orig → tang | YES (handle) | Not used, check reverse |
| star → orig | No | Check source types |
| star → star | No (tv ≠ tw) | Absurd on tv≠tw refl |
| star → tang | YES (star-to-tang) | Not used, check reverse |
| tang → orig | **NO** | Tang is terminal! ✓ |
| tang → star | **NO** | Tang is terminal! ✓ |
| tang → tang | No (tv ≠ tw) | Absurd on tv≠tw refl |

**✓ = Successfully proven**

## Problems Encountered

### Problem 1: `orig → star` via `orig-edge`

```agda
no-edge-to-star-from-same-node (orig-edge x y edge nc pv pw) =
  -- pw : mid ≡ (y, v-original)
  -- rest : EdgePath mid (a, v-fork-star)
  -- Issue: mid might be at different node y ≠ a
  -- Need to either:
  --   a) Show y = a (via acyclicity), then recurse
  --   b) Show path y → a in Γ̄ projects to path in G, use G-acyclic
```

**Attempted**: Recursion with transport
**Failed**: Implicit argument unification issues in Cubical Agda

### Problem 2: `orig → star` via `tip-to-star`

```agda
no-edge-to-star-from-same-node (tip-to-star a' a'' conv edge pv pw) =
  -- pv : (a, v-original) ≡ (a', v-original), so a ≡ a'
  -- edge : Edge a' a'' in G, becomes Edge a a'' via substitution
  -- pw : mid ≡ (a'', v-fork-star)
  -- rest : EdgePath mid (a, v-fork-star)
  -- Goal: Show a'' ≡ a (creating Edge a a in G, contradicting no-loops)
```

**Attempted**: Use G-acyclicity on forward path (edge a → a'') and return path (project-to-G rest)
**Failed**: `subst` with implicit `mid` argument causes type mismatch `b₁ != mid`

## Comparison with Previous Approaches

### Architecture.agda Approach (HIT with Path Constructor)

**File**: `src/Neural/Topos/Architecture.agda` lines 837-860

```agda
data _≤ˣ_ : X-Vertex → X-Vertex → Type (o ⊔ ℓ) where
  ≤ˣ-refl : ∀ {x} → x ≤ˣ x
  ≤ˣ-orig : ...
  ≤ˣ-tang-handle : ...
  ≤ˣ-tip-tang : ...
  ≤ˣ-trans : ∀ {x y z} → x ≤ˣ y → y ≤ˣ z → x ≤ˣ z
  ≤ˣ-thin : ∀ {x y} (p q : x ≤ˣ y) → p ≡ q  -- PATH CONSTRUCTOR
```

**Key difference**: Made thinness AXIOMATIC via path constructor
- Avoided need to prove uniqueness
- Category laws trivial: `≤ˣ-idl f = ≤ˣ-thin (≤ˣ-trans ≤ˣ-refl f) f`

**Why it worked**: HIT path constructor bypasses K axiom issues

### Our Current Approach (Inductive EdgePath from 1Lab)

**File**: `ForkCategorical.agda`

```agda
-- Using 1Lab's EdgePath (free category)
open EdgePath Γ̄

-- Proving acyclicity via:
Γ̄-acyclic : ∀ (v w : ForkVertex) → EdgePath v w → EdgePath w v → v ≡ w
```

**Advantages**:
- Uses 1Lab infrastructure (products, limits inherit)
- More compositional (EdgePath is standard)
- Avoids custom HIT definition

**Challenges**:
- Must actually PROVE properties (not axiomatize)
- Complex pattern matching on paths
- Implicit argument unification in Cubical Agda

## Why Current Proof is Hard

### Issue 1: Implicit Mid Parameter

When pattern matching on `cons e rest`:
```agda
path-between-different-types-impossible a v-original v-fork-star tv≠tw (cons e rest)
  -- rest has type: EdgePath mid (a, v-fork-star)
  -- BUT: mid is an implicit argument that Agda generates fresh
  -- When we match on edge constructor:
  --   tip-to-star a' a'' conv edge pv pw
  --   pw : mid ≡ (a'', v-fork-star)
  -- Agda doesn't automatically know mid in `rest` type equals mid in `pw`!
```

**What works**: Checking source/target type directly from `pv`/`pw`
**What fails**: Using `rest` after transporting via `pw`

### Issue 2: Projection and Transport

```agda
-- Want to say:
rest' = subst (λ v → EdgePath v (a, v-fork-star)) pw rest
-- But Agda complains:
--   b₁ != mid of type ForkVertex
--   when checking rest has type EdgePath mid ...
```

**Root cause**: `mid` in type of `rest` is existentially quantified by `cons`
- Not the same `mid` we can name in our proof
- Cubical Agda's pattern matching doesn't auto-unify

## Proposed Solutions

### Option A: Strengthen Induction Hypothesis

Instead of proving `EdgePath (a, tv) (a, tw) → ⊥`, prove stronger statement:

```agda
no-path-to-star : ∀ (a : Node) (v : ForkVertex)
                → fst v ≡ a
                → snd v ≡ v-original
                → EdgePath v (a, v-fork-star)
                → ⊥
```

**Benefits**:
- Explicit parameter `v` instead of implicit `mid`
- Can pattern match on `v` structure
- Easier to transport

### Option B: Use 1Lab's Path Lemmas

Search 1Lab for path manipulation lemmas:
- `Cat.Instances.Graphs` - how do they handle paths?
- `Cat.Diagram.Pushout` - similar cospan reasoning?
- Check if there's a "path induction" principle

### Option C: Axiomatize the Tricky Cases

Keep the mathematical insight (tang terminal), but postulate:
```agda
postulate
  orig-to-star-impossible : ∀ (a : Node) → EdgePath (a, v-original) (a, v-fork-star) → ⊥
```

**Trade-off**: Less constructive, but documents the proof strategy

### Option D: Simplify via Graph Quotient

Define equivalence relation on fork vertices:
```agda
_≈_ : ForkVertex → ForkVertex → Type
(a, tv) ≈ (b, tw) = (a ≡ b) × (tv ≡ tw)
```

Work in quotient graph `Γ̄ / ≈`, where path uniqueness is easier?

## Recommended Next Steps

### Step 1: Check 1Lab Examples

Search for similar proofs:
```bash
grep -r "EdgePath\|acyclic" /nix/store/.../1lab/src/Cat/Instances/
```

Look for:
- How they handle path induction
- Transport along path equalities
- Dealing with implicit mid arguments

### Step 2: Try Option A (Stronger Induction)

Refactor to explicit vertex parameter:
```agda
path-from-orig-to-star-impossible : ∀ (v w : ForkVertex)
                                  → fst v ≡ fst w
                                  → snd v ≡ v-original
                                  → snd w ≡ v-fork-star
                                  → EdgePath v w → ⊥
```

### Step 3: Document Cospan Structure

Even if we postulate, create explicit cospan functor:
```agda
fork-cospan : ∀ (a : Node) → is-convergent a
            → Functor ·→·←· (OrientedGraphs o ℓ)
fork-cospan a conv .F₀ cs-a = (a, v-original)
fork-cospan a conv .F₀ cs-b = (a, v-fork-star)
fork-cospan a conv .F₀ cs-c = (a, v-fork-tang)
...
```

This makes the categorical structure explicit.

## Key Mathematical Facts (TO PRESERVE)

1. **Tang is terminal at same node** (no outgoing edges) ✓ PROVEN
2. **Handle direction is a → A** (original → tang) ✓ FIXED
3. **Cospan structure** ✓ DOCUMENTED
4. **G-acyclicity preserves** ✓ USING

Don't lose these insights when refactoring!

## Files to Reference

1. `/Users/faezs/homotopy-nn/src/Neural/Graph/ForkCategorical.agda` - Current work
2. `/Users/faezs/homotopy-nn/src/Neural/Topos/Architecture.agda` - HIT approach
3. `/nix/store/.../1lab/src/Cat/Instances/Graphs.lagda.md` - 1Lab graphs
4. `/nix/store/.../1lab/src/Cat/Instances/Shape/Cospan.lagda.md` - Cospan category
5. `/Users/faezs/homotopy-nn/src/ToposOfDNNs.agda` - Paper text (line 514-519 for handle direction)

## Session Summary

**What worked**:
- Identified correct edge direction (handle: a → A)
- Documented cospan structure
- Proved tang is terminal (no outgoing edges)
- Completed star → orig, tang → orig, tang → star cases

**What's blocked**:
- orig → star case (2 holes: orig-edge, tip-to-star)
- Implicit argument unification with `mid` parameter
- Transport of `rest` along path equality `pw`

**Mathematical understanding**: COMPLETE ✓
**Technical implementation**: PARTIAL (cubical Agda subtleties)
