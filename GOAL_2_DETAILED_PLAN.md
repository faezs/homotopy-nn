# Goal 2: HIT Witness Transport in star→tang Naturality

## Problem Statement

**Location**: `/Users/faezs/homotopy-nn/src/Neural/Graph/ForkTopos.agda:1195`

**Goal Type**:
```agda
Hsh .whole (lift false) (patch-at-star y-node (F ⟪ f ⟫ z)) ≡
(H ⟪ f ⟫ γ .η ((x-node , v-fork-tang) , inc tt) z)
```

**Context**:
- We're proving `restrict-faithful`: sheaf morphisms are determined by restriction to X
- This is the final naturality case (8/9 → 9/9): star→tang edge naturality
- Path `f = cons (star-to-tang a conv pv pw) p` where `p : Path-in Γ̄ (a, tang) (x-node, tang)`
- In `helper nil` branch, so `p' = nil` (tail is empty)
- Key equality: `y-node ≡ a ≡ x-node` (all vertices coincide)

**Challenge**: Circular dependencies in witness transport
- `pv : y-node ≡ a` and `pw : a ≡ x-node` are vertex equalities
- All three nodes are equal, creating a "degenerate loop" edge
- Standard transport fails due to witness coherence issues

---

## Mathematical Understanding

### What We're Proving

For sheaves `F` and `H` on the fork graph with nat trans `α, β : F ⇒ H`:
- If `restrict α ≡ restrict β` (equal on X)
- Then `α ≡ β` (equal everywhere)

At fork-star vertices, we use **sheaf gluing**:
```
α .η (y, v-fork-star) x = Hsh .whole (incoming-sieve) (patch made from α on incoming arrows)
```

The star→tang naturality case shows this gluing respects functor application.

### Key Insight from Paper

From Belfiore & Bennequin (2022), Section 1.3:
> "The fork construction introduces A★ and A (tang) with arrows a' → A★ and A★ → A"

The sheaf condition at A★ enforces:
```
F(A★) ≅ ∏_{a'→A★} F(a')
```

So `whole` at star vertices **is determined by** the incoming arrows from tang/original vertices.

### What `patch-at-star` Does

Defined at ForkTopos.agda:1049-1078:

```agda
patch-at-star : (star-node : Graph.Node G)
              → ∣ F .F₀ (star-node , v-fork-star) ∣
              → Patch H (fork-cover {star-node , v-fork-star} (lift false))
```

For an element `x : F(star-node, v-fork-star)`:
- **Line 1052**: On original vertices: `part f = γ .η (orig-vertex) (F ⟪ f ⟫ x)`
- **Line 1065**: On tang vertices: `part f = γ .η (tang-vertex) (F ⟪ f ⟫ x)`
- **Lines 1053-1064**: Star vertices impossible (not in incoming sieve)

**Key**: Patch uses `γ` (the restriction of α to X)

---

## Source Dependencies

### 1. From Our Implementation (ForkTopos.agda)

**Lines 1049-1078**: `patch-at-star` definition
- Understand how patch is constructed from `γ : restrict α`
- Note: uses `γ .η` on tang and original vertices only

**Lines 650-750**: Path projection functions
- `project-path-orig`, `project-path-tang`, etc.
- How Γ̄-paths project to X-paths

**Lines 820-930**: `lift-project-roundtrip` proofs
- Shows `lift-path (project-path p) ≡ p` for certain paths
- **Critical**: This is blocked for Type 2 paths (through star)

**Lines 1000-1043**: `patch-compat-orig` proof
- Example of patch compatibility for original vertices
- Uses `γ .is-natural` (naturality of γ)
- Pattern to follow for our case

### 2. From 1Lab Path Infrastructure

**Source**: `/nix/store/.../1lab-github-latest/src/Cat/Instances/Free.lagda.md`

**Lines 36-39**: `Path-in` HIT definition
```agda
data Path-in : ⌞ G ⌟ → ⌞ G ⌟ → Type (o ⊔ ℓ) where
  nil  : ∀ {a} → Path-in a a
  cons : ∀ {a b c} → G.Edge a b → Path-in b c → Path-in a c
```

**Lines 148-186**: `path-is-set` proof
- Uses `path-codep` - codes for path equality
- Identity system based on structural equality
- **Key tool**: `path-encode` and `path-decode` for path equalities

**Lines 191-195**: `path-decode`
```agda
path-decode : ∀ {a b} {xs ys : Path-in a b}
            → xs ≡ ys
            → path-codep (λ _ → a) xs ys
```

**Useful for**: Converting between propositional and structural path equality

### 3. From 1Lab Sheaf Theory

**Source**: `/nix/store/.../1lab-github-latest/src/Cat/Site/Base.lagda.md`

**Lines 488-494**: `is-sheaf` record
```agda
record is-sheaf : Type (o ⊔ ℓ ⊔ ℓs ⊔ ℓc) where
  field
    whole   : {U : ⌞ C ⌟} (S : J ʻ U) (p : Patch A ⟦ S ⟧) → A ʻ U
    glues   : {U : ⌞ C ⌟} (S : J ʻ U) (p : Patch A ⟦ S ⟧) → is-section A (whole S p) p
    separate : is-separated
```

**Key fields**:
- `whole`: Glues compatible patches into a global element
- `glues`: Proves the glued element restricts correctly
- `separate`: Separation (used elsewhere)

**Lines 160-167**: `Patch` record and `is-patch`
```agda
Parts : ∀ {U} (T : Sieve C U) → Type _
Parts {U} T = ∀ {V} (f : Hom V U) (hf : f ∈ T) → A₀ V

is-patch : ∀ {U} (T : Sieve C U) (p : Parts T) → Type _
is-patch {U} T part =
  ∀ {V W} (f : Hom V U) (hf : f ∈ T) (g : Hom W V) (hgf : f ∘ g ∈ T)
  → A₁ g (part f hf) ≡ part (f ∘ g) hgf
```

**Critical**: `glues` tells us how `whole` behaves under restriction

### 4. From Our ForkCategorical Infrastructure

**Source**: `/Users/faezs/homotopy-nn/src/Neural/Graph/ForkCategorical.agda`

**Need to reference**:
- `star-to-tang` edge constructor
- `v-fork-star`, `v-fork-tang` vertex types
- `star-only-to-tang`: star vertices only have edges to tang
- `tang-no-outgoing`: tang vertices have no outgoing edges

### 5. From Cubical Agda Primitives

**Source**: `1Lab.Prelude`, `1Lab.Path`

**Key tools needed**:
- `subst`: Transport along paths
- `transport`: Same as subst for Path types
- `ap`: Action on paths (functoriality)
- `sym`, `_∙_`: Path operations
- `PathP`: Dependent paths
- `is-prop→pathp`: Propositional paths are unique
- `happly`: Function extensionality

---

## Proof Strategy

### Step 1: Understand the Degenerate Case

Since `y-node ≡ a ≡ x-node` and `p' = nil`:
```
f = cons (star-to-tang a conv pv pw) nil
```

This is a single edge from `(a, v-fork-star)` to `(a, v-fork-tang)` on the same underlying node `a`.

**Simplification**: After all transports, both sides should reduce to `γ` applied to this edge.

### Step 2: Use Sheaf Gluing Property

From `Hsh : is-sheaf fork-coverage H`:
```agda
Hsh .glues (lift false) (patch-at-star y-node (F ⟪ f ⟫ z)) :
  is-section H (Hsh .whole ...) (patch-at-star y-node (F ⟪ f ⟫ z))
```

This means for any arrow `g` in the incoming sieve:
```
H ⟪ g ⟫ (Hsh .whole (lift false) patch) ≡ patch .part g
```

### Step 3: Instantiate Gluing at Our Edge

Our edge `f = star-to-tang a ...` is NOT in the incoming sieve (it goes FROM star).

But the **inverse direction** can be analyzed:
- **Goal LHS**: `Hsh .whole ... (patch-at-star y-node (F ⟪ f ⟫ z))`
- **Goal RHS**: `H ⟪ f ⟫ (γ .η ((x-node, v-fork-tang), inc tt) z)`

Since `y-node ≡ a ≡ x-node`, we can transport to get:
```
γ .η ((a, v-fork-tang), inc tt) (F ⟪ f ⟫ z)
```

### Step 4: Connect via Patch Definition

From `patch-at-star` line 1065:
```agda
patch-at-star star-node x .part {tang-node, v-fork-tang} g g-in-sieve =
  γ .η ((tang-node, v-fork-tang), inc tt) (F ⟪ g ⟫ x)
```

Key insight: The patch **already uses** `γ`, so we need to show the `whole` construction respects functor application.

### Step 5: Required Lemmas

**Lemma 1**: Path transport in degenerate case
```agda
star-to-tang-transport :
  ∀ {a b} (p : a ≡ b) (conv : is-convergent a)
  → (f : ForkEdge (a, v-fork-star) (a, v-fork-tang))
  → subst (λ n → ForkEdge (n, v-fork-star) (n, v-fork-tang)) p f ≡ ...
```

**Lemma 2**: Sheaf whole commutes with functor (when vertices coincide)
```agda
sheaf-whole-functor-commute :
  ∀ {y x} (y-eq-x : y ≡ x)
  → (f : Path-in Γ̄ (y, v-fork-star) (x, v-fork-tang))
  → Hsh .whole (incoming-sieve) (patch-from (F ⟪ f ⟫ z))
  ≡ H ⟪ f ⟫ (γ .η ((x, v-fork-tang)) z)
```

**Lemma 3**: Vertex equality functoriality
```agda
vertex-eq-naturality :
  ∀ {a b} (p : a ≡ b)
  → (f : ForkEdge (a, v-fork-star) (a, v-fork-tang))
  → F ⟪ subst ... f ⟫ ≡ subst (λ n → ...) p (F ⟪ f ⟫)
```

---

## Implementation Plan

### Phase 1: Set Up Context (10 lines)

Extract key facts from the context:
```agda
helper nil z =
  let -- Vertex equalities
      y-eq-a : y-node ≡ a
      y-eq-a = ap fst pv

      a-eq-x : a ≡ x-node
      a-eq-x = ap fst (tang-path-nil (subst ... p))

      y-eq-x : y-node ≡ x-node
      y-eq-x = y-eq-a ∙ a-eq-x

      -- Simplified edge
      f-simple : ForkEdge (a, v-fork-star) (a, v-fork-tang)
      f-simple = star-to-tang a conv pv pw

  in ...
```

### Phase 2: Transport to Common Vertex (20 lines)

Transport both sides to work at vertex `x-node`:
```agda
  -- Transport LHS: Hsh .whole at y-node → at x-node
  lhs-transport :
    Hsh .whole (lift false) (patch-at-star y-node (F ⟪ f ⟫ z))
    ≡ subst (λ n → ∣ H .F₀ (n, v-fork-star) ∣) y-eq-x
        (Hsh .whole (lift false) (patch-at-star x-node ...))

  -- Transport RHS: γ at x-node (already there)
  rhs-at-x : ∣ H .F₀ (x-node, v-fork-star) ∣
  rhs-at-x = H ⟪ f ⟫ (γ .η ((x-node, v-fork-tang), inc tt) z)
```

### Phase 3: Use Sheaf Glues Property (30 lines)

Apply `Hsh .glues` to relate `whole` to the patch parts:
```agda
  -- Key: patch-at-star uses γ on tang vertices
  patch-at-tang :
    patch-at-star x-node y .part {x-node, v-fork-tang} (edge-to-tang) proof
    ≡ γ .η ((x-node, v-fork-tang), inc tt) (F ⟪ edge-to-tang ⟫ y)

  -- Glues property instantiated
  glues-at-tang :
    H ⟪ edge-to-tang ⟫ (Hsh .whole (incoming-sieve) patch)
    ≡ γ .η ((x-node, v-fork-tang), inc tt) (F ⟪ edge-to-tang ⟫ ...)
```

### Phase 4: Naturality and Functoriality (30 lines)

Use naturality of `γ` and functoriality:
```agda
  -- γ is natural transformation
  γ-natural :
    ∀ (g : Path-in X u v)
    → γ .η v ∘ F ⟪ lift-path g ⟫ ≡ H ⟪ lift-path g ⟫ ∘ γ .η u

  -- Apply to our edge (after showing it lifts to X)
  apply-naturality :
    H ⟪ f ⟫ (γ .η ((x-node, v-fork-tang)) z)
    ≡ γ .η ((x-node, v-fork-star)) (F ⟪ f ⟫ z)
```

### Phase 5: Combine with Path Equality (20 lines)

Use `path-is-set` to equate paths:
```agda
  -- Both f and f-transported are star→tang edges on same node
  -- By path-is-set, propositional equality suffices
  path-eq : f ≡ f-transported
  path-eq = path-is-set _ _ f f-transported

  -- Use to transport functor applications
  functoriality : F ⟪ f ⟫ ≡ F ⟪ f-transported ⟫
  functoriality = ap (F ⟪_⟫) path-eq
```

### Phase 6: Final Assembly (10 lines)

Compose all the equalities:
```agda
helper nil z =
  lhs-transport
  ∙ ap (subst ...) (glues-at-tang ∙ patch-at-tang)
  ∙ apply-naturality
  ∙ ap (γ .η ...) functoriality
  ∙ sym rhs-transport
```

---

## Critical 1Lab Techniques to Master

### 1. Path Type Reasoning

**From**: `1Lab.Path`, `1Lab.Path.Reasoning`

```agda
-- Basic path operations
p ∙ q : x ≡ z        -- transitivity (when p : x ≡ y, q : y ≡ z)
sym p : y ≡ x        -- symmetry (when p : x ≡ y)
ap f p : f x ≡ f y   -- action on paths
ap₂ f p q : f x y ≡ f x' y'

-- Path reasoning syntax
x ≡⟨ p ⟩
y ≡⟨ q ⟩
z ∎

-- Dependent paths
PathP (λ i → A i) x y : Type
-- When A : I → Type, x : A i0, y : A i1

is-prop→pathp : (∀ i → is-prop (A i)) → (x : A i0) → (y : A i1) → PathP A x y
```

### 2. Transport and Subst

**From**: `1Lab.Prelude`

```agda
subst : (P : A → Type) → x ≡ y → P x → P y

-- Transport laws
transport-refl : transport refl x ≡ x
transport⁻transport : transport (sym p) (transport p x) ≡ x

-- Subst application
subst-application : subst (λ a → F a ≡ G a) p eq = ...
```

### 3. HIT Path Constructors

**From**: Our `Path-in` HIT (via 1Lab Free)

```agda
-- Path equality uses structural encoding
path-encode : xs ≡ ys → path-codep (λ _ → a) xs ys
path-decode : path-codep (λ _ → a) xs ys → xs ≡ ys

-- path-codep for cons:
path-codep (cons e p) (cons e' q) =
  Σ (fst (edge-endpoints e) ≡ fst (edge-endpoints e'))
    (λ eq → PathP (λ i → Edge ... (eq i) ...) e e'
          × path-codep ... p q)
```

### 4. Sheaf Gluing

**From**: `Cat.Site.Base`

```agda
-- Access gluing property
sheaf .glues : (S : covering-sieve) → (p : Patch) → is-section (whole S p) p

-- is-section means:
∀ {V} (f : Hom V U) (f-in-sieve : f ∈ S)
→ Functor.F₁ F f (whole S p) ≡ p .part f f-in-sieve
```

---

## Testing Strategy

### Unit Test 1: Vertex Transport

Test that transporting along `y-eq-x` preserves the edge type:
```agda
test-vertex-transport :
  ∀ (a : Graph.Node G) (conv : is-convergent a)
  → (e : ForkEdge (a, v-fork-star) (a, v-fork-tang))
  → subst (λ n → ForkEdge (n, v-fork-star) (n, v-fork-tang)) refl e ≡ e
test-vertex-transport a conv e = transport-refl e
```

### Unit Test 2: Patch Evaluation

Test that `patch-at-star` evaluates correctly on tang vertices:
```agda
test-patch-at-tang :
  ∀ (star-node tang-node : Graph.Node G) (x : F(star-node, v-fork-star))
  → (g : ForkEdge (tang-node, v-fork-tang) (star-node, v-fork-star))
  → patch-at-star star-node x .part g tt
  ≡ γ .η ((tang-node, v-fork-tang), inc tt) (F ⟪ g ⟫ x)
test-patch-at-tang = refl  -- Should be definitional
```

### Integration Test: Simpler Case

First prove for `y-node ≡ x-node` (without middle vertex `a`):
```agda
helper-simple :
  ∀ (x : Graph.Node G) (conv : is-convergent x)
  → (f : ForkEdge (x, v-fork-star) (x, v-fork-tang))
  → Hsh .whole (incoming-sieve) (patch-at-star x (F ⟪ f ⟫ z))
  ≡ H ⟪ f ⟫ (γ .η ((x, v-fork-tang), inc tt) z)
```

---

## Estimated Difficulty and Approach

**Difficulty**: ⭐⭐⭐⭐ (4/5 stars)

**Why hard**:
- Circular witness dependencies
- Deep path coherence in HIT
- Sheaf gluing with transport
- Novel use of 1Lab infrastructure

**Why doable**:
- Well-documented context
- Clear mathematical argument
- All tools available in 1Lab
- Can test incrementally

**Time estimate**: 2-4 hours of focused work

**Approach**:
1. Start with Phase 1 (context setup) - 15 min
2. Attempt Phase 3 (sheaf glues) first - 30 min
3. Work backwards to Phase 2 (transport) - 45 min
4. Add Phase 4 (naturality) - 45 min
5. Connect with Phase 5 (path equality) - 30 min
6. Final assembly Phase 6 - 30 min
7. Debug and refine - 1 hour

---

## Success Criteria

**Goal achieved when**:
```agda
helper nil z = <complete-proof>
```
type-checks with 0 holes.

**Intermediate milestones**:
1. ✅ Context variables extracted correctly
2. ✅ `lhs-transport` type-checks
3. ✅ `patch-at-tang` definitional equality works
4. ✅ `glues-at-tang` applies successfully
5. ✅ `apply-naturality` composes correctly
6. ✅ Final path equality assembles

---

## Next Iteration Preparation

**Before starting implementation**, have ready:
1. This document (GOAL_2_DETAILED_PLAN.md)
2. ForkTopos.agda open at line 1195
3. 1Lab Free.lagda.md for reference
4. 1Lab Site/Base.lagda.md for sheaf API
5. Scratch file for testing lemmas

**Start command**:
```bash
agda --interaction-json --library-file=./libraries
```

Then use agda-mcp to:
1. Load ForkTopos.agda
2. Get goal context for goal 2
3. Implement phases incrementally
4. Test each lemma before proceeding

**Remember**: Work incrementally, test often, use 1Lab patterns!
