# Naturality Proof Strategy for α .is-natural

## Goal

Prove: `α .is-natural x y f` for all morphisms `f : x → y` in `Γ̄-Category`.

**Naturality square**:
```
F(x) ----α.η x----> G(x)
 |                   |
F(f)|               |G(f)
 |                   |
 v                   v
F(y) ----α.η y----> G(y)
```

Must prove: `α .η y ∘ F.F₁ f ≡ G.F₁ f ∘ α .η x`

## Construction of α (Recap)

```agda
α .η (v , v-original) = γ .η ((v , v-original) , inc tt)
α .η (v , v-fork-star) = λ x → Gsh .whole (lift false) (patch-at-star x)
α .η (v , v-fork-tang) = γ .η ((v , v-fork-tang) , inc tt)
```

## Case Analysis

Need to case-split on both **source vertex x** and **target vertex y**, plus the **edge f**.

### ForkVertex Type Combinations

|  x \ y  | v-original | v-fork-star | v-fork-tang |
|---------|------------|-------------|-------------|
| **v-original** | Case 1 | Case 2 | Case 3 |
| **v-fork-star** | IMPOSSIBLE | Case 4 | Case 5 |
| **v-fork-tang** | IMPOSSIBLE | IMPOSSIBLE | Case 6 |

**Impossibility reasoning**:
- **star → orig**: star-only-to-tang + tang-no-outgoing + tang≠orig
- **tang → orig**: tang-no-outgoing + tang-path-nil + tang≠orig
- **tang → star**: tang-no-outgoing + tang-path-nil + tang≠star

### Case 1: orig → orig

**Source**: `x = (x-node, v-original)`
**Target**: `y = (y-node, v-original)`
**Edge**: `f : Path-in Γ̄ (x-node, v-original) (y-node, v-original)`

**LHS**: `γ .η ((y-node, v-original), inc tt) ∘ F.F₁ f`
**RHS**: `G.F₁ f ∘ γ .η ((x-node, v-original), inc tt)`

**Strategy**:
1. Project `f` to X-path: `f-X = project-path-orig f`
2. Use `lift-project-roundtrip`: `lift-path f-X ≡ f`
3. Rewrite using roundtrip to get X-morphism
4. Apply γ's naturality: `γ .is-natural ((x-node, v-original), inc tt) ((y-node, v-original), inc tt) f-X`
5. Rewrite back using roundtrip

**Code pattern**:
```agda
case-orig-orig : ∀ {x-node y-node} (f : Path-in Γ̄ (x-node, v-original) (y-node, v-original))
               → α .η (y-node, v-original) ∘ F.F₁ f ≡ G.F₁ f ∘ α .η (x-node, v-original)
case-orig-orig f =
  let f-X = project-path-orig f
      rt = lift-project-roundtrip f
  in ap (λ p → α .η ... ∘ F.F₁ p) (sym rt)
     ∙ γ .is-natural ... f-X
     ∙ ap (λ p → G.F₁ p ∘ α .η ...) rt
```

### Case 2: orig → star

**Source**: `x = (x-node, v-original)`
**Target**: `y = (y-node, v-fork-star)`
**Edge**: `f : Path-in Γ̄ (x-node, v-original) (y-node, v-fork-star)`

**LHS**: `(λ z → Gsh .whole (lift false) (patch-at-star z)) ∘ F.F₁ f`
        `= λ x → Gsh .whole (lift false) (patch-at-star (F.F₁ f x))`

**RHS**: `G.F₁ f ∘ γ .η ((x-node, v-original), inc tt)`

**Strategy**:
This is tricky because LHS involves sheaf gluing, RHS involves γ.

**Key insight**: Need to use the **sheaf axiom** that `Gsh .whole` is natural.

**Substrategy**:
1. Expand `Gsh .whole (lift false) (patch-at-star (F.F₁ f x))`
2. Show this equals `G.F₁ f (Gsh .whole (lift false) (patch-at-star x))`
3. May need to use properties of patches and sheaf naturality

**Research needed**:
- How does `whole` behave under functoriality?
- Is there a lemma about `whole (F.F₁ f ∘ patch)`?

### Case 3: orig → tang (COMPLEX - Two Path Types!)

**Source**: `x = (x-node, v-original)` [in opposite category, this is the target in Γ̄]
**Target**: `y = (y-node, v-fork-tang)` [in opposite category, this is the source in Γ̄]
**Path**: `f : Path-in Γ̄ (y-node, v-original) (x-node, v-fork-tang)` [opposite direction!]

**LHS**: `γ .η ((x-node, v-fork-tang), inc tt) ∘ F.F₁ f`
**RHS**: `G.F₁ f ∘ γ .η ((y-node, v-original), inc tt)`

**CRITICAL**: Only TWO edges can reach tang vertices:
1. **handle**: `(a, v-original) → (a, v-fork-tang)` (same node)
2. **star-to-tang**: `(a, v-fork-star) → (a, v-fork-tang)` (same node)

**Path Classification** (see ORIG_TANG_PATH_ANALYSIS.md):

**Type 1: Via Handle (X-paths)**
- Structure: `(y, v-original) --orig-edges--> (x, v-original) --handle--> (x, v-fork-tang)`
- Last edge is `handle`
- Prefix stays in X (can use project-path-orig)
- **Strategy**:
  1. Pattern match: `f = cons ... (... (cons (handle ...) nil))`
  2. Decompose into prefix (X-path) + handle
  3. Use γ.is-natural on prefix
  4. Handle the final handle edge using functoriality

**Type 2: Via Star (NON-X-paths)**
- Structure: `(y, v-original) --...--> (a', v-original) --tip-to-star--> (x, v-fork-star) --star-to-tang--> (x, v-fork-tang)`
- Last edge is `star-to-tang`, second-to-last is `tip-to-star`
- Goes through `(x, v-fork-star)` which is NOT in X!
- CANNOT project to X-path
- **Strategy**:
  1. Pattern match: `f = cons ... (cons (tip-to-star ...) (cons (star-to-tang ...) nil))`
  2. Use sheaf gluing properties at star vertex
  3. Similar to orig→star case (see Case 2 below)

**Implementation approach**: Case split on path structure

### Case 4: star → star

**Source**: `x = (x-node, v-fork-star)`
**Target**: `y = (y-node, v-fork-star)`
**Edge**: `f : Path-in Γ̄ (x-node, v-fork-star) (y-node, v-fork-star)`

**Analysis**: What edges exist from star to star?
- Can't be `nil` (would require x = y as vertices)
- Can't be single `star-to-tang` (lands in tang, not star)
- Could be length ≥ 2 path going through tang... but tang has no outgoing!

**Expected**: This case is **IMPOSSIBLE** or very degenerate.

**Strategy**: Prove impossibility using edge structure.

### Case 5: star → tang

**Source**: `x = (x-node, v-fork-star)`
**Target**: `y = (y-node, v-fork-tang)`
**Edge**: `f : Path-in Γ̄ (x-node, v-fork-star) (y-node, v-fork-tang)`

**Analysis**: Only possibility is `cons (star-to-tang ...) nil`

**LHS**: `γ .η ((y-node, v-fork-tang), inc tt) ∘ F.F₁ f`
**RHS**: `G.F₁ f ∘ (λ z → Gsh .whole (lift false) (patch-at-star z))`

**Strategy**:
- This is the "dual" of Case 2
- May need sheaf gluing properties

### Case 6: tang → tang

**Source**: `x = (x-node, v-fork-tang)`
**Target**: `y = (y-node, v-fork-tang)`
**Edge**: `f : Path-in Γ̄ (x-node, v-fork-tang) (y-node, v-fork-tang)`

**Analysis**:
- Tang has no outgoing edges
- So `f` must be `nil` (x = y)
- This is the **identity case**

**Strategy**:
```agda
case-tang-tang nil =
  α .η (x-node, v-fork-tang) ∘ F.F-id
  ≡ γ .η ... ∘ id
  ≡ γ .η ...
  ≡ id ∘ γ .η ...
  ≡ G.F-id ∘ α .η (x-node, v-fork-tang)
```

## 1Lab Reasoning Modules

### Cat.Reasoning

**Import**:
```agda
open import Cat.Reasoning
module Γ̄ = Cat.Reasoning Γ̄-Category
module Sets = Cat.Reasoning (Sets (o ⊔ ℓ))
```

**Key combinators**:
- `eliml` / `elimr` - eliminate identity morphisms
- `introl` / `intror` - introduce identity morphisms
- `pulll` / `pullr` - reassociate pulling morphisms
- `pushl` / `pushr` - reassociate pushing morphisms
- `assoc` / `sym-assoc` - associativity

### Cat.Natural.Reasoning

**Import**:
```agda
import Cat.Natural.Reasoning as NatR
module γ-nat = NatR γ
```

**Key combinators**:
- `naturall` / `naturalr` - conjugate by naturality
- `viewl` / `viewr` - apply naturality and focus on subexpression
- `pulll` / `pullr` - pull natural transformation through composition
- `popl` / `popr` - pop functor composition and apply naturality
- `shiftl` / `shiftr` - shift natural transformation across composition

**Example usage**:
```agda
α .is-natural x y f =
  α .η y Sets.∘ F.F₁ f     ≡⟨ ... ⟩
  G.F₁ f Sets.∘ α .η x     ∎
```

### Cat.Functor.Reasoning

**Import**:
```agda
import Cat.Functor.Reasoning as FR
module F-R = FR F
module G-R = FR G
```

**Key combinators**:
- `F.collapse` - collapse functor composition
- `F.expand` - expand functor composition
- `F.pulll` / `F.pullr` - pull functor through composition
- `F.weave` - complex reassociation patterns

## Proof Organization Strategy

### Option 1: Direct Pattern Match

```agda
α .is-natural x y f with snd x | snd y
... | v-original | v-original = case-orig-orig f
... | v-original | v-fork-star = case-orig-star f
... | v-original | v-fork-tang = case-orig-tang f
... | v-fork-star | v-original = absurd (star→orig-impossible f)
... | v-fork-star | v-fork-star = case-star-star f
... | v-fork-star | v-fork-tang = case-star-tang f
... | v-fork-tang | v-original = absurd (tang→orig-impossible f)
... | v-fork-tang | v-fork-star = absurd (tang→star-impossible f)
... | v-fork-tang | v-fork-tang = case-tang-tang f
```

### Option 2: Helper Function

```agda
naturality-by-vertices : ∀ (x-type y-type : VertexType)
                       → (f : Path-in Γ̄ (x-node, x-type) (y-node, y-type))
                       → α .η (y-node, y-type) ∘ F.F₁ f
                         ≡ G.F₁ f ∘ α .η (x-node, x-type)
naturality-by-vertices v-original v-original f = case-orig-orig f
... (continue for each case)

α .is-natural x y f = naturality-by-vertices (snd x) (snd y) f
```

### Option 3: Path Reasoning with Modules

```agda
α .is-natural x y f with snd x | snd y
... | v-original | v-original =
  let f-X = project-path-orig f
      rt = lift-project-roundtrip f
      module _ = NatR γ
  in
    α .η y Sets.∘ F.F₁ f                ≡⟨ ap (α .η y Sets.∘_) (F-R.collapse (sym rt)) ⟩
    α .η y Sets.∘ F.F₁ (lift-path f-X)  ≡⟨ naturall ... ⟩
    G.F₁ (lift-path f-X) Sets.∘ α .η x  ≡⟨ ap (Sets._∘ α .η x) (G-R.collapse rt) ⟩
    G.F₁ f Sets.∘ α .η x                ∎
```

## Research Needed

### 1. Sheaf Whole Naturality

**Question**: Does `Gsh .whole` commute with functoriality?

**Look for**: Lemmas about sheafification and naturality in `Cat.Site.Base` or `Cat.Site.Sheafification`.

**Pattern to find**:
```agda
whole (cover) (F.F₁ f ∘ patch) ≡? F.F₁ f (whole (cover) patch)
```

### 2. Patch Functoriality

**Question**: How do patches compose with morphisms?

**Look for**: Properties of `Patch` in sheaf construction.

### 3. Identity Edge Cases

**Question**: How to handle `nil` edges efficiently?

**Pattern**:
```agda
with f
... | nil = idl _ ∙ sym (idr _)
```

### 4. Edge Impossibility Lemmas

Already have:
- `star-only-to-tang`
- `tang-no-outgoing`
- `tang-path-nil`
- `star≠orig`, `star≠tang`, `tang≠orig`

May need:
- General impossible edge characterizations
- Length constraints on paths between vertex types

## Implementation Plan

### Phase 1: Impossible Cases (Easy)

1. Prove `tang→orig` impossible
2. Prove `tang→star` impossible
3. Prove `star→orig` impossible

### Phase 2: Identity and Trivial Cases

1. `tang→tang` (only nil)
2. `star→star` (if any - may be impossible)

### Phase 3: γ-Naturality Cases (Medium)

1. `orig→orig` - use project-path-orig + γ naturality + roundtrip
2. `orig→tang` - similar to orig→orig

### Phase 4: Sheaf Gluing Cases (Hard)

1. `orig→star` - involves Gsh .whole on RHS
2. `star→tang` - involves Gsh .whole on LHS

**Note**: These may require understanding sheaf gluing naturality properties.

## Expected Challenges

1. **Sheaf whole naturality**: May need to use is-sheaf axioms
2. **Type-directed case analysis**: Lots of with-abstraction
3. **Witness management**: Need to track is-non-star proofs
4. **Path equalities**: Extensive use of ap and transport

## Resources

**1Lab Modules**:
- `Cat.Reasoning` - category reasoning combinators
- `Cat.Natural.Reasoning` - natural transformation reasoning
- `Cat.Functor.Reasoning` - functor reasoning
- `Cat.Site.Base` - sheaf axioms
- `1Lab.Path.Reasoning` - path equational reasoning

**Our Previous Work**:
- `project-path-orig` - converts Γ̄-paths to X-paths
- `lift-project-roundtrip` - proves they roundtrip
- `patch-compat-orig` - patch compatibility for orig→orig
- Edge impossibility lemmas

**Paper Reference**:
- Belfiore & Bennequin (2022), Section 1.5
- Theorem: DNN-Topos ≃ PSh(X)
- Fullness proof sketch (informal)

## Next Steps

1. ✅ Write this strategy document
2. Start with impossible cases (tang→orig, tang→star, star→orig)
3. Implement trivial cases (tang→tang, possibly star→star)
4. Tackle orig→orig using path projection machinery
5. Research sheaf whole naturality for orig→star and star→tang
6. Complete all cases

## Success Metrics

- **NO postulates** in the proof
- All 9 cases (or prove some impossible)
- Clean, readable proof using reasoning combinators
- Under 200 lines of proof code total
