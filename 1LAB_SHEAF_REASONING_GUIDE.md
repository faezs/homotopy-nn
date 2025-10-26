# 1Lab Sheaf and Naturality Reasoning Guide

## Date: 2025-10-24

## Key Discovery: `map-patch` is the Sheaf Naturality Lemma!

From `/nix/store/.../src/Cat/Site/Base.lagda.md` line 288-293:

```agda
map-patch
  : ∀ {B : Functor (C ^op) (Sets ℓs)} {U} {S : Sieve C U} (eta : A => B)
  → Patch A S
  → Patch B S
map-patch eta x .part f hf = eta .η _ (x .part f hf)
map-patch eta x .patch f hf g hgf = sym (eta .is-natural _ _ _ $ₚ _) ∙ ap (eta .η _) (x .patch f hf g hgf)
```

**What this does**: Given a natural transformation `η : A => B` and a patch for `A`, produces a patch for `B` by applying `η` pointwise.

**Key property** (line 293): The compatibility condition uses `η.is-natural`!

## Sheaf Structure (From Cat/Site/Base.lagda.md)

### Patch Definition (lines 202-217)

```agda
record Patch (A : Functor (C ^op) (Sets ℓs)) (S : Sieve C U) : Type where
  field
    part  : ∀ {V} (f : Hom V U) → f ∈ S → A ʻ V
    patch : ∀ {V W} (f : Hom V U) (hf : f ∈ S) (g : Hom W V) (hgf : (f ∘ g) ∈ S)
          → A ⟪ g ⟫ (part f hf) ≡ part (f ∘ g) hgf
```

**Interpretation**: A patch assigns values to each covering morphism, with compatibility on composites.

### Sheaf Gluing (lines 488-505)

```agda
record is-sheaf : Type where
  field
    whole   : {U : ⌞ C ⌟} (S : J ʻ U) (p : Patch A ⟦ S ⟧) → A ʻ U
    glues   : {U : ⌞ C ⌟} (S : J ʻ U) (p : Patch A ⟦ S ⟧) → is-section A (whole S p) p
    separate : is-separated
```

**Key functions**:
- `whole S p`: Glues local patches into global section
- `glues S p`: Proves that restricting `whole S p` gives back the original patches

### Section (lines 222-231)

```agda
record Section (A : Functor (C ^op) (Sets ℓs)) (p : Patch A T) : Type where
  field
    whole : A ʻ U
    glues : is-section A whole p
```

**is-section** (line 219-220):
```agda
is-section : ∀ {U} {T : Sieve C U} (s : A ʻ U) (p : Patch A T) → Type _
is-section {T = T} s p = ∀ {V} (f : Hom V _) (hf : f ∈ T) → A ⟪ f ⟫ s ≡ p .part f hf
```

## Cat.Natural.Reasoning Combinators

From `/nix/store/.../src/Cat/Natural/Reasoning.lagda.md`:

### Naturality Rearrangement (lines 48-71)

```agda
naturall : η _ ∘ F.₁ a ≡ η _ ∘ F.₁ b
         → G.₁ a ∘ η _ ≡ G.₁ b ∘ η _

naturalr : G.₁ a ∘ η _ ≡ G.₁ b ∘ η _
         → η _ ∘ F.₁ a ≡ η _ ∘ F.₁ b

viewr : G.₁ a ≡ G.₁ b
      → η _ ∘ F.₁ a ≡ η _ ∘ F.₁ b

viewl : F.₁ a ≡ F.₁ b
      → G.₁ a ∘ η _ ≡ G.₁ b ∘ η _
```

### Pulling Through Composites (lines 78-111)

```agda
pulll : (p : η _ ∘ f ≡ g)
      → η _ ∘ F.₁ a ∘ f ≡ G.₁ a ∘ g

pullr : (p : f ∘ η _ ≡ g)
      → (f ∘ G.₁ a) ∘ η _ ≡ g ∘ F.₁ a

popl : (p : η _ ∘ F.₁ b ≡ f)
     → η _ ∘ F.₁ (a ∘ b) ≡ G.₁ a ∘ f

popr : (p : G.₁ a ∘ η _ ≡ f)
     → G.₁ (a ∘ b) ∘ η _ ≡ f ∘ F.₁ b

shiftl : (p : F.₁ a ∘ f ≡ g)
       → G.₁ a ∘ η _ ∘ f ≡ η _ ∘ g

shiftr : (p : f ∘ G.₁ a ≡ g)
       → (f ∘ η _) ∘ F.₁ a ≡ g ∘ η _
```

## Cat.Reasoning Combinators

From `/nix/store/.../src/Cat/Reasoning.lagda.md`:

### Identity Manipulation (lines 73-95)

```agda
eliml : id ∘ f ≡ f
elimr : f ∘ id ≡ f
introl : f ≡ id ∘ f
intror : f ≡ f ∘ id
```

### Composition Rearrangement (lines 106-164)

```agda
pulll : (p : a ∘ b ≡ c)
      → a ∘ (b ∘ f) ≡ c ∘ f

pullr : (p : a ∘ b ≡ c)
      → (f ∘ a) ∘ b ≡ f ∘ c

pushl : (p : a ∘ b ≡ c)
      → c ∘ f ≡ a ∘ (b ∘ f)

pushr : (p : a ∘ b ≡ c)
      → f ∘ c ≡ (f ∘ a) ∘ b
```

## Application to Our Naturality Proofs

### For Type 1 orig→tang (Via Handle - X-paths)

**Goal**: Prove `α .η (x, tang) ∘ F.F₁ f ≡ G.F₁ f ∘ α .η (y, orig)`

**Strategy using projection**:
```agda
α .is-natural (x , v-fork-tang) (y , v-original) f =
  let f-X = project-path-orig-to-tang f  -- Type 1 projection
      roundtrip = lift-project-roundtrip-tang f  -- Need to prove
      module γ-nat = NatR γ  -- Import reasoning combinators
  in ext λ z →
    α .η (x , tang) (F.F₁ f z)                           ≡⟨⟩
    γ .η ((x , tang), inc tt) (F.F₁ f z)                 ≡⟨ ap (γ .η ...) (ap (λ p → F.F₁ p z) (sym roundtrip)) ⟩
    γ .η ((x , tang), inc tt) (F.F₁ (lift-path f-X) z)   ≡⟨ happly (γ .is-natural ... f-X) z ⟩
    G.F₁ (lift-path f-X) (γ .η ((y , orig), inc tt) z)   ≡⟨ ap (λ p → G.F₁ p ...) roundtrip ⟩
    G.F₁ f (γ .η ((y , orig), inc tt) z)                 ≡⟨⟩
    G.F₁ f (α .η (y , orig) z)                           ∎
```

**Using Natural.Reasoning combinators** (cleaner):
```agda
module γ-nat = NatR γ
in ext λ z →
  γ-nat.viewr (ap (λ p → F.F₁ p z) (sym roundtrip))   -- Apply roundtrip on F side
  ∙ happly (γ .is-natural ... f-X) z                  -- Main naturality
  ∙ γ-nat.viewl (ap (λ p → G.F₁ p ...) roundtrip)     -- Apply roundtrip on G side
```

### For Type 2 orig→tang (Via Star - Non-X-paths)

**Challenge**: Path goes through star, cannot use γ.is-natural directly.

**Structure**: `f = prefix ++ [tip-to-star] ++ [star-to-tang]`

**Key insight from map-patch**: We can use patch naturality!

**Approach**:
1. Recognize that `α .η (x, star)` is defined via sheaf gluing
2. The patch used for gluing involves γ on original vertices
3. Use `map-patch` to relate patches under natural transformations
4. Show that `F.F₁ f` and `G.F₁ f` respect the patch structure

**Concrete strategy** (needs development):
```agda
α .is-natural (x , v-fork-tang) (y , v-original) (cons e-tip (cons e-star nil)) =
  -- e-tip : tip-to-star, e-star : star-to-tang
  -- α.η at star uses: λ z → Gsh .whole (lift false) (patch-at-star z)
  -- Need to show naturality through sheaf gluing
  ext λ z →
    -- LHS: α .η (x, tang) (F.F₁ (tip+star) z)
    --    = γ .η ((x, tang), ...) (F.F₁ (tip+star) z)
    -- RHS: G.F₁ (tip+star) (α .η (y, orig) z)
    --    = G.F₁ (tip+star) (γ .η ((y, orig), ...) z)
    {!!}  -- Use sheaf gluing properties + map-patch
```

### For orig→star (Sheaf Gluing Case)

**Goal**: Prove `α .η (x, star) ∘ F.F₁ f ≡ G.F₁ f ∘ α .η (y, orig)`

where `α .η (x, star) = λ z → Gsh .whole (lift false) (patch-at-star z)`

**Structure of patch-at-star** (from ForkTopos.agda lines 910-970):
```agda
patch-at-star x .part f hf = γ .η ((vertex f , v-original), inc tt) (F.F₁ f x)
patch-at-star x .patch f hf g hgf = patch-compat-orig
```

**Key observation**: The patch already uses γ! So naturality follows from:
1. `map-patch γ (patch-at-star (F.F₁ path x))` relates to `patch-at-star (G.F₁ path (γ.η y x))`
2. Sheaf gluing `whole` respects natural transformations via patches

**Proof sketch**:
```agda
α .is-natural (x , v-fork-star) (y , v-original) f =
  ext λ z →
    Gsh .whole (lift false) (patch-at-star (F.F₁ f z))    ≡⟨ {! use map-patch !} ⟩
    Gsh .whole (lift false) (map-patch γ (patch-for-F))   ≡⟨ {! sheaf gluing naturality !} ⟩
    G.F₁ f (Gsh .whole (lift false) (patch-at-star z))     ≡⟨⟩
    G.F₁ f (α .η (y , orig) z)                             ∎
```

## What We Still Need

### 1. Sheaf Gluing Naturality Lemma

**Conjecture**: If `η : A => B` and `Ash, Bsh` are sheaves, then:
```agda
whole-natural : (p : Patch A S)
              → η .η U (Ash .whole S p) ≡ Bsh .whole S (map-patch η p)
```

**Status**: NOT found explicitly in 1Lab, but should follow from:
- Sheaf condition (uniqueness of gluing)
- `map-patch` preserves patch structure
- Likely needs to be proven using `separate` (uniqueness)

### 2. Roundtrip for orig-to-tang Paths

**Need to prove**: `lift-path (project-path-orig-to-tang f) ≡ f` for Type 1 paths

**Similar to**: `lift-project-roundtrip` for orig→orig (already done)

**Strategy**: Induction on path structure, pattern match on edges

### 3. Star→Star Path Analysis

**Question**: Are there non-nil paths from star to star?

**Analysis**:
- Star only goes to tang (star-to-tang)
- Tang has no outgoing edges
- Therefore: star → tang → ??? (stuck!)
- **Conclusion**: Only nil is possible, making this an identity case

## Recommended Implementation Order

### Phase 1: Complete Type 1 Paths (Projection-Based)

1. ✅ Implement `project-path-orig-to-tang` for handle case (DONE)
2. **Next**: Prove `lift-project-roundtrip-tang` (roundtrip for orig-to-tang)
3. **Then**: Prove naturality for Type 1 orig→tang using projection + γ.is-natural

### Phase 2: Sheaf Gluing Lemmas (Foundation for Type 2)

1. **Prove or postulate**: `whole-natural` lemma
2. **Document**: How `map-patch` connects patches under natural transformations
3. **Understand**: Relationship between sheaf gluing and functoriality

### Phase 3: Prove Sheaf Gluing Cases

1. **orig→star**: Use `whole-natural` + patch structure
2. **star→tang**: Similar to orig→star
3. **orig→tang Type 2**: Combines both patterns

### Phase 4: Complete Remaining Cases

1. **star→star**: Prove only nil possible, use identity case
2. **restrict-ess-surj**: Essential surjectivity (extension functor)

## Practical Examples

### Using Cat.Natural.Reasoning

```agda
-- Import the module with your natural transformation
import Cat.Natural.Reasoning as NatR
module γ-nat = NatR γ

-- In your proof:
proof = ext λ z →
  γ-nat.pulll lemma1    -- Pull through composition on left
  ∙ happly (γ .is-natural x y f-X) z    -- Main naturality
  ∙ γ-nat.pullr lemma2  -- Pull through composition on right
```

### Using Cat.Reasoning

```agda
module Sets = Cat.Reasoning (Sets (o ⊔ ℓ))

proof = ext λ z →
  ap (α .η y) (Sets.eliml F.F-id)    -- Eliminate identity on F side
  ∙ naturality-core z                 -- Main proof
  ∙ Sets.elimr G.F-id                 -- Eliminate identity on G side
```

## Key Takeaways

1. **`map-patch` is crucial**: It's the naturality lemma for patches
2. **Sheaf gluing respects naturality**: Via `map-patch` and uniqueness
3. **Reasoning combinators are powerful**: Use Cat.Natural.Reasoning extensively
4. **Type 1 and Type 2 need different strategies**: Projection vs sheaf gluing
5. **We have the tools**: Just need to assemble them correctly!
