# Research Summary: Species Core FinSets & Equivalence Composition

**Date**: 2025-10-16
**Focus**: Understanding how to prove `Fin-injective-id` using equivalence composition laws

---

## Key Insight: Equivalence Composition Pattern

Your observation about the structure of `Fin-peel` revealed the solution strategy:

```agda
Fin-peel (id , id-equiv)
  = Maybe-injective (Equiv.inverse Fin-suc ∙e (id , id-equiv) ∙e Fin-suc)
```

The composition flow:
1. `Fin-suc` : `Fin (suc n) ≃ Maybe (Fin n)`
2. `(id , id-equiv)` : `Fin (suc n) ≃ Fin (suc n)` (identity)
3. `Equiv.inverse Fin-suc` : `Maybe (Fin n) ≃ Fin (suc n)`

When composed:
```
Maybe (Fin n) --[Equiv.inverse Fin-suc]--> Fin (suc n) --[id]--> Fin (suc n) --[Fin-suc]--> Maybe (Fin n)
```

This should equal the identity on `Maybe (Fin n)` by:
1. **Identity composition**: `(id , id-equiv) ∙e Fin-suc = Fin-suc`
2. **Inverse cancellation**: `(Equiv.inverse Fin-suc) ∙e Fin-suc = id`

---

## Missing Equivalence Laws

### What We Need

```agda
∙e-idl : ∀ {A B : Type ℓ} (e : A ≃ B) → (id , id-equiv) ∙e e ≡ e
∙e-idr : ∀ {A B : Type ℓ} (e : A ≃ B) → e ∙e (id , id-equiv) ≡ e
∙e-invl : ∀ {A B : Type ℓ} (e : A ≃ B) → (e e⁻¹) ∙e e ≡ (id , id-equiv)
∙e-invr : ∀ {A B : Type ℓ} (e : A ≃ B) → e ∙e (e e⁻¹) ≡ (id , id-equiv)
```

### What 1Lab Provides

From `/nix/store/.../1Lab/Equiv.lagda.md`:

```agda
equiv→unit : ∀ {f : A → B} (eqv : is-equiv f) x
           → equiv→inverse eqv (f x) ≡ x

equiv→counit : ∀ {f : A → B} (eqv : is-equiv f) (y : B)
             → f (equiv→inverse eqv y) ≡ y
```

These are the **pointwise** inverse laws. We need to lift them to **equivalence-level** laws.

---

## Proof Strategy

### Step 1: Prove ∙e-idr

```agda
∙e-idr : ∀ {A B : Type ℓ} (e : A ≃ B) → e ∙e (id , id-equiv) ≡ e
∙e-idr (f , ef) = Σ-prop-path is-equiv-is-prop (funext λ x → refl)
```

**Reasoning**:
- First component: `f ∘ id = f` definitionally
- Second component: `is-equiv` is a proposition, so any two proofs are equal

### Step 2: Prove ∙e-invl

```agda
∙e-invl : ∀ {A B : Type ℓ} (e : A ≃ B) → (e e⁻¹) ∙e e ≡ (id , id-equiv)
∙e-invl (f , ef) = Σ-prop-path is-equiv-is-prop
  (funext λ x → equiv→unit ef x)
```

**Reasoning**:
- First component: `equiv→inverse ef ∘ f = id` pointwise by `equiv→unit`
- Second component: `is-equiv` is a proposition

### Step 3: Apply to Fin-peel

```agda
Fin-peel-id : Fin-peel (id , id-equiv) ≡ (id , id-equiv)
Fin-peel-id =
  Fin-peel (id , id-equiv)                                    ≡⟨⟩
  Maybe-injective (Equiv.inverse Fin-suc ∙e (id , id-equiv) ∙e Fin-suc)
                                                              ≡⟨ ap Maybe-injective (ap (_∙e Fin-suc) (∙e-idl _)) ⟩
  Maybe-injective (Equiv.inverse Fin-suc ∙e Fin-suc)         ≡⟨ ap Maybe-injective (∙e-invl Fin-suc) ⟩
  Maybe-injective (id , id-equiv)                             ≡⟨ Maybe-injective-id ⟩
  (id , id-equiv)                                             ∎
```

Where `Maybe-injective-id : Maybe-injective (id , id-equiv) ≡ (id , id-equiv)` would also need to be proven.

### Step 4: Complete Fin-injective-id

```agda
Fin-injective-id (suc n) =
  ap suc (Fin-injective (Fin-peel (id , id-equiv)))  ≡⟨ ap (ap suc ∘ Fin-injective) Fin-peel-id ⟩
  ap suc (Fin-injective (id , id-equiv))              ≡⟨ ap (ap suc) (Fin-injective-id n) ⟩
  ap suc refl                                         ≡⟨⟩
  refl                                                ∎
```

---

## Implementation Plan

### Phase 1: Prove Equivalence Composition Laws (2-3 hours)

Add to Species.agda after imports:

```agda
-- Equivalence composition identity laws
-- These should probably be contributed to 1Lab!

∙e-idl : ∀ {ℓ} {A B : Type ℓ} (e : A ≃ B) → (id , id-equiv) ∙e e ≡ e
∙e-idl (f , ef) = Σ-prop-path is-equiv-is-prop (funext λ x → refl)

∙e-idr : ∀ {ℓ} {A B : Type ℓ} (e : A ≃ B) → e ∙e (id , id-equiv) ≡ e
∙e-idr (f , ef) = Σ-prop-path is-equiv-is-prop (funext λ x → refl)

∙e-invl : ∀ {ℓ} {A B : Type ℓ} (e : A ≃ B) → (e e⁻¹) ∙e e ≡ (id , id-equiv)
∙e-invl (f , ef) = Σ-prop-path is-equiv-is-prop (funext (equiv→unit ef))

∙e-invr : ∀ {ℓ} {A B : Type ℓ} (e : A ≃ B) → e ∙e (e e⁻¹) ≡ (id , id-equiv)
∙e-invr (f , ef) = Σ-prop-path is-equiv-is-prop (funext (equiv→counit ef))
```

### Phase 2: Prove Maybe-injective-id (1 hour)

```agda
Maybe-injective-id : ∀ {ℓ} {A : Type ℓ} → Maybe-injective (id , id-equiv {A = Maybe A}) ≡ (id , id-equiv {A = A})
Maybe-injective-id = {! Unfold Maybe-injective definition and show it reduces to id !}
```

### Phase 3: Prove Fin-peel-id (30 min)

Using the above lemmas:

```agda
Fin-peel-id : ∀ {n} → Fin-peel (id , id-equiv {A = Fin (suc n)}) ≡ (id , id-equiv {A = Fin n})
Fin-peel-id =
  path-algebra-using-∙e-laws
```

### Phase 4: Complete Fin-injective-id (15 min)

```agda
Fin-injective-id (suc n) =
  ap (ap suc) (ap Fin-injective Fin-peel-id ∙ Fin-injective-id n)
```

---

## Technical Notes

### Σ-prop-path Usage

From `1Lab.Type.Sigma`:

```agda
Σ-prop-path : ∀ {B : A → Type ℓ'}
            → (∀ x → is-prop (B x))
            → {u v : Σ A B}
            → u .fst ≡ v .fst
            → u ≡ v
```

This is perfect for proving equality of equivalences where the second component (`is-equiv`) is a proposition.

### funext Usage

From `1Lab.Type.Pi`:

```agda
funext : {f g : (x : A) → B x} → (∀ x → f x ≡ g x) → f ≡ g
```

Needed to lift pointwise equalities to function equalities.

### Path Reasoning

Use the equational reasoning combinators:

```agda
open import 1Lab.Path.Reasoning

proof : a ≡ d
proof =
  a ≡⟨ reason-1 ⟩
  b ≡⟨ reason-2 ⟩
  c ≡⟨ reason-3 ⟩
  d ∎
```

---

## Why This Matters

### Mathematical Correctness

The proof of `Fin-injective-id` is a **foundational result** about how cardinality extraction behaves on the identity equivalence. It's needed for:

1. **Functor identity law** for Composition species
2. **Correctness of species operations** under relabeling
3. **Connection to classical combinatorics** where relabeling by identity is a no-op

### Broader Impact

These equivalence composition laws (`∙e-idl`, `∙e-idr`, `∙e-invl`, `∙e-invr`) are **generally useful** and should be in 1Lab. They would be valuable for:

- Any proof involving equivalence composition
- Univalence-based reasoning
- Category theory in HoTT (equivalences as isomorphisms)

**Recommendation**: Once proven, submit as a pull request to 1Lab with proper documentation.

---

## Estimated Effort

| Task | Time | Difficulty |
|------|------|------------|
| Prove ∙e-idl, ∙e-idr | 30 min | Easy (definitional) |
| Prove ∙e-invl, ∙e-invr | 1 hour | Medium (use equiv→unit/counit) |
| Prove Maybe-injective-id | 1-2 hours | Medium-Hard (unfold definition) |
| Prove Fin-peel-id | 30 min | Easy (use above lemmas) |
| Complete Fin-injective-id | 15 min | Easy (path algebra) |
| **Total** | **3-4 hours** | **Medium overall** |

---

## Alternative Approach

If the equivalence laws prove too difficult, we could:

1. **Postulate the composition laws** with clear documentation that they're foundational and should be provable
2. **Complete the species work** that depends on them
3. **Return later** to prove the laws from first principles

However, given your insight about the composition structure, I believe the proof is tractable and worth pursuing.

---

## Next Action

**Immediate**: Try implementing Phase 1 (equivalence composition laws) in Species.agda.

```agda
-- Add after line 145 (after 1Lab.Equiv import)
open import 1Lab.Type.Sigma using (Σ-prop-path)
open import 1Lab.HLevel using (is-equiv-is-prop)

-- Equivalence composition identity laws
∙e-idl : ∀ {ℓ} {A B : Type ℓ} (e : A ≃ B) → (id , id-equiv) ∙e e ≡ e
∙e-idl (f , ef) = Σ-prop-path is-equiv-is-prop (funext λ x → refl)

∙e-idr : ∀ {ℓ} {A B : Type ℓ} (e : A ≃ B) → e ∙e (id , id-equiv) ≡ e
∙e-idr (f , ef) = Σ-prop-path is-equiv-is-prop (funext λ x → refl)

∙e-invl : ∀ {ℓ} {A B : Type ℓ} (e : A ≃ B) → (e e⁻¹) ∙e e ≡ (id , id-equiv)
∙e-invl (f , ef) = Σ-prop-path is-equiv-is-prop (funext (equiv→unit ef))
```

Then test if these type-check and proceed to Phases 2-4.

---

**Generated**: 2025-10-16
**Key Discovery**: Equivalence composition structure in `Fin-peel`
**Impact**: Clear path to completing Goal 1 (Fin-injective-id)
**Status**: Ready to implement
