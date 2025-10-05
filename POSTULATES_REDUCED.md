# Postulate Reduction in Shannon.agda

## Summary

Successfully refactored `src/Neural/Information/Shannon.agda` to use **module parameterization** instead of scattered postulates. This is the mathematically correct approach in dependent type theory.

**Status**: ✅ Type-checks successfully with no errors or unsolved metas.

**Axiom Reduction**: 21 → 18 axioms (removed 3 redundant axioms, converted 2 to definitions)

## Key Changes

### 1. Created RealAnalysisStructure Record (18 axioms, down from 21)

All real analysis axioms are now grouped in a single record type:

**Decidability** (1 axiom):
- `is-zeroℝ : ℝ → Bool`

**Logarithm properties** (3 axioms):
- `log-product : (x y : ℝ) → logℝ (x *ℝ y) ≡ logℝ x +ℝ logℝ y`
- `log-concave : (x y t : ℝ) → ...`
- `log-term-nonpos : (p : ℝ) → zeroℝ ≤ℝ p → p ≤ℝ oneℝ → (p *ℝ logℝ p) ≤ℝ zeroℝ`

**Sum properties** (3 axioms):
- `sum-distrib : {n : Nat} → (f g : Fin n → ℝ) → sumℝ (λ i → f i +ℝ g i) ≡ sumℝ f +ℝ sumℝ g`
- `sum-factor : {n : Nat} → (c : ℝ) → (f : Fin n → ℝ) → sumℝ (λ i → c *ℝ f i) ≡ c *ℝ sumℝ f`
- `sum-nonneg : {n : Nat} → (f : Fin n → ℝ) → ((i : Fin n) → zeroℝ ≤ℝ f i) → zeroℝ ≤ℝ sumℝ f`

**Order structure** (6 axioms):
- `≥ℝ-refl, ≥ℝ-trans, ≥ℝ-antisym, ≥ℝ-prop` (poset axioms)
- `addition-right-monotone : {x y : ℝ} → y ≥ℝ zeroℝ → (x +ℝ y) ≥ℝ x`
- `product-nonneg : {x y : ℝ} → x ≥ℝ zeroℝ → y ≥ℝ zeroℝ → (x *ℝ y) ≥ℝ zeroℝ`

**Probability bounds** (1 axiom):
- `prob-le-one : {X : FinitePointedSet} → (P : ProbabilityMeasure X) → (x : underlying-set X) → P .prob x ≤ℝ oneℝ`
  - NOTE: `prob-nonneg` was REMOVED because ProbabilityMeasure.prob-nonneg already provides it!

**Fiberwise measures** (1 axiom):
- `fiberwise-nonneg : {X Y : FinitePointedSet} → (f : underlying-set X → underlying-set Y) → (Λ : FiberwiseMeasure f) → (y : underlying-set Y) → (x : underlying-set X) → zeroℝ ≤ℝ Λ y x`
  - TODO: Should be moved to PfSurjectiveMorphism (not "real analysis")

**Negation properties** (3 axioms):
- `neg-sum : {n : Nat} → (f : Fin n → ℝ) → -ℝ (sumℝ f) ≡ sumℝ (-ℝ ∘ f)`
- `neg-≤ : {x y : ℝ} → x ≤ℝ y → (-ℝ y) ≤ℝ (-ℝ x)`
- `neg-zero : -ℝ zeroℝ ≡ zeroℝ`

### 1.1 Order Relation Definitions

In `Neural.Information.agda`, we now define `≥ℝ` in terms of `≤ℝ`:
```agda
_≥ℝ_ : ℝ → ℝ → Type
x ≥ℝ y = y ≤ℝ x
```

This makes the order conversions `≥ℝ-to-≤ℝ` and `≤ℝ-to-≥ℝ` trivial (just `id`), so they're now **definitions** not axioms:
```agda
≥ℝ-to-≤ℝ : {x y : ℝ} → x ≥ℝ y → y ≤ℝ x
≥ℝ-to-≤ℝ p = p

≤ℝ-to-≥ℝ : {x y : ℝ} → x ≤ℝ y → y ≥ℝ x
≤ℝ-to-≥ℝ p = p
```

### 2. Module Parameterization

```agda
module _ (R-analysis : RealAnalysisStructure) where
  open RealAnalysisStructure R-analysis public

  -- All Shannon entropy theory proven from these axioms
```

### 3. Actual Proofs Written

✅ **shannon-entropy-nonneg** (lines 232-247):
Proved using:
- `transport` along `neg-sum` equality
- `sum-nonneg` for sums of non-negative terms
- `neg-≤` to flip order when negating
- `log-term-nonpos-lem` helper
- `neg-zero` simplification
- **Uses `P .prob-nonneg` field from ProbabilityMeasure (not axiom!)**

✅ **weighted-conditional-entropy-nonneg** (lines 297-300):
Proved using:
- `sum-nonneg` for weighted sum
- `product-nonneg` for probability × entropy terms
- `≥ℝ-to-≤ℝ` and `≤ℝ-to-≥ℝ` conversions (now trivial definitions)
- `shannon-entropy-nonneg` recursively
- **Uses `P .prob-nonneg` field from ProbabilityMeasure (not axiom!)**

✅ **fiberwise-to-prob** (lines 507-516):
Helper to convert fiberwise measure into ProbabilityMeasure.
Constructs using copattern matching:
- `prob x = Λ y x`
- `prob-nonneg` from `fiberwise-nonneg` axiom
- `prob-normalized = tt`
- `basepoint-positive` from `fiberwise-nonneg` at basepoint

✅ **information-loss** (lines 526-536):
Defined as weighted conditional entropy using `fiberwise-to-prob` helper.

✅ **information-loss-formula** (lines 560-566):
Proved by `refl` (definitional equality).

### 4. Remaining Postulates (7 blocks)

Inside the parameterized module, we have:

1. **shannon-extensivity-lemma** - Chain rule S(P') = S(P) + Σ_y P(y) S(Q|y)
   - Requires algebraic proof using log-product, sum-distrib, sum-factor

2. **Pf-surjective** - Subcategory of Pf with surjective morphisms
   - Standard construction, tedious but straightforward

3. **shannon-entropy-decreasing** - Morphisms decrease entropy
   - Follows from shannon-extensivity-lemma + weighted-conditional-entropy-nonneg + addition-right-monotone
   - Cannot be proven until shannon-extensivity-lemma is proven

4. **shannon-entropy-functor** - Functoriality of Shannon entropy
   - Follows from shannon-entropy-decreasing + morphism composition

5-7. Various postulates for embeddings, dilation factors, simplex category, binary entropy

## Comparison: Before vs After

### Before (scattered postulates):
- ~13 top-level postulates mixed throughout code
- Hard to see dependencies
- Unclear what's axiom vs. what needs proof
- **Had unsolved metas** preventing type-checking

### After (module parameterization):
- **✅ Type-checks successfully with NO errors or unsolved metas**
- **18 axioms** in RealAnalysisStructure (clear, grouped, mathematically principled)
  - **Removed 3 redundant axioms**:
    - `prob-nonneg` → use ProbabilityMeasure.prob-nonneg field
    - `≥ℝ-to-≤ℝ` → now a trivial definition (just `id`)
    - `≤ℝ-to-≥ℝ` → now a trivial definition (just `id`)
  - **Added ≥ℝ definition** in Neural.Information.agda: `x ≥ℝ y = y ≤ℝ x`
- **5 actual definitions/proofs written**:
  - shannon-entropy-nonneg
  - weighted-conditional-entropy-nonneg
  - fiberwise-to-prob (helper construction)
  - information-loss (definition)
  - information-loss-formula (definitional equality)
- **2 conversion functions** now definitions instead of axioms
- 7 remaining postulates inside module (mostly category-theoretic constructions)
- Clear separation: axioms are parameters, postulates are TODOs
- **All unsolved metas resolved**

## Next Steps

1. Prove **shannon-extensivity-lemma** using algebraic manipulation with:
   - log-product for log(xy) = log x + log y
   - sum-distrib for distributing sums
   - sum-factor for factoring constants

2. Replace **shannon-entropy-decreasing** postulate with actual proof:
   ```agda
   shannon-entropy-decreasing ϕ =
     transport (λ i → shannon-extensivity-lemma (...) i ≥ℝ shannon-entropy (YP .snd))
               (addition-right-monotone (weighted-conditional-entropy-nonneg ...))
   ```

3. Build **Pf-surjective** category using 1Lab's subcategory construction
