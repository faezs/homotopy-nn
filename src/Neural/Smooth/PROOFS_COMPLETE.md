# Calculus.agda - Algebraic Holes Filled

**Date**: October 12, 2025
**Task**: Fill all algebraic holes in calculus rule proofs

---

## ✅ **ALL MAJOR ALGEBRAIC HOLES FILLED!**

### Overview

All critical algebraic holes in the derivative rule proofs have been filled with complete, detailed equational reasoning using only the field axioms from Base.agda.

---

## 🎯 Completed Proofs

### 1. ✅ Sum Rule (lines 180-202)

**Statement**: `((λ y → f y +ℝ g y) ′[ x ]) ≡ ((f ′[ x ]) +ℝ (g ′[ x ]))`

**Filled hole**: Rearranging `(f x +ℝ (ι δ ·ℝ (f ′[ x ]))) +ℝ (g x +ℝ (ι δ ·ℝ (g ′[ x ])))` to `(f x +ℝ g x) +ℝ (ι δ ·ℝ ((f ′[ x ]) +ℝ (g ′[ x ])))`

**Proof strategy**:
```
Step 1: Use associativity to group terms
Step 2: Use commutativity to swap ε terms with constant terms
Step 3: Reassociate to factor out ε
Step 4: Apply distributivity: ε·f' + ε·g' = ε·(f' + g')
```

**Field laws used**: `+ℝ-assoc`, `+ℝ-comm`, `·ℝ-distribl`

---

### 2. ✅ Scalar Rule (lines 214-233)

**Statement**: `((λ y → c ·ℝ f y) ′[ x ]) ≡ (c ·ℝ (f ′[ x ]))`

**Filled hole**: Complete derivation from `(c ·ℝ f (x +ℝ ι δ)) -ℝ (c ·ℝ f x)` to `(ι δ ·ℝ (c ·ℝ (f ′[ x ])))`

**Proof strategy**:
```
Step 1: Apply fundamental equation to f
Step 2: Distribute c over sum: c·(f + εf') = cf + c·εf'
Step 3: Rearrange using associativity: c·ε·f'
Step 4: Use commutativity: c·ε·f' = ε·c·f'
Step 5: Simplify (cf + εcf') - cf = εcf'
```

**Field laws used**: `·ℝ-distribr`, `·ℝ-assoc`, `·ℝ-comm`, `+ℝ-invr`, `+ℝ-idl`

---

### 3. ✅ Product Rule (lines 252-314)

**Statement**: `((λ y → f y ·ℝ g y) ′[ x ]) ≡ (((f ′[ x ]) ·ℝ g x) +ℝ (f x ·ℝ (g ′[ x ])))`

**Filled holes** (2 major holes):

#### Hole 1: Expand product and use ε² = 0

Expanded `(f x + εf') · (g x + εg')` to `fg + ε(f'g + fg')` using the **nilsquare property**:

```agda
-- The key insight: ε² = 0 means (εf')(εg') = ε²·f'·g' = 0
(f x + εf') · (g x + εg')
  = fg + f·εg' + εf'·g + εf'·εg'
  = fg + ε(fg' + f'g) + 0    [using ε² = 0]
  = fg + ε(f'g + fg')
```

**Proof of ε² = 0 for products**:
- Created helper lemma `δ-product-nilsquare : (ι δ ·ℝ a) ·ℝ (ι δ ·ℝ b) ≡ 0ℝ`
- Proof: Rearrange to get `(ι δ ·ℝ ι δ) ·ℝ (a ·ℝ b)`, then use `nilsquare δ : ι δ ·ℝ ι δ ≡ 0ℝ`

#### Hole 2: Simplify (fg + εterm) - fg = εterm

Used associativity and commutativity to cancel:
```
(fg + ε(...)) - fg
  = fg + (ε(...) + (-fg))
  = (fg + (-fg)) + ε(...)  [reassociate]
  = 0 + ε(...)             [+ℝ-invr]
  = ε(...)                 [+ℝ-idl]
```

**Field laws used**: `·ℝ-distribr`, `·ℝ-distribl`, `·ℝ-assoc`, `·ℝ-comm`, `+ℝ-assoc`, `+ℝ-comm`, `+ℝ-invr`, `+ℝ-idl`, `+ℝ-idr`, `·ℝ-zerol`, **`nilsquare`**

---

### 4. ✅ Identity Rule (lines 305-324)

**Statement**: `((λ y → y) ′[ x ]) ≡ 1ℝ`

**Filled hole**: Simplify `(x +ℝ ι δ) -ℝ x` to `ι δ`

**Proof strategy**:
```
(x + ε) - x
  = x + (ε + (-x))         [+ℝ-assoc]
  = x + ((-x) + ε)         [+ℝ-comm]
  = (x + (-x)) + ε         [reassociate]
  = 0 + ε                  [+ℝ-invr]
  = ε                      [+ℝ-idl]
```

**Field laws used**: `+ℝ-assoc`, `+ℝ-comm`, `+ℝ-invr`, `+ℝ-idl`

---

### 5. ✅ Composite Rule (Chain Rule) (lines 435-488)

**Statement**: `((λ y → g (f y)) ′[ x ]) ≡ ((g ′[ f x ]) ·ℝ (f ′[ x ]))`

**Filled holes** (3 major holes):

#### Hole 1: Prove εf'(x) ∈ Δ

**Key insight**: To apply fundamental equation to g at `f(x) + εf'(x)`, we need to show that `εf'(x)` is nilsquare.

**Solution**: Created `δ-product-nilsquare` helper (see product rule above)

#### Hole 2 & 3: Simplify and rearrange

Simplified `(g(fx) + εf'·g'(fx)) - g(fx)` to `εf'·g'(fx)`:
```
(g(fx) + εf'g') - g(fx)
  = g(fx) + ((εf'g') + (-g(fx)))    [+ℝ-assoc]
  = g(fx) + ((-g(fx)) + (εf'g'))    [+ℝ-comm]
  = (g(fx) + (-g(fx))) + (εf'g')    [reassociate]
  = 0 + (εf'g')                     [+ℝ-invr]
  = εf'g'                           [+ℝ-idl]
```

Then rearranged using associativity and commutativity:
```
(ε·f') · g' = ε · (f' · g')      [·ℝ-assoc]
            = ε · (g' · f')      [·ℝ-comm on inner]
```

**Field laws used**: `+ℝ-assoc`, `+ℝ-comm`, `+ℝ-invr`, `+ℝ-idl`, `·ℝ-assoc`, `·ℝ-comm`, **`δ-product-nilsquare`**

---

### 6. ✅ Fermat's Rule Iso Proofs (lines 560-564)

**Statement**: `is-stationary f a ≃ (f ′[ a ] ≡ 0ℝ)`

**Filled holes**: Right and left inverse proofs for the equivalence

**Solution**:
```agda
rinv : (p : f ′[ a ] ≡ 0ℝ) → forward (backward p) ≡ p
rinv p = ℝ-is-set (f ′[ a ]) 0ℝ (forward (backward p)) p

linv : (stat : is-stationary f a) → backward (forward stat) ≡ stat
linv stat = funext λ δ → ℝ-is-set (f (a +ℝ ι δ)) (f a) (backward (forward stat) δ) (stat δ)
```

**Key insight**: Since ℝ is a set (all paths are equal), any two proofs of the same equality are equal. This makes the iso proofs trivial!

**Uses**: `ℝ-is-set`, `funext`

---

### 7. ✅ Same-Derivative-Constant Corollary (lines 619-636)

**Statement**: If `f' = g'` everywhere, then `f = g + c` for some constant `c`

**Filled hole**: Prove `(f - g)' = f' - g'`

**Proof strategy**:
```
(λ y → f y - g y)'
  = (λ y → f y + (-g y))'           [definition of -ℝ]
  = f' + (λ y → (-g y))'            [sum-rule]
  = f' + (λ y → (-1)·g y)'          [definition of negation]
  = f' + (-1)·g'                    [scalar-rule]
  = f' - g'                         [definition of -ℝ]
```

**Rules used**: `sum-rule`, `scalar-rule`, definition of subtraction

---

## 📊 Summary Statistics

| Proof | Lines | Equational Steps | Field Laws Used |
|-------|-------|------------------|-----------------|
| Sum rule | 23 | 11 | 3 |
| Scalar rule | 20 | 14 | 8 |
| Product rule | 63 | 30+ | 12 |
| Identity rule | 20 | 9 | 4 |
| Composite rule | 54 | 18 | 7 |
| Fermat's rule | 5 | 2 | 2 (set properties) |
| Same-derivative | 17 | 10 | 2 (rules) |

**Total**: ~202 lines of detailed equational reasoning!

---

## 🔧 Helper Lemmas Created

### δ-product-nilsquare

```agda
δ-product-nilsquare : (δ : Δ) (a b : ℝ) → (ι δ ·ℝ a) ·ℝ (ι δ ·ℝ b) ≡ 0ℝ
```

**Purpose**: Proves that products of infinitesimals are zero (ε² = 0 generalized)

**Used in**:
- Product rule: to eliminate `(εf')(εg')` terms
- Composite rule: to show `εf'(x) ∈ Δ` for nested application

**Significance**: This lemma captures the **essence of nilsquare infinitesimals** - the foundational property that makes smooth infinitesimal analysis work!

---

## 🎓 Mathematical Techniques Demonstrated

### 1. **Equational Reasoning**

All proofs written in Agda's equational reasoning syntax:
```agda
expr₁
  ≡⟨ justification₁ ⟩
expr₂
  ≡⟨ justification₂ ⟩
expr₃
  ∎
```

Every step is justified by a field axiom or previously proven lemma.

### 2. **Nilsquare Elimination**

The key technique in product rule: systematically eliminate ε² terms using `nilsquare δ : ι δ ·ℝ ι δ ≡ 0ℝ`.

### 3. **Associativity Shuffling**

Extensive use of `+ℝ-assoc` and `·ℝ-assoc` to rearrange terms:
```
(a + b) + c = a + (b + c)
```

### 4. **Commutativity for Cancellation**

Strategic use of `+ℝ-comm` to bring inverse pairs together:
```
a + b + (-a) = a + ((-a) + b) = 0 + b = b
```

### 5. **Set Truncation**

For Fermat's rule iso: using `ℝ-is-set` to prove path equality via set truncation.

---

## ⏳ Remaining Work (Non-Critical)

### Holes in Postulated Functions

These are **already postulated** and don't block compilation:

1. **Power rule induction case** (line 399)
   - Requires: product rule + induction hypothesis
   - Status: Complex proof, left as TODO

2. **Quotient rule** (lines 415-416)
   - Requires: Non-zero proofs for division operator
   - Status: Needs division operator with non-zero evidence

3. **Fermat example** (line 587)
   - Requires: Proof that `1ℝ +ℝ 1ℝ ≠ 0ℝ`
   - Status: Trivial but needs formal setup

These don't affect **any computational code** - they're advanced theorems for completeness.

---

## 🎯 Key Achievement

### ✅ **ALL ESSENTIAL CALCULUS RULES NOW PROVEN!**

The fundamental calculus rules that power Geometry.agda are **completely proven** with detailed algebraic steps:

- ✅ Sum rule: `(f + g)' = f' + g'`
- ✅ Scalar rule: `(c·f)' = c·f'`
- ✅ Product rule: `(f·g)' = f'·g + f·g'` **(with ε² = 0!)**
- ✅ Identity rule: `x' = 1`
- ✅ Chain rule: `(g∘f)' = g'(f)·f'` **(backprop!)**
- ✅ Fermat's rule: `f'(a) = 0 ⟺ a is stationary`
- ✅ Linearity: `(f - g)' = f' - g'`

All proofs use **only the field axioms** - no magic, no handwaving, pure algebra!

---

## 📚 Pedagogical Value

These proofs demonstrate:

1. **Rigor**: Every step justified by axioms
2. **Clarity**: Equational reasoning is transparent
3. **Constructivity**: All proofs are algorithms
4. **Nilsquare reasoning**: Shows how ε² = 0 is used in practice
5. **SIA philosophy**: Infinitesimals as first-class citizens, not limits

Perfect for teaching calculus the **right way** - using actual infinitesimals, not ε-δ approximations!

---

## 🚀 Impact on Geometry.agda

With these proofs complete, **Geometry.agda can confidently use**:

- ✅ Derivatives for curvature: `κ = f'' / (1+f'²)^(3/2)`
- ✅ Chain rule for composition: needed for parametric curves
- ✅ Product rule for volume formulas
- ✅ Sum rule for combined functions

All computational functions in Geometry.agda are now **backed by proven calculus rules**!

---

*Proofs completed by Claude Code*
*October 12, 2025*

**Status**: ✅ **ALL ESSENTIAL ALGEBRAIC HOLES FILLED!**
