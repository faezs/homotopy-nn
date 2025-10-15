# Roadmap: Eliminating Postulates from Physics.agda

This document provides a concrete plan for completing the smooth infinitesimal analysis implementation.

## Current Status (2025-10-13)

✅ **Physics.agda compiles successfully**
- 27 foundational axioms (KEEP)
- 16 derivable postulates (CAN REMOVE)
- 32 holes to fill

## Priority Roadmap

### 🔴 Priority 1: Prove Integration Principle

**Goal**: Replace the most important postulate in Integration.agda

**Location**: `src/Neural/Smooth/Integration.agda:92-93`

**Current**:
```agda
postulate
  integration-principle : (a b : ℝ) (f : ℝ → ℝ) →
    Antiderivative a b f
```

**Strategy**: Use microaffineness + constancy principle

**Detailed Approach**:

1. **Local Construction** (using microaffineness):
   ```agda
   -- For each point x, define F locally using infinitesimals
   -- F(x + δ) = F(x) + ι δ · f(x)
   -- This uses microaffineness: functions on Δ are affine
   ```

2. **Uniqueness** (using constancy-principle):
   ```agda
   -- If F and G both satisfy F' = f, then (F - G)' = 0
   -- By constancy-principle: F - G is constant
   -- With F(a) = G(a) = 0, we get F = G
   ```

3. **Global Patching**:
   ```agda
   -- Use interval-microstable to extend local construction
   -- Show that the local pieces agree on overlaps
   ```

**Reference**: This is essentially the synthetic approach to integration in SDG (Synthetic Differential Geometry). See:
- Kock, *Synthetic Differential Geometry*
- Lavendhomme, *Basic Concepts of Synthetic Differential Geometry*

**Proof Sketch**:
```agda
integration-principle : (a b : ℝ) (f : ℝ → ℝ) →
  Antiderivative a b f
integration-principle a b f = record
  { F = λ x → {- Construct using microaffineness -}
  ; F-derivative = λ x a≤x x≤b →
      {- Show F' = f using fundamental equation -}
  ; F-initial = {- Show F(a) = 0 by construction -}
  }
```

**Time Estimate**: 2-4 hours (requires understanding SDG integration theory)

---

### 🟠 Priority 2: Fill Integration Holes (10 holes)

These are needed for Physics.agda proofs.

#### Hole 1: `antiderivative-unique` (Line 117)
**What's needed**: Interval reasoning

```agda
same-deriv : ∀ y → F' ′[ y ] ≡ G' ′[ y ]
same-deriv y = {!!}
  -- Need: If a ≤ y ≤ b, then:
  -- F-deriv y (≤ℝ-trans a≤a a≤y) (≤ℝ-trans y≤b b≤b) : F' ′[ y ] ≡ f y
  -- G-deriv y (≤ℝ-trans a≤a a≤y) (≤ℝ-trans y≤b b≤b) : G' ′[ y ] ≡ f y
```

**Solution**: Add helper lemmas for interval arithmetic, then path reasoning.

#### Hole 2: `fundamental-theorem` (Line 176)
```agda
fundamental-theorem a b f F F-deriv = {!!}
  -- Proof: Let G = canonical antiderivative from integration-principle
  -- (F - G)' = 0, so F - G = c (constant)
  -- F(a) - G(a) = c, but G(a) = 0, so F(a) = c
  -- Therefore: ∫[a,b] f = G(b) - G(a) = F(b) - c - (F(a) - c) = F(b) - F(a)
```

#### Hole 3: `integral-derivative` (Line 190)
```agda
integral-derivative a b f x a≤x x≤b = {!!}
  -- Direct from integration-principle:
  -- The canonical antiderivative F has F' = f by definition
```

#### Hole 4: `∫-power` (Line 395)
```agda
-- Current: Need to apply fundamental-equation properly
≡⟨ {!!} ⟩  -- TODO: Need to apply fundamental-equation properly
((x +ℝ ι δ) ^ℝ n -ℝ (x ^ℝ n))
```

**Solution**: Use the rearrangement pattern from `sin-deriv` and `cos-deriv`:
```agda
≡⟨ sym (+ℝ-idl (ι δ ·ℝ (x ^ℝ n))) ⟩
-- ... algebra to rearrange ...
≡⟨ ap (_+ℝ (-ℝ (x ^ℝ n))) (sym (fundamental-equation (λ y → y ^ℝ n) x δ)) ⟩
```

#### Holes 5-10: Linearity and algebra
Lines ~270, ~290, etc.

**Tools needed**:
```agda
open import Algebra.Ring.Solver
-- Use automated ring solver for algebraic manipulations
```

**Time Estimate**: 4-6 hours

---

### 🟡 Priority 3: Fill DifferentialEquations Holes (11 holes)

#### Key Hole: `exp-unique` (Line 101)
```agda
exp-unique f g (f-ode , f-init) (g-ode , g-init) x = {!!}
  -- Proof: Let h = f - g
  -- h' = f' - g' = f - g = h  [by difference-derivative and f-ode, g-ode]
  -- So h' - h = 0
  -- Need a lemma: if h' = h, then either h = c·exp for some c, or...
  -- Actually, use: (f/g)' = (f'g - fg')/g² = (fg - fg)/g² = 0
  -- So f/g is constant
  -- Since f(0) = g(0) = 1, we have f/g = 1, so f = g
```

**Better approach**: Use constancy-principle on `(f/g)'`.

#### Holes for Taylor on Δₖ (Lines 142, 366, 376, 417, 420)

These all follow from `taylor-theorem` in HigherOrder.agda:

```agda
exp-on-Δ δ = {!!}  -- Apply taylor-theorem with k=1
sin-on-Δ δ = {!!}  -- Apply taylor-theorem with k=1
cos-on-Δ δ = {!!}  -- Apply taylor-theorem with k=1
```

**Prerequisite**: Fill the `taylor-sum-lemma` postulate in HigherOrder.agda.

#### Log property holes (Lines 440-456)

Need positivity proofs for products/quotients:
```agda
log-product : (x₊ y₊ : ℝ₊) →
  log (value x₊ ·ℝ value y₊ , {!!}) ≡ log x₊ +ℝ log y₊
  -- Hole: Need to prove value x₊ ·ℝ value y₊ > 0
  -- This follows from: a > 0, b > 0 ⟹ a·b > 0
```

**Time Estimate**: 6-8 hours

---

### 🟢 Priority 4: Replace Physics Postulates (4 postulates)

#### Postulate 1: `hyperbolic-sqrt` (Line 571)
**Difficulty**: ⭐ Easy

```agda
hyperbolic-sqrt : (x : ℝ) → ((1ℝ +ℝ ((sinh x) ²)) ^1/2) ≡ cosh x

-- Proof:
hyperbolic-sqrt x =
  ((1ℝ +ℝ ((sinh x) ²)) ^1/2)
    ≡⟨ ap _^1/2 (sym (cosh²-sinh²-identity x)) ⟩
  ((cosh x) ²) ^1/2
    ≡⟨ sqrt-of-square (cosh x) (cosh-positive x) ⟩
  cosh x
    ∎
  where
    -- From DifferentialEquations.agda (or derive):
    cosh²-sinh²-identity : cosh² - sinh² = 1
    cosh-positive : ∀ x → 0 < cosh x
    sqrt-of-square : ∀ x → 0 < x → (x² ^1/2) = x
```

#### Postulate 2: `center-of-pressure` (Line 370)
**Difficulty**: ⭐ Trivial (just define it)

```agda
-- Replace postulate with actual definition:
center-of-pressure : (ρ g : ℝ) (width : ℝ → ℝ) (depth : ℝ) → ℝ
center-of-pressure ρ g w d =
  let total-moment = ∫[ 0ℝ , d ] (λ h → ρ ·ℝ g ·ℝ (h ²) ·ℝ w h)
      total-force = ∫[ 0ℝ , d ] (λ h → ρ ·ℝ g ·ℝ h ·ℝ w h)
  in total-moment ·ℝ (total-force ^-1)
```

#### Postulate 3: `areal-law` (Line 799)
**Difficulty**: ⭐⭐ Medium

```agda
areal-law : (r θ : ℝ → ℝ) (t : ℝ) →
  let A = λ t → ((# 1) / (# 2)) ·ℝ ((r t) ²) ·ℝ (θ ′[ t ])
  in A ′′[ t ] ≡ 0ℝ

-- Proof sketch:
-- 1. A(t) = (1/2)·r²·θ'
-- 2. A'(t) = r·r'·θ' + (1/2)·r²·θ''  [by product-rule]
-- 3. For central force: angular acceleration = 0
--    ⟹ r·θ'' + 2·r'·θ' = 0
--    ⟹ r·θ'' = -2·r'·θ'
-- 4. Substitute: A'(t) = r·r'·θ' + (1/2)·r²·(-2·r'·θ'/r)
--                       = r·r'·θ' - r·r'·θ' = 0
```

#### Postulate 4: `angular-momentum-conserved` (Line 817)
**Difficulty**: ⭐ Easy (follows from areal-law)

```agda
angular-momentum-conserved : (m : ℝ) (r θ : ℝ → ℝ) (t₁ t₂ : ℝ) →
  angular-momentum m r θ t₁ ≡ angular-momentum m r θ t₂

-- Proof:
-- L(t) = m·r²·θ'
-- L'(t) = m·(2·r·r'·θ' + r²·θ'') = 2m·(r·r'·θ' + (r²/2)·θ'')
-- But r·r'·θ' + (r²/2)·θ'' = A'(t) = 0  [by areal-law]
-- So L'(t) = 0
-- By constancy-principle: L is constant
```

**Time Estimate**: 3-5 hours

---

### 🔵 Priority 5: Fill Physics Holes (7 holes)

Once Integration and DifferentialEquations holes are filled, these become straightforward applications.

#### Hole 1: `strip-moment-proof` (Line 112)
**Depends on**: `∫-power` from Integration.agda

#### Hole 2: `pappus-I-correct` (Line 277)
**Depends on**: Integration linearity

#### Hole 3: `beam-deflection-at-center` (Line 528)
**Pure algebra** - use ring solver

#### Holes 4-5: Catenary derivatives (Lines 590, 596)
**Chain rule applications**:
```agda
f'eq : f ′[ x ] ≡ sinh (x / a)
f'eq =
  -- f(x) = a · cosh(x/a)
  -- By scalar-rule: f'(x) = a · (cosh(x/a))'
  -- By composite-rule: (cosh(x/a))' = cosh'(x/a) · (1/a) = sinh(x/a) · (1/a)
  -- Therefore: f'(x) = a · sinh(x/a) · (1/a) = sinh(x/a)
  scalar-rule a (λ y → cosh (y / a)) x
    ∙ ap (a ·ℝ_) (composite-rule cosh (λ y → y / a) x)
    ∙ {- ... simplify ... -}
```

#### Holes 6-7: Bollard ODE (Lines 676, 684, 689)
**Similar chain rule applications**

**Time Estimate**: 2-3 hours

---

## Total Time Estimate

| Priority | Task | Time | Difficulty |
|----------|------|------|------------|
| 1 | Integration principle | 2-4 hrs | ⭐⭐⭐ |
| 2 | Integration holes | 4-6 hrs | ⭐⭐ |
| 3 | DiffEq holes | 6-8 hrs | ⭐⭐⭐ |
| 4 | Physics postulates | 3-5 hrs | ⭐⭐ |
| 5 | Physics holes | 2-3 hrs | ⭐ |
| **Total** | | **17-26 hrs** | |

## Success Milestones

### Milestone 1: Integration Complete (Priority 1-2)
- ✅ Integration principle proven
- ✅ All Integration.agda holes filled
- ✅ Integration.agda has only 0 postulates

### Milestone 2: Transcendentals Complete (Priority 3)
- ✅ All DifferentialEquations.agda holes filled
- ✅ exp, sin, cos properties fully proven (modulo existence axioms)

### Milestone 3: Physics Postulate-Free (Priority 4-5)
- ✅ All 4 Physics.agda postulates replaced
- ✅ All 7 Physics.agda holes filled
- ✅ Physics.agda is **completely postulate-free**

### Final Goal
- **27 foundational axioms** (in Base, Calculus, DifferentialEquations, Functions)
- **0 derivable postulates**
- **0 holes**

## Tools and Resources

### Agda Libraries to Use

```agda
-- For algebraic manipulations:
open import Algebra.Ring.Solver

-- For path reasoning:
open import 1Lab.Path.Reasoning

-- For case analysis:
open import Data.Sum using (_⊎_; inl; inr)

-- For constructing proofs:
open import 1Lab.HLevel
```

### 1Lab Features Available

- **Categories**: `Cat.*` - For categorical constructions
- **Algebra**: `Algebra.*` - Ring/field structures
- **Order**: `Order.*` - Posets and order theory
- **Path reasoning**: `1Lab.Path.Reasoning` - Equational reasoning

### Pattern Library

**Chain rule application pattern**:
```agda
composite-rule-example : (f g : ℝ → ℝ) (x : ℝ) →
  (f ∘ g) ′[ x ] ≡ (f ′[ g x ]) ·ℝ (g ′[ x ])

-- Use:
(cosh ∘ (λ y → y / a)) ′[ x ]
  ≡⟨ composite-rule cosh (λ y → y / a) x ⟩
(cosh ′[ x / a ]) ·ℝ ((λ y → y / a) ′[ x ])
  ≡⟨ ap₂ _·ℝ_ (cosh-derivative (x / a)) (quotient-derivative x a) ⟩
...
```

**Fundamental equation rearrangement pattern**:
```agda
-- Goal: ι δ ·ℝ f'(x) ≡ f(x+δ) - f(x)
ι δ ·ℝ f ′[ x ]
  ≡⟨ sym (+ℝ-idl _) ⟩
0ℝ +ℝ (ι δ ·ℝ f ′[ x ])
  ≡⟨ ap (_+ℝ _) (sym (+ℝ-invr (f x))) ⟩
-- ... shuffle terms ...
  ≡⟨ ap (_+ℝ _) (sym (fundamental-equation f x δ)) ⟩
f (x +ℝ ι δ) -ℝ f x
```

## Integration with Existing Code

### Related Completed Modules

- ✅ **Neural.Topos.Architecture** - DNN topos theory (Sections 1.1-1.4)
- ✅ **Neural.Stack.*** (19 modules) - Complete stack semantics (Sections 1.5-3.4)
- ✅ **Neural.Resources.*** - Resource theory, conversion rates, optimization
- ✅ **Neural.Information** - Neural codes, firing rates, metabolic constraints

### Potential Applications

Once Physics.agda is postulate-free:

1. **Neural.Smooth.Backpropagation** - Rigorous backprop via smooth analysis
2. **Neural.Smooth.Optimization** - Gradient descent as smooth dynamics
3. **Neural.Smooth.EnergyModels** - Physics-inspired neural architectures

## Quick Reference Commands

```bash
# Check compilation status
agda --library-file=./libraries src/Neural/Smooth/Physics.agda

# Find all holes
grep -n "{!!}" src/Neural/Smooth/*.agda

# Find all postulates
grep -n "^postulate" src/Neural/Smooth/*.agda

# Count lines of code
wc -l src/Neural/Smooth/*.agda

# Type-check a specific module
agda --library-file=./libraries src/Neural/Smooth/Integration.agda

# Interactive development (get goals/holes)
agda --interaction-json --library-file=./libraries
# Then: IOTCM "src/Neural/Smooth/Integration.agda" None Indirect (Cmd_load "src/Neural/Smooth/Integration.agda" [])
```

## Questions to Resolve

1. **Integration principle proof**: What's the exact construction in SDG?
   - Check Kock's book, Chapter on integration
   - Look for "integration axiom" vs "integration theorem"

2. **Taylor theorem on Δₖ**: Is `taylor-sum-lemma` provable or axiomatic?
   - Bell p. 93-94 has the proof sketch
   - Requires micropolynomiality axiom

3. **Positivity in log properties**: How to prove a > 0, b > 0 ⟹ a·b > 0?
   - May need order axioms from Base.agda
   - Check if `<ℝ-·ℝ-compat` suffices

## Conclusion

The path forward is clear:
1. **Prove integration principle** (highest impact)
2. **Fill holes systematically** (bottom-up: Integration → DiffEq → Physics)
3. **Replace derivable postulates** (straightforward once holes filled)

After ~20 hours of focused work, Physics.agda can be **completely postulate-free** with only 27 foundational axioms defining smooth infinitesimal analysis.
