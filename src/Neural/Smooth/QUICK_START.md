# Quick Start Guide: Resuming Work on Physics.agda

This guide helps you quickly resume work on eliminating postulates from the smooth infinitesimal analysis implementation.

## TL;DR

```bash
# 1. Verify everything still compiles
cd /Users/faezs/homotopy-nn
agda --library-file=./libraries src/Neural/Smooth/Physics.agda

# 2. Pick a task from ROADMAP.md

# 3. Make edits, then recompile
agda --library-file=./libraries src/Neural/Smooth/Physics.agda

# 4. Commit when done
git add src/Neural/Smooth/
git commit -m "Fill [specific hole/postulate name]"
```

## Current Status Snapshot

**Date**: 2025-10-13

✅ **Physics.agda compiles** with `--allow-unsolved-metas`

**What we have**:
- 27 foundational axioms (keep these)
- 16 derivable postulates (can eliminate)
- 32 holes to fill ({!!})

**See full details**: `COMPILATION_STATUS.md`

## File Overview

| File | Purpose | Status |
|------|---------|--------|
| `COMPILATION_STATUS.md` | Current state, what compiles, what doesn't | ✅ Up to date |
| `ROADMAP.md` | Detailed plan with time estimates | ✅ Ready to use |
| `POSTULATES_AUDIT.md` | Comprehensive analysis of all postulates | 📋 Reference |
| This file | Quick commands to get started | 📖 You are here |

## Recommended Workflow

### Option 1: Fill a Specific Hole (Easiest)

**Choose from**:
- Integration.agda: Lines 117, 176, 190, 395, ~270, ~290
- DifferentialEquations.agda: Lines 101, 142, ~215, ~222, ~366, ~376, ~417, ~420, ~440-456
- Physics.agda: Lines 112, 277, 528, 590, 596, 676, 684, 689

**Example**: Fill the `∫-power` hole (Integration.agda:395)

1. **Open the file**:
   ```bash
   # Use your preferred editor
   vim src/Neural/Smooth/Integration.agda +395
   # or
   code src/Neural/Smooth/Integration.agda:395
   ```

2. **Read the context**:
   - Look at the comment above the hole
   - Check what lemmas are available (scroll up)
   - See the proof strategy in the comments

3. **Fill the hole**:
   ```agda
   ≡⟨ {!!} ⟩  -- TODO: Need to apply fundamental-equation properly

   -- Replace with:
   ≡⟨ sym (+ℝ-idl (ι δ ·ℝ (x ^ℝ n))) ⟩
   0ℝ +ℝ (ι δ ·ℝ (x ^ℝ n))
     ≡⟨ ap (_+ℝ (ι δ ·ℝ (x ^ℝ n))) (sym (+ℝ-invr (x ^ℝ n))) ⟩
   -- ... continue rearrangement pattern ...
   ```

4. **Test**:
   ```bash
   agda --library-file=./libraries src/Neural/Smooth/Integration.agda
   ```

5. **Commit**:
   ```bash
   git add src/Neural/Smooth/Integration.agda
   git commit -m "Fill ∫-power hole using fundamental-equation"
   ```

### Option 2: Replace a Postulate (Medium)

**Best candidates**:
- `center-of-pressure` (Physics.agda:370) - Just define it
- `hyperbolic-sqrt` (Physics.agda:571) - Use cosh²-sinh² identity

**Example**: Replace `center-of-pressure`

1. **Find the postulate**:
   ```bash
   grep -n "center-of-pressure" src/Neural/Smooth/Physics.agda
   # Line 370
   ```

2. **Replace it**:
   ```agda
   postulate
     center-of-pressure : (ρ g : ℝ) (width : ℝ → ℝ) (depth : ℝ) → ℝ

   -- Replace with:
   center-of-pressure : (ρ g : ℝ) (width : ℝ → ℝ) (depth : ℝ) → ℝ
   center-of-pressure ρ g w d =
     let moment = ∫[ 0ℝ , d ] (λ h → ρ ·ℝ g ·ℝ (h ²) ·ℝ w h)
         force = ∫[ 0ℝ , d ] (λ h → ρ ·ℝ g ·ℝ h ·ℝ w h)
     in moment ·ℝ (force ^-1)
   ```

3. **Test and commit** (as above)

### Option 3: Prove Integration Principle (Advanced)

**This is the highest-impact task** - eliminates a fundamental axiom.

See detailed strategy in `ROADMAP.md` Priority 1.

**Prerequisites**:
- Understand microaffineness (Base.agda)
- Understand constancy-principle (Calculus.agda)
- Understand SDG integration theory (Kock's book)

**Time**: 2-4 hours

**Start here**:
```bash
vim src/Neural/Smooth/Integration.agda +92
# Find the integration-principle postulate
```

## Common Patterns

### Pattern 1: Rearranging fundamental-equation

**Use when**: You need to show `ι δ ·ℝ f'(x) ≡ f(x+δ) - f(x)`

**Template**:
```agda
ι δ ·ℝ f ′[ x ]
  ≡⟨ sym (+ℝ-idl (ι δ ·ℝ f ′[ x ])) ⟩
0ℝ +ℝ (ι δ ·ℝ f ′[ x ])
  ≡⟨ ap (_+ℝ (ι δ ·ℝ f ′[ x ])) (sym (+ℝ-invr (f x))) ⟩
((f x) +ℝ (-ℝ f x)) +ℝ (ι δ ·ℝ f ′[ x ])
  ≡⟨ +ℝ-assoc (f x) (-ℝ f x) (ι δ ·ℝ f ′[ x ]) ⟩
(f x) +ℝ ((-ℝ f x) +ℝ (ι δ ·ℝ f ′[ x ]))
  ≡⟨ ap (f x +ℝ_) (+ℝ-comm (-ℝ f x) (ι δ ·ℝ f ′[ x ])) ⟩
(f x) +ℝ ((ι δ ·ℝ f ′[ x ]) +ℝ (-ℝ f x))
  ≡⟨ sym (+ℝ-assoc (f x) (ι δ ·ℝ f ′[ x ]) (-ℝ f x)) ⟩
((f x) +ℝ (ι δ ·ℝ f ′[ x ])) +ℝ (-ℝ f x)
  ≡⟨ ap (_+ℝ (-ℝ f x)) (sym (fundamental-equation f x δ)) ⟩
f (x +ℝ ι δ) +ℝ (-ℝ f x)
  ≡⟨⟩
f (x +ℝ ι δ) -ℝ f x
  ∎
```

**See examples**: Functions.agda lines 204-232, Geometry.agda lines 297-316

### Pattern 2: Chain rule application

**Use when**: Computing derivative of `f ∘ g`

**Template**:
```agda
(f ∘ g) ′[ x ]
  ≡⟨ composite-rule f g x ⟩
(f ′[ g x ]) ·ℝ (g ′[ x ])
  ≡⟨ ap₂ _·ℝ_ (f-derivative (g x)) (g-derivative x) ⟩
-- ... continue with known derivatives ...
```

**Example**: Catenary f(x) = a·cosh(x/a)
```agda
f ′[ x ]
  ≡⟨ scalar-rule a (λ y → cosh (y / a)) x ⟩
a ·ℝ ((λ y → cosh (y / a)) ′[ x ])
  ≡⟨ ap (a ·ℝ_) (composite-rule cosh (λ y → y / a) x) ⟩
a ·ℝ ((cosh ′[ x / a ]) ·ℝ ((λ y → y / a) ′[ x ]))
  ≡⟨ ap (λ z → a ·ℝ (z ·ℝ _)) (cosh-derivative (x / a)) ⟩
a ·ℝ ((sinh (x / a)) ·ℝ ((λ y → y / a) ′[ x ]))
  -- ... continue ...
```

### Pattern 3: Using constancy-principle

**Use when**: Proving uniqueness (f = g because f' = g')

**Template**:
```agda
-- Show: f ≡ g
-- Given: ∀ x → f ′[ x ] ≡ g ′[ x ]

let h = λ x → f x -ℝ g x
    h' = λ x → h ′[ x ]

    h'-is-zero : ∀ x → h ′[ x ] ≡ 0ℝ
    h'-is-zero x =
      h ′[ x ]
        ≡⟨ difference-derivative f g x ⟩
      f ′[ x ] -ℝ g ′[ x ]
        ≡⟨ ap (_-ℝ g ′[ x ]) (same-deriv x) ⟩
      g ′[ x ] -ℝ g ′[ x ]
        ≡⟨ +ℝ-invr (g ′[ x ]) ⟩
      0ℝ
        ∎

    (c , h-is-constant) = constancy-principle h h'-is-zero

    -- Now show c = 0 using initial conditions
```

## Useful Lemmas Reference

### From Base.agda
- `+ℝ-assoc`, `+ℝ-comm`, `+ℝ-idl`, `+ℝ-idr`
- `+ℝ-invl`, `+ℝ-invr`
- `·ℝ-assoc`, `·ℝ-comm`, `·ℝ-idl`, `·ℝ-idr`
- `·ℝ-distribl`, `·ℝ-distribr`
- `·ℝ-zerol`, `·ℝ-zeror`
- `nilsquare : ∀ (δ : Δ) → (ι δ ·ℝ ι δ) ≡ 0ℝ`

### From Calculus.agda
- `fundamental-equation : f (x +ℝ ι δ) ≡ f x +ℝ (ι δ ·ℝ f ′[ x ])`
- `microcancellation : (∀ δ → ι δ ·ℝ a ≡ ι δ ·ℝ b) → a ≡ b`
- `sum-rule`, `product-rule`, `quotient-rule`, `composite-rule`
- `scalar-rule`, `identity-rule`, `constant-rule`
- `power-rule : (λ x → x ^ℝ n) ′[ x ] ≡ natToℝ n ·ℝ (x ^ℝ (n ∸ 1))`
- `constancy-principle : (∀ x → f ′[ x ] ≡ 0ℝ) → Σ[ c ∈ ℝ ] (∀ x → f x ≡ c)`

### From DifferentialEquations.agda
- `exp-derivative : exp ′[ x ] ≡ exp x`
- `exp-initial : exp 0ℝ ≡ 1ℝ`
- `sin-derivative : sin ′[ x ] ≡ cos x` (after filling hole)
- `cos-derivative : cos ′[ x ] ≡ -ℝ sin x` (after filling hole)
- `cosh-derivative : cosh ′[ x ] ≡ sinh x`
- `sinh-derivative : sinh ′[ x ] ≡ cosh x`

### From Functions.agda
- `# : Nat → ℝ` - Natural number embedding
- `_^_ : ℝ → Nat → ℝ` - Natural number powers
- `_^-1 : ℝ → ℝ` - Reciprocal
- `_/_ : ℝ → ℝ → ℝ` - Division (defined as `x / y = x ·ℝ (y ^-1)`)
- `_^1/2`, `_^3/2` : ℝ → ℝ` - Fractional powers

## Debugging Tips

### Type Error: Terms don't match

**Error**: `a != b of type ℝ`

**Solution**: Add intermediate steps
```agda
-- Instead of:
≡⟨ big-leap ⟩

-- Do:
≡⟨ step1 ⟩
intermediate-term
  ≡⟨ step2 ⟩
another-intermediate
  ≡⟨ step3 ⟩
```

### Can't Apply Lemma

**Error**: `Expected type T, but got type S`

**Solution**: Check the exact type
```agda
-- In Agda REPL or via interaction:
-- C-c C-, shows goal type
-- C-c C-. shows context and goal

-- Use `ap` to apply under a constructor:
≡⟨ ap f lemma ⟩  -- If lemma : a ≡ b, gives f a ≡ f b

-- Use `ap₂` for binary functions:
≡⟨ ap₂ _+ℝ_ lemma1 lemma2 ⟩  -- Applies to both arguments
```

### Hole Won't Fill

**Error**: Goal has unsolved metavariables

**Solution**:
1. Check that all implicit arguments are resolved
2. Use `?` placeholders to see what Agda infers
3. Make arguments explicit if needed: `@x` or `{x = ...}`

### Circular Import

**Error**: Module X imports Y which imports X

**Solution**: Move shared definitions to a new module that both import.

## Testing Strategy

### Quick Test (Just Check It Compiles)
```bash
agda --library-file=./libraries src/Neural/Smooth/Physics.agda
```

### Verbose Test (See All Warnings)
```bash
agda --library-file=./libraries src/Neural/Smooth/Physics.agda 2>&1 | tee compilation.log
```

### Test Specific Module
```bash
agda --library-file=./libraries src/Neural/Smooth/Integration.agda
```

### Check for Unsolved Metas
```bash
agda --library-file=./libraries src/Neural/Smooth/Physics.agda 2>&1 | grep "Unsolved"
```

### Interactive Development
```bash
# Start Agda in JSON interaction mode
agda --interaction-json --library-file=./libraries

# Then send commands (e.g., via Emacs agda-mode or other editor integration)
```

## Progress Tracking

### Count Remaining Work
```bash
# Count holes
echo "Holes remaining:"
grep -c "{!!}" src/Neural/Smooth/*.agda

# Count derivable postulates (in Physics.agda)
echo "Physics.agda postulates:"
grep -c "^postulate" src/Neural/Smooth/Physics.agda

# Count all postulates
echo "All postulates:"
grep -c "^postulate" src/Neural/Smooth/*.agda | awk '{sum+=$1} END {print sum}'
```

### Mark Progress in This File

When you complete a task, update ROADMAP.md:
```markdown
### ✅ Milestone 1: Integration Complete
- ✅ Integration principle proven (DATE)
- ✅ All Integration.agda holes filled (DATE)
```

## Getting Help

### Documentation
1. Bell (2008), *A Primer of Infinitesimal Analysis* - Primary reference
2. 1Lab docs: https://1lab.dev/ - For library features
3. Agda docs: https://agda.readthedocs.io/ - Language reference

### Examples in Codebase
- Look at completed proofs in Calculus.agda for patterns
- See Functions.agda lines 204-232 for fundamental-equation rearrangement
- See Geometry.agda lines 297-316 for another example

### When Stuck
1. Check ROADMAP.md for strategy hints
2. Look for similar proofs in the same module
3. Check what lemmas are available (scroll through imports)
4. Try proving a simpler version first
5. Add intermediate steps to see where the issue is

## Next Session Checklist

When you start your next session:

1. ✅ Read this file
2. ✅ Verify compilation: `agda --library-file=./libraries src/Neural/Smooth/Physics.agda`
3. ✅ Check ROADMAP.md for recommended next task
4. ✅ Pick a specific hole or postulate
5. ✅ Open the file and start editing
6. ✅ Test frequently
7. ✅ Commit when done
8. ✅ Update progress in ROADMAP.md

## Summary

**You have**:
- ✅ A compiling Physics.agda
- ✅ Clear roadmap (ROADMAP.md)
- ✅ Detailed status (COMPILATION_STATUS.md)
- ✅ This quick-start guide

**You need**:
- 17-26 hours to complete all tasks
- Agda skills (you have these)
- Patience for equational reasoning

**Recommended first task**: Fill `∫-power` hole (Integration.agda:395)
- Clear pattern to follow
- Already documented
- Enables many Physics proofs

**Good luck! 🚀**
