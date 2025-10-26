# Training Loss Analysis - Why It's Not Falling

**Date**: October 23, 2025
**Issue**: Training loss stuck at 9.5021 with no improvement over 150 epochs

---

## Problem Diagnosis

### Issue 1: No Gradients Through Early Recursion Cycles ✅ FIXED

**Original TRM implementation**:
```python
# T-1 cycles WITHOUT gradients
for t in range(self.num_cycles - 1):
    with torch.no_grad():  # ← Blocks gradients!
        z = self.refine_z(...)
        y = self.refine_y(...)

# Only last cycle has gradients
z = self.refine_z(...)
y = self.refine_y(...)
```

**Problem**: Encoder and refiners never receive gradients because early cycles use `torch.no_grad()`.

**Fix Applied**: Remove `torch.no_grad()` wrapper, allow gradients through all cycles:
```python
for t in range(self.num_cycles):
    z = self.refine_z(...)  # ← Gradients flow
    y = self.refine_y(...)  # ← Gradients flow
```

**Result**: Encoder, refiners, and formula selector now receive gradients ✅

---

### Issue 2: No Gradients to Neural Predicates ⚠️ FUNDAMENTAL ISSUE

**Cell-by-cell evaluation breaks gradient graph**:
```python
def _apply_formula(self, grid, formula):
    for i in range(H_out):
        for j in range(W_out):  # ← Python loop!
            truth = self.interpreter.force(formula, context)
            # Predicates called here, but gradients don't accumulate
```

**Why this breaks gradients**:
1. Python for-loops are not differentiable operations
2. Each cell evaluation is independent (no tensor operations)
3. PyTorch computational graph fragmented across loop iterations
4. Predicates get called but gradients don't flow back properly

**Evidence**:
```
Gradient flow check:
  Encoder layer1: True ✅
  Refine_z: True ✅
  Refine_y: True ✅
  Formula selector: True ✅
  Neural predicate (color_eq): False ❌  ← NO GRADIENTS!
```

**Root cause**: Our architecture mixes:
- **Tensor operations** (encoder, refiners, selector) → gradients flow
- **Python loops** (formula evaluation) → gradients break
- **Predicate calls** (inside loops) → no gradient accumulation

---

## Why TRM Recursion Doesn't Help

The TRM paper achieves 45% on ARC-AGI-1 by:
1. Using **end-to-end differentiable** networks
2. Operating on **grid-level tensors** throughout
3. No symbolic formulas or cell-by-cell evaluation

**Our hybrid approach**:
- ✅ Interpretable formulas (symbolic)
- ✅ Differentiable logic operators (smooth)
- ❌ Cell-by-cell evaluation (breaks gradients)
- ❌ Predicates don't learn (no gradients)

**Conclusion**: Adding recursion to a non-differentiable evaluation loop doesn't improve learning.

---

## Solutions

### Option 1: Vectorize Formula Evaluation (Ideal, but hard)

**Goal**: Evaluate formula on all cells at once using tensor operations.

**Requirements**:
- Rewrite `_apply_formula` to operate on full grids
- Vectorize all predicates (batch operations)
- Rewrite Kripke-Joyal interpreter for batched evaluation
- Handle variable grid sizes with padding/masking

**Pros**:
- Full gradient flow to all components
- Much faster (GPU parallelism)
- TRM recursion would then help

**Cons**:
- Major refactoring (~2000+ lines)
- Complex tensor reshaping
- Need to handle dynamic formulas (quantifiers)

**Estimated effort**: 2-3 weeks

---

### Option 2: Simplify to Baseline + Improvements (Pragmatic)

**Observation**: Baseline already achieves 2.07-2.74% binary accuracy without recursion.

**Improvements to baseline**:

1. **Increase learning rate**:
   ```python
   lr=5e-3  # vs current 1e-3
   ```

2. **Add learning rate warmup**:
   ```python
   # Epochs 1-10: lr gradually 1e-5 → 5e-3
   # Epochs 11+: lr = 5e-3 with decay
   ```

3. **Increase model capacity**:
   ```python
   feature_dim=128  # vs current 64
   hidden_dim=256   # for formula selector
   ```

4. **Better template diversity**:
   - Add more relational templates (touching, adjacent, same_row, same_col)
   - Add pattern completion templates (fill_gaps, extend_pattern)
   - Add counting templates (count_color, majority_color)

5. **Curriculum learning**:
   - Train longer on small grids (scale 0: 100 epochs)
   - Fine-tune on larger grids (scale 1-2: 50 epochs each)
   - Gradually increase temperature annealing

6. **Data augmentation**:
   - Rotate grids (90°, 180°, 270°)
   - Flip grids (horizontal, vertical)
   - Recolor (permute color indices)

**Expected improvement**: 2-3% → 5-10% binary accuracy

**Estimated effort**: 1-2 days

---

### Option 3: Hybrid Approach (Future work)

**Phase 1**: Fix baseline with improvements (Option 2)
**Phase 2**: Partially vectorize critical predicates
**Phase 3**: Add recursive refinement once gradients flow

---

## Recommendation

**Go with Option 2** (Improve baseline) because:

1. **Faster results**: 1-2 days vs 2-3 weeks
2. **Proven approach**: Baseline already learning (2-3%)
3. **Incremental**: Can add vectorization later
4. **Practical**: Focus on what works

**Next steps**:
1. Create improved config with higher LR, warmup, larger model
2. Add more diverse templates
3. Implement curriculum learning
4. Run training and measure improvement

Target: **5-10% binary accuracy** (2-4x improvement over baseline)

---

## Code Changes Made

### trm_neural_symbolic.py (Line 565-576)

**Before** (blocked gradients):
```python
for t in range(self.num_cycles - 1):
    with torch.no_grad():  # ← No learning!
        z = self.refine_z(...)
        y = self.refine_y(...)

z = self.refine_z(...)  # Only this learns
y = self.refine_y(...)
```

**After** (gradients flow):
```python
for t in range(self.num_cycles):
    z = self.refine_z(...)  # All cycles learn
    y = self.refine_y(...)
```

### formula_templates.py (Line 270-294)

**Before** (missing eval method):
```python
class SequentialFormula:  # ← Not a Formula!
    steps: List[Formula]
```

**After** (proper inheritance):
```python
@dataclass
class SequentialFormula(Formula):  # ← Inherits eval()
    steps: List[Formula]

    def eval(self, context, interpreter):
        return self.steps[-1].eval(context, interpreter)
```

---

## Lessons Learned

1. **Gradient flow is critical**: Always verify gradients reach all trainable parameters
2. **Python loops break gradients**: Use tensor operations for differentiability
3. **Hybrid architectures are tricky**: Symbolic + neural requires careful design
4. **TRM requires end-to-end differentiability**: Recursion only helps if gradients flow
5. **Baseline first, optimize later**: Get something working before adding complexity

---

**Status**: TRM training paused. Focus on improving baseline neural-symbolic solver.
