# Size-Polymorphic Training - Status Report

**Date**: October 25, 2025
**Task**: Make topos-theoretic ARC solver work with variable-sized grids

---

## ✅ Fixed Issues

### 1. **Compatibility Matrix Indexing** (differentiable_gluing.py:127-141)
**Problem**: Overlap indices were used directly to index `section.values`, but overlap contains cell indices (e.g., [0..80]) which might not exist as positions in a smaller section's values array.

**Solution**: Use `torch.isin()` to create boolean masks, then index by position:
```python
mask_i = torch.isin(sections[i].base_indices, overlap)
mask_j = torch.isin(sections[j].base_indices, overlap)
s1_overlap = sections[i].values[mask_i]
s2_overlap = sections[j].values[mask_j]
```

### 2. **Weighted Section Average** (differentiable_gluing.py:221-246)
**Problem**: Same issue - `target_idx` (a cell index) was used to directly index `section.values[target_idx]`.

**Solution**: Find position in `base_indices` where value matches `target_idx`:
```python
mask = torch.isin(target_base, section.base_indices)
for j, target_idx in enumerate(target_base):
    if mask[j]:
        position = (section.base_indices == target_idx).nonzero(as_tuple=True)[0]
        pos = position[0].item()
        glued_values[j] += w * section.values[pos]
```

### 3. **Composition Law Edge Masking** (train_topos_arc.py:243-259)
**Problem**: Computing sheaf axiom losses using full 30×30 edge_index but with features from actual grid size (e.g., 10×10 = 100 cells).

**Solution**: Use masked edges matching actual grid dimensions:
```python
first_train_input = train_pairs[0][0]  # [h, w, 10]
h, w = first_train_input.shape[0:2]
actual_cells = h * w
masked_edge_index = self.model._get_masked_edges(h, w)
```

---

## ❌ Remaining Issues

### 4. **SheafLearner Feature Dimension Mismatch**
**Error**: `mat1 and mat2 shapes cannot be multiplied (332x128 and 16x32)`

**Location**: `cellular_sheaf_nn.py:208` in `SheafLearner.forward()`

**Root cause**: The `SheafLearner` is initialized with hardcoded dimensions that don't match the actual feature dimensions from the model:
- Model uses `feature_dim=64`, so concatenated edge features are 128-dim
- But `SheafLearner` was initialized expecting 8-dim stalks (16-dim concatenated)

**Diagnosis**:
```python
# In topos_arc_solver.py __init__:
self.sheaf_nn = CellularSheafNN(
    in_channels=stalk_dim,  # = 8
    out_channels=stalk_dim,
    ...
)

# But in extract_features:
features = self.feature_extractor(grid_flat)  # Returns [batch, cells, 64]
sheaf_features = self.sheaf_nn(features[0], masked_edges)  # Expects 8-dim!
```

**Need to investigate**: How are features flowing through the network? The dimensions don't match up.

### 5. **Output Size Prediction**
**Status**: Temporarily fixed with interpolation, but not ideal.

**Current approach**: Resize prediction to match test_output size using `F.interpolate()`.

**Limitation**: In ARC, output size is part of the puzzle. We should predict it, not force-resize.

**Better solutions**:
1. Add output size predictor network
2. Multi-resolution predictions (try multiple sizes, pick best via compatibility)
3. Learn size transformation as part of the pattern

---

## 🔍 Debug Scripts Created

1. **debug_shapes.py** - Tests forward pass with single task (3×3 grid)
   Status: ✅ Passes

2. **debug_overlap.py** - Tests overlap computation (100 vs 81 cells)
   Status: ✅ Passes

3. **debug_sections.py** - Inspects section.base_indices structure
   Status: ✅ Passes

4. **test_single_task.py** - Tests task with uniform sizes
   Status: ✅ Passes (compatibility: 0.991528)

5. **test_variable_task.py** - Tests task with variable train sizes (6×6, 10×10, 20×20)
   Status: ✅ Passes (compatibility: 0.352928)

6. **test_training_step.py** - Tests full training step with backward pass
   Status: ❌ Fails on sheaf learner dimension mismatch

---

## 📊 What Works

1. **Forward pass** - Model can process variable-sized grids
2. **Sheaf gluing** - Sections of different sizes can be glued correctly
3. **Compatibility scoring** - Works across size boundaries
4. **Edge masking** - Correctly filters graph topology for actual grid size

---

## 📋 Next Steps (Priority Order)

### Immediate (30 min)
1. **Fix SheafLearner dimensions**: Trace feature flow from `feature_extractor` → `sheaf_nn` → `sheaf_learner`
2. **Verify stalk dimensionality**: Should `in_channels` = `feature_dim` or `stalk_dim`?
3. **Test full training loop**: Run one epoch, ensure no errors

### Short-term (2 hours)
4. **Monitor training**: Check if losses decrease over epochs
5. **Hyperparameter tuning**: Adjust loss weights, learning rate
6. **Validation metrics**: Track compatibility scores, coverage

### Medium-term (1 day)
7. **Output size prediction**: Add explicit size prediction network
8. **Batch training**: Ensure multiple tasks can be batched together
9. **Scale to full dataset**: Test on all 400 tasks

---

## 💡 Key Insights

### What We Learned

1. **Single model with masking > Dynamic models**
   - Maintains weight sharing across grid sizes
   - Can batch arbitrary tasks together
   - Simpler implementation

2. **Index vs Position**
   - `base_indices` contains CELL INDICES (which cells this section covers)
   - `values[k]` corresponds to `base_indices[k]`, not to cell k directly
   - Must map cell indices → positions in values array

3. **Edge masking is critical**
   - Can't use full 30×30 edge_index with partial features
   - Must filter edges to match actual grid topology
   - Applies to BOTH forward pass and loss computation

### Architecture Strengths

✅ Solid mathematical foundation (real sheaf theory)
✅ Differentiable end-to-end
✅ Category theory enforced via losses
✅ Handles variable sizes (after fixes)

### Remaining Weaknesses

⚠️ Feature dimension confusion between components
⚠️ Output size prediction is hacky
⚠️ Not tested at scale (only 20 tasks so far)

---

## 📝 Files Modified

| File | Changes | Status |
|------|---------|--------|
| `differentiable_gluing.py` | Fixed compatibility matrix and weighted average for size polymorphism | ✅ Working |
| `train_topos_arc.py` | Fixed edge masking in composition law computation | ✅ Working |
| `topos_arc_solver.py` | Added `_get_masked_edges()`, updated `extract_features()` | ✅ Working |
| `cellular_sheaf_nn.py` | No changes yet | ⚠️ Needs dimension fix |

---

## 🎯 Current Blocker

**Feature dimension mismatch in SheafLearner**

Need to trace:
1. What dimensions does `feature_extractor` output?
2. What dimensions does `sheaf_nn` expect as input?
3. What dimensions does `sheaf_learner` expect for edge features?

Once we fix this final dimension issue, training should work!
