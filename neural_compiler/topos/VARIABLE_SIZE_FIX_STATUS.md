# Variable-Size Grid Fix - Current Status

**Date**: October 25, 2025

---

## ✅ What's Fixed

1. **Padding in extract_features**: Model now pads grids internally and masks edges correctly
2. **Variable input/output in extract_section**: Handles different-sized input/output grids within a task
3. **Variable test size in forward**: Outputs correct size based on test input dimensions

## ❌ What's Still Broken

### Issue 1: Soft Gluing with Variable-Sized Sections

**Problem**: `soft_glue_sheaf_sections()` assumes all sections have the same base space.

**Error**: `index 663 is out of bounds for dimension 0 with size 100`

**Root cause**:
- Task has multiple train examples: (10×10 input, ...), (5×5 input, ...), (3×3 input, ...)
- Sections have base_indices of sizes: 100, 25, 9
- Target base (test input 10×10) has size 100
- Gluing tries to access index 663 from a section with only 100 elements

**Current gluing assumes**:
```python
def soft_glue_sheaf_sections(sections, target_base, temperature):
    # Assumes all sections.base_indices are same size and aligned
    for i in target_base:
        for section in sections:
            section.values[i]  # ❌ FAILS if section has < i elements
```

**Need**: Interpolate/resample sections to target base size

---

### Issue 2: Output Size Prediction

**Problem**: Model predicts output with same size as test INPUT, but actual test OUTPUT might be different.

**Error**: `The size of tensor a (7) must match the size of tensor b (3) at non-singleton dimension 2`

**Example**:
- Test input: 3×7
- Test output: 3×3
- Model prediction: 3×7 (matches input, wrong!)

**Root cause**: `forward()` returns `[1, test_h, test_w, 10]` based on test_input size, not test_output size.

**But we don't know test_output size during inference!**

This is a fundamental ARC challenge: Output size can be different from input size, and we need to PREDICT it.

---

## Solutions Needed

### Solution 1: Resample Sections for Gluing

Options:

**A) Interpolation**: Resample each section's values to match target_base size
```python
def resample_section(section, target_size):
    # If section has 9 values, target has 100:
    # Use nearest neighbor or linear interpolation
    indices = torch.linspace(0, len(section.values)-1, target_size)
    resampled = interp1d(section.values, indices)
    return SheafSection(target_base, resampled)
```

**B) Padding**: Pad smaller sections with zeros
```python
def pad_section(section, target_size):
    padded_values = torch.zeros(target_size, section.values.shape[-1])
    padded_values[:len(section.values)] = section.values
    return SheafSection(target_base, padded_values)
```

**C) Subset matching** (BEST): Only glue overlapping regions
```python
def soft_glue_with_subset(sections, target_base, temperature):
    glued_values = torch.zeros(len(target_base), value_dim)

    for i in target_base:
        compatible_values = []
        for section in sections:
            if i < len(section.base_indices):
                compatible_values.append(section.values[i])

        if compatible_values:
            glued_values[i] = weighted_average(compatible_values)
        # else: keep zero (no data for this cell)
```

---

### Solution 2: Predict Output Size

Options:

**A) Fixed size assumption**: Always predict same size as input (current)
- ❌ Wrong for size-changing tasks

**B) Learn output size**: Add a size predictor network
```python
class OutputSizePredictor(nn.Module):
    def forward(self, input_features, train_examples):
        # From train pairs, learn h_out / h_in ratio
        # Predict output dimensions
        return (pred_h, pred_w)
```

**C) Multi-resolution predictions**: Predict at multiple scales, pick best
- Generate 1×1, input_size, 2×input_size outputs
- Use gluing compatibility to pick correct scale

**D) Simplest**: For now, use ground truth test_output size during training
```python
# In train_step:
test_output_h, test_output_w = test_output.shape[0:2]
# Pass to model somehow, or reshape prediction
```

---

## Next Steps

1. **Fix gluing** with subset matching (Solution 1C)
2. **Handle output size** with size predictor (Solution 2B) OR accept that we predict input-size outputs for now
3. **Test** on ARC tasks with variable sizes
4. **Monitor** if losses decrease

---

**Current Training Status**: All 14 tasks failing, losses at 0.0

**Next Action**: Implement subset-based soft gluing
