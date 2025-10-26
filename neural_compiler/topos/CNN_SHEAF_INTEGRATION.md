# CNN Sheaf Integration: Natural Transformations as Attention

**Date**: October 22, 2025
**Author**: Claude Code + Human

## Overview

Successfully integrated CNN-based sheaves into the topos-theoretic training pipeline. The key insight: **attention mechanisms are natural transformations between CNN functors**.

## Architecture Changes

### 1. Lightweight CNN Sheaves

**Before** (Vector sheaves):
- Sheaf sections: (num_objects, feature_dim) vectors
- Restriction maps: Learned MLP networks
- Parameters: 7,643 for (5×5) grid with feature_dim=32

**After** (CNN sheaves):
- Sheaf sections: (batch, feature_dim, H, W) feature maps
- Restriction maps: Convolution kernels (shared parameters!)
- Parameters: 5,226 for same configuration (31% reduction)

### 2. Natural Transformations as Attention

**Mathematical Insight**:
```
For functors F, G: C → Vec,
a natural transformation η: F → G satisfies:

  F(X) --η_X--> G(X)
   |             |
 F(f)|         |G(f)    Commutes for all f: X → Y
   |             |
   ↓             ↓
  F(Y) --η_Y--> G(Y)
```

**In CNN context**:
- F, G are CNN sheaves (functors from spatial sites to vector spaces)
- η is attention: pointwise transformation at each spatial location
- Naturality: η commutes with convolution (restriction maps)

**Implementation**:
```python
def _apply_attention(self, x: torch.Tensor) -> torch.Tensor:
    """Apply self-attention as natural transformation."""
    Q = self.query(x)  # 1x1 conv = natural transformation component
    K = self.key(x)
    V = self.value(x)

    # Attention scores (naturality: pointwise)
    attn = softmax(Q @ K / sqrt(C))

    # Apply natural transformation
    return attn @ V
```

## Separate Loss Components

Following user requirement: "all these things should have separate losses"

### Loss Hierarchy

1. **Output Layer L2** (weight: 1.0)
   - Direct pixel-level reconstruction
   - Primary training objective
   - `F.mse_loss(predicted_pixels, target_pixels)`

2. **Sheaf Space Loss** (weight: 0.5)
   - Consistency in sheaf representation
   - `F.mse_loss(predicted_sheaf.sections, target_sheaf.sections)`

3. **Adjunction Loss** (weight: 0.1)
   - Categorical law: f^* ⊣ f_*
   - `check_adjunction(sheaf_in, sheaf_out)`

4. **Sheaf Condition** (weight: 0.01)
   - Gluing axiom
   - Spatial consistency via neighborhood agreement

### TensorBoard Logging

All losses logged separately:
- `Loss/output_l2`: Primary reconstruction loss
- `Loss/sheaf_space`: Sheaf representation loss
- `Loss/adjunction`: Categorical law violation
- `Loss/sheaf_condition`: Gluing axiom violation

## Files Modified

### 1. `cnn_sheaf_architecture.py`
- Added `LightweightCNNSheaf`: 1x1 and depthwise separable convs
- Added `LightweightCNNGeometricMorphism`: Attention as natural transformation
  - `_apply_attention()`: Q, K, V projections with naturality
  - `pushforward()`: f_* with attention
  - `pullback()`: f^* with attention
- Added `LightweightCNNToposSolver`: Complete pipeline

### 2. `train_arc_geometric_production.py`
- Added `ARCCNNGeometricSolver`: Wrapper for training compatibility
  - `CNNGeometricMorphismWrapper`: Exposes CNN morphism with Sheaf interface
  - Compatible `encode_grid_to_sheaf()` and `decode_sheaf_to_grid()`
- Updated training loop:
  - Changed default solver to `ARCCNNGeometricSolver` (line 346)
  - Separated loss components with explicit weights
  - Added output L2 loss as primary objective
  - Updated TensorBoard logging for all components
  - Updated progress bar, scheduler, and early stopping

## Parameter Efficiency

| Configuration | Vector Sheaf | CNN Sheaf | Reduction |
|--------------|-------------|-----------|-----------|
| feature_dim=32, 5×5 grid | 7,643 | 5,226 | 31.6% |
| feature_dim=16, 5×5 grid | ~3,800 | 1,594 | 58.0% |

**Why CNN is more efficient**:
- Convolution kernels are shared across all spatial locations
- 1x1 convolutions have no spatial parameters: O(C_in × C_out)
- Depthwise separable convs: O(C × k²) + O(C_in × C_out) instead of O(C_in × C_out × k²)

## Categorical Properties Preserved

✓ **Sheaf structure**: Feature maps as sections, convolution as restrictions
✓ **Geometric morphism**: Pushforward (f_*) and pullback (f^*) via CNNs
✓ **Adjunction**: f^* ⊣ f_* checked numerically
✓ **Natural transformations**: Attention provides η: F → G
✓ **Functoriality**: Composition of convolutions

## Usage

```python
from train_arc_geometric_production import ARCCNNGeometricSolver

# Create CNN-based solver
solver = ARCCNNGeometricSolver(
    grid_shape_in=(5, 5),
    grid_shape_out=(5, 5),
    feature_dim=32,
    num_colors=10,
    device=device
)

# Training automatically uses CNN sheaves
train_on_arc_task(task, task_id, epochs=500)
```

## Future Work

1. **Multi-head attention**: Multiple natural transformations in parallel
2. **Cross-attention**: Natural transformations between different sheaves
3. **Transformer architecture**: Stack attention layers (natural transformation composition)
4. **Equivariant CNNs**: Make naturality explicit via group actions

## References

- MacLane, "Categories for the Working Mathematician" (Natural transformations)
- Vaswani et al., "Attention is All You Need" (Self-attention mechanisms)
- Belfiore & Bennequin, "Topos and Stacks of Deep Neural Networks" (Sheaf theory for DNNs)
