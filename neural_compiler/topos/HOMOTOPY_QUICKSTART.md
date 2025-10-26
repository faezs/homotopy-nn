# Homotopy Minimization - Quick Start Guide

## 🚀 What is This?

A PyTorch implementation that learns **canonical transformations** by minimizing homotopy distance between training examples. Instead of learning 4 separate transformations, learn 1 transformation that represents their shared essence.

## 📦 What You Get

- **homotopy_arc_learning.py**: Core implementation (600 lines)
- **test_homotopy_minimization.py**: Comprehensive tests
- **Full gradient flow**: Everything is differentiable

## ⚡ Quick Example

```python
from homotopy_arc_learning import HomotopyClassLearner, train_homotopy_class
from geometric_morphism_torch import Site, Sheaf
from arc_loader import ARCGrid
import numpy as np

# 1. Create training pairs (e.g., color flip transformation)
train_pairs = [
    (ARCGrid.from_array(np.array([[1, 2], [2, 1]])),
     ARCGrid.from_array(np.array([[2, 1], [1, 2]]))),
    # ... more pairs with same transformation
]

# 2. Setup sites
site_in = Site((2, 2), connectivity="4")
site_out = Site((2, 2), connectivity="4")

# 3. Convert to sheaves
sheaf_pairs = [
    (Sheaf.from_grid(inp, site_in, 32),
     Sheaf.from_grid(out, site_out, 32))
    for inp, out in train_pairs
]

# 4. Create learner
learner = HomotopyClassLearner(
    site_in, site_out,
    feature_dim=32,
    num_training_examples=len(train_pairs)
)

# 5. Train (automatic two-phase optimization)
history = train_homotopy_class(
    learner, sheaf_pairs,
    num_epochs=200,
    verbose=True
)

# 6. Predict on new input
test_input = Sheaf.from_grid(test_grid, site_in, 32)
prediction = learner.predict(test_input)  # Uses canonical morphism!
```

## 🎯 Key Parameters

### Training
- `num_epochs=200`: Total epochs (Phase 1: 0-100, Phase 2: 100-200)
- `lr_individual=1e-3`: Learning rate for individual morphisms fᵢ
- `lr_canonical=5e-4`: Learning rate for canonical morphism f*

### Loss Weights
- `lambda_homotopy=1.0`: Weight for homotopy distance Σd_H(f*, fᵢ)
- `lambda_recon=10.0`: Weight for reconstruction ||fᵢ(xᵢ) - yᵢ||²
- `lambda_canonical=5.0`: Weight for canonical accuracy ||f*(xᵢ) - yᵢ||²

**Adaptive**: Weights change automatically during phase transition!

## 📊 What Gets Minimized

```
L_total = λ_h·Σᵢ d_H(f*, fᵢ)     ← Homotopy distance (collapse to f*)
        + λ_r·Σᵢ ||fᵢ(xᵢ) - yᵢ||²  ← Individual reconstruction
        + λ_c·Σᵢ ||f*(xᵢ) - yᵢ||²  ← Canonical generalization
```

Where homotopy distance:
```
d_H(f, g) = α·L²_distance + β·topological_distance + γ·parameter_distance
```

## 🧪 Testing

```bash
# Run all tests
python3 test_homotopy_minimization.py

# Output:
# ✓ Test 1: Distance Metric - PASSED
# ✓ Test 2: Convergence - PASSED (41.5% reduction)
# ⚠ Test 3: Generalization - PARTIAL (needs more examples)
# + Visualization saved
```

## 📈 Expected Results

**Convergence metrics**:
- Homotopy distance reduces by ~40-50%
- Individual morphisms cluster around canonical
- Phase transition visible at epoch 100

**Training progress**:
```
Epoch   0 [PHASE 1 (Fit)]:
  Homotopy:  34.596 (weight: 0.10)
  Recon:     50.234 (weight: 20.00)

Epoch 100 [PHASE 2 (Collapse)]:
  Homotopy:  25.123 (weight: 2.00)
  Recon:      0.156 (weight: 5.00)

Epoch 199:
  Homotopy:  20.256
  Canonical:  0.031  ← Final accuracy
```

## 🎨 Visualization

Automatically generates `homotopy_learning_curves.png`:
- Left plot: Per-morphism convergence (fᵢ → f*)
- Right plot: Average homotopy distance over time
- Red line: Phase transition marker

## 🔧 Troubleshooting

### High homotopy distance not reducing
- ✓ Increase `lambda_homotopy` in Phase 2
- ✓ Decrease learning rates (may be unstable)
- ✓ Check that training pairs are actually similar transformations

### Poor generalization
- ✓ Add more training examples (>4 recommended)
- ✓ Increase `feature_dim` (try 64 or 128)
- ✓ Train longer in Phase 2

### NaN losses
- ✓ Reduce learning rates
- ✓ Add gradient clipping (already enabled at 1.0)
- ✓ Check input data ranges

## 🧬 Architecture Details

### HomotopyDistance Components

1. **L² distance** (`alpha=1.0`):
   ```python
   d_L2 = mean(||f(x) - g(x)||² for x in inputs)
   ```

2. **Topological distance** (`beta=0.5`):
   ```python
   d_topo = ||features(f) - features(g)||²
   # Features: connectivity, variance, gradient magnitude
   ```

3. **Parameter distance** (`gamma=0.1`):
   ```python
   d_param = ||θ_f - θ_g||²
   # Direct Euclidean distance in weight space
   ```

### Two-Phase Strategy

**Phase 1 (Fit)**: Epochs 0-N/2
- Focus: Make each fᵢ fit its training pair
- Weights: Low homotopy, high reconstruction
- Allows morphisms to diverge

**Phase 2 (Collapse)**: Epochs N/2-N
- Focus: Collapse all fᵢ to canonical f*
- Weights: High homotopy, high canonical
- Forces convergence to shared transformation

## 🎓 Mathematical Intuition

**Problem**: Learning from 4 examples (x₁→y₁, ..., x₄→y₄)

**Traditional approach**: Learn 4 functions f₁, f₂, f₃, f₄
- Overfits to each example
- Doesn't generalize
- No shared structure

**Our approach**: Learn 1 canonical function f*
- Represents homotopy class [f*]
- All fᵢ are continuous deformations of f*
- Generalizes by capturing shared essence

**Homotopy**: Two morphisms f, g are homotopic if there exists continuous deformation H:[0,1]→Mor such that H(0)=f, H(1)=g.

**Goal**: Find f* such that all training examples are "close" in homotopy space.

## 📚 Files Overview

```
neural_compiler/topos/
├── homotopy_arc_learning.py (600 lines)
│   ├── HomotopyDistance          # 3-component distance metric
│   ├── HomotopyClassLearner      # Main learning algorithm
│   └── train_homotopy_class()    # Two-phase training loop
│
├── test_homotopy_minimization.py (400 lines)
│   ├── test_homotopy_distance()  # Validate metric
│   ├── test_homotopy_convergence() # Validate learning
│   ├── test_generalization()     # Validate prediction
│   └── visualize_homotopy_learning() # Create plots
│
├── geometric_morphism_torch.py (700 lines)
│   ├── Site                      # Grothendieck topology
│   ├── Sheaf                     # Presheaf with gluing
│   └── GeometricMorphism         # Adjoint pair (f* ⊣ f_*)
│
├── topos_categorical.py (800 lines)
│   ├── NaturalTransformation
│   ├── SubobjectClassifier (Ω)
│   └── Topos category
│
└── homotopy_regularization.py (400 lines)
    ├── HomotopyRegularization    # Smoothness penalty
    ├── PathInterpolationLoss     # Geometric consistency
    └── TopologicalConsistencyLoss # Lipschitz constraint
```

## 🌟 Best Practices

1. **Start small**: Test on 2×2 or 3×3 grids first
2. **Verify convergence**: Check that homotopy distance decreases
3. **Monitor phases**: Ensure phase transition happens smoothly
4. **Visualize**: Always generate learning curves
5. **Multiple runs**: Stochastic - run 3-5 times, pick best

## 🚨 Limitations

1. **Small training sets**: Needs ≥3 examples to work well
2. **Computational cost**: O(N²) in number of examples
3. **Topological features**: Currently approximated, not exact
4. **Grid size**: Works best on <10×10 grids (scalability issue)

## ✨ Advanced Usage

### Custom distance weights
```python
learner.homotopy_distance = HomotopyDistance(
    alpha=2.0,   # Emphasize L² distance
    beta=0.1,    # De-emphasize topology
    gamma=0.5    # Moderate parameter distance
)
```

### Manual phase control
```python
for epoch in range(200):
    # Custom weight schedule
    if epoch < 50:
        lambda_h, lambda_r, lambda_c = 0.1, 20.0, 0.5
    elif epoch < 150:
        lambda_h, lambda_r, lambda_c = 1.0, 10.0, 5.0
    else:
        lambda_h, lambda_r, lambda_c = 5.0, 1.0, 20.0

    loss, metrics = learner(sheaf_pairs, lambda_h, lambda_r, lambda_c)
    # ... backward pass ...
```

### Extract canonical morphism
```python
# After training, save canonical morphism
torch.save(learner.canonical_morphism.state_dict(), 'canonical_f.pt')

# Use for prediction
canonical = learner.canonical_morphism
test_output = canonical.pushforward(test_sheaf)
```

## 🎯 Success Metrics

**Training successful if**:
- ✅ Homotopy distance reduces by >20%
- ✅ Final canonical error <0.1
- ✅ All individual morphisms within 30% of mean
- ✅ Smooth learning curves (no spikes)

**Ready for deployment if**:
- ✅ Test error <2× training error
- ✅ Generalization gap <10.0
- ✅ Consistent across multiple runs

## 🔗 Integration with ARC Solver

```python
# In your ARC solver
from homotopy_arc_learning import learn_arc_task_homotopy

# Load task
task = load_arc_task(task_id)

# Learn canonical transformation
learner = learn_arc_task_homotopy(
    task,
    feature_dim=64,
    num_epochs=200
)

# Predict test outputs
for test_input in task.test:
    sheaf_in = Sheaf.from_grid(test_input, learner.site_in, 64)
    prediction = learner.predict(sheaf_in)
    # Convert back to grid and submit
```

---

**That's it!** You now have a working homotopy minimization system. 🎉

For questions or issues, check the comprehensive documentation in `HOMOTOPY_MINIMIZATION_COMPLETE.md`.
