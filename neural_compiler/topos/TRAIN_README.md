# Training Meta-Learner on ARC-AGI

## Quick Start

```bash
cd neural_compiler/topos

# Quick training (50 epochs, 100 tasks, ~30 min)
./train_arc.sh

# Or with custom parameters
./train_arc.sh 100 200 16  # epochs tasks batch_size

# Or directly with Python
python train_meta_learner.py --epochs 50 --tasks 100
```

## What This Does

1. **Downloads ARC-AGI data** (if not present)
2. **Loads training tasks** from ARC-AGI dataset
3. **Trains universal topos** via meta-learning
4. **Exports to ONNX** for deployment
5. **Tests few-shot adaptation** on held-out tasks

## Training Parameters

```bash
python train_meta_learner.py \
    --data ../../ARC-AGI/data \    # ARC data path
    --output trained_models \      # Where to save
    --epochs 100 \                 # Meta-training epochs
    --tasks 200 \                  # Number of tasks to use
    --batch-size 8 \               # Tasks per meta-batch
    --shots 3 \                    # Few-shot examples
    --seed 42                      # Random seed
```

## Expected Output

```
======================================================================
TRAINING META-LEARNER ON ARC-AGI
======================================================================

✓ Loaded 200 training tasks

Dataset Statistics:
  Total tasks: 200
  Total examples: 800
  Avg examples/task: 4.0
  Avg grid size: 12.5 × 12.5

Initializing meta-learner...
✓ Meta-learner initialized
  Objects: 20
  Feature dim: 32
  Embedding dim: 64

Starting meta-training...
  Epochs: 100
  Batch size: 8
  N-shot: 3
  Tasks: 200

Epoch   0/100: Meta-loss = 0.4523
Epoch  10/100: Meta-loss = 0.3214
Epoch  20/100: Meta-loss = 0.2567
...
Epoch 100/100: Meta-loss = 0.1234

✓ Meta-training completed!
✓ Saved trained model to trained_models/universal_topos.pkl

Exporting to ONNX...
✓ ONNX export successful!

======================================================================
TRAINING COMPLETE
======================================================================
Trained on: 200 ARC tasks
Model saved: trained_models/universal_topos.pkl
ONNX export: trained_models/onnx_export/
  - task_encoder.onnx ✓
  - sheaf_network.onnx ✓
  - universal_topos.pkl ✓
======================================================================
```

## Output Files

After training, you'll have:

```
trained_models/
├── universal_topos.pkl           # Trained meta-learner
├── config.json                   # Training configuration
├── training_results.json         # Statistics
└── onnx_export/
    ├── task_encoder.onnx         # Task embedding network
    ├── sheaf_network.onnx        # Sheaf prediction network
    ├── universal_topos.pkl       # Full model
    └── metadata.json             # Export info
```

## Using the Trained Model

```python
from meta_learner import MetaToposLearner
from arc_loader import load_arc_dataset

# Load trained meta-learner
meta = MetaToposLearner.load('trained_models/universal_topos.pkl')

# Load a new task
tasks = load_arc_dataset('../../ARC-AGI/data', 'evaluation')
new_task = tasks['task_id_123']

# Few-shot adapt
import jax.random as random
key = random.PRNGKey(0)
adapted_site = meta.few_shot_adapt(new_task, n_shots=3, key=key)

# Now use adapted_site for predictions
# (Full inference pipeline coming soon)
```

## Training Times

Approximate training times on different hardware:

| Hardware | 100 tasks, 50 epochs | 200 tasks, 100 epochs | 400 tasks, 200 epochs |
|----------|----------------------|------------------------|------------------------|
| CPU      | ~30 min              | ~2 hours               | ~8 hours               |
| GPU (T4) | ~5 min               | ~20 min                | ~1.5 hours             |
| TPU v3   | ~2 min               | ~8 min                 | ~30 min                |

## Troubleshooting

### ARC Data Not Found
```bash
cd ../..
git clone https://github.com/fchollet/ARC-AGI.git
cd neural_compiler/topos
```

### Missing Dependencies
```bash
pip install jax jaxlib flax optax onnx onnxruntime numpy
```

### Out of Memory
Reduce batch size:
```bash
python train_meta_learner.py --batch-size 4 --tasks 50
```

### Training Takes Too Long
Quick test run:
```bash
python train_meta_learner.py --epochs 10 --tasks 20
```

## Next Steps

After training:

1. **Evaluate on test set**:
   ```bash
   python evaluate_meta_learner.py --model trained_models/universal_topos.pkl
   ```

2. **Deploy with ONNX**:
   ```python
   import onnxruntime as ort
   session = ort.InferenceSession('trained_models/onnx_export/task_encoder.onnx')
   ```

3. **Continue training**:
   ```bash
   python train_meta_learner.py --resume trained_models/universal_topos.pkl
   ```

## Theory

This implements **Phase 2: Meta-Learning** from ARC-AGI-2-STRATEGY.md:

- **Phase 1**: Task-specific evolution (done)
- **Phase 2**: Meta-learn universal topos ← **YOU ARE HERE**
- **Phase 3**: Deploy and evaluate

The universal topos learns common abstract patterns across all ARC tasks,
enabling few-shot generalization to new tasks!

See `src/Neural/Topos/MetaLearning.agda` for formal specification.
