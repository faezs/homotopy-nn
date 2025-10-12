#!/bin/bash
# Quick launcher for ARC-AGI meta-learning training

set -e

echo "======================================================================"
echo "ARC-AGI Meta-Learning Training"
echo "======================================================================"
echo ""

# Check if ARC data exists
ARC_DATA="../../ARC-AGI/data"
if [ ! -d "$ARC_DATA" ]; then
    echo "⚠ ARC-AGI data not found at $ARC_DATA"
    echo ""
    echo "Cloning ARC-AGI repository..."
    cd ../..
    git clone https://github.com/fchollet/ARC-AGI.git
    cd neural_compiler/topos
    echo "✓ ARC-AGI data downloaded"
    echo ""
fi

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import jax, flax, optax, onnx" 2>/dev/null || {
    echo "⚠ Missing dependencies. Installing..."
    pip install jax jaxlib flax optax onnx onnxruntime numpy
}
echo "✓ Dependencies OK"
echo ""

# Parse arguments
EPOCHS=${1:-50}      # Default: 50 epochs (quick training)
TASKS=${2:-100}      # Default: 100 tasks (subset for speed)
BATCH=${3:-8}        # Default: 8 tasks per batch

echo "Training configuration:"
echo "  Epochs: $EPOCHS"
echo "  Tasks: $TASKS"
echo "  Batch size: $BATCH"
echo ""

# Run training
python3 train_meta_learner.py \
    --data "$ARC_DATA" \
    --output "trained_models" \
    --epochs "$EPOCHS" \
    --tasks "$TASKS" \
    --batch-size "$BATCH" \
    --shots 3 \
    --seed 42

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ TRAINING COMPLETE!"
    echo "======================================================================"
    echo ""
    echo "Trained model saved to: trained_models/"
    echo ""
    echo "Files:"
    echo "  - universal_topos.pkl       (trained meta-learner)"
    echo "  - onnx_export/              (ONNX deployment files)"
    echo "  - config.json               (training configuration)"
    echo "  - training_results.json     (training statistics)"
    echo ""
    echo "To use the trained model:"
    echo "  python3"
    echo "  >>> from meta_learner import MetaToposLearner"
    echo "  >>> meta = MetaToposLearner.load('trained_models/universal_topos.pkl')"
    echo "  >>> # meta.few_shot_adapt(task, n_shots=3)"
    echo ""
else
    echo ""
    echo "✗ Training failed. Check errors above."
    exit 1
fi
