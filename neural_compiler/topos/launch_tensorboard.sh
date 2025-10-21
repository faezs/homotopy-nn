#!/bin/bash
# Launch TensorBoard to monitor ARC geometric morphism training

RUNS_DIR="/Users/faezs/homotopy-nn/neural_compiler/topos/runs"
TENSORBOARD="/nix/store/lnqf82hc6ljyb26s0h2jx6kw953v6a7z-python3-3.12.11-env/bin/tensorboard"

echo "========================================"
echo "TensorBoard - Geometric Morphism Training"
echo "========================================"
echo ""
echo "Logs directory: $RUNS_DIR"
echo ""
echo "Starting TensorBoard..."
echo "Open browser to: http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop"
echo ""

$TENSORBOARD --logdir="$RUNS_DIR" --port=6006
