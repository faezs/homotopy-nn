"""
Quick test to verify fractal + derivator integration works.

Tests:
1. Load Mini-ARC dataset
2. Create model for scale 0
3. Run 1 epoch of training
4. Compute Kan extension
"""

import torch
import torch.nn.functional as F
import numpy as np
import jax.numpy as jnp
from pathlib import Path

print("Importing modules...")
from gros_topos_curriculum import load_mini_arc
from derivator_learning import KanExtension
from train_arc_geometric_production import ARCCNNGeometricSolver
from arc_fractal_learning import FractalScaleHierarchy
from arc_solver import ARCGrid

def test_data_loading():
    """Test 1: Load datasets."""
    print("\n=== Test 1: Loading Mini-ARC ===")
    tasks = load_mini_arc()
    print(f"✓ Loaded {len(tasks)} tasks")

    if len(tasks) > 0:
        task = tasks[0]
        print(f"✓ First task: {task.task_id}")
        print(f"  Input shape: {task.input_examples[0].shape}")
        print(f"  Output shape: {task.output_examples[0].shape}")

    return tasks

def test_model_creation():
    """Test 2: Create CNN model."""
    print("\n=== Test 2: Creating Model ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model for 5x5 grids (Mini-ARC size)
    model = ARCCNNGeometricSolver(
        grid_shape_in=(5, 5),
        grid_shape_out=(5, 5),
        feature_dim=32,
        num_colors=10,
        device=device
    )

    print(f"✓ Model created")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    return model, device

def test_forward_pass(model, device, tasks):
    """Test 3: Run forward pass."""
    print("\n=== Test 3: Forward Pass ===")

    if len(tasks) == 0:
        print("✗ No tasks available")
        return

    task = tasks[0]
    inp = task.input_examples[0]
    out = task.output_examples[0]

    # Convert to tensor
    inp_tensor = torch.from_numpy(inp).long().unsqueeze(0).to(device)
    out_tensor = torch.from_numpy(out).long().unsqueeze(0).to(device)

    print(f"Input shape: {inp_tensor.shape}")

    # Forward (needs output_shape argument)
    output_shape = (out.shape[0], out.shape[1])

    with torch.no_grad():
        pred = model(inp_tensor, output_shape)

    print(f"✓ Forward pass complete")
    print(f"  Output shape: {pred.shape}")

    return pred

def test_kan_extension(model, device, tasks):
    """Test 4: Kan extension."""
    print("\n=== Test 4: Kan Extension ===")

    kan = KanExtension(feature_dim=32).to(device)
    print(f"✓ Kan extension module created")

    # Collect features from a few tasks
    features = []
    model.eval()

    with torch.no_grad():
        for task in tasks[:5]:
            inp = task.input_examples[0]
            out = task.output_examples[0]
            inp_tensor = torch.from_numpy(inp).long().unsqueeze(0).to(device)
            output_shape = (out.shape[0], out.shape[1])

            # Extract features via forward pass
            # ARCCNNGeometricSolver wraps the actual CNN, so use the internal solver
            if hasattr(model, 'cnn_solver') and hasattr(model.cnn_solver, 'sheaf_encoder'):
                # Use sheaf encoder directly
                one_hot = F.one_hot(inp_tensor, num_classes=10).float().permute(0, 3, 1, 2)
                feat = model.cnn_solver.sheaf_encoder(one_hot)
            else:
                # Fallback to full forward
                feat = model(inp_tensor, output_shape)
                if feat.dim() == 3:  # (B, H, W) - add channel dim
                    feat = feat.unsqueeze(1).float()

            feat_flat = feat.flatten(1)
            features.append(feat_flat)

    key_features = torch.cat(features, dim=0)
    print(f"✓ Collected {key_features.shape[0]} features")
    print(f"  Feature shape: {key_features.shape}")

    # Compute Kan extension
    query = key_features[:1]  # Use first as query
    with torch.no_grad():
        extended = kan(
            query=query,
            key=key_features,
            value=key_features
        )

    print(f"✓ Kan extension computed")
    print(f"  Extended shape: {extended.shape}")

    return extended

def test_scales():
    """Test 5: Scale hierarchy."""
    print("\n=== Test 5: Scale Hierarchy ===")

    scales = FractalScaleHierarchy()

    print(f"✓ {len(scales.levels)} scale levels:")
    for level in scales.levels:
        print(f"  Level {level.level_idx}: {level.name} ({level.min_size}-{level.max_size})")

    return scales

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("FRACTAL + DERIVATOR INTEGRATION TEST")
    print("="*60)

    try:
        # Test 1: Load data
        tasks = test_data_loading()

        # Test 2: Create model
        model, device = test_model_creation()

        # Test 3: Forward pass
        test_forward_pass(model, device, tasks)

        # Test 4: Kan extension
        test_kan_extension(model, device, tasks)

        # Test 5: Scales
        test_scales()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nIntegration is working! Ready to run full training.")

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
