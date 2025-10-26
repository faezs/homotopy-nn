"""
Diagnose why loss=0 but accuracy is low.

Check:
1. What is the model actually predicting?
2. What is the loss measuring?
3. Why doesn't it correlate with pixel accuracy?
"""

import torch
import torch.nn.functional as F
import numpy as np
import jax.numpy as jnp

from train_arc_geometric_production import ARCCNNGeometricSolver
from arc_solver import ARCGrid
from gros_topos_curriculum import load_mini_arc

def diagnose():
    """Run diagnostics on a fresh model."""

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load one task
    tasks = load_mini_arc()
    task = tasks[0]

    inp = task.input_examples[0]
    out = task.output_examples[0]

    print(f"\n=== Task Info ===")
    print(f"Task ID: {task.task_id}")
    print(f"Input shape: {inp.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Input:\n{inp}")
    print(f"Target:\n{out}")

    # Create model
    grid_shape = (5, 5)
    model = ARCCNNGeometricSolver(
        grid_shape_in=grid_shape,
        grid_shape_out=grid_shape,
        feature_dim=64,
        num_colors=10,
        device=device
    )

    print(f"\n=== Model Info ===")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Convert to ARCGrid
    inp_grid = ARCGrid(height=inp.shape[0], width=inp.shape[1], cells=jnp.array(inp))
    out_grid = ARCGrid(height=out.shape[0], width=out.shape[1], cells=jnp.array(out))

    # Before training
    print(f"\n=== BEFORE Training ===")
    with torch.no_grad():
        pred_grid = model(inp_grid, (out.shape[0], out.shape[1]))
        pred_cells = np.array(pred_grid.cells)

        loss_dict = model.cnn_solver.compute_topos_loss(inp_grid, out_grid)

    print(f"Prediction:\n{pred_cells}")
    print(f"Unique colors: {np.unique(pred_cells)}")
    print(f"Pixel accuracy: {(pred_cells == out).mean():.2%}")
    print(f"\nLoss components:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val.item():.6f}")

    # Train for 10 steps on this one example
    print(f"\n=== Training 10 steps ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(10):
        loss_dict = model.cnn_solver.compute_topos_loss(inp_grid, out_grid)
        loss = loss_dict['total']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_grid = model(inp_grid, (out.shape[0], out.shape[1]))
            pred_cells = np.array(pred_grid.cells)
            pixel_acc = (pred_cells == out).mean()

        if step % 2 == 0:
            print(f"Step {step}: loss={loss.item():.6f}, pixel_acc={pixel_acc:.2%}")

    # After training
    print(f"\n=== AFTER Training ===")
    with torch.no_grad():
        pred_grid = model(inp_grid, (out.shape[0], out.shape[1]))
        pred_cells = np.array(pred_grid.cells)

        loss_dict = model.cnn_solver.compute_topos_loss(inp_grid, out_grid)

    print(f"Prediction:\n{pred_cells}")
    print(f"Target:\n{out}")
    print(f"Unique colors: {np.unique(pred_cells)}")
    print(f"Pixel accuracy: {(pred_cells == out).mean():.2%}")
    print(f"\nLoss components:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val.item():.6f}")

    # Check what the sheaf representation looks like
    print(f"\n=== Sheaf Space Analysis ===")
    with torch.no_grad():
        # Encode input and target
        inp_one_hot = F.one_hot(torch.from_numpy(inp).long().to(device), num_classes=10).float()
        inp_one_hot = inp_one_hot.permute(2, 0, 1).unsqueeze(0)

        out_one_hot = F.one_hot(torch.from_numpy(out).long().to(device), num_classes=10).float()
        out_one_hot = out_one_hot.permute(2, 0, 1).unsqueeze(0)

        inp_sheaf = model.cnn_solver.sheaf_encoder(inp_one_hot)
        out_sheaf = model.cnn_solver.sheaf_encoder(out_one_hot)
        pred_sheaf = model.cnn_solver.geometric_morphism.pushforward(inp_sheaf)

        print(f"Input sheaf shape: {inp_sheaf.shape}")
        print(f"Target sheaf shape: {out_sheaf.shape}")
        print(f"Predicted sheaf shape: {pred_sheaf.shape}")
        print(f"Sheaf feature range: [{pred_sheaf.min():.3f}, {pred_sheaf.max():.3f}]")

        # Sheaf MSE
        sheaf_mse = F.mse_loss(pred_sheaf, out_sheaf)
        print(f"Sheaf MSE: {sheaf_mse.item():.6f}")

        # Now decode
        pred_logits = model.cnn_solver.decoder(pred_sheaf)
        print(f"Decoder output shape: {pred_logits.shape}")
        print(f"Logit range: [{pred_logits.min():.3f}, {pred_logits.max():.3f}]")

        # Check if logits are discriminative
        pred_probs = F.softmax(pred_logits, dim=1)
        max_probs = pred_probs.max(dim=1)[0]
        print(f"Max probabilities: mean={max_probs.mean():.3f}, min={max_probs.min():.3f}, max={max_probs.max():.3f}")

        if max_probs.mean() < 0.15:  # Close to 1/10 = random
            print(f"⚠️  WARNING: Model is uncertain (near-uniform predictions)")
        elif max_probs.mean() > 0.95:
            print(f"✓ Model is confident")

        # Check if decoder is working
        pred_colors = pred_logits.argmax(dim=1).squeeze(0).cpu().numpy()
        print(f"Decoded colors:\n{pred_colors}")
        print(f"Match with model output: {np.array_equal(pred_colors, pred_cells)}")

    print(f"\n=== DIAGNOSIS ===")
    if loss_dict['total'].item() < 0.01 and pixel_acc < 0.1:
        print("🔴 PROBLEM: Loss is low but pixel accuracy is terrible!")
        print("   Cause: Loss function (MSE in sheaf space) ≠ pixel accuracy")
        print("   Solution: Use cross-entropy loss on decoded pixels")
    elif np.unique(pred_cells).size == 1:
        print("🔴 PROBLEM: Model predicts constant color!")
        print("   Cause: Degenerate solution minimizes loss")
        print("   Solution: Add pixel-level loss term")
    elif max_probs.mean() < 0.2:
        print("🟡 PROBLEM: Model is very uncertain")
        print("   Cause: Underfitting - needs more training")
        print("   Solution: Train longer or increase capacity")
    else:
        print("✓ Model seems to be learning correctly")

if __name__ == "__main__":
    diagnose()
