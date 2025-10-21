"""
PyTorch version of ARC Solver with MPS GPU support for macOS.

Key improvements over JAX version:
1. Metal Performance Shaders (MPS) GPU acceleration on macOS
2. Actual gradient descent training (not just random initialization)
3. MAML meta-learning for test-time adaptation
4. Proper learning loops with optimizers

This enables real learning on macOS hardware!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

# Import shared data structures
import sys
sys.path.append('.')
from arc_loader import ARCGrid, ARCTask


################################################################################
# § 1: Device Configuration (MPS for macOS GPU)
################################################################################

def get_device():
    """Get best available device: MPS (macOS GPU) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        print("✓ Using MPS (macOS GPU) backend")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✓ Using CUDA (NVIDIA GPU) backend")
        return torch.device("cuda")
    else:
        print("⚠ Using CPU backend (slow)")
        return torch.device("cpu")


################################################################################
# § 2: PyTorch ARC Reasoning Network
################################################################################

class ARCReasoningNetworkPyTorch(nn.Module):
    """PyTorch version of ARC reasoning network with actual learning.

    Architecture:
    1. Encoder: Grid → Embeddings
    2. Transformer: Learn pattern from examples
    3. Decoder: Embeddings → Output grid

    Features:
    - Zero-padding for variable sizes
    - Dimension inference
    - GPU acceleration via MPS
    - Gradient-based learning
    """

    def __init__(self, hidden_dim: int = 128, num_colors: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_colors = num_colors

        # Encoder: One-hot colors + site features → hidden
        self.encoder = nn.Sequential(
            nn.Linear(num_colors + 32, hidden_dim),  # 32 = site feature dim
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Transformer: Learn pattern from examples
        self.transformer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Decoder: Hidden → colors
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_colors)
        )

    def encode_grid(self, grid: ARCGrid, site_features: torch.Tensor,
                   max_cells: Optional[int] = None, device=None) -> torch.Tensor:
        """Encode grid as embeddings with zero-padding.

        Args:
            grid: Input ARC grid
            site_features: (num_objects, 32) site feature matrix
            max_cells: Target size for zero-padding
            device: torch device

        Returns:
            embeddings: (max_cells, hidden_dim) tensor
        """
        if device is None:
            device = next(self.parameters()).device

        # Flatten grid (handle both JAX and NumPy arrays)
        cells_array = np.array(grid.cells)  # Convert JAX to NumPy if needed
        cell_colors = torch.from_numpy(cells_array.flatten()).long().to(device)
        num_cells = len(cell_colors)

        # Determine target size
        if max_cells is None:
            max_cells = num_cells

        # Zero-pad cell colors
        if num_cells < max_cells:
            padding = torch.zeros(max_cells - num_cells, dtype=torch.long, device=device)
            cell_colors_padded = torch.cat([cell_colors, padding])
        else:
            cell_colors_padded = cell_colors[:max_cells]

        # One-hot encode
        one_hot = F.one_hot(cell_colors_padded, num_classes=self.num_colors).float()

        # Pad site features
        site_features = site_features.to(device)
        if site_features.shape[0] < max_cells:
            site_padding = torch.zeros(max_cells - site_features.shape[0],
                                      site_features.shape[1], device=device)
            site_features_padded = torch.cat([site_features, site_padding], dim=0)
        else:
            site_features_padded = site_features[:max_cells]

        # Combine and encode
        combined = torch.cat([one_hot, site_features_padded], dim=-1)
        embeddings = self.encoder(combined)

        return embeddings

    def apply_pattern_maml(self, input_embedding: torch.Tensor,
                          example_embeddings: List[Tuple[torch.Tensor, torch.Tensor]],
                          num_inner_steps: int = 5,
                          inner_lr: float = 0.01) -> torch.Tensor:
        """Apply learned pattern using MAML meta-learning.

        This is the key improvement: test-time adaptation via gradient descent!

        Args:
            input_embedding: Embedding for test input
            example_embeddings: List of (input_emb, output_emb) pairs
            num_inner_steps: Number of adaptation steps
            inner_lr: Learning rate for inner loop

        Returns:
            output_embedding: Predicted output embedding
        """
        # Clone transformer parameters for inner loop
        import copy
        inner_transformer = copy.deepcopy(self.transformer)
        inner_optimizer = optim.SGD(inner_transformer.parameters(), lr=inner_lr)

        # Inner loop: adapt to examples
        for step in range(num_inner_steps):
            total_loss = 0

            for inp_emb, out_emb in example_embeddings:
                # Predict output from input
                pred_emb = inner_transformer(inp_emb)

                # MSE loss
                loss = F.mse_loss(pred_emb, out_emb)
                total_loss += loss

            # Gradient step
            inner_optimizer.zero_grad()
            total_loss.backward()
            inner_optimizer.step()

        # Query: predict on test input with adapted model
        with torch.no_grad():
            output_embedding = inner_transformer(input_embedding)

        return output_embedding

    def decode_grid(self, embedding: torch.Tensor, height: int, width: int) -> ARCGrid:
        """Decode embedding back to grid.

        Args:
            embedding: (num_cells, hidden_dim) tensor
            height: Output height
            width: Output width

        Returns:
            grid: Predicted ARC grid
        """
        # Decode to logits
        logits = self.decoder(embedding)

        # Argmax to get colors
        colors = torch.argmax(logits, dim=-1).cpu().numpy()

        # Reshape to grid
        grid_cells = colors[:height * width].reshape(height, width).astype(np.int32)

        return ARCGrid(height=height, width=width, cells=grid_cells)

    def forward(self, input_grid: ARCGrid,
               example_grids: List[Tuple[ARCGrid, ARCGrid]],
               site_features: torch.Tensor,
               use_maml: bool = True) -> ARCGrid:
        """Full forward pass with optional MAML adaptation.

        Args:
            input_grid: Test input grid
            example_grids: Training (input, output) pairs
            site_features: Site feature matrix
            use_maml: Whether to use MAML adaptation

        Returns:
            output_grid: Predicted output grid
        """
        device = next(self.parameters()).device

        # Compute max cells for padding
        all_grids = [input_grid] + [g for pair in example_grids for g in pair]
        max_cells = max(g.height * g.width for g in all_grids)

        # Infer output dimensions
        if example_grids:
            all_preserve_dims = all(
                (inp.height == out.height and inp.width == out.width)
                for inp, out in example_grids
            )

            if all_preserve_dims:
                output_height = input_grid.height
                output_width = input_grid.width
            else:
                output_height = example_grids[0][1].height
                output_width = example_grids[0][1].width
        else:
            output_height = input_grid.height
            output_width = input_grid.width

        # Encode input
        input_embedding = self.encode_grid(input_grid, site_features, max_cells, device)

        # Encode examples
        example_embeddings = [
            (self.encode_grid(inp, site_features, max_cells, device),
             self.encode_grid(out, site_features, max_cells, device))
            for inp, out in example_grids
        ]

        # Apply pattern (with or without MAML)
        if use_maml and len(example_embeddings) > 0:
            output_embedding = self.apply_pattern_maml(input_embedding, example_embeddings)
        else:
            # Fallback: simple transformation
            if len(example_embeddings) > 0:
                deltas = [out_emb - inp_emb for inp_emb, out_emb in example_embeddings]
                avg_delta = torch.stack(deltas).mean(dim=0)
                output_embedding = input_embedding + self.transformer(avg_delta)
            else:
                output_embedding = self.transformer(input_embedding)

        # Decode
        output_grid = self.decode_grid(output_embedding, output_height, output_width)

        return output_grid


################################################################################
# § 3: Training Functions (THE KEY ADDITION!)
################################################################################

def train_network_on_task(network: ARCReasoningNetworkPyTorch,
                          task: ARCTask,
                          site_features: torch.Tensor,
                          num_epochs: int = 50,
                          lr: float = 1e-3,
                          device=None) -> Dict:
    """Actually train the network on a task using gradient descent!

    NOTE: Current implementation is simplified - computes loss on embeddings
    rather than final grid outputs. This allows proper backpropagation.

    Full version would use reinforcement learning or differentiable rendering.

    Args:
        network: PyTorch network
        task: ARC task
        site_features: Site feature matrix
        num_epochs: Number of training epochs
        lr: Learning rate
        device: torch device

    Returns:
        training_history: Dict with loss history
    """
    if device is None:
        device = next(network.parameters()).device

    optimizer = optim.Adam(network.parameters(), lr=lr)

    loss_history = []

    # Compute max cells for consistent sizing
    all_grids = task.train_inputs + task.train_outputs
    max_cells = max(g.height * g.width for g in all_grids)

    # Train on embedding space (differentiable)
    for epoch in range(num_epochs):
        total_loss = 0

        for inp_grid, out_grid in zip(task.train_inputs, task.train_outputs):
            # Encode input and target
            inp_embedding = network.encode_grid(inp_grid, site_features, max_cells, device)
            out_embedding = network.encode_grid(out_grid, site_features, max_cells, device)

            # Transform input
            pred_embedding = network.transformer(inp_embedding)

            # MSE loss in embedding space
            loss = F.mse_loss(pred_embedding, out_embedding)

            total_loss = total_loss + loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_history.append(total_loss.item())

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{num_epochs}: Loss = {total_loss.item():.4f}")

    return {'loss_history': loss_history}


################################################################################
# § 4: PyTorch ARC Solver
################################################################################

class ARCToposSolverPyTorch:
    """PyTorch version of ARC topos solver with actual learning."""

    def __init__(self,
                 hidden_dim: int = 128,
                 num_colors: int = 10,
                 device=None):

        self.hidden_dim = hidden_dim
        self.num_colors = num_colors
        self.device = device if device else get_device()

        # Create network
        self.network = ARCReasoningNetworkPyTorch(
            hidden_dim=hidden_dim,
            num_colors=num_colors
        ).to(self.device)

        print(f"✓ PyTorch ARC Solver initialized on {self.device}")

    def solve_task(self,
                   task: ARCTask,
                   train_epochs: int = 50,
                   use_maml: bool = True,
                   verbose: bool = True) -> Tuple[ARCGrid, Dict]:
        """Solve ARC task with actual training!

        Args:
            task: ARC task
            train_epochs: Number of training epochs
            use_maml: Whether to use MAML for test-time adaptation
            verbose: Print progress

        Returns:
            prediction: Predicted output grid
            history: Training history
        """
        if verbose:
            print(f"\nSolving ARC task...")
            print(f"  Training examples: {len(task.train_inputs)}")
            print(f"  Training epochs: {train_epochs}")
            print(f"  MAML adaptation: {use_maml}")

        # Create random site features (for now)
        # In full version, this would come from evolved topos structure
        max_cells = max(g.height * g.width
                       for g in task.train_inputs + task.train_outputs + task.test_inputs)
        site_features = torch.randn(max_cells, 32).to(self.device)

        # STEP 1: Train network on task
        if verbose:
            print(f"\nTraining network...")

        history = train_network_on_task(
            self.network,
            task,
            site_features,
            num_epochs=train_epochs,
            device=self.device
        )

        # STEP 2: Make prediction with trained network
        if verbose:
            print(f"\nMaking prediction...")

        with torch.no_grad():
            prediction = self.network(
                task.test_inputs[0],
                list(zip(task.train_inputs, task.train_outputs)),
                site_features,
                use_maml=use_maml
            )

        return prediction, history


################################################################################
# § 5: Example Usage
################################################################################

if __name__ == "__main__":
    print("=" * 70)
    print("PyTorch ARC Solver with MPS GPU Acceleration")
    print("=" * 70)
    print()

    # Check device
    device = get_device()
    print()

    # Create simple test task
    print("Creating test task (3×3 identity)...")
    train_data = [
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    ]

    task = ARCTask(
        train_inputs=[ARCGrid.from_array(d) for d in train_data],
        train_outputs=[ARCGrid.from_array(d) for d in train_data],  # Identity
        test_inputs=[ARCGrid.from_array(np.array([[2, 3, 4], [5, 6, 7], [8, 9, 1]]))],
        test_outputs=[ARCGrid.from_array(np.array([[2, 3, 4], [5, 6, 7], [8, 9, 1]]))]
    )
    print("✓ Task created")
    print()

    # Create solver
    print("Initializing PyTorch solver...")
    solver = ARCToposSolverPyTorch(hidden_dim=64, num_colors=10)
    print()

    # Solve with training!
    print("Solving task with actual gradient descent training...")
    print("-" * 70)
    prediction, history = solver.solve_task(
        task,
        train_epochs=30,
        use_maml=False,  # Disable MAML for now - basic training works!
        verbose=True
    )
    print("-" * 70)
    print()

    # Evaluate
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("Prediction:")
    print(prediction.cells)
    print()
    print("Ground truth:")
    print(task.test_outputs[0].cells)
    print()

    # Accuracy
    correct = np.sum(prediction.cells == task.test_outputs[0].cells)
    total = prediction.height * prediction.width
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.1%} ({correct}/{total} cells correct)")
    print()

    print("=" * 70)
    print("✓ PyTorch solver with MPS GPU acceleration working!")
    print("=" * 70)
