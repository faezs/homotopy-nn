"""
Pure TRM Baseline for ARC Tasks

Implements two-level recursive reasoning exactly as in the TRM paper:
- z_H (high-level reasoning): H_cycles
- z_L (low-level answer): L_cycles per H cycle
- No symbolic formulas, pure sequence-to-sequence
- Grid → flatten to sequence → predict output sequence → unflatten

Architecture:
    Input grid [H, W]
        ↓ flatten + embed
    Tokens [900] (30×30 padded grid)
        ↓ encoder
    (z_H, z_L) initial states [B, L, D]
        ↓
    For h in H_cycles-1 (no grad):
        For l in L_cycles:
            z_L = refine_L(z_L, z_H + input_embed)
        z_H = refine_H(z_H, z_L)
    For l in L_cycles (with grad):
        z_L = refine_L(z_L, z_H + input_embed)
    z_H = refine_H(z_H, z_L)
        ↓
    LM head(z_H) → output tokens [900]
        ↓ unflatten
    Output grid [H, W]

Author: Claude Code
Date: October 23, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


################################################################################
# § 1: Grid Encoding/Decoding
################################################################################

MAX_GRID_SIZE = 30
SEQ_LEN = MAX_GRID_SIZE * MAX_GRID_SIZE  # 900

# Vocabulary: PAD=0, EOS=1, colors 0-9 → tokens 2-11
PAD_TOKEN = 0
EOS_TOKEN = 1
VOCAB_SIZE = 12  # 0-11


def grid_to_sequence(grid: torch.Tensor, max_size: int = MAX_GRID_SIZE) -> torch.Tensor:
    """Convert grid to padded sequence with EOS markers.

    Args:
        grid: [B, H, W] or [H, W] color indices (0-9)
        max_size: Maximum grid dimension (30)

    Returns:
        seq: [B, max_size*max_size] token sequence
    """
    if grid.dim() == 2:
        grid = grid.unsqueeze(0)

    B, H, W = grid.shape
    device = grid.device

    # Initialize with PAD
    seq = torch.zeros(B, max_size, max_size, dtype=torch.long, device=device)

    # Fill with grid values (shift by 2: color 0 → token 2)
    seq[:, :H, :W] = grid + 2

    # Add EOS markers
    # Horizontal EOS at end of each row
    if W < max_size:
        seq[:, :H, W] = EOS_TOKEN

    # Vertical EOS at end of grid
    if H < max_size:
        seq[:, H, :W] = EOS_TOKEN

    # Flatten
    return seq.view(B, -1)


def sequence_to_grid(seq: torch.Tensor, target_H: int, target_W: int) -> torch.Tensor:
    """Convert sequence back to grid.

    Args:
        seq: [B, max_size*max_size] token sequence
        target_H, target_W: Output grid dimensions

    Returns:
        grid: [B, target_H, target_W] color indices
    """
    B = seq.shape[0]
    max_size = int(math.sqrt(seq.shape[1]))

    # Reshape to 2D
    seq_2d = seq.view(B, max_size, max_size)

    # Extract grid region and convert tokens to colors (token 2 → color 0)
    grid = seq_2d[:, :target_H, :target_W]
    grid = torch.clamp(grid - 2, min=0, max=9)

    return grid


################################################################################
# § 2: RMS Norm
################################################################################

def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS normalization (same as TRM)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


################################################################################
# § 3: SwiGLU MLP
################################################################################

class SwiGLU(nn.Module):
    """SwiGLU activation MLP (same as TRM)."""

    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        expanded_size = int(hidden_size * expansion)

        self.gate = nn.Linear(hidden_size, expanded_size, bias=False)
        self.up = nn.Linear(hidden_size, expanded_size, bias=False)
        self.down = nn.Linear(expanded_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


################################################################################
# § 4: Transformer Block
################################################################################

class TransformerBlock(nn.Module):
    """Simple transformer block with self-attention + MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float = 4.0,
        rms_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm_eps = rms_norm_eps

        assert hidden_size % num_heads == 0

        # Self-attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # MLP
        self.mlp = SwiGLU(hidden_size, expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            out: [B, L, D]
        """
        B, L, D = x.shape

        # Self-attention with post-norm
        residual = x

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        attn = self.o_proj(attn)

        x = rms_norm(residual + attn, eps=self.norm_eps)

        # MLP with post-norm
        residual = x
        x = rms_norm(residual + self.mlp(x), eps=self.norm_eps)

        return x


################################################################################
# § 5: Reasoning Module (H or L level)
################################################################################

class ReasoningModule(nn.Module):
    """Stack of transformer blocks for H or L level reasoning."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        expansion: float = 4.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, expansion)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, D] current state (z_H or z_L)
            input_injection: [B, L, D] injected input

        Returns:
            updated: [B, L, D]
        """
        # Add input injection
        x = hidden_states + input_injection

        # Pass through layers
        for layer in self.layers:
            x = layer(x)

        return x


################################################################################
# § 6: TRM Baseline Model
################################################################################

class TRMBaseline(nn.Module):
    """Pure TRM with two-level recursion for ARC tasks.

    No symbolic formulas - pure sequence-to-sequence prediction.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        H_layers: int = 2,
        L_layers: int = 2,
        H_cycles: int = 3,
        L_cycles: int = 2,
        expansion: float = 4.0,
        device: str = 'cuda'
    ):
        """
        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            H_layers: Number of layers in H-level reasoning
            L_layers: Number of layers in L-level reasoning
            H_cycles: Number of high-level recursion cycles
            L_cycles: Number of low-level cycles per H cycle
            expansion: MLP expansion factor
            device: Device
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.device = torch.device(device)

        # Token embedding
        embed_scale = math.sqrt(hidden_size)
        self.embed_scale = embed_scale
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, hidden_size)
        nn.init.normal_(self.embed_tokens.weight, std=1.0 / embed_scale)

        # Output head
        self.lm_head = nn.Linear(hidden_size, VOCAB_SIZE, bias=False)

        # Reasoning modules
        self.L_level = ReasoningModule(hidden_size, L_layers, num_heads, expansion)
        self.H_level = ReasoningModule(hidden_size, H_layers, num_heads, expansion)

        # Initial states (learnable)
        self.H_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.L_init = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(
        self,
        input_grid: torch.Tensor,
        target_grid: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_grid: [B, H, W] color indices
            target_grid: [B, H_out, W_out] optional target for loss
            target_size: (H_out, W_out) output size

        Returns:
            output_grid: [B, H_out, W_out] predicted colors
            loss: scalar (if target_grid provided)
        """
        if input_grid.dim() == 2:
            input_grid = input_grid.unsqueeze(0)

        B = input_grid.shape[0]

        # Determine output size
        if target_size is None:
            if target_grid is not None:
                target_size = (target_grid.shape[1], target_grid.shape[2])
            else:
                target_size = (input_grid.shape[1], input_grid.shape[2])

        H_out, W_out = target_size

        # Convert grid to sequence
        input_seq = grid_to_sequence(input_grid).to(self.device)  # [B, 900]

        # Embed
        input_embeddings = self.embed_scale * self.embed_tokens(input_seq)  # [B, 900, D]

        # Initialize states
        L = SEQ_LEN
        z_H = self.H_init.view(1, 1, -1).expand(B, L, -1).clone()
        z_L = self.L_init.view(1, 1, -1).expand(B, L, -1).clone()

        # Two-level recursion
        # H_cycles-1 without gradients
        with torch.no_grad():
            for h in range(self.H_cycles - 1):
                # L cycles
                for l in range(self.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings)
                # H update
                z_H = self.H_level(z_H, z_L)

        # Final cycle with gradients
        for l in range(self.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings)
        z_H = self.H_level(z_H, z_L)

        # Predict output sequence
        logits = self.lm_head(z_H)  # [B, 900, VOCAB_SIZE]

        # Convert to grid
        output_tokens = torch.argmax(logits, dim=-1)  # [B, 900]
        output_grid = sequence_to_grid(output_tokens, H_out, W_out)

        # Compute loss if target provided
        loss = None
        if target_grid is not None:
            target_seq = grid_to_sequence(target_grid).to(self.device)
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                target_seq.view(-1),
                ignore_index=PAD_TOKEN
            )

        return output_grid, loss


################################################################################
# § 7: Training Function
################################################################################

def train_trm_baseline(
    train_inputs: torch.Tensor,
    train_outputs: torch.Tensor,
    test_input: torch.Tensor,
    test_output: torch.Tensor,
    num_epochs: int = 200,
    lr: float = 1e-3,
    H_cycles: int = 3,
    L_cycles: int = 2,
    device: str = 'cuda'
):
    """Train TRM baseline on ARC task.

    Args:
        train_inputs: [N, H, W] training input grids
        train_outputs: [N, H_out, W_out] training output grids
        test_input: [H, W] test input
        test_output: [H_out, W_out] test output
        num_epochs: Number of training epochs
        lr: Learning rate
        H_cycles: High-level recursion cycles
        L_cycles: Low-level cycles per H cycle
        device: Device
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device)

    print("="*80)
    print("TRM BASELINE TRAINING")
    print("="*80)
    print(f"Device: {device}")
    print(f"H_cycles: {H_cycles}, L_cycles: {L_cycles}")
    print(f"Training examples: {train_inputs.shape[0]}")
    print(f"Input size: {train_inputs.shape[1:]}")
    print(f"Output size: {train_outputs.shape[1:]}")

    # Create model
    model = TRMBaseline(
        hidden_size=256,
        num_heads=8,
        H_layers=2,
        L_layers=2,
        H_cycles=H_cycles,
        L_cycles=L_cycles,
        device=device
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Move data to device
    train_inputs = train_inputs.to(device)
    train_outputs = train_outputs.to(device)
    test_input = test_input.to(device)
    test_output = test_output.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    from tqdm import tqdm

    best_loss = float('inf')
    losses = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()

        # Forward
        _, loss = model(train_inputs, train_outputs)

        # Backward
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, best={best_loss:.4f}")

    print(f"\n✅ Training complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Best loss: {best_loss:.4f}")

    # Test
    print("\n=== Testing ===")
    model.eval()
    with torch.no_grad():
        output_pred, _ = model(test_input, target_size=test_output.shape)

        correct = (output_pred == test_output).sum().item()
        total = test_output.numel()
        accuracy = 100.0 * correct / total

        print(f"Test accuracy: {accuracy:.2f}% ({correct}/{total} pixels)")

        print("\nPredicted output:")
        for row in output_pred[0].cpu().numpy():
            print("  ", list(row))

        print("\nExpected output:")
        for row in test_output.cpu().numpy():
            print("  ", list(row))

        if accuracy == 100.0:
            print("\n🎉 ✅ PERFECT SOLUTION!")
            return True
        elif accuracy > 90.0:
            print(f"\n✅ Nearly perfect ({accuracy:.1f}%)")
            return True
        else:
            print(f"\n❌ Failed ({accuracy:.1f}%)")
            return False


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing TRM baseline with synthetic data...")

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    # Simple copy task
    train_inputs = torch.randint(0, 10, (4, 3, 3))
    train_outputs = train_inputs.clone()

    test_input = torch.randint(0, 10, (3, 3))
    test_output = test_input.clone()

    success = train_trm_baseline(
        train_inputs,
        train_outputs,
        test_input,
        test_output,
        num_epochs=50,
        H_cycles=2,
        L_cycles=1,
        device=device
    )

    print(f"\nSuccess: {success}")
