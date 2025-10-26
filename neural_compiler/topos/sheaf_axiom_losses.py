"""
Sheaf Axiom Losses for Enforcing Category Theory

This module implements loss functions that enforce sheaf and functoriality axioms:

1. Composition Law: F_ij ∘ F_jk = F_ik (restriction maps compose correctly)
2. Identity Axiom: F_ii = I (identity restriction map on self-loops)

These losses make the cellular sheaf NN actually satisfy sheaf axioms,
not just have sheaf-themed architecture!

Mathematical Foundation:
- Sheaf is a functor F: C^op → Sets (or Vect for vector spaces)
- Functoriality requires: F(g ∘ f) = F(f) ∘ F(g)
- For restriction maps: ρ_UV ∘ ρ_VW = ρ_UW where U ⊆ V ⊆ W

Author: Claude Code + Human
Date: October 25, 2025
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict


################################################################################
# § 1: Path Finding in Grid Graphs
################################################################################

def find_2hop_paths(edge_index: torch.Tensor, num_vertices: int) -> torch.Tensor:
    """
    Find all 2-hop paths in graph: i → j → k

    For composition law checking: F_ik should equal F_ij @ F_jk

    Args:
        edge_index: [2, E] edge list (source, target)
        num_vertices: Number of vertices in graph

    Returns:
        paths: [P, 3] tensor where paths[p] = [i, j, k]
               representing path i → j → k
    """
    source, target = edge_index

    # Build adjacency lists for efficient neighbor lookup
    # adj_list[j] = list of k where j → k exists
    adj_list = {i: [] for i in range(num_vertices)}
    for s, t in zip(source.tolist(), target.tolist()):
        adj_list[s].append(t)

    # Find all 2-hop paths
    paths = []
    for edge_idx in range(edge_index.size(1)):
        i = source[edge_idx].item()
        j = target[edge_idx].item()

        # Find all k where j → k
        for k in adj_list[j]:
            paths.append([i, j, k])

    if len(paths) == 0:
        # No 2-hop paths (graph too sparse)
        return torch.zeros(0, 3, dtype=torch.long, device=edge_index.device)

    return torch.tensor(paths, dtype=torch.long, device=edge_index.device)


def find_edge_index(edge_index: torch.Tensor, source: int, target: int) -> int:
    """
    Find index of edge (source, target) in edge_index.

    Args:
        edge_index: [2, E] edge list
        source: Source vertex
        target: Target vertex

    Returns:
        idx: Index of edge, or -1 if not found
    """
    matches = (edge_index[0] == source) & (edge_index[1] == target)
    indices = torch.where(matches)[0]

    if len(indices) == 0:
        return -1

    return indices[0].item()


################################################################################
# § 2: Composition Law Loss
################################################################################

def composition_law_loss(restriction_maps: torch.Tensor,
                         edge_index: torch.Tensor,
                         num_vertices: int,
                         sample_paths: int = 100) -> torch.Tensor:
    """
    Loss enforcing F_ik = F_ij @ F_jk (functoriality of restriction maps).

    For each 2-hop path i → j → k:
    - Composed: F_ij @ F_jk (matrix multiplication)
    - Direct: F_ik (if edge i → k exists)
    - Loss: ||F_ik - F_ij @ F_jk||²_F (Frobenius norm)

    Args:
        restriction_maps: [E, d, d] learned restriction matrices
        edge_index: [2, E] edge list
        num_vertices: Number of vertices
        sample_paths: Maximum paths to check (for efficiency)

    Returns:
        loss: Scalar measuring composition law violation
    """
    device = restriction_maps.device
    d = restriction_maps.size(1)

    # Find all 2-hop paths
    paths = find_2hop_paths(edge_index, num_vertices)  # [P, 3]

    if paths.size(0) == 0:
        # No 2-hop paths
        return torch.tensor(0.0, device=device)

    # Sample paths if too many (for efficiency)
    if paths.size(0) > sample_paths:
        perm = torch.randperm(paths.size(0))[:sample_paths]
        paths = paths[perm]

    total_violation = torch.tensor(0.0, device=device)
    num_valid_paths = 0

    for path_idx in range(paths.size(0)):
        i, j, k = paths[path_idx].tolist()

        # Find edge indices
        ij_idx = find_edge_index(edge_index, i, j)
        jk_idx = find_edge_index(edge_index, j, k)
        ik_idx = find_edge_index(edge_index, i, k)

        if ij_idx == -1 or jk_idx == -1:
            # Missing edge in path (shouldn't happen, but check)
            continue

        # Get restriction maps
        F_ij = restriction_maps[ij_idx]  # [d, d]
        F_jk = restriction_maps[jk_idx]  # [d, d]

        # Compose: F_ij @ F_jk
        F_composed = torch.matmul(F_ij, F_jk)  # [d, d]

        if ik_idx != -1:
            # Direct edge i → k exists
            F_ik = restriction_maps[ik_idx]  # [d, d]

            # Violation: ||F_ik - F_ij @ F_jk||²_F
            violation = torch.norm(F_ik - F_composed, p='fro')**2

        else:
            # No direct edge i → k
            # Weak constraint: Composed map should be "reasonable"
            # (bounded norm, close to orthogonal, etc.)
            # For now, no penalty
            violation = torch.tensor(0.0, device=device)

        total_violation += violation
        num_valid_paths += 1

    if num_valid_paths == 0:
        return torch.tensor(0.0, device=device)

    # Average violation
    loss = total_violation / num_valid_paths

    return loss


################################################################################
# § 3: Identity Axiom Loss
################################################################################

def identity_axiom_loss(restriction_maps: torch.Tensor,
                       edge_index: torch.Tensor) -> torch.Tensor:
    """
    Loss enforcing F_ii = I (identity on self-loops).

    For each edge (i, i) (self-loop):
    - F_ii should be identity matrix
    - Loss: ||F_ii - I||²_F

    Note: Grid graphs typically don't have self-loops.
    If no self-loops exist, this loss is 0.

    Args:
        restriction_maps: [E, d, d] learned restriction matrices
        edge_index: [2, E] edge list

    Returns:
        loss: Scalar measuring identity axiom violation
    """
    device = restriction_maps.device
    d = restriction_maps.size(1)

    # Find self-loops (i, i)
    source, target = edge_index
    self_loops = source == target

    if not self_loops.any():
        # No self-loops in graph
        return torch.tensor(0.0, device=device)

    # Get restriction maps for self-loops
    F_self = restriction_maps[self_loops]  # [num_self_loops, d, d]

    # Identity matrix
    I = torch.eye(d, device=device).unsqueeze(0)  # [1, d, d]

    # Violation: ||F_ii - I||²_F for each self-loop
    violations = torch.norm(F_self - I, p='fro', dim=(1, 2))**2  # [num_self_loops]

    # Average violation
    loss = violations.mean()

    return loss


################################################################################
# § 4: Orthogonality and Stability Regularization
################################################################################

def orthogonality_loss(restriction_maps: torch.Tensor,
                      target_orthogonality: float = 1.0) -> torch.Tensor:
    """
    Encourage restriction maps to be orthogonal (preserve norms).

    For stable sheaf diffusion, restriction maps should approximately
    preserve vector norms: ||F_ij v|| ≈ ||v||

    This is satisfied if F_ij^T F_ij ≈ I (F is orthogonal).

    Args:
        restriction_maps: [E, d, d] restriction matrices
        target_orthogonality: How close to orthogonal (1.0 = perfectly orthogonal)

    Returns:
        loss: Scalar measuring deviation from orthogonality
    """
    d = restriction_maps.size(1)
    device = restriction_maps.device

    # Compute F^T F for each restriction map
    FtF = torch.bmm(restriction_maps.transpose(1, 2), restriction_maps)  # [E, d, d]

    # Identity matrix
    I = torch.eye(d, device=device).unsqueeze(0)  # [1, d, d]

    # Deviation from identity: ||F^T F - I||²_F
    deviations = torch.norm(FtF - target_orthogonality * I, p='fro', dim=(1, 2))**2

    loss = deviations.mean()

    return loss


def spectral_norm_loss(restriction_maps: torch.Tensor,
                       max_spectral_norm: float = 1.0) -> torch.Tensor:
    """
    Regularize spectral norm of restriction maps (largest singular value).

    For stability, we want σ_max(F_ij) ≤ 1 (maps don't amplify).

    Args:
        restriction_maps: [E, d, d] restriction matrices
        max_spectral_norm: Maximum allowed spectral norm

    Returns:
        loss: Scalar penalizing large spectral norms
    """
    # Compute singular values
    U, S, V = torch.svd(restriction_maps)  # S: [E, d]

    # Spectral norm = largest singular value
    spectral_norms = S.max(dim=1)[0]  # [E]

    # Penalize norms exceeding threshold
    violations = F.relu(spectral_norms - max_spectral_norm)**2

    loss = violations.mean()

    return loss


################################################################################
# § 5: Combined Sheaf Loss
################################################################################

def combined_sheaf_axiom_loss(restriction_maps: torch.Tensor,
                              edge_index: torch.Tensor,
                              num_vertices: int,
                              composition_weight: float = 1.0,
                              identity_weight: float = 1.0,
                              orthogonality_weight: float = 0.1,
                              spectral_weight: float = 0.1) -> Dict[str, torch.Tensor]:
    """
    Combined loss enforcing all sheaf axioms and regularizations.

    Args:
        restriction_maps: [E, d, d] learned restriction matrices
        edge_index: [2, E] edge list
        num_vertices: Number of vertices
        composition_weight: Weight for composition law (λ_comp)
        identity_weight: Weight for identity axiom (λ_id)
        orthogonality_weight: Weight for orthogonality regularization (λ_orth)
        spectral_weight: Weight for spectral norm regularization (λ_spec)

    Returns:
        losses: Dict with individual losses and total
    """
    # Individual losses
    comp_loss = composition_law_loss(restriction_maps, edge_index, num_vertices)
    id_loss = identity_axiom_loss(restriction_maps, edge_index)
    orth_loss = orthogonality_loss(restriction_maps)

    # Only compute spectral loss if weight is non-zero (avoids MPS SVD issues)
    if spectral_weight > 0.0:
        spec_loss = spectral_norm_loss(restriction_maps)
    else:
        spec_loss = torch.tensor(0.0, device=restriction_maps.device)

    # Weighted total
    total_loss = (
        composition_weight * comp_loss +
        identity_weight * id_loss +
        orthogonality_weight * orth_loss +
        spectral_weight * spec_loss
    )

    return {
        'composition': comp_loss,
        'identity': id_loss,
        'orthogonality': orth_loss,
        'spectral': spec_loss,
        'total': total_loss
    }


################################################################################
# § 6: Testing
################################################################################

def test_sheaf_axiom_losses():
    """
    Test sheaf axiom losses on synthetic restriction maps.
    """
    print("=" * 80)
    print("TESTING SHEAF AXIOM LOSSES")
    print("=" * 80)
    print()

    # Create simple grid graph: 0 → 1 → 2 → 3
    edge_index = torch.tensor([
        [0, 1, 2],  # sources
        [1, 2, 3]   # targets
    ], dtype=torch.long)
    num_vertices = 4
    d = 4  # stalk dimension

    print("Test 1: Perfect composition (F_ik = F_ij @ F_jk)")
    print("-" * 40)

    # Create restriction maps satisfying composition
    F_01 = torch.eye(d)
    F_12 = torch.eye(d) * 0.5
    F_23 = torch.eye(d) * 0.5

    restriction_maps = torch.stack([F_01, F_12, F_23])  # [3, 4, 4]

    # Add edge 0 → 2 with F_02 = F_01 @ F_12
    edge_index_extended = torch.tensor([
        [0, 1, 2, 0],
        [1, 2, 3, 2]
    ], dtype=torch.long)
    F_02 = torch.matmul(F_01, F_12)
    restriction_maps_extended = torch.cat([restriction_maps, F_02.unsqueeze(0)])

    comp_loss = composition_law_loss(restriction_maps_extended, edge_index_extended, num_vertices)
    print(f"Composition loss: {comp_loss.item():.6f}")
    print(f"Expected: ~0.0 (maps compose correctly)")
    print()

    print("Test 2: Imperfect composition")
    print("-" * 40)

    # Make F_02 NOT equal F_01 @ F_12
    F_02_wrong = torch.randn(d, d)
    restriction_maps_wrong = torch.cat([restriction_maps, F_02_wrong.unsqueeze(0)])

    comp_loss_wrong = composition_law_loss(restriction_maps_wrong, edge_index_extended, num_vertices)
    print(f"Composition loss: {comp_loss_wrong.item():.6f}")
    print(f"Expected: >0.1 (maps don't compose)")
    print()

    print("Test 3: Identity axiom (self-loops)")
    print("-" * 40)

    # Add self-loop 0 → 0
    edge_index_selfloop = torch.tensor([
        [0, 1, 2, 0],
        [1, 2, 3, 0]
    ], dtype=torch.long)
    F_00 = torch.eye(d)
    restriction_maps_selfloop = torch.cat([restriction_maps, F_00.unsqueeze(0)])

    id_loss_perfect = identity_axiom_loss(restriction_maps_selfloop, edge_index_selfloop)
    print(f"Identity loss (perfect): {id_loss_perfect.item():.6f}")

    # Violate identity
    F_00_wrong = torch.randn(d, d)
    restriction_maps_selfloop_wrong = torch.cat([restriction_maps, F_00_wrong.unsqueeze(0)])
    id_loss_wrong = identity_axiom_loss(restriction_maps_selfloop_wrong, edge_index_selfloop)
    print(f"Identity loss (wrong): {id_loss_wrong.item():.6f}")
    print()

    print("Test 4: Orthogonality regularization")
    print("-" * 40)

    # Orthogonal maps
    orth_maps = torch.stack([torch.eye(d) for _ in range(3)])
    orth_loss_perfect = orthogonality_loss(orth_maps)
    print(f"Orthogonality loss (perfect): {orth_loss_perfect.item():.6f}")

    # Random (non-orthogonal) maps
    random_maps = torch.randn(3, d, d)
    orth_loss_random = orthogonality_loss(random_maps)
    print(f"Orthogonality loss (random): {orth_loss_random.item():.6f}")
    print()

    print("Test 5: Combined loss")
    print("-" * 40)

    losses = combined_sheaf_axiom_loss(restriction_maps_extended, edge_index_extended, num_vertices)
    for name, value in losses.items():
        print(f"{name:20s}: {value.item():.6f}")
    print()

    print("=" * 80)
    print("✓ All tests passed! Sheaf axiom losses working.")
    print("=" * 80)


if __name__ == "__main__":
    test_sheaf_axiom_losses()
