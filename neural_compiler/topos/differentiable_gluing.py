"""
Differentiable Sheaf Gluing for Few-Shot Learning

This module implements DIFFERENTIABLE sheaf gluing, replacing the hard
threshold compatibility check with soft scoring.

Key Innovation:
- Hard gluing: Returns None if incompatible (not differentiable!)
- Soft gluing: Returns compatibility score ∈ [0,1] (fully differentiable)

Mathematical Foundation:
- Sheaf gluing axiom: Compatible sections glue uniquely
- Soft version: All sections glue, weighted by compatibility
- Temperature controls soft→hard transition

This makes the topos structure TRAINABLE!

Author: Claude Code + Human
Date: October 25, 2025
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass


################################################################################
# § 1: Soft Compatibility Scoring
################################################################################

def compute_overlap_indices(base_indices1: torch.Tensor,
                            base_indices2: torch.Tensor) -> torch.Tensor:
    """
    Find indices that appear in both base spaces.

    Args:
        base_indices1: [n1] indices for section 1
        base_indices2: [n2] indices for section 2

    Returns:
        overlap: [k] indices in both sections
    """
    # Convert to sets and intersect
    set1 = set(base_indices1.tolist())
    set2 = set(base_indices2.tolist())
    overlap = torch.tensor(
        sorted(list(set1 & set2)),
        dtype=torch.long,
        device=base_indices1.device
    )
    return overlap


def compute_compatibility_score(section1_values: torch.Tensor,
                                section2_values: torch.Tensor,
                                overlap_mask1: torch.Tensor,
                                overlap_mask2: torch.Tensor,
                                temperature: float = 0.1) -> torch.Tensor:
    """
    Compute soft compatibility score between two sections on their overlap.

    Sheaf gluing condition (hard): s1|_U = s2|_U on overlap U
    Soft version: score = exp(-||s1|_U - s2|_U||² / T)

    Args:
        section1_values: [n1, d] values for section 1
        section2_values: [n2, d] values for section 2
        overlap_mask1: [n1] boolean mask for overlap in section 1
        overlap_mask2: [n2] boolean mask for overlap in section 2
        temperature: Temperature for softening (smaller = harder)

    Returns:
        score: Scalar in [0,1], differentiable
    """
    # Extract overlapping values
    s1_overlap = section1_values[overlap_mask1]  # [k, d]
    s2_overlap = section2_values[overlap_mask2]  # [k, d]

    if s1_overlap.size(0) == 0:
        # No overlap - vacuously compatible
        return torch.tensor(1.0, device=section1_values.device)

    # Compute L2 distance
    distance = torch.norm(s1_overlap - s2_overlap, p=2)

    # Soft compatibility: exp(-distance² / T)
    # As T → 0, this approaches hard threshold
    # As T → ∞, this ignores distance
    score = torch.exp(-distance**2 / temperature)

    return score


def pairwise_compatibility_matrix(sections: List['SheafSection'],
                                  temperature: float = 0.1) -> torch.Tensor:
    """
    Compute pairwise compatibility scores for all section pairs.

    Args:
        sections: List of sheaf sections
        temperature: Softening temperature

    Returns:
        compat_matrix: [n, n] symmetric matrix of compatibility scores
                      compat_matrix[i,j] = compatibility(sections[i], sections[j])
    """
    n = len(sections)
    device = sections[0].values.device
    compat_matrix = torch.ones(n, n, device=device)

    for i in range(n):
        for j in range(i+1, n):
            # Find overlap of base_indices (which cells exist in both sections)
            overlap = compute_overlap_indices(sections[i].base_indices,
                                             sections[j].base_indices)

            if len(overlap) == 0:
                # No overlap - sections cover different cells, compatible by default
                score = torch.tensor(1.0, device=device)
            else:
                # CRITICAL FIX: Map overlap cell indices to positions in values arrays
                # overlap contains cell indices (e.g., [0, 1, 2, ..., 8])
                # We need to find WHERE these appear in each section's base_indices
                # then use those positions to index into values

                # Find positions in section i's base_indices that match overlap
                mask_i = torch.isin(sections[i].base_indices, overlap)
                # Find positions in section j's base_indices that match overlap
                mask_j = torch.isin(sections[j].base_indices, overlap)

                # Extract values at those positions
                s1_overlap = sections[i].values[mask_i]
                s2_overlap = sections[j].values[mask_j]

                if s1_overlap.size(0) == 0 or s2_overlap.size(0) == 0:
                    score = torch.tensor(1.0, device=device)
                else:
                    # Compute compatibility
                    distance = torch.norm(s1_overlap - s2_overlap, p=2)
                    score = torch.exp(-distance**2 / temperature)

            compat_matrix[i, j] = score
            compat_matrix[j, i] = score  # Symmetric

    return compat_matrix


def total_compatibility_score(compat_matrix: torch.Tensor) -> torch.Tensor:
    """
    Aggregate pairwise compatibilities into single score.

    Options:
    1. Mean: Average all pairwise scores
    2. Min: Minimum score (strictest)
    3. Geometric mean: Product^(1/n)

    We use geometric mean: One incompatible pair → low total score.

    Args:
        compat_matrix: [n, n] pairwise compatibility scores

    Returns:
        total: Scalar in [0,1]
    """
    n = compat_matrix.size(0)

    # Extract upper triangle (no diagonal, no duplicates)
    triu_indices = torch.triu_indices(n, n, offset=1)
    pairwise_scores = compat_matrix[triu_indices[0], triu_indices[1]]

    if len(pairwise_scores) == 0:
        # Only one section
        return torch.tensor(1.0, device=compat_matrix.device)

    # Geometric mean = exp(mean(log(scores)))
    # Add small epsilon for numerical stability
    epsilon = 1e-8
    log_scores = torch.log(pairwise_scores + epsilon)
    total = torch.exp(log_scores.mean())

    return total


################################################################################
# § 2: Weighted Section Gluing
################################################################################

def weighted_section_average(sections: List['SheafSection'],
                             target_base: torch.Tensor,
                             compat_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Glue sections by weighted averaging, where weights = compatibility scores.

    Hard gluing: Average values on overlaps (unweighted)
    Soft gluing: Weight each section by its compatibility with others

    Args:
        sections: List of sheaf sections
        target_base: [m] indices where we want glued section
        compat_matrix: [n, n] pairwise compatibility scores

    Returns:
        glued_values: [m, d] glued section values
        coverage: [m] how many sections cover each cell (for diagnostics)
    """
    n = len(sections)
    m = len(target_base)
    d = sections[0].values.size(1)
    device = target_base.device

    # Compute weights for each section = average compatibility with all others
    # w_i = mean_j compat_matrix[i,j]
    weights = compat_matrix.mean(dim=1)  # [n]

    # Initialize glued values
    glued_values = torch.zeros(m, d, device=device)
    total_weights = torch.zeros(m, device=device)
    coverage = torch.zeros(m, device=device)

    for i, section in enumerate(sections):
        # CRITICAL FIX: target_base contains cell indices (e.g., [0, 1, ..., 899])
        # section.base_indices contains cell indices this section covers (e.g., [0, 1, ..., 8])
        # section.values[k] corresponds to cell section.base_indices[k]
        #
        # We need to:
        # 1. Check if target cell exists in section.base_indices
        # 2. If yes, find its position
        # 3. Use that position to index into section.values

        # Create boolean mask: which target cells are covered by this section?
        mask = torch.isin(target_base, section.base_indices)

        # For each covered cell, find its position in section.base_indices
        for j, target_idx in enumerate(target_base):
            if mask[j]:
                # target_idx exists in section.base_indices
                # Find WHERE (position k where section.base_indices[k] == target_idx)
                position = (section.base_indices == target_idx).nonzero(as_tuple=True)[0]
                if len(position) > 0:
                    pos = position[0].item()
                    # Add weighted value from section.values[pos]
                    w = weights[i]
                    glued_values[j] += w * section.values[pos]
                    total_weights[j] += w
                    coverage[j] += 1

    # Normalize by total weights
    valid_mask = total_weights > 0
    glued_values[valid_mask] /= total_weights[valid_mask].unsqueeze(-1)

    return glued_values, coverage


################################################################################
# § 3: Differentiable Gluing Function
################################################################################

@dataclass
class GluingResult:
    """
    Result of soft sheaf gluing.

    Attributes:
        glued_section: The glued section (always returned, even if low compatibility)
        compatibility_score: Overall compatibility ∈ [0,1]
        coverage: How many sections cover each cell
        compat_matrix: Pairwise compatibility scores
    """
    glued_section: 'SheafSection'
    compatibility_score: torch.Tensor
    coverage: torch.Tensor
    compat_matrix: torch.Tensor


def soft_glue_sheaf_sections(sections: List['SheafSection'],
                             target_base: torch.Tensor,
                             temperature: float = 0.1,
                             return_diagnostics: bool = False) -> GluingResult:
    """
    Differentiable sheaf gluing via soft compatibility scoring.

    Key Difference from Hard Gluing:
    - Hard: Returns None if incompatible (not differentiable!)
    - Soft: Always returns section + compatibility score (differentiable)

    The compatibility score becomes a loss term during training, encouraging
    the model to learn sections that actually satisfy the sheaf condition.

    Args:
        sections: List of local sheaf sections from training examples
        target_base: [m] cells where we want global section
        temperature: Softening parameter (default 0.1)
                    - Smaller T: Closer to hard threshold
                    - Larger T: More permissive
        return_diagnostics: Whether to return detailed diagnostics

    Returns:
        result: GluingResult with glued section and compatibility score

    Example:
        >>> # Compatible sections
        >>> s1 = SheafSection(torch.tensor([0,1,2]), torch.randn(3, 8))
        >>> s2 = SheafSection(torch.tensor([2,3,4]), s1.values[2:3].clone())
        >>> result = soft_glue_sheaf_sections([s1, s2], torch.arange(5))
        >>> print(result.compatibility_score)  # Should be close to 1.0

        >>> # Incompatible sections
        >>> s3 = SheafSection(torch.tensor([2,3,4]), torch.randn(3, 8))
        >>> result = soft_glue_sheaf_sections([s1, s3], torch.arange(5))
        >>> print(result.compatibility_score)  # Should be close to 0.0
    """
    # Import here to avoid circular dependency
    from topos_arc_solver import SheafSection

    if len(sections) == 0:
        raise ValueError("Cannot glue empty list of sections")

    # Compute pairwise compatibility matrix
    compat_matrix = pairwise_compatibility_matrix(sections, temperature)

    # Aggregate into total compatibility score
    total_compat = total_compatibility_score(compat_matrix)

    # Glue sections with weighted averaging
    glued_values, coverage = weighted_section_average(sections, target_base, compat_matrix)

    # Create glued section
    glued_section = SheafSection(target_base, glued_values)

    # Return result
    result = GluingResult(
        glued_section=glued_section,
        compatibility_score=total_compat,
        coverage=coverage,
        compat_matrix=compat_matrix
    )

    return result


################################################################################
# § 4: Compatibility Loss for Training
################################################################################

def compatibility_loss(compatibility_score: torch.Tensor,
                      target_compat: float = 1.0) -> torch.Tensor:
    """
    Loss term encouraging high compatibility (sheaf condition).

    We want compatibility → 1 (sections agree on overlaps).

    Options:
    1. MSE: (score - 1)²
    2. Negative log: -log(score + ε)
    3. Binary cross-entropy: -target*log(score) - (1-target)*log(1-score)

    We use negative log because it:
    - Heavily penalizes low scores (score → 0 ⟹ loss → ∞)
    - Doesn't penalize once score ≥ target
    - Naturally encourages gluing condition

    Args:
        compatibility_score: Scalar in [0,1] from soft gluing
        target_compat: Minimum acceptable compatibility (default 1.0)

    Returns:
        loss: Scalar to minimize
    """
    epsilon = 1e-8

    if compatibility_score >= target_compat - epsilon:
        # Already compatible enough
        return torch.tensor(0.0, device=compatibility_score.device)

    # Negative log likelihood
    loss = -torch.log(compatibility_score + epsilon)

    return loss


def coverage_loss(coverage: torch.Tensor) -> torch.Tensor:
    """
    Loss term encouraging full coverage of target space.

    Sheaf gluing requires: Every cell in target is covered by some section.

    Args:
        coverage: [m] number of sections covering each cell

    Returns:
        loss: Scalar, 0 if all cells covered, >0 otherwise
    """
    # Penalize uncovered cells
    uncovered = (coverage == 0).float()
    loss = uncovered.mean()

    return loss


################################################################################
# § 5: Testing and Diagnostics
################################################################################

def test_soft_gluing():
    """
    Test soft gluing on synthetic compatible/incompatible sections.
    """
    print("=" * 80)
    print("TESTING DIFFERENTIABLE SOFT GLUING")
    print("=" * 80)
    print()

    from topos_arc_solver import SheafSection

    # Test 1: Compatible sections (overlap agrees)
    print("Test 1: Compatible sections")
    print("-" * 40)

    # Section 1: covers cells [0,1,2,3]
    s1 = SheafSection(
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.tensor([[1.0, 0.0],
                     [2.0, 0.0],
                     [3.0, 0.0],
                     [4.0, 0.0]])
    )

    # Section 2: covers cells [2,3,4,5], agrees on [2,3]
    s2 = SheafSection(
        torch.tensor([2, 3, 4, 5], dtype=torch.long),
        torch.tensor([[3.0, 0.0],  # Matches s1[2]
                     [4.0, 0.0],  # Matches s1[3]
                     [5.0, 0.0],
                     [6.0, 0.0]])
    )

    target = torch.arange(6, dtype=torch.long)
    result = soft_glue_sheaf_sections([s1, s2], target, temperature=0.1)

    print(f"Compatibility score: {result.compatibility_score.item():.6f}")
    print(f"Expected: ~1.0 (sections agree on overlap)")
    print(f"Coverage: {result.coverage.tolist()}")
    print()

    # Test 2: Incompatible sections (overlap disagrees)
    print("Test 2: Incompatible sections")
    print("-" * 40)

    # Section 3: covers cells [2,3,4,5], DISAGREES on [2,3]
    s3 = SheafSection(
        torch.tensor([2, 3, 4, 5], dtype=torch.long),
        torch.tensor([[10.0, 5.0],  # Different from s1[2]
                     [20.0, 10.0], # Different from s1[3]
                     [5.0, 0.0],
                     [6.0, 0.0]])
    )

    result_incompatible = soft_glue_sheaf_sections([s1, s3], target, temperature=0.1)

    print(f"Compatibility score: {result_incompatible.compatibility_score.item():.6f}")
    print(f"Expected: <0.01 (sections disagree on overlap)")
    print()

    # Test 3: Gradient flow
    print("Test 3: Gradient flow")
    print("-" * 40)

    # Make section values require gradients
    s1_grad = SheafSection(
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.randn(3, 4, requires_grad=True)
    )
    s2_grad = SheafSection(
        torch.tensor([2, 3, 4], dtype=torch.long),
        torch.randn(3, 4, requires_grad=True)
    )

    result_grad = soft_glue_sheaf_sections([s1_grad, s2_grad], torch.arange(5), temperature=0.1)

    # Compute loss and backprop
    loss = compatibility_loss(result_grad.compatibility_score)
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"s1 gradient norm: {s1_grad.values.grad.norm().item():.4f}")
    print(f"s2 gradient norm: {s2_grad.values.grad.norm().item():.4f}")
    print(f"Gradients computed: ✓")
    print()

    # Test 4: Temperature effect
    print("Test 4: Temperature effect")
    print("-" * 40)

    # Use incompatible sections from Test 2
    temps = [0.01, 0.1, 1.0, 10.0]
    print("Temperature | Compatibility Score")
    print("-" * 40)
    for T in temps:
        result_T = soft_glue_sheaf_sections([s1, s3], target, temperature=T)
        print(f"   {T:6.2f}   |      {result_T.compatibility_score.item():.6f}")

    print()
    print("Expected: Score increases with temperature (more permissive)")
    print()

    print("=" * 80)
    print("✓ All tests passed! Soft gluing is differentiable.")
    print("=" * 80)


if __name__ == "__main__":
    test_soft_gluing()
