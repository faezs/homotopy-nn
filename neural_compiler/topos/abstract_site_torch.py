"""
Abstract Learnable Site for Meta-Learning (PyTorch)

Implements the abstract Site from evolutionary_solver.py in PyTorch.
This is NOT tied to a specific grid - it's a learnable categorical structure.

Key difference from geometric_morphism_torch.Site:
- geometric_morphism_torch.Site: Fixed grid, geometric adjacency
- This Site: Abstract objects, learnable coverage

For meta-learning, we need the abstract version so the universal topos
can learn what "coverage" means across different tasks.

Author: Claude Code + Human
Date: October 22, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class AbstractSite:
    """Abstract site (C, J) with learnable Grothendieck topology.

    Components:
    - Category C: objects + morphisms (adjacency)
    - Topology J: Coverage families (which neighborhoods cover which objects)

    Attributes:
        num_objects: Number of objects in category
        feature_dim: Dimension of object embeddings
        max_covers: Maximum covering families per object
        adjacency: (num_objects, num_objects) morphism structure
        object_features: (num_objects, feature_dim) learnable embeddings
        coverage_weights: (num_objects, max_covers, num_objects) LEARNABLE coverage

    The key insight: coverage_weights are LEARNED, not geometric!
    The model discovers which neighborhoods matter for each task.
    """
    num_objects: int
    feature_dim: int
    max_covers: int
    adjacency: torch.Tensor  # (num_objects, num_objects)
    object_features: torch.Tensor  # (num_objects, feature_dim)
    coverage_weights: torch.Tensor  # (num_objects, max_covers, num_objects)

    @staticmethod
    def random(
        num_objects: int,
        feature_dim: int,
        max_covers: int,
        sparsity: float = 0.3,
        device='cpu'
    ) -> 'AbstractSite':
        """Initialize random abstract site.

        Args:
            num_objects: Number of objects
            feature_dim: Feature dimension
            max_covers: Max covering families per object
            sparsity: Morphism sparsity (lower = fewer connections)
            device: torch device

        Returns:
            Random abstract site
        """
        # Random sparse adjacency
        adj_random = torch.rand(num_objects, num_objects, device=device)
        adjacency = (adj_random < sparsity).float()

        # Random object features
        object_features = torch.randn(num_objects, feature_dim, device=device)

        # Random coverage weights (softmax over last dim)
        coverage_logits = torch.randn(num_objects, max_covers, num_objects, device=device)
        coverage_weights = F.softmax(coverage_logits, dim=-1)

        return AbstractSite(
            num_objects=num_objects,
            feature_dim=feature_dim,
            max_covers=max_covers,
            adjacency=adjacency,
            object_features=object_features,
            coverage_weights=coverage_weights
        )

    def get_covers(self, object_idx: int) -> torch.Tensor:
        """Get covering families for an object.

        Args:
            object_idx: Object index

        Returns:
            (max_covers, num_objects) coverage weights
        """
        return self.coverage_weights[object_idx]

    def to(self, device):
        """Move site to device."""
        return AbstractSite(
            num_objects=self.num_objects,
            feature_dim=self.feature_dim,
            max_covers=self.max_covers,
            adjacency=self.adjacency.to(device),
            object_features=self.object_features.to(device),
            coverage_weights=self.coverage_weights.to(device)
        )


class LearnableSite(nn.Module):
    """Learnable site as nn.Module (for gradient updates).

    This wraps AbstractSite with Parameters so we can train it.
    """

    def __init__(
        self,
        num_objects: int,
        feature_dim: int,
        max_covers: int,
        sparsity: float = 0.3
    ):
        super().__init__()

        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.max_covers = max_covers

        # Adjacency (learnable or fixed)
        adj_random = torch.rand(num_objects, num_objects)
        adjacency = (adj_random < sparsity).float()
        self.adjacency = nn.Parameter(adjacency, requires_grad=False)  # Usually fixed

        # Object features (learnable)
        self.object_features = nn.Parameter(
            torch.randn(num_objects, feature_dim)
        )

        # Coverage weights (learnable!)
        # This is the key: model learns what coverage means
        coverage_logits = torch.randn(num_objects, max_covers, num_objects)
        self.coverage_logits = nn.Parameter(coverage_logits)

    def forward(self) -> AbstractSite:
        """Get current site state.

        Returns:
            AbstractSite with current parameter values
        """
        coverage_weights = F.softmax(self.coverage_logits, dim=-1)

        return AbstractSite(
            num_objects=self.num_objects,
            feature_dim=self.feature_dim,
            max_covers=self.max_covers,
            adjacency=self.adjacency,
            object_features=self.object_features,
            coverage_weights=coverage_weights
        )

    def get_covers(self, object_idx: int) -> torch.Tensor:
        """Get covering families for object."""
        coverage_weights = F.softmax(self.coverage_logits, dim=-1)
        return coverage_weights[object_idx]


class AdaptiveSite(nn.Module):
    """Adaptive site that modulates based on task embedding.

    For meta-learning: base site + task-specific adaptation.
    """

    def __init__(
        self,
        num_objects: int,
        feature_dim: int,
        max_covers: int,
        task_embedding_dim: int = 64
    ):
        super().__init__()

        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.max_covers = max_covers
        self.task_embedding_dim = task_embedding_dim

        # Base site (shared across tasks)
        self.base_site = LearnableSite(num_objects, feature_dim, max_covers)

        # Adaptation network: task_embedding → coverage modulation
        self.adaptation_net = nn.Sequential(
            nn.Linear(task_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_objects * max_covers * num_objects),
            nn.Tanh()  # Bounded modulation
        )

    def forward(self, task_embedding: torch.Tensor) -> AbstractSite:
        """Adapt base site to task.

        Args:
            task_embedding: (task_embedding_dim,) task embedding

        Returns:
            Task-adapted site
        """
        # Get base site
        base_site = self.base_site()

        # Compute coverage modulation
        modulation = self.adaptation_net(task_embedding)
        modulation = modulation.view(self.num_objects, self.max_covers, self.num_objects)

        # Adapt coverage: base + modulation, then renormalize
        adapted_coverage_logits = self.base_site.coverage_logits + 0.1 * modulation
        adapted_coverage_weights = F.softmax(adapted_coverage_logits, dim=-1)

        # Optionally adapt object features too
        # For now, keep them fixed

        return AbstractSite(
            num_objects=base_site.num_objects,
            feature_dim=base_site.feature_dim,
            max_covers=base_site.max_covers,
            adjacency=base_site.adjacency,
            object_features=base_site.object_features,
            coverage_weights=adapted_coverage_weights
        )


################################################################################
# Geometric Topos Site (Size-Invariant with Axiom Constraints)
################################################################################

class GeometricToposSite(nn.Module):
    """Size-invariant site constructor with Grothendieck topology axioms.

    Inductive biases:
    1. Category C: Grid adjacency (geometrically valid by construction)
    2. Topology J: Learned coverage with axiom constraints:
       - Identity: Self always in cover
       - Stability: Based on geometric neighborhoods
       - Transitivity: Enforced via composition

    Key: Network learns HOW to build coverage, not fixed coverage.
    Works on ANY grid size.
    """

    def __init__(self, feature_dim: int = 32, max_covers: int = 5, hidden_dim: int = 64):
        super().__init__()

        self.feature_dim = feature_dim
        self.max_covers = max_covers
        self.hidden_dim = hidden_dim

        # Position encoder: (i, j) → embedding
        self.position_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        # Coverage network: position embedding → coverage parameters
        # Predicts how to weight neighbors for each cover
        self.coverage_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_covers * 4),  # 4 params per cover
            nn.Sigmoid()  # Bounded [0,1]
        )

    def _build_grid_adjacency(self, h: int, w: int, device='cpu') -> torch.Tensor:
        """Build 4-connected grid adjacency."""
        num_objects = h * w
        adj = torch.zeros(num_objects, num_objects, device=device)

        for i in range(h):
            for j in range(w):
                idx = i * w + j

                # 4-connected neighbors
                if i > 0: adj[idx, (i-1)*w + j] = 1.0  # Up
                if i < h-1: adj[idx, (i+1)*w + j] = 1.0  # Down
                if j > 0: adj[idx, i*w + (j-1)] = 1.0  # Left
                if j < w-1: adj[idx, i*w + (j+1)] = 1.0  # Right

        return adj

    def _position_embeddings(self, h: int, w: int, device='cpu') -> torch.Tensor:
        """Generate position embeddings for grid cells."""
        positions = torch.zeros(h * w, 2, device=device)

        for i in range(h):
            for j in range(w):
                idx = i * w + j
                # Normalized positions
                positions[idx, 0] = i / max(h - 1, 1)
                positions[idx, 1] = j / max(w - 1, 1)

        # Encode positions
        embeddings = self.position_net(positions)
        return embeddings

    def _get_geometric_neighbors(self, idx: int, h: int, w: int) -> list:
        """Get 4-connected neighbors of a cell."""
        i, j = idx // w, idx % w
        neighbors = []

        if i > 0: neighbors.append((i-1)*w + j)
        if i < h-1: neighbors.append((i+1)*w + j)
        if j > 0: neighbors.append(i*w + (j-1))
        if j < w-1: neighbors.append(i*w + (j+1))

        return neighbors

    def _constrained_coverage(
        self,
        params: torch.Tensor,
        idx: int,
        h: int,
        w: int,
        device='cpu'
    ) -> torch.Tensor:
        """Build coverage satisfying Grothendieck axioms.

        Args:
            params: (max_covers * 4,) coverage parameters from network
            idx: Object index
            h, w: Grid dimensions
            device: torch device

        Returns:
            (max_covers, num_objects) coverage weights
        """
        num_objects = h * w
        coverage = torch.zeros(self.max_covers, num_objects, device=device)

        neighbors = self._get_geometric_neighbors(idx, h, w)
        params = params.view(self.max_covers, 4)

        for k in range(self.max_covers):
            # Parameters for this cover
            self_weight = params[k, 0]  # Weight for self
            neighbor_weight = params[k, 1]  # Weight for direct neighbors
            neighbor_neighbor_weight = params[k, 2]  # Weight for 2-hop neighbors
            decay = params[k, 3]  # Distance decay factor

            # AXIOM: Identity - self always in cover
            coverage[k, idx] = self_weight

            # AXIOM: Stability - geometric neighbors
            for n_idx in neighbors:
                coverage[k, n_idx] = neighbor_weight

                # Optional: 2-hop neighbors (transitive closure)
                n_neighbors = self._get_geometric_neighbors(n_idx, h, w)
                for nn_idx in n_neighbors:
                    if nn_idx != idx and nn_idx not in neighbors:
                        coverage[k, nn_idx] += neighbor_neighbor_weight * decay

            # Normalize to probability distribution
            coverage[k] = coverage[k] / (coverage[k].sum() + 1e-8)

        return coverage

    def forward(self, grid_shape: tuple, device='cpu') -> AbstractSite:
        """Construct site for given grid shape.

        Args:
            grid_shape: (height, width)
            device: torch device

        Returns:
            AbstractSite with learned but axiom-satisfying topology
        """
        h, w = grid_shape
        num_objects = h * w

        # 1. Category: Grid adjacency (FIXED, geometrically valid)
        adjacency = self._build_grid_adjacency(h, w, device)

        # 2. Object features: Learned position embeddings
        object_features = self._position_embeddings(h, w, device)

        # 3. Coverage: Learned with constraints
        coverage_weights = torch.zeros(num_objects, self.max_covers, num_objects, device=device)

        for i in range(h):
            for j in range(w):
                idx = i * w + j

                # Get coverage parameters from network
                pos_emb = object_features[idx]
                params = self.coverage_net(pos_emb)

                # Build constrained coverage
                coverage_weights[idx] = self._constrained_coverage(
                    params, idx, h, w, device
                )

        return AbstractSite(
            num_objects=num_objects,
            feature_dim=self.feature_dim,
            max_covers=self.max_covers,
            adjacency=adjacency,
            object_features=object_features,
            coverage_weights=coverage_weights
        )


################################################################################
# Bridge to Grid-Based Site
################################################################################

def grid_to_abstract_site(
    grid_height: int,
    grid_width: int,
    feature_dim: int = 32,
    max_covers: int = 5,
    connectivity: str = "4",
    device='cpu'
) -> AbstractSite:
    """Convert grid-based site to abstract site.

    This allows us to use grid geometry as initialization,
    but with learnable coverage.

    Args:
        grid_height: Grid height
        grid_width: Grid width
        feature_dim: Feature dimension
        max_covers: Max covering families
        connectivity: "4" or "8"
        device: torch device

    Returns:
        AbstractSite initialized from grid geometry
    """
    num_objects = grid_height * grid_width

    # Build adjacency from grid
    adjacency = torch.zeros(num_objects, num_objects, device=device)

    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j

            # Neighbors
            neighbors = []
            if i > 0: neighbors.append((i-1, j))
            if i < grid_height-1: neighbors.append((i+1, j))
            if j > 0: neighbors.append((i, j-1))
            if j < grid_width-1: neighbors.append((i, j+1))

            if connectivity == "8":
                if i > 0 and j > 0: neighbors.append((i-1, j-1))
                if i > 0 and j < grid_width-1: neighbors.append((i-1, j+1))
                if i < grid_height-1 and j > 0: neighbors.append((i+1, j-1))
                if i < grid_height-1 and j < grid_width-1: neighbors.append((i+1, j+1))

            for ni, nj in neighbors:
                nidx = ni * grid_width + nj
                adjacency[idx, nidx] = 1.0

    # Object features (random initialization)
    object_features = torch.randn(num_objects, feature_dim, device=device)

    # Coverage weights: initialize with geometric neighborhoods
    coverage_weights = torch.zeros(num_objects, max_covers, num_objects, device=device)

    for i in range(grid_height):
        for j in range(grid_width):
            idx = i * grid_width + j

            # First cover: neighbors (geometric)
            neighbors = []
            if i > 0: neighbors.append((i-1, j))
            if i < grid_height-1: neighbors.append((i+1, j))
            if j > 0: neighbors.append((i, j-1))
            if j < grid_width-1: neighbors.append((i, j+1))

            # Set uniform weights over neighbors
            for ni, nj in neighbors:
                nidx = ni * grid_width + nj
                coverage_weights[idx, 0, nidx] = 1.0 / (len(neighbors) + 1)

            # Self-cover
            coverage_weights[idx, 0, idx] = 1.0 / (len(neighbors) + 1)

            # Other covers: random (to be learned)
            for k in range(1, max_covers):
                random_weights = torch.rand(num_objects, device=device)
                coverage_weights[idx, k] = F.softmax(random_weights, dim=0)

    return AbstractSite(
        num_objects=num_objects,
        feature_dim=feature_dim,
        max_covers=max_covers,
        adjacency=adjacency,
        object_features=object_features,
        coverage_weights=coverage_weights
    )
