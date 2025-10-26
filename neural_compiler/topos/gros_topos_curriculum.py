"""
Gros Topos Curriculum Learning for ARC-AGI

Implements Urs Schreiber's Gros topos framework where:
- Petit topos: Sheaves over ONE grid (Sh(G))
- Gros topos: Sheaves over the CATEGORY of all grids (Sh(GridCat))

Key Insight:
An ARC task Input → Output is NOT just a function between grids,
it's a SHEAF MORPHISM in the Gros topos!

    T: Sh(GridCat_in) → Sh(GridCat_out)

Curriculum learning = learning in the 2-category of sheaf morphisms:
- Objects: Sheaf morphisms (ARC transforms)
- Morphisms: Natural transformations (curriculum steps)
- 2-Morphisms: Modifications (learning updates)

References:
- Urs Schreiber, "Higher Topos Theory in Physics" (2013)
- Lurie, "Higher Topos Theory" (2009)
- Belfiore & Bennequin, "Topos and Stacks of Deep Neural Networks" (2022)

Author: Claude Code + Human
Date: October 22, 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from abc import ABC, abstractmethod


# ============================================================================
# § 1: Gros Topos - Category of Sites
# ============================================================================

@dataclass
class GridSite:
    """A site in the Gros topos.

    In Gros topos, we don't work with a SINGLE space, but with a CATEGORY
    of spaces. Each grid configuration is a site.

    Coverage: Neighborhoods in the grid (Moore/Von Neumann)
    """
    height: int
    width: int
    num_colors: int = 10

    def __post_init__(self):
        self.size = (self.height, self.width)

    def coverage(self, i: int, j: int, radius: int = 1) -> List[Tuple[int, int]]:
        """Coverage at position (i, j) = Moore neighborhood."""
        neighbors = []
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    neighbors.append((ni, nj))
        return neighbors

    def __hash__(self):
        return hash((self.height, self.width, self.num_colors))

    def __eq__(self, other):
        return (self.height == other.height and
                self.width == other.width and
                self.num_colors == other.num_colors)


class GridCategory:
    """Category of all grid sites.

    This is the base category for the Gros topos Sh(GridCat).

    Objects: GridSite instances (all possible grid configurations)
    Morphisms: Grid embeddings (padding, cropping)
    """

    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.sites = {}  # Cache of GridSite objects

    def site(self, height: int, width: int, num_colors: int = 10) -> GridSite:
        """Get or create a GridSite."""
        key = (height, width, num_colors)
        if key not in self.sites:
            self.sites[key] = GridSite(height, width, num_colors)
        return self.sites[key]

    def morphism_exists(self, src: GridSite, tgt: GridSite) -> bool:
        """Check if morphism exists (embedding or projection)."""
        # Morphism exists if src can embed into tgt (padding)
        # or tgt can embed into src (cropping)
        return (src.height <= tgt.height and src.width <= tgt.width) or \
               (tgt.height <= src.height and tgt.width <= src.width)


# ============================================================================
# § 2: Sheaf Morphisms (ARC Transforms)
# ============================================================================

class TransformComplexity(Enum):
    """Curriculum levels based on transform complexity."""
    TRIVIAL = 0      # Identity, copy
    GEOMETRIC = 1    # Rotation, reflection, translation
    COLOR = 2        # Color substitution, inversion
    TOPOLOGICAL = 3  # Fill, connect, boundary
    PATTERN = 4      # Repetition, tiling, scaling
    RELATIONAL = 5   # Spatial relations, above/below
    COMPOSITIONAL = 6  # Multi-step reasoning
    ABSTRACT = 7     # Conceptual, high-level rules

    def __lt__(self, other):
        return self.value < other.value


@dataclass
class SheafMorphism:
    """A sheaf morphism in the Gros topos.

    This represents an ARC task: Input grid → Output grid

    NOT just a function f: Grid_in → Grid_out
    BUT a natural transformation η: F_in ⇒ F_out between sheaf functors!

    F_in: GridCat^op → Set  (input sheaf functor)
    F_out: GridCat^op → Set (output sheaf functor)
    η: F_in ⇒ F_out  (the transform we learn!)
    """
    task_id: str
    name: str
    input_examples: List[np.ndarray]
    output_examples: List[np.ndarray]
    test_inputs: List[np.ndarray]
    test_outputs: List[np.ndarray]

    # Topos structure
    source_sites: List[GridSite] = field(default_factory=list)
    target_sites: List[GridSite] = field(default_factory=list)

    # Curriculum metadata
    complexity: TransformComplexity = TransformComplexity.COMPOSITIONAL
    concepts: List[str] = field(default_factory=list)

    # Dataset provenance
    dataset_name: str = ""

    def __post_init__(self):
        """Infer source and target sites from examples."""
        grid_cat = GridCategory()

        for inp in self.input_examples:
            h, w = inp.shape
            self.source_sites.append(grid_cat.site(h, w))

        for out in self.output_examples:
            h, w = out.shape
            self.target_sites.append(grid_cat.site(h, w))

    def max_input_size(self) -> Tuple[int, int]:
        """Maximum input grid size."""
        if not self.input_examples:
            return (0, 0)
        heights = [g.shape[0] for g in self.input_examples + self.test_inputs]
        widths = [g.shape[1] for g in self.input_examples + self.test_inputs]
        return (max(heights), max(widths))

    def max_output_size(self) -> Tuple[int, int]:
        """Maximum output grid size."""
        if not self.output_examples:
            return (0, 0)
        heights = [g.shape[0] for g in self.output_examples + self.test_outputs]
        widths = [g.shape[1] for g in self.output_examples + self.test_outputs]
        return (max(heights), max(widths))


# ============================================================================
# § 3: Curriculum as 2-Category
# ============================================================================

class CurriculumLevel:
    """A level in the curriculum = collection of sheaf morphisms.

    In 2-category theory:
    - 0-cells: Sheaf morphisms (ARC tasks)
    - 1-cells: Natural transformations (skill transfers)
    - 2-cells: Modifications (learning updates)
    """

    def __init__(
        self,
        level: int,
        complexity: TransformComplexity,
        tasks: List[SheafMorphism]
    ):
        self.level = level
        self.complexity = complexity
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __repr__(self):
        return f"Level {self.level} ({self.complexity.name}): {len(self.tasks)} tasks"


class GrosToposCurriculum:
    """Curriculum learning in the Gros topos.

    The curriculum is organized as a 2-category:
    - Objects: Curriculum levels (collections of sheaf morphisms)
    - Morphisms: Skill transfers between levels
    - 2-Morphisms: Learning updates

    Training progresses through levels of increasing complexity.
    """

    def __init__(self):
        self.levels: List[CurriculumLevel] = []
        self.all_tasks: List[SheafMorphism] = []
        self.task_index: Dict[str, SheafMorphism] = {}

    def add_task(self, task: SheafMorphism):
        """Add a task to the curriculum."""
        self.all_tasks.append(task)
        self.task_index[task.task_id] = task

    def organize_by_complexity(self):
        """Organize tasks into curriculum levels by complexity."""
        # Group by complexity
        by_complexity: Dict[TransformComplexity, List[SheafMorphism]] = {}

        for task in self.all_tasks:
            if task.complexity not in by_complexity:
                by_complexity[task.complexity] = []
            by_complexity[task.complexity].append(task)

        # Create curriculum levels
        self.levels = []
        for i, (complexity, tasks) in enumerate(sorted(by_complexity.items())):
            level = CurriculumLevel(i, complexity, tasks)
            self.levels.append(level)

    def get_level(self, level_idx: int) -> Optional[CurriculumLevel]:
        """Get curriculum level by index."""
        if 0 <= level_idx < len(self.levels):
            return self.levels[level_idx]
        return None

    def summary(self) -> str:
        """Summary of curriculum structure."""
        lines = ["=" * 70]
        lines.append("GROS TOPOS CURRICULUM - Sheaf Morphisms")
        lines.append("=" * 70)
        lines.append(f"Total tasks: {len(self.all_tasks)}")
        lines.append(f"Curriculum levels: {len(self.levels)}")
        lines.append("")

        for level in self.levels:
            lines.append(f"  {level}")

        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# § 4: Dataset Loaders
# ============================================================================

def load_arc_agi_1(data_path: str = "~/ARC-AGI/data") -> List[SheafMorphism]:
    """Load ARC-AGI-1 dataset as sheaf morphisms."""
    data_path = Path(data_path).expanduser()
    tasks = []

    for split in ["training", "evaluation"]:
        split_dir = data_path / split
        if not split_dir.exists():
            continue

        for json_file in split_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)

            task_id = f"agi1-{split}-{json_file.stem}"

            input_examples = [np.array(pair['input'], dtype=np.int32)
                            for pair in data['train']]
            output_examples = [np.array(pair['output'], dtype=np.int32)
                             for pair in data['train']]
            test_inputs = [np.array(pair['input'], dtype=np.int32)
                          for pair in data['test']]
            test_outputs = [np.array(pair['output'], dtype=np.int32)
                           for pair in data['test']]

            task = SheafMorphism(
                task_id=task_id,
                name=json_file.stem,
                input_examples=input_examples,
                output_examples=output_examples,
                test_inputs=test_inputs,
                test_outputs=test_outputs,
                dataset_name="ARC-AGI-1"
            )

            tasks.append(task)

    return tasks


def load_arc_agi_2(data_path: str = "~/ARC-AGI-2/data") -> List[SheafMorphism]:
    """Load ARC-AGI-2 dataset as sheaf morphisms."""
    data_path = Path(data_path).expanduser()
    tasks = []

    for split in ["training", "evaluation"]:
        split_dir = data_path / split
        if not split_dir.exists():
            continue

        for json_file in split_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)

            task_id = f"agi2-{split}-{json_file.stem}"

            input_examples = [np.array(pair['input'], dtype=np.int32)
                            for pair in data['train']]
            output_examples = [np.array(pair['output'], dtype=np.int32)
                             for pair in data['train']]
            test_inputs = [np.array(pair['input'], dtype=np.int32)
                          for pair in data['test']]
            test_outputs = [np.array(pair['output'], dtype=np.int32)
                           for pair in data['test']]

            task = SheafMorphism(
                task_id=task_id,
                name=json_file.stem,
                input_examples=input_examples,
                output_examples=output_examples,
                test_inputs=test_inputs,
                test_outputs=test_outputs,
                dataset_name="ARC-AGI-2"
            )

            tasks.append(task)

    return tasks


def load_concept_arc(data_path: str = "~/ConceptARC/corpus") -> List[SheafMorphism]:
    """Load ConceptARC dataset as sheaf morphisms."""
    data_path = Path(data_path).expanduser()
    tasks = []

    # ConceptARC is organized by concept
    for concept_dir in data_path.iterdir():
        if not concept_dir.is_dir():
            continue

        concept_name = concept_dir.name

        for json_file in concept_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)

            task_id = f"concept-{concept_name}-{json_file.stem}"

            input_examples = [np.array(pair['input'], dtype=np.int32)
                            for pair in data['train']]
            output_examples = [np.array(pair['output'], dtype=np.int32)
                             for pair in data['train']]
            test_inputs = [np.array(pair['input'], dtype=np.int32)
                          for pair in data['test']]
            test_outputs = [np.array(pair['output'], dtype=np.int32)
                           for pair in data['test']]

            task = SheafMorphism(
                task_id=task_id,
                name=json_file.stem,
                input_examples=input_examples,
                output_examples=output_examples,
                test_inputs=test_inputs,
                test_outputs=test_outputs,
                concepts=[concept_name],
                dataset_name="ConceptARC"
            )

            tasks.append(task)

    return tasks


def load_mini_arc(data_path: str = "~/MINI-ARC/data/MiniARC") -> List[SheafMorphism]:
    """Load Mini-ARC dataset (5x5 grids) as sheaf morphisms."""
    data_path = Path(data_path).expanduser()
    tasks = []

    if not data_path.exists():
        return tasks

    for json_file in data_path.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)

        task_id = f"mini-{json_file.stem}"

        input_examples = [np.array(pair['input'], dtype=np.int32)
                        for pair in data['train']]
        output_examples = [np.array(pair['output'], dtype=np.int32)
                         for pair in data['train']]
        test_inputs = [np.array(pair['input'], dtype=np.int32)
                      for pair in data['test']]
        test_outputs = [np.array(pair['output'], dtype=np.int32)
                       for pair in data['test']]

        task = SheafMorphism(
            task_id=task_id,
            name=json_file.stem,
            input_examples=input_examples,
            output_examples=output_examples,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            complexity=TransformComplexity.GEOMETRIC,  # Mini-ARC tends to be simpler
            dataset_name="Mini-ARC"
        )

        tasks.append(task)

    return tasks


# ============================================================================
# § 5: Complexity Estimation
# ============================================================================

class ComplexityEstimator:
    """Estimate transform complexity for curriculum ordering.

    Uses heuristics based on:
    - Grid size changes
    - Color palette changes
    - Spatial transformations
    - Pattern complexity
    """

    @staticmethod
    def estimate(task: SheafMorphism) -> TransformComplexity:
        """Estimate task complexity."""
        if not task.input_examples or not task.output_examples:
            return TransformComplexity.COMPOSITIONAL

        inp = task.input_examples[0]
        out = task.output_examples[0]

        # Check for identity
        if np.array_equal(inp, out):
            return TransformComplexity.TRIVIAL

        # Check for geometric transforms (same size)
        if inp.shape == out.shape:
            # Check rotation/reflection
            if ComplexityEstimator._is_geometric(inp, out):
                return TransformComplexity.GEOMETRIC

            # Check color substitution
            if ComplexityEstimator._is_color_transform(inp, out):
                return TransformComplexity.COLOR

        # Check for pattern repetition (size multiplication)
        if ComplexityEstimator._is_tiling(inp, out):
            return TransformComplexity.PATTERN

        # Check for topological operations
        if ComplexityEstimator._is_topological(inp, out):
            return TransformComplexity.TOPOLOGICAL

        # Default to high complexity
        return TransformComplexity.COMPOSITIONAL

    @staticmethod
    def _is_geometric(inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if transform is geometric (rotation/reflection)."""
        # Check 90° rotations
        for k in [1, 2, 3]:
            if np.array_equal(np.rot90(inp, k), out):
                return True

        # Check flips
        if np.array_equal(np.flip(inp, 0), out) or np.array_equal(np.flip(inp, 1), out):
            return True

        return False

    @staticmethod
    def _is_color_transform(inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if transform is pure color substitution."""
        # Same non-zero pattern, different colors
        inp_mask = inp != 0
        out_mask = out != 0
        return np.array_equal(inp_mask, out_mask)

    @staticmethod
    def _is_tiling(inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if output is tiling of input."""
        h_in, w_in = inp.shape
        h_out, w_out = out.shape

        # Check if output size is multiple of input
        if h_out % h_in == 0 and w_out % w_in == 0:
            # Check if tiles match
            for i in range(0, h_out, h_in):
                for j in range(0, w_out, w_in):
                    tile = out[i:i+h_in, j:j+w_in]
                    if not np.array_equal(tile, inp):
                        return False
            return True

        return False

    @staticmethod
    def _is_topological(inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if transform involves topological operations (fill, connect)."""
        # Heuristic: number of non-zero cells changes significantly
        inp_filled = np.count_nonzero(inp)
        out_filled = np.count_nonzero(out)

        return abs(out_filled - inp_filled) > inp.size * 0.1


# ============================================================================
# § 6: Main - Build Curriculum
# ============================================================================

def build_gros_topos_curriculum() -> GrosToposCurriculum:
    """Build complete curriculum from all datasets."""
    print("=" * 70)
    print("BUILDING GROS TOPOS CURRICULUM")
    print("=" * 70)
    print()

    curriculum = GrosToposCurriculum()
    estimator = ComplexityEstimator()

    # Load all datasets
    print("Loading datasets...")

    print("  Loading ARC-AGI-1...")
    agi1_tasks = load_arc_agi_1()
    print(f"    Loaded {len(agi1_tasks)} tasks")

    print("  Loading ARC-AGI-2...")
    agi2_tasks = load_arc_agi_2()
    print(f"    Loaded {len(agi2_tasks)} tasks")

    print("  Loading ConceptARC...")
    concept_tasks = load_concept_arc()
    print(f"    Loaded {len(concept_tasks)} tasks")

    print("  Loading Mini-ARC...")
    mini_tasks = load_mini_arc()
    print(f"    Loaded {len(mini_tasks)} tasks")

    print()
    print(f"Total: {len(agi1_tasks) + len(agi2_tasks) + len(concept_tasks) + len(mini_tasks)} tasks")
    print()

    # Estimate complexity for all tasks
    print("Estimating complexity...")
    all_tasks = agi1_tasks + agi2_tasks + concept_tasks + mini_tasks

    for task in all_tasks:
        if task.dataset_name != "ConceptARC":  # ConceptARC has known concepts
            task.complexity = estimator.estimate(task)
        curriculum.add_task(task)

    # Organize into curriculum levels
    print("Organizing curriculum...")
    curriculum.organize_by_complexity()

    print()
    print(curriculum.summary())

    return curriculum


if __name__ == "__main__":
    curriculum = build_gros_topos_curriculum()

    print()
    print("Curriculum ready for training!")
    print()
    print("Next steps:")
    print("  1. Train on Level 0 (TRIVIAL) - warm-up")
    print("  2. Progress through levels (GEOMETRIC → COLOR → ...)")
    print("  3. Learn sheaf morphism representations")
    print("  4. Generate NEW transforms synthetically (Gros topos objects!)")
