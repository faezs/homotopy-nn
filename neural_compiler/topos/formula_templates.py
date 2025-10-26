"""
Formula Templates for ARC Tasks

Library of common transformation patterns expressed as formulas in the
internal language of the topos.

THEORETICAL FOUNDATION:

ARC tasks follow recurring patterns that can be expressed compositionally
using the internal logic:

**Pattern categories**:
1. **Color transformations**: Change colors based on conditions
2. **Geometric transformations**: Rotate, reflect, translate
3. **Topological transformations**: Fill holes, extract boundaries
4. **Relational transformations**: Copy patterns, extend structures

Each pattern is a formula template with holes (parameters) that can be:
- Filled by neural search (Gumbel-Softmax selection)
- Learned end-to-end (differentiable formulas)
- Composed into complex programs

The key insight: Instead of learning f: Grid → Grid as black box,
we learn f = ⟦formula⟧ where formula is interpretable and composable.

Author: Claude Code
Date: October 23, 2025
"""

import torch
from typing import List, Callable, Dict, Any
from dataclasses import dataclass

from internal_language import (
    Formula, Var, Const,
    atom, forall, exists, implies, and_, or_, not_, assign
)


################################################################################
# § 1: Color Transformation Templates
################################################################################

def fill_if(condition_pred: str, target_color: int) -> Formula:
    """Template: Fill cells satisfying condition with target color.

    Formula: ∀cell. condition(cell) ⇒ color(cell) := target_color

    Examples:
    - fill_if("is_inside", 2) → Fill interior with red
    - fill_if("is_boundary", 0) → Clear boundary
    """
    return forall(
        "cell",
        implies(
            atom(condition_pred, Var("cell")),
            assign("color", Const(target_color))
        )
    )


def recolor_matching(source_color: int, target_color: int) -> Formula:
    """Template: Recolor all cells of source_color to target_color.

    Formula: ∀cell. color(cell) = source ⇒ color(cell) := target
    """
    return forall(
        "cell",
        implies(
            atom("color_eq", Var("cell"), Const(source_color)),
            assign("color", Const(target_color))
        )
    )


def conditional_recolor(
    condition_pred: str,
    source_color: int,
    target_color: int
) -> Formula:
    """Template: Recolor source→target only if condition holds.

    Formula: ∀cell. (condition(cell) ∧ color(cell) = source) ⇒ color(cell) := target
    """
    return forall(
        "cell",
        implies(
            and_(
                atom(condition_pred, Var("cell")),
                atom("color_eq", Var("cell"), Const(source_color))
            ),
            assign("color", Const(target_color))
        )
    )


def copy_color_from_neighbor(direction: str) -> Formula:
    """Template: Copy color from neighboring cell.

    Formula: ∀cell. color(cell) := color(neighbor(cell, direction))

    Args:
        direction: "left", "right", "up", "down"
    """
    return forall(
        "cell",
        assign(
            "color",
            atom(f"neighbor_color_{direction}", Var("cell"))
        )
    )


################################################################################
# § 2: Geometric Transformation Templates
################################################################################

def reflect_horizontal() -> Formula:
    """Template: Reflect grid horizontally.

    Formula: ∀cell. color(cell) := color(reflect_h(cell))
    """
    return forall(
        "cell",
        assign("color", atom("reflected_color_h", Var("cell")))
    )


def reflect_vertical() -> Formula:
    """Template: Reflect grid vertically.

    Formula: ∀cell. color(cell) := color(reflect_v(cell))
    """
    return forall(
        "cell",
        assign("color", atom("reflected_color_v", Var("cell")))
    )


def rotate_90() -> Formula:
    """Template: Rotate grid 90° clockwise.

    Formula: ∀cell. color(cell) := color(rotate(cell, 90))
    """
    return forall(
        "cell",
        assign("color", atom("rotated_color_90", Var("cell")))
    )


def translate(dx: int, dy: int) -> Formula:
    """Template: Translate pattern by (dx, dy).

    Formula: ∀cell. color(cell) := color(cell + (dx, dy))
    """
    return forall(
        "cell",
        assign("color", atom("translated_color", Var("cell"), Const(dx), Const(dy)))
    )


################################################################################
# § 3: Topological Transformation Templates
################################################################################

def fill_interior(region_pred: str, fill_color: int) -> Formula:
    """Template: Fill interior of region.

    Formula: ∀cell. (¬boundary(cell) ∧ in_region(cell)) ⇒ color(cell) := fill_color
    """
    return forall(
        "cell",
        implies(
            and_(
                not_(atom("is_boundary", Var("cell"))),
                atom(region_pred, Var("cell"))
            ),
            assign("color", Const(fill_color))
        )
    )


def extract_boundary(region_pred: str, boundary_color: int) -> Formula:
    """Template: Extract boundary of region.

    Formula: ∀cell. boundary_of_region(cell) ⇒ color(cell) := boundary_color
    """
    return forall(
        "cell",
        implies(
            atom(f"{region_pred}_boundary", Var("cell")),
            assign("color", Const(boundary_color))
        )
    )


def remove_interior(region_pred: str) -> Formula:
    """Template: Remove interior (keep only boundary).

    Formula: ∀cell. (¬boundary_of_region(cell) ∧ in_region(cell)) ⇒ color(cell) := 0
    """
    return forall(
        "cell",
        implies(
            and_(
                not_(atom(f"{region_pred}_boundary", Var("cell"))),
                atom(region_pred, Var("cell"))
            ),
            assign("color", Const(0))
        )
    )


################################################################################
# § 4: Pattern Matching Templates
################################################################################

def if_pattern_then_color(pattern_pred: str, target_color: int) -> Formula:
    """Template: If cell matches pattern, apply color.

    Formula: ∀cell. matches_pattern(cell) ⇒ color(cell) := target_color
    """
    return forall(
        "cell",
        implies(
            atom(pattern_pred, Var("cell")),
            assign("color", Const(target_color))
        )
    )


def extend_line(direction: str, color: int) -> Formula:
    """Template: Extend lines in given direction.

    Formula: ∀cell. (is_line_end(cell) ∧ direction_clear(cell)) ⇒
                    extend_color(cell, direction, color)
    """
    return forall(
        "cell",
        implies(
            and_(
                atom("is_line_end", Var("cell")),
                atom(f"clear_{direction}", Var("cell"))
            ),
            atom(f"extend_{direction}", Var("cell"), Const(color))
        )
    )


def complete_symmetry(axis: str) -> Formula:
    """Template: Complete symmetric pattern.

    Formula: ∀cell. ¬has_symmetric_pair(cell, axis) ⇒
                    create_symmetric_pair(cell, axis)
    """
    return forall(
        "cell",
        implies(
            not_(atom(f"has_pair_{axis}", Var("cell"))),
            atom(f"create_pair_{axis}", Var("cell"))
        )
    )


################################################################################
# § 5: Composite Templates (Sequences)
################################################################################

@dataclass
class SequentialFormula(Formula):
    """Sequence of formulas applied in order.

    Each formula transforms the grid, passing result to next.

    For now, we evaluate only the last step (sequential composition
    requires iterative grid transformation which is handled at a higher level).
    """
    steps: List[Formula]

    def eval(self, context: Dict[str, Any], interpreter) -> torch.Tensor:
        """Evaluate sequential formula.

        For simplicity, we just evaluate the last step.
        Full sequential evaluation would require grid-level iteration.
        """
        if len(self.steps) == 0:
            return torch.tensor(1.0, device=interpreter.device)

        # Evaluate last step (approximation for now)
        return self.steps[-1].eval(context, interpreter)

    def __str__(self):
        return " ; ".join(str(step) for step in self.steps)


def two_step_transform(step1: Formula, step2: Formula) -> SequentialFormula:
    """Template: Apply two transformations sequentially."""
    return SequentialFormula([step1, step2])


def three_step_transform(step1: Formula, step2: Formula, step3: Formula) -> SequentialFormula:
    """Template: Apply three transformations sequentially."""
    return SequentialFormula([step1, step2, step3])


################################################################################
# § 6: Meta-Templates (Parameterized Families)
################################################################################

class TemplateFamily:
    """Family of related templates with parameters.

    Enables neural search over template parameters.
    """

    def __init__(self, name: str, generator: Callable):
        self.name = name
        self.generator = generator

    def instantiate(self, *params) -> Formula:
        """Instantiate template with specific parameters."""
        return self.generator(*params)


# Example families
FILL_FAMILIES = TemplateFamily(
    "fill_if",
    lambda cond, color: fill_if(cond, color)
)

RECOLOR_FAMILIES = TemplateFamily(
    "recolor",
    lambda src, tgt: recolor_matching(src, tgt)
)

GEOMETRIC_FAMILIES = TemplateFamily(
    "geometric",
    lambda transform: {
        "reflect_h": reflect_horizontal(),
        "reflect_v": reflect_vertical(),
        "rotate_90": rotate_90(),
    }[transform]
)


################################################################################
# § 7: Template Library
################################################################################

class TemplateLibrary:
    """Curated library of formula templates for ARC tasks.

    Organizes templates by category for easy lookup and composition.
    """

    def __init__(self):
        self.templates: Dict[str, Formula] = {}
        self.families: Dict[str, TemplateFamily] = {}

        self._register_templates()
        self._register_families()

    def _register_templates(self):
        """Register all predefined templates."""

        # Color transformations
        self.templates['fill_inside_red'] = fill_if("is_inside", 2)
        self.templates['fill_boundary_black'] = fill_if("is_boundary", 0)
        self.templates['clear_inside'] = fill_if("is_inside", 0)

        # Geometric
        self.templates['reflect_h'] = reflect_horizontal()
        self.templates['reflect_v'] = reflect_vertical()
        self.templates['rotate_90'] = rotate_90()

        # Topological
        self.templates['extract_boundary_red'] = extract_boundary("region", 2)
        self.templates['remove_interior'] = remove_interior("region")

        # Composite examples
        self.templates['boundary_then_fill'] = two_step_transform(
            extract_boundary("region", 2),
            fill_interior("region", 1)
        )

    def _register_families(self):
        """Register template families for parameterized search."""
        self.families['fill_if'] = FILL_FAMILIES
        self.families['recolor'] = RECOLOR_FAMILIES
        self.families['geometric'] = GEOMETRIC_FAMILIES

    def get(self, name: str) -> Formula:
        """Look up template by name."""
        if name not in self.templates:
            raise ValueError(f"Unknown template: {name}")
        return self.templates[name]

    def get_family(self, name: str) -> TemplateFamily:
        """Look up template family by name."""
        if name not in self.families:
            raise ValueError(f"Unknown family: {name}")
        return self.families[name]

    def search_space_size(self) -> int:
        """Total number of templates in library."""
        return len(self.templates)

    def all_template_names(self) -> List[str]:
        """Get list of all template names."""
        return list(self.templates.keys())


################################################################################
# § 8: Template Enumeration (for Search)
################################################################################

def enumerate_color_transforms(num_colors: int = 10) -> List[Formula]:
    """Enumerate all color transformation templates.

    Returns list of formulas for:
    - fill_if(cond, color) for all conditions × colors
    - recolor(src, tgt) for all src × tgt pairs
    """
    templates = []

    conditions = ["is_inside", "is_boundary", "is_corner"]
    for cond in conditions:
        for color in range(num_colors):
            templates.append(fill_if(cond, color))

    for src in range(num_colors):
        for tgt in range(num_colors):
            if src != tgt:
                templates.append(recolor_matching(src, tgt))

    return templates


def enumerate_geometric_transforms() -> List[Formula]:
    """Enumerate all geometric transformations."""
    return [
        reflect_horizontal(),
        reflect_vertical(),
        rotate_90(),
        translate(1, 0),   # Right
        translate(-1, 0),  # Left
        translate(0, 1),   # Down
        translate(0, -1),  # Up
    ]


def enumerate_composite_transforms(
    base_templates: List[Formula],
    max_depth: int = 2
) -> List[Formula]:
    """Enumerate composite templates up to max_depth.

    Args:
        base_templates: List of atomic templates
        max_depth: Maximum composition depth

    Returns:
        All compositions up to max_depth
    """
    if max_depth == 1:
        return base_templates

    composites = []

    # Depth 2: pairs
    for t1 in base_templates:
        for t2 in base_templates:
            composites.append(two_step_transform(t1, t2))

    if max_depth >= 3:
        # Depth 3: triples
        for t1 in base_templates:
            for t2 in base_templates:
                for t3 in base_templates:
                    composites.append(three_step_transform(t1, t2, t3))

    return composites


################################################################################
# § 9: Example Usage
################################################################################

if __name__ == "__main__":
    """
    Demonstrate template library usage.
    """

    library = TemplateLibrary()

    print("=== Formula Template Library ===\n")

    print("Predefined templates:")
    for name in library.all_template_names():
        template = library.get(name)
        print(f"  {name}: {template}")

    print(f"\nTotal predefined: {library.search_space_size()}")

    print("\n=== Template Families ===\n")

    # Instantiate from family
    fill_family = library.get_family('fill_if')
    custom_template = fill_family.instantiate("is_corner", 5)
    print(f"Custom: {custom_template}")

    print("\n=== Enumerated Search Space ===\n")

    color_templates = enumerate_color_transforms(num_colors=4)
    print(f"Color transforms (4 colors): {len(color_templates)}")

    geom_templates = enumerate_geometric_transforms()
    print(f"Geometric transforms: {len(geom_templates)}")

    composite_templates = enumerate_composite_transforms(geom_templates[:3], max_depth=2)
    print(f"Composite transforms (depth 2): {len(composite_templates)}")

    total_search_space = len(color_templates) + len(geom_templates) + len(composite_templates)
    print(f"\nTotal search space size: {total_search_space}")

    print("\n✓ Template library initialized successfully!")
