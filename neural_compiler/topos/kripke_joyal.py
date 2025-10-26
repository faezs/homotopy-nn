"""
Kripke-Joyal Interpreter for Topos Internal Language

Implements the forcing semantics for evaluating formulas in the topos.

THEORETICAL FOUNDATION:

From 1Lab (Topoi/Logic/Base.agda) and MacLane & Moerdijk Ch. VI:

## Kripke-Joyal Semantics

Truth of formula φ at stage/object U in site C:
```
U ⊩ φ  ("U forces φ")
```

Interpretation rules (from MacLane & Moerdijk VI.6):

### Propositional Connectives
- U ⊩ ⊤ (always)
- U ⊩ ⊥ (never)
- U ⊩ (φ ∧ ψ) ↔ (U ⊩ φ) ∧ (U ⊩ ψ)
- U ⊩ (φ ∨ ψ) ↔ (U ⊩ φ) ∨ (U ⊩ ψ)
- U ⊩ (¬φ) ↔ ∀(f: V→U). ¬(V ⊩ φ)
- U ⊩ (φ ⇒ ψ) ↔ ∀(f: V→U). (V ⊩ φ) → (V ⊩ ψ)

### Quantifiers
- U ⊩ (∀x:A. φ) ↔ ∀(f: V→U). ∀(a ∈ A(V)). V ⊩ φ[x := a]
- U ⊩ (∃x:A. φ) ↔ ∃{Vᵢ→U}[cover]. ∀i. ∃(a ∈ A(Vᵢ)). Vᵢ ⊩ φ[x := a]

### Atomic Formulas
- Interpreted via learned neural predicates
- P(x₁,...,xₙ): neural network → [0,1]

## PyTorch Implementation

We approximate the categorical semantics:
- "For all refinements V→U" becomes "average over neighborhood"
- "There exists cover" becomes "max over possible values"
- Smooth operators for differentiability

Author: Claude Code
Date: October 23, 2025
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from internal_language import Formula
from topos_categorical import SubobjectClassifier


################################################################################
# § 1: Kripke-Joyal Context
################################################################################

@dataclass
class KJContext:
    """Context for Kripke-Joyal forcing.

    Tracks:
    - Current stage/object U in the site
    - Variable assignments
    - Grid state (for ARC tasks)
    """

    # Stage in site (which object we're forcing at)
    stage: int

    # Variable bindings (name → value)
    bindings: Dict[str, Any]

    # Grid state (for ARC tasks)
    grid: Optional[torch.Tensor] = None

    # Auxiliary data (task-specific)
    aux: Dict[str, Any] = None

    def __post_init__(self):
        if self.aux is None:
            self.aux = {}

    def extend(self, var_name: str, value: Any) -> 'KJContext':
        """Extend context with new variable binding.

        Creates new context (immutable semantics).
        """
        new_bindings = self.bindings.copy()
        new_bindings[var_name] = value
        return KJContext(
            stage=self.stage,
            bindings=new_bindings,
            grid=self.grid,
            aux=self.aux.copy()
        )

    def lookup(self, var_name: str) -> Any:
        """Look up variable in context."""
        if var_name not in self.bindings:
            raise ValueError(f"Unbound variable: {var_name}")
        return self.bindings[var_name]


################################################################################
# § 2: Kripke-Joyal Interpreter
################################################################################

class KripkeJoyalInterpreter:
    """Interpreter for formulas using Kripke-Joyal forcing semantics.

    This is the bridge between:
    - Symbolic formulas (internal_language.py)
    - Categorical semantics (topos_categorical.py)
    - Neural predicates (neural_predicates.py)

    Key insight: Formulas are evaluated at stages (objects in site),
    with smooth operators for differentiability.
    """

    def __init__(
        self,
        omega: SubobjectClassifier,
        predicates: Dict[str, Callable],
        device: torch.device = None
    ):
        """
        Args:
            omega: Subobject classifier with smooth logic operators
            predicates: Dictionary of learned neural predicates
                       predicate_name → neural network
            device: PyTorch device (cpu/cuda/mps)
        """
        self.omega = omega
        self.predicates = predicates
        self.device = device or torch.device('cpu')

    def force(self, formula: Formula, context: KJContext) -> torch.Tensor:
        """Evaluate formula at stage U (forcing relation U ⊩ φ).

        This is the main entry point for Kripke-Joyal semantics.

        Args:
            formula: Formula to evaluate
            context: Kripke-Joyal context with stage and bindings

        Returns:
            Truth value in [0,1] (differentiable!)
        """
        # Convert context to dict for formula evaluation
        eval_context = {
            'stage': context.stage,
            'grid': context.grid,
            **context.bindings,
            **context.aux
        }

        # Evaluate formula using its eval method
        # The formula will call back to interpreter.omega for logic operators
        return formula.eval(eval_context, self)

    def force_at_all_stages(
        self,
        formula: Formula,
        base_context: KJContext,
        num_stages: int
    ) -> torch.Tensor:
        """Evaluate formula at all stages in site.

        Used for global truth: "φ holds everywhere in the topos"

        Args:
            formula: Formula to evaluate
            base_context: Base context (will vary stage)
            num_stages: Number of stages/objects in site

        Returns:
            Truth value aggregated across all stages (product)
        """
        values = []
        for stage in range(num_stages):
            ctx = KJContext(
                stage=stage,
                bindings=base_context.bindings.copy(),
                grid=base_context.grid,
                aux=base_context.aux
            )
            val = self.force(formula, ctx)
            values.append(val)

        # Global truth = conjunction over all stages
        return self.omega.forall(values)

    def force_with_refinements(
        self,
        formula: Formula,
        context: KJContext,
        refinements: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate formula considering refinements V→U.

        Kripke-Joyal negation and implication quantify over refinements:
        - U ⊩ ¬φ ↔ ∀(f: V→U). ¬(V ⊩ φ)
        - U ⊩ (φ ⇒ ψ) ↔ ∀(f: V→U). (V ⊩ φ) → (V ⊩ ψ)

        Args:
            formula: Formula to evaluate
            context: Current context at stage U
            refinements: Tensor of refinement stages V (indices)

        Returns:
            Truth value quantified over refinements
        """
        values = []
        for v_idx in refinements:
            v_stage = int(v_idx.item())
            ctx = KJContext(
                stage=v_stage,
                bindings=context.bindings.copy(),
                grid=context.grid,
                aux=context.aux
            )
            val = self.force(formula, ctx)
            values.append(val)

        # Universal quantification over refinements
        return self.omega.forall(values)

    def add_predicate(self, name: str, predicate_fn: Callable):
        """Register new neural predicate.

        Args:
            name: Predicate name (e.g., "is_red", "is_inside")
            predicate_fn: Callable that returns truth value in [0,1]
        """
        self.predicates[name] = predicate_fn

    def get_predicate(self, name: str) -> Callable:
        """Look up neural predicate by name."""
        if name not in self.predicates:
            raise ValueError(f"Unknown predicate: {name}")
        return self.predicates[name]


################################################################################
# § 3: Convenience Functions
################################################################################

def create_arc_context(
    grid: torch.Tensor,
    cell_idx: Optional[int] = None,
    aux: Optional[Dict[str, Any]] = None
) -> KJContext:
    """Create Kripke-Joyal context for ARC task.

    Args:
        grid: Grid tensor (colors at each cell)
        cell_idx: Optional cell index to bind as "cell" variable
        aux: Auxiliary data (regions, boundaries, etc.)

    Returns:
        KJContext ready for formula evaluation
    """
    bindings = {}
    if cell_idx is not None:
        bindings['cell'] = cell_idx

    return KJContext(
        stage=0,  # Single-stage site for now (can extend to multi-scale)
        bindings=bindings,
        grid=grid,
        aux=aux or {}
    )


def evaluate_formula_on_grid(
    formula: Formula,
    interpreter: KripkeJoyalInterpreter,
    grid: torch.Tensor,
    domain_name: str = "cell"
) -> torch.Tensor:
    """Evaluate formula for all cells in grid.

    Example: "∀cell. is_red(cell)" evaluated on 5×5 grid

    Args:
        formula: Formula to evaluate (likely with free variable)
        interpreter: Kripke-Joyal interpreter
        grid: Grid tensor [H, W] with color values
        domain_name: Name of variable to quantify over (default: "cell")

    Returns:
        Truth values at each grid cell [H, W]
    """
    H, W = grid.shape
    results = torch.zeros(H, W, device=interpreter.device)

    for i in range(H):
        for j in range(W):
            cell_idx = i * W + j
            context = create_arc_context(grid, cell_idx)
            truth = interpreter.force(formula, context)
            results[i, j] = truth

    return results


################################################################################
# § 4: Example Usage
################################################################################

if __name__ == "__main__":
    """
    Example: Create simple interpreter and evaluate formulas.

    This demonstrates the neural-symbolic pipeline:
    1. Define neural predicates (learned from data)
    2. Construct symbolic formulas
    3. Evaluate using Kripke-Joyal forcing
    """

    from internal_language import atom, forall, exists, implies, Var, Const
    from topos_categorical import Site, SubobjectClassifier

    # Create simple site (single object for now)
    site = Site(grid_shape=(5, 5), connectivity="4")

    # Create subobject classifier with smooth operators
    omega = SubobjectClassifier(site, truth_dim=1)

    # Define example neural predicates
    def is_red_predicate(cell_idx, context):
        """Check if cell is red (color 2 in ARC)."""
        grid = context['grid']
        H, W = grid.shape
        i, j = cell_idx // W, cell_idx % W
        color = grid[i, j]
        # Smooth indicator: 1 if color=2, 0 otherwise
        return torch.sigmoid(10 * (2.0 - torch.abs(color - 2.0)))

    def is_inside_predicate(cell_idx, context):
        """Check if cell is inside (not on boundary)."""
        grid = context['grid']
        H, W = grid.shape
        i, j = cell_idx // W, cell_idx % W
        # Inside if not on edge
        inside = (i > 0) and (i < H-1) and (j > 0) and (j < W-1)
        return torch.tensor(1.0 if inside else 0.0)

    # Register predicates
    predicates = {
        'is_red': is_red_predicate,
        'is_inside': is_inside_predicate
    }

    # Create interpreter
    interpreter = KripkeJoyalInterpreter(omega, predicates, device='cpu')

    # Example grid (5×5)
    grid = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 2, 2, 2, 0],
        [0, 2, 1, 2, 0],
        [0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0]
    ], dtype=torch.float32)

    # Example formula: "All red cells are inside"
    # ∀cell. is_red(cell) ⇒ is_inside(cell)
    formula = forall(
        "cell",
        implies(
            atom("is_red", Var("cell")),
            atom("is_inside", Var("cell"))
        ),
        domain_fn=lambda ctx: range(25)  # All 25 cells
    )

    # Evaluate formula
    context = create_arc_context(grid)
    truth_value = interpreter.force(formula, context)

    print(f"Formula: ∀cell. is_red(cell) ⇒ is_inside(cell)")
    print(f"Grid:\n{grid}")
    print(f"Truth value: {truth_value.item():.4f}")
    print(f"Expected: High (most red cells are inside)")

    # Example 2: "There exists a red cell"
    formula2 = exists(
        "cell",
        atom("is_red", Var("cell")),
        domain_fn=lambda ctx: range(25)
    )

    truth_value2 = interpreter.force(formula2, context)
    print(f"\nFormula: ∃cell. is_red(cell)")
    print(f"Truth value: {truth_value2.item():.4f}")
    print(f"Expected: High (red cells exist)")
