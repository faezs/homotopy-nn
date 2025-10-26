"""
Internal Language of the Topos - Formula DSL

Implements the internal logic of a Grothendieck topos as differentiable formulas.

THEORETICAL FOUNDATION:

From 1Lab (Topoi/Logic/Base.agda) and our formalization (Neural/Stack/Languages.agda):

## Kripke-Joyal Semantics

Truth of formula φ at stage/object U:
```
U ⊩ φ  ("U forces φ")
```

Interpretation rules:
- U ⊩ (φ ∧ ψ) ↔ (U ⊩ φ) ∧ (U ⊩ ψ)
- U ⊩ (φ ∨ ψ) ↔ (U ⊩ φ) ∨ (U ⊩ ψ)
- U ⊩ (¬φ) ↔ ∀(f: V→U). ¬(V ⊩ φ)
- U ⊩ (φ ⇒ ψ) ↔ ∀(f: V→U). (V ⊩ φ) → (V ⊩ ψ)
- U ⊩ (∀x.φ) ↔ ∀(f: V→U). V ⊩ φ[x := f]
- U ⊩ (∃x.φ) ↔ ∃[cover]. ∀(f ∈ cover). V ⊩ φ[x := f_V]

## PyTorch Approximation

We use:
- Smooth logic operators (product t-norms, probabilistic sums)
- Neural networks for atomic predicates
- Differentiable quantifiers (product for ∀, logsumexp for ∃)

Author: Claude Code
Date: October 23, 2025
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


################################################################################
# § 1: Formula Base Classes
################################################################################

class Formula(ABC):
    """Abstract base class for formulas in the internal language.

    All formulas must implement eval() which returns a smooth truth value [0,1].
    """

    @abstractmethod
    def eval(self, context: Dict[str, Any], interpreter: 'KripkeJoyalInterpreter') -> torch.Tensor:
        """Evaluate formula in given context.

        Args:
            context: Variable bindings (e.g., {'cell': tensor_value, 'U': object_index})
            interpreter: Kripke-Joyal interpreter with smooth operators

        Returns:
            Truth value in [0, 1] (differentiable!)
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Human-readable formula representation."""
        pass


################################################################################
# § 2: Variables and Terms
################################################################################

@dataclass
class Var:
    """Variable in the internal language."""
    name: str

    def __str__(self):
        return self.name


@dataclass
class Const:
    """Constant value."""
    value: Any

    def __str__(self):
        return str(self.value)


################################################################################
# § 3: Atomic Formulas (Learned Predicates)
################################################################################

class Atomic(Formula):
    """Atomic formula with learned predicate.

    Examples:
    - color(cell) = red
    - inside(cell)
    - is_square(region)
    """

    def __init__(self, predicate_name: str, args: List[Any], neural_net: Optional[nn.Module] = None):
        """
        Args:
            predicate_name: Name of predicate (e.g., "inside", "color_eq")
            args: Arguments to predicate (variables or constants)
            neural_net: Learned predicate as neural network (returns probabilities [0,1])
        """
        self.predicate_name = predicate_name
        self.args = args
        self.neural_net = neural_net

    def eval(self, context: Dict[str, Any], interpreter) -> torch.Tensor:
        """Evaluate atomic predicate using neural network."""
        # Get neural predicates from interpreter
        if self.neural_net is not None:
            predicate_fn = self.neural_net
        elif self.predicate_name in interpreter.predicates:
            predicate_fn = interpreter.predicates[self.predicate_name]
        else:
            raise ValueError(f"Unknown predicate: {self.predicate_name}")

        # Evaluate arguments in context
        arg_values = []
        for arg in self.args:
            if isinstance(arg, Var):
                arg_values.append(context.get(arg.name))
            elif isinstance(arg, Const):
                arg_values.append(arg.value)
            else:
                arg_values.append(arg)

        # Call neural predicate
        return predicate_fn(*arg_values, context=context)

    def __str__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.predicate_name}({args_str})"


################################################################################
# § 4: Logical Connectives
################################################################################

class And(Formula):
    """Conjunction: φ ∧ ψ

    Kripke-Joyal: U ⊩ (φ ∧ ψ) ↔ (U ⊩ φ) ∧ (U ⊩ ψ)
    Smooth: p * q (product t-norm, fully differentiable)
    """

    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right

    def eval(self, context: Dict[str, Any], interpreter) -> torch.Tensor:
        v1 = self.left.eval(context, interpreter)
        v2 = self.right.eval(context, interpreter)
        return interpreter.omega.conjunction(v1, v2)

    def __str__(self):
        return f"({self.left} ∧ {self.right})"


class Or(Formula):
    """Disjunction: φ ∨ ψ

    Kripke-Joyal: U ⊩ (φ ∨ ψ) ↔ (U ⊩ φ) ∨ (U ⊩ ψ)
    Smooth: p + q - p*q (probabilistic sum, differentiable)
    """

    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right

    def eval(self, context: Dict[str, Any], interpreter) -> torch.Tensor:
        v1 = self.left.eval(context, interpreter)
        v2 = self.right.eval(context, interpreter)
        return interpreter.omega.disjunction(v1, v2)

    def __str__(self):
        return f"({self.left} ∨ {self.right})"


class Not(Formula):
    """Negation: ¬φ

    Kripke-Joyal: U ⊩ (¬φ) ↔ ∀(f: V→U). ¬(V ⊩ φ)
    Smooth: 1 - p (already smooth!)
    """

    def __init__(self, body: Formula):
        self.body = body

    def eval(self, context: Dict[str, Any], interpreter) -> torch.Tensor:
        v = self.body.eval(context, interpreter)
        return interpreter.omega.negation(v)

    def __str__(self):
        return f"¬{self.body}"


class Implies(Formula):
    """Implication: φ ⇒ ψ

    Kripke-Joyal: U ⊩ (φ ⇒ ψ) ↔ ∀(f: V→U). (V ⊩ φ) → (V ⊩ ψ)
    Smooth: max(1-p, q) or equivalently: (1-p) + q - (1-p)*q
    """

    def __init__(self, antecedent: Formula, consequent: Formula):
        self.antecedent = antecedent
        self.consequent = consequent

    def eval(self, context: Dict[str, Any], interpreter) -> torch.Tensor:
        p = self.antecedent.eval(context, interpreter)
        q = self.consequent.eval(context, interpreter)
        return interpreter.omega.implication(p, q)

    def __str__(self):
        return f"({self.antecedent} ⇒ {self.consequent})"


################################################################################
# § 5: Quantifiers
################################################################################

class Forall(Formula):
    """Universal quantification: ∀x. φ

    Kripke-Joyal: U ⊩ (∀x.φ) ↔ ∀(f: V→U). V ⊩ φ[x := f]

    Smooth: Product over all values
    torch.prod([φ[x := v1], φ[x := v2], ...])

    Gradient: ∂(∏ vᵢ)/∂vⱼ = ∏_{i≠j} vᵢ (all values contribute!)
    """

    def __init__(self, var: Var, body: Formula, domain_fn: Optional[Callable] = None):
        """
        Args:
            var: Variable to quantify over
            body: Formula with free variable
            domain_fn: Function that returns domain values given context
                      (defaults to all grid cells)
        """
        self.var = var
        self.body = body
        self.domain_fn = domain_fn

    def eval(self, context: Dict[str, Any], interpreter) -> torch.Tensor:
        # Get domain to quantify over
        if self.domain_fn is not None:
            domain = self.domain_fn(context)
        else:
            # Default: all objects in site
            domain = range(context.get('num_objects', 1))

        # Evaluate body for each value in domain
        values = []
        for value in domain:
            new_context = context.copy()
            new_context[self.var.name] = value
            v = self.body.eval(new_context, interpreter)
            values.append(v)

        if len(values) == 0:
            return torch.tensor(1.0)  # Vacuous truth

        # Smooth universal: product (differentiable!)
        return torch.prod(torch.stack(values))

    def __str__(self):
        return f"∀{self.var}. {self.body}"


class Exists(Formula):
    """Existential quantification: ∃x. φ

    Kripke-Joyal: U ⊩ (∃x.φ) ↔ ∃[cover]. ∀(f ∈ cover). V ⊩ φ

    Smooth: Logsumexp (differentiable approximation of max)
    torch.logsumexp([φ[x := v1], φ[x := v2], ...])

    Gradient flows to all values, with more weight to larger values
    """

    def __init__(self, var: Var, body: Formula, domain_fn: Optional[Callable] = None):
        """
        Args:
            var: Variable to quantify over
            body: Formula with free variable
            domain_fn: Function that returns domain values given context
        """
        self.var = var
        self.body = body
        self.domain_fn = domain_fn

    def eval(self, context: Dict[str, Any], interpreter) -> torch.Tensor:
        # Get domain
        if self.domain_fn is not None:
            domain = self.domain_fn(context)
        else:
            domain = range(context.get('num_objects', 1))

        # Evaluate body for each value
        values = []
        for value in domain:
            new_context = context.copy()
            new_context[self.var.name] = value
            v = self.body.eval(new_context, interpreter)
            values.append(v)

        if len(values) == 0:
            return torch.tensor(0.0)  # No witnesses

        # Smooth existential: logsumexp (differentiable approximation of max)
        # Normalize to [0,1] range
        stacked = torch.stack(values)
        return torch.sigmoid(torch.logsumexp(stacked, dim=0))

    def __str__(self):
        return f"∃{self.var}. {self.body}"


################################################################################
# § 6: Assignment (for transformations)
################################################################################

class Assign(Formula):
    """Assignment: x := value

    Used in transformations like "color(cell) := red"
    Returns 1.0 (always true) but has side effect of modifying context
    """

    def __init__(self, var: Var, value: Any):
        self.var = var
        self.value = value

    def eval(self, context: Dict[str, Any], interpreter) -> torch.Tensor:
        # Evaluate value
        if isinstance(self.value, Formula):
            val = self.value.eval(context, interpreter)
        elif isinstance(self.value, Var):
            val = context.get(self.value.name)
        elif isinstance(self.value, Const):
            val = self.value.value
        else:
            val = self.value

        # Store in context (side effect!)
        context[f"assigned_{self.var.name}"] = val

        # Assignment always "succeeds" (returns true)
        return torch.tensor(1.0)

    def __str__(self):
        return f"{self.var} := {self.value}"


################################################################################
# § 7: Convenience Functions
################################################################################

def atom(predicate: str, *args) -> Atomic:
    """Shorthand for creating atomic formulas."""
    return Atomic(predicate, list(args))


def forall(var_name: str, body: Formula, domain_fn=None) -> Forall:
    """Shorthand for universal quantification."""
    return Forall(Var(var_name), body, domain_fn)


def exists(var_name: str, body: Formula, domain_fn=None) -> Exists:
    """Shorthand for existential quantification."""
    return Exists(Var(var_name), body, domain_fn)


def assign(var_name: str, value: Any) -> Assign:
    """Shorthand for assignment."""
    return Assign(Var(var_name), value)


# Infix operators for cleaner syntax
def and_(φ: Formula, ψ: Formula) -> And:
    return And(φ, ψ)


def or_(φ: Formula, ψ: Formula) -> Or:
    return Or(φ, ψ)


def not_(φ: Formula) -> Not:
    return Not(φ)


def implies(φ: Formula, ψ: Formula) -> Implies:
    return Implies(φ, ψ)


################################################################################
# § 8: Example Formulas
################################################################################

# Example: "All cells are red"
# ∀cell. color(cell) = red
example_all_red = forall("cell", atom("color_eq", Var("cell"), Const("red")))

# Example: "If cell is inside, then fill with blue"
# ∀cell. inside(cell) ⇒ color(cell) := blue
example_fill_interior = forall(
    "cell",
    implies(
        atom("inside", Var("cell")),
        assign("color", Const("blue"))
    )
)

# Example: "There exists a 3×3 square"
# ∃region. is_square(region) ∧ size(region) = 3
example_find_square = exists(
    "region",
    and_(
        atom("is_square", Var("region")),
        atom("size_eq", Var("region"), Const(3))
    )
)
