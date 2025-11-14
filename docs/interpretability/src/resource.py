"""
Neural Interpretability via Resource Functors

Core resource theory implementation based on:
- src/Neural/Resources.agda
- src/Neural/Resources/Convertibility.agda
- src/Neural/Resources/Optimization.agda
"""

from typing import Protocol, TypeVar, Generic, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

# Type variables
R = TypeVar('R')  # Resource type
S = TypeVar('S')  # Semiring type


class SemiringProtocol(Protocol):
    """A semiring (R, +, *, 0, 1) with order ≥"""

    def add(self, other: 'SemiringProtocol') -> 'SemiringProtocol':
        """Additive operation (parallel composition)"""
        ...

    def mul(self, other: 'SemiringProtocol') -> 'SemiringProtocol':
        """Multiplicative operation (sequential composition)"""
        ...

    @property
    def zero(self) -> 'SemiringProtocol':
        """Additive identity"""
        ...

    @property
    def one(self) -> 'SemiringProtocol':
        """Multiplicative identity"""
        ...

    def __ge__(self, other: 'SemiringProtocol') -> bool:
        """Preorder relation (convertibility)"""
        ...


class Resource(ABC, Generic[R]):
    """
    A resource in a symmetric monoidal category.

    Based on Definition (Section 3.2): Resource Theory
    - Objects are resources (activations, weights, information)
    - Morphisms are conversions (structure-preserving transformations)
    """

    @abstractmethod
    def tensor(self, other: 'Resource[R]') -> 'Resource[R]':
        """
        Parallel composition A ⊗ B

        Represents independent resources composed in parallel.
        """
        pass

    @abstractmethod
    def compose(self, other: 'Resource[R]') -> 'Resource[R]':
        """
        Sequential composition A ◦ B

        Represents resource transformation (B after A).
        """
        pass

    @abstractmethod
    def is_convertible_to(self, other: 'Resource[R]') -> bool:
        """
        Convertibility preorder A ⪰ B

        Returns True if resource A can be converted to resource B.
        """
        pass

    @property
    @abstractmethod
    def identity(self) -> 'Resource[R]':
        """Identity resource (neutral element)"""
        pass


@dataclass
class PreorderedMonoid:
    """
    A preordered monoid (R, +, ⪰, 0) from isomorphism classes.

    Based on Section 3.2.2: The preordered monoid is obtained from
    the symmetric monoidal category by taking isomorphism classes
    under the tensor product.
    """

    elements: list  # Elements of the monoid
    addition: Callable  # Binary operation + : R × R → R
    order: Callable  # Preorder ⪰ : R × R → Bool
    zero: object  # Neutral element

    def __post_init__(self):
        """Verify monoid axioms"""
        # Could add runtime checks for:
        # - Associativity: (a + b) + c = a + (b + c)
        # - Identity: a + 0 = a = 0 + a
        # - Preorder: reflexive, transitive
        # - Compatibility: a ⪰ b ⟹ a + c ⪰ b + c
        pass


class ConversionRate:
    """
    Conversion rate ρ_{A→B} between resources.

    Definition (Section 3.2.2):
        ρ_{A→B} = sup { m/n | n·A ⪰ m·B }

    Interpretation: Maximum ratio m/n such that n copies of A
    can be converted to m copies of B.
    """

    def __init__(self, source: Resource, target: Resource):
        self.source = source
        self.target = target
        self._rate_cache = None

    def compute(self, max_copies: int = 100) -> float:
        """
        Compute conversion rate via bulk conversion.

        Args:
            max_copies: Maximum number of copies to test

        Returns:
            ρ_{A→B} = sup { m/n }
        """
        if self._rate_cache is not None:
            return self._rate_cache

        max_rate = 0.0

        # Try all n ∈ [1, max_copies]
        for n in range(1, max_copies + 1):
            # Tensor power: A⊗n
            n_copies_source = self._tensor_power(self.source, n)

            # Find maximum m such that n·A ⪰ m·B
            m = self._find_max_m(n_copies_source, self.target, max_copies)

            if m > 0:
                rate = m / n
                max_rate = max(max_rate, rate)

        self._rate_cache = max_rate
        return max_rate

    def _tensor_power(self, resource: Resource, n: int) -> Resource:
        """Compute A⊗n (n copies in parallel)"""
        result = resource.identity
        for _ in range(n):
            result = result.tensor(resource)
        return result

    def _find_max_m(self, n_source: Resource, target: Resource,
                    max_m: int) -> int:
        """Find maximum m such that n_source ⪰ m·target"""
        for m in range(max_m, 0, -1):
            m_target = self._tensor_power(target, m)
            if n_source.is_convertible_to(m_target):
                return m
        return 0


class MeasuringHomomorphism(ABC):
    """
    An S-measuring homomorphism M: R → S.

    Definition (Section 3.2.2):
    A monoid homomorphism M: (R, +, ⪰, 0) → (S, +, ≥, 0) such that:
    1. M(A + B) = M(A) + M(B)  (homomorphism)
    2. M(0) = 0                 (preserves zero)
    3. A ⪰ B ⟹ M(A) ≥ M(B)    (preserves order)

    Examples:
    - Energy measure (power consumption)
    - Entropy measure (information content)
    - Parameter count
    - FLOPs (computational cost)
    """

    @abstractmethod
    def measure(self, resource: Resource) -> float:
        """Apply the measuring homomorphism M(A)"""
        pass

    def verify_homomorphism(self, a: Resource, b: Resource) -> bool:
        """Verify M(A ⊗ B) = M(A) + M(B)"""
        lhs = self.measure(a.tensor(b))
        rhs = self.measure(a) + self.measure(b)
        return np.isclose(lhs, rhs)

    def verify_order_preserving(self, a: Resource, b: Resource) -> bool:
        """Verify A ⪰ B ⟹ M(A) ≥ M(B)"""
        if a.is_convertible_to(b):
            return self.measure(a) >= self.measure(b)
        return True  # Vacuously true if not convertible


class EntropyMeasure(MeasuringHomomorphism):
    """
    Shannon entropy as a measuring homomorphism.

    H(p) = -Σ p_i log(p_i)

    Properties:
    - H(p ⊗ q) = H(p) + H(q) for independent distributions
    - H(0) = 0 (degenerate distribution)
    - More uncertain → higher entropy
    """

    def measure(self, resource: Resource) -> float:
        """Compute Shannon entropy of resource distribution"""
        # Assume resource has a .distribution() method
        if hasattr(resource, 'distribution'):
            p = resource.distribution()
            # Shannon entropy: H(p) = -Σ p_i log(p_i)
            p = p[p > 0]  # Avoid log(0)
            return -np.sum(p * np.log2(p))
        else:
            raise ValueError("Resource must have distribution() method")


class ParameterCountMeasure(MeasuringHomomorphism):
    """
    Parameter count as a measuring homomorphism.

    Properties:
    - count(A ⊗ B) = count(A) + count(B)  (parallel adds)
    - count(A ◦ B) = count(A) × count(B)  (sequential multiplies)
    """

    def measure(self, resource: Resource) -> float:
        """Count total parameters in resource"""
        if hasattr(resource, 'param_count'):
            return float(resource.param_count())
        else:
            raise ValueError("Resource must have param_count() method")


class FLOPsMeasure(MeasuringHomomorphism):
    """
    Floating-point operations as a measuring homomorphism.

    Measures computational cost.
    """

    def measure(self, resource: Resource) -> float:
        """Count FLOPs for resource computation"""
        if hasattr(resource, 'flops'):
            return float(resource.flops())
        else:
            raise ValueError("Resource must have flops() method")


def theorem_5_6(rate: ConversionRate, measure: MeasuringHomomorphism) -> bool:
    """
    Verify Theorem 5.6: ρ_{A→B} · M(B) ≤ M(A)

    Args:
        rate: Conversion rate ρ_{A→B}
        measure: Measuring homomorphism M

    Returns:
        True if theorem holds (within numerical tolerance)
    """
    rho = rate.compute()
    M_A = measure.measure(rate.source)
    M_B = measure.measure(rate.target)

    lhs = rho * M_B
    rhs = M_A

    return lhs <= rhs + 1e-6  # Numerical tolerance


# Example usage
if __name__ == "__main__":
    print("Resource Theory for Neural Interpretability")
    print("=" * 50)
    print()
    print("Core concepts:")
    print("1. Resource: Objects in symmetric monoidal category")
    print("2. Conversion rate: ρ_{A→B} = sup { m/n | n·A ⪰ m·B }")
    print("3. Measuring: M: R → ℝ (entropy, params, FLOPs)")
    print("4. Theorem 5.6: ρ_{A→B} · M(B) ≤ M(A)")
    print()
    print("See examples/ for practical applications.")
