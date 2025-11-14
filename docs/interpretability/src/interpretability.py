"""
High-level interpretability API using resource functors.

Provides practical tools for:
- Attention head redundancy detection
- Layer importance analysis
- Information flow tracking
- Minimal circuit discovery
"""

from typing import List, Dict, Optional, Callable, Any
import numpy as np
from dataclasses import dataclass

from .resource import (
    Resource, ConversionRate, MeasuringHomomorphism,
    EntropyMeasure, ParameterCountMeasure, FLOPsMeasure
)


@dataclass
class NeuralResource(Resource):
    """
    A neural network component as a resource.

    Can represent:
    - Attention heads
    - Layers
    - Neurons
    - Activations
    """

    name: str
    activations: np.ndarray  # Activation values
    weights: Optional[np.ndarray] = None  # Weight parameters
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def tensor(self, other: 'NeuralResource') -> 'NeuralResource':
        """Parallel composition (independent resources)"""
        return NeuralResource(
            name=f"{self.name}⊗{other.name}",
            activations=np.concatenate([self.activations, other.activations]),
            weights=None,  # Combined weights not well-defined
            metadata={'composed': [self.name, other.name]}
        )

    def compose(self, other: 'NeuralResource') -> 'NeuralResource':
        """Sequential composition (other after self)"""
        # Apply other's transformation to self's activations
        # Simplified: just pass through
        return NeuralResource(
            name=f"{self.name}◦{other.name}",
            activations=other.activations,  # Simplified
            weights=None,
            metadata={'sequence': [self.name, other.name]}
        )

    def is_convertible_to(self, other: 'NeuralResource') -> bool:
        """
        Check if this resource can be converted to other.

        Heuristic: Based on mutual information between activations.
        """
        # Compute correlation between activation distributions
        if self.activations.shape != other.activations.shape:
            return False

        # Normalized correlation
        corr = np.corrcoef(
            self.activations.flatten(),
            other.activations.flatten()
        )[0, 1]

        # High correlation means convertible
        return abs(corr) > 0.7

    @property
    def identity(self) -> 'NeuralResource':
        """Identity resource (pass-through)"""
        return NeuralResource(
            name="id",
            activations=np.zeros_like(self.activations)
        )

    def distribution(self) -> np.ndarray:
        """Get probability distribution from activations"""
        # Softmax to get probability distribution
        exp_acts = np.exp(self.activations - np.max(self.activations))
        return exp_acts / exp_acts.sum()

    def param_count(self) -> int:
        """Count parameters"""
        if self.weights is not None:
            return self.weights.size
        return 0

    def flops(self) -> int:
        """Estimate FLOPs"""
        if self.weights is not None:
            # Simplified: matrix multiplication FLOPs
            return 2 * self.weights.size
        return 0


class ResourceNetwork:
    """
    A neural network viewed as a collection of resources.

    Provides interpretability tools based on resource theory.
    """

    def __init__(self, resources: List[NeuralResource]):
        self.resources = resources
        self._conversion_cache: Dict[tuple, float] = {}

    @classmethod
    def from_pretrained(cls, model_name: str) -> 'ResourceNetwork':
        """
        Load a pretrained model as resources.

        Args:
            model_name: HuggingFace model name (e.g., "gpt2")

        Returns:
            ResourceNetwork with extracted resources
        """
        # TODO: Implement actual model loading
        # For now, create dummy resources
        print(f"[Mock] Loading {model_name}...")

        dummy_resources = [
            NeuralResource(
                name=f"layer_{i}",
                activations=np.random.randn(512),  # d_model=512
                weights=np.random.randn(512, 512)
            )
            for i in range(12)  # 12 layers
        ]

        return cls(dummy_resources)

    def conversion_rate(self, source_idx: int, target_idx: int) -> float:
        """
        Compute conversion rate ρ_{source → target}.

        Args:
            source_idx: Index of source resource
            target_idx: Index of target resource

        Returns:
            Conversion rate (0.0 to 1.0+)
        """
        key = (source_idx, target_idx)
        if key in self._conversion_cache:
            return self._conversion_cache[key]

        source = self.resources[source_idx]
        target = self.resources[target_idx]

        rate = ConversionRate(source, target).compute()
        self._conversion_cache[key] = rate

        return rate

    def find_redundant_resources(self, threshold: float = 0.85) -> List[tuple]:
        """
        Find pairs of redundant resources via high conversion rates.

        Args:
            threshold: Minimum ρ_{A→B} to consider redundant

        Returns:
            List of (i, j, rate) tuples for redundant pairs
        """
        redundant = []

        for i in range(len(self.resources)):
            for j in range(i + 1, len(self.resources)):
                rate_ij = self.conversion_rate(i, j)
                rate_ji = self.conversion_rate(j, i)

                # Symmetric redundancy
                if rate_ij >= threshold and rate_ji >= threshold:
                    redundant.append((i, j, (rate_ij + rate_ji) / 2))

        return redundant

    def measure_all(self, measure: MeasuringHomomorphism) -> List[float]:
        """
        Apply measuring homomorphism to all resources.

        Args:
            measure: Measuring function (entropy, params, FLOPs)

        Returns:
            List of measurements for each resource
        """
        return [measure.measure(r) for r in self.resources]

    def information_flow(self,
                        source_idx: int,
                        target_idx: int,
                        measure: MeasuringHomomorphism) -> float:
        """
        Measure information flow from source to target.

        Uses conversion rates weighted by measurement.

        Args:
            source_idx: Source resource index
            target_idx: Target resource index
            measure: Measuring homomorphism

        Returns:
            Flow value (higher = more information flow)
        """
        rate = self.conversion_rate(source_idx, target_idx)
        source_measure = measure.measure(self.resources[source_idx])

        # Flow = rate × source information
        return rate * source_measure


def attention_redundancy(model: ResourceNetwork,
                        threshold: float = 0.85) -> np.ndarray:
    """
    Compute redundancy matrix for attention heads.

    Args:
        model: ResourceNetwork representing the model
        threshold: Minimum conversion rate for redundancy

    Returns:
        n×n matrix where entry (i,j) is ρ_{head_i → head_j}
    """
    n = len(model.resources)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = model.conversion_rate(i, j)

    return matrix


def layer_importance(model: ResourceNetwork,
                    measure: MeasuringHomomorphism,
                    output_idx: int = -1) -> Dict[str, float]:
    """
    Measure layer importance via information flow to output.

    Args:
        model: ResourceNetwork
        measure: Measuring homomorphism (e.g., entropy)
        output_idx: Index of output layer (-1 for last)

    Returns:
        Dict mapping layer name to importance score
    """
    if output_idx < 0:
        output_idx = len(model.resources) + output_idx

    importance = {}
    for i, resource in enumerate(model.resources):
        if i != output_idx:
            flow = model.information_flow(i, output_idx, measure)
            importance[resource.name] = flow

    return importance


def minimal_circuit(model: ResourceNetwork,
                   task_examples: Any,
                   optimality_criterion: str = "minimal_params") -> ResourceNetwork:
    """
    Find minimal subnetwork sufficient for task.

    Based on optimal resource assignment (Section 3.3).

    Args:
        model: Full ResourceNetwork
        task_examples: Task data for evaluation
        optimality_criterion: "minimal_params", "minimal_flops", etc.

    Returns:
        Pruned ResourceNetwork
    """
    # TODO: Implement actual circuit discovery
    # For now, return original model
    print(f"[Mock] Finding minimal circuit with {optimality_criterion}...")
    return model


# Example usage
if __name__ == "__main__":
    print("Neural Interpretability API")
    print("=" * 50)
    print()

    # Load model
    model = ResourceNetwork.from_pretrained("gpt2")
    print(f"Loaded model with {len(model.resources)} resources\n")

    # Find redundant resources
    print("Finding redundant resources...")
    redundant = model.find_redundant_resources(threshold=0.8)
    print(f"Found {len(redundant)} redundant pairs\n")

    # Measure entropy
    print("Computing entropy for each layer...")
    entropy = EntropyMeasure()
    entropies = model.measure_all(entropy)
    for i, h in enumerate(entropies):
        print(f"  Layer {i}: H = {h:.3f}")
    print()

    # Layer importance
    print("Computing layer importance...")
    importance = layer_importance(model, entropy)
    for name, score in sorted(importance.items(),
                             key=lambda x: x[1],
                             reverse=True)[:5]:
        print(f"  {name}: {score:.3f}")
