"""Package initialization for neural interpretability library."""

from .src.resource import (
    Resource,
    ConversionRate,
    MeasuringHomomorphism,
    EntropyMeasure,
    ParameterCountMeasure,
    FLOPsMeasure,
    PreorderedMonoid,
    theorem_5_6,
)

from .src.interpretability import (
    NeuralResource,
    ResourceNetwork,
    attention_redundancy,
    layer_importance,
    minimal_circuit,
)

__version__ = "0.1.0"

__all__ = [
    # Core resource theory
    "Resource",
    "ConversionRate",
    "MeasuringHomomorphism",
    "EntropyMeasure",
    "ParameterCountMeasure",
    "FLOPsMeasure",
    "PreorderedMonoid",
    "theorem_5_6",
    # Interpretability API
    "NeuralResource",
    "ResourceNetwork",
    "attention_redundancy",
    "layer_importance",
    "minimal_circuit",
]
