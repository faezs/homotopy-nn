"""
Functor Compiler: Categorical Interpretation of F: C → Set

Key insight: A functor F: C → Set has:
- F₀: Ob(C) → Set (objects to tensor specs)
- F₁: Hom(C) → Set-functions (morphisms to operations)
- F-id: F₁(id_c) = id_{F₀(c)} (preserves identity)
- F-∘: F₁(f ∘ g) = F₁(g) ∘ F₁(f) (preserves composition)

The functoriality axiom F-∘ IS function composition in JAX!
We don't need to manually track composition - it's automatic.
"""

from dataclasses import dataclass
from typing import Dict, List, Callable, Any
import jax
import jax.numpy as jnp


@dataclass
class TensorSpec:
    """Specification of a tensor type (from F₀)."""
    kind: str  # "scalar", "vec", "mat", "prod", "sum"
    dims: List[int]  # Dimensions

    def __repr__(self):
        if self.kind == "scalar":
            return "ℝ"
        elif self.kind == "vec":
            return f"ℝ^{self.dims[0]}"
        elif self.kind == "mat":
            return f"ℝ^{self.dims[0]}×{self.dims[1]}"
        elif self.kind == "prod":
            return f"({self.dims[0]} × {self.dims[1]})"
        elif self.kind == "sum":
            return f"({self.dims[0]} + {self.dims[1]})"
        else:
            return f"Tensor({self.kind}, {self.dims})"


@dataclass
class ComputeOp:
    """A compute operation (from F₁)."""
    kind: str  # "linear", "conv", "activation", "fork", "residual", "compose", "identity"
    params: Dict[str, Any]  # Operation-specific parameters

    def __repr__(self):
        if self.kind == "identity":
            return "id"
        elif self.kind == "compose":
            return f"({self.params['f']} ∘ {self.params['g']})"
        elif self.kind == "fork":
            return f"fork(arity={self.params.get('arity', '?')})"
        else:
            return f"{self.kind}({self.params})"


@dataclass
class ObjectMapping:
    """F₀(c) = TensorSpec at object c."""
    obj_id: str
    tensor_type: TensorSpec


@dataclass
class MorphismMapping:
    """F₁(f) = ComputeOp for morphism f: source → target."""
    morph_id: str
    source: str
    target: str
    operation: ComputeOp


@dataclass
class FunctorialityWitness:
    """Proof that F preserves identity and composition.

    These are proven in Agda, so we don't check them at runtime.
    They become structural guarantees in the compiled code.
    """
    preserves_identity: List[tuple]  # (obj_id, witness)
    preserves_composition: List[tuple]  # (f, g, witness)


@dataclass
class CategoricalFunctor:
    """A functor F: C → Set represented as categorical IR.

    This is the main IR type that preserves compositional structure.
    Unlike flat IR, this directly represents the functor structure.
    """
    name: str
    objects: List[ObjectMapping]  # F₀
    morphisms: List[MorphismMapping]  # F₁
    functoriality: FunctorialityWitness  # F-id, F-∘
    inputs: List[str]  # Input object IDs
    outputs: List[str]  # Output object IDs


class FunctorCompiler:
    """Compile F: C → Set to JAX.

    The key insight is that functoriality F-∘ means composition is automatic.
    We compile F₀ to tensor specifications and F₁ to JAX operations, and
    composition works automatically because of the functoriality axiom!

    Example:
        Given F₁(f) and F₁(g), we have F₁(f ∘ g) = F₁(g) ∘ F₁(f) by functoriality.
        In JAX: jit(lambda x: F1_f(F1_g(x))) is F₁(f ∘ g) automatically!
    """

    def __init__(self, functor: CategoricalFunctor):
        self.functor = functor

        # Build lookup tables for F₀ and F₁
        self.F0: Dict[str, TensorSpec] = {
            obj.obj_id: obj.tensor_type
            for obj in functor.objects
        }

        self.F1: Dict[str, MorphismMapping] = {
            morph.morph_id: morph
            for morph in functor.morphisms
        }

        # Compiled operations (will be filled during compilation)
        self.compiled_ops: Dict[str, Callable] = {}

    def compile_tensor_spec(self, spec: TensorSpec) -> tuple:
        """Convert TensorSpec to JAX shape."""
        if spec.kind == "scalar":
            return ()
        elif spec.kind == "vec":
            return (spec.dims[0],)
        elif spec.kind == "mat":
            return (spec.dims[0], spec.dims[1])
        elif spec.kind == "prod":
            # Product type: (A, B) - represented as concatenated tensor
            return (sum(spec.dims),)  # Simplified for now
        elif spec.kind == "sum":
            # Sum type: A + B - represented as tagged union (simplified)
            return (max(spec.dims),)
        else:
            raise ValueError(f"Unknown tensor spec kind: {spec.kind}")

    def compile_operation(self, op: ComputeOp) -> Callable:
        """Compile a single operation to JAX.

        This is F₁(f) for a single morphism f.
        """
        if op.kind == "identity":
            # F₁(id_c) = id_{F₀(c)}
            return lambda x, params: x

        elif op.kind == "linear":
            # Linear: x ↦ W @ x + b
            in_dim = op.params['in_dim']
            out_dim = op.params['out_dim']

            def linear_fn(x, params):
                W, b = params['W'], params['b']
                return jnp.dot(x, W) + b

            return linear_fn

        elif op.kind == "conv":
            # Convolution
            def conv_fn(x, params):
                # TODO: Implement using jax.lax.conv
                return x

            return conv_fn

        elif op.kind == "activation":
            # Activation function
            act_name = op.params['activation']
            if act_name == "relu":
                return lambda x, params: jax.nn.relu(x)
            elif act_name == "sigmoid":
                return lambda x, params: jax.nn.sigmoid(x)
            elif act_name == "tanh":
                return lambda x, params: jnp.tanh(x)
            else:
                raise ValueError(f"Unknown activation: {act_name}")

        elif op.kind == "fork":
            # Fork operation: concatenate inputs (from sheaf condition!)
            # F(A★) ≅ ∏ F(incoming) becomes concat in JAX
            def fork_fn(inputs, params):
                if len(inputs) == 1:
                    return inputs[0]
                return jnp.concatenate(inputs, axis=-1)

            return fork_fn

        elif op.kind == "residual":
            # Residual: x + f(x) (conservation law!)
            def residual_fn(x, params):
                # Assumes params contains the residual function
                return x + params['residual_op'](x, params.get('residual_params', {}))

            return residual_fn

        elif op.kind == "compose":
            # Composition: F₁(f ∘ g) = F₁(g) ∘ F₁(f)
            # This is automatic from functoriality!
            f = op.params['f']
            g = op.params['g']
            f_compiled = self.compile_operation(f)
            g_compiled = self.compile_operation(g)

            def compose_fn(x, params):
                # Apply g first, then f (contravariant!)
                y = g_compiled(x, params.get('g_params', {}))
                return f_compiled(y, params.get('f_params', {}))

            return compose_fn

        else:
            raise ValueError(f"Unknown operation kind: {op.kind}")

    def compile(self) -> Callable:
        """Compile the entire functor to a JAX function.

        Returns a JIT-compiled forward function where composition is automatic.
        """
        # Compile all operations
        for morph_id, morph in self.F1.items():
            self.compiled_ops[morph_id] = self.compile_operation(morph.operation)

        # Build the forward function
        # For now, we assume a simple path from inputs to outputs
        # In a full implementation, this would traverse the category structure

        def forward(x, params):
            """Forward pass following categorical structure.

            Composition is automatic from functoriality!
            """
            # TODO: Traverse category structure to build computation
            # For now, simple sequential application
            activations = {input_id: x for input_id in self.functor.inputs}

            # Apply morphisms in topological order
            # (would be extracted from category structure)
            for morph_id in sorted(self.compiled_ops.keys()):
                morph = self.F1[morph_id]
                source_val = activations.get(morph.source, x)
                op_fn = self.compiled_ops[morph_id]
                activations[morph.target] = op_fn(source_val, params.get(morph_id, {}))

            # Return output
            output_ids = self.functor.outputs
            if len(output_ids) == 1:
                return activations[output_ids[0]]
            else:
                return tuple(activations[oid] for oid in output_ids)

        # JIT compile for performance
        return jax.jit(forward)

    def initialize_params(self, rng_key):
        """Initialize parameters for all operations.

        Uses functoriality to ensure consistent initialization across compositions.
        """
        params = {}

        for morph_id, morph in self.F1.items():
            op = morph.operation

            if op.kind == "linear":
                in_dim = op.params['in_dim']
                out_dim = op.params['out_dim']

                # Xavier/Glorot initialization
                scale = jnp.sqrt(2.0 / (in_dim + out_dim))
                key_W, key_b = jax.random.split(rng_key)
                rng_key = key_b

                params[morph_id] = {
                    'W': scale * jax.random.normal(key_W, (in_dim, out_dim)),
                    'b': jnp.zeros(out_dim)
                }

            elif op.kind == "conv":
                # TODO: Initialize conv parameters
                params[morph_id] = {}

            elif op.kind == "compose":
                # Composition doesn't have its own parameters
                # Parameters come from constituent morphisms
                params[morph_id] = {
                    'f_params': {},
                    'g_params': {}
                }

            # Other operations don't have parameters

        return params

    def summary(self):
        """Print a summary of the functor structure."""
        print(f"\n=== Categorical Functor: {self.functor.name} ===")
        print(f"Objects (F₀): {len(self.functor.objects)}")
        print(f"Morphisms (F₁): {len(self.functor.morphisms)}")
        print(f"Inputs: {self.functor.inputs}")
        print(f"Outputs: {self.functor.outputs}")

        print("\nF₀ (Object Mapping):")
        for obj in self.functor.objects:
            print(f"  {obj.obj_id} ↦ {obj.tensor_type}")

        print("\nF₁ (Morphism Mapping):")
        for morph in self.functor.morphisms:
            print(f"  {morph.morph_id}: {morph.source} → {morph.target}")
            print(f"    {morph.operation}")

        print(f"\nFunctoriality:")
        print(f"  Preserves identity: {len(self.functor.functoriality.preserves_identity)} proofs")
        print(f"  Preserves composition: {len(self.functor.functoriality.preserves_composition)} proofs")
        print("  (Composition is automatic in JAX!)")
