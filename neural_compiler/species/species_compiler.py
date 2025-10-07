"""
Tensor Species Compiler: Direct Einsum Interpretation

Compile tensor species (functors core(FinSet)^d → FinVect) to JAX via einsums.

Key insight from Dudzik (2024): "Einsums represent the bulk of shape-changing
operations in neural networks" and they are closed under gradients.

For einsum 'ij,jk->ik' (matmul), gradient is automatic - just permute the feet!

Architecture:
1. Parse TensorSpecies JSON from Agda
2. Compile einsum operations to jnp.einsum
3. Composition is automatic (no manual graph traversal!)
4. Learnable monoids for fork vertices (O(log n) aggregation)
"""

from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional
import json
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap


@dataclass
class IndexVar:
    """An index variable with name and shape.

    Example: IndexVar("I", 784) represents index I with dimension 784.
    In einsum "ij->j", this would be the 'i' index.
    """
    name: str
    shape: int

    def __repr__(self):
        return f"{self.name}({self.shape})"


@dataclass
class EinsumOp:
    """An einsum operation.

    Represents a polynomial functor as an einsum string.

    Example:
        EinsumOp("ij,jk->ik", [IndexVar("I", 784), IndexVar("J", 256), ...], [...])
    compiles to:
        lambda W1, W2: jnp.einsum('ij,jk->ik', W1, W2)
    """
    spec: str  # Einsum specification (e.g., "ij,jk->ik")
    input_indices: List[IndexVar]
    output_indices: List[IndexVar]
    op_type: str  # "einsum", "identity", "elementwise"

    # For elementwise ops
    elementwise_name: Optional[str] = None

    def __repr__(self):
        if self.op_type == "einsum":
            return f"einsum('{self.spec}')"
        elif self.op_type == "identity":
            return "id"
        elif self.op_type == "elementwise":
            return f"{self.elementwise_name}"
        return f"{self.op_type}"


@dataclass
class LearnableMonoid:
    """A learnable commutative monoid for aggregation.

    From Ong & Veličković (2022): "A well-behaved aggregator for a GNN is a
    commutative monoid over its latent space."

    Instead of hardcoded sum/concat, we learn the binary operator:
        combine: (x, y) ↦ MLP([x; y])

    Training includes commutativity regularization:
        loss += λ * ||combine(x,y) - combine(y,x)||²
    """
    input_arities: List[int]
    output_dim: int
    monoid_type: str  # "sum", "max", "concat", "learnable"
    commutative_reg: bool

    # For learnable monoids
    depth: Optional[int] = None  # MLP depth

    def __repr__(self):
        if self.monoid_type == "learnable":
            return f"LearnableMonoid(depth={self.depth}, comm_reg={self.commutative_reg})"
        return f"{self.monoid_type}-monoid"


@dataclass
class TensorSpecies:
    """A tensor species: functor core(FinSet)^d → FinVect.

    Represents a neural network as a categorical structure where:
    - Objects (F₀): Index variables → tensor dimensions
    - Morphisms (F₁): Einsum operations
    - Functoriality: Composition is automatic!
    - Monoids: Learnable aggregators for fork vertices

    Example: Simple MLP
        dimension = 3  # Three index variables: I, J, K
        index_shapes = [IndexVar("I", 784), IndexVar("J", 256), IndexVar("K", 10)]
        operations = [
            EinsumOp("ij->j", [I], [J], "einsum"),
            EinsumOp("relu", [J], [J], "elementwise", elementwise_name="relu"),
            EinsumOp("jk->k", [J], [K], "einsum")
        ]
    """
    name: str
    dimension: int  # Number of index variables
    index_shapes: Dict[str, int]  # Name → dimension
    operations: List[EinsumOp]
    monoids: List[LearnableMonoid]
    inputs: List[str]  # Input index names
    outputs: List[str]  # Output index names

    @classmethod
    def from_json(cls, json_path: str) -> 'TensorSpecies':
        """Load tensor species from Agda-exported JSON."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Parse index shapes
        index_shapes = {
            idx['name']: idx['shape']
            for idx in data['index_shapes']
        }

        # Parse operations
        operations = []
        for op in data['operations']:
            if op['type'] == 'einsum':
                operations.append(EinsumOp(
                    spec=op['spec'],
                    input_indices=[IndexVar(name, index_shapes[name]) for name in op['inputs']],
                    output_indices=[IndexVar(name, index_shapes[name]) for name in op['outputs']],
                    op_type='einsum'
                ))
            elif op['type'] == 'identity':
                idx_name = op['index']
                operations.append(EinsumOp(
                    spec='',
                    input_indices=[IndexVar(idx_name, index_shapes[idx_name])],
                    output_indices=[IndexVar(idx_name, index_shapes[idx_name])],
                    op_type='identity'
                ))
            elif op['type'] == 'elementwise':
                idx_names = op['indices']
                operations.append(EinsumOp(
                    spec='',
                    input_indices=[IndexVar(name, index_shapes[name]) for name in idx_names],
                    output_indices=[IndexVar(name, index_shapes[name]) for name in idx_names],
                    op_type='elementwise',
                    elementwise_name=op['name']
                ))

        # Parse monoids
        monoids = []
        for mon in data.get('monoids', []):
            mtype = mon['monoid_type']
            if isinstance(mtype, dict) and mtype['type'] == 'learnable':
                monoids.append(LearnableMonoid(
                    input_arities=mon['input_arities'],
                    output_dim=mon['output_dim'],
                    monoid_type='learnable',
                    commutative_reg=mon['commutative_reg'],
                    depth=mtype['depth']
                ))
            else:
                monoids.append(LearnableMonoid(
                    input_arities=mon['input_arities'],
                    output_dim=mon['output_dim'],
                    monoid_type=mtype,
                    commutative_reg=False
                ))

        return cls(
            name=data['name'],
            dimension=data['dimension'],
            index_shapes=index_shapes,
            operations=operations,
            monoids=monoids,
            inputs=data['inputs'],
            outputs=data['outputs']
        )


class SpeciesCompiler:
    """Compile tensor species to JAX via direct einsum interpretation.

    The key insight is that einsum operations compose automatically via
    functoriality: F₁(f ∘ g) = F₁(g) ∘ F₁(f)

    In JAX: jit(lambda x: einsum2(einsum1(x))) is the composition!
    """

    def __init__(self, species: TensorSpecies):
        self.species = species
        self.compiled_ops: Dict[int, Callable] = {}

    def compile_einsum(self, op: EinsumOp) -> Callable:
        """Compile a single einsum operation to JAX.

        This is where the magic happens: jnp.einsum is a direct interpretation
        of the polynomial functor represented by the einsum string!
        """
        if op.op_type == "identity":
            return lambda x: x

        elif op.op_type == "einsum":
            # Direct einsum compilation!
            spec = op.spec

            def einsum_fn(*args):
                return jnp.einsum(spec, *args)

            return einsum_fn

        elif op.op_type == "elementwise":
            # Elementwise operations (applied pointwise)
            name = op.elementwise_name

            if name == "relu":
                return jax.nn.relu
            elif name == "sigmoid":
                return jax.nn.sigmoid
            elif name == "tanh":
                return jnp.tanh
            elif name == "softmax":
                return jax.nn.softmax
            else:
                raise ValueError(f"Unknown elementwise operation: {name}")

        else:
            raise ValueError(f"Unknown operation type: {op.op_type}")

    def compile(self) -> Callable:
        """Compile the entire species to a JIT-compiled JAX function.

        Returns a function: (x, params) → y where composition is automatic!
        """
        # Compile all operations
        compiled_ops = [self.compile_einsum(op) for op in self.species.operations]

        def forward(x, params):
            """Forward pass. Composition is automatic from functoriality!"""
            activation = x

            for i, op_fn in enumerate(compiled_ops):
                op_params = params.get(i, None)

                if op_params is not None:
                    # Operation has parameters (e.g., linear layer)
                    if isinstance(op_params, dict):
                        # Parameters are a dict (e.g., W and b)
                        activation = op_fn(activation, **op_params)
                    else:
                        activation = op_fn(activation, op_params)
                else:
                    # Operation has no parameters (e.g., activation)
                    activation = op_fn(activation)

            return activation

        # JIT compile for performance
        return jax.jit(forward)

    def initialize_params(self, rng_key):
        """Initialize parameters for einsum operations.

        For einsum 'ij,jk->ik' (linear layer), this creates W with shape (i,j,k).
        Uses Xavier/Glorot initialization.
        """
        params = {}

        for i, op in enumerate(self.species.operations):
            if op.op_type == "einsum":
                # Parse einsum spec to determine parameter shapes
                # For now, simplified: assume linear layers
                # TODO: Full einsum parameter inference

                if len(op.input_indices) == 2 and len(op.output_indices) == 1:
                    # Binary einsum like 'ij,jk->ik' → matrix multiplication
                    # Need weight matrix
                    in_dim = op.input_indices[0].shape
                    out_dim = op.output_indices[0].shape

                    scale = jnp.sqrt(2.0 / (in_dim + out_dim))
                    key_W, rng_key = jax.random.split(rng_key)

                    params[i] = {
                        'W': scale * jax.random.normal(key_W, (in_dim, out_dim))
                    }

            # No parameters for identity or elementwise ops

        return params

    def compile_with_grad(self) -> tuple[Callable, Callable]:
        """Compile both forward and backward pass.

        From Dudzik: "The gradient flow through an einsum is an einsum."

        For einsum 'ij,jk->ik', the gradient automatically becomes
        'ik,jk->ij' (permute the feet!)

        JAX handles this automatically via autodiff.
        """
        forward_fn = self.compile()

        # Gradient is automatic!
        grad_fn = jax.grad(lambda params, x: jnp.sum(forward_fn(x, params)))

        return jax.jit(forward_fn), jax.jit(grad_fn)

    def summary(self):
        """Print summary of the tensor species."""
        print(f"\n{'='*60}")
        print(f"Tensor Species: {self.species.name}")
        print(f"{'='*60}")
        print(f"Dimension: {self.species.dimension} index variables")
        print(f"Operations: {len(self.species.operations)}")
        print(f"Monoids: {len(self.species.monoids)}")
        print()

        print("Index Shapes (F₀):")
        for name, shape in self.species.index_shapes.items():
            print(f"  {name} → {shape}")
        print()

        print("Operations (F₁):")
        for i, op in enumerate(self.species.operations):
            print(f"  {i}. {op}")
        print()

        if self.species.monoids:
            print("Learnable Monoids (Fork Aggregators):")
            for mon in self.species.monoids:
                print(f"  {mon}")
        print()

        print(f"Inputs: {self.species.inputs}")
        print(f"Outputs: {self.species.outputs}")
        print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # This would load from Agda-exported JSON
    # For now, create a simple example programmatically

    mlp_species = TensorSpecies(
        name="SimpleMLP",
        dimension=3,
        index_shapes={"I": 784, "J": 256, "K": 10},
        operations=[
            EinsumOp("ij->j", [IndexVar("I", 784)], [IndexVar("J", 256)], "einsum"),
            EinsumOp("", [IndexVar("J", 256)], [IndexVar("J", 256)], "elementwise", elementwise_name="relu"),
            EinsumOp("jk->k", [IndexVar("J", 256)], [IndexVar("K", 10)], "einsum"),
        ],
        monoids=[],
        inputs=["I"],
        outputs=["K"]
    )

    compiler = SpeciesCompiler(mlp_species)
    compiler.summary()

    # Compile
    forward_fn = compiler.compile()
    params = compiler.initialize_params(jax.random.PRNGKey(0))

    # Test
    x = jax.random.normal(jax.random.PRNGKey(1), (32, 784))
    output = forward_fn(x, params)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (32, 10)")
    print(f"✓ Shapes match!" if output.shape == (32, 10) else "✗ Shape mismatch!")
