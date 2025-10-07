"""
JAX Backend: Compile polynomial functors to optimized JAX code.

Generates JIT-compiled functions that run on CPU/GPU/TPU.
"""

from typing import Dict, List, Tuple, Any, Callable
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from flax import linen as nn
import numpy as np

from .parser import (
    Operation, LinearOp, Conv2DOp, ActivationOp, ForkOp, ResidualOp,
    BatchNormOp, LayerNormOp, MaxPoolOp, AvgPoolOp, AttentionOp,
    ActivationType, Shape, VecShape, MatShape, TensorShape, Vertex
)
from .polyfunctor import PolynomialFunctor, StringDiagram, polyfunctor_to_diagram


# ============================================================================
# JAX Operation Compilation
# ============================================================================

class JAXBackend:
    """
    Compile polynomial functors to JAX functions.

    Usage:
        backend = JAXBackend()
        jax_fn = backend.compile(poly_functor)
        output = jax_fn(input, params)
    """

    def __init__(self):
        self.compiled_ops: Dict[int, Callable] = {}

    def compile_operation(self, op: Operation, shape: Shape) -> Callable:
        """Compile a single operation to JAX function."""

        if isinstance(op, LinearOp):
            def linear_fn(x, params):
                W, b = params['W'], params['b']
                return jnp.dot(x, W) + b
            return linear_fn

        elif isinstance(op, Conv2DOp):
            def conv_fn(x, params):
                W, b = params['W'], params['b']
                # x: (batch, height, width, in_channels)
                # W: (kernel, kernel, in_channels, out_channels)
                out = jax.lax.conv_general_dilated(
                    x, W,
                    window_strides=(1, 1),
                    padding='SAME'
                )
                return out + b[None, None, None, :]
            return conv_fn

        elif isinstance(op, ActivationOp):
            activation_map = {
                ActivationType.RELU: jax.nn.relu,
                ActivationType.SIGMOID: jax.nn.sigmoid,
                ActivationType.TANH: jnp.tanh,
                ActivationType.GELU: jax.nn.gelu,
                ActivationType.IDENTITY: lambda x: x,
            }
            act_fn = activation_map[op.activation]
            return lambda x, params: act_fn(x)

        elif isinstance(op, ForkOp):
            # Fork = concatenate inputs (from sheaf condition)
            def fork_fn(inputs, params):
                # inputs: List of arrays to merge
                return jnp.concatenate(inputs, axis=-1)
            return fork_fn

        elif isinstance(op, ResidualOp):
            # Residual = identity (added to main path elsewhere)
            return lambda x, params: x

        elif isinstance(op, BatchNormOp):
            def batchnorm_fn(x, params):
                gamma, beta = params['gamma'], params['beta']
                mean = jnp.mean(x, axis=0, keepdims=True)
                var = jnp.var(x, axis=0, keepdims=True)
                return gamma * (x - mean) / jnp.sqrt(var + 1e-5) + beta
            return batchnorm_fn

        elif isinstance(op, LayerNormOp):
            def layernorm_fn(x, params):
                gamma, beta = params['gamma'], params['beta']
                mean = jnp.mean(x, axis=-1, keepdims=True)
                var = jnp.var(x, axis=-1, keepdims=True)
                return gamma * (x - mean) / jnp.sqrt(var + 1e-5) + beta
            return layernorm_fn

        elif isinstance(op, MaxPoolOp):
            def maxpool_fn(x, params):
                return jax.lax.reduce_window(
                    x, -jnp.inf, jax.lax.max,
                    (1, op.kernel_size, op.kernel_size, 1),
                    (1, op.stride, op.stride, 1),
                    'SAME'
                )
            return maxpool_fn

        elif isinstance(op, AvgPoolOp):
            def avgpool_fn(x, params):
                return jax.lax.reduce_window(
                    x, 0., jax.lax.add,
                    (1, op.kernel_size, op.kernel_size, 1),
                    (1, op.stride, op.stride, 1),
                    'SAME'
                ) / (op.kernel_size ** 2)
            return avgpool_fn

        elif isinstance(op, AttentionOp):
            def attention_fn(x, params):
                # Simplified multi-head attention
                Q, K, V = params['Q'], params['K'], params['V']
                qk = jnp.dot(jnp.dot(x, Q), jnp.dot(x, K).T)
                attention_weights = jax.nn.softmax(qk / jnp.sqrt(op.d_k))
                return jnp.dot(attention_weights, jnp.dot(x, V))
            return attention_fn

        else:
            raise ValueError(f"Unsupported operation: {op}")

    def compile_diagram(self, diagram: StringDiagram, poly: PolynomialFunctor) -> Callable:
        """
        Compile string diagram to executable JAX function.

        Returns a function: (input, params) → output
        """
        # Build execution plan (topological order)
        node_order = [node['id'] for node in diagram.nodes]

        # Compile each operation
        compiled_ops = {}
        for node in diagram.nodes:
            node_id = node['id']
            op = node['op']
            shape = node['shape']
            compiled_ops[node_id] = self.compile_operation(op, shape)

        # Build adjacency for execution
        incoming: Dict[int, List[int]] = {nid: [] for nid in node_order}
        for wire in diagram.wires:
            source, _, target, _ = wire
            incoming[target].append(source)

        def forward(x, params):
            """Execute the network."""
            # Activations: node_id → tensor
            activations = {}

            # Assume first node is input
            input_node = node_order[0]
            activations[input_node] = x

            # Execute in topological order
            for node_id in node_order[1:]:
                # Get inputs
                input_nodes = incoming[node_id]

                if len(input_nodes) == 0:
                    # No inputs (shouldn't happen after input)
                    continue
                elif len(input_nodes) == 1:
                    # Single input
                    node_input = activations[input_nodes[0]]
                else:
                    # Multiple inputs (fork)
                    node_input = [activations[src] for src in input_nodes]

                # Get parameters for this node
                node_params = params.get(node_id, {})

                # Execute operation
                compiled_op = compiled_ops[node_id]
                activations[node_id] = compiled_op(node_input, node_params)

            # Return output (last node)
            output_node = node_order[-1]
            return activations[output_node]

        return forward

    def compile(self, poly: PolynomialFunctor) -> Callable:
        """
        Compile polynomial functor to JAX function.

        Returns JIT-compiled function ready for training/inference.
        """
        # Convert to string diagram
        diagram = polyfunctor_to_diagram(poly)

        # Compile diagram to JAX
        forward_fn = self.compile_diagram(diagram, poly)

        # JIT compile for performance
        return jit(forward_fn)


# ============================================================================
# Parameter Initialization
# ============================================================================

def initialize_params(poly: PolynomialFunctor, rng_key) -> Dict[int, Dict[str, jnp.ndarray]]:
    """
    Initialize parameters for a compiled network.

    Uses Xavier/Glorot initialization.
    """
    params = {}

    for pos in poly.positions:
        op = poly.operations[pos]
        rng_key, subkey = jax.random.split(rng_key)

        if isinstance(op, LinearOp):
            W = jax.random.normal(subkey, (op.in_dim, op.out_dim)) * jnp.sqrt(2.0 / op.in_dim)
            b = jnp.zeros(op.out_dim)
            params[pos] = {'W': W, 'b': b}

        elif isinstance(op, Conv2DOp):
            W_shape = (op.kernel_size, op.kernel_size, op.in_channels, op.out_channels)
            W = jax.random.normal(subkey, W_shape) * jnp.sqrt(2.0 / op.in_channels)
            b = jnp.zeros(op.out_channels)
            params[pos] = {'W': W, 'b': b}

        elif isinstance(op, BatchNormOp) or isinstance(op, LayerNormOp):
            gamma = jnp.ones(op.features)
            beta = jnp.zeros(op.features)
            params[pos] = {'gamma': gamma, 'beta': beta}

        elif isinstance(op, AttentionOp):
            Q = jax.random.normal(subkey, (op.d_model, op.d_k)) * 0.01
            rng_key, subkey = jax.random.split(rng_key)
            K = jax.random.normal(subkey, (op.d_model, op.d_k)) * 0.01
            rng_key, subkey = jax.random.split(rng_key)
            V = jax.random.normal(subkey, (op.d_model, op.d_v)) * 0.01
            params[pos] = {'Q': Q, 'K': K, 'V': V}

    return params


# ============================================================================
# Training Utilities
# ============================================================================

def create_train_step(forward_fn: Callable, loss_fn: Callable) -> Callable:
    """
    Create a training step function.

    Returns: (params, x, y) → (loss, new_params)
    """
    def train_step(params, x, y, learning_rate=0.001):
        def loss_wrapper(p):
            pred = forward_fn(x, p)
            return loss_fn(pred, y)

        loss, grads = jax.value_and_grad(loss_wrapper)(params)

        # Simple SGD update
        new_params = jax.tree_map(
            lambda p, g: p - learning_rate * g,
            params, grads
        )

        return loss, new_params

    return jit(train_step)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example compilation pipeline
    from .parser import parse_ir
    from .polyfunctor import compile_to_polyfunctor

    # # Load and compile
    # ir = parse_ir("mlp.json")
    # poly = compile_to_polyfunctor(ir)
    #
    # backend = JAXBackend()
    # forward_fn = backend.compile(poly)
    #
    # # Initialize parameters
    # rng = jax.random.PRNGKey(0)
    # params = initialize_params(poly, rng)
    #
    # # Test forward pass
    # x = jnp.ones((1, 784))  # Batch of 1, input dim 784
    # output = forward_fn(x, params)
    # print(f"Output shape: {output.shape}")
    pass
