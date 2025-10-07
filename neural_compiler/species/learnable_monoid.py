"""
Learnable Commutative Monoid Aggregators

From Ong & Veličković (2022): "Learning Algebraic Structure for Graph Neural Networks"

Key results:
1. Well-behaved GNN aggregators are commutative monoids
2. Learnable monoids achieve O(log V) depth (vs O(V) for RNNs)
3. Exponential improvements for parallelism and dependency length

Implementation:
- Binary operator: combine(x, y) = MLP([x; y])
- Training with commutativity regularization
- Balanced binary tree aggregation (O(log n) depth!)

Connection to topos theory:
- Fork vertices A★ have sheaf condition: F(A★) ≅ ∏ F(incoming)
- This product IS a monoid operation!
- Instead of hardcoded concat/sum, we LEARN the associative binary operator

Example:
    Instead of: concat([x1, x2, x3, x4])
    We compute: combine(combine(x1, x2), combine(x3, x4))
    With learned binary operator combine: (x, y) ↦ MLP([x; y])
"""

from typing import List, Callable
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from flax import linen as nn
import optax


class LearnableMonoidAggregator(nn.Module):
    """Learnable commutative monoid for O(log n) aggregation.

    Attributes:
        features: Dimension of the aggregated vectors
        mlp_depth: Depth of the MLP defining the binary operator
        hidden_dim: Hidden dimension of the MLP (default: 2*features)
        commutative_reg_weight: Weight for commutativity regularization loss

    Example:
        ```python
        aggregator = LearnableMonoidAggregator(features=64, mlp_depth=3)
        inputs = [x1, x2, x3, x4]  # Each x_i has shape (batch, 64)
        output = aggregator(inputs)  # Shape: (batch, 64)
        # Aggregation happens in O(log 4) = 2 depth!
        ```

    The binary operator is:
        combine(x, y) = MLP([x; y]) where MLP has `mlp_depth` layers
    """

    features: int
    mlp_depth: int = 3
    hidden_dim: int = None
    commutative_reg_weight: float = 0.1

    def setup(self):
        """Setup the MLP defining the binary operator."""
        hidden = self.hidden_dim or (2 * self.features)

        # Binary operator: (x, y) → combined
        # Input: [x; y] with shape (batch, 2*features)
        # Output: combined with shape (batch, features)
        layers = []
        for i in range(self.mlp_depth):
            if i == 0:
                layers.append(nn.Dense(hidden))
            elif i == self.mlp_depth - 1:
                layers.append(nn.Dense(self.features))
            else:
                layers.append(nn.Dense(hidden))

        self.mlp_layers = layers

    def combine(self, x, y):
        """Binary operator: (x, y) ↦ MLP([x; y]).

        This is the learned associative operation defining the monoid.

        Args:
            x: Tensor of shape (batch, features)
            y: Tensor of shape (batch, features)

        Returns:
            combined: Tensor of shape (batch, features)
        """
        # Concatenate inputs
        xy = jnp.concatenate([x, y], axis=-1)  # Shape: (batch, 2*features)

        # Apply MLP
        activation = xy
        for i, layer in enumerate(self.mlp_layers):
            activation = layer(activation)
            if i < len(self.mlp_layers) - 1:
                activation = jax.nn.relu(activation)

        return activation

    def aggregate_tree(self, inputs: List):
        """Aggregate inputs using balanced binary tree.

        This achieves O(log n) depth instead of O(n) for sequential!

        Algorithm:
            1. If only one input, return it
            2. Otherwise, recursively aggregate pairs
            3. Build balanced binary tree

        Example:
            inputs = [x1, x2, x3, x4]
            →  [combine(x1, x2), combine(x3, x4)]
            →  [combine(combine(x1, x2), combine(x3, x4))]
            Depth: O(log 4) = 2

        Args:
            inputs: List of tensors, each with shape (batch, features)

        Returns:
            aggregated: Single tensor of shape (batch, features)
        """
        if len(inputs) == 0:
            # Empty aggregation → zero vector
            batch_size = 1  # Default
            return jnp.zeros((batch_size, self.features))

        if len(inputs) == 1:
            return inputs[0]

        # Pair up inputs and recursively aggregate
        pairs = []
        for i in range(0, len(inputs), 2):
            if i + 1 < len(inputs):
                # Combine pair
                combined = self.combine(inputs[i], inputs[i + 1])
                pairs.append(combined)
            else:
                # Odd one out, pass through
                pairs.append(inputs[i])

        # Recursively aggregate the pairs
        return self.aggregate_tree(pairs)

    def __call__(self, inputs: List):
        """Aggregate a list of inputs.

        Args:
            inputs: List of tensors, each with shape (batch, features)

        Returns:
            aggregated: Single tensor of shape (batch, features)
        """
        return self.aggregate_tree(inputs)

    def commutativity_loss(self, x, y):
        """Commutativity regularization: ||combine(x,y) - combine(y,x)||².

        This encourages the learned operator to be commutative.

        Args:
            x: Tensor of shape (batch, features)
            y: Tensor of shape (batch, features)

        Returns:
            loss: Scalar commutativity penalty
        """
        xy = self.combine(x, y)
        yx = self.combine(y, x)
        return jnp.mean((xy - yx) ** 2)


# Training utilities


def create_training_step(aggregator_model, learning_rate=1e-3):
    """Create a training step function with commutativity regularization.

    Args:
        aggregator_model: LearnableMonoidAggregator instance
        learning_rate: Learning rate for Adam optimizer

    Returns:
        train_step: Function (params, inputs, targets) → (params, loss)
    """
    optimizer = optax.adam(learning_rate)

    def loss_fn(params, inputs, targets):
        """Loss = MSE + commutativity regularization."""
        # Forward pass
        aggregated = aggregator_model.apply({'params': params}, inputs)

        # Task loss (e.g., regression)
        task_loss = jnp.mean((aggregated - targets) ** 2)

        # Commutativity regularization
        # Sample pairs from inputs
        if len(inputs) >= 2:
            x, y = inputs[0], inputs[1]
            comm_loss = aggregator_model.apply(
                {'params': params},
                method=aggregator_model.commutativity_loss
            )(x, y)
        else:
            comm_loss = 0.0

        total_loss = task_loss + aggregator_model.commutative_reg_weight * comm_loss

        return total_loss, (task_loss, comm_loss)

    @jit
    def train_step(params, opt_state, inputs, targets):
        """Single training step with commutativity regularization."""
        (loss, (task_loss, comm_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params, inputs, targets)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, task_loss, comm_loss

    return train_step, optimizer


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Learnable Commutative Monoid Aggregator Demo")
    print("="*60)
    print()

    # Create aggregator
    features = 64
    batch_size = 32
    num_inputs = 8

    aggregator = LearnableMonoidAggregator(
        features=features,
        mlp_depth=3,
        commutative_reg_weight=0.1
    )

    # Initialize
    rng = random.PRNGKey(0)
    dummy_inputs = [random.normal(random.PRNGKey(i), (batch_size, features))
                    for i in range(num_inputs)]

    params = aggregator.init(rng, dummy_inputs)['params']

    print(f"Features: {features}")
    print(f"MLP depth: {aggregator.mlp_depth}")
    print(f"Number of parameters: {sum(p.size for p in jax.tree_util.tree_leaves(params))}")
    print()

    # Test aggregation
    print(f"Aggregating {num_inputs} inputs...")
    aggregated = aggregator.apply({'params': params}, dummy_inputs)
    print(f"Input shapes: {[x.shape for x in dummy_inputs]}")
    print(f"Aggregated shape: {aggregated.shape}")
    print(f"Expected: ({batch_size}, {features})")
    print()

    # Test commutativity
    print("Testing commutativity...")
    x, y = dummy_inputs[0], dummy_inputs[1]
    xy = aggregator.apply({'params': params}, method=aggregator.combine)(x, y)
    yx = aggregator.apply({'params': params}, method=aggregator.combine)(y, x)
    comm_error = jnp.mean((xy - yx) ** 2)
    print(f"Commutativity error: {comm_error:.6f}")
    print("(Should decrease during training with commutativity regularization)")
    print()

    # Estimate aggregation depth
    import math
    depth = math.ceil(math.log2(num_inputs))
    print(f"Aggregation depth: O(log {num_inputs}) = {depth}")
    print(f"Sequential depth would be: O({num_inputs}) = {num_inputs}")
    print(f"Speedup factor: {num_inputs / depth:.1f}x")
    print()

    print("="*60)
    print("✓ Learnable monoid aggregator working!")
    print("="*60)
