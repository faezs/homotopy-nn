#!/usr/bin/env python3
"""
MLP MUP Runner: JSON REPL for MLP execution with MUP scaling

Receives network specifications from Haskell via JSON protocol,
executes them using JAX/Flax, and returns training results.

## JSON Protocol

**Request**:
```json
{
  "command": "train",
  "network": {
    "layers": [...],
    "connections": [...],
    "batch_size": 32,
    "num_epochs": 100,
    "optimizer": "adam"
  }
}
```

**Response**:
```json
{
  "status": "success",
  "loss": 0.45,
  "accuracy": 0.89,
  "feature_norms": {"hidden_0": 12.3, "hidden_1": 11.8, "output": 5.4},
  "error": null
}
```
"""

import sys
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, List, Optional
from dataclasses import dataclass

# Configure JAX
jax.config.update("jax_enable_x64", False)  # Use float32 for speed


@dataclass
class LayerSpec:
    """Layer specification from Agda/Haskell"""
    type: str  # "input", "hidden", or "output"
    dim: Optional[int] = None
    width: Optional[int] = None
    init_std: Optional[float] = None
    lr: Optional[float] = None


@dataclass
class Connection:
    """Connection between layers"""
    from_idx: int
    to_idx: int


@dataclass
class NetworkSpec:
    """Complete network specification"""
    layers: List[LayerSpec]
    connections: List[Connection]
    batch_size: int
    num_epochs: int
    optimizer: str


class MUPInitializer:
    """Custom initializer following MUP scaling rules"""

    def __init__(self, std: float):
        self.std = std

    def __call__(self, key, shape, dtype=jnp.float32):
        """Initialize with MUP-scaled standard deviation"""
        return random.normal(key, shape, dtype) * self.std


class MUPMLP(nn.Module):
    """MLP with MUP parameterization"""

    spec: NetworkSpec

    def setup(self):
        """
        Build MLP following MUP scaling:
        - Hidden layers: init_std / √width, lr = base_lr
        - Output layer: init_std / width, lr = base_lr / width
        """
        self.dense_layers = []

        for i, layer in enumerate(self.spec.layers):
            if layer.type == "input":
                continue  # Input layer is just a placeholder

            elif layer.type == "hidden":
                self.dense_layers.append(
                    nn.Dense(
                        layer.width,
                        kernel_init=MUPInitializer(layer.init_std),
                        bias_init=nn.initializers.zeros,
                        name=f'hidden_{i}'
                    )
                )

            elif layer.type == "output":
                self.dense_layers.append(
                    nn.Dense(
                        layer.dim,
                        kernel_init=MUPInitializer(layer.init_std),
                        bias_init=nn.initializers.zeros,
                        name='output'
                    )
                )

    def __call__(self, x, training: bool = False):
        """Forward pass through MUP-parameterized network"""
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.reshape((x.shape[0], -1))

        # Apply all layers
        for i, layer in enumerate(self.dense_layers):
            x = layer(x)
            # ReLU for all except last layer
            if i < len(self.dense_layers) - 1:
                x = nn.relu(x)

        return x


def create_mup_optimizer(spec: NetworkSpec) -> optax.GradientTransformation:
    """
    Create optimizer with per-layer learning rates (MUP scaling).

    Uses optax.multi_transform to apply different LRs:
    - Hidden layers: base_lr
    - Output layer: base_lr / width
    """
    def label_fn(path, _):
        """Label parameters by layer type"""
        path_str = '/'.join(path)
        if 'output' in path_str:
            return 'output'
        elif 'hidden' in path_str:
            return 'hidden'
        else:
            return 'other'

    # Extract learning rates from spec
    hidden_lr = None
    output_lr = None
    for layer in spec.layers:
        if layer.type == "hidden" and layer.lr is not None:
            hidden_lr = layer.lr
        elif layer.type == "output" and layer.lr is not None:
            output_lr = layer.lr

    # Default fallback
    if hidden_lr is None:
        hidden_lr = 0.1
    if output_lr is None:
        output_lr = 0.01

    # Create optimizers for each group
    if spec.optimizer == 'sgd':
        optimizers = {
            'hidden': optax.sgd(hidden_lr),
            'output': optax.sgd(output_lr),
            'other': optax.sgd(hidden_lr)
        }
    elif spec.optimizer == 'adam':
        optimizers = {
            'hidden': optax.adam(hidden_lr),
            'output': optax.adam(output_lr),
            'other': optax.adam(hidden_lr)
        }
    else:
        raise ValueError(f"Unknown optimizer: {spec.optimizer}")

    return optax.multi_transform(optimizers, label_fn)


def create_train_state(spec: NetworkSpec, rng_key) -> train_state.TrainState:
    """Initialize MUP network and training state"""
    model = MUPMLP(spec)

    # Get input dimension from first layer
    input_dim = next(l.dim for l in spec.layers if l.type == "input")
    dummy_input = jnp.ones((1, input_dim))

    # Initialize parameters
    params = model.init(rng_key, dummy_input, training=False)

    # Create MUP optimizer
    tx = create_mup_optimizer(spec)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


@jit
def train_step(state: train_state.TrainState, batch_x, batch_y):
    """Single training step with cross-entropy loss"""
    def loss_fn(params):
        logits = state.apply_fn(params, batch_x, training=True)
        # Cross-entropy loss
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        labels_one_hot = jax.nn.one_hot(batch_y, num_classes=logits.shape[-1])
        loss = -jnp.mean(jnp.sum(log_probs * labels_one_hot, axis=-1))
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    # Compute accuracy
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == batch_y)

    return state, loss, accuracy


def compute_feature_norms(state: train_state.TrainState) -> Dict[str, float]:
    """
    Compute feature norms at each layer.

    MUP property: These should stay O(1) across widths.
    """
    norms = {}

    # Extract parameters from flattened params dict
    flat_params = flax.traverse_util.flatten_dict(state.params['params'])

    for key_path, value in flat_params.items():
        if 'kernel' in key_path:
            layer_name = '_'.join(key_path[:-1])  # Remove 'kernel' suffix
            norms[layer_name] = float(jnp.linalg.norm(value))

    return norms


def generate_synthetic_dataset(input_dim: int, output_dim: int, num_samples: int = 1000):
    """Generate synthetic classification dataset"""
    rng = np.random.RandomState(42)
    X = rng.randn(num_samples, input_dim).astype(np.float32)
    y = rng.randint(0, output_dim, size=num_samples)
    return X, y


def train_network(spec: NetworkSpec) -> Dict:
    """Train network with MUP parameterization"""
    # Get dimensions
    input_dim = next(l.dim for l in spec.layers if l.type == "input")
    output_dim = next(l.dim for l in spec.layers if l.type == "output")

    # Generate dataset
    train_x, train_y = generate_synthetic_dataset(input_dim, output_dim)
    num_samples = train_x.shape[0]

    # Initialize
    rng_key = random.PRNGKey(42)
    state = create_train_state(spec, rng_key)

    # Training loop
    losses = []
    accuracies = []

    for epoch in range(spec.num_epochs):
        # Shuffle data
        perm = np.random.permutation(num_samples)
        train_x_shuffled = train_x[perm]
        train_y_shuffled = train_y[perm]

        # Mini-batch training
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for i in range(0, num_samples, spec.batch_size):
            batch_x = train_x_shuffled[i:i+spec.batch_size]
            batch_y = train_y_shuffled[i:i+spec.batch_size]

            state, loss, acc = train_step(state, batch_x, batch_y)
            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        losses.append(float(avg_loss))
        accuracies.append(float(avg_acc))

    # Compute final feature norms
    feature_norms = compute_feature_norms(state)

    return {
        'status': 'success',
        'loss': losses[-1],
        'accuracy': accuracies[-1],
        'feature_norms': feature_norms,
        'error': None
    }


def parse_network_spec(json_spec: Dict) -> NetworkSpec:
    """Parse network specification from JSON"""
    layers = []
    for layer_data in json_spec['layers']:
        layer = LayerSpec(
            type=layer_data['type'],
            dim=layer_data.get('dim'),
            width=layer_data.get('width'),
            init_std=layer_data.get('init_std'),
            lr=layer_data.get('lr')
        )
        layers.append(layer)

    connections = []
    for conn_data in json_spec.get('connections', []):
        conn = Connection(
            from_idx=conn_data['from'],
            to_idx=conn_data['to']
        )
        connections.append(conn)

    return NetworkSpec(
        layers=layers,
        connections=connections,
        batch_size=json_spec.get('batch_size', 32),
        num_epochs=json_spec.get('num_epochs', 100),
        optimizer=json_spec.get('optimizer', 'adam')
    )


def handle_request(request: Dict) -> Dict:
    """Handle a single JSON request"""
    try:
        command = request.get('command')

        if command == 'train':
            network_json = request.get('network')
            spec = parse_network_spec(network_json)
            return train_network(spec)

        else:
            return {
                'status': 'error',
                'loss': None,
                'accuracy': None,
                'feature_norms': None,
                'error': f"Unknown command: {command}"
            }

    except Exception as e:
        return {
            'status': 'error',
            'loss': None,
            'accuracy': None,
            'feature_norms': None,
            'error': str(e)
        }


def main():
    """JSON REPL: read requests from stdin, write responses to stdout"""
    print("MLP MUP Runner ready", file=sys.stderr, flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = handle_request(request)
            print(json.dumps(response), flush=True)

        except json.JSONDecodeError as e:
            error_response = {
                'status': 'error',
                'loss': None,
                'accuracy': None,
                'feature_norms': None,
                'error': f"JSON decode error: {e}"
            }
            print(json.dumps(error_response), flush=True)

        except Exception as e:
            error_response = {
                'status': 'error',
                'loss': None,
                'accuracy': None,
                'feature_norms': None,
                'error': f"Unexpected error: {e}"
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    main()
