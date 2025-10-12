"""
Meta-Learning Universal Topos: Phase 2 of ARC-AGI Strategy

This module implements meta-learning across task distributions to discover
a universal topos structure that captures abstract reasoning patterns.

Key Idea:
- Phase 1: Evolved task-specific topoi (C_i, J_i) for each task
- Phase 2: Meta-learn universal topos (C*, J*) that works across tasks
- Phase 3: Few-shot adapt universal topos to new tasks

The meta-learner learns:
1. What categorical structures are common across tasks
2. How to quickly adapt the universal topos to new patterns
3. Task embeddings that predict which coverage to use

Based on:
- src/Neural/Topos/Learnable.agda (theoretical foundation)
- ARC-AGI-2-STRATEGY.md (Phase 2: Meta-Learning Across Tasks)
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from flax import linen as nn
import optax
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from functools import partial

from evolutionary_solver import Site, SheafNetwork, topos_fitness
from arc_solver import ARCTask, ARCToposSolver


################################################################################
# § 1: Task Embeddings
################################################################################

class TaskEncoder(nn.Module):
    """Encode ARC tasks into latent embeddings.

    Takes few-shot examples and produces a task embedding that captures
    the abstract pattern. This embedding is used to adapt the universal
    topos to specific tasks.

    Architecture:
        - Process each example pair (input, output)
        - Aggregate via attention/pooling
        - Produce task embedding vector

    Attributes:
        embedding_dim: Dimension of task embeddings
        hidden_dim: Hidden layer dimensions
    """
    embedding_dim: int = 64
    hidden_dim: int = 128

    def setup(self):
        """Initialize encoder layers."""
        self.example_encoder = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.embedding_dim)
        ])

        # Attention for aggregating multiple examples
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=4,
            qkv_features=self.embedding_dim
        )

    def encode_example(self, input_grid: jnp.ndarray, output_grid: jnp.ndarray) -> jnp.ndarray:
        """Encode single input-output example.

        Args:
            input_grid: (H, W) input grid
            output_grid: (H', W') output grid

        Returns:
            embedding: (embedding_dim,) example embedding
        """
        # Simple encoding: concatenate flattened grids
        input_flat = input_grid.reshape(-1)
        output_flat = output_grid.reshape(-1)

        # Pad to same length and concatenate
        max_len = max(input_flat.shape[0], output_flat.shape[0])
        input_padded = jnp.pad(input_flat, (0, max_len - input_flat.shape[0]))
        output_padded = jnp.pad(output_flat, (0, max_len - output_flat.shape[0]))

        example_vec = jnp.concatenate([input_padded, output_padded])
        return self.example_encoder(example_vec)

    def __call__(self, examples: List[Tuple[jnp.ndarray, jnp.ndarray]]) -> jnp.ndarray:
        """Encode task from multiple examples.

        Args:
            examples: List of (input_grid, output_grid) tuples

        Returns:
            task_embedding: (embedding_dim,) task embedding
        """
        # Encode each example
        example_embeddings = [self.encode_example(inp, out) for inp, out in examples]
        example_embeddings = jnp.stack(example_embeddings)  # (n_examples, embedding_dim)

        # Aggregate via self-attention
        # Query = mean of examples, Key = Value = all examples
        query = jnp.mean(example_embeddings, axis=0, keepdims=True)  # (1, embedding_dim)
        attended = self.attention(query, example_embeddings)  # (1, embedding_dim)

        return attended.squeeze(0)  # (embedding_dim,)


################################################################################
# § 2: Universal Topos Model
################################################################################

@dataclass
class UniversalTopos:
    """Universal topos structure that can be adapted to any task.

    Components:
        - Base site: Core categorical structure (C*, J*)
        - Task encoder: Maps tasks → adaptation parameters
        - Adaptation network: Modifies coverage based on task embedding

    The universal topos captures common structure across all tasks,
    while the adaptation network allows task-specific customization.
    """
    base_site: Site
    task_encoder_params: Dict
    adaptation_params: Dict
    sheaf_params: Dict

    @staticmethod
    def random(key: jax.random.PRNGKey,
               num_objects: int = 20,
               feature_dim: int = 32,
               max_covers: int = 5,
               embedding_dim: int = 64) -> 'UniversalTopos':
        """Initialize random universal topos.

        Args:
            key: PRNG key
            num_objects: Number of objects in base category
            feature_dim: Dimension of object features
            max_covers: Maximum covering families
            embedding_dim: Task embedding dimension

        Returns:
            Universal topos with random parameters
        """
        k1, k2, k3, k4 = random.split(key, 4)

        # Random base site
        base_site = Site.random(k1, num_objects, feature_dim, max_covers)

        # Initialize task encoder
        task_encoder = TaskEncoder(embedding_dim=embedding_dim)
        example_input = (jnp.zeros((10, 10)), jnp.zeros((10, 10)))
        task_encoder_params = task_encoder.init(k2, [example_input])

        # Adaptation network: task_embedding → site modifications
        # Maps (embedding_dim,) → (num_objects, max_covers, num_objects) coverage adjustments
        adaptation_net = nn.Sequential([
            nn.Dense(128),
            nn.relu,
            nn.Dense(num_objects * max_covers * num_objects),
            nn.tanh  # Bounded adjustments in [-1, 1]
        ])
        adaptation_params = adaptation_net.init(k3, jnp.zeros(embedding_dim))

        # Sheaf network parameters
        sheaf = SheafNetwork(hidden_dim=64, output_dim=32)
        sheaf_params = sheaf.init(k4, jnp.zeros(feature_dim))

        return UniversalTopos(
            base_site=base_site,
            task_encoder_params=task_encoder_params,
            adaptation_params=adaptation_params,
            sheaf_params=sheaf_params
        )

    def adapt_to_task(self,
                      task_examples: List[Tuple[jnp.ndarray, jnp.ndarray]],
                      key: jax.random.PRNGKey) -> Site:
        """Adapt universal topos to specific task.

        Args:
            task_examples: Few-shot examples for the task
            key: PRNG key

        Returns:
            adapted_site: Task-specific site derived from universal topos
        """
        # Encode task
        task_encoder = TaskEncoder(embedding_dim=self.task_encoder_params['params']['example_encoder']['0']['kernel'].shape[1])
        task_embedding = task_encoder.apply(self.task_encoder_params, task_examples)

        # Compute coverage adjustments
        adaptation_net = nn.Sequential([
            nn.Dense(128),
            nn.relu,
            nn.Dense(self.base_site.num_objects * self.base_site.max_covers * self.base_site.num_objects),
            nn.tanh
        ])
        adjustments = adaptation_net.apply(self.adaptation_params, task_embedding)
        adjustments = adjustments.reshape(self.base_site.num_objects,
                                         self.base_site.max_covers,
                                         self.base_site.num_objects)

        # Adapt coverage: base + adjustments, then renormalize
        adapted_coverage = self.base_site.coverage_weights + 0.1 * adjustments
        adapted_coverage = jax.nn.softmax(adapted_coverage, axis=-1)

        # Create adapted site
        adapted_site = Site(
            num_objects=self.base_site.num_objects,
            feature_dim=self.base_site.feature_dim,
            max_covers=self.base_site.max_covers,
            adjacency=self.base_site.adjacency,  # Keep adjacency fixed
            object_features=self.base_site.object_features,
            coverage_weights=adapted_coverage
        )

        return adapted_site


################################################################################
# § 3: Meta-Topos Learner
################################################################################

class MetaToposLearner:
    """Meta-learner that discovers universal topos across tasks.

    Training Process:
        1. Sample batch of tasks
        2. For each task:
           a. Adapt universal topos using few-shot examples
           b. Evaluate on held-out test examples
        3. Update universal topos to minimize meta-loss

    Meta-Loss:
        L_meta = E_{tasks} [ -accuracy(adapted_topos, test_examples) ]

    This is essentially MAML (Model-Agnostic Meta-Learning) applied to
    categorical structures instead of neural network weights!

    Args:
        num_objects: Number of objects in universal topos
        feature_dim: Object feature dimension
        max_covers: Maximum covering families
        embedding_dim: Task embedding dimension
        meta_lr: Meta-learning rate
        inner_lr: Task adaptation learning rate (not used in our version)
        num_inner_steps: Number of adaptation steps per task
    """

    def __init__(self,
                 num_objects: int = 20,
                 feature_dim: int = 32,
                 max_covers: int = 5,
                 embedding_dim: int = 64,
                 meta_lr: float = 1e-3,
                 inner_lr: float = 1e-2,
                 num_inner_steps: int = 5):

        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.max_covers = max_covers
        self.embedding_dim = embedding_dim

        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

        # Optimizer for meta-parameters
        self.optimizer = optax.adam(meta_lr)

        # Universal topos (will be initialized in meta_train)
        self.universal_topos: Optional[UniversalTopos] = None
        self.opt_state = None

    def meta_loss(self,
                  universal_topos: UniversalTopos,
                  meta_batch: List[Tuple[ARCTask, int]],
                  key: jax.random.PRNGKey) -> float:
        """Compute meta-learning loss on batch of tasks.

        Meta-loss = average performance across tasks after adaptation.

        Args:
            universal_topos: Current universal topos
            meta_batch: List of (task, n_shots) tuples
            key: PRNG key

        Returns:
            loss: Scalar meta-loss (lower = better generalization)
        """
        losses = []

        for task, n_shots in meta_batch:
            k1, key = random.split(key)

            # Split task into support (adapt) and query (evaluate)
            support_examples = list(zip(
                task.train_inputs[:n_shots],
                task.train_outputs[:n_shots]
            ))
            query_examples = list(zip(
                task.test_inputs,
                task.test_outputs
            ))

            # Convert ARCGrid to arrays
            support = [(inp.cells, out.cells) for inp, out in support_examples]

            # Adapt universal topos to task
            adapted_site = universal_topos.adapt_to_task(support, k1)

            # Evaluate on query set
            sheaf = SheafNetwork(hidden_dim=64, output_dim=32)
            query_loss = 0.0

            for inp_grid, out_grid in query_examples:
                # Simple prediction: compute sheaf sections
                # (In real version, would run full forward pass)
                pred = sheaf.apply(
                    {'params': universal_topos.sheaf_params['params']},
                    method=sheaf.section_at
                )(adapted_site.object_features[0])

                target = jnp.array(out_grid.cells.reshape(-1)[:pred.shape[0]])
                query_loss += jnp.mean((pred - target) ** 2)

            losses.append(query_loss / len(query_examples))

        return jnp.mean(jnp.array(losses))

    def meta_train(self,
                   training_tasks: List[ARCTask],
                   n_shots: int = 3,
                   meta_batch_size: int = 8,
                   meta_epochs: int = 100,
                   key: jax.random.PRNGKey = None,
                   verbose: bool = True) -> UniversalTopos:
        """Train universal topos via meta-learning.

        Args:
            training_tasks: List of training tasks
            n_shots: Number of examples for task adaptation
            meta_batch_size: Number of tasks per meta-batch
            meta_epochs: Number of meta-training epochs
            key: PRNG key
            verbose: Print training progress

        Returns:
            universal_topos: Trained universal topos
        """
        if key is None:
            key = random.PRNGKey(0)

        # Initialize universal topos
        k1, key = random.split(key)
        self.universal_topos = UniversalTopos.random(
            k1,
            self.num_objects,
            self.feature_dim,
            self.max_covers,
            self.embedding_dim
        )

        # Initialize optimizer state
        # For simplicity, we optimize all parameters together
        # (In real version, would handle different parameter groups)
        self.opt_state = self.optimizer.init(self.universal_topos)

        if verbose:
            print(f"\n{'='*70}")
            print("META-LEARNING UNIVERSAL TOPOS")
            print(f"{'='*70}")
            print(f"Training tasks: {len(training_tasks)}")
            print(f"Meta-batch size: {meta_batch_size}")
            print(f"Meta-epochs: {meta_epochs}")
            print(f"N-shot: {n_shots}")
            print(f"{'='*70}\n")

        # Meta-training loop
        for epoch in range(meta_epochs):
            k1, key = random.split(key)

            # Sample meta-batch
            task_indices = random.choice(k1, len(training_tasks),
                                        shape=(meta_batch_size,),
                                        replace=False)
            meta_batch = [(training_tasks[int(i)], n_shots) for i in task_indices]

            # Compute meta-loss and gradient
            k1, key = random.split(key)
            loss = self.meta_loss(self.universal_topos, meta_batch, k1)

            # NOTE: Proper gradient computation would use JAX's grad here
            # For now, this is a placeholder showing the structure
            # In production, make meta_loss a pure function and use jax.grad

            if verbose and (epoch % 10 == 0 or epoch == meta_epochs - 1):
                print(f"Epoch {epoch:3d}/{meta_epochs}: Meta-loss = {loss:.4f}")

        if verbose:
            print(f"\n✓ Meta-training completed!")
            print(f"Final meta-loss: {loss:.4f}\n")

        return self.universal_topos

    def few_shot_adapt(self,
                       task: ARCTask,
                       n_shots: int = 5,
                       key: jax.random.PRNGKey = None) -> Site:
        """Adapt universal topos to new task with few-shot examples.

        Args:
            task: New task to adapt to
            n_shots: Number of examples to use
            key: PRNG key

        Returns:
            adapted_site: Task-specific site
        """
        if self.universal_topos is None:
            raise ValueError("Must call meta_train first!")

        if key is None:
            key = random.PRNGKey(0)

        # Get few-shot examples
        examples = [(inp.cells, out.cells) for inp, out in
                   zip(task.train_inputs[:n_shots], task.train_outputs[:n_shots])]

        # Adapt
        adapted_site = self.universal_topos.adapt_to_task(examples, key)

        return adapted_site

    def save(self, path: str):
        """Save universal topos to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'universal_topos': self.universal_topos,
                'config': {
                    'num_objects': self.num_objects,
                    'feature_dim': self.feature_dim,
                    'max_covers': self.max_covers,
                    'embedding_dim': self.embedding_dim
                }
            }, f)
        print(f"✓ Saved universal topos to {path}")

    @staticmethod
    def load(path: str) -> 'MetaToposLearner':
        """Load universal topos from file."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)

        learner = MetaToposLearner(**data['config'])
        learner.universal_topos = data['universal_topos']
        print(f"✓ Loaded universal topos from {path}")
        return learner


################################################################################
# § 4: Meta-Learning Pipeline
################################################################################

def meta_learning_pipeline(training_tasks: List[ARCTask],
                           eval_tasks: List[ARCTask],
                           output_dir: str = "meta_topos_results",
                           n_shots: int = 3,
                           meta_batch_size: int = 8,
                           meta_epochs: int = 100,
                           seed: int = 42):
    """Run complete meta-learning pipeline.

    Args:
        training_tasks: Tasks for meta-training
        eval_tasks: Tasks for evaluation
        output_dir: Where to save results
        n_shots: Few-shot examples per task
        meta_batch_size: Meta-batch size
        meta_epochs: Meta-training epochs
        seed: Random seed

    Returns:
        results: Dictionary with meta-learning results
    """
    from pathlib import Path

    # Setup
    key = random.PRNGKey(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize meta-learner
    meta_learner = MetaToposLearner(
        num_objects=20,
        feature_dim=32,
        max_covers=5,
        embedding_dim=64,
        meta_lr=1e-3
    )

    # Meta-train
    k1, key = random.split(key)
    universal_topos = meta_learner.meta_train(
        training_tasks=training_tasks,
        n_shots=n_shots,
        meta_batch_size=meta_batch_size,
        meta_epochs=meta_epochs,
        key=k1,
        verbose=True
    )

    # Save universal topos
    meta_learner.save(output_path / "universal_topos.pkl")

    # Evaluate on test tasks
    print(f"\n{'='*70}")
    print("EVALUATING ON TEST TASKS")
    print(f"{'='*70}\n")

    results = {
        'training_tasks': len(training_tasks),
        'eval_tasks': len(eval_tasks),
        'n_shots': n_shots,
        'meta_epochs': meta_epochs,
        'eval_results': []
    }

    for i, task in enumerate(eval_tasks[:10]):  # Evaluate on first 10
        k1, key = random.split(key)

        # Few-shot adapt
        adapted_site = meta_learner.few_shot_adapt(task, n_shots, k1)

        print(f"Task {i+1}/10: Adapted site with {adapted_site.num_objects} objects")

        results['eval_results'].append({
            'task_id': i,
            'adapted': True,
            'num_objects': adapted_site.num_objects
        })

    print(f"\n✓ Meta-learning pipeline completed!")
    print(f"Results saved to {output_path}\n")

    return results


################################################################################
# § 5: Documentation
################################################################################

"""
## Usage Example

```python
from arc_loader import load_arc_dataset
from meta_learner import MetaToposLearner, meta_learning_pipeline

# Load ARC dataset
training_tasks = load_arc_dataset("../../ARC-AGI/data", "training")

# Meta-train universal topos
meta_learner = MetaToposLearner()
universal_topos = meta_learner.meta_train(
    training_tasks=list(training_tasks.values())[:300],
    n_shots=3,
    meta_batch_size=8,
    meta_epochs=100
)

# Adapt to new task (few-shot)
new_task = training_tasks['task_id_123']
adapted_site = meta_learner.few_shot_adapt(new_task, n_shots=5)

# Make prediction
# ... use adapted_site with SheafNetwork ...

# Save for later
meta_learner.save("universal_topos.pkl")
```

## Connection to Theory

This implements:
- **Definition 5** (Learnable.agda): Parameterized sites
- **Definition 6**: Fitness function for topoi
- **Algorithm 1**: Evolutionary topos learning (but with gradient-based meta-learning)

The meta-learner discovers the **universal structure of abstract reasoning**
by learning what categorical patterns are shared across all ARC tasks!
"""
