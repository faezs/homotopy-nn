"""
Tensor Program Integration with ARC-AGI Solver

This module bridges the tensor program encoding (from tensor_program_stack.py)
with the existing ARC-AGI evolutionary topos solver.

Key Integration Points:
1. TensorProgram stack structure → ARC grid topoi
2. Fork construction → Multi-path grid transformations
3. Semantic tensor encoding → Abstract pattern representation
4. Sheaf gluing → Spatial consistency constraints

Usage:
    # Create integrated solver
    solver = TensorARCSolver(population_size=20, generations=30)

    # Solve ARC task using tensor program framework
    best_topos, prediction = solver.solve_with_tensor_program(key, task)

    # Analyze learned categorical structure
    analyze_learned_topos(best_topos, task)
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Import existing ARC infrastructure
from arc_solver import (
    ARCGrid, ARCTask, ARCToposSolver, ARCReasoningNetwork,
    create_grid_site
)
from arc_loader import (
    load_arc_dataset, visualize_task, visualize_prediction,
    visualize_site_structure, evaluate_task
)
from evolutionary_solver import Site, SheafNetwork


################################################################################
# § 1: Tensor Program Components (from shared code)
################################################################################

@dataclass
class Layer:
    """Represents a layer in the neural network (vertex in the directed graph)."""
    id: int
    name: str
    dimension: int  # Number of neurons/features
    type: str  # 'input', 'hidden', 'output'


@dataclass
class Edge:
    """Represents a connection between layers (morphism in the category)."""
    source: int  # Source layer id
    target: int  # Target layer id
    weight_shape: Tuple[int, int]


class DirectedGraph:
    """The underlying directed graph Γ of the neural network (Section 1.1)."""

    def __init__(self):
        self.layers: Dict[int, Layer] = {}
        self.edges: List[Edge] = []
        self.partial_order = {}

    def add_layer(self, layer: Layer):
        self.layers[layer.id] = layer

    def add_edge(self, edge: Edge):
        self.edges.append(edge)
        self._update_partial_order(edge.source, edge.target)

    def _update_partial_order(self, source: int, target: int):
        if source not in self.partial_order:
            self.partial_order[source] = set()
        self.partial_order[source].add(target)


@dataclass
class Fork:
    """Fork construction for handling convergent multiplicity (Section 1.3)."""
    convergence_point: int
    star_node: int  # A*
    handle_node: int  # A
    input_layers: List[int]


class Category:
    """The category C(Γ) with fork construction (Section 1.3)."""

    def __init__(self, graph: DirectedGraph):
        self.graph = graph
        self.forks: Dict[int, Fork] = {}
        self._construct_forks()

    def _construct_forks(self):
        convergence_points = {}

        for edge in self.graph.edges:
            if edge.target not in convergence_points:
                convergence_points[edge.target] = []
            convergence_points[edge.target].append(edge.source)

        next_id = max(self.graph.layers.keys()) + 1
        for target, sources in convergence_points.items():
            if len(sources) > 1:
                fork = Fork(
                    convergence_point=target,
                    star_node=next_id,
                    handle_node=next_id + 1,
                    input_layers=sources
                )
                self.forks[target] = fork
                next_id += 2


@dataclass
class Fiber:
    """A fiber in the stack over a layer (Section 2)."""
    layer_id: int
    local_category: str
    invariance_group: Optional[str] = None


class Stack:
    """The stack F → C from Chapter 2."""

    def __init__(self, base_category: Category):
        self.base = base_category
        self.fibers: Dict[int, Fiber] = {}
        self._construct_fibers()

    def _construct_fibers(self):
        for layer_id, layer in self.base.graph.layers.items():
            if layer.type == 'conv':
                fiber = Fiber(
                    layer_id=layer_id,
                    local_category="TranslationGroupoid",
                    invariance_group="Z^2"
                )
            else:
                fiber = Fiber(
                    layer_id=layer_id,
                    local_category="TrivialCategory"
                )
            self.fibers[layer_id] = fiber

    def get_fiber(self, layer_id: int) -> Fiber:
        return self.fibers.get(layer_id)

    def transport(self, source: int, target: int, element):
        return element


class TensorEncoding:
    """Encodes semantic information as tensors (Section 3.4)."""

    def __init__(self, stack: Stack):
        self.stack = stack
        self.theories: Dict[str, any] = {}
        self.propositions: Dict[str, any] = {}

    def encode_theory(self, layer_id: int, theory: Dict[str, any]) -> jnp.ndarray:
        """Encode semantic theory at a layer as tensor."""
        fiber = self.stack.get_fiber(layer_id)

        if fiber.invariance_group == "Z^2":
            # CNN layer: spatial tensor
            h, w = 28, 28
            c = len(theory)
            tensor = jnp.zeros((h, w, c))

            for i, (prop, value) in enumerate(theory.items()):
                if isinstance(value, bool) and value:
                    tensor = tensor.at[:, :, i].set(1.0)
                elif isinstance(value, float):
                    tensor = tensor.at[:, :, i].set(value)
        else:
            # Regular layer: vector encoding
            tensor = jnp.array([float(v) if isinstance(v, (bool, float)) else 0.0
                               for v in theory.values()])

        return tensor

    def tensor_product(self, tensor1: jnp.ndarray, tensor2: jnp.ndarray) -> jnp.ndarray:
        """Tensor product for combining semantic information."""
        return jnp.outer(tensor1.flatten(), tensor2.flatten()).reshape(-1)

    def conditioning(self, theory_tensor: jnp.ndarray,
                    proposition_tensor: jnp.ndarray) -> jnp.ndarray:
        """Condition theory on proposition (semantic conditioning)."""
        return theory_tensor * proposition_tensor


################################################################################
# § 2: ARC Grid → Tensor Program Mapping
################################################################################

def arc_grid_to_directed_graph(task: ARCTask, grid_size: int = 30) -> DirectedGraph:
    """
    Convert ARC task structure to directed graph for tensor program.

    For ARC grids, we create a spatial processing graph:
    - Input layer: flattened grid cells
    - Processing layers: spatial transformations
    - Output layer: predicted grid cells

    Args:
        task: ARC task with training examples
        grid_size: Maximum grid dimension

    Returns:
        graph: DirectedGraph representing the processing structure
    """
    graph = DirectedGraph()

    # Layer 0: Input (flattened grid)
    max_cells = grid_size * grid_size
    graph.add_layer(Layer(0, "input", max_cells, "input"))

    # Layer 1: Local features (spatial neighborhoods)
    graph.add_layer(Layer(1, "local_features", 256, "hidden"))
    graph.add_edge(Edge(0, 1, (256, max_cells)))

    # Layer 2-3: Two parallel processing paths (for multi-object patterns)
    graph.add_layer(Layer(2, "path_a", 128, "hidden"))
    graph.add_layer(Layer(3, "path_b", 128, "hidden"))
    graph.add_edge(Edge(1, 2, (128, 256)))
    graph.add_edge(Edge(1, 3, (128, 256)))

    # Layer 4: Merge (convergence point - creates fork)
    graph.add_layer(Layer(4, "merge", 256, "hidden"))
    graph.add_edge(Edge(2, 4, (128, 128)))
    graph.add_edge(Edge(3, 4, (128, 128)))

    # Layer 5: Output (reconstructed grid)
    graph.add_layer(Layer(5, "output", max_cells, "output"))
    graph.add_edge(Edge(4, 5, (max_cells, 256)))

    return graph


def site_to_tensor_program_weights(site: Site, graph: DirectedGraph) -> Dict:
    """
    Convert evolved site structure to tensor program weight matrices.

    The site's adjacency and coverage encode the optimal information flow,
    which we translate to connection weights.

    Args:
        site: Evolved topos structure
        graph: Directed graph structure

    Returns:
        weights: Dictionary mapping (source, target) → weight matrix
    """
    weights = {}

    # Map site adjacency to edge weights
    for edge in graph.edges:
        source_dim, target_dim = edge.weight_shape

        # Use site's learned structure to initialize weights
        # This is a simplified mapping - could be more sophisticated
        if edge.source < site.num_objects and edge.target < site.num_objects:
            # Extract relevant submatrix from site adjacency
            base_weight = site.adjacency[edge.source, edge.target]

            # Initialize with Glorot scaling
            scale = jnp.sqrt(2.0 / (source_dim + target_dim))
            W = jnp.ones((target_dim, source_dim)) * base_weight * scale

            weights[(edge.source, edge.target)] = W
        else:
            # Random initialization for layers beyond site objects
            scale = jnp.sqrt(2.0 / (source_dim + target_dim))
            W = jnp.ones((target_dim, source_dim)) * scale
            weights[(edge.source, edge.target)] = W

    return weights


################################################################################
# § 3: Enhanced ARC Solver with Tensor Program
################################################################################

class TensorARCSolver(ARCToposSolver):
    """
    ARC solver enhanced with tensor program framework.

    Extends the evolutionary topos solver to use the complete tensor program
    stack structure, including:
    - Fork construction for convergent patterns
    - Stack fibers for invariances
    - Semantic tensor encoding
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor_program = None
        self.semantic_encoding = None

    def build_tensor_program_for_task(self, task: ARCTask) -> Dict:
        """
        Build complete tensor program for an ARC task.

        Args:
            task: ARC task

        Returns:
            program: Dictionary with graph, category, stack, encoding
        """
        # Build directed graph from task structure
        graph = arc_grid_to_directed_graph(task, self.grid_size)

        # Construct category with forks
        category = Category(graph)

        # Build stack with fibers
        stack = Stack(category)

        # Create tensor encoding
        tensor_encoding = TensorEncoding(stack)

        return {
            'graph': graph,
            'category': category,
            'stack': stack,
            'encoding': tensor_encoding,
            'forks': category.forks
        }

    def extract_semantic_context(self, task: ARCTask) -> Dict[str, any]:
        """
        Extract semantic information from ARC task examples.

        Analyzes training examples to identify abstract patterns:
        - Symmetries (rotation, reflection)
        - Repetitions
        - Color transformations
        - Spatial relationships

        Args:
            task: ARC task with training examples

        Returns:
            semantic_context: Dictionary of semantic properties
        """
        context = {}

        # Analyze input/output sizes
        size_changes = []
        for inp, out in zip(task.train_inputs, task.train_outputs):
            size_changes.append((
                out.height / inp.height if inp.height > 0 else 1.0,
                out.width / inp.width if inp.width > 0 else 1.0
            ))

        avg_h_scale = np.mean([s[0] for s in size_changes])
        avg_w_scale = np.mean([s[1] for s in size_changes])

        context['scales_grid'] = abs(avg_h_scale - 1.0) > 0.1 or abs(avg_w_scale - 1.0) > 0.1
        context['scale_factor_h'] = float(avg_h_scale)
        context['scale_factor_w'] = float(avg_w_scale)

        # Check for color transformations
        color_changes = []
        for inp, out in zip(task.train_inputs, task.train_outputs):
            if inp.height == out.height and inp.width == out.width:
                same_structure = jnp.sum(jnp.abs(inp.cells - out.cells)) > 0
                color_changes.append(same_structure)

        context['transforms_colors'] = any(color_changes) if color_changes else False

        # Check for symmetry (simplified)
        context['has_symmetry'] = False  # Would need more sophisticated analysis
        context['pattern_type'] = 'transformation'

        return context

    def solve_with_tensor_program(self, key: jax.random.PRNGKey,
                                  task: ARCTask,
                                  verbose: bool = True) -> Tuple[Site, ARCGrid, Dict]:
        """
        Solve ARC task using full tensor program framework.

        This is the main integration point that combines:
        1. Evolutionary topos optimization
        2. Tensor program structure
        3. Semantic encoding

        Args:
            key: PRNG key
            task: ARC task to solve
            verbose: Print progress

        Returns:
            best_site: Optimal topos structure
            prediction: Predicted output grid
            analysis: Dictionary with analysis of learned structure
        """
        if verbose:
            print(f"\n{'='*70}")
            print("TENSOR PROGRAM ARC SOLVER")
            print(f"{'='*70}")
            print(f"Training examples: {len(task.train_inputs)}")
            print(f"Test examples: {len(task.test_inputs)}")

        # Build tensor program structure
        if verbose:
            print("\n1. Building tensor program structure...")

        program = self.build_tensor_program_for_task(task)
        self.tensor_program = program

        if verbose:
            print(f"   ✓ Graph: {len(program['graph'].layers)} layers")
            print(f"   ✓ Forks: {len(program['forks'])} convergence points")
            print(f"   ✓ Stack fibers: {len(program['stack'].fibers)}")

        # Extract semantic context
        if verbose:
            print("\n2. Extracting semantic patterns...")

        semantic_context = self.extract_semantic_context(task)
        self.semantic_encoding = semantic_context

        if verbose:
            print(f"   ✓ Semantic properties detected:")
            for prop_key, prop_value in semantic_context.items():
                print(f"     - {prop_key}: {prop_value}")

        # Evolve optimal topos (using parent class method)
        if verbose:
            print("\n3. Evolving optimal topos structure...")

        k1, k2 = random.split(key, 2)
        best_site, prediction, fitness_history = self.solve_arc_task(
            k1, task, verbose=verbose
        )

        # Convert site to tensor program weights
        if verbose:
            print("\n4. Converting topos to tensor program weights...")

        weights = site_to_tensor_program_weights(best_site, program['graph'])

        if verbose:
            print(f"   ✓ Generated {len(weights)} weight matrices")

        # Analyze learned structure
        analysis = {
            'semantic_context': semantic_context,
            'tensor_program': program,
            'learned_weights': weights,
            'fitness_history': fitness_history,
            'category_structure': {
                'num_objects': len(program['graph'].layers),
                'num_morphisms': len(program['graph'].edges),
                'num_forks': len(program['forks']),
                'fork_details': {k: {
                    'inputs': v.input_layers,
                    'star': v.star_node,
                    'handle': v.handle_node
                } for k, v in program['forks'].items()}
            },
            'site_properties': {
                'num_objects': best_site.num_objects,
                'sparsity': float(jnp.sum(best_site.adjacency) / (best_site.num_objects ** 2)),
                'max_covers': best_site.max_covers
            }
        }

        if verbose:
            print(f"\n{'='*70}")
            print("RESULTS")
            print(f"{'='*70}")
            print(f"Category structure:")
            print(f"  - Layers: {analysis['category_structure']['num_objects']}")
            print(f"  - Morphisms: {analysis['category_structure']['num_morphisms']}")
            print(f"  - Forks: {analysis['category_structure']['num_forks']}")
            print(f"\nLearned topos:")
            print(f"  - Objects: {analysis['site_properties']['num_objects']}")
            print(f"  - Sparsity: {analysis['site_properties']['sparsity']:.1%}")
            print(f"  - Coverage families: {analysis['site_properties']['max_covers']}")
            print(f"{'='*70}\n")

        return best_site, prediction, analysis


################################################################################
# § 4: Visualization and Analysis
################################################################################

def analyze_learned_topos(analysis: Dict, task: ARCTask,
                          output_dir: str = "tensor_arc_analysis"):
    """
    Generate comprehensive analysis of learned topos structure.

    Args:
        analysis: Analysis dictionary from solve_with_tensor_program
        task: Original ARC task
        output_dir: Directory to save visualizations
    """
    from pathlib import Path
    import json

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print("TENSOR PROGRAM TOPOS ANALYSIS")
    print(f"{'='*70}\n")

    # 1. Category structure analysis
    print("1. CATEGORY STRUCTURE")
    print("-" * 70)
    cat = analysis['category_structure']
    print(f"Objects (layers): {cat['num_objects']}")
    print(f"Morphisms (connections): {cat['num_morphisms']}")
    print(f"Fork points (convergence): {cat['num_forks']}")

    if cat['num_forks'] > 0:
        print(f"\nFork details:")
        for fork_id, fork_info in cat['fork_details'].items():
            print(f"  Fork at layer {fork_id}:")
            print(f"    - Inputs: {fork_info['inputs']}")
            print(f"    - Star node: {fork_info['star']}")
            print(f"    - Handle node: {fork_info['handle']}")

    # 2. Site properties
    print(f"\n2. LEARNED TOPOS (SITE)")
    print("-" * 70)
    site = analysis['site_properties']
    print(f"Objects: {site['num_objects']}")
    print(f"Sparsity: {site['sparsity']:.2%}")
    print(f"Max covering families: {site['max_covers']}")

    # 3. Semantic analysis
    print(f"\n3. SEMANTIC PATTERNS DETECTED")
    print("-" * 70)
    semantic = analysis['semantic_context']
    for key, value in semantic.items():
        print(f"{key}: {value}")

    # 4. Save detailed JSON
    json_output = {
        'category': cat,
        'site': site,
        'semantic': {k: str(v) for k, v in semantic.items()}
    }

    with open(output_path / "analysis.json", 'w') as f:
        json.dump(json_output, f, indent=2)

    print(f"\n✓ Saved detailed analysis to {output_path / 'analysis.json'}")
    print(f"{'='*70}\n")


################################################################################
# § 5: Main Test Runner
################################################################################

def run_tensor_arc_test(task_id: str = "synthetic",
                        num_real_tasks: int = 10,
                        verbose: bool = True):
    """
    Run complete test of tensor program on ARC tasks.

    Args:
        task_id: If "synthetic", creates synthetic task; else loads from dataset
        num_real_tasks: Number of real ARC tasks to test on
        verbose: Print detailed output
    """
    key = random.PRNGKey(42)

    print("="*70)
    print("TENSOR PROGRAM ARC-AGI INTEGRATION TEST")
    print("="*70)
    print()

    # Initialize solver
    print("Initializing Tensor ARC Solver...")
    solver = TensorARCSolver(
        population_size=20,  # Small for faster testing
        generations=30,
        mutation_rate=0.15,
        grid_size=10,
        coverage_type="local"
    )
    print(f"✓ Solver configured:")
    print(f"  - Population: {solver.population_size}")
    print(f"  - Generations: {solver.generations}")
    print(f"  - Grid size: {solver.grid_size}")
    print()

    # Test on synthetic task first
    if task_id == "synthetic":
        print("="*70)
        print("PHASE 1: SYNTHETIC TASK TEST")
        print("="*70)
        print("\nCreating synthetic identity transformation task...")

        # Create simple synthetic task
        train_inputs = [
            ARCGrid.from_array(np.random.randint(0, 10, (5, 5))),
            ARCGrid.from_array(np.random.randint(0, 10, (5, 5)))
        ]
        train_outputs = [inp for inp in train_inputs]  # Identity

        test_inputs = [ARCGrid.from_array(np.random.randint(0, 10, (5, 5)))]
        test_outputs = [test_inputs[0]]

        synthetic_task = ARCTask(
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            test_inputs=test_inputs,
            test_outputs=test_outputs
        )

        print(f"✓ Created synthetic task")
        print(f"  - Training examples: {len(train_inputs)}")
        print(f"  - Grid size: {train_inputs[0].height}×{train_inputs[0].width}")

        # Solve with tensor program
        key, k_solve = random.split(key)
        best_site, prediction, analysis = solver.solve_with_tensor_program(
            k_solve, synthetic_task, verbose=verbose
        )

        # Evaluate
        correct = jnp.sum(prediction.cells == test_outputs[0].cells)
        total = prediction.height * prediction.width
        accuracy = float(correct) / total

        print(f"\nSynthetic Task Results:")
        print(f"  Accuracy: {accuracy:.1%} ({correct}/{total} cells correct)")

        # Analyze
        analyze_learned_topos(analysis, synthetic_task)

    # Test on real ARC tasks if requested
    if num_real_tasks > 0:
        print("\n" + "="*70)
        print(f"PHASE 2: REAL ARC TASKS (n={num_real_tasks})")
        print("="*70)
        print()

        # Load dataset
        print("Loading ARC training dataset...")
        try:
            tasks = load_arc_dataset(
                "../../ARC-AGI/data",
                split="training",
                limit=num_real_tasks
            )

            print(f"\nTesting on {len(tasks)} ARC tasks...")

            results = []
            for i, (tid, task) in enumerate(tasks.items()):
                print(f"\n{'-'*70}")
                print(f"Task {i+1}/{len(tasks)}: {tid}")
                print(f"{'-'*70}")

                key, k_solve = random.split(key)
                try:
                    best_site, prediction, task_analysis = solver.solve_with_tensor_program(
                        k_solve, task, verbose=False
                    )

                    # Evaluate
                    eval_results = evaluate_task(task, [prediction])

                    print(f"✓ Accuracy: {eval_results['avg_accuracy']:.1%}")
                    print(f"✓ Solved: {'YES' if eval_results['task_solved'] else 'NO'}")

                    results.append({
                        'task_id': tid,
                        'accuracy': eval_results['avg_accuracy'],
                        'solved': eval_results['task_solved'],
                        'analysis': task_analysis
                    })

                except Exception as e:
                    print(f"✗ Error: {e}")
                    results.append({
                        'task_id': tid,
                        'accuracy': 0.0,
                        'solved': False,
                        'error': str(e)
                    })

            # Summary
            print(f"\n{'='*70}")
            print("SUMMARY: REAL ARC TASKS")
            print(f"{'='*70}")

            solved = sum(1 for r in results if r['solved'])
            avg_acc = np.mean([r['accuracy'] for r in results])

            print(f"Tasks tested: {len(results)}")
            print(f"Tasks solved: {solved} ({solved/len(results):.1%})")
            print(f"Average accuracy: {avg_acc:.1%}")
            print(f"{'='*70}\n")

        except Exception as e:
            print(f"Could not load ARC dataset: {e}")
            print("Skipping real task testing...")

    print("\n" + "="*70)
    print("✓ TENSOR PROGRAM ARC INTEGRATION TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test tensor program integration with ARC-AGI"
    )
    parser.add_argument("--synthetic-only", action="store_true",
                       help="Only run synthetic test")
    parser.add_argument("--num-tasks", type=int, default=10,
                       help="Number of real ARC tasks to test")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    num_real = 0 if args.synthetic_only else args.num_tasks

    run_tensor_arc_test(
        task_id="synthetic",
        num_real_tasks=num_real,
        verbose=args.verbose
    )
