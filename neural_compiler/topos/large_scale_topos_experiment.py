"""
Large-Scale Topos Quantization Experiment

Comprehensive evaluation of 3-bit quantized topos solver across:
1. Varying dataset sizes (100 → 10,000 samples)
2. Varying graph sizes (4 → 8 vertices)
3. Different quantization levels (2-bit, 3-bit, 4-bit, full precision)
4. Generalization tests on unseen graph structures

Author: Claude Code + Human
Date: October 24, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import time
import json
from tiny_quantized_topos import (
    GraphTopos, TinyQuantizedSheafNet, QuantizedLinear,
    generate_random_graphs
)


################################################################################
# § 1: Extended Quantization Levels
################################################################################

class FlexibleQuantizedLinear(nn.Module):
    """Linear layer with configurable bit-width quantization."""

    def __init__(self, in_features: int, out_features: int, num_bits: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.num_bits = num_bits

        # Compute quantization levels
        num_levels = 2 ** num_bits
        max_val = num_levels // 2 - 1
        min_val = -num_levels // 2
        self.levels = torch.tensor(
            list(range(min_val, max_val + 1)),
            dtype=torch.float32
        )

    def quantize(self, w: torch.Tensor) -> torch.Tensor:
        """Quantize weights to configured bit levels."""
        w_expanded = w.unsqueeze(-1)
        levels = self.levels.to(w.device).view(1, 1, -1)
        distances = torch.abs(w_expanded - levels)
        indices = torch.argmin(distances, dim=-1)
        return self.levels.to(w.device)[indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with quantized weights, STE for backward."""
        if self.num_bits == 32:  # Full precision
            return F.linear(x, self.weight, self.bias)

        w_quant = self.quantize(self.weight)
        w_ste = self.weight + (w_quant - self.weight).detach()
        return F.linear(x, w_ste, self.bias)


class FlexibleSheafNet(nn.Module):
    """Sheaf network with configurable quantization."""

    def __init__(self, num_vertices: int, num_bits: int = 3,
                 hidden_size: int = 16):
        super().__init__()

        input_size = num_vertices * num_vertices + num_vertices

        self.fc1 = FlexibleQuantizedLinear(input_size, hidden_size, num_bits)
        self.fc2 = FlexibleQuantizedLinear(hidden_size, 8, num_bits)
        self.fc3 = FlexibleQuantizedLinear(8, 1, num_bits)

        self.num_vertices = num_vertices
        self.num_bits = num_bits

    def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(graph_features))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def effective_bits(self) -> int:
        num_weights = sum(p.numel() for name, p in self.named_parameters()
                         if 'weight' in name)
        num_biases = sum(p.numel() for name, p in self.named_parameters()
                        if 'bias' in name)
        return self.num_bits * num_weights + 32 * num_biases


################################################################################
# § 2: Experiment Runner
################################################################################

def run_single_experiment(
    num_vertices: int,
    num_samples: int,
    num_bits: int,
    epochs: int = 100,
    lr: float = 0.1,
    hidden_size: int = 16,
    test_size: int = 50,
    verbose: bool = False
) -> Dict:
    """Run single experiment configuration."""

    # Create model
    model = FlexibleSheafNet(num_vertices, num_bits, hidden_size)

    # Generate data
    train_graphs = generate_random_graphs(num_vertices, num_samples, balanced=True)
    test_graphs = generate_random_graphs(num_vertices, test_size, balanced=True)

    # Training
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    start_time = time.time()
    train_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for graph, label in train_graphs:
            optimizer.zero_grad()
            features = graph.to_feature_vector()
            logit = model(features)
            target = torch.tensor([1.0 if label else 0.0])
            loss = F.binary_cross_entropy_with_logits(logit, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = (torch.sigmoid(logit) > 0.5).item()
            correct += (pred == label)

        avg_loss = total_loss / len(train_graphs)
        train_acc = 100 * correct / len(train_graphs)

        # Test
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for graph, label in test_graphs:
                features = graph.to_feature_vector()
                logit = model(features)
                pred = (torch.sigmoid(logit) > 0.5).item()
                test_correct += (pred == label)

        test_acc = 100 * test_correct / len(test_graphs)

        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                  f"Train={train_acc:.1f}%, Test={test_acc:.1f}%")

    elapsed = time.time() - start_time

    # Test on Königsberg if 4 vertices
    konigsberg_correct = None
    if num_vertices == 4:
        konigsberg_edges = [
            (0, 2), (0, 2), (1, 2), (1, 2),
            (0, 3), (1, 3), (2, 3)
        ]
        konigsberg = GraphTopos(4, konigsberg_edges)
        true_answer = konigsberg.has_eulerian_path()

        model.eval()
        with torch.no_grad():
            features = konigsberg.to_feature_vector()
            logit = model(features)
            pred = (torch.sigmoid(logit) > 0.5).item()
            konigsberg_correct = (pred == true_answer)

    return {
        'num_vertices': num_vertices,
        'num_samples': num_samples,
        'num_bits': num_bits,
        'epochs': epochs,
        'lr': lr,
        'hidden_size': hidden_size,
        'total_params': model.count_parameters(),
        'effective_bits': model.effective_bits(),
        'compression_ratio': model.count_parameters() * 32 / model.effective_bits(),
        'training_time': elapsed,
        'final_train_loss': train_losses[-1],
        'final_train_acc': train_accs[-1],
        'final_test_acc': test_accs[-1],
        'best_test_acc': max(test_accs),
        'konigsberg_correct': konigsberg_correct,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs
    }


################################################################################
# § 3: Large-Scale Experiment Suite
################################################################################

def run_experiment_suite(save_file: str = "topos_experiment_results.json"):
    """Run comprehensive experiment suite."""

    print("=" * 80)
    print("LARGE-SCALE TOPOS QUANTIZATION EXPERIMENT")
    print("=" * 80)
    print()

    results = []

    # Experiment 1: Varying dataset size (4 vertices, 3 bits)
    print("Experiment 1: Varying Dataset Size")
    print("-" * 80)
    for num_samples in [100, 500, 1000, 2000, 5000, 10000]:
        print(f"Running with {num_samples} samples...")
        result = run_single_experiment(
            num_vertices=4,
            num_samples=num_samples,
            num_bits=3,
            epochs=100,
            verbose=True
        )
        results.append(result)
        print(f"  → Test accuracy: {result['final_test_acc']:.1f}%, "
              f"Time: {result['training_time']:.2f}s")
        print()

    # Experiment 2: Varying graph size (1000 samples, 3 bits)
    print("\nExperiment 2: Varying Graph Size")
    print("-" * 80)
    for num_vertices in [4, 5, 6, 7, 8]:
        print(f"Running with {num_vertices} vertices...")
        # Adjust hidden size for larger graphs
        hidden_size = 16 if num_vertices <= 5 else 32
        result = run_single_experiment(
            num_vertices=num_vertices,
            num_samples=1000,
            num_bits=3,
            epochs=150,
            hidden_size=hidden_size,
            verbose=True
        )
        results.append(result)
        print(f"  → Params: {result['total_params']}, "
              f"Test accuracy: {result['final_test_acc']:.1f}%, "
              f"Time: {result['training_time']:.2f}s")
        print()

    # Experiment 3: Varying quantization (4 vertices, 1000 samples)
    print("\nExperiment 3: Varying Quantization Levels")
    print("-" * 80)
    for num_bits in [2, 3, 4, 8, 32]:
        bit_name = "Full precision" if num_bits == 32 else f"{num_bits}-bit"
        print(f"Running with {bit_name} quantization...")
        result = run_single_experiment(
            num_vertices=4,
            num_samples=1000,
            num_bits=num_bits,
            epochs=100,
            verbose=True
        )
        results.append(result)
        print(f"  → Compression: {result['compression_ratio']:.1f}x, "
              f"Test accuracy: {result['final_test_acc']:.1f}%, "
              f"Storage: {result['effective_bits']} bits")
        print()

    # Experiment 4: Scalability test (large graphs, large datasets)
    print("\nExperiment 4: Scalability Test")
    print("-" * 80)
    configs = [
        (6, 5000, 32, 200),  # 6 vertices, 5k samples, 32 hidden, 200 epochs
        (7, 3000, 64, 200),  # 7 vertices, 3k samples, 64 hidden, 200 epochs
        (8, 2000, 64, 250),  # 8 vertices, 2k samples, 64 hidden, 250 epochs
    ]

    for num_vertices, num_samples, hidden_size, epochs in configs:
        print(f"Running {num_vertices} vertices, {num_samples} samples, "
              f"{hidden_size} hidden units...")
        result = run_single_experiment(
            num_vertices=num_vertices,
            num_samples=num_samples,
            num_bits=3,
            epochs=epochs,
            hidden_size=hidden_size,
            verbose=True
        )
        results.append(result)
        print(f"  → Params: {result['total_params']}, "
              f"Test accuracy: {result['final_test_acc']:.1f}%, "
              f"Time: {result['training_time']:.2f}s")
        print()

    # Save results
    print("=" * 80)
    print(f"Saving results to {save_file}...")
    with open(save_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary statistics
    print("\nSUMMARY STATISTICS")
    print("=" * 80)

    # Group by experiment type
    dataset_size_exps = [r for r in results if r['num_vertices'] == 4 and
                         r['num_bits'] == 3 and r['num_samples'] <= 10000]
    graph_size_exps = [r for r in results if r['num_samples'] == 1000 and
                       r['num_bits'] == 3 and r['num_vertices'] <= 8]
    quant_exps = [r for r in results if r['num_vertices'] == 4 and
                  r['num_samples'] == 1000]

    print("\n1. Dataset Size Impact (4 vertices, 3-bit):")
    for r in dataset_size_exps:
        print(f"   {r['num_samples']:5d} samples → {r['final_test_acc']:5.1f}% test acc, "
              f"{r['training_time']:6.2f}s")

    print("\n2. Graph Size Impact (1000 samples, 3-bit):")
    for r in graph_size_exps:
        print(f"   {r['num_vertices']} vertices → {r['final_test_acc']:5.1f}% test acc, "
              f"{r['total_params']:4d} params, {r['training_time']:6.2f}s")

    print("\n3. Quantization Impact (4 vertices, 1000 samples):")
    for r in quant_exps:
        bit_name = "Full" if r['num_bits'] == 32 else f"{r['num_bits']}-bit"
        print(f"   {bit_name:8s} → {r['final_test_acc']:5.1f}% test acc, "
              f"{r['compression_ratio']:5.1f}x compression, "
              f"{r['effective_bits']:5d} bits")

    print("\n4. Best Performing Configurations:")
    sorted_by_acc = sorted(results, key=lambda r: r['final_test_acc'], reverse=True)
    for i, r in enumerate(sorted_by_acc[:5]):
        print(f"   #{i+1}: {r['final_test_acc']:.1f}% acc - "
              f"{r['num_vertices']}v, {r['num_samples']}s, {r['num_bits']}-bit, "
              f"{r['total_params']}p")

    print("\n5. Königsberg Test Results (4-vertex models):")
    konigsberg_results = [r for r in results if r['konigsberg_correct'] is not None]
    correct = sum(1 for r in konigsberg_results if r['konigsberg_correct'])
    print(f"   {correct}/{len(konigsberg_results)} models correctly solved Königsberg")

    print("\n" + "=" * 80)
    print("Experiment complete!")
    print(f"Total configurations tested: {len(results)}")
    print(f"Results saved to: {save_file}")
    print("=" * 80)

    return results


################################################################################
# § 4: Main
################################################################################

if __name__ == "__main__":
    results = run_experiment_suite()
