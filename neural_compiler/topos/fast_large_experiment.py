"""
Fast Large-Scale Topos Experiment (Optimized)

Faster version with reduced epoch counts for rapid iteration.

Author: Claude Code + Human
Date: October 24, 2025
"""

import torch
import numpy as np
from large_scale_topos_experiment import run_single_experiment
import json
import time

def run_fast_experiment_suite(save_file: str = "topos_fast_results.json"):
    """Run optimized experiment suite with fewer epochs."""

    print("=" * 80)
    print("FAST LARGE-SCALE TOPOS QUANTIZATION EXPERIMENT")
    print("=" * 80)
    print()

    results = []
    start_total = time.time()

    # Experiment 1: Varying dataset size (reduced epochs)
    print("Experiment 1: Varying Dataset Size (4 vertices, 3-bit)")
    print("-" * 80)
    for num_samples in [200, 1000, 5000]:
        print(f"Running with {num_samples} samples...")
        result = run_single_experiment(
            num_vertices=4,
            num_samples=num_samples,
            num_bits=3,
            epochs=50,  # Reduced from 100
            verbose=False
        )
        results.append(result)
        print(f"  ✓ Test: {result['final_test_acc']:.1f}%, "
              f"Train: {result['final_train_acc']:.1f}%, "
              f"Time: {result['training_time']:.2f}s, "
              f"Königsberg: {'✓' if result['konigsberg_correct'] else '✗'}")

    # Experiment 2: Varying graph size
    print("\nExperiment 2: Varying Graph Size (1000 samples, 3-bit)")
    print("-" * 80)
    for num_vertices in [4, 6, 8]:
        print(f"Running with {num_vertices} vertices...")
        hidden_size = 16 if num_vertices <= 5 else 32
        result = run_single_experiment(
            num_vertices=num_vertices,
            num_samples=1000,
            num_bits=3,
            epochs=80,  # Reduced from 150
            hidden_size=hidden_size,
            verbose=False
        )
        results.append(result)
        print(f"  ✓ Params: {result['total_params']}, "
              f"Test: {result['final_test_acc']:.1f}%, "
              f"Time: {result['training_time']:.2f}s")

    # Experiment 3: Varying quantization
    print("\nExperiment 3: Varying Quantization (4 vertices, 1000 samples)")
    print("-" * 80)
    for num_bits in [2, 3, 4, 32]:
        bit_name = "Full" if num_bits == 32 else f"{num_bits}-bit"
        print(f"Running with {bit_name} quantization...")
        result = run_single_experiment(
            num_vertices=4,
            num_samples=1000,
            num_bits=num_bits,
            epochs=50,
            verbose=False
        )
        results.append(result)
        print(f"  ✓ Compression: {result['compression_ratio']:.1f}x, "
              f"Test: {result['final_test_acc']:.1f}%, "
              f"Storage: {result['effective_bits']/8:.0f} bytes")

    # Experiment 4: Stress test - Very large dataset
    print("\nExperiment 4: Stress Test (4 vertices, 10k samples, 3-bit)")
    print("-" * 80)
    print("Running stress test...")
    result = run_single_experiment(
        num_vertices=4,
        num_samples=10000,
        num_bits=3,
        epochs=80,
        test_size=200,
        verbose=False
    )
    results.append(result)
    print(f"  ✓ Test: {result['final_test_acc']:.1f}%, "
          f"Time: {result['training_time']:.2f}s")

    # Experiment 5: Large graph stress test
    print("\nExperiment 5: Large Graph Stress Test (8 vertices, 2k samples)")
    print("-" * 80)
    print("Running large graph test...")
    result = run_single_experiment(
        num_vertices=8,
        num_samples=2000,
        num_bits=3,
        epochs=100,
        hidden_size=64,
        verbose=False
    )
    results.append(result)
    print(f"  ✓ Params: {result['total_params']}, "
          f"Test: {result['final_test_acc']:.1f}%, "
          f"Time: {result['training_time']:.2f}s")

    total_time = time.time() - start_total

    # Save results
    print("\n" + "=" * 80)
    print(f"Saving results to {save_file}...")
    with open(save_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 80)

    print("\n📊 EXPERIMENT 1: Dataset Size Impact")
    print("-" * 80)
    dataset_exps = results[0:3]
    for r in dataset_exps:
        acc_bar = "█" * int(r['final_test_acc'] / 10)
        print(f"  {r['num_samples']:5d} samples: {acc_bar:10s} {r['final_test_acc']:5.1f}% "
              f"({r['training_time']:5.2f}s)")

    print("\n📏 EXPERIMENT 2: Graph Size Impact")
    print("-" * 80)
    graph_exps = results[3:6]
    for r in graph_exps:
        acc_bar = "█" * int(r['final_test_acc'] / 10)
        print(f"  {r['num_vertices']} vertices:  {acc_bar:10s} {r['final_test_acc']:5.1f}% "
              f"({r['total_params']:4d} params, {r['training_time']:5.2f}s)")

    print("\n🗜️  EXPERIMENT 3: Quantization Impact")
    print("-" * 80)
    quant_exps = results[6:10]
    for r in quant_exps:
        bit_name = "Full" if r['num_bits'] == 32 else f"{r['num_bits']}-bit"
        acc_bar = "█" * int(r['final_test_acc'] / 10)
        comp_ratio = r['compression_ratio']
        print(f"  {bit_name:8s}: {acc_bar:10s} {r['final_test_acc']:5.1f}% "
              f"({comp_ratio:4.1f}x compression, {r['effective_bits']/8:5.0f} bytes)")

    print("\n🚀 EXPERIMENT 4: Stress Test Results")
    print("-" * 80)
    stress = results[10]
    print(f"  10,000 samples: {stress['final_test_acc']:.1f}% test accuracy")
    print(f"  Training time: {stress['training_time']:.2f}s")
    print(f"  Samples/sec: {stress['num_samples'] * stress['epochs'] / stress['training_time']:.0f}")

    print("\n🔬 EXPERIMENT 5: Large Graph Test")
    print("-" * 80)
    large = results[11]
    print(f"  8 vertices: {large['final_test_acc']:.1f}% test accuracy")
    print(f"  Parameters: {large['total_params']}")
    print(f"  Storage: {large['effective_bits']/8:.0f} bytes")

    print("\n🏆 KEY FINDINGS")
    print("-" * 80)

    # Best accuracy
    best_acc = max(results, key=lambda r: r['final_test_acc'])
    print(f"  Best accuracy: {best_acc['final_test_acc']:.1f}% "
          f"({best_acc['num_vertices']}v, {best_acc['num_samples']}s, {best_acc['num_bits']}-bit)")

    # Best compression
    best_comp = max(results, key=lambda r: r['compression_ratio'])
    print(f"  Best compression: {best_comp['compression_ratio']:.1f}x "
          f"({best_comp['num_bits']}-bit)")

    # Fastest training
    fastest = min(results, key=lambda r: r['training_time'])
    print(f"  Fastest training: {fastest['training_time']:.2f}s "
          f"({fastest['num_samples']} samples)")

    # Smallest model
    smallest = min(results, key=lambda r: r['effective_bits'])
    print(f"  Smallest model: {smallest['effective_bits']/8:.0f} bytes "
          f"({smallest['num_bits']}-bit, {smallest['num_vertices']}v)")

    # Königsberg success rate
    konigsberg_results = [r for r in results if r['konigsberg_correct'] is not None]
    k_correct = sum(1 for r in konigsberg_results if r['konigsberg_correct'])
    print(f"  Königsberg success: {k_correct}/{len(konigsberg_results)} "
          f"({100*k_correct/len(konigsberg_results):.0f}%)")

    print("\n📈 SCALING ANALYSIS")
    print("-" * 80)

    # Time complexity
    print("  Dataset scaling (4v, 3-bit):")
    for i in range(len(dataset_exps)-1):
        r1, r2 = dataset_exps[i], dataset_exps[i+1]
        sample_ratio = r2['num_samples'] / r1['num_samples']
        time_ratio = r2['training_time'] / r1['training_time']
        print(f"    {r1['num_samples']}→{r2['num_samples']} samples: "
              f"{sample_ratio:.1f}x data → {time_ratio:.1f}x time")

    # Parameter scaling
    print("\n  Parameter scaling (1000s, 3-bit):")
    for r in graph_exps:
        input_size = r['num_vertices'] * r['num_vertices'] + r['num_vertices']
        print(f"    {r['num_vertices']} vertices: {input_size:3d} input features → "
              f"{r['total_params']:4d} params")

    print("\n⚡ PERFORMANCE METRICS")
    print("-" * 80)
    print(f"  Total experiment time: {total_time:.2f}s")
    print(f"  Total configurations: {len(results)}")
    print(f"  Average time per config: {total_time/len(results):.2f}s")
    print(f"  Total graphs processed: {sum(r['num_samples'] for r in results):,}")
    print(f"  Total training epochs: {sum(r['epochs'] for r in results):,}")

    print("\n" + "=" * 80)
    print("✅ EXPERIMENT COMPLETE!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = run_fast_experiment_suite()
