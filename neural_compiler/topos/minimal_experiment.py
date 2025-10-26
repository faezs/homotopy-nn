"""Minimal Topos Experiment - Saves incrementally"""

import torch
import numpy as np
from large_scale_topos_experiment import run_single_experiment
import json
import os

results_file = "minimal_results.json"

# Load existing if available
if os.path.exists(results_file):
    with open(results_file) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} existing results")
else:
    results = []

def save_result(r):
    results.append(r)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

print("MINIMAL TOPOS EXPERIMENT - 6 configs")
print("=" * 60)

# Config 1: Baseline
print("\n1. Baseline (4v, 500s, 3-bit, 30 epochs)...")
r = run_single_experiment(4, 500, 3, epochs=30, verbose=False)
save_result(r)
print(f"   ✓ Test: {r['final_test_acc']:.1f}%, Time: {r['training_time']:.2f}s")

# Config 2: Large dataset
print("\n2. Large dataset (4v, 2000s, 3-bit, 30 epochs)...")
r = run_single_experiment(4, 2000, 3, epochs=30, verbose=False)
save_result(r)
print(f"   ✓ Test: {r['final_test_acc']:.1f}%, Time: {r['training_time']:.2f}s")

# Config 3: Large graph
print("\n3. Large graph (8v, 500s, 3-bit, 30 epochs)...")
r = run_single_experiment(8, 500, 3, epochs=30, hidden_size=32, verbose=False)
save_result(r)
print(f"   ✓ Test: {r['final_test_acc']:.1f}%, Params: {r['total_params']}, Time: {r['training_time']:.2f}s")

# Config 4: 2-bit quantization
print("\n4. 2-bit quant (4v, 500s, 2-bit, 30 epochs)...")
r = run_single_experiment(4, 500, 2, epochs=30, verbose=False)
save_result(r)
print(f"   ✓ Test: {r['final_test_acc']:.1f}%, Compression: {r['compression_ratio']:.1f}x, Time: {r['training_time']:.2f}s")

# Config 5: 4-bit quantization
print("\n5. 4-bit quant (4v, 500s, 4-bit, 30 epochs)...")
r = run_single_experiment(4, 500, 4, epochs=30, verbose=False)
save_result(r)
print(f"   ✓ Test: {r['final_test_acc']:.1f}%, Compression: {r['compression_ratio']:.1f}x, Time: {r['training_time']:.2f}s")

# Config 6: Full precision
print("\n6. Full precision (4v, 500s, 32-bit, 30 epochs)...")
r = run_single_experiment(4, 500, 32, epochs=30, verbose=False)
save_result(r)
print(f"   ✓ Test: {r['final_test_acc']:.1f}%, Storage: {r['effective_bits']//8}B, Time: {r['training_time']:.2f}s")

# Final summary
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print("\n📊 All Configurations:")
for i, r in enumerate(results, 1):
    bits = "Full" if r['num_bits'] == 32 else f"{r['num_bits']}-bit"
    print(f"  {i}. {r['num_vertices']}v, {r['num_samples']:4d}s, {bits:5s}: "
          f"Test={r['final_test_acc']:5.1f}%, "
          f"Params={r['total_params']:4d}, "
          f"Time={r['training_time']:5.2f}s")

print(f"\n✅ Results saved to {results_file}")
