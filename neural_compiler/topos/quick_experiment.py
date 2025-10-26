"""Quick Topos Experiment - Ultra-fast version for demonstration"""

import torch
import numpy as np
from large_scale_topos_experiment import run_single_experiment
import json

print("=" * 70)
print("QUICK TOPOS QUANTIZATION EXPERIMENT")
print("=" * 70)

results = []

# Test 1: Small dataset scaling
print("\n1️⃣  Dataset Scaling (20 epochs each)")
for n in [200, 1000, 2000]:
    print(f"  Running {n} samples...", end=" ", flush=True)
    r = run_single_experiment(4, n, 3, epochs=20, verbose=False)
    results.append(r)
    print(f"✓ {r['final_test_acc']:.0f}% ({r['training_time']:.1f}s)")

# Test 2: Graph sizes
print("\n2️⃣  Graph Size Scaling (20 epochs each)")
for v in [4, 6, 8]:
    h = 16 if v <= 5 else 32
    print(f"  Running {v} vertices...", end=" ", flush=True)
    r = run_single_experiment(v, 500, 3, epochs=20, hidden_size=h, verbose=False)
    results.append(r)
    print(f"✓ {r['final_test_acc']:.0f}% ({r['total_params']} params, {r['training_time']:.1f}s)")

# Test 3: Quantization levels
print("\n3️⃣  Quantization Levels (20 epochs each)")
for bits in [2, 3, 4, 32]:
    name = "Full" if bits == 32 else f"{bits}-bit"
    print(f"  Running {name}...", end=" ", flush=True)
    r = run_single_experiment(4, 500, bits, epochs=20, verbose=False)
    results.append(r)
    print(f"✓ {r['final_test_acc']:.0f}% ({r['compression_ratio']:.1f}x, {r['effective_bits']//8}B)")

# Save
with open("quick_results.json", 'w') as f:
    json.dump(results, f, indent=2)

# Summary
print("\n" + "=" * 70)
print("📊 SUMMARY")
print("=" * 70)

print("\nDataset Impact:")
for i, r in enumerate(results[0:3]):
    print(f"  {r['num_samples']:4d} samples → {r['final_test_acc']:5.1f}%")

print("\nGraph Size Impact:")
for r in results[3:6]:
    print(f"  {r['num_vertices']}v ({r['total_params']:3d}p) → {r['final_test_acc']:5.1f}%")

print("\nQuantization Impact:")
for r in results[6:10]:
    n = "Full" if r['num_bits'] == 32 else f"{r['num_bits']}-bit"
    print(f"  {n:5s} ({r['compression_ratio']:4.1f}x) → {r['final_test_acc']:5.1f}%")

k_results = [r for r in results if r['konigsberg_correct'] is not None]
k_success = sum(1 for r in k_results if r['konigsberg_correct'])
print(f"\nKönigsberg: {k_success}/{len(k_results)} solved correctly")

print("\n✅ Done! Results saved to quick_results.json")
