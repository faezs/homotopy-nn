Homotopy NN Demo Guide

Prerequisites
- Nix + flakes enabled.
- Dev shell: `nix develop` (Agda + 1Lab + Python stack). Verify with `agda --version` and `python3 --version`.

Fast Path
- Everything: `agda --library-file=./libraries src/Everything.agda`
- Focused demo (recommended): `bash scripts/check-demo.sh`

Modules Shown
- Neural.Network.Conservation: conservation laws via equalizers/coequalizers; one concrete functor law filled in.
- Neural.Computational.TransitionSystems: structure and operations; gaps (reachability/SCCs) are called out.
- Neural.Dynamics.IntegratedInformation: Φ definition and the explicit feedforward‑zero‑Φ claim (currently relying on postulates).

Notes on Postulates
- Some theorems are intentionally postulated to surface proof obligations (see AUDIT.md, PRIORITY.md). The demo points to exact locations.

Python Homotopy/Interpretability Add‑On
- Produce a toy graph and codes from data/weights:
  - `python scripts/trace_model.py --weights examples/weights.json --edge-threshold 0.2`
  - or `python scripts/trace_model.py --dataset examples/data.csv --mi-threshold 0.02`
- Compute Betti numbers of the clique complex:
  - `python scripts/topology.py --graph out/G.json --out out/homology.json`
- Outputs:
  - `out/G.json` with nodes/edges
  - `out/codes.npz` with boolean codes and metadata
  - `out/homology.json` with Betti numbers (uses GUDHI when available)

