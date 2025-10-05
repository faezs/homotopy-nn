Homotopy Neural Networks — Formal Framework (Cubical Agda)

Overview
- Formal, type‑checked framework for categorical and information‑theoretic models of neural networks in Cubical Agda (1Lab).
- Focus areas: directed graphs as functors, network summing functors, conservation via (co)equalizers, resource theory, transition systems, and integrated information (Φ) with homotopy‑theoretic scaffolding.

What’s Implemented
- Category‑theoretic plumbing: ΣC(G), conservation laws, resource theory constructs, transition system structure, and Section 8 scaffolding for Φ and homotopy.
- Clear proof surfaces with explicit postulates where theory is not yet formalized.

What’s Not Proven (Key Postulates)
- Lemma 8.1: feedforward networks have Φ = 0 (used by the demo; currently relies on postulates).
- Shannon entropy and basic properties; some probability/functorial constructions.
- Advanced homotopy theory (Gamma‑spaces, spectra) as future work scaffolding.

Why It Matters (Interpretability/Safety)
- Treats models as compositional categorical systems with explicit invariants and resource monotones.
- ΣC(G) and conservation laws expose bottlenecks and compositional contracts.
- Φ and homotopy views offer global measures of integration and higher‑order structure.

Quick Start
- Nix dev shell: `nix develop`
- Type‑check everything: `agda --library-file=./libraries src/Everything.agda`
- Demo checks: `bash scripts/check-demo.sh`

Optional: Python Homotopy Add‑On
- Generate a toy graph/codes from example data: `python scripts/trace_model.py --dataset examples/data.csv --mi-threshold 0.02`
- Or from weights: `python scripts/trace_model.py --weights examples/weights.json --edge-threshold 0.2`
- Compute Betti numbers: `python scripts/topology.py --graph out/G.json --out out/homology.json`

Demo Highlights
- Conservation: concrete categorical laws on small graphs.
- Transition systems: products/coproducts and grafting structure with clearly marked gaps (reachability/SCCs).
- Integrated information: definition site and the feedforward‑zero‑Φ claim called out as a proof obligation.
 - Homotopy view: build clique complex from exported graph and report Betti numbers; accurate via GUDHI in the Nix devshell.

Status and Honesty
- This is a rigorous framework with a transparent proof audit (see AUDIT.md, PRIORITY.md). Some central claims are intentionally left as postulates with clear pointers.

Contact/Pitch Angles
- Interpretability/Safety: formal specs of agent subsystems and information‑flow guarantees.
- Formal methods: proof‑engineering for ML; potential Lean port for core results.
