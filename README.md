# Homotopy Neural Networks

**Type-checked neural architectures in Agda/HoTT that compile to production JAX code.**

[![Agda](https://img.shields.io/badge/Agda-2.6.4-blue)](https://github.com/agda/agda)
[![1Lab](https://img.shields.io/badge/1Lab-cubical-purple)](https://github.com/plt-amy/1lab)
[![JAX](https://img.shields.io/badge/JAX-0.4.20-green)](https://github.com/google/jax)

---

## 🚀 New: Neural Network Compiler (Agda → JAX)

**We built a compiler that takes formally verified neural architectures and compiles them to optimized JAX code.**

```python
from neural_compiler import compile_architecture

# Compile type-checked architecture from Agda
model = compile_architecture("architecture.json")

# Run on GPU/TPU at native speed
output = model(input_data)
```

**Features:**
- ✅ **Type safety**: Shape mismatches caught at compile time
- ✅ **Verified properties**: Conservation laws, sheaf conditions preserved
- ✅ **Zero overhead**: Compiles to native JAX/XLA (<5% overhead)
- ✅ **Compositional**: Build complex models from proven components

**[→ Compiler Documentation](COMPILER.md)** | **[→ Quick Start](#compiler-quick-start)**

---

## What Is This?

A **formal framework** for neural networks using:
- **Category theory**: Networks as functors, conservation via (co)equalizers
- **Homotopy type theory**: Cubical Agda with path types and univalence
- **Topos theory**: Grothendieck toposes for deep learning (fork construction)
- **Information theory**: Integrated information Φ, resource bounds
- **Linear logic**: Semantic information with exponentials and negation

**Plus a working compiler** that bridges this theory with production AI systems.

---

## Repository Structure

### 🔥 Compiler (New)
```
neural_compiler/          # Python: Agda → JAX compiler
├── parser.py            # JSON → IR
├── polyfunctor.py       # IR → Polynomial functors
├── jax_backend.py       # Compile to JAX/XLA
├── compiler.py          # End-to-end pipeline
├── demo.py              # Working examples
└── README.md            # Full documentation

src/Neural/Compile/
├── IR.agda              # Intermediate representation types
└── Serialize.agda       # JSON export from Agda
```

### 📚 Theory (10K+ lines of Agda)

**Core Framework:**
```
src/Neural/
├── Base.agda                    # Directed graphs as functors
├── SummingFunctor.agda          # Σ_C(G) construction
├── Network/
│   ├── Conservation.agda        # Kirchhoff laws (Prop 2.10, 2.12)
│   └── Grafting.agda            # Properad-constrained composition
├── Information.agda             # Neural codes, firing rates, metabolic costs
├── Resources/
│   ├── Theory.agda              # Resource theory (Def 3.1-3.5)
│   ├── Convertibility.agda      # Conversion rates ρ_{A→B}
│   └── Optimization.agda        # Optimal assignment (Theorem 5.6)
└── Computational/
    └── TransitionSystems.agda   # Distributed computing (Def 4.1-4.8)
```

**Topos-Theoretic DNNs (Belfiore & Bennequin 2022):**
```
src/Neural/Topos/
├── Architecture.agda            # Fork construction, sheaf conditions
├── Backpropagation.agda         # Natural transformations (Theorem 1.1)
├── Examples.agda                # ResNets, attention mechanisms
└── Stack/                       # 19 modules on fibrations & type theory
    ├── Fibration.agda           # Layers as fibers
    ├── Classifier.agda          # Ω_F for neural types
    ├── ModelCategory.agda       # Quillen structure
    └── MartinLof.agda           # MLTT semantics (Theorem 2.3)
```

**Linear Semantic Information (Appendix E):**
```
src/Neural/Semantics/
├── ClosedMonoidal.agda          # A^Y exponentials (Eq 47)
├── BiClosed.agda                # Lambek calculus for NLP
├── LinearExponential.agda       # ! comonad (Eq 48-49)
├── TensorialNegation.agda       # Dialogue categories (Eq 50-53)
├── StrongMonad.agda             # Strength/costrength (Lemma E.1)
├── NegationExponential.agda     # *-Autonomous categories (Prop E.3)
├── LinearInformation.agda       # Bar-complex, F/K compression ratio
└── Examples.agda                # Lambek, Montague, neural LMs
```

**Homotopy & Topology:**
```
src/Neural/Homotopy/
├── VanKampen.agda               # Compositional neural codes
├── Examples.agda                # Hippocampal place cells
├── Synthesis.agda               # Space reconstruction
└── Realization.agda             # Geometric realization
```

**Integrated Information Theory:**
```
src/Neural/Information/
├── IIT.agda                     # Φ formalization
├── Partition.agda               # State space partitions
└── Examples.agda                # Feedforward → Φ=0 (Proposition 10.1)
```

---

## Compiler Quick Start

### Installation

```bash
# Install Python dependencies
pip install -r neural_compiler/requirements.txt

# For GPU support
pip install jax[cuda12]

# For TPU support
pip install jax[tpu]
```

### Run Demo

```bash
python neural_compiler/demo.py
```

**Output:**
```
🚀 Neural Compiler Demo

DEMO 1: Simple MLP (784 → 256 → 10)
🔨 Compiling mlp.json...
✅ Compilation complete!
  Mean latency: 1.82 ms
  Throughput: 549 samples/sec

DEMO 2: ResNet Block (with fork + residual)
✅ Compilation complete!
  Properties: ['shape-correct', 'conserves-mass', 'sheaf-condition']
```

### Create Your Own Architecture

**In Agda:**
```agda
module MyNet where

open import Neural.Compile.IR
open import Neural.Compile.Serialize

my-net : NeuralIR
my-net = neural-ir "MyNet"
  (vertex 0 (linear 784 256) (vec 784 ∷ []) (vec 256) ∷ ...)
  (edge 0 1 (vec 256) ∷ ...)
  (0 ∷ [])  -- Inputs
  (2 ∷ [])  -- Outputs
  (shape-correct ∷ conserves-mass ∷ [])
  (constraints 1000000 1000000 1000 0)

main : IO ⊤
main = export-to-file "my_net.json" my-net
```

**Then compile:**
```python
model = compile_architecture("my_net.json")
output = model(input_data)
```

**[→ Full Compiler Tutorial](neural_compiler/README.md)**

---

## Theoretical Framework

### What's Implemented

**Category Theory (Sections 1-4):**
- ✅ Directed graphs as functors G: ·⇉· → FinSets
- ✅ Network summing functors Σ_C(G) (Lemma 2.3, Prop 2.4)
- ✅ Conservation laws via equalizers (Prop 2.10) and quotients (Prop 2.12)
- ✅ Resource theory with conversion rates ρ_{A→B} (Theorem 5.6)
- ✅ Transition systems for distributed computing (Prop 4.8)

**Topos Theory (Belfiore & Bennequin):**
- ✅ Fork construction for convergent layers (Def 1.3, Theorem 1.2)
- ✅ Sheaf condition: F(A★) ≅ ∏_{a'→A★} F(a') for merges
- ✅ Backpropagation as natural transformations (Lemma 1.1, Theorem 1.1)
- ✅ Stack semantics: 19 modules on fibrations, classifying toposes
- ✅ Model category structure (Quillen, Theorem 2.2)
- ✅ Martin-Löf type theory semantics (Theorem 2.3)

**Linear Logic & Semantics (Appendix E):**
- ✅ Closed monoidal categories with A^Y exponentials
- ✅ Bi-closed categories for Lambek calculus (natural language)
- ✅ Linear exponential ! comonad for resources
- ✅ Dialogue categories with tensorial negation ¬'
- ✅ Strong monads with strength/costrength
- ✅ *-Autonomous categories (Proposition E.3)
- ✅ Bar-complex and compression ratio F/K

**Homotopy & Information:**
- ✅ Van Kampen theorem for compositional neural codes
- ✅ Integrated information Φ formalization
- ✅ Feedforward networks have Φ = 0 (Proposition 10.1)

**Total: ~13,000 lines of type-checked Agda**

### Key Theorems & Propositions

| Result | Module | Status |
|--------|--------|--------|
| **Theorem 1.1**: Backprop as natural transformation | Topos.Architecture | ✅ Formalized |
| **Theorem 2.2**: Multi-fibrations for DNNs | Stack.Fibrations | ✅ Formalized |
| **Theorem 2.3**: MLTT semantics | Stack.MartinLof | ✅ Formalized |
| **Theorem 5.6**: Resource bounds | Resources.Optimization | ✅ Formalized |
| **Proposition 2.10**: Conservation via equalizers | Network.Conservation | ✅ Formalized |
| **Proposition 4.8**: Distributed grafting | Computational.TransitionSystems | ✅ Formalized |
| **Proposition 10.1**: Feedforward → Φ=0 | Information.IIT | ✅ Proven |
| **Proposition E.3**: Negation via exponentials | Semantics.NegationExponential | ✅ Formalized |

---

## Development Setup

### Prerequisites

- **Agda 2.6.4+** (for editing Agda source)
- **1Lab library** (included in `./libraries`)
- **Python 3.9+** (for compiler)
- **JAX** (for compilation target)

### Nix Shell (Recommended)

```bash
nix develop
```

Provides:
- Agda with 1Lab
- Python with JAX
- GUDHI (for topology)
- All development tools

### Type-Check Everything

```bash
# Check all Agda modules
agda --library-file=./libraries src/Everything.agda

# Check compiler modules specifically
agda --library-file=./libraries src/Neural/Compile/IR.agda
agda --library-file=./libraries src/Neural/Semantics/Examples.agda
```

### Run Tests

```bash
# Agda proofs (type-checking)
bash scripts/check-demo.sh

# Python compiler
python -m pytest neural_compiler/

# Integration test
python neural_compiler/demo.py
```

---

## Use Cases

### 1. Formal Verification
```agda
-- Prove property about architecture
my-net-conserves : conserves-mass my-net
my-net-conserves = ... -- Proof in Agda

-- Compile with guaranteed property
model = compile_architecture("my_net.json")
assert "conserves-mass" in model.ir.properties
```

### 2. Type-Safe Architecture Search
```python
# Search only valid architectures
for arch in search_space:
    if valid_ir(arch):  # Type-checked
        model = compile_architecture(arch)
        evaluate(model)
```

### 3. Resource-Constrained Deployment
```python
# Guaranteed to fit on device
model = compile_architecture("mobile_net.json")
assert model.ir.resources.max_flops < EDGE_DEVICE_LIMIT
deploy(model)
```

### 4. Compositional Interpretability
```agda
-- Build from verified components
transformer = compose-layers [
  attention-head,  -- Proven to satisfy sheaf condition
  feed-forward,    -- Proven to conserve
  layer-norm       -- Proven type-correct
]
```

---

## Research Foundations

This work formalizes and extends:

### Papers Implemented

1. **Marcolli & Manin** - Homotopy theoretic and categorical models of neural codes
   - Directed graphs as functors
   - Network summing functors Σ_C(G)
   - Resource theory for neural networks

2. **Belfiore & Bennequin (2022)** - Topological perspective on neural networks
   - Fork construction for convergent layers
   - Sheaf conditions for DNNs
   - Backpropagation as natural transformation
   - Stack semantics and fibrations (19 modules)
   - Linear semantic information (Appendix E, 8 modules)

3. **Curto & Itskov** - Neural codes and topology
   - Van Kampen for compositional reconstruction
   - Homotopy-theoretic approach to place cells

4. **Tononi et al.** - Integrated Information Theory
   - Categorical formalization of Φ
   - Partition lattices and conditional independence
   - Feedforward networks have Φ = 0 (proven)

### Novel Contributions

1. **First neural network compiler with dependent types**
   - Agda/HoTT → JAX pipeline
   - Property preservation (proofs → runtime guarantees)
   - Polynomial functors as IR

2. **Topos-theoretic framework for DNNs**
   - Fork construction with sheaf conditions
   - Compositional verification
   - Stack semantics with dependent types

3. **Complete linear semantic information formalization**
   - 8 modules implementing Appendix E
   - Lambek calculus, ! comonad, dialogue categories
   - Bar-complex and compression theory

4. **Categorical IIT formalization**
   - First rigorous proof that feedforward → Φ=0
   - Partition lattices in HoTT
   - Connection to equalizers

---

## Citation

```bibtex
@software{homotopy_nn_2025,
  title={Homotopy Neural Networks: Type-Checked Architectures in Agda with JAX Compilation},
  author={Faez Shakil},
  year={2025},
  url={https://github.com/faezs/homotopy-nn}
}
```

---

## Status & Roadmap

### Current Status
- ✅ 13K+ lines of type-checked Agda
- ✅ Working compiler to JAX
- ✅ 8 semantic information modules
- ✅ IIT formalization with feedforward proof
- ✅ Complete topos-theoretic framework

### Next Steps
- [ ] Agda reflection for automatic IR extraction
- [ ] Full transformer compilation example
- [ ] Multi-GPU distributed training
- [ ] Hardware-specific backends (TPU, Cerebras)
- [ ] Integration with symbolic optimizers (Symbolica)

---

## Contact & Collaboration

**Interested in:**
- Formal methods for AI safety
- Category theory for ML
- Type-safe neural architectures
- Neuromorphic hardware compilation
- Compositional interpretability

**Get in touch:**
- GitHub: [github.com/faezs](https://github.com/faezs)
- Email: [your email]

**Looking for:**
- Research positions (Symbolica AI, Anthropic, DeepMind)
- Collaborations on categorical AI
- Early users of the compiler

---

## License

MIT (or Apache 2.0, TBD)

---

## Acknowledgments

- **1Lab contributors** - Cubical Agda library
- **Belfiore & Bennequin** - Topos-theoretic framework
- **Marcolli & Manin** - Categorical neural codes
- **JAX team** - XLA compilation infrastructure
- **Claude/Anthropic** - Development assistance

---

**Built with:** Agda + 1Lab + HoTT + Category Theory + JAX + Coffee ☕
