# Chains and Dynamical Objects: Computational vs Denotational

## TL;DR: It's Both!

The categorical approach in Belfiore & Bennequin (2022) is **neither purely computational nor purely denotational** - it's a **categorical semantics** that unifies both:

```
Syntax (Graph Γ) → Semantics (Functors) → Dynamics (Natural Transformations)
  ↑ computational      ↑ denotational          ↑ transformational
```

## The Three Layers of Meaning

### 1. Computational Graph (Syntax)

The oriented graph Γ is the **syntax** - the network architecture itself:
- Vertices = layers (L₀, L₁, ..., Lₙ)
- Edges = connections between layers
- Acyclic = feedforward (no recurrent loops)

**This is computational**: It describes the structure of the computation.

### 2. Functors to Set (Denotational Semantics)

The three functors give **meaning** to the syntax:

#### X^w: Activity Functor (for fixed weights w)
- **Denotational**: "Layer k means the set Xₖ of possible neuron activities"
- **Computational**: "Edge k→k+1 computes X^w_{k+1,k}: Xₖ → Xₖ₊₁"

**Both!** The mathematical structure describes both what it *means* and how it *computes*.

#### W = Π: Weight Functor
- **Denotational**: "The configuration space of all possible trained networks"
- **Computational**: "How we 'forget' used weights as we propagate forward"

**Key insight**: Weights are **morphisms in X^w** (transformations between activities) but **objects in W** (points in configuration space). This "dual representation" is crucial!

#### X: Total Dynamics (Crossed Product)
```agda
X(k) = Xₖ × Πₖ
     = (activity at layer k) × (all weights for layers ≥ k)
```

**This describes the state space of the learning system!**
- Not just one forward pass, but ALL possible forward passes for ALL possible weights
- A point in X represents a potential configuration of the network
- Morphisms in X represent both forward propagation AND weight updates

### 3. Natural Transformations (Dynamics)

From Section 1.4 (Theorem 1.1):
> "Backpropagation is a flow of natural transformations of W to itself"

This is **neither computational nor denotational** - it's **transformational**:
- It describes how the system **evolves** over time
- It's a morphism between functors, not within a functor
- It operates at the "meta-level" of changing the network's behavior

## Why Category Theory?

### Traditional Approach (Computational Only)
```python
# Just the algorithm
def forward(x, weights):
    for layer in layers:
        x = layer.activate(x, weights[layer])
    return x
```

### Categorical Approach (Unified)
```agda
-- The meaning
X^w : Functor C₀(Γ) Set
W   : Functor C₀(Γ) Set
X   : Functor C₀(Γ) Set

-- The dynamics
backprop : ∀ ξ₀ → NatTrans W W
```

**Benefits**:
1. **Compositional**: Build complex networks from simple pieces
2. **Equational reasoning**: Prove properties about the network
3. **Abstraction**: Separate "what" from "how"
4. **Generalization**: Works for any architecture (chains, forks, ResNets, etc.)

## From Chains to General DNNs (Section 1.3)

### The Problem

Chains are too simple! Modern networks have:
- Multiple inputs converging to one layer (ResNet skip connections)
- Multiple paths from input to output (complex topologies)

**Solution**: The **Fork Construction** (Figure 1.2)

### Fork Construction

When layers a', a'', ... converge at layer a:

**Before** (in original graph Γ):
```
a'  a''
 ↘  ↙
   a
```

**After** (in category C):
```
a' → A★ ← a''
     ↓
     A
     ↓
     a
```

New elements:
- **A★** (fork-star): Where inputs join
- **A** (fork-tang/socket): Intermediate point
- **Arrows**: a'→A★, a''→A★ (tines), A★→A (tang), a→A (handle)

### Why Fork?

The fork allows the functors to work:

**In X^w** (activity functor):
- X_A★ = X_a' × X_a'' (product of incoming activities)
- X_A = X_A★ (same space)
- X_a → X_A is the learned combination function

**In W** (weight functor):
- Same structure, but with weight spaces

**The sheaf condition**: At A★ vertices, we have a special covering that makes this a **Grothendieck topos**!

## The Topos C^ (Sections 1.3-1.5)

This is where it gets deep. The category C (opposite of C₀(Γ) with forks) has a Grothendieck topology J:

**Coverings**:
- At regular vertices x: only the maximal sieve (all morphisms into x)
- At fork-stars A★: ALSO the sieve of tines {a'→A★, a''→A★, ...}

**Result**: The topos C^ = Sh(C, J) of sheaves is:
1. **Localic**: Equivalent to sheaves on a topological space (Alexandrov topology)
2. **Coherent**: Has good logical properties
3. **Sub-extensional**: Propositions determined by subobjects

### What This Means

The **internal logic** of the topos corresponds to the **logic of the network**:

From Section 1.2:
> "the classifying object of subobjects Ω is given by the subobjects of 1;
> All these subobjects are increasing sequences (∅,...,∅,★,...,★). This
> can be interpreted as the fact that a proposition in the language (and
> internal semantic theory) of the topos is more and more determined when
> we approach the last layer."

**Translation**: As you move through the network from input to output, **information becomes more determinate**!

This connects to:
- Information theory (Section 3)
- Integrated information theory (IIT)
- The epistemology of neural computation

## Computational vs Denotational: Final Answer

It's a **false dichotomy**! The categorical approach transcends this distinction:

| Aspect | Role | Example |
|--------|------|---------|
| **Graph Γ** | Syntax/Computation | Network architecture |
| **Functors X^w, W, X** | Semantics/Denotation | What layers mean |
| **Morphisms in functors** | Computation | Forward propagation |
| **Natural transformations** | Dynamics | Backpropagation |
| **Topos C^** | Logic | Internal reasoning of network |
| **Sheaf condition** | Constraint | Information integration |

The beauty is that **all these levels coexist and interact** through the universal properties of category theory.

## Implementation Status

In our codebase:
- ✅ Basic DirectedGraph (Definition 2.6)
- ✅ Fork construction (Architecture.agda)
- ✅ Poset CX (Proposition 1.1)
- ✅ Alexandrov topology (Section 1.5)
- 🚧 Dynamical functors X^w, W, X (Chain.agda - started)
- 🚧 Natural transformations (backpropagation)
- 🚧 Free category C₀(Γ)
- 🚧 Grothendieck topology J
- 🚧 Sheafification

## Next Steps

To fully implement Section 1.2-1.4:

1. **Complete free category construction** from graph Γ
2. **Define functors to Sets** (need good category of manifolds/differentiable maps)
3. **Implement natural transformations** (backprop as flow)
4. **Connect to topos theory** (sheaf condition, coverage, logic)
5. **Add concrete examples** (MLP, ResNet, Transformer)

## References

- Belfiore & Bennequin (2022): "Topos and Stacks of Deep Neural Networks"
- Spivak et al. (FST19): Earlier categorical approach (functorial backprop)
- Functorial Semantics of Algebraic Theories (Lawvere)
- Sheaves in Geometry and Logic (Mac Lane & Moerdijk)
