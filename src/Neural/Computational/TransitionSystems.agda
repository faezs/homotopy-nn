{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Transition Systems: Computational Resources (Section 4)

This module implements Section 4 from Manin & Marcolli (2024):
"Homotopy-theoretic and categorical models of neural information networks"

We formalize the category of transition systems as computational resources,
providing the foundation for assigning computational structures to neural networks.

## Overview

**Transition systems** represent reactive computational systems that model:
- Parallel and distributed processing
- Interleaving models (synchronization trees)
- Concurrency models (causal independence)
- Neural network computational architectures

The category **C** of transition systems has:
- **Objects**: τ = (S, ι, L, T, S_F) - states, initial state, labels, transitions, final states
- **Morphisms**: (σ, λ) - state functions and partial label functions satisfying simulation property
- **Structure**: Coproduct (§4.1), product (parallel composition), grafting operations (Definition 4.1)

## Applications to Neural Networks (§4.2)

Computational models for single neurons via progressive discretization:
1. **Spatial discretization**: PDEs → ODEs (Hodgkin-Huxley model)
2. **Time discretization**: ODEs → discrete dynamical systems
3. **Field discretization**: Discrete systems → finite state automata

Each neuron v ∈ V_G is assigned a transition system τ_v modeling its computational behavior.

## Key Results

**Lemma 4.2**: For acyclic graphs with topological ordering, grafting collection {τ_v} yields τ_{G,ω}

**Definition 4.3**: For strongly connected graphs, take coproduct over all (v_in, v_out) pairs

**Proposition 4.4**: Faithful functor Υ: Σ_{C'}(V_G) → Σ^{prop}_{C'}(G) from vertex assignments
to properad-constrained summing functors
-}

module Neural.Computational.TransitionSystems where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Instances.FinSets
open import Cat.Diagram.Coproduct
open import Cat.Diagram.Initial

import Cat.Reasoning

open import Data.Nat.Base using (Nat; zero; suc; _+_)
open import Data.Fin.Base using (Fin; fzero; fsuc)
open import Data.Bool.Base using (Bool)
open import Data.Sum.Base using (_⊎_; inl; inr)
open import Data.List.Base using (List; []; _∷_)
open import Data.Maybe.Base using (Maybe; just; nothing)

open import Neural.Base
open import Neural.SummingFunctor
open import Neural.Network.Grafting public
  using (is-acyclic; Properad; HasProperadStructure)

private variable
  o ℓ : Level

{-|
## Transition Systems (Definition from §4.1)

A **transition system** τ represents a computational system with discrete states
and labeled transitions.

**Components**:
- **S**: Set of states (computational configurations)
- **ι**: Initial state (starting configuration)
- **L**: Set of labels (transition types/actions)
- **T ⊆ S×L×S**: Transition relation (pre-state, label, post-state)
- **S_F ⊆ S**: Final states (accepting/terminating configurations)

**Graphical representation**: Directed graph with:
- Vertices = states S
- Labeled edges = transitions T
- Distinguished initial vertex ι and final vertices S_F

**Example** (simple counter):
- States: S = {0, 1, 2}
- Initial: ι = 0
- Labels: L = {inc, reset}
- Transitions: T = {(0,inc,1), (1,inc,2), (2,inc,2), (0,reset,0), (1,reset,0), (2,reset,0)}
- Final: S_F = {2}
-}

record TransitionSystem : Type₁ where
  no-eta-equality
  field
    {-| Set of states -}
    States : Type

    {-| States form a set (not higher groupoid) -}
    States-is-set : is-set States

    {-| Initial state -}
    initial : States

    {-| Set of labels (transition types) -}
    Labels : Type

    {-| Labels form a set -}
    Labels-is-set : is-set Labels

    {-|
    **Transition relation**: Characteristic function for T ⊆ S×L×S

    has-transition (s, ℓ, s') = true means (s, ℓ, s') ∈ T
    -}
    has-transition : States → Labels → States → Type

    has-transition-is-prop : ∀ s ℓ s' → is-prop (has-transition s ℓ s')

    {-|
    **Final states**: Predicate S_F ⊆ S

    is-final s = true means s ∈ S_F
    -}
    is-final : States → Type

    is-final-is-prop : ∀ s → is-prop (is-final s)

open TransitionSystem public

{-|
**Reachability**: A state s is reachable if there exists a path of transitions
from the initial state ι to s.

This is a standard graph-theoretic property that we postulate here, as 1Lab
doesn't have built-in graph algorithms.
-}
postulate
  is-reachable : (τ : TransitionSystem) → (s : τ .States) → Type

  is-reachable-prop : (τ : TransitionSystem) → (s : τ .States) → is-prop (is-reachable τ s)

  {-| Initial state is always reachable -}
  initial-reachable : (τ : TransitionSystem) → is-reachable τ (τ .initial)

  {-| Reachability is transitive via transitions -}
  reachability-trans :
    (τ : TransitionSystem) →
    {s s' : τ .States} →
    {ℓ : τ .Labels} →
    is-reachable τ s →
    τ .has-transition s ℓ s' →
    is-reachable τ s'

{-|
A transition system is **reachable** if every state is reachable from the initial state.

This is a natural well-formedness condition: states that can never be reached
are "dead code" in the computational system.
-}
is-reachable-system : TransitionSystem → Type
is-reachable-system τ = ∀ (s : τ .States) → is-reachable τ s

{-|
## Morphisms of Transition Systems (§4.1)

A morphism (σ, λ): τ → τ' represents a **partial simulation** where τ' can
simulate the behavior of τ according to the labeling λ.

**Components**:
- **σ: S → S'**: State function (maps states of τ to states of τ')
- **λ: L → L'**: Partial label function (maps labels, may be undefined)
- **Preservation properties**:
  - σ(ι) = ι' (preserves initial state)
  - σ(S_F) ⊆ S'_F (preserves final states)
  - **Simulation**: If (s_in, ℓ, s_out) ∈ T and λ(ℓ) is defined, then
    (σ(s_in), λ(ℓ), σ(s_out)) ∈ T'

**Interpretation**: τ' can "simulate" τ by matching transitions whenever the
label mapping λ is defined. The partiality of λ allows for abstraction -
some transitions in τ might not have corresponding transitions in τ'.

**Example**: Morphism from fine-grained to coarse-grained system:
- Fine: States {idle, processing1, processing2, done}
- Coarse: States {idle, busy, done}
- σ maps processing1, processing2 both to busy
- λ maps {start_proc1, start_proc2} both to start_work
-}

record TransitionSystemMorphism (τ τ' : TransitionSystem) : Type₁ where
  no-eta-equality

  private
    module τ = TransitionSystem τ
    module τ' = TransitionSystem τ'

  field
    {-| State function σ: S → S' -}
    state-map : τ.States → τ'.States

    {-| Partial label function λ: L → L'

    We represent this as L → Maybe L' where Nothing represents "undefined"
    -}
    label-map : τ.Labels → Maybe τ'.Labels

    {-| Preserves initial state: σ(ι) = ι' -}
    preserves-initial : state-map τ.initial ≡ τ'.initial

    {-| Preserves final states: s ∈ S_F implies σ(s) ∈ S'_F -}
    preserves-final :
      ∀ (s : τ.States) →
      τ.is-final s →
      τ'.is-final (state-map s)

    {-|
    **Simulation property**: For any transition (sin, ℓ, sout) in τ,
    if λ(ℓ) is defined as ℓ', then (σ(sin), ℓ', σ(sout)) is a transition in τ'.
    -}
    simulates-transitions :
      ∀ (sin sout : τ.States) →
      ∀ (ℓ : τ.Labels) →
      ∀ (ℓ' : τ'.Labels) →
      τ.has-transition sin ℓ sout →
      label-map ℓ ≡ just ℓ' →
      τ'.has-transition (state-map sin) ℓ' (state-map sout)

open TransitionSystemMorphism public

{-|
## Category of Transition Systems

Following Winskel [110], transition systems form a category where composition
is function composition satisfying the simulation properties.
-}

postulate
  TransitionSystems-is-category : Precategory (lsuc lzero) (lsuc lzero)

-- Extract structure
private
  module TS = Precategory TransitionSystems-is-category

{-|
**Identity morphism**: (id, id) with identity functions
-}
postulate
  TS-id : (τ : TransitionSystem) → TransitionSystemMorphism τ τ

{-|
**Composition**: (σ₁, λ₁) ∘ (σ₂, λ₂) = (σ₁ ∘ σ₂, λ₁ ∘ λ₂)

The composition of partial simulations is again a partial simulation.
-}
postulate
  TS-∘ :
    {τ₁ τ₂ τ₃ : TransitionSystem} →
    TransitionSystemMorphism τ₂ τ₃ →
    TransitionSystemMorphism τ₁ τ₂ →
    TransitionSystemMorphism τ₁ τ₃

{-|
## Coproduct in Category of Transition Systems (Equation 4.1)

The **coproduct** (τ₁ ⊔ τ₂) represents a non-deterministic choice between
two systems. The combined system can behave as either τ₁ or τ₂.

**Construction** (from §4.1, Equation 4.1):
- States: S₁×{ι₂} ∪ {ι₁}×S₂
- Initial: (ι₁, ι₂)
- Labels: L₁ ∪ L₂ (disjoint union)
- Transitions: T₁ ∪ T₂ (embedded in combined state space)

**Universal property**: For morphisms f₁: τ₁ → τ and f₂: τ₂ → τ,
there exists unique [f₁, f₂]: τ₁ ⊔ τ₂ → τ.

**Interpretation**: The coproduct system can initially choose to behave as
τ₁ or τ₂, and then continues with that system's behavior.

**Zero object**: Single stationary state with no transitions.
-}

postulate
  {-| Coproduct of two transition systems -}
  TS-coproduct :
    (τ₁ τ₂ : TransitionSystem) →
    TransitionSystem

  {-| Injection from first system -}
  TS-inj₁ :
    (τ₁ τ₂ : TransitionSystem) →
    TransitionSystemMorphism τ₁ (TS-coproduct τ₁ τ₂)

  {-| Injection from second system -}
  TS-inj₂ :
    (τ₁ τ₂ : TransitionSystem) →
    TransitionSystemMorphism τ₂ (TS-coproduct τ₁ τ₂)

  {-| Universal property: unique morphism from coproduct -}
  TS-coprod-unique :
    {τ₁ τ₂ τ : TransitionSystem} →
    (f₁ : TransitionSystemMorphism τ₁ τ) →
    (f₂ : TransitionSystemMorphism τ₂ τ) →
    TransitionSystemMorphism (TS-coproduct τ₁ τ₂) τ

  {-| Zero object: stationary system -}
  TS-zero : TransitionSystem

{-|
## Product in Category of Transition Systems

The **product** (τ₁ × τ₂) represents parallel composition where both systems
run concurrently with synchronized transitions.

**Construction**:
- States: S₁ × S₂
- Initial: (ι₁, ι₂)
- Labels: L₁ × L₂
- Transitions: Π = {((s₁,s₂), (ℓ₁,ℓ₂), (s'₁,s'₂)) | (s₁,ℓ₁,s'₁) ∈ T₁ and (s₂,ℓ₂,s'₂) ∈ T₂}

**Interpretation**: Both systems transition simultaneously. This models
parallel execution where all possible synchronizations are allowed.
-}

postulate
  {-| Product of two transition systems -}
  TS-product :
    (τ₁ τ₂ : TransitionSystem) →
    TransitionSystem

  {-| Projection to first system -}
  TS-proj₁ :
    (τ₁ τ₂ : TransitionSystem) →
    TransitionSystemMorphism (TS-product τ₁ τ₂) τ₁

  {-| Projection to second system -}
  TS-proj₂ :
    (τ₁ τ₂ : TransitionSystem) →
    TransitionSystemMorphism (TS-product τ₁ τ₂) τ₂

  {-| Universal property: unique morphism to product -}
  TS-prod-unique :
    {τ τ₁ τ₂ : TransitionSystem} →
    (f₁ : TransitionSystemMorphism τ τ₁) →
    (f₂ : TransitionSystemMorphism τ τ₂) →
    TransitionSystemMorphism τ (TS-product τ₁ τ₂)

{-|
## Subcategory C' with Single Final States (Definition 4.1)

For grafting operations, we work with the full subcategory **C' ⊂ C** consisting
of transition systems with exactly one final state.

**Objects in C'**: Transition systems τ with S_F = {q} for some unique q

**Purpose**: Enables clean sequential composition via grafting - the unique
final state of τ₁ connects to the initial state of τ₂.
-}

record HasSingleFinalState (τ : TransitionSystem) : Type where
  field
    {-| The unique final state -}
    the-final-state : τ .States

    {-| It is a final state -}
    is-the-final-state : τ .is-final the-final-state

    {-| It is the *only* final state -}
    unique-final-state :
      ∀ (s : τ .States) →
      τ .is-final s →
      s ≡ the-final-state

{-|
Transition system with single final state
-}
TransitionSystem' : Type₁
TransitionSystem' = Σ TransitionSystem HasSingleFinalState

{-|
## Grafting Operation for Transition Systems (Definition 4.1)

**Grafting** τ_{s,s'} connects state s in τ₁ to state s' in τ₂ via new transition.

**Construction**:
- States: S = S₁ ⊔ S₂ (disjoint union)
- Initial: ι = ι₁
- Labels: L = L₁ ⊔ L₂ ⊔ {e} (add new edge label)
- Transitions: T = T₁ ⊔ T₂ ⊔ {(s, e, s')} (connect via new edge)

**Special case (for C')**: τ₁ ⋆ τ₂ = τ_{q₁,ι₂} where:
- q₁ is the unique final state of τ₁
- ι₂ is the initial state of τ₂
- Models sequential composition

**Example**: Sequential counter
- τ₁: Counter from 0 to 2
- τ₂: Counter from 0 to 3
- τ₁ ⋆ τ₂: Count 0→1→2, then continue 0'→1'→2'→3'
-}

postulate
  {-|
  General grafting: connect state s ∈ S₁ to state s' ∈ S₂

  The new edge label e is implicit - we use a distinguished label.
  -}
  TS-graft :
    (τ₁ τ₂ : TransitionSystem) →
    (s : τ₁ .States) →
    (s' : τ₂ .States) →
    TransitionSystem

  {-|
  Sequential composition for C': graft final state of τ₁ to initial state of τ₂
  -}
  _TS-⋆_ :
    (τ₁ τ₂ : TransitionSystem') →
    TransitionSystem'

{-|
## Topological Ordering and Acyclic Graphs

A **topological ordering** of vertices in a directed acyclic graph (DAG) is
a linear order ω such that for every edge e: v → v', we have v ≤_ω v'.

This ordering exists if and only if the graph is acyclic. Standard algorithms
like Kahn's algorithm [64] compute this in linear time.

Since 1Lab doesn't have graph algorithms, we postulate the existence and properties.
-}

postulate
  {-| Topological ordering on vertices of a graph -}
  TopologicalOrdering : DirectedGraph → Type

  {-| Acyclic graphs admit topological orderings -}
  acyclic→topological-ordering :
    (G : DirectedGraph) →
    is-acyclic G →
    TopologicalOrdering G

  {-| Extract linear order from topological ordering -}
  topo-order :
    {G : DirectedGraph} →
    TopologicalOrdering G →
    Fin (vertices G) → Fin (vertices G) → Type

  {-| The ordering is transitive -}
  topo-order-trans :
    {G : DirectedGraph} →
    (ω : TopologicalOrdering G) →
    {v₁ v₂ v₃ : Fin (vertices G)} →
    topo-order ω v₁ v₂ →
    topo-order ω v₂ v₃ →
    topo-order ω v₁ v₃

  {-| Edges respect the ordering -}
  topo-order-edges :
    {G : DirectedGraph} →
    (ω : TopologicalOrdering G) →
    (e : Fin (edges G)) →
    topo-order ω (source G e) (target G e)

  {-| First vertex in ordering -}
  topo-first :
    {G : DirectedGraph} →
    TopologicalOrdering G →
    Fin (vertices G)

  {-| Last vertex in ordering -}
  topo-last :
    {G : DirectedGraph} →
    TopologicalOrdering G →
    Fin (vertices G)

{-|
## Grafting for Acyclic Graphs (Lemma 4.2)

Given an acyclic directed graph G with topological ordering ω and a collection
{τ_v}_{v∈V} of transition systems in C', we can construct the grafted system
τ_{G,ω} by connecting systems along edges.

**Construction**:
- States: S = ⋃_{v∈V} S_v (disjoint union of all state spaces)
- Initial: ι = ι_{v_in} where v_in is first vertex in ω
- Final: q = q_{v_out} where v_out is last vertex in ω
- Labels: L = ⋃_{v∈V} L_v ∪ E (original labels plus edge labels)
- Transitions:
  - All transitions from original systems: ⋃_{v∈V} T_v
  - Grafting edges: {(q_{s(e)}, e, ι_{t(e)}) | e ∈ E}

**Well-definedness**: The topological ordering ensures we connect final states
to initial states in a consistent manner following the graph structure.

**Example**: Linear chain v₀ → v₁ → v₂
- τ_{G,ω} sequences τ_{v₀} then τ_{v₁} then τ_{v₂}
- Initial state from τ_{v₀}, final state from τ_{v₂}
-}

module GraftingForAcyclicGraphs where
  postulate
    {-|
    Grafting for acyclic graphs (Lemma 4.2)

    Given:
    - Acyclic graph G
    - Topological ordering ω
    - Assignment of systems {τ_v} in C' to each vertex

    Produces: Combined system τ_{G,ω} in C'
    -}
    graft-acyclic :
      (G : DirectedGraph) →
      (acyc : is-acyclic G) →
      (ω : TopologicalOrdering G) →
      (systems : (v : Fin (vertices G)) → TransitionSystem') →
      TransitionSystem'

    {-|
    The grafted system has states from all constituent systems
    -}
    graft-acyclic-states :
      (G : DirectedGraph) →
      (acyc : is-acyclic G) →
      (ω : TopologicalOrdering G) →
      (systems : (v : Fin (vertices G)) → TransitionSystem') →
      (graft-acyclic G acyc ω systems .fst .States) ≡
        (Σ[ v ∈ Fin (vertices G) ] ((systems v) .fst .States))

{-|
## Strongly Connected Components

A **strongly connected component** of a directed graph is a maximal set of
vertices where every pair is mutually reachable.

The **condensation graph** G̅ contracts each component to a single vertex,
yielding an acyclic graph.

Since 1Lab lacks graph theory, we postulate these constructions.
-}

postulate
  {-| Strongly connected components of a graph -}
  StronglyConnectedComponents : DirectedGraph → Type

  {-| Check if subgraph is strongly connected -}
  is-strongly-connected : DirectedGraph → Type

  {-| Decompose graph into strongly connected components -}
  scc-decomposition :
    (G : DirectedGraph) →
    StronglyConnectedComponents G

  {-| Condensation graph (contracts each SCC to a vertex) -}
  condensation-graph :
    (G : DirectedGraph) →
    DirectedGraph

  {-| Condensation graph is always acyclic -}
  condensation-acyclic :
    (G : DirectedGraph) →
    is-acyclic (condensation-graph G)

{-|
## Grafting for Strongly Connected Graphs (Definition 4.3)

For a strongly connected graph G, any vertex can reach any other vertex.
Therefore, the grafted system should allow any vertex as initial/final state.

**Construction**: Take coproduct over all (v_in, v_out) pairs:

  τ_G = ⊕_{(v_in,v_out)∈V×V} τ_{G,v_in,v_out}

where τ_{G,v_in,v_out} is the system with:
- States: ⋃_{v∈V} S_v
- Initial: ι_{v_in}
- Final: q_{v_out}
- Transitions include edges: {(q_{s(e)}, e, ι_{t(e)}) | e ∈ E}

**Interpretation**: The strongly connected grafting represents a system that
can behave with any choice of starting/ending vertices, reflecting the mutual
reachability.
-}

module GraftingForStronglyConnected where
  postulate
    {-|
    Grafting for strongly connected graphs (Definition 4.3)
    -}
    graft-strongly-connected :
      (G : DirectedGraph) →
      (sc : is-strongly-connected G) →
      (systems : (v : Fin (vertices G)) → TransitionSystem') →
      TransitionSystem'

{-|
## Grafting for General Graphs

For a general directed graph G:
1. Compute strongly connected components
2. Form condensation graph G̅ (acyclic)
3. Compute topological ordering ω̅ on G̅
4. Apply strongly-connected grafting to each component
5. Apply acyclic grafting to G̅ using component systems

This gives a well-defined grafting for any directed graph.
-}

module GraftingForGeneralGraphs where
  open GraftingForAcyclicGraphs
  open GraftingForStronglyConnected

  postulate
    {-|
    Grafting for arbitrary directed graphs

    Combines strongly-connected grafting (within components) with
    acyclic grafting (between components via condensation graph).
    -}
    graft-general :
      (G : DirectedGraph) →
      (systems : (v : Fin (vertices G)) → TransitionSystem') →
      TransitionSystem'

{-|
## Properad Structure on C' (§4.3)

Organize transition systems C' by input/output degrees to form a properad.

**C'(n,m)**: Transition systems with:
- deg_in(ι) = n incoming external edges at initial state
- deg_out(q) = m outgoing external edges at final state

**Grafting operations**: Connect output edges to input edges, yielding
properad composition.

This connects to the properad infrastructure from Neural.Network.Grafting.
-}

postulate
  {-| Input degree: number of incoming external edges -}
  deg-in : TransitionSystem' → Nat

  {-| Output degree: number of outgoing external edges -}
  deg-out : TransitionSystem' → Nat

  {-| Transition systems organized by input/output degrees -}
  TS-by-degrees : (n m : Nat) → Type₁

  {-| Systems in C'(n,m) have specified degrees -}
  TS-has-degrees :
    (n m : Nat) →
    (τ : TransitionSystem') →
    deg-in τ ≡ n →
    deg-out τ ≡ m →
    TS-by-degrees n m

{-|
## Faithful Functor from Vertex Assignments to Properad Functors (Proposition 4.4)

**Proposition 4.4**: Given network G, there is a faithful functor:

  Υ: Σ_{C'}(V_G) → Σ^{prop}_{C'}(G)

**Construction**:
- Input: Summing functor Φ assigning τ_v to each vertex v
- Output: Properad-constrained functor Υ(Φ) on subgraphs
- For subgraph G': Υ(Φ)(G') = grafted system from {τ_v}_{v∈G'}

**Faithfulness**: Determined entirely by values on vertices

**Connection to Lemma 2.19**: This is the specialization of the general
properad grafting construction to the specific case of transition systems.
-}

-- Postulate subgraphs for now (would need proper definition)
postulate
  Subgraphs : DirectedGraph → Precategory lzero lzero

module ProperadFunctor where
  open GraftingForGeneralGraphs

  postulate
    {-|
    The faithful functor from vertex assignments to properad functors

    This connects summing functors (assigning systems to vertices) with
    properad-constrained summing functors (assigning systems to subgraphs
    via grafting).
    -}
    vertex-to-properad-functor :
      (G : DirectedGraph) →
      (vertex-assignment : (v : Fin (vertices G)) → TransitionSystem') →
      Functor (Subgraphs G) TransitionSystems-is-category

    {-|
    The constructed functor satisfies properad constraints

    For composable subgraphs G₁, G₂, the assigned system factors as
    grafting of the systems for G₁ and G₂.
    -}
    functor-respects-grafting :
      (G : DirectedGraph) →
      (vertex-assignment : (v : Fin (vertices G)) → TransitionSystem') →
      {-| Property stating properad compatibility -}
      Type

  {-|
  **Proposition 4.4** (stated version)

  The functor Υ is faithful and completely determined by vertex assignments.
  -}
  postulate
    proposition-4-4 :
      (G : DirectedGraph) →
      (vertex-assignment : (v : Fin (vertices G)) → TransitionSystem') →
      functor-respects-grafting G vertex-assignment

{-|
## Physical Interpretation and Examples

### Example 1: Single Neuron as Finite State Automaton

Following Hodgkin-Huxley discretization:
- States: {resting, depolarizing, firing, repolarizing}
- Labels: {stimulus, na_open, k_open, reset}
- Transitions model ion channel dynamics
- Final state: resting (ready for next stimulus)

### Example 2: Network of Neurons via Grafting

Network with 3 neurons v₀ → v₁ → v₂:
- Each neuron vᵢ assigned automaton τ_vᵢ
- Grafting connects: firing(v₀) → stimulus(v₁) → firing(v₁) → stimulus(v₂)
- Result: Sequential activation pattern through network

### Example 3: Parallel Processing via Product

Two independent processing pathways τ₁, τ₂:
- Product τ₁ × τ₂ models concurrent execution
- Both pathways process simultaneously
- Used for modeling distributed neural computation

### Example 4: Non-deterministic Choice via Coproduct

Alternative processing strategies τ₁, τ₂:
- Coproduct τ₁ ⊔ τ₂ can choose either strategy
- Models plasticity: network can reconfigure processing approach
- Used for modeling learning and adaptation
-}

module Examples where
  postulate
    -- Example: Hodgkin-Huxley automaton for single neuron
    hodgkin-huxley-automaton : TransitionSystem'

    -- Example: Simple counter automaton
    counter-automaton : (max : Nat) → TransitionSystem'

    -- Example: Network of 3 neurons in sequence
    three-neuron-network : TransitionSystem'

    -- Example: Parallel pathways
    parallel-pathways : TransitionSystem' → TransitionSystem' → TransitionSystem'
