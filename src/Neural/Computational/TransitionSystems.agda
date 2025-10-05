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

open import Data.Nat.Base using (Nat; zero; suc; _+_; _==_)
open import Data.Fin.Base using (Fin; fzero; fsuc)
open import Data.Bool.Base using (Bool; if_then_else_)
open import Data.Sum.Base using (_⊎_; inl; inr)
open import Data.List.Base using (List; []; _∷_; _++_)
open import Data.Maybe.Base using (Maybe; just; nothing)

open import Neural.Base
open import Neural.SummingFunctor
open import Neural.Network.Grafting public
  using (is-acyclic; Properad; HasProperadStructure)

-- Algorithmic implementations over DirectedGraph
open import Neural.Graph.Algorithms as GraphAlgos public
  using ( verticesList
        ; edgesList
        ; neighborsOut
        ; neighborsIn
        ; reachableFrom
        ; coReachableTo
        ; topoSort
        ; sccs
        )

-- Small helpers
list-length : ∀ {A : Type} → List A → Nat
list-length [] = 0
list-length (_ ∷ xs) = suc (list-length xs)

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

{-| Topological ordering on vertices of a graph, with an order and a check. -}
record TopologicalOrdering (G : DirectedGraph) : Type where
  field
    ord : List (Fin (vertices G))
    edgesOK : Bool  -- cache: order-respects-edges G ord

open TopologicalOrdering public

-- Position of a vertex in a given order
position : {G : DirectedGraph} → TopologicalOrdering G → Fin (vertices G) → Nat
position {G} ω v = pos (ord ω) v
  where
    pos : List (Fin (vertices G)) → Fin (vertices G) → Nat
    pos []       v = 0
    pos (x ∷ xs) v = if GraphAlgos.eqFin? x v then 0 else suc (pos xs v)

-- Type-level ≤ on Nat
data _≤ₙ_ : Nat → Nat → Type where
  z≤n : ∀ {n} → 0 ≤ₙ n
  s≤s : ∀ {m n} → m ≤ₙ n → suc m ≤ₙ suc n

≤ₙ-trans : ∀ {m n p} → m ≤ₙ n → n ≤ₙ p → m ≤ₙ p
≤ₙ-trans z≤n _ = z≤n
≤ₙ-trans (s≤s mn) (s≤s np) = s≤s (≤ₙ-trans mn np)

-- Extract linear order from topological ordering (by positions)
topo-order :
  {G : DirectedGraph} →
  TopologicalOrdering G →
  Fin (vertices G) → Fin (vertices G) → Type
topo-order {G} ω v₁ v₂ = position ω v₁ ≤ₙ position ω v₂

-- The ordering is transitive
topo-order-trans :
  {G : DirectedGraph} →
  (ω : TopologicalOrdering G) →
  {v₁ v₂ v₃ : Fin (vertices G)} →
  topo-order ω v₁ v₂ →
  topo-order ω v₂ v₃ →
  topo-order ω v₁ v₃
topo-order-trans ω p₁₂ p₂₃ = ≤ₙ-trans p₁₂ p₂₃

-- Concrete, list-based topological orders and helpers
TopologicalOrderList : (G : DirectedGraph) → Type
TopologicalOrderList G = List (Fin (vertices G))

compute-topological-order : (G : DirectedGraph) → Maybe (TopologicalOrderList G)
compute-topological-order = topoSort

private
  posList : ∀ {n} → List (Fin n) → Fin n → Nat
  posList []       v = 0
  posList (x ∷ xs) v = if GraphAlgos.eqFin? x v then 0 else suc (posList xs v)

order-respects-edges : (G : DirectedGraph) → TopologicalOrderList G → Bool
order-respects-edges G ord =
  allEdgesOK (edgesList G)
  where
    ps = posList ord
    _<=?_ : Nat → Nat → Bool
    zero    <=? _       = true
    (suc m) <=? zero    = false
    (suc m) <=? (suc n) = m <=? n

    allEdgesOK : List (Fin (edges G)) → Bool
    allEdgesOK [] = true
    allEdgesOK (e ∷ es) =
      let s = source G e
          t = target G e
      in if (ps s) <=? (ps t) then allEdgesOK es else false

-- Acyclic graphs admit topological orderings (via Kahn's algorithm)
acyclic→topological-ordering :
  (G : DirectedGraph) →
  is-acyclic G →
  TopologicalOrdering G
acyclic→topological-ordering G _ with compute-topological-order G
... | just o  = record { ord = o ; edgesOK = order-respects-edges G o }
... | nothing = record { ord = verticesList G ; edgesOK = false }

-- Note: Edges-respect-order, first/last element depend on additional
-- properties of ω (being a valid topological order and non-empty graph).
-- They can be derived when ω is constructed by compute-topological-order
-- on a non-empty acyclic graph. If you want, I can add refined variants
-- guarded by those preconditions.


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

We provide concrete SCCs via Kosaraju from Graph.Algorithms.
-}

-- SCCs as a list of components (each component is a list of vertices)
SCCsList : (G : DirectedGraph) → Type
SCCsList G = List (List (Fin (vertices G)))

compute-sccs : (G : DirectedGraph) → SCCsList G
compute-sccs = sccs

{-| Strongly connected components of a graph -}
StronglyConnectedComponents : DirectedGraph → Type
StronglyConnectedComponents = SCCsList

{-| Check if a graph is strongly connected: exactly one SCC -}
is-strongly-connected : DirectedGraph → Type
is-strongly-connected G =
  let n = list-length (compute-sccs G) in
  if n == 1 then ⊤ else ⊥

{-| Decompose graph into strongly connected components -}
scc-decomposition :
  (G : DirectedGraph) →
  StronglyConnectedComponents G
scc-decomposition = compute-sccs

{-| Condensation graph (contracts each SCC to a vertex) -}
condensation-graph :
  (G : DirectedGraph) →
  DirectedGraph
condensation-graph = GraphAlgos.condensationDirected

-- Optional: Boolean acyclicity checker for the condensation graph
isAcyclicCondensation? : (G : DirectedGraph) → Bool
isAcyclicCondensation? G with compute-topological-order (condensation-graph G)
... | just _  = true
... | nothing = false

{-|
## Concrete SCC condensation summary

Provide a lightweight condensation summary (component count and edges between
components as Nat × Nat pairs).
-}

-- Find the first index of a vertex inside a list of components
index-of-vertex : {G : DirectedGraph} →
                  (comps : SCCsList G) →
                  Fin (vertices G) →
                  Nat
index-of-vertex [] v = 0
index-of-vertex (c ∷ cs) v = if member v c then 0 else suc (index-of-vertex cs v)
  where
    member : ∀ {n} → Fin n → List (Fin n) → Bool
    member x [] = false
    member x (y ∷ ys) = if GraphAlgos.eqFin? x y then true else member x ys

-- Condensation summary: number of components and inter-component edges
CondensationSummary : (G : DirectedGraph) → Type
CondensationSummary G = Σ Nat (λ n → List (Nat × Nat))

condensation-summary : (G : DirectedGraph) → CondensationSummary G
condensation-summary G =
  let comps = compute-sccs G
      n     = list-length' comps
      edges = map (λ e →
                    let i = index-of-vertex comps (source G e)
                        j = index-of-vertex comps (target G e) in
                    (i , j)) (edgesList G)
  in n , edges

  where
    -- Local list length (shadow-safe)
    list-length' : ∀ {A : Type} → List A → Nat
    list-length' [] = 0
    list-length' (_ ∷ xs) = suc (list-length' xs)

-- Concrete construction of the condensation graph as a DirectedGraph
compute-condensation-graph : (G : DirectedGraph) → DirectedGraph
compute-condensation-graph = GraphAlgos.condensationDirected

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

{-|
## Section 4.4: Larger-Scale Structures and Distributed Computing

Beyond local synaptic connectivity, neural networks exhibit larger-scale structures
such as non-local neuromodulation. Neuromodulators are generated in specific brain
regions (brainstem, basal forebrain) and transmitted to multiple regions via
long-range connections.

This section extends the transition system framework with:
1. **Time-delay automata** (Definition 4.5): Transitions carry time-delay blocks
2. **Distributed structures** (Definition 4.6): Partitions with neuromodulator vertices
3. **Category Gdist** (Definition 4.7): Graphs with distributed structure
4. **Modified summing functor** (Proposition 4.8): Grafting with distributed structure
-}

{-|
## Time-Delay Transitions (Definition 4.5)

**Automata with time delay blocks** generalize finite state automata by allowing
transitions to carry time-delay information. These produce a class of formal
languages strictly containing regular languages and incomparable to context-free
languages.

**Key idea**: Transitions labeled (a, n) where:
- a ∈ L': underlying label
- n ∈ ℤ₊: time delay block

**Example** (non-context-free language {aⁿbⁿcⁿ : n ∈ ℕ}):
- States: s₀, s₁, s₂
- Transitions: (a,0): s₀→s₁, (b,1): s₁→s₂, (c,2): s₂→s₀
- Time-zero: deposit 'a' symbols
- Time-one: deposit 'b' symbols
- Time-two: deposit 'c' symbols
- Result: {(a,0)ⁿ(b,1)ⁿ(c,2)ⁿ : n ∈ ℕ}

**Subcategory Cₜ ⊂ C'**: Transition systems with:
- Label set L = L' × ℤ₊
- Unique final state q
- Optional: incoming half-edges at ι, outgoing half-edges at q
-}

{-|
Time-delay label: pair of base label and time delay
-}
TimeDelayLabel : Type → Type
TimeDelayLabel L' = L' × Nat

{-|
Transition system with time-delay labels (Definition 4.5)
-}
record TimeDelayTransitionSystem : Type₁ where
  no-eta-equality
  field
    {-| Base label set L' -}
    BaseLabels : Type
    BaseLabels-is-set : is-set BaseLabels

    {-| Underlying transition system with labels L = L' × ℤ₊ -}
    underlying : TransitionSystem

    {-| Label set is time-delay labels -}
    labels-are-time-delay :
      underlying .Labels ≡ TimeDelayLabel BaseLabels

    {-| Has unique final state -}
    has-single-final : HasSingleFinalState underlying

open TimeDelayTransitionSystem public

{-|
Time-delay transition system with single final state
-}
TimeDelayTransitionSystem' : Type₁
TimeDelayTransitionSystem' = TimeDelayTransitionSystem

{-|
**Subcategories Cₜ(n,m)**: Time-delay systems with specified input/output degrees

- deg_in(ι) = n: n incoming half-edges at initial state
- deg_out(q) = m: m outgoing half-edges at final state
-}
postulate
  {-| Input degree for time-delay systems -}
  td-deg-in : TimeDelayTransitionSystem' → Nat

  {-| Output degree for time-delay systems -}
  td-deg-out : TimeDelayTransitionSystem' → Nat

  {-| Time-delay systems organized by input/output degrees -}
  TDT-by-degrees : (n m : Nat) → Type₁

  {-| Systems in Cₜ(n,m) have specified degrees -}
  TDT-has-degrees :
    (n m : Nat) →
    (τ : TimeDelayTransitionSystem') →
    td-deg-in τ ≡ n →
    td-deg-out τ ≡ m →
    TDT-by-degrees n m

{-|
**Remark**: When time delay is not explicitly written, it defaults to n=0,
corresponding to usual transitions with labeling set L'.
-}

{-|
## Distributed Structures on Graphs (Definition 4.6)

A **distributed structure** m on a directed graph G represents the partition
into N machines with neuromodulator transmission.

**Components**:
1. Partition of vertices V into N subsets Vᵢ (machines)
2. Partition of edges E by rule: e ∈ mᵢ iff target(e) ∈ Vᵢ
3. Subsets Vs,ᵢ, Vt,ᵢ ⊆ Vᵢ (source/target vertices for neuromodulation)
4. Augmented graph G₀ with:
   - New vertices v₀,ᵢ (neuromodulator nodes)
   - Incoming edges to v₀,ᵢ from any v ∈ Vs,ⱼ (any machine)
   - Outgoing edges from v₀,ᵢ to vertices in Vt,ᵢ (same machine)
   - Time delays nₑ ∈ ℤ₊ on all edges (nₑ = 0 for original edges)

**Physical interpretation**:
- v₀,ᵢ: Neuromodulator collection node in machine i
- Incoming edges: Neurons releasing neuromodulator
- Outgoing edges: Neuromodulated synapses
- Time delays: Transmission delays in multiples of Δt
-}

{-|
Machine partition: Assignment of vertices to machines
-}
MachinePartition : DirectedGraph → Nat → Type
MachinePartition G N = Fin (vertices G) → Fin N

{-|
Distributed structure on a directed graph (Definition 4.6)
-}
record DistributedStructure (G : DirectedGraph) : Type₁ where
  no-eta-equality
  field
    {-| Number of machines -}
    num-machines : Nat

    {-| Partition vertices into machines -}
    machine-of-vertex : MachinePartition G num-machines

    {-|
    Edge partitioning rule: edge e belongs to machine i iff target(e) ∈ Vᵢ
    -}
    machine-of-edge : Fin (edges G) → Fin num-machines
    edge-machine-via-target :
      ∀ (e : Fin (edges G)) →
      machine-of-edge e ≡ machine-of-vertex (target G e)

    {-|
    Source vertices Vs,ᵢ for neuromodulator transmission
    (vertices that release neuromodulator in machine i)
    -}
    is-source-vertex : Fin num-machines → Fin (vertices G) → Type
    is-source-vertex-prop :
      ∀ i v → is-prop (is-source-vertex i v)

    {-|
    Target vertices Vt,ᵢ for neuromodulation
    (vertices receiving neuromodulator in machine i)
    -}
    is-target-vertex : Fin num-machines → Fin (vertices G) → Type
    is-target-vertex-prop :
      ∀ i v → is-prop (is-target-vertex i v)

    {-|
    Augmented graph G₀ obtained by adding:
    - One neuromodulator vertex v₀,ᵢ per machine
    - Edges from source vertices to v₀,ᵢ
    - Edges from v₀,ᵢ to target vertices
    -}
    augmented-graph : DirectedGraph

    {-| Original graph G embeds in G₀ (vertices) -}
    original-vertices-count :
      vertices augmented-graph ≡ vertices G + num-machines

    {-| Embedding of original edges into augmented graph -}
    embed-edge : Fin (edges G) → Fin (edges augmented-graph)

    {-|
    Time delay assignment to edges in G₀
    - Original edges from G have delay 0
    - New edges have specified delays
    -}
    time-delay : Fin (edges augmented-graph) → Nat

    {-| Original edges have zero delay -}
    original-edges-zero-delay :
      ∀ (e : Fin (edges G)) →
      time-delay (embed-edge e) ≡ 0

open DistributedStructure public

{-|
**Condensation graph for distributed structure**

Given (G,m), the condensation graph Ḡ₀(m) is obtained by contracting each
subgraph Gᵢ (vertices Vᵢ and edges between them) to a single vertex.

**Property**: Ḡ₀(m) is always acyclic
-}
postulate
  {-| Condensation graph of distributed structure -}
  condensation-distributed :
    (G : DirectedGraph) →
    (m : DistributedStructure G) →
    DirectedGraph

  {-| Condensation of distributed structure is acyclic -}
  condensation-distributed-acyclic :
    (G : DirectedGraph) →
    (m : DistributedStructure G) →
    is-acyclic (condensation-distributed G m)

{-|
## Category Gdist (Definition 4.7)

**Objects**: Pairs (G,m) where:
- G: Finite directed graph
- m: Distributed structure on G
- Induced subgraphs Gᵢ (vertex set Vᵢ ∪ {v₀,ᵢ}) are strongly connected

**Morphisms**: α: (G,m) → (G',m') are graph morphisms α: G → G' that:
- Preserve distributed structure
- Map subgraphs αᵢ = α|Gᵢ : Gᵢ → G'ⱼ₍ᵢ₎ (compatible with partitions)

**Note**: We use natural transformations as graph morphisms (Definition 2.6),
allowing edge identifications but not edge contractions.
-}

{-|
Graph with distributed structure
-}
DistributedGraph : Type₁
DistributedGraph = Σ DirectedGraph DistributedStructure

{-|
Subgraph induced by machine i in distributed structure
-}
postulate
  induced-subgraph :
    (G : DirectedGraph) →
    (m : DistributedStructure G) →
    (i : Fin (m .num-machines)) →
    DirectedGraph

  {-| Induced subgraphs must be strongly connected -}
  induced-subgraph-strongly-connected :
    (G : DirectedGraph) →
    (m : DistributedStructure G) →
    (i : Fin (m .num-machines)) →
    Type

{-|
Morphism of distributed graphs (Definition 4.7)

α: (G,m) → (G',m') must:
1. Be a graph morphism α: G → G'
2. Preserve distributed structure (map machines to machines)
-}
record DistributedGraphMorphism (Gm G'm' : DistributedGraph) : Type₁ where
  no-eta-equality

  private
    G = Gm .fst
    m = Gm .snd
    G' = G'm' .fst
    m' = G'm' .snd

  field
    {-|
    Underlying graph morphism

    For now we postulate the structure - would be natural transformation
    of functors ·⇉· → FinSets (Definition 2.6)
    -}
    vertex-map : Fin (vertices G) → Fin (vertices G')
    edge-map : Fin (edges G) → Fin (edges G')

    {-| Machine mapping -}
    machine-map : Fin (m .num-machines) → Fin (m' .num-machines)

    {-|
    Compatibility: vertices map to vertices in compatible machines

    If v ∈ Vᵢ in G, then vertex-map(v) ∈ Vⱼ₍ᵢ₎ in G'
    -}
    preserves-machine-assignment :
      ∀ (v : Fin (vertices G)) →
      m' .machine-of-vertex (vertex-map v) ≡
        machine-map (m .machine-of-vertex v)

open DistributedGraphMorphism public

{-|
Category of distributed graphs (Definition 4.7)
-}
postulate
  Gdist : Precategory (lsuc lzero) (lsuc lzero)

  {-| Objects are distributed graphs -}
  Gdist-Ob : Precategory.Ob Gdist ≡ DistributedGraph

  {-| Morphisms are distributed graph morphisms -}
  Gdist-Hom :
    (Gm G'm' : DistributedGraph) →
    Precategory.Hom Gdist {!!} {!!} ≡ DistributedGraphMorphism Gm G'm'
    -- Note: Using {!!} as we need to transport along Gdist-Ob

{-|
## Modified Summing Functor for Distributed Systems (Proposition 4.8)

Given an object (G,m) ∈ Gdist and a summing functor Φ ∈ ΣCₜ(VG) with values
in the time-delay subcategory Cₜ, we construct a properad-constrained functor:

  Υ(Φ) ∈ Σᵖʳᵒᵖ_Cₜ(G,m)

**Construction procedure**:
1. For each machine i with vertex set Vᵢ:
   - Consider objects Φ(v) for v ∈ Vᵢ
   - Determine τGᵢ by grafting as in Definition 4.3
2. For condensation graph Ḡ₀(m):
   - Perform grafting τḠ₀(m),ω̄ as in Lemma 4.2
   - Combines the machine systems {τGᵢ}

**Result**: For (G',m') ∈ P(G,m) (category of subgraphs with compatible
distributed structure), Υ(Φ)(G',m') = τḠ₀(m'),ω̄

**Connection to Proposition 4.4**: This extends the faithful functor
construction to account for larger-scale distributed structure and
time-delay transitions.
-}

module DistributedSummingFunctor where
  open GraftingForGeneralGraphs
  open GraftingForAcyclicGraphs
  open GraftingForStronglyConnected

  {-|
  Category of subgraphs with compatible distributed structure
  -}
  postulate
    SubgraphsWithDistribution :
      (G : DirectedGraph) →
      (m : DistributedStructure G) →
      Precategory lzero lzero

  {-|
  Summing functor with values in time-delay subcategory
  -}
  SummingFunctorTD : DirectedGraph → Type₁
  SummingFunctorTD G =
    Functor (SubgraphsWithDistribution G {!!}) {!!}
    -- Would need proper category structure for TDT systems

  {-|
  Grafting within a single machine (Definition 4.3)

  For strongly connected subgraph Gᵢ, combine time-delay systems
  assigned to vertices in Vᵢ.
  -}
  postulate
    graft-within-machine :
      (G : DirectedGraph) →
      (m : DistributedStructure G) →
      (i : Fin (m .num-machines)) →
      (vertex-systems : (v : Fin (vertices G)) → TimeDelayTransitionSystem') →
      TimeDelayTransitionSystem'

  {-|
  Grafting between machines via condensation graph (Lemma 4.2)

  For acyclic condensation graph Ḡ₀(m), combine machine systems
  {τGᵢ} following topological ordering.
  -}
  postulate
    graft-between-machines :
      (G : DirectedGraph) →
      (m : DistributedStructure G) →
      (machine-systems : (i : Fin (m .num-machines)) → TimeDelayTransitionSystem') →
      TimeDelayTransitionSystem'

  {-|
  **Proposition 4.8**: Modified summing functor construction

  Given:
  - (G,m): Distributed graph
  - Φ ∈ ΣCₜ(VG): Summing functor assigning time-delay systems to vertices

  Produces:
  - Υ(Φ) ∈ Σᵖʳᵒᵖ_Cₜ(G,m): Properad-constrained functor on subgraphs

  The construction uses:
  1. Strongly-connected grafting within each machine
  2. Acyclic grafting via condensation graph between machines
  -}
  postulate
    proposition-4-8 :
      (G : DirectedGraph) →
      (m : DistributedStructure G) →
      (vertex-assignment : (v : Fin (vertices G)) → TimeDelayTransitionSystem') →
      {-| Produces properad-constrained summing functor -}
      Type₁

{-|
## Topological Questions (§4.4.3)

Several mathematical questions remain for future work:

1. **Protocol simplicial complexes**: Describe the topological structure of
   distributed computing algorithms implementing neuromodulated networks

2. **Relation to neural codes**: Investigate how the topology of protocol
   complexes relates to other topological structures (place field codes,
   overlap patterns, etc.)

3. **Embedded graph invariants**: Networks are embedded in 3-dimensional space.
   Can invariants of embedded graphs (e.g., fundamental group of complement,
   as in knot theory) carry relevant information about computational structure?

These questions connect discrete graph topology with spatial embedding geometry.
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

    -- Example: Time-delay automaton for {aⁿbⁿcⁿ}
    abc-time-delay-automaton : TimeDelayTransitionSystem'

    -- Example: Distributed neuromodulated network
    neuromodulated-network :
      (G : DirectedGraph) →
      (m : DistributedStructure G) →
      TimeDelayTransitionSystem'
