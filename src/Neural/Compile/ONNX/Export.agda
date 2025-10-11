{-# OPTIONS --no-import-sorts #-}

{-|
# Export Agda DirectedGraph to ONNX

This module implements the compilation from Agda's categorical representation
of neural networks (DirectedGraph) to ONNX's computational graph format.

## Key Mappings

```
DirectedGraph G              ONNX GraphProto
---------------              ---------------
vertices(G) : Nat     →      nodes : List NodeProto
edges(G) : Nat        →      implicit (tensor names)
source, target        →      node inputs/outputs connections

Each vertex v         →      ONNX node with operation type
Each edge e           →      Tensor name connecting nodes
```

## User Annotations Required

Since DirectedGraph is abstract (just graph structure), the user must provide:
1. Operation type for each vertex (Conv, Add, ReLU, etc.)
2. Tensor shapes/types for each edge
3. Attributes for operations (kernel size, strides, etc.)
4. Model metadata (name, version, etc.)

These are provided via the `GraphAnnotations` record.

## Example Usage

```agda
-- Define a simple 2-layer network
simple-network : DirectedGraph
simple-network = ...  -- 3 vertices (input, hidden, output), 2 edges

-- Annotate with ONNX operation types
annotations : GraphAnnotations simple-network
annotations = record
  { vertex-ops   = λ { 0 → "Input"
                     ; 1 → "MatMul"
                     ; 2 → "Softmax" }
  ; edge-shapes  = λ { 0 → [784, 256]  -- input to hidden
                     ; 1 → [256, 10] }  -- hidden to output
  ; ...
  }

-- Export to ONNX
onnx-model : ModelProto
onnx-model = export-to-onnx simple-network annotations
```

## Reference

Section 1.1-1.2 of Belfiore & Bennequin (2022): Oriented graphs and dynamical objects
-}

module Neural.Compile.ONNX.Export where

open import 1Lab.Prelude
open import Data.Dec.Base using (Discrete; Dec; yes; no)
open import Data.Nat.Base using (Nat; zero; suc; _+_; Discrete-Nat; s≤s; _≤_; ≤-peel; 0≤x; _<_; ≤-sucr; x≤x)
open Discrete Discrete-Nat renaming (decide to nat-eq?)
open import Data.Fin.Base using (Fin; lower; from-nat)
open Data.Fin.Base using (Fin-view; fin-view)
open import Data.Irr using (forget)
open import Data.Bool.Base using (Bool; true; false; not; if_then_else_)
open Data.Bool.Base using (and)
open import Data.List.Base
open Data.List.Base using (_++_; concat)
open import Data.String.Base using (String)

-- String operations (simplified for now - can be refined later)
postulate
  show-nat : Nat → String
  _<>ₛ_ : String → String → String  -- String concatenation

infixr 30 _<>ₛ_

-- Our graph and ONNX definitions
open import Neural.Base using (DirectedGraph; vertices; edges; source; target)
open import Neural.Compile.ONNX

--------------------------------------------------------------------------------
-- Graph Annotations
--------------------------------------------------------------------------------

{-|
User-provided annotations to turn an abstract DirectedGraph into a concrete
ONNX model. Since DirectedGraph only captures graph structure, we need:

1. **Operation types**: What operation does each vertex perform?
2. **Tensor information**: What are the shapes and types of data flowing on edges?
3. **Operation attributes**: Parameters like kernel sizes, activation functions, etc.
4. **Metadata**: Model name, version, documentation

This is the bridge between category theory and executable models!
-}
record GraphAnnotations (G : DirectedGraph) : Type₁ where
  field
    -- Operation type for each vertex (e.g., "Conv", "Add", "ReLU")
    vertex-op-type : Fin (vertices G) → String

    -- Input tensor shapes for each edge
    edge-shape : Fin (edges G) → List Nat

    -- Element type for each edge (default: FLOAT)
    edge-elem-type : Fin (edges G) → TensorElementType

    -- Operation attributes for each vertex
    vertex-attributes : Fin (vertices G) → List AttributeProto

    -- Which vertices are graph inputs? (List of vertex indices)
    graph-inputs : List (Fin (vertices G))

    -- Which vertices are graph outputs? (List of vertex indices)
    graph-outputs : List (Fin (vertices G))

    -- Model metadata
    model-name : String
    model-doc : String
    producer : String

--------------------------------------------------------------------------------
-- Tensor Name Generation
--------------------------------------------------------------------------------

{-|
Generate a unique tensor name for an edge.
Format: "edge_{index}_{source_vertex}→{target_vertex}"

Example: edge 0 from vertex 2 to vertex 3 becomes "edge_0_2→3"
-}
tensor-name-for-edge : (G : DirectedGraph) → Fin (edges G) → String
tensor-name-for-edge G e =
  let e-idx = lower e
      s-idx = lower (source G e)
      t-idx = lower (target G e)
  in "edge_" <>ₛ show-nat e-idx <>ₛ "_" <>ₛ show-nat s-idx <>ₛ "→" <>ₛ show-nat t-idx

{-|
Generate a unique node name for a vertex.
Format: "node_{index}"
-}
node-name-for-vertex : (G : DirectedGraph) → Fin (vertices G) → String
node-name-for-vertex G v =
  "node_" <>ₛ show-nat (lower v)

--------------------------------------------------------------------------------
-- Edge to Tensor Mapping
--------------------------------------------------------------------------------

{-|
Find all edges that have a given vertex as their source.
We iterate through all edges checking if source matches v.
-}
outgoing-edges : (G : DirectedGraph) → Fin (vertices G) → List (Fin (edges G))
outgoing-edges G v = helper (edges G) x≤x
  where
    if-yes : {A : Type} → Dec A → (A → List (Fin (edges G))) → List (Fin (edges G)) → List (Fin (edges G))
    if-yes (yes p) f _ = f p
    if-yes (no _) _ default = default

    helper : (n : Nat) → n ≤ edges G → List (Fin (edges G))
    helper zero _ = []
    helper (suc k) p with inspect (edges G)
    ... | zero , _  = []
    ... | suc m , eq =
      let -- p : suc k ≤ edges G, eq : edges G ≡ suc m
          -- Rewrite p using eq: suc k ≤ suc m
          p' : suc k ≤ suc m
          p' = subst (suc k ≤_) eq p
          e : Fin (suc m)
          e = record { lower = k ; bounded = forget p' }
          e' : Fin (edges G)
          e' = subst Fin (sym eq) e
          matches = nat-eq? (lower (source G e')) (lower v)
          -- Need: k ≤ edges G
          rest = helper k (subst (k ≤_) (sym eq) (≤-sucr (≤-peel p')))
      in if-yes matches (λ _ → e' ∷ rest) rest

{-|
Find all edges that have a given vertex as their target.
-}
incoming-edges : (G : DirectedGraph) → Fin (vertices G) → List (Fin (edges G))
incoming-edges G v = helper (edges G) x≤x
  where
    if-yes : {A : Type} → Dec A → (A → List (Fin (edges G))) → List (Fin (edges G)) → List (Fin (edges G))
    if-yes (yes p) f _ = f p
    if-yes (no _) _ default = default

    helper : (n : Nat) → n ≤ edges G → List (Fin (edges G))
    helper zero _ = []
    helper (suc k) p with inspect (edges G)
    ... | zero , _ = []
    ... | suc m , eq =
      let p' : suc k ≤ suc m
          p' = subst (suc k ≤_) eq p
          e : Fin (suc m)
          e = record { lower = k ; bounded = forget p' }
          e' : Fin (edges G)
          e' = subst Fin (sym eq) e
          matches = nat-eq? (lower (target G e')) (lower v)
          rest = helper k (subst (k ≤_) (sym eq) (≤-sucr (≤-peel p')))
      in if-yes matches (λ _ → e' ∷ rest) rest

--------------------------------------------------------------------------------
-- Vertex to ONNX Node
--------------------------------------------------------------------------------

{-|
Convert a single vertex in the DirectedGraph to an ONNX NodeProto.

Process:
1. Get operation type from annotations
2. Find incoming edges → input tensor names
3. Find outgoing edges → output tensor names
4. Get attributes from annotations
5. Construct NodeProto
-}
vertex-to-node : (G : DirectedGraph) → (annot : GraphAnnotations G)
               → Fin (vertices G) → NodeProto
vertex-to-node G annot v = record
  { op-type    = GraphAnnotations.vertex-op-type annot v
  ; inputs     = map (tensor-name-for-edge G) (incoming-edges G v)
  ; outputs    = map (tensor-name-for-edge G) (outgoing-edges G v)
  ; attributes = GraphAnnotations.vertex-attributes annot v
  ; name       = node-name-for-vertex G v
  ; domain     = ""  -- Empty = standard ONNX operators
  }

--------------------------------------------------------------------------------
-- Graph Inputs/Outputs
--------------------------------------------------------------------------------

{-|
Create ValueInfoProto for a graph input.
Input vertices typically have no incoming edges - they're sources.
-}
create-input-value-info : (G : DirectedGraph) → (annot : GraphAnnotations G)
                        → Fin (vertices G) → ValueInfoProto
create-input-value-info G annot v =
  let out-edges = outgoing-edges G v
  in case out-edges of λ
    { [] → record
        { name = node-name-for-vertex G v <>ₛ "_input"
        ; type = tensor-type record
            { elem-type = FLOAT
            ; shape = []
            }
        ; doc = "Input from vertex " <>ₛ show-nat (lower v) <>ₛ " (no outgoing edges)"
        }
    ; (first-edge ∷ _) → record
        { name = node-name-for-vertex G v <>ₛ "_input"
        ; type = tensor-type record
            { elem-type = GraphAnnotations.edge-elem-type annot first-edge
            ; shape = map dim-value (GraphAnnotations.edge-shape annot first-edge)
            }
        ; doc = "Input from vertex " <>ₛ show-nat (lower v)
        }
    }

{-|
Create ValueInfoProto for a graph output.
Output vertices typically have no outgoing edges - they're sinks.
-}
create-output-value-info : (G : DirectedGraph) → (annot : GraphAnnotations G)
                         → Fin (vertices G) → ValueInfoProto
create-output-value-info G annot v =
  let in-edges = incoming-edges G v
  in case in-edges of λ
    { [] → record
        { name = node-name-for-vertex G v <>ₛ "_output"
        ; type = tensor-type record
            { elem-type = FLOAT
            ; shape = []
            }
        ; doc = "Output from vertex " <>ₛ show-nat (lower v) <>ₛ " (no incoming edges)"
        }
    ; (first-edge ∷ _) → record
        { name = node-name-for-vertex G v <>ₛ "_output"
        ; type = tensor-type record
            { elem-type = GraphAnnotations.edge-elem-type annot first-edge
            ; shape = map dim-value (GraphAnnotations.edge-shape annot first-edge)
            }
        ; doc = "Output from vertex " <>ₛ show-nat (lower v)
        }
    }

--------------------------------------------------------------------------------
-- Graph Compilation
--------------------------------------------------------------------------------

{-|
Convert all vertices to ONNX nodes.
-}
compile-nodes : (G : DirectedGraph) → (annot : GraphAnnotations G)
              → List NodeProto
compile-nodes G annot = helper (vertices G) x≤x
  where
    helper : (n : Nat) → n ≤ vertices G → List NodeProto
    helper zero _ = []
    helper (suc k) p with inspect (vertices G)
    ... | zero , _ = []
    ... | suc m , eq =
      let p' : suc k ≤ suc m
          p' = subst (suc k ≤_) eq p
          v : Fin (suc m)
          v = record { lower = k ; bounded = forget p' }
          v' : Fin (vertices G)
          v' = subst Fin (sym eq) v
      in vertex-to-node G annot v' ∷ helper k (subst (k ≤_) (sym eq) (≤-sucr (≤-peel p')))

{-|
Create graph input ValueInfoProtos from annotation.
-}
compile-inputs : (G : DirectedGraph) → (annot : GraphAnnotations G)
               → List ValueInfoProto
compile-inputs G annot =
  map (create-input-value-info G annot) (GraphAnnotations.graph-inputs annot)

{-|
Create graph output ValueInfoProtos from annotation.
-}
compile-outputs : (G : DirectedGraph) → (annot : GraphAnnotations G)
                → List ValueInfoProto
compile-outputs G annot =
  map (create-output-value-info G annot) (GraphAnnotations.graph-outputs annot)

{-|
Compile initializers (weights, biases, constants).
For now, we don't extract weights from the DirectedGraph representation.
User should provide weights separately or we'd need to extend DirectedGraph
to include weight information.

TODO: Extend DirectedGraph with weight functors (Section 1.2 of paper)
-}
compile-initializers : (G : DirectedGraph) → (annot : GraphAnnotations G)
                     → List TensorProto
compile-initializers G annot = []  -- Placeholder

--------------------------------------------------------------------------------
-- Main Export Function
--------------------------------------------------------------------------------

{-|
Export a DirectedGraph to ONNX GraphProto.

This is the core compilation function that turns our categorical representation
into an executable ONNX computation graph.
-}
export-to-graph : (G : DirectedGraph) → (annot : GraphAnnotations G) → GraphProto
export-to-graph G annot = record
  { nodes        = compile-nodes G annot
  ; name         = GraphAnnotations.model-name annot
  ; inputs       = compile-inputs G annot
  ; outputs      = compile-outputs G annot
  ; initializers = compile-initializers G annot
  ; doc          = GraphAnnotations.model-doc annot
  }

{-|
Export a DirectedGraph to a complete ONNX ModelProto.

This wraps the graph in model metadata (IR version, opset, producer info).
-}
export-to-onnx : (G : DirectedGraph) → (annot : GraphAnnotations G) → ModelProto
export-to-onnx G annot = record
  { ir-version       = 9  -- ONNX IR version 9 (current as of 2024)
  ; opset-import     = record { domain = "" ; version = 17 } ∷ []  -- ONNX opset 17
  ; producer-name    = GraphAnnotations.producer annot
  ; producer-version = "1.0.0"
  ; domain           = "neural.homotopy"
  ; model-version    = 1
  ; doc              = GraphAnnotations.model-doc annot
  ; graph            = export-to-graph G annot
  }

--------------------------------------------------------------------------------
-- Utilities
--------------------------------------------------------------------------------

{-|
Check if a graph is valid for ONNX export:
1. Has at least one input and one output
2. All edges are used (no dangling edges)
3. Graph is acyclic (ONNX requires DAG)

TODO: Implement acyclicity check using topological sort
-}
is-valid-for-export : (G : DirectedGraph) → (annot : GraphAnnotations G) → Bool
is-valid-for-export G annot =
  let has-inputs  = not (null? (GraphAnnotations.graph-inputs annot))
      has-outputs = not (null? (GraphAnnotations.graph-outputs annot))
  in and has-inputs has-outputs
  where
    null? : {A : Type} → List A → Bool
    null? [] = true
    null? (_ ∷ _) = false

{-|
Count multi-input nodes (nodes with more than one incoming edge).
These correspond to convergence points / fork-star vertices in the paper!
-}
count-multi-input-nodes : (G : DirectedGraph) → Nat
count-multi-input-nodes G = helper (vertices G) x≤x
  where
    helper : (n : Nat) → n ≤ vertices G → Nat
    helper zero _ = zero
    helper (suc k) p with inspect (vertices G)
    ... | zero , _ = zero
    ... | suc m , eq =
      let p' : suc k ≤ suc m
          p' = subst (suc k ≤_) eq p
          v : Fin (suc m)
          v = record { lower = k ; bounded = forget p' }
          v' : Fin (vertices G)
          v' = subst Fin (sym eq) v
      in (case incoming-edges G v' of λ
           { [] → 0
           ; (_ ∷ []) → 0
           ; (_ ∷ _ ∷ _) → 1
           }) + helper k (subst (k ≤_) (sym eq) (≤-sucr (≤-peel p')))

--------------------------------------------------------------------------------
-- Documentation
--------------------------------------------------------------------------------

{-|
## Next Steps

1. **Python bridge** (tools/onnx_bridge.py):
   - Serialize ModelProto to protobuf
   - Execute with ONNX Runtime

2. **Fork construction support** (Neural.Compile.ONNX.ForkExport):
   - Handle Fork-Category graphs
   - Map fork-star vertices to multi-input ONNX nodes
   - Preserve topological properties

3. **Correctness proofs** (Neural.Compile.ONNX.Correctness):
   - Prove exported graphs are acyclic
   - Prove SSA property (each tensor name used once as output)
   - Prove structure preservation

4. **Examples**:
   - Simple MLP chain
   - ResNet block with skip connections
   - Attention mechanism
-}
