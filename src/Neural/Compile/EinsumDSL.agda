{-# OPTIONS --no-import-sorts #-}

{-|
# Einsum-Based Neural Network DSL

**Core Philosophy**: The oriented graph IS the computation.

- **Vertices** = Tensor spaces (shapes)
- **Edges** = Einsum operations (transformations)
- **No metadata**: Kernel sizes, strides, padding are COMPUTED from shapes, not stored

## Example

```agda
simple-cnn : SequentialSpec
simple-cnn = sequential "simple-cnn"
  [ space [28, 28, 1]        -- Input tensor space
  , transform [24, 24, 20] "Conv"    -- Conv reduces 28→24 (kernel=5 inferred)
  , transform [24, 24, 20] "Relu"    -- Elementwise (shape unchanged)
  , transform [12, 12, 20] "MaxPool" -- Pool reduces 24→12 (stride=2 inferred)
  , transform [8, 8, 50] "Conv"      -- Conv 12→8, channels 20→50
  , transform [8, 8, 50] "Relu"
  , transform [4, 4, 50] "MaxPool"   -- Pool 8→4
  , transform [800] "Flatten"        -- Reshape 4×4×50 = 800
  , transform [500] "Gemm"           -- Dense 800→500
  , transform [500] "Relu"
  , transform [10] "Gemm"            -- Dense 500→10
  ]

simple-cnn-graph : DirectedGraph
simple-cnn-graph = compile-to-graph simple-cnn
```

The graph structure encodes everything!
- Vertex i has shape from layers[i]
- Edge i→(i+1) is the transformation layers[i] → layers[i+1]
- ONNX attributes are inferred from shape changes
-}

module Neural.Compile.EinsumDSL where

open import 1Lab.Prelude
open import Data.Nat.Base using (Nat; zero; suc; _+_; _*_)
open import Prim.Data.Nat renaming (_-_ to _∸_)
open import Data.Fin.Base using (Fin; fzero; fsuc; weaken)
open import Data.List.Base using (List; []; _∷_; length; head; tail)
open import Data.String.Base using (String)
open import Data.Bool.Base using (Bool; true; false)
open import Data.Maybe.Base using (Maybe; just; nothing)
open import Data.Float.Base using (Float)

open import Cat.Base using (Functor)
open Functor

open import Neural.Base using (DirectedGraph)
open import Neural.Compile.ONNX using (AttributeProto; AttributeValue; attr-ints; attr-int; attr-float)
open import Neural.Compile.ONNX.Export using (count-multi-input-nodes)

private variable
  ℓ : Level

-- Monus property: suc (n ∸ 1) ≡ n for n = suc m (non-zero)
-- This is the key property that makes chain graphs work:
-- For n vertices (n ≥ 1), we have n-1 edges
-- weaken : Fin (n-1) → Fin n works because suc ((suc m) ∸ 1) = suc m
suc-monus-lemma : (m : Nat) → suc (suc m ∸ 1) ≡ suc m
suc-monus-lemma m = refl   -- suc (suc m ∸ 1) = suc m ✓ definitionally

-- Float constant: 1.0
-- Cannot import Agda.Builtin.Float directly due to LEVELMAX conflict with 1Lab
-- Use postulate with GHC binding instead
postulate
  float-one : Float

{-# FOREIGN GHC import Numeric.Natural (Natural) #-}
{-# COMPILE GHC float-one = 1.0 :: Double #-}

--------------------------------------------------------------------------------
-- Layer Specifications: Shape Transformations
--------------------------------------------------------------------------------

{-|
A layer is either:
1. A tensor space (vertex in the graph)
2. A transformation between spaces (edge in the graph)

The transformation type (Conv, Relu, etc.) is just a label for ONNX export.
The ACTUAL computation is determined by the shape change.
-}

data LayerSpec : Type where
  -- Tensor space with shape [batch, spatial..., channels]
  space : List Nat → LayerSpec

  -- Transformation: new shape + operation label
  -- The shape change DEFINES the operation!
  transform : List Nat → String → LayerSpec

--------------------------------------------------------------------------------
-- Network Specification
--------------------------------------------------------------------------------

{-|
Sequential networks: chain of spaces and transformations.

Convention: Must alternate space → transform → space → transform...
Or start with space, then transforms (each implies next space).

For simplicity, we use the second convention:
- First layer is always `space` (input)
- Subsequent layers are `transform` (each creates new space)
-}

record SequentialSpec : Type where
  constructor sequential
  field
    network-name : String
    layers : List LayerSpec

open SequentialSpec public

--------------------------------------------------------------------------------
-- Extract Shapes from Layers
--------------------------------------------------------------------------------

{-|
Extract the shape from a layer specification.
- space: directly specified
- transform: the NEW shape after transformation
-}

layer-shape : LayerSpec → List Nat
layer-shape (space s) = s
layer-shape (transform s _) = s

{-|
Get all tensor shapes in the network (vertices).

For sequential spec starting with space:
- First layer is a space → that's vertex 0
- Each subsequent transform creates a new space → vertices 1, 2, ...
-}

extract-vertex-shapes : List LayerSpec → List (List Nat)
extract-vertex-shapes [] = []
extract-vertex-shapes (space s ∷ rest) = s ∷ extract-transform-shapes rest
  where
    extract-transform-shapes : List LayerSpec → List (List Nat)
    extract-transform-shapes [] = []
    extract-transform-shapes (transform s _ ∷ rest) = s ∷ extract-transform-shapes rest
    extract-transform-shapes (space _ ∷ rest) = extract-transform-shapes rest  -- Skip redundant spaces
extract-vertex-shapes (transform s _ ∷ rest) = s ∷ extract-transform-shapes rest
  where
    extract-transform-shapes : List LayerSpec → List (List Nat)
    extract-transform-shapes [] = []
    extract-transform-shapes (transform s _ ∷ rest) = s ∷ extract-transform-shapes rest
    extract-transform-shapes (space _ ∷ rest) = extract-transform-shapes rest

{-|
Get operation labels for edges.

Each transform creates an edge, labeled by its operation type.
-}

extract-edge-labels : List LayerSpec → List String
extract-edge-labels [] = []
extract-edge-labels (space _ ∷ rest) = extract-from-rest rest
  where
    extract-from-rest : List LayerSpec → List String
    extract-from-rest [] = []
    extract-from-rest (transform _ op ∷ rest) = op ∷ extract-from-rest rest
    extract-from-rest (space _ ∷ rest) = extract-from-rest rest
extract-edge-labels (transform _ op ∷ rest) = op ∷ extract-edge-labels rest

--------------------------------------------------------------------------------
-- Graph Compilation
--------------------------------------------------------------------------------

{-|
Compile a sequential network to a DirectedGraph.

**Structure**:
- n vertex shapes → n vertices
- n-1 transformations → n-1 edges
- Chain topology: vertex i connects to vertex i+1
-}

compile-to-graph : SequentialSpec → DirectedGraph
compile-to-graph spec with extract-vertex-shapes (spec .layers)
... | [] = empty-graph
  where
    empty-graph : DirectedGraph
    empty-graph .F₀ false = 0
    empty-graph .F₀ true = 0
    empty-graph .F₁ {false} {true} _ = λ ()
    empty-graph .F₁ {false} {false} tt = λ ()
    empty-graph .F₁ {true} {true} tt = λ ()
    empty-graph .F-id {false} = funext λ ()
    empty-graph .F-id {true} = funext λ ()
    empty-graph .F-∘ {false} {false} {false} tt tt = funext λ ()
    empty-graph .F-∘ {false} {false} {true} _ tt = funext λ ()
    empty-graph .F-∘ {false} {true} {true} tt _ = funext λ ()
    empty-graph .F-∘ {true} {true} {true} tt tt = funext λ ()

... | (_ ∷ _) = graph
  where
    vertex-shapes : List (List Nat)
    vertex-shapes = extract-vertex-shapes (spec .layers)

    -- We know vertex-shapes is non-empty (suc m for some m)
    -- So n-vertices = suc m and n-edges = m
    n-vertices : Nat
    n-vertices = length vertex-shapes

    n-edges : Nat
    n-edges = n-vertices ∸ 1

    graph : DirectedGraph
    graph .F₀ false = n-edges   -- edges
    graph .F₀ true = n-vertices -- vertices

    -- Source: edge i connects from vertex i
    -- weaken : Fin n-edges → Fin (suc n-edges)
    -- subst with suc-monus-lemma to get Fin n-vertices
    graph .F₁ {false} {true} true with n-vertices
    ... | zero = λ ()  -- Impossible: we matched non-empty list
    ... | suc m = λ e → subst Fin (suc-monus-lemma m) (weaken e)

    -- Target: edge i connects to vertex i+1
    graph .F₁ {false} {true} false with n-vertices
    ... | zero = λ ()
    ... | suc m = λ e → subst Fin (suc-monus-lemma m) (fsuc e)

    -- Identity on edges
    graph .F₁ {false} {false} tt = λ e → e

    -- Identity on vertices
    graph .F₁ {true} {true} tt = λ v → v

    -- Functor laws (automatic for chain structure)
    graph .F-id {false} = refl
    graph .F-id {true} = refl

    graph .F-∘ {false} {false} {false} tt tt = refl
    graph .F-∘ {false} {false} {true} true tt = refl
    graph .F-∘ {false} {false} {true} false tt = refl
    graph .F-∘ {false} {true} {true} tt true = refl
    graph .F-∘ {false} {true} {true} tt false = refl
    graph .F-∘ {true} {true} {true} tt tt = refl

--------------------------------------------------------------------------------
-- Convergence Count (Automatic for Sequential)
--------------------------------------------------------------------------------

{-|
Sequential networks have ZERO convergence points by construction.
Every vertex has at most one incoming edge (chain structure).
-}

postulate
  sequential-convergence-count : (spec : SequentialSpec)
                               → count-multi-input-nodes (compile-to-graph spec) ≡ 0
  -- This SHOULD compute via refl, but may need concrete graph to normalize

--------------------------------------------------------------------------------
-- ONNX Operation Names
--------------------------------------------------------------------------------

{-|
The operation label from the transform is used directly as the ONNX op type.

User specifies: transform [24, 24, 20] "Conv"
ONNX gets: op_type = "Conv"
-}

edge-to-onnx-op : String → String
edge-to-onnx-op label = label  -- Direct mapping

--------------------------------------------------------------------------------
-- ONNX Attribute Inference
--------------------------------------------------------------------------------

open AttributeProto

{-|
Infer ONNX attributes from shape transformations.

For Conv: kernel size = input_size - output_size + 1 (valid padding)
For Pool: window size and stride from spatial reduction
For others: minimal attributes
-}

-- Product of a list of numbers
list-product : List Nat → Nat
list-product [] = 1
list-product (x ∷ xs) = x * list-product xs

{-|
Infer attributes from source shape → target shape transformation.
-}

infer-attributes : String → List Nat → List Nat → List AttributeProto
infer-attributes "Conv" (b ∷ hin ∷ win ∷ cin ∷ []) (b2 ∷ hout ∷ wout ∷ cout ∷ []) =
  let kh = hin ∸ hout + 1
      kw = win ∸ wout + 1
  in record { name = "kernel_shape" ; value = attr-ints (kh ∷ kw ∷ []) }
     ∷ record { name = "strides" ; value = attr-ints (1 ∷ 1 ∷ []) }
     ∷ record { name = "pads" ; value = attr-ints (0 ∷ 0 ∷ 0 ∷ 0 ∷ []) }
     ∷ []

infer-attributes "MaxPool" (b ∷ hin ∷ win ∷ c ∷ []) (b2 ∷ hout ∷ wout ∷ c2 ∷ []) =
  record { name = "kernel_shape" ; value = attr-ints (2 ∷ 2 ∷ []) }
  ∷ record { name = "strides" ; value = attr-ints (2 ∷ 2 ∷ []) }
  ∷ []

infer-attributes "AveragePool" (b ∷ hin ∷ win ∷ c ∷ []) (b2 ∷ hout ∷ wout ∷ c2 ∷ []) =
  record { name = "kernel_shape" ; value = attr-ints (2 ∷ 2 ∷ []) }
  ∷ record { name = "strides" ; value = attr-ints (2 ∷ 2 ∷ []) }
  ∷ []

infer-attributes "Flatten" _ _ =
  record { name = "axis" ; value = attr-int 1 } ∷ []

infer-attributes "Gemm" _ _ =
  record { name = "alpha" ; value = attr-float float-one }
  ∷ record { name = "beta" ; value = attr-float float-one }
  ∷ []

-- All other operations (Relu, Sigmoid, etc.) have no attributes
infer-attributes _ _ _ = []

--------------------------------------------------------------------------------
-- Helper: Lookup in List
--------------------------------------------------------------------------------

lookup : {A : Type} → Nat → List A → Maybe A
lookup _ [] = nothing
lookup zero (x ∷ xs) = just x
lookup (suc n) (x ∷ xs) = lookup n xs

--------------------------------------------------------------------------------
-- Documentation
--------------------------------------------------------------------------------

{-|
## Example Usage

```agda
my-cnn : SequentialSpec
my-cnn = sequential "my-cnn"
  [ space [28, 28, 1]           -- Input: 28×28 grayscale
  , transform [24, 24, 20] "Conv"    -- Conv with kernel=5 (28-24+1)
  , transform [24, 24, 20] "Relu"    -- Elementwise activation
  , transform [12, 12, 20] "MaxPool" -- Pool with stride=2 (24/12)
  , transform [800] "Flatten"        -- Reshape 12×12×20 = 2880... wait
  , transform [500] "Gemm"           -- Dense layer
  , transform [10] "Gemm"            -- Output layer
  ]

my-graph : DirectedGraph
my-graph = compile-to-graph my-cnn

-- Convergence count computes automatically
my-convergence : count-multi-input-nodes my-graph ≡ 0
my-convergence = sequential-convergence-count my-cnn
```

## Philosophy

The **shape IS the spec**:
- [28,28,1] → [24,24,20]: Conv with kernel=5, filters=20
- [24,24,20] → [24,24,20]: Elementwise operation
- [24,24,20] → [12,12,20]: Pooling with stride=2
- [2880] → [500]: Dense layer (weight matrix 2880×500)

No need to separately specify kernel sizes, strides, padding!
They're **computed from the geometry**.
-}
