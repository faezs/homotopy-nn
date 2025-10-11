{-# OPTIONS --no-import-sorts #-}

{-|
# ONNX Intermediate Representation in Agda

This module defines the ONNX graph structure in Agda, mirroring the
official ONNX protobuf specification. This allows us to:

1. Represent ONNX models in Agda
2. Export Agda DirectedGraph to ONNX format
3. Import ONNX models and analyze them in Agda
4. Prove properties about ONNX graphs

## ONNX Overview

ONNX (Open Neural Network Exchange) is an open format for representing
machine learning models. An ONNX model is a **directed acyclic graph (DAG)**
where:
- **Nodes** represent operations (Conv, Add, ReLU, etc.)
- **Edges** are implicit, represented by tensor names flowing between nodes
- Each tensor name is the output of exactly one node (SSA property)
- Multi-input nodes (Add, Concat) correspond to convergence points

## Mapping to Paper's Oriented Graphs

```
Paper's Γ (Section 1.1)      ONNX Graph
-----------------------      ----------
Vertices (layers Lₖ)   →     Subgraphs (sets of nodes)
Edges (connections)    →     Tensor flow (implicit via names)
Acyclic property       →     DAG constraint (no cycles)

Fork Construction (Section 1.3):
fork-star a'           →     Multi-input node (Add/Concat)
fork-tang A            →     Output tensor of multi-input node
handle a               →     Next node consuming that tensor
```

## Reference

ONNX specification: https://onnx.ai/onnx/repo-docs/IR.html
Official protobuf: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
-}

module Neural.Compile.ONNX where

open import 1Lab.Prelude
open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin)
open import Data.Bool.Base using (Bool; true; false)
open import Data.List.Base
open Data.List.Base using (_++_; concat)
open import Data.String.Base using (String)
open import Data.Float.Base using (Float)

--------------------------------------------------------------------------------
-- Tensor Element Types
--------------------------------------------------------------------------------

{-|
ONNX tensor element types, corresponding to TensorProto.DataType enum.
These are the primitive types that can be stored in tensors.
-}
data TensorElementType : Type where
  UNDEFINED  : TensorElementType  -- 0
  FLOAT      : TensorElementType  -- 1 - 32-bit float
  UINT8      : TensorElementType  -- 2
  INT8       : TensorElementType  -- 3
  UINT16     : TensorElementType  -- 4
  INT16      : TensorElementType  -- 5
  INT32      : TensorElementType  -- 6
  INT64      : TensorElementType  -- 7
  STRING     : TensorElementType  -- 8
  BOOL       : TensorElementType  -- 9
  FLOAT16    : TensorElementType  -- 10 - IEEE 754 half-precision
  DOUBLE     : TensorElementType  -- 11 - 64-bit float
  UINT32     : TensorElementType  -- 12
  UINT64     : TensorElementType  -- 13
  COMPLEX64  : TensorElementType  -- 14
  COMPLEX128 : TensorElementType  -- 15
  BFLOAT16   : TensorElementType  -- 16 - Brain float16

--------------------------------------------------------------------------------
-- Tensor Shapes
--------------------------------------------------------------------------------

{-|
A dimension can be either a fixed size or a symbolic parameter.
Example: [1, 224, 224, 3] for an image, or ["batch", 224, 224, 3] for batched.
-}
data Dimension : Type where
  dim-value : Nat → Dimension           -- Fixed size dimension
  dim-param : String → Dimension        -- Symbolic dimension (e.g., "batch_size")

{-|
TensorShape represents the shape of a tensor as a list of dimensions.
Example: [1, 3, 224, 224] for a single RGB image
-}
TensorShape : Type
TensorShape = List Dimension

--------------------------------------------------------------------------------
-- Type Information
--------------------------------------------------------------------------------

{-|
TypeProto defines the type of a value (input/output of a node).
Currently simplified to tensor types only.
Full ONNX also supports: Map, Sequence, Optional, SparseTensor
-}
record TensorTypeProto : Type where
  field
    elem-type : TensorElementType
    shape     : TensorShape

{-|
TypeProto is the top-level type descriptor.
For now, we only implement tensor types.
-}
data TypeProto : Type where
  tensor-type : TensorTypeProto → TypeProto

--------------------------------------------------------------------------------
-- Value Information
--------------------------------------------------------------------------------

{-|
ValueInfoProto describes an input or output of the graph.
Contains: name, type, and optional documentation.

Example:
  name = "input_image"
  type = tensor<float, [1, 3, 224, 224]>
  doc  = "Input RGB image"
-}
record ValueInfoProto : Type where
  field
    name : String
    type : TypeProto
    doc  : String  -- Documentation string (optional, use "" if none)

--------------------------------------------------------------------------------
-- Attributes
--------------------------------------------------------------------------------

{-|
AttributeValue represents the value of an operator attribute.
Attributes configure operators (e.g., kernel_size=3 for Conv).

Examples:
  - Conv operator: kernel_shape = [3, 3], strides = [1, 1]
  - Concat operator: axis = 1
-}
data AttributeValue : Type where
  attr-float   : Float → AttributeValue
  attr-int     : Nat → AttributeValue
  attr-string  : String → AttributeValue
  attr-floats  : List Float → AttributeValue
  attr-ints    : List Nat → AttributeValue
  attr-strings : List String → AttributeValue
  -- Note: attr-tensor and attr-graph exist in full ONNX but omitted for simplicity

{-|
AttributeProto is a named attribute.
Example: { name = "kernel_shape", value = attr-ints [3, 3] }
-}
record AttributeProto : Type where
  field
    name  : String
    value : AttributeValue

--------------------------------------------------------------------------------
-- Tensor Data (Initializers/Weights)
--------------------------------------------------------------------------------

{-|
TensorProto represents actual tensor data.
Used for initializers (weights, biases) in the graph.

Simplified version - in real ONNX, data can be stored in multiple formats:
- raw_data (bytes)
- float_data, int32_data, etc. (typed arrays)
- external_data (file reference)

We use a simple representation for now.
-}
record TensorProto : Type where
  field
    name      : String
    elem-type : TensorElementType
    dims      : List Nat  -- Shape as concrete dimensions
    -- data would go here in a real implementation
    -- For now, we treat tensors as abstract

--------------------------------------------------------------------------------
-- Nodes (Operations)
--------------------------------------------------------------------------------

{-|
NodeProto represents a single operation in the computation graph.

Fields:
- op-type: Operation name (e.g., "Conv", "Add", "Concat", "ReLU")
- inputs:  List of tensor names consumed by this node
- outputs: List of tensor names produced by this node (usually 1)
- attributes: Operator-specific parameters

Example (Conv node):
  op-type    = "Conv"
  inputs     = ["input_image", "conv1_weight", "conv1_bias"]
  outputs    = ["conv1_output"]
  attributes = [kernel_shape=[3,3], strides=[1,1], pads=[1,1,1,1]]

Example (Add node - multi-input!):
  op-type    = "Add"
  inputs     = ["branch1_output", "branch2_output"]
  outputs    = ["merged_output"]
  attributes = []

Key insight: Multi-input nodes like Add, Concat correspond to the
fork-star vertices in the paper's fork construction!
-}
record NodeProto : Type where
  field
    op-type    : String                  -- Operation type (e.g., "Conv", "Add")
    inputs     : List String             -- Input tensor names
    outputs    : List String             -- Output tensor names
    attributes : List AttributeProto     -- Operator attributes
    name       : String                  -- Optional node name (use "" if none)
    domain     : String                  -- Operator domain (use "" for default)

--------------------------------------------------------------------------------
-- Graphs (Computation Graphs)
--------------------------------------------------------------------------------

{-|
GraphProto represents the entire computation graph.

This is the core ONNX structure - a DAG of nodes where:
1. nodes: List of operations, topologically sorted
2. inputs: Graph-level inputs (ValueInfoProto)
3. outputs: Graph-level outputs (ValueInfoProto)
4. initializers: Constant tensors (weights, biases)

Edges are implicit: An edge from node A to node B exists if
some output tensor name of A appears in the inputs of B.

Example structure:
  inputs       = ["input_image"]
  outputs      = ["classification_logits"]
  initializers = [conv1_weight, conv1_bias, ...]
  nodes        = [Conv1, ReLU1, Conv2, ReLU2, ..., Dense, Softmax]

Constraints:
- Must be acyclic (no cycles)
- Each output tensor name is produced by exactly one node (SSA property)
- Topologically sorted (dependencies come before dependents)
-}
record GraphProto : Type where
  field
    nodes        : List NodeProto         -- Computation nodes
    name         : String                 -- Graph name
    inputs       : List ValueInfoProto    -- Graph inputs
    outputs      : List ValueInfoProto    -- Graph outputs
    initializers : List TensorProto       -- Constant tensors (weights)
    doc          : String                 -- Documentation string

--------------------------------------------------------------------------------
-- Operator Sets
--------------------------------------------------------------------------------

{-|
OperatorSetIdProto specifies which version of which operator set is used.
ONNX operators are versioned - different versions may have different semantics.

Example: { domain = "", version = 17 }
  means "use ONNX standard operators version 17"
-}
record OperatorSetIdProto : Type where
  field
    domain  : String  -- Operator domain (empty string = ONNX standard ops)
    version : Nat     -- Version number

--------------------------------------------------------------------------------
-- Models (Top Level)
--------------------------------------------------------------------------------

{-|
ModelProto is the top-level container for an ONNX model.

Contains:
- graph: The computation graph
- opset-import: Which operator sets are used
- ir-version: ONNX IR version
- producer-name/version: Tool that generated the model
- domain: Model namespace

This is what gets serialized to a .onnx file.
-}
record ModelProto : Type where
  field
    ir-version     : Nat                        -- ONNX IR version (currently 9)
    opset-import   : List OperatorSetIdProto    -- Operator set versions
    producer-name  : String                     -- e.g., "pytorch", "tensorflow"
    producer-version : String                   -- e.g., "2.0.0"
    domain         : String                     -- Model namespace
    model-version  : Nat                        -- Model version number
    doc            : String                     -- Model documentation
    graph          : GraphProto                 -- The actual computation graph

--------------------------------------------------------------------------------
-- Utility Functions
--------------------------------------------------------------------------------

{-|
Create a simple tensor type (float32 with given shape).
-}
mk-float-tensor-type : List Nat → TypeProto
mk-float-tensor-type shape =
  tensor-type record
    { elem-type = FLOAT
    ; shape = map dim-value shape
    }

{-|
Create a value info with float32 type.
-}
mk-float-value-info : String → List Nat → ValueInfoProto
mk-float-value-info name shape = record
  { name = name
  ; type = mk-float-tensor-type shape
  ; doc = ""
  }

{-|
Check if a node has multiple inputs (i.e., is a convergence point).
These correspond to fork-star vertices in the paper's construction.
-}
is-multi-input-node : NodeProto → Bool
is-multi-input-node node with NodeProto.inputs node
... | []           = false
... | x ∷ []       = false
... | x ∷ y ∷ rest = true

{-|
Get all tensor names used in a graph (inputs + all node outputs).
-}
get-all-tensor-names : GraphProto → List String
get-all-tensor-names graph =
  let input-names = map ValueInfoProto.name (GraphProto.inputs graph)
      node-output-names = concat (map NodeProto.outputs (GraphProto.nodes graph))
  in input-names ++ node-output-names

{-|
Common ONNX operator types as constants.
-}
module OpTypes where
  Conv       = "Conv"
  Add        = "Add"
  Concat     = "Concat"
  MatMul     = "MatMul"
  Gemm       = "Gemm"
  Relu       = "Relu"
  Sigmoid    = "Sigmoid"
  Tanh       = "Tanh"
  Softmax    = "Softmax"
  BatchNorm  = "BatchNormalization"
  MaxPool    = "MaxPool"
  AvgPool    = "AveragePool"
  Reshape    = "Reshape"
  Flatten    = "Flatten"
  Transpose  = "Transpose"

--------------------------------------------------------------------------------
-- Documentation
--------------------------------------------------------------------------------

{-|
## Next Steps

To use this module:

1. **Export DirectedGraph → ONNX**:
   - Map vertices to subgraphs of nodes
   - Map edges to tensor name connections
   - Handle fork construction (multi-input nodes)

2. **Import ONNX → DirectedGraph**:
   - Identify multi-input nodes (convergence points)
   - Generate fork construction
   - Prove acyclicity is preserved

3. **Prove Properties**:
   - ONNX graphs are DAGs (no cycles)
   - SSA property (each tensor name output exactly once)
   - Type soundness (inputs/outputs match)
   - Roundtrip correctness

See Neural.Compile.ONNX.Export and Neural.Compile.ONNX.Import for implementations.
-}
