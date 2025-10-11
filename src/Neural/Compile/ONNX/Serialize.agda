{-# OPTIONS --no-import-sorts #-}

{-|
# JSON Serialization for ONNX

Serialize Agda ONNX types to JSON strings for interop with Python bridge.
-}

module Neural.Compile.ONNX.Serialize where

open import 1Lab.Prelude
open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin; lower)
open import Data.List.Base
open import Data.String.Base using (String; _<>_)
open import Data.Float.Base using (Float)

open import Neural.Compile.ONNX

-- Postulate string operations for JSON building
postulate
  show-nat : Nat → String
  show-float : Float → String
  quote-string : String → String  -- Add quotes around string

--------------------------------------------------------------------------------
-- JSON Utilities
--------------------------------------------------------------------------------

-- Join list of strings with separator
join : String → List String → String
join sep [] = ""
join sep (x ∷ []) = x
join sep (x ∷ xs) = x <> sep <> join sep xs

-- Wrap in brackets
brackets : String → String
brackets s = "[" <> s <> "]"

-- Wrap in braces
braces : String → String
braces s = "{" <> s <> "}"

-- JSON key-value pair
field : String → String → String
field key value = quote-string key <> ": " <> value

--------------------------------------------------------------------------------
-- Serialize ONNX Types
--------------------------------------------------------------------------------

serialize-elem-type : TensorElementType → String
serialize-elem-type UNDEFINED  = quote-string "UNDEFINED"
serialize-elem-type FLOAT      = quote-string "FLOAT"
serialize-elem-type UINT8      = quote-string "UINT8"
serialize-elem-type INT8       = quote-string "INT8"
serialize-elem-type UINT16     = quote-string "UINT16"
serialize-elem-type INT16      = quote-string "INT16"
serialize-elem-type INT32      = quote-string "INT32"
serialize-elem-type INT64      = quote-string "INT64"
serialize-elem-type STRING     = quote-string "STRING"
serialize-elem-type BOOL       = quote-string "BOOL"
serialize-elem-type FLOAT16    = quote-string "FLOAT16"
serialize-elem-type DOUBLE     = quote-string "DOUBLE"
serialize-elem-type UINT32     = quote-string "UINT32"
serialize-elem-type UINT64     = quote-string "UINT64"
serialize-elem-type COMPLEX64  = quote-string "COMPLEX64"
serialize-elem-type COMPLEX128 = quote-string "COMPLEX128"
serialize-elem-type BFLOAT16   = quote-string "BFLOAT16"

serialize-dimension : Dimension → String
serialize-dimension (dim-value n) =
  braces (field "dim-value" (show-nat n))
serialize-dimension (dim-param s) =
  braces (field "dim-param" (quote-string s))

serialize-shape : TensorShape → String
serialize-shape dims = brackets (join ", " (map serialize-dimension dims))

serialize-tensor-type : TensorTypeProto → String
serialize-tensor-type tt = braces (join ", "
  ( field "elem-type" (serialize-elem-type (TensorTypeProto.elem-type tt))
  ∷ field "shape" (serialize-shape (TensorTypeProto.shape tt))
  ∷ [] ))

serialize-type-proto : TypeProto → String
serialize-type-proto (tensor-type tt) =
  braces (field "tensor-type" (serialize-tensor-type tt))

serialize-value-info : ValueInfoProto → String
serialize-value-info vi = braces (join ", "
  ( field "name" (quote-string (ValueInfoProto.name vi))
  ∷ field "type" (serialize-type-proto (ValueInfoProto.type vi))
  ∷ field "doc" (quote-string (ValueInfoProto.doc vi))
  ∷ [] ))

serialize-attribute-value : AttributeValue → String
serialize-attribute-value (attr-float f) =
  braces (field "attr-float" (show-float f))
serialize-attribute-value (attr-int n) =
  braces (field "attr-int" (show-nat n))
serialize-attribute-value (attr-string s) =
  braces (field "attr-string" (quote-string s))
serialize-attribute-value (attr-floats fs) =
  braces (field "attr-floats" (brackets (join ", " (map show-float fs))))
serialize-attribute-value (attr-ints ns) =
  braces (field "attr-ints" (brackets (join ", " (map show-nat ns))))
serialize-attribute-value (attr-strings ss) =
  braces (field "attr-strings" (brackets (join ", " (map quote-string ss))))

serialize-attribute : AttributeProto → String
serialize-attribute attr = braces (join ", "
  ( field "name" (quote-string (AttributeProto.name attr))
  ∷ field "value" (serialize-attribute-value (AttributeProto.value attr))
  ∷ [] ))

serialize-string-list : List String → String
serialize-string-list ss = brackets (join ", " (map quote-string ss))

serialize-node : NodeProto → String
serialize-node node = braces (join ", "
  ( field "op-type" (quote-string (NodeProto.op-type node))
  ∷ field "inputs" (serialize-string-list (NodeProto.inputs node))
  ∷ field "outputs" (serialize-string-list (NodeProto.outputs node))
  ∷ field "attributes" (brackets (join ", " (map serialize-attribute (NodeProto.attributes node))))
  ∷ field "name" (quote-string (NodeProto.name node))
  ∷ field "domain" (quote-string (NodeProto.domain node))
  ∷ [] ))

serialize-tensor : TensorProto → String
serialize-tensor tensor = braces (join ", "
  ( field "name" (quote-string (TensorProto.name tensor))
  ∷ field "elem-type" (serialize-elem-type (TensorProto.elem-type tensor))
  ∷ field "dims" (brackets (join ", " (map show-nat (TensorProto.dims tensor))))
  ∷ [] ))

serialize-graph : GraphProto → String
serialize-graph graph = braces (join ", "
  ( field "name" (quote-string (GraphProto.name graph))
  ∷ field "nodes" (brackets (join ", " (map serialize-node (GraphProto.nodes graph))))
  ∷ field "inputs" (brackets (join ", " (map serialize-value-info (GraphProto.inputs graph))))
  ∷ field "outputs" (brackets (join ", " (map serialize-value-info (GraphProto.outputs graph))))
  ∷ field "initializers" (brackets (join ", " (map serialize-tensor (GraphProto.initializers graph))))
  ∷ field "doc" (quote-string (GraphProto.doc graph))
  ∷ [] ))

serialize-opset : OperatorSetIdProto → String
serialize-opset opset = braces (join ", "
  ( field "domain" (quote-string (OperatorSetIdProto.domain opset))
  ∷ field "version" (show-nat (OperatorSetIdProto.version opset))
  ∷ [] ))

serialize-model : ModelProto → String
serialize-model model = braces (join ", "
  ( field "ir-version" (show-nat (ModelProto.ir-version model))
  ∷ field "opset-import" (brackets (join ", " (map serialize-opset (ModelProto.opset-import model))))
  ∷ field "producer-name" (quote-string (ModelProto.producer-name model))
  ∷ field "producer-version" (quote-string (ModelProto.producer-version model))
  ∷ field "domain" (quote-string (ModelProto.domain model))
  ∷ field "model-version" (show-nat (ModelProto.model-version model))
  ∷ field "doc" (quote-string (ModelProto.doc model))
  ∷ field "graph" (serialize-graph (ModelProto.graph model))
  ∷ [] ))

--------------------------------------------------------------------------------
-- Main Export Function
--------------------------------------------------------------------------------

{-|
Serialize ModelProto to JSON string.
This can be written to a file and consumed by onnx_bridge.py
-}
model-to-json : ModelProto → String
model-to-json = serialize-model
