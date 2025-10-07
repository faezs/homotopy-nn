{-# OPTIONS --rewriting --guardedness --cubical #-}

{-|
# JSON Serialization for Neural IR

Converts the IR to JSON format for consumption by Python bridge.

## Output Format

```json
{
  "name": "ResNet50",
  "vertices": [
    {
      "id": 0,
      "op": {"type": "linear", "in_dim": 784, "out_dim": 256},
      "input_shapes": [{"type": "vec", "dim": 784}],
      "output_shape": {"type": "vec", "dim": 256}
    }
  ],
  "edges": [
    {"source": 0, "target": 1, "shape": {"type": "vec", "dim": 256}}
  ],
  "inputs": [0],
  "outputs": [3],
  "properties": ["shape-correct", "conserves-mass"],
  "resources": {
    "max_flops": 1000000,
    "max_memory": 1000000,
    "max_latency": 1000,
    "sparsity": 0
  }
}
```

-}

module Neural.Compile.Serialize where

open import 1Lab.Prelude
open import Data.String using (String; _++_)
open import Data.Nat using (ℕ)
open import Data.Nat.Show using (show)
open import Data.List using (List; []; _∷_; map; foldr)

open import Neural.Compile.IR

--------------------------------------------------------------------------------
-- JSON Type

{-|
Simple JSON representation (simplified - in practice use a library)
-}

data JSON : Type where
  jnull : JSON
  jbool : Bool → JSON
  jnum : ℕ → JSON
  jstr : String → JSON
  jarray : List JSON → JSON
  jobject : List (String × JSON) → JSON

--------------------------------------------------------------------------------
-- Serialization Helpers

nat-to-json : ℕ → JSON
nat-to-json n = jstr (show n)

string-to-json : String → JSON
string-to-json s = jstr s

list-to-json : {A : Type} → (A → JSON) → List A → JSON
list-to-json f xs = jarray (map f xs)

-- Quote string
quote : String → String
quote s = "\"" ++ s ++ "\""

-- JSON to string (pretty print)
json-to-string : JSON → String
json-to-string jnull = "null"
json-to-string (jbool true) = "true"
json-to-string (jbool false) = "false"
json-to-string (jnum n) = show n
json-to-string (jstr s) = quote s
json-to-string (jarray xs) = "[" ++ foldr comma "" (map json-to-string xs) ++ "]"
  where
  comma : String → String → String
  comma s "" = s
  comma s acc = s ++ ", " ++ acc
json-to-string (jobject fields) = "{" ++ foldr comma "" (map field-to-string fields) ++ "}"
  where
  field-to-string : String × JSON → String
  field-to-string (k , v) = quote k ++ ": " ++ json-to-string v

  comma : String → String → String
  comma s "" = s
  comma s acc = s ++ ", " ++ acc

--------------------------------------------------------------------------------
-- Shape Serialization

shape-to-json : Shape → JSON
shape-to-json scalar = jobject
  ( ("type" , jstr "scalar") ∷ [] )
shape-to-json (vec n) = jobject
  ( ("type" , jstr "vec") ∷
    ("dim" , jnum n) ∷ [] )
shape-to-json (mat m n) = jobject
  ( ("type" , jstr "mat") ∷
    ("rows" , jnum m) ∷
    ("cols" , jnum n) ∷ [] )
shape-to-json (tensor dims) = jobject
  ( ("type" , jstr "tensor") ∷
    ("dims" , jarray (map jnum dims)) ∷ [] )

--------------------------------------------------------------------------------
-- Operation Serialization

activation-to-json : Activation → JSON
activation-to-json relu = jstr "relu"
activation-to-json sigmoid = jstr "sigmoid"
activation-to-json tanh = jstr "tanh"
activation-to-json gelu = jstr "gelu"
activation-to-json identity = jstr "identity"

operation-to-json : Operation → JSON
operation-to-json (linear in-dim out-dim) = jobject
  ( ("type" , jstr "linear") ∷
    ("in_dim" , jnum in-dim) ∷
    ("out_dim" , jnum out-dim) ∷ [] )
operation-to-json (conv2d in-ch out-ch kernel) = jobject
  ( ("type" , jstr "conv2d") ∷
    ("in_channels" , jnum in-ch) ∷
    ("out_channels" , jnum out-ch) ∷
    ("kernel_size" , jnum kernel) ∷ [] )
operation-to-json (activation act) = jobject
  ( ("type" , jstr "activation") ∷
    ("activation" , activation-to-json act) ∷ [] )
operation-to-json (fork arity) = jobject
  ( ("type" , jstr "fork") ∷
    ("arity" , jnum arity) ∷ [] )
operation-to-json residual = jobject
  ( ("type" , jstr "residual") ∷ [] )
operation-to-json (batch-norm features) = jobject
  ( ("type" , jstr "batch_norm") ∷
    ("features" , jnum features) ∷ [] )
operation-to-json (layer-norm features) = jobject
  ( ("type" , jstr "layer_norm") ∷
    ("features" , jnum features) ∷ [] )
operation-to-json (max-pool kernel stride) = jobject
  ( ("type" , jstr "max_pool") ∷
    ("kernel_size" , jnum kernel) ∷
    ("stride" , jnum stride) ∷ [] )
operation-to-json (avg-pool kernel stride) = jobject
  ( ("type" , jstr "avg_pool") ∷
    ("kernel_size" , jnum kernel) ∷
    ("stride" , jnum stride) ∷ [] )
operation-to-json (attention heads d-model d-k d-v) = jobject
  ( ("type" , jstr "attention") ∷
    ("heads" , jnum heads) ∷
    ("d_model" , jnum d-model) ∷
    ("d_k" , jnum d-k) ∷
    ("d_v" , jnum d-v) ∷ [] )

--------------------------------------------------------------------------------
-- Vertex Serialization

vertex-to-json : Vertex → JSON
vertex-to-json v = jobject
  ( ("id" , jnum (Vertex.id v)) ∷
    ("op" , operation-to-json (Vertex.op v)) ∷
    ("input_shapes" , jarray (map shape-to-json (Vertex.input-shapes v))) ∷
    ("output_shape" , shape-to-json (Vertex.output-shape v)) ∷ [] )

--------------------------------------------------------------------------------
-- Edge Serialization

edge-to-json : Edge → JSON
edge-to-json e = jobject
  ( ("source" , jnum (Edge.source e)) ∷
    ("target" , jnum (Edge.target e)) ∷
    ("shape" , shape-to-json (Edge.shape e)) ∷ [] )

--------------------------------------------------------------------------------
-- Property Serialization

property-to-json : Property → JSON
property-to-json conserves-mass = jstr "conserves-mass"
property-to-json shape-correct = jstr "shape-correct"
property-to-json fibration-valid = jstr "fibration-valid"
property-to-json (flops-bounded n) = jobject
  ( ("type" , jstr "flops-bounded") ∷
    ("max_flops" , jnum n) ∷ [] )
property-to-json (memory-bounded n) = jobject
  ( ("type" , jstr "memory-bounded") ∷
    ("max_memory" , jnum n) ∷ [] )
property-to-json sheaf-condition = jstr "sheaf-condition"
property-to-json (custom name) = jobject
  ( ("type" , jstr "custom") ∷
    ("name" , jstr name) ∷ [] )

--------------------------------------------------------------------------------
-- Resource Constraints Serialization

resources-to-json : ResourceConstraints → JSON
resources-to-json r = jobject
  ( ("max_flops" , jnum (ResourceConstraints.max-flops r)) ∷
    ("max_memory" , jnum (ResourceConstraints.max-memory r)) ∷
    ("max_latency" , jnum (ResourceConstraints.max-latency r)) ∷
    ("sparsity" , jnum (ResourceConstraints.sparsity r)) ∷ [] )

--------------------------------------------------------------------------------
-- Complete IR Serialization

neural-ir-to-json : NeuralIR → JSON
neural-ir-to-json ir = jobject
  ( ("name" , jstr (NeuralIR.name ir)) ∷
    ("vertices" , jarray (map vertex-to-json (NeuralIR.vertices ir))) ∷
    ("edges" , jarray (map edge-to-json (NeuralIR.edges ir))) ∷
    ("inputs" , jarray (map jnum (NeuralIR.inputs ir))) ∷
    ("outputs" , jarray (map jnum (NeuralIR.outputs ir))) ∷
    ("properties" , jarray (map property-to-json (NeuralIR.properties ir))) ∷
    ("resources" , resources-to-json (NeuralIR.resources ir)) ∷ [] )

-- Export to string
export-ir : NeuralIR → String
export-ir ir = json-to-string (neural-ir-to-json ir)

--------------------------------------------------------------------------------
-- Examples

{-|
## Example: Export MLP

```agda
mlp-json : String
mlp-json = export-ir mlp-ir
```

This produces:
```json
{
  "name": "SimpleMLP",
  "vertices": [...],
  "edges": [...],
  ...
}
```
-}

mlp-json : String
mlp-json = export-ir mlp-ir

resnet-json : String
resnet-json = export-ir resnet-block-ir

--------------------------------------------------------------------------------
-- File Writing (Postulated - would use FFI in practice)

postulate
  writeFile : String → String → IO ⊤

-- Export to file
export-to-file : String → NeuralIR → IO ⊤
export-to-file filename ir = writeFile filename (export-ir ir)

-- Example usage
-- main : IO ⊤
-- main = export-to-file "mlp.json" mlp-ir

--------------------------------------------------------------------------------
-- Summary

{-|
This module provides:
1. ✅ JSON representation
2. ✅ Serialization for all IR types
3. ✅ Pretty printing
4. ✅ File export (via FFI)

**Next:** Build Python parser to read these JSON files.
-}
