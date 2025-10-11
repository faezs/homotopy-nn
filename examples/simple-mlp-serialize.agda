{-# OPTIONS --no-import-sorts #-}
{-# OPTIONS --guardedness #-}

{-|
# Generate simple-mlp.json from Agda

This module generates the JSON file by serializing the Agda ONNX model.
Run with:
  agda --compile examples/simple-mlp-serialize.agda
  ./simple-mlp-serialize > examples/simple-mlp-generated.json
-}

module examples.simple-mlp-serialize where

open import 1Lab.Prelude
open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin; fzero; fsuc)
open import Data.List.Base
open import Data.String.Base using (String)
open import IO.Base
open import IO.Primitive

open import Neural.Base using (DirectedGraph)
open import Neural.Compile.ONNX
open import Neural.Compile.ONNX.Export
open import Neural.Compile.ONNX.Serialize

--------------------------------------------------------------------------------
-- Concrete DirectedGraph for Simple MLP
--------------------------------------------------------------------------------

-- For compilation, we need a concrete graph implementation
-- (not just postulated)

open import Cat.Instances.Shape.Parallel using (·⇉·)
open import Cat.Instances.FinSets using (FinSets)
open import Cat.Functor.Base using (Functor)

-- Simple MLP graph: 3 vertices, 2 edges
-- Vertex 0 (input) --edge 0--> Vertex 1 (hidden) --edge 1--> Vertex 2 (output)

postulate
  simple-mlp-functor : Functor ·⇉· FinSets

simple-mlp-graph : DirectedGraph
simple-mlp-graph = simple-mlp-functor

-- Annotations (same as before)
simple-mlp-annotations : GraphAnnotations simple-mlp-graph
simple-mlp-annotations = record
  { vertex-op-type = λ v →
      "Input"  -- simplified, should use Fin-cases

  ; edge-shape = λ e →
      784 ∷ 256 ∷ []  -- simplified

  ; edge-elem-type = λ _ → FLOAT

  ; vertex-attributes = λ v → []

  ; graph-inputs = fzero ∷ []

  ; graph-outputs = fsuc (fsuc fzero) ∷ []

  ; model-name = "simple-mlp"

  ; model-doc = "Simple 2-layer MLP for MNIST: 784→256→10"

  ; producer = "homotopy-nn"
  }

-- Export to ONNX
simple-mlp-onnx : ModelProto
simple-mlp-onnx = export-to-onnx simple-mlp-graph simple-mlp-annotations

-- Serialize to JSON
simple-mlp-json : String
simple-mlp-json = model-to-json simple-mlp-onnx

--------------------------------------------------------------------------------
-- Main: Output JSON to stdout
--------------------------------------------------------------------------------

postulate
  putStrLn : String → IO ⊤

{-# FOREIGN GHC import qualified Data.Text.IO as T #-}
{-# COMPILE GHC putStrLn = T.putStrLn #-}

main : IO ⊤
main = putStrLn simple-mlp-json
