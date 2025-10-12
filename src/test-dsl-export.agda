{-# OPTIONS --no-import-sorts #-}

{-|
Test: Normalize simple-cnn-json-dsl to verify DSL → ONNX pipeline works.

This should produce a JSON string representing the complete ONNX model.
-}

module test-dsl-export where

open import examples.CNN.SimpleCNN using (simple-cnn-json-dsl)

-- Force normalization
test : _
test = simple-cnn-json-dsl
