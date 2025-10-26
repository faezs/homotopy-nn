module Neural.Graph.TestModule where

open import Neural.Graph.Base
open import Neural.Graph.Oriented
open import 1Lab.Prelude

private variable
  o ℓ : Level

module Test (G : Graph o ℓ) where
  open Graph G

  test : Node → Node → Type ℓ
  test x y = Edge x y
