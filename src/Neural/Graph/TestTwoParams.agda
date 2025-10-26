module Neural.Graph.TestTwoParams where

open import Neural.Graph.Base
open import Neural.Graph.Oriented
open import Neural.Graph.Path

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.HLevel.Closure

open import Cat.Instances.Graphs.Omega
open import Cat.Functor.Subcategory

open import Data.List
open import Data.Nat

private variable
  o ℓ o' ℓ' : Level

module Test (G : Graph o ℓ) (oriented : is-oriented G) where
  open Graph G

  test : Node → Node → Type ℓ
  test x y = Edge x y
