{-# OPTIONS --cubical --no-import-sorts #-}
module Neural.Code where

open import Neural.Base

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path
open import 1Lab.Type

open import Cat.Base
open import Cat.Functor.Base

open import Data.Bool.Base
open import Data.Fin.Base
open import Data.List.Base
open import Data.Power

-- | Generate neural code from neural network responses
-- Collects all possible activation patterns across stimuli
generate-neural-code : {X : StimulusSpace} (N : NeuralNetwork X)
                     → (stimuli : List X)
                     → NeuralCode (vertices (NeuralNetwork.graph N))
generate-neural-code N stimuli = map (neural-response N) stimuli

-- | Code words: individual binary response patterns
CodeWord : (n : Nat) → Type
CodeWord n = Fin n → Bool

-- | Active neurons for a given code word
active-neurons : {n : Nat} → CodeWord n → List (Fin n)
active-neurons {n} cw = filter (cw) (fin-list n)
  where
    fin-list : (k : Nat) → List (Fin k)
    fin-list zero = []
    fin-list (suc k) = fzero ∷ map fsuc (fin-list k)

-- | Support of a neural code: all neurons that appear in some code word
code-support : {n : Nat} → NeuralCode n → List (Fin n)
code-support code = concat (map active-neurons code)

-- | Overlap pattern: neurons that fire together
-- This captures the combinatorial structure needed for homotopy reconstruction
record OverlapPattern (n : Nat) : Type where
  field
    neurons : List (Fin n)
    occurs-in-code : CodeWord n → Bool