{-# OPTIONS --guardedness #-}

module examples.CNN.ExportJSON where

open import Agda.Builtin.IO using (IO)
open import Agda.Builtin.Unit using (⊤)
open import Agda.Builtin.String using (String)
open import Data.String.Base using (_++_)

-- Import our CNN model
open import examples.CNN.SimpleCNN using (simple-cnn-json)

{-# FOREIGN GHC import qualified Data.Text.IO as Text #-}
{-# FOREIGN GHC import System.IO #-}

postulate
  putStrLn : String → IO ⊤
  writeFile : String → String → IO ⊤

{-# COMPILE GHC putStrLn = Text.putStrLn #-}
{-# COMPILE GHC writeFile = \path content -> Text.writeFile path content #-}

main : IO ⊤
main = writeFile "examples/simple-cnn.json" simple-cnn-json
