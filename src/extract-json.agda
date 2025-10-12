{-# OPTIONS --no-import-sorts #-}

{-|
Extract simple-cnn-json-dsl value by compiling with GHC.

This will compile to Haskell and write the JSON to stdout.
-}

module extract-json where

open import Data.String.Base using (String)
open import examples.CNN.SimpleCNN using (simple-cnn-json-dsl)

-- For GHC compilation
{-# FOREIGN GHC import qualified Data.Text.IO as TIO #-}
{-# FOREIGN GHC import System.IO (stdout, hSetEncoding, utf8) #-}

postulate
  writeString : String → IO String

{-# COMPILE GHC writeString = \s -> do
      hSetEncoding stdout utf8
      TIO.putStrLn s
      return s
    #-}

-- Main function that writes the JSON
main : IO String
main = writeString simple-cnn-json-dsl
