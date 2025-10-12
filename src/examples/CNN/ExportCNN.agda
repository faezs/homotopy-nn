{-# OPTIONS --guardedness #-}

module examples.CNN.ExportCNN where

open import Agda.Builtin.IO
open import Agda.Builtin.Unit

{-# FOREIGN GHC
import qualified Data.Text.IO as T
import qualified Data.Text as T

-- Hardcoded JSON for now - shows the structure that would be generated
simpleCnnJson :: T.Text
simpleCnnJson = T.unlines [
  T.pack "{",
  T.pack "  \"ir-version\": 9,",
  T.pack "  \"opset-import\": [{\"domain\": \"\", \"version\": 17}],",
  T.pack "  \"producer-name\": \"homotopy-nn\",",
  T.pack "  \"producer-version\": \"1.0.0\",",
  T.pack "  \"domain\": \"neural.homotopy\",",
  T.pack "  \"model-version\": 1,",
  T.pack "  \"doc\": \"LeNet-style CNN with Z^2 translation equivariance: 28x28x1 -> Conv(5x5,20) -> MaxPool -> Conv(5x5,50) -> MaxPool -> Dense(500) -> Dense(10)\",",
  T.pack "  \"graph\": {",
  T.pack "    \"name\": \"simple-cnn\",",
  T.pack "    \"nodes\": [],",
  T.pack "    \"inputs\": [],",
  T.pack "    \"outputs\": [],",
  T.pack "    \"initializers\": []",
  T.pack "  }",
  T.pack "}"
  ]
#-}

postulate
  main : IO ⊤

{-# COMPILE GHC main = T.putStrLn simpleCnnJson #-}
