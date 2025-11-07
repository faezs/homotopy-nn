{-# OPTIONS --cubical --allow-unsolved-metas #-}

{-|
# MLP to JAX Compilation with MUP Scaling

Compiles SimpleMLP-MUP chain networks to JAX/Flax modules.

## Key Insight
SimpleMLP is a **chain**: x₀ → h₁ → h₂ → y (no convergence, no forks!)
MUP scaling ensures O(1) feature learning across widths.

## Compilation Pipeline
1. Extract chain structure from SimpleMLP
2. Apply MUP scaling (hidden: std/√w, output: std/w)
3. Generate JAX NetworkSpec with layer configs
4. Serialize to JSON for Python runtime

## References
- Yang et al. "Tensor Programs V" (2022) - MUP scaling
- Belfiore & Bennequin (2022) - Chain as simplest feedforward network
-}

module Neural.Compile.MLPToJAX where

open import 1Lab.Prelude
open import 1Lab.Path
open import 1Lab.Type

-- Basic types
open import Data.Nat using (Nat; _+_; _*_; zero; suc)
open import Data.Fin using (Fin; fzero; fsuc)
open import Agda.Builtin.String using (String)

-- Import SimpleMLP and MUP
open import Neural.Topos.Examples using (SimpleMLP)
open import Neural.Stack.MUP using (MUPBase; MUPConfig; mk-mup-config; MLPArch)
open import Neural.Smooth.Base using (ℝ; _/ₙ_)

-- List type for JSON arrays
data List (A : Type) : Type where
  [] : List A
  _∷_ : A → List A → List A

infixr 5 _∷_

-- JSON type
data JSON : Type where
  json-null : JSON
  json-nat : Nat → JSON
  json-string : String → JSON
  json-real : ℝ → JSON
  json-array : List JSON → JSON
  json-object : List (String × JSON) → JSON

--------------------------------------------------------------------------------
-- § 1: Layer Specification for JAX
--------------------------------------------------------------------------------

-- Layer types in MLP chain
data LayerSpec : Type where
  input-layer  : (dim : Nat) → LayerSpec
  hidden-layer : (width : Nat) (init-std : ℝ) (lr : ℝ) → LayerSpec
  output-layer : (dim : Nat) (init-std : ℝ) (lr : ℝ) → LayerSpec

-- Connection between layers (edge in chain)
record Connection : Type where
  field
    from-idx : Nat  -- Source layer index
    to-idx   : Nat  -- Target layer index

-- Complete network specification for JAX
record JAXNetworkSpec : Type where
  field
    layers : List LayerSpec
    connections : List Connection

    -- Global hyperparameters
    batch-size : Nat
    num-epochs : Nat
    optimizer : String

{-# COMPILE GHC LayerSpec = data LayerSpec (InputLayer | HiddenLayer | OutputLayer) #-}
{-# COMPILE GHC Connection = data Connection (Connection) #-}
{-# COMPILE GHC JAXNetworkSpec = data JAXNetworkSpec (JAXNetworkSpec) #-}

--------------------------------------------------------------------------------
-- § 2: SimpleMLP to JAX Compilation
--------------------------------------------------------------------------------

-- Extract SimpleMLP chain structure with MUP scaling
compile-simple-mlp : (w : Nat) → MUPBase → JAXNetworkSpec
compile-simple-mlp w base =
  let config = mk-mup-config base w
  in record
    { layers =
        input-layer (28 * 28) ∷  -- MNIST input: 784
        hidden-layer w (MUPConfig.hidden-std config) (MUPConfig.hidden-lr config) ∷
        hidden-layer w (MUPConfig.hidden-std config) (MUPConfig.hidden-lr config) ∷
        output-layer 10 (MUPConfig.output-std config) (MUPConfig.output-lr config) ∷
        []
    ; connections =
        record { from-idx = 0 ; to-idx = 1 } ∷  -- input → h₁
        record { from-idx = 1 ; to-idx = 2 } ∷  -- h₁ → h₂
        record { from-idx = 2 ; to-idx = 3 } ∷  -- h₂ → output
        []
    ; batch-size = 32
    ; num-epochs = 100
    ; optimizer = "adam"
    }

-- Compile multiple widths for MUP transfer test
compile-mlp-multi-width : MUPBase → List Nat → List JAXNetworkSpec
compile-mlp-multi-width base [] = []
compile-mlp-multi-width base (w ∷ ws) =
  compile-simple-mlp w base ∷ compile-mlp-multi-width base ws

--------------------------------------------------------------------------------
-- § 3: JSON Serialization
--------------------------------------------------------------------------------

-- Convert LayerSpec to JSON
layer-to-json : LayerSpec → JSON
layer-to-json (input-layer dim) = json-object (
  ("type", json-string "input") ∷
  ("dim", json-nat dim) ∷
  [])
layer-to-json (hidden-layer width init-std lr) = json-object (
  ("type", json-string "hidden") ∷
  ("width", json-nat width) ∷
  ("init_std", json-real init-std) ∷
  ("lr", json-real lr) ∷
  [])
layer-to-json (output-layer dim init-std lr) = json-object (
  ("type", json-string "output") ∷
  ("dim", json-nat dim) ∷
  ("init_std", json-real init-std) ∷
  ("lr", json-real lr) ∷
  [])

-- Convert Connection to JSON
connection-to-json : Connection → JSON
connection-to-json conn = json-object (
  ("from", json-nat (Connection.from-idx conn)) ∷
  ("to", json-nat (Connection.to-idx conn)) ∷
  [])

-- Convert list to JSON array
postulate
  list-map : ∀ {A B : Type} → (A → B) → List A → List B

-- Convert JAXNetworkSpec to JSON
network-to-json : JAXNetworkSpec → JSON
network-to-json spec = json-object (
  ("layers", json-array (list-map layer-to-json (JAXNetworkSpec.layers spec))) ∷
  ("connections", json-array (list-map connection-to-json (JAXNetworkSpec.connections spec))) ∷
  ("batch_size", json-nat (JAXNetworkSpec.batch-size spec)) ∷
  ("num_epochs", json-nat (JAXNetworkSpec.num-epochs spec)) ∷
  ("optimizer", json-string (JAXNetworkSpec.optimizer spec)) ∷
  [])

--------------------------------------------------------------------------------
-- § 4: FFI Exports
--------------------------------------------------------------------------------

-- Export for Haskell FFI
postulate
  json-to-string : JSON → String

{-# FOREIGN GHC import MLP.Runtime #-}
{-# COMPILE GHC compile-simple-mlp = mlpCompileFFI #-}
{-# COMPILE GHC network-to-json = networkToJsonFFI #-}

--------------------------------------------------------------------------------
-- § 5: Example: MUP Transfer Test Configuration
--------------------------------------------------------------------------------

-- Standard MUP base (0.1 lr, 0.02 std)
example-mup-base : MUPBase
example-mup-base = record
  { base-lr = 1 /ₙ 10
  ; base-std = 1 /ₙ 50
  ; hidden-depth = 2
  }

-- Compile networks at different widths for transfer test
example-widths : List Nat
example-widths = 64 ∷ 256 ∷ 1024 ∷ []

example-networks : List JAXNetworkSpec
example-networks = compile-mlp-multi-width example-mup-base example-widths

-- Generate JSON for Python runtime
example-json : List JSON
example-json = list-map network-to-json example-networks

{-|
## Output Format

For each width, generates JSON like:
```json
{
  "layers": [
    {"type": "input", "dim": 784},
    {"type": "hidden", "width": 64, "init_std": 0.0025, "lr": 0.1},
    {"type": "hidden", "width": 64, "init_std": 0.0025, "lr": 0.1},
    {"type": "output", "dim": 10, "init_std": 0.00031, "lr": 0.0016}
  ],
  "connections": [
    {"from": 0, "to": 1},
    {"from": 1, "to": 2},
    {"from": 2, "to": 3}
  ],
  "batch_size": 32,
  "num_epochs": 100,
  "optimizer": "adam"
}
```

Python runtime will:
1. Parse JSON
2. Build Flax MLP with MUP initialization
3. Train and verify feature norms stay O(1)
-}
