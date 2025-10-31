{-# OPTIONS --cubical #-}

{-|
# Einsum Execution Oracle via PyTorch

**Trust Boundary Architecture**:
- ✅ Verified in Agda: Type safety, index tracking, optimization correctness
- 🔒 Trusted external oracle: PyTorch's `torch.einsum` implementation
- 🔌 Realized via extraction: Agda postulate → Haskell FFI → Python subprocess

## Key Design

**Postulated Execution**: We postulate `eval-pytorch` as an opaque function that
executes einsum formulas. This postulate is:
1. **Not axiomatically justified** - We trust PyTorch's correctness
2. **Realized via extraction** - Haskell code provides FFI implementation
3. **External to type system** - No computational rules needed

**Trust Model**: "Trust PyTorch, verify everything else"
- Agda verifies: Einsum construction, string conversion, optimization
- PyTorch handles: GPU execution, numerical precision, performance

## Example

```agda
-- Matrix multiply: A[i,j] × B[j,k] → C[i,k]
test : Tensor
test = evalEinsum matmul
  [ mkTensor (i ∷ j ∷ []) [ 1.0 , 2.0 , 3.0 , 4.0 ]  -- 2×2 A
  , mkTensor (j ∷ k ∷ []) [ 5.0 , 6.0 , 7.0 , 8.0 ]  -- 2×2 B
  ]
-- Result: mkTensor (i ∷ k ∷ []) [ 19.0 , 22.0 , 43.0 , 50.0 ]
```

## Extraction

In Haskell backend:
```haskell
evalPytorch :: String -> [Tensor] -> IO Tensor
evalPytorch formula tensors = do
  session <- getPythonSession
  sendJSON session $ object ["formula" .= formula, "tensors" .= tensors]
  receiveTensor session
```

-}

module Neural.Compile.Einsum.Oracle where

open import 1Lab.Prelude
open import 1Lab.Path

open import Data.List using (List; []; _∷_; length)
open import Data.String.Base using (String)
open import Neural.Compile.Einsum.Index
open import Neural.Compile.Einsum.Expression

--------------------------------------------------------------------------------
-- § 1: Numeric Types
--------------------------------------------------------------------------------

{-|
## Floating Point Numbers

Postulated for now - could be refined to IEEE 754 later.
-}

postulate
  Float : Type
  Float-is-set : is-set Float

--------------------------------------------------------------------------------
-- § 2: Tensor Type (Concrete)
--------------------------------------------------------------------------------

{-|
## Tensor: Shaped Multi-dimensional Array

Concrete representation as shape + flattened data.

**Design choice**: Store data as flat list, not nested structure
- Matches PyTorch's internal representation
- Easy to serialize/deserialize
- Size invariant: length data ≡ product of shape dimensions

**TODO**: Add size invariant as field
-}

record Tensor : Type where
  constructor mkTensor
  field
    shape : IndexCtx            -- Dimension indices [batch, seq, hidden]
    elements : List Float       -- Flattened elements in row-major order

open Tensor public

-- Size of a tensor (number of elements)
tensor-size : Tensor → Nat
tensor-size tens = length (tens .elements)

-- Rank of a tensor (number of dimensions)
tensor-rank : Tensor → Nat
tensor-rank tens = length (tens .shape)

--------------------------------------------------------------------------------
-- § 3: PyTorch Oracle (Postulated)
--------------------------------------------------------------------------------

{-|
## Execution Oracle

**POSTULATED** - No computational rules, no verification.

This is our **trust boundary**:
- Agda verifies the *formula* is type-safe
- PyTorch computes the *result*

**Implementation** (Haskell extraction):
```haskell
evalPytorch :: String -> [Tensor] -> IO Tensor
evalPytorch formula tensors = do
  -- Start persistent Python session
  session <- getOrCreatePythonSession

  -- Send: {"formula": "ij,jk->ik", "tensors": [...]}
  let request = object
        [ "formula" .= formula
        , "tensors" .= map tensorToJSON tensors
        ]
  hPutStrLn (pyStdin session) (encode request)

  -- Receive: {"shape": [2,3], "data": [1.0,2.0,...]}
  response <- hGetLine (pyStdout session)
  return $ tensorFromJSON $ decode response
```

**Type signature**:
- Input: Einsum formula as string (e.g., `"ij,jk->ik"`)
- Input: List of tensors matching formula's input contexts
- Output: Result tensor matching formula's output context

**Properties** (trusted, not proven):
1. **Type safety**: Output shape matches einsum output context
2. **Correctness**: Implements Einstein summation convention
3. **Determinism**: Same inputs → same output
4. **Termination**: Always terminates (no infinite loops)

**Non-properties** (not guaranteed):
- Numerical precision (floating point rounding)
- Performance (implementation-dependent)
- Memory efficiency (may allocate large intermediates)
-}

postulate
  eval-pytorch : String → List Tensor → Tensor

--------------------------------------------------------------------------------
-- § 4: Smart Wrapper
--------------------------------------------------------------------------------

{-|
## Verified Wrapper

Convert well-typed Einsum expression to string formula, then evaluate.

**Two-stage verification**:
1. **Type safety** (Agda): Einsum construction is well-typed
2. **String conversion** (Agda): Formula correctly represents Einsum
3. **Execution** (PyTorch - trusted): Compute result

**Example**:
```agda
evalEinsum matmul [A, B]
  ≡ eval-pytorch "ij,jk->ik" [A, B]
```
-}

-- Forward reference to toString function (defined in ToString.agda)
postulate
  einsumToString : {ins : List IndexCtx} {out : IndexCtx}
                 → Einsum ins out
                 → String

-- Smart wrapper combining type safety + execution
evalEinsum : {ins : List IndexCtx} {out : IndexCtx}
           → Einsum ins out
           → List Tensor
           → Tensor
evalEinsum expr tensors = eval-pytorch (einsumToString expr) tensors

--------------------------------------------------------------------------------
-- § 5: Examples (Placeholders)
--------------------------------------------------------------------------------

{- TODO: Uncomment when SmartConstructors is fixed
module OracleExamples where
  open SmartConstructors

  -- These will execute when we have the full pipeline
  postulate
    example-A : Tensor  -- 2×3 matrix
    example-B : Tensor  -- 3×4 matrix

  -- Matrix multiply via oracle
  example-matmul : Tensor
  example-matmul = evalEinsum matmul (example-A ∷ example-B ∷ [])

  -- Expected: 2×4 matrix
  -- (This would be verified by running the extracted Haskell code)
-}

--------------------------------------------------------------------------------
-- § 6: Future Work
--------------------------------------------------------------------------------

{-|
## Potential Extensions

1. **Tensor validation**: Add predicate `valid-tensor : Tensor → Type`
   - Check length data ≡ product of dimensions
   - Use in eval-pytorch precondition

2. **Dtype support**: Parameterize Tensor by dtype (Float32, Int64, etc.)
   ```agda
   data DType : Type where
     float32 int64 bool : DType

   record Tensor (dt : DType) : Type where
     ...
   ```

3. **Device specification**: CPU vs GPU
   ```agda
   data Device : Type where
     cpu gpu : Device

   eval-pytorch-on : Device → String → List Tensor → Tensor
   ```

4. **Gradients**: Extend oracle to return gradients
   ```agda
   eval-pytorch-grad : String → List Tensor → (Tensor × List Tensor)
   ```

5. **Numerical properties**: Postulate axioms about precision
   ```agda
   postulate
     eval-associative : (e1 e2 e3 : Einsum _ _)
                      → eval (e1 ⨾ (e2 ⨾ e3)) ≈ eval ((e1 ⨾ e2) ⨾ e3)
   ```
   where `≈` is approximate equality up to floating point error

6. **Performance model**: Cost estimation
   ```agda
   cost : Einsum ins out → Nat  -- FLOP count
   ```
-}
