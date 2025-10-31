# Haskell → Python Einsum Bridge

FFI layer connecting Agda-extracted Haskell code to Python PyTorch einsum execution.

## Architecture

```
Agda (ToString.agda)
    ↓ Extract to Haskell
Haskell (PythonBridge.hs)
    ↓ JSON over subprocess stdin/stdout
Python (einsum_session.py)
    ↓ Parse S-expressions
PyTorch (torch.einsum)
    ↓ GPU execution
Tensor Result
```

## Files

- `src/Einsum/PythonBridge.hs` - Main bridge module (~300 lines)
- `app/Main.hs` - Interactive REPL for testing (~80 lines)
- `einsum-bridge.cabal` - Cabal package definition

## Building

### With Nix (recommended)

```bash
# Enter dev environment (has GHC, Cabal, Agda, Python)
$ nix develop

# Build the library
$ cabal build einsum-bridge

# Build and run the REPL
$ cabal run einsum-repl
```

### Without Nix

Requirements:
- GHC 9.10
- Cabal 3.0+
- Python 3.12+ with PyTorch

```bash
$ cabal build
$ cabal run einsum-repl
```

## Usage

### Interactive REPL

```bash
$ cabal run einsum-repl
Einsum Python Bridge REPL
=========================

Starting Python session...
Python session ready!

Available commands:
  matmul  - Matrix multiplication example
  dot     - Dot product example
  quit    - Exit REPL

> matmul
Matrix multiplication: A[j,i] × B[j,k] → C[i,k]
A = Tensor [3,2] [1.0,2.0,3.0,4.0,5.0,6.0]
B = Tensor [3,2] [7.0,8.0,9.0,10.0,11.0,12.0]
Result: Tensor [2,2] [89.0,98.0,116.0,128.0]

> quit
Goodbye!
```

### Library API

```haskell
import Einsum.PythonBridge

main :: IO ()
main = withPythonSession $ \session -> do
  -- Define tensors
  let a = Tensor [3, 2] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let b = Tensor [3, 2] [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

  -- Execute einsum
  result <- evalEinsum session "(contract [j] [[i] [k]])" [a, b]

  print result
  -- Tensor [2,2] [89.0,98.0,116.0,128.0]
```

## Module: `Einsum.PythonBridge`

### Session Management

- `startPythonSession :: IO PythonSession` - Start Python subprocess
- `stopPythonSession :: PythonSession -> IO ()` - Stop subprocess
- `withPythonSession :: (PythonSession -> IO a) -> IO a` - Managed session

### Types

```haskell
data Tensor = Tensor
  { tensorShape :: [Int]
  , tensorData  :: [Double]
  }

data PythonSession = PythonSession
  { pyStdin  :: Handle
  , pyStdout :: Handle
  , pyStderr :: Handle
  , pyProc   :: ProcessHandle
  }
```

### Execution

- `evalEinsum :: PythonSession -> Text -> [Tensor] -> IO Tensor`
  - Execute einsum formula with input tensors
  - Returns result tensor
  - Throws `EinsumError` on failure

### Errors

```haskell
data EinsumError
  = PythonProcessError String      -- Subprocess failed to start
  | ParseError String               -- JSON response malformed
  | EinsumExecutionError Text       -- Python reported error
```

## Protocol

### Request (JSON line to Python stdin)

```json
{
  "formula": "(contract [j] [[i] [k]])",
  "tensors": [
    {"shape": [3, 2], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
    {"shape": [3, 2], "data": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}
  ]
}
```

### Response (JSON line from Python stdout)

**Success**:
```json
{
  "success": true,
  "shape": [2, 2],
  "data": [89.0, 98.0, 116.0, 128.0]
}
```

**Error**:
```json
{
  "success": false,
  "error": "ParseError: Expected '(', got ..."
}
```

## Integration with Agda

The `evalEinsum` function is designed to be called from Agda-extracted Haskell code:

```agda
-- In Oracle.agda
postulate
  eval-pytorch : String → List Tensor → Tensor

-- After extraction to Haskell, implement as:
eval_pytorch :: Text -> [Tensor] -> IO Tensor
eval_pytorch formula tensors = do
  session <- getGlobalSession  -- Persistent session
  evalEinsum session formula tensors
```

## Testing

Run the Python backend tests first:

```bash
$ python3 python-runtime/test_session.py
Testing PyTorch Einsum Session
============================================================
Test 1: Matrix multiplication
✅ Success! Result shape: [2, 2]

Test 2: Dot product
✅ Success! Result: 15.0

Test 3: Transpose
✅ Success! Result shape: [3, 2]

Test 4: Reduce (sum over dimension)
✅ Success! Result shape: [2]

Test 5: Error handling
✅ Error correctly reported

============================================================
All tests completed!
```

Then test the Haskell bridge:

```bash
$ cabal run einsum-repl
> matmul
✅ Success!

> dot
✅ Success!
```

## Dependencies

### Haskell
- `base >= 4.14`
- `aeson >= 2.0` - JSON encoding/decoding
- `bytestring >= 0.10` - Efficient byte arrays
- `text >= 1.2` - Text handling
- `process >= 1.6` - Subprocess management

### Python (runtime)
- `python3` - Python interpreter
- `torch` - PyTorch for einsum execution
- `python-runtime/einsum_parser.py` - S-expression parser
- `python-runtime/einsum_session.py` - REPL server

## Future Work

1. **Persistent global session** - Reuse subprocess across calls
2. **Batch execution** - Send multiple formulas at once
3. **Async API** - Non-blocking einsum execution
4. **Error recovery** - Restart Python on crash
5. **Type-level shapes** - Dependent types for tensor shapes
6. **GPU selection** - Choose specific GPU device

## See Also

- `../src/Neural/Compile/Einsum/` - Agda einsum modules
- `../python-runtime/` - Python backend
- `../EINSUM_PIPELINE_STATUS.md` - Complete session summary
