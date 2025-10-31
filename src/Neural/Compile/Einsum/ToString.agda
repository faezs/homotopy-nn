{-# OPTIONS --cubical #-}

{-|
# Einsum to String Conversion (Zero Postulates!)

Convert well-typed Einsum expressions to S-expression strings for Python parsing.

## Format

S-expression format chosen for:
- Simple to generate in Agda (recursive string concatenation)
- Easy to parse in Python (ast.literal_eval or simple parser)
- Human-readable for debugging

## Examples

```agda
-- Matrix multiply: Contract [j] [[i], [k]]
-- → "(contract [j] [[i] [k]])"

-- Sequential: Seq e1 e2
-- → "(seq (contract ...) (broadcast ...))"

-- Parallel: Par e1 e2
-- → "(par (contract ...) (contract ...))"
```

## Philosophy

**Zero Postulates**: All string operations are concrete and executable.
- String concatenation: _++_ from Data.String
- Show functions: Exhaustive pattern matching
- Type-driven: Einsum structure dictates output format

-}

module Neural.Compile.Einsum.ToString where

open import 1Lab.Prelude
open import 1Lab.Path

open import Data.String.Base using (String)
open import Data.List using (List; []; _∷_; _++_)
open import Data.Nat.Base using (Nat)
open import Meta.Append
open import Neural.Compile.Einsum.Index
open import Neural.Compile.Einsum.Expression

private variable
  ℓ : Level

--------------------------------------------------------------------------------
-- § 1: Basic String Utilities
--------------------------------------------------------------------------------

{-|
## Intercalate

Join list of strings with separator.

**Example**: `intercalate ", " ["a", "b", "c"]` → `"a, b, c"`
-}
intercalate : String → List String → String
intercalate sep [] = ""
intercalate sep (x ∷ []) = x
intercalate sep (x ∷ xs) = x <> sep <> intercalate sep xs

--------------------------------------------------------------------------------
-- § 2: Index to String Conversion
--------------------------------------------------------------------------------

{-|
## Index to String

Exhaustive mapping from Idx constructors to string names.

**Coverage**: All 21 Idx constructors explicitly handled.
-}
idx-to-string : Idx → String
idx-to-string i = "i"
idx-to-string j = "j"
idx-to-string k = "k"
idx-to-string l = "l"
idx-to-string m = "m"
idx-to-string n = "n"
idx-to-string b = "b"
idx-to-string t = "t"
idx-to-string s = "s"
idx-to-string q = "q"
idx-to-string kk = "kk"
idx-to-string v = "v"
idx-to-string h = "h"
idx-to-string e = "e"
idx-to-string d = "d"
idx-to-string c = "c"
idx-to-string ic = "ic"
idx-to-string oc = "oc"
idx-to-string ks = "ks"
idx-to-string w = "w"
idx-to-string o = "o"

{-|
## Index Context to String

Convert list of indices to Python list notation.

**Example**: `[i, j, k]` → `"[i j k]"`

**Note**: Using space-separated for simplicity (Python can split on whitespace)
-}
ctx-to-string : IndexCtx → String
ctx-to-string [] = "[]"
ctx-to-string ctx = "[" <> intercalate " " (list-map idx-to-string ctx) <> "]"
  where
    -- Local map function (Data.List doesn't export map)
    list-map : (Idx → String) → List Idx → List String
    list-map f [] = []
    list-map f (x ∷ xs) = f x ∷ list-map f xs

{-|
## List of Contexts to String

Convert list of index contexts (e.g., for Contract remaining indices).

**Example**: `[[i], [k]]` → `"[[i] [k]]"`
-}
ctxs-to-string : List IndexCtx → String
ctxs-to-string [] = "[]"
ctxs-to-string ctxs = "[" <> intercalate " " (list-map ctx-to-string ctxs) <> "]"
  where
    list-map : (IndexCtx → String) → List IndexCtx → List String
    list-map f [] = []
    list-map f (x ∷ xs) = f x ∷ list-map f xs

--------------------------------------------------------------------------------
-- § 3: Main Conversion Function
--------------------------------------------------------------------------------

{-|
## Einsum to S-Expression String

Recursively convert Einsum AST to S-expression format.

**Format**:
- Contract: `(contract [contracted-indices] [remaining-contexts])`
- Seq: `(seq <sub-expr-1> <sub-expr-2>)`
- Par: `(par <sub-expr-1> <sub-expr-2>)`
- Broadcast: `(broadcast <input-ctx> [new-dims])`
- Reduce: `(reduce <input-ctx> <dim>)`
- Transpose: `(transpose <input-ctx> <perm-ctx>)`
- Reshape: `(reshape <input-ctx> <new-shape>)`

**Zero Postulates**: Exhaustive pattern matching on all 7 constructors.
-}
einsumToString : {ins : List IndexCtx} {out : IndexCtx}
               → Einsum ins out
               → String

-- Contract: Sum over contracted indices, keep remaining
einsumToString (Contract {σ₁} {σ₂} contracted remaining len-proof eq1 eq2) =
  "(contract " <> ctx-to-string contracted <> " " <> ctxs-to-string remaining <> ")"

-- Sequential composition
einsumToString (Seq e1 e2) =
  "(seq " <> einsumToString e1 <> " " <> einsumToString e2 <> ")"

-- Parallel composition
einsumToString (Par e1 e2) =
  "(par " <> einsumToString e1 <> " " <> einsumToString e2 <> ")"

-- Broadcast: Add new dimensions
einsumToString (Broadcast {σ} new-dims) =
  "(broadcast " <> ctx-to-string σ <> " " <> ctx-to-string new-dims <> ")"

-- Reduce: Sum over a dimension
einsumToString (Reduce {σ} dim mem-proof) =
  "(reduce " <> ctx-to-string σ <> " " <> idx-to-string dim <> ")"

-- Transpose: Reorder dimensions
einsumToString (Transpose {σ} perm) =
  "(transpose " <> ctx-to-string σ <> " " <> ctx-to-string (permute-ctx perm) <> ")"

-- Reshape: Change shape (preserve size)
einsumToString (Reshape {σ} new-shape size-proof) =
  "(reshape " <> ctx-to-string σ <> " " <> ctx-to-string new-shape <> ")"

--------------------------------------------------------------------------------
-- § 4: Examples (Concrete Outputs)
--------------------------------------------------------------------------------

{-
module ToStringExamples where
  open StandardIndices

  -- Matrix multiply: A[i,j] × B[j,k] → C[i,k]
  matmul-string : String
  matmul-string = einsumToString matmul
  -- Expected: "(contract [j] [[i] [k]])"

  -- Note: matmul needs to be defined first in Expression.agda SmartConstructors
  -- This is commented out until SmartConstructors is implemented

  -- Broadcast example: v[i] → M[i,j]
  broadcast-example : Einsum ((i ∷ []) ∷ []) (i ∷ j ∷ [])
  broadcast-example = Broadcast {i ∷ []} (j ∷ [])

  broadcast-string : String
  broadcast-string = einsumToString broadcast-example
  -- Expected: "(broadcast [i] [j])"

  -- Sequential example: (A·B)·C
  -- seq-example : Einsum [[i,j], [j,k], [k,m]] [i,m]
  -- seq-example = Seq matmul matmul
  -- (Commented: needs matmul definition)
-}

--------------------------------------------------------------------------------
-- § 5: Python Parser (Reference Implementation)
--------------------------------------------------------------------------------

{-|
## Python Side Parser

Reference implementation for parsing S-expressions into PyTorch code.

```python
def parse_einsum_sexpr(sexpr: str) -> Callable[[List[torch.Tensor]], torch.Tensor]:
    \"\"\"Parse S-expression einsum to PyTorch executor.\"\"\"

    def parse(tokens):
        if tokens[0] == '(':
            op = tokens[1]
            if op == 'contract':
                contracted = parse_list(tokens[2])
                remaining = parse_list_of_lists(tokens[3])
                # Build einsum string: "ij,jk->ik"
                return make_contract_executor(contracted, remaining)
            elif op == 'seq':
                e1 = parse(tokens[2:])
                e2 = parse(...)
                return lambda tensors: e2([e1(tensors)])
            # ... handle other ops

    tokens = tokenize(sexpr)
    return parse(tokens)
```

**Advantages**:
- Simple recursive descent parser
- Direct mapping to PyTorch operations
- No eval() needed (security)
-}

--------------------------------------------------------------------------------
-- § 6: Integration with Oracle
--------------------------------------------------------------------------------

{-|
## Note on Oracle Integration

This module provides the implementation for the `einsumToString` postulate
in Oracle.agda:

```agda
-- In Oracle.agda:
postulate
  einsumToString : {ins : List IndexCtx} {out : IndexCtx}
                 → Einsum ins out
                 → String

-- In ToString.agda:
einsumToString : {ins : List IndexCtx} {out : IndexCtx}
               → Einsum ins out
               → String
einsumToString = ... (concrete implementation)
```

**Status**: No longer a postulate! Fully concrete implementation with:
- ✅ Zero postulates
- ✅ Exhaustive pattern matching
- ✅ Simple string concatenation
- ✅ ~170 lines as estimated
-}

--------------------------------------------------------------------------------
-- § 7: Future Extensions
--------------------------------------------------------------------------------

{-|
## Potential Improvements

1. **Optimized format**: Switch to binary encoding for large expressions
   - Current: Human-readable S-expressions
   - Future: Compact binary format with length prefixes

2. **Direct PyTorch string**: Generate `"ij,jk->ik"` notation directly
   - Pros: Native PyTorch format, no parser needed
   - Cons: Harder to generate for complex Seq/Par compositions

3. **LISP parser on Python side**: Use actual LISP parser library
   - More robust than hand-rolled parser
   - Handles nested structures automatically

4. **JSON format**: Alternative to S-expressions
   - Pros: Standard format, many parsers
   - Cons: More verbose, harder to generate in Agda

5. **Type annotations**: Include shape information in output
   - Debug tool: Verify Python execution matches expected shapes
   - Format: `(contract [j] [[i] [k]] :type [[i j] [j k]] -> [i k])`
-}
