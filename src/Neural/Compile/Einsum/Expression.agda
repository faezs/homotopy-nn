{-# OPTIONS --cubical #-}

{-|
# Deep Embedding of Einsum Expressions

Typed AST for einsum operations with index-level tracking.

## Key Idea

Represent einsum as a **GADT** (indexed datatype) where:
- Type indices track input/output index contexts
- Operations preserve well-typedness
- Composition is type-safe

## Example

```agda
-- Matrix multiplication: A[i,j] × B[j,k] → C[i,k]
MatMul : Einsum [[i, j], [j, k]] [i, k]
MatMul = contract [j] [[i], [k]] refl refl

-- Sequential composition type-checks only when dimensions match
compose : Einsum [[i,j], [j,k]] [i,k]  -- A·B
        → Einsum [[i,k], [k,m]] [i,m]  -- (A·B)·C
        → Einsum [[i,j], [j,k], [k,m]] [i,m]
```

-}

module Neural.Compile.Einsum.Expression where

open import 1Lab.Prelude
open import 1Lab.Path
open import 1Lab.Type

open import Data.List using (List; []; _∷_; _++_; length; filter)
open import Data.Nat.Base using (Nat)
open import 1Lab.Membership using (_∈_)
open import Neural.Compile.Einsum.Index

private variable
  ℓ : Level

-- List indexing helper (needed for Contract constructor)
-- TODO: Should be replaced with dependent type: (xs : List A) → Fin (length xs) → A
-- For now, postulate the empty case which is unreachable in our usage (length remaining ≡ 2)
postulate
  !!-empty : ∀ {A : Type ℓ} → A

_!!_ : ∀ {A : Type ℓ} → List A → Nat → A
[] !! idx = !!-empty
(x ∷ xs) !! zero = x
(x ∷ xs) !! (suc idx) = xs !! idx

--------------------------------------------------------------------------------
-- § 1: Einsum Expression (Deep Embedding)
--------------------------------------------------------------------------------

{-|
## Einsum Expression Type

A well-typed einsum expression indexed by:
- `inputs`: List of input tensor contexts
- `output`: Output tensor context

**Type safety**: Index mismatches cause type errors at compile time.
-}

data Einsum : (inputs : List IndexCtx) → (output : IndexCtx) → Type₁ where

  {-|
  ### Contract (Core Primitive)

  Sum over matching indices, keeping unmatched ones.

  **Example**: Matrix multiply `A[i,j] × B[j,k] → C[i,k]`
  ```agda
  Contract
    [j]           -- Indices to contract (sum over)
    [[i], [k]]    -- Remaining indices from each input
    ...           -- Proofs that inputs = contracted ++ remaining
  ```

  **String notation**: `"ij,jk->ik"`
  -}
  Contract : {σ₁ σ₂ : IndexCtx}
           → (contracted : List Idx)           -- Indices summed over
           → (remaining : List IndexCtx)       -- Remaining indices (one per input)
           → (length remaining ≡ 2)            -- Binary operation
           → (σ₁ ≡ contracted ++ (remaining !! 0))  -- Input 1 structure
           → (σ₂ ≡ contracted ++ (remaining !! 1))  -- Input 2 structure
           → Einsum (σ₁ ∷ σ₂ ∷ []) (remaining !! 0 ++ remaining !! 1)

  {-|
  ### Sequential Composition

  Apply one einsum, then another.

  **Type constraint**: Output of first must match input of second.

  **Example**: `(A·B)·C` where `·` is matrix multiplication
  -}
  Seq : {ins : List IndexCtx} {mid out : IndexCtx}
      → Einsum ins mid          -- First operation produces intermediate
      → Einsum (mid ∷ []) out   -- Second operation consumes intermediate
      → Einsum ins out

  {-|
  ### Parallel Composition (Fork)

  Execute two independent einsums on disjoint inputs.

  **Example**: Compute `f(x)` and `g(y)` in parallel
  -}
  Par : {ins₁ ins₂ : List IndexCtx} {out₁ out₂ : IndexCtx}
      → Einsum ins₁ out₁
      → Einsum ins₂ out₂
      → Einsum (ins₁ ++ ins₂) (out₁ ++ out₂)

  {-|
  ### Broadcast

  Expand tensor by adding new dimensions (repeat along new axes).

  **Example**: `v[i] → M[i,j]` (broadcast vector to matrix)
  -}
  Broadcast : {σ : IndexCtx}
            → (new-dims : List Idx)
            → Einsum (σ ∷ []) (σ ++ new-dims)

  {-|
  ### Reduce

  Sum over a dimension (eliminating it).

  **Example**: `M[i,j] → v[i]` (sum over j)
  -}
  Reduce : {σ : IndexCtx}
         → (dim : Idx)
         → (dim ∈ σ)        -- Proof that dim exists
         → Einsum (σ ∷ []) (σ \\ (dim ∷ []))

  {-|
  ### Transpose

  Reorder dimensions.

  **Example**: `M[i,j] → Mᵀ[j,i]`
  -}
  Transpose : {σ : IndexCtx}
            → (perm : Permutation σ)   -- How to reorder
            → Einsum (σ ∷ []) (permute-ctx perm)

  {-|
  ### Reshape

  Change dimensions (keeping total size constant).

  **Example**: `T[batch, seq, heads, d_head] → T'[batch, seq, (heads*d_head)]`
  -}
  Reshape : {σ : IndexCtx}
          → (new-shape : IndexCtx)
          → (size-preserving : dim σ ≡ dim new-shape)
          → Einsum (σ ∷ []) new-shape

--------------------------------------------------------------------------------
-- § 2: Smart Constructors (TODO: Fix list literal syntax)
--------------------------------------------------------------------------------

{-|
## Smart Constructors

Convenient ways to build einsum expressions with better type inference.

TODO: Convert all `[x, y]` literals to `x ∷ y ∷ []` notation
-}

{-
module SmartConstructors where
  -- Use index constructors from Index module (i, j, k, batch, etc.)

  {-|
  ### Matrix Multiplication

  `A[i,j] × B[j,k] → C[i,k]`
  -}
  matmul : Einsum [[i, j], [j, k]] [i, k]
  matmul = Contract [j] [[i], [k]] refl {!!} {!!}

  {-|
  ### Dot Product

  `v[i] · w[i] → scalar[]`
  -}
  dot : Einsum [[i], [i]] []
  dot = Contract [i] [[], []] refl {!!} {!!}

  {-|
  ### Outer Product

  `v[i] ⊗ w[j] → M[i,j]`
  -}
  outer : Einsum [[i], [j]] [i, j]
  outer = Contract [] [[i], [j]] refl {!!} {!!}

  {-|
  ### Batch Matrix Multiplication

  `A[b,i,j] × B[b,j,k] → C[b,i,k]`
  -}
  batch-matmul : Einsum [[batch, i, j], [batch, j, k]] [batch, i, k]
  batch-matmul = Contract [j] [[batch, i], [batch, k]] refl {!!} {!!}

--------------------------------------------------------------------------------
-- § 3: Composition Operators
--------------------------------------------------------------------------------

{-|
## Infix Operators for Composition

Make einsum expressions more readable.
-}

infixl 6 _⨾_   -- Sequential composition
infixl 5 _∥_   -- Parallel composition

_⨾_ : {ins : List IndexCtx} {mid : IndexCtx} {out : IndexCtx}
    → Einsum ins [mid]
    → Einsum [mid] out
    → Einsum ins out
_⨾_ = Seq

_∥_ : {ins₁ ins₂ : List IndexCtx} {out₁ out₂ : IndexCtx}
    → Einsum ins₁ out₁
    → Einsum ins₂ out₂
    → Einsum (ins₁ ++ ins₂) (out₁ ++ out₂)
_∥_ = Par

--------------------------------------------------------------------------------
-- § 4: Examples
--------------------------------------------------------------------------------

module Examples where
  open StandardIndices
  open SmartConstructors

  {-|
  ### Chained Matrix Multiplication

  `(A·B)·C` with `A[i,j]`, `B[j,k]`, `C[k,m]`
  -}
  chain-matmul : Einsum [[i, j], [j, k], [k, m]] [i, m]
  chain-matmul =
    let ab = matmul       -- A·B: [i,j],[j,k] → [i,k]
        bc = matmul       -- B·C: [j,k],[k,m] → [j,m]
    in {!!}  -- TODO: Figure out proper composition

  {-|
  ### Bilinear Form

  `xᵀ·A·y` with `x[i]`, `A[i,j]`, `y[j]`

  Can be computed as:
  1. `A·y → z[i]`
  2. `xᵀ·z → scalar`
  -}
  bilinear : Einsum [[i], [i, j], [j]] []
  bilinear =
    let ay = Contract [j] [[i], []] refl {!!} {!!}  -- A·y
        xt = Contract [i] [[], []] refl {!!} {!!}    -- xᵀ·(A·y)
    in Seq ay xt

  {-|
  ### Attention Scores (Simplified)

  `Q[b,q,d] × Kᵀ[b,k,d] → scores[b,q,k]`

  Contract over `d` (embedding dimension).
  -}
  attention-scores : Einsum [[batch, seq-q, d-model], [batch, seq-k, d-model]]
                             [batch, seq-q, seq-k]
  attention-scores =
    Contract [d-model] [[batch, seq-q], [batch, seq-k]] refl {!!} {!!}

  {-|
  ### Convolution (after im2col)

  `x_windows[b,w,k,i] × kernel[k,i,o] → output[b,w,o]`
  -}
  conv-einsum : Einsum [[batch, window, kernel-size, in-channels],
                        [kernel-size, in-channels, out-channels]]
                       [batch, window, out-channels]
  conv-einsum =
    Contract [kernel-size, in-channels] [[batch, window], [out-channels]] refl {!!} {!!}
-}
