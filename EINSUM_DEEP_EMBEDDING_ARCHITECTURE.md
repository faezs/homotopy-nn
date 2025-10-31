# Species → Einsum: Deep Embedding Architecture

**Date**: 2025-11-01
**Status**: Foundation laid - Index system + Expression AST created
**Vision**: Typed compilation from combinatorial species to GPU-optimized einsum

---

## 🎯 The Big Picture

```
NeuralNet (high-level operations)
    ↓ network-to-species
Species (combinatorial structure + symmetries)
    ↓ compileSpecies
Einsum Expression (typed AST with index tracking)
    ↓ optimize
Einsum Expression (fused/reordered)
    ↓ emit-triton
Triton GPU Kernel (Python code)
    ↓ execute
Tensor Result
```

**Key insight**: Species capture the **semantics** (what computation means), Einsum captures the **syntax** (how to execute it efficiently).

---

## 📐 Type Structure

### Index System (`Einsum/Index.agda`)

**Index**: Named dimension label (String for flexibility)
```agda
Idx : Type
Idx = String

-- Examples
"i", "j", "k"           -- Generic dimensions
"b"                      -- Batch
"t", "s"                 -- Time/sequence
"h", "e"                 -- Heads, embedding (attention)
"c", "i", "o"            -- Channels (conv)
```

**IndexCtx**: List of dimensions (open/extensible product)
```agda
IndexCtx : Type
IndexCtx = List Idx

-- Examples
[i, j]                   -- 2D matrix
[b, t, d]                -- Batch of sequences
[b, h, q, k]             -- Attention scores
```

**Operations**:
- `_∈ᵢ_` : Index membership
- `_\\_` : Remove indices (set difference)
- `_⊎ᵢ_` : Disjoint union
- `_++ᵢ_` : Append
- `dim` : Count dimensions

### Einsum Expression (`Einsum/Expression.agda`)

**Deep embedding** - GADT indexed by input/output contexts:

```agda
data Einsum : (inputs : List IndexCtx) → (output : IndexCtx) → Type₁ where

  -- Contract: Sum over matching indices
  Contract : (contracted : List Idx)
           → (remaining : List IndexCtx)
           → ...
           → Einsum [σ₁, σ₂] (remaining !! 0 ++ remaining !! 1)

  -- Sequential composition
  Seq : Einsum ins [mid] → Einsum [mid] out → Einsum ins out

  -- Parallel (fork)
  Par : Einsum ins₁ out₁ → Einsum ins₂ out₂
      → Einsum (ins₁ ++ ins₂) (out₁ ++ out₂)

  -- Broadcast (add dimensions)
  Broadcast : (new-dims : List Idx) → Einsum [σ] (σ ++ new-dims)

  -- Reduce (sum over dimension)
  Reduce : (dim : Idx) → (dim ∈ σ) → Einsum [σ] (σ \\ [dim])

  -- Transpose (reorder)
  Transpose : Permutation σ → Einsum [σ] (permute-ctx perm σ)

  -- Reshape (change shape, preserve size)
  Reshape : (new-shape : IndexCtx) → ... → Einsum [σ] new-shape
```

---

## 🔧 Concrete Examples

### Matrix Multiplication

**Operation**: `A[i,j] × B[j,k] → C[i,k]`

**String notation**: `"ij,jk->ik"`

**Deep embedding**:
```agda
MatMul : Einsum [[i, j], [j, k]] [i, k]
MatMul = Contract
  [j]              -- Contract over 'j'
  [[i], [k]]       -- Keep 'i' from A, 'k' from B
  refl ...
```

**Type safety**: Attempting `Contract [x]` on `[[i,j], [j,k]]` would fail because `x ∉ both inputs`.

### Dense Layer

**Operation**: `W[i,j] × x[j] + b[j] → y[i]` (with batch: `x[b,j]`)

**Einsum**: `"bi,ij,j->bj"` (batch-input × weights × bias)

**Deep embedding**:
```agda
DenseEinsum : Einsum [[batch, in], [in, out], [out]] [batch, out]
DenseEinsum =
  let wx = Contract [in] [[batch], [out]] ...  -- W·x
      add-bias = Broadcast [out] ...            -- Add bias
  in Seq wx add-bias
```

### Conv1D (after im2col)

**Operation**: `x_windows[b,w,k,i] × kernel[k,i,o] → output[b,w,o]`

**Einsum**: `"bwki,kio->bwo"`

**Deep embedding**:
```agda
ConvEinsum : Einsum
  [[batch, window, kernel-size, in-channels],
   [kernel-size, in-channels, out-channels]]
  [batch, window, out-channels]

ConvEinsum = Contract [kernel-size, in-channels] [[batch, window], [out-channels]] ...
```

### Multi-Head Attention

**Full sequence** (6 einsum operations):

1. **Project Q**: `Q[b,q,d] × W_q[d,h,e] → Q'[b,q,h,e]`
   `"bqd,dhe->bqhe"`

2. **Project K**: `K[b,k,d] × W_k[d,h,e] → K'[b,k,h,e]`
   `"bkd,dhe->bkhe"`

3. **Project V**: `V[b,v,d] × W_v[d,h,e] → V'[b,v,h,e]`
   `"bvd,dhe->bvhe"`

4. **Scores**: `Q'[b,q,h,e] × K'ᵀ[b,k,h,e] → scores[b,h,q,k]`
   `"bqhe,bkhe->bhqk"`

5. **Apply attention**: `scores[b,h,q,k] × V'[b,k,h,e] → out[b,q,h,e]`
   `"bhqk,bkhe->bqhe"`

6. **Output proj**: `out_flat[b,q,f] × W_out[f,o] → final[b,q,o]`
   `"bqf,fo->bqo"`

**Deep embedding**:
```agda
AttentionEinsum : Einsum [...complex inputs...] [batch, seq-q, d-out]
AttentionEinsum =
  let proj-q = Contract [d-model] [[batch, seq-q], [heads, d-head]] ...
      proj-k = Contract [d-model] [[batch, seq-k], [heads, d-head]] ...
      proj-v = Contract [d-model] [[batch, seq-v], [heads, d-head]] ...
      scores = Contract [heads, d-head] [[batch, seq-q], [batch, seq-k]] ...
      apply = Contract [seq-k] [[batch, heads, seq-q], [heads, d-head]] ...
      output = Contract [heads_x_d-head] [[batch, seq-q], [d-out]] ...
  in Seq (Par (Par (Par proj-q proj-k) proj-v) id)
         (Seq scores (Seq apply output))
```

**Non-linear step**: Softmax applied after step 4 (not represented in einsum)

---

## 🧬 Species Integration

### Species Definition

**Species**: Functor `FinSet → Type` describing combinatorial structures

```agda
Species : Type₁
Species = Nat → Type  -- Structures on n-element sets

-- Operations
_⊕_ : Species → Species → Species  -- Coproduct (disjoint union)
_⊗_ : Species → Species → Species  -- Product (pairs)
_∘ₛ_ : Species → Species → Species  -- Composition
```

### Neural Network → Species

```agda
network-to-species : NeuralNet m n → Species

-- Examples
network-to-species (Prim (Dense W b)) = DenseSpecies m n
network-to-species (f ⊙ g) = network-to-species f ∘ₛ network-to-species g
network-to-species (Fork f g) = network-to-species f ⊕ network-to-species g
```

### Species → Einsum Compilation

```agda
compileSpecies : (S : Species)
               → (weights : WeightSpec S)
               → Σ[ ctx ∈ (List IndexCtx × IndexCtx) ] Einsum (fst ctx) (snd ctx)

-- Examples
compileSpecies (DenseSpecies m n) weights =
  ( [[batch, in], [in, out], [out]]
  , [batch, out]
  , DenseEinsum
  )

compileSpecies (AttentionSpecies d h dh) weights =
  ( [...6 weight matrices...]
  , [batch, seq-q, d-out]
  , AttentionEinsum
  )
```

---

## 🚀 Optimization & Fusion

### Algebraic Laws

**Associativity**:
```agda
(e₁ ⨾ e₂) ⨾ e₃ ≡ e₁ ⨾ (e₂ ⨾ e₃)
```

**Fusion** (merge consecutive contractions):
```agda
optimize : Einsum ins out → Einsum ins out
optimize (Seq (Contract ...) (Contract ...)) = Contract ...  -- Single fused op
```

**Reordering** (for cache locality):
```agda
reorder-for-cache : Einsum ins out → Einsum ins out
```

### Compilation Passes

1. **Fusion**: Merge adjacent operations
2. **CSE**: Eliminate common subexpressions
3. **Loop reordering**: Optimize memory access patterns
4. **Kernel selection**: Choose best Triton kernel for pattern

---

## 📝 Code Generation

### Python/PyTorch

```agda
emit-python : Einsum ins out → String

-- Example
emit-python MatMul
-- ⇒ "torch.einsum('ij,jk->ik', A, B)"

emit-python DenseEinsum
-- ⇒ "torch.einsum('bi,ij,j->bj', x, W, b)"
```

### Triton GPU Kernel

```agda
emit-triton : Einsum ins out → String

-- Example (conceptual)
emit-triton AttentionEinsum
-- ⇒ Multi-kernel fused attention with:
--    - Shared memory tiling
--    - Warp-level primitives
--    - Flash attention optimizations
```

---

## 📁 File Structure

### Created
- ✅ `src/Neural/Compile/Einsum/Index.agda` (~266 lines, 2 holes)
- ✅ `src/Neural/Compile/Einsum/Expression.agda` (~237 lines, created)

### Planned
- ⏳ `src/Neural/Compile/Einsum/Eval.agda` - Execution semantics
- ⏳ `src/Neural/Compile/Einsum/Optimize.agda` - Fusion & rewriting
- ⏳ `src/Neural/Compile/Einsum/Emit.agda` - Code generation
- ⏳ `src/Neural/Combinatorial/Species.agda` - Species theory
- ⏳ `src/Neural/Combinatorial/SpeciesToEinsum.agda` - Compilation

---

## 🎯 Integration Points

### With ForkExtract.agda

```agda
-- Alternative extraction path
extract-via-einsum : NeuralNet m n → String  -- Python code
extract-via-einsum net =
  let species = network-to-species net
      einsum = compileSpecies species (extract-weights net)
      optimized = optimize einsum
  in emit-python optimized
```

### With TritonModel.agda

```agda
-- Prove correctness
einsum-fork-equiv : (net : NeuralNet m n)
                  → einsum-eval (compileSpecies (network-to-species net))
                  ≡ fork-exec (extract net)
```

---

## 🔬 Benefits

### 1. Type Safety
**Index mismatches detected at compile time**:
```agda
-- Type error: Can't contract 'k' from inputs that don't contain it
bad-contract : Einsum [[i, j], [m, n]] [i, n]
bad-contract = Contract [k] [[i], [n]] ...  -- ❌ Type error
```

### 2. Compositionality
**Einsum expressions compose like neural networks**:
```agda
mlp : Einsum [[batch, in1]] [batch, out2]
mlp = dense1 ⨾ relu ⨾ dense2
```

### 3. Optimization
**Algebraic rewriting**:
```agda
-- Fuse two matrix multiplies into single triply-nested loop
(A · B) · C  ⟿  optimized-triple-product A B C
```

### 4. Verification
**Prove correctness**:
```agda
eval-preserves-semantics : (e : Einsum ins out)
                          → eval e ≡ denotational-meaning e
```

### 5. Multiple Backends
**Single source, multiple targets**:
- PyTorch (`torch.einsum`)
- NumPy (`np.einsum`)
- JAX (`jax.numpy.einsum`)
- Triton (custom kernels)
- XLA (TensorFlow compiler)

---

## 🚧 Current Status

**Completed**:
- ✅ Index system with contexts
- ✅ Deep embedding GADT structure
- ✅ Smart constructors for common operations
- ✅ Examples (MatMul, Dense, Conv, Attention)

**Next Steps**:
1. Fill Expression.agda holes (proofs for Contract constructors)
2. Implement Eval.agda (tensor execution semantics)
3. Create Species.agda (basic species theory)
4. Implement SpeciesToEinsum.agda (compilation function)
5. Add Optimize.agda (fusion passes)
6. Add Emit.agda (code generation)

**Timeline**: ~2-3 hours for core infrastructure, then integration with existing modules.

---

## 💡 Vision: End-to-End Verified Compilation

```
User writes:
    mlp = Dense 128 784 ⊙ ReLU ⊙ Dense 10 128

Compiler produces:
    torch.einsum('bi,ij->bj',
                 torch.relu(torch.einsum('bi,ij->bj', x, W1)),
                 W2)

Theorem guarantees:
    exec(compile(mlp)) ≡ ⟦mlp⟧  (denotational semantics)
```

**The dream**: Write high-level neural network code with mathematical semantics, get optimized GPU kernels with verified correctness!

---

**Session**: 2025-11-01
**Achievement**: Laid foundation for Species → Einsum compilation with deep embedding
**Status**: Index + Expression modules created, ready for evaluation semantics
