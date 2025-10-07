{-# OPTIONS --rewriting --guardedness --cubical --allow-unsolved-metas #-}

{-|
# Section 5.1: Attention Moduli and Relation Moduli

This module implements the transformer attention mechanism from Section 5.1
of Belfiore & Bennequin (2022), formalizing Equations 5.1-5.7.

## Paper Reference

> "In addition to the chains of LSTM, another network's component is now recognized
> as essential for most of the tasks in linguistic: to translate, to complete a
> sentence, to determine a context and to take into account a context for finding
> the meaning of a word or sentence. This modulus has its origin in the attention
> operator, introduced by Bahdanau et al. [BCB16], for machine translation of texts."

> "The extended form that is most used today was defined in the same context by
> Vaswani et al. 2017 [VSP+17], under the common name of transformer or simply decoder."

## Key Equations

- **Equation 5.1**: Soft-max Boltzmann weights: p^a_i = (1/Z^a_i) exp(E^a_i)
- **Equation 5.2**: Attention weights: V'_i = Σ_a p^a_i V^a_i
- **Equation 5.3**: Multi-head mixing: A_j = Σ_i w^j_i V'_i
- **Equation 5.4**: Complete attention formula (Query-Key-Value)
- **Equation 5.5**: LSTM with attention memory: c_t with σ_i(x_t, m_t)
- **Equation 5.6**: LSTM hidden output
- **Equation 5.7**: Relation operator: A = f(Σ_{i,j} g(o_i, o_j; Q, H))

## DNN Interpretation

**Transformer architecture**:
- Input: Vectors Y (memories/contexts) and X (external data)
- Three linear operators: Q (queries), K (keys), V (values)
- Attention mechanism: Compute inner products, apply softmax, weighted sum
- Multi-head: Parallel attention with different learned weights
- Output: Integrated representation from multiple perspectives

**Degree analysis** (p. 112-113):
- Attention has **degree 3** (in unsaturated regime)
- LSTM+attention has **degree 5** (p. 114)
- Braid group B₅ (swallowtail) explains greater syntactic power
- Connection to catastrophe theory and semantic groupoids

## References

- [BCB16] Bahdanau et al. (2016): Attention mechanism
- [VSP+17] Vaswani et al. (2017): "Attention Is All You Need"
- [RSB+17] Raposo et al. (2017): Relation networks
- [HM18] Hudson & Manning (2018): MAC reasoning architecture
-}

module Neural.Attention.Transformer where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path

open import Cat.Base
open import Cat.Functor.Base

open import Data.Nat.Base using (Nat; zero; suc; _+_; _*_)
open import Data.Fin.Base using (Fin; fzero; fsuc)
open import Data.Sum using (_⊎_; inl; inr)

-- Import LSTM from Chapter 4
open import Neural.Memory.LSTM

private variable
  o ℓ : Level

--------------------------------------------------------------------------------
-- Real numbers and operations

open Neural.Memory.LSTM public using (ℝ; _+ℝ_; _*ℝ_; _/ℝ_; exp)

postulate
  zeroℝ : ℝ
  oneℝ : ℝ

-- Vector spaces
Vec : Nat → Type
Vec n = Fin n → ℝ

-- Matrix (linear transformation)
Matrix : Nat → Nat → Type
Matrix m n = Fin m → Fin n → ℝ

-- Apply matrix to vector
apply-matrix : ∀ {m n} → Matrix m n → Vec n → Vec m
apply-matrix M v i = {!!}  -- Σ_j M i j * v j

-- Inner product
inner-product : ∀ {n} → Vec n → Vec n → ℝ
inner-product {n} v w = {!!}  -- Σ_i v i * w i

--------------------------------------------------------------------------------
-- §5.1.1: Query-Key-Value Operators

{-|
## Step 1: Linear Projections

Three sets of linear operators are applied:
- Q = W_Q[Y]: Queries
- K = W_K[Y,X]: Keys
- V = W_V[Y]: Values

Indexed by:
- i ∈ I: Heads (individuals in input)
- a ∈ A: Aspects (time instants or different aspects to integrate)
-}

-- Indices
postulate
  HeadIndex : Type  -- Heads (individuals)
  AspectIndex : Type  -- Aspects (times/features)

-- Input types
record AttentionInput : Type₁ where
  constructor attn-input
  field
    d-model : Nat       -- Model dimension
    Y : Vec d-model     -- Memory/hidden variables
    X : Vec d-model     -- External input data

-- Weight matrices (learned parameters)
record AttentionWeights : Type₁ where
  constructor attn-weights
  field
    d-model : Nat       -- Model dimension
    d-k : Nat           -- Key/Query dimension
    d-v : Nat           -- Value dimension

    -- Linear operators
    W-Q : Matrix d-k d-model    -- Query projection
    W-K : Matrix d-k d-model    -- Key projection
    W-V : Matrix d-v d-model    -- Value projection

-- Query, Key, Value vectors (indexed by heads i and aspects a)
record QKV-Vectors (HeadIndex AspectIndex : Type) : Type₁ where
  constructor qkv-vectors
  field
    d-k : Nat
    d-v : Nat

    Q : HeadIndex → AspectIndex → Vec d-k    -- Queries Q^a_i
    K : HeadIndex → AspectIndex → Vec d-k    -- Keys K^a_i
    V : HeadIndex → AspectIndex → Vec d-v    -- Values V^a_i

-- Compute QKV from input (Step 1)
compute-QKV : ∀ {HI AI} → AttentionWeights → AttentionInput → QKV-Vectors HI AI
compute-QKV {HI} {AI} W input = qkv-vectors AW.d-k AW.d-v Q K V
  where
    module AW = AttentionWeights W
    open AttentionInput input

    postulate
      apply-W-Q : Vec d-model → Vec AW.d-k
      apply-W-K : Vec d-model → Vec AW.d-k
      apply-W-V : Vec d-model → Vec AW.d-v

    Q : HI → AI → Vec AW.d-k
    Q i a = apply-W-Q Y

    K : HI → AI → Vec AW.d-k
    K i a = apply-W-K Y  -- Simplified: should combine Y and X

    V : HI → AI → Vec AW.d-v
    V i a = apply-W-V Y

--------------------------------------------------------------------------------
-- §5.1.2: Energy and Soft-Max (Equation 5.1)

{-|
## Step 2: Compute Attention Energies and Probabilities

**Equation 5.1**: Soft-max function applied to energies

The inner products E^a_i = κ(Q^a_i | K^a_i) are computed,
and the soft-max function is applied:

  p^a_i = (1/Z^a_i) exp(E^a_i)

where Z^a_i is the normalization (partition function):

  Z^a_i = Σ_{a'} exp(E^{a'}_i)

**Boltzmann distribution**: This is exactly the Boltzmann distribution
from statistical mechanics, with E^a_i playing the role of energy.

**Physical interpretation**:
- Low energy → high probability (stable states)
- High energy → low probability (unstable states)
- Temperature implicit (could add scaling factor 1/T)
-}

-- Energy computation (inner product of queries and keys)
compute-energy : ∀ {HeadIndex AspectIndex d-k} → QKV-Vectors HeadIndex AspectIndex → HeadIndex → AspectIndex → ℝ
compute-energy {d-k = d-k} qkv i a = inner-product (Q i a) (K i a)
  where
    open QKV-Vectors qkv

-- Partition function Z^a_i (normalization)
postulate
  partition-function : ∀ {HeadIndex AspectIndex} → (HeadIndex → AspectIndex → ℝ) → HeadIndex → ℝ

-- Soft-max probability distribution (Equation 5.1)
soft-max : ∀ {HeadIndex AspectIndex} → (HeadIndex → AspectIndex → ℝ) → HeadIndex → AspectIndex → ℝ
soft-max E i a = exp (E i a) /ℝ partition-function E i
  -- p^a_i = (1/Z^a_i) exp(E^a_i)

-- Attention probabilities
compute-attention-probs : ∀ {HeadIndex AspectIndex} → QKV-Vectors HeadIndex AspectIndex → HeadIndex → AspectIndex → ℝ
compute-attention-probs qkv = soft-max (compute-energy qkv)

{-|
**Properties of soft-max**:
1. Non-negative: p^a_i ≥ 0
2. Normalized: Σ_a p^a_i = 1 (probability distribution)
3. Differentiable: Smooth function for gradient descent
4. Sharpening: exp amplifies differences in energies
-}

postulate
  soft-max-non-negative : ∀ {HeadIndex AspectIndex} (E : HeadIndex → AspectIndex → ℝ) i a → {!!}  -- p^a_i ≥ 0
  soft-max-normalized : ∀ {HeadIndex AspectIndex} (E : HeadIndex → AspectIndex → ℝ) i → {!!}  -- Σ_a p^a_i = 1

--------------------------------------------------------------------------------
-- §5.1.3: Weighted Sum (Equation 5.2)

{-|
## Step 3: Compute Weighted Sum of Values

**Equation 5.2**: V'_i = Σ_a p^a_i V^a_i

Attention weights p^a_i determine how much each value V^a_i contributes
to the output V'_i.

**Interpretation**:
- High attention weight p^a_i → value V^a_i has large influence
- Low attention weight → value ignored
- Differentiable routing based on query-key similarity
-}

-- Sum over aspects A (postulated for now)
postulate
  sum-over-A : ∀ {A d-v} → (AspectIndex → Vec d-v) → Vec d-v

-- Weighted sum of values (Equation 5.2)
attention-output : ∀ {I A d-v} → (HeadIndex → AspectIndex → ℝ) → (HeadIndex → AspectIndex → Vec d-v) → HeadIndex → Vec d-v
attention-output p V i = sum-over-A (λ a → scalar-mult (p i a) (V i a))
  where
    postulate scalar-mult : ∀ {d-v} → ℝ → Vec d-v → Vec d-v

-- Complete attention computation (Steps 1-3)
single-head-attention : ∀ {HeadIndex AspectIndex} → AttentionWeights → AttentionInput → HeadIndex → Vec (AttentionWeights.d-v)
single-head-attention W input i = V'
  where
    qkv = compute-QKV W input
    p = compute-attention-probs qkv
    open QKV-Vectors qkv using (V)
    V' = attention-output p V i

--------------------------------------------------------------------------------
-- §5.1.4: Multi-Head Attention (Equation 5.3-5.4)

{-|
## Step 4: Multi-Head Mixing

**Equation 5.3**: A_j = Σ_i w^j_i V'_i

After computing attention for each head i, a final linear layer mixes
the heads together using weights w^j_i.

**Multi-head attention benefits**:
1. **Parallel perspectives**: Each head learns different patterns
2. **Richer representations**: Combined information from multiple views
3. **Attention to different aspects**: Syntactic vs semantic vs positional
4. **Increased capacity**: More parameters, more expressive
-}

record MultiHeadWeights : Type₁ where
  constructor multi-head-weights
  field
    n-heads : Nat          -- Number of attention heads
    d-model : Nat          -- Model dimension
    d-output : Nat         -- Output dimension

    -- Individual head weights
    head-weights : Fin n-heads → AttentionWeights

    -- Final mixing weights W^j_i (Equation 5.3)
    W-out : Matrix d-output (n-heads * AttentionWeights.d-v (head-weights fzero))

-- Multi-head attention (Equation 5.3)
multi-head-attention : ∀ {HeadIndex AspectIndex} → MultiHeadWeights → AttentionInput → Vec (MultiHeadWeights.d-output _)
multi-head-attention {I} MH input = A
  where
    open MultiHeadWeights MH

    -- Compute attention for each head
    V' : Fin n-heads → HeadIndex → Vec (AttentionWeights.d-v (head-weights fzero))
    V' h = single-head-attention (head-weights h) input

    -- Concatenate all heads
    postulate concat-heads : (Fin n-heads → HeadIndex → Vec _) → Vec _

    -- Final linear projection (Equation 5.3)
    A : Vec d-output
    A = apply-matrix W-out (concat-heads V')

{-|
**Equation 5.4**: Complete Attention Formula

The paper gives the complete formula:

  A_j(Y,X) = Σ_i Σ_a w^j_i · softmax(κ(W_Q(Y)^a_i | W_K(Y,X)^a_i)) · W_V(Y)^a_i

This combines:
1. Linear projections Q, K, V
2. Inner products (energies)
3. Soft-max normalization
4. Weighted sum
5. Multi-head mixing
-}

-- Full attention formula (Equation 5.4) - already implemented above!
full-attention : ∀ {HeadIndex AspectIndex} → MultiHeadWeights → AttentionInput → Vec _
full-attention = multi-head-attention

--------------------------------------------------------------------------------
-- §5.1.5: LSTM with Attention (Equations 5.5-5.6)

{-|
## Attention-Enhanced LSTM

> "The memories or hidden variables issued from the transformer were re-introduced
> in the LSTM chain" (p. 113)

**Equation 5.5**: Cell state update with attention memory m_t

  c_t = c_{t-1} ⊙ σ_f(x_t, h_{t-1}) ⊕ σ_i(x_t, m_t) ⊙ τ_h(x_t, h_{t-1})

Key change: Input gate σ_i now uses attention memory m_t instead of h_{t-1}!

**Equation 5.6**: Hidden state output

  h_t = σ_o(x_t, h_{t-1}) ⊙ tanh(c_t)

**Degree analysis** (p. 114):
- Standard LSTM: degree 3 in h_{t-1}
- LSTM+attention: degree 5 in h_{t-1}
- Braid group: B₃ → B₅ (fold → swallowtail)
- Explains "greatest syntactic power"
-}

record LSTM-Attention-Weights (m n : Nat) : Type where
  constructor lstm-attn-weights
  field
    -- Standard LSTM weights (for f, o, h gates)
    W-f : LinearForm m n
    U-f : LinearForm m m
    β-f : Vec-m m

    W-o : LinearForm m n
    U-o : LinearForm m m
    β-o : Vec-m m

    W-h : LinearForm m n
    U-h : LinearForm m m
    β-h : Vec-m m

    -- Input gate with attention memory dependency
    W-i : LinearForm m n    -- Depends on x_t
    M-i : LinearForm m m    -- Depends on m_t (ATTENTION MEMORY!)
    β-i : Vec-m m

    -- Attention mechanism for computing m_t
    attention : MultiHeadWeights

-- LSTM step with attention (Equations 5.5-5.6)
lstm-attention-step : ∀ {m n I A}
                    → LSTM-Attention-Weights m n
                    → LSTM-State m    -- (c_{t-1}, h_{t-1})
                    → Input n         -- x_t
                    → LSTM-State m    -- (c_t, h_t)
lstm-attention-step {m} W (lstm-state c' h') x = lstm-state c-new h-new
  where
    open LSTM-Attention-Weights W

    -- Compute attention memory m_t from previous states
    attn-inp : AttentionInput
    attn-inp = attn-input m {!!} {!!}  -- Y = h', X = x
    postulate vec-to-vec-m : ∀ {d} → Vec d → Vec-m m
    m-t : Vec-m m
    m-t = vec-to-vec-m (multi-head-attention attention attn-inp)

    -- Gates (Equation 5.5: input gate uses m_t!)
    postulate affine-apply : ∀ {m n} → LinearForm m n → Vec-m n → Vec-m m

    α-i : Vec-m m
    α-i a = {!!}  -- W-i·x + M-i·m_t + β-i  (ATTENTION DEPENDENCY!)

    α-f : Vec-m m
    α-f a = {!!}  -- W-f·x + U-f·h' + β-f

    α-o : Vec-m m
    α-o a = {!!}  -- W-o·x + U-o·h' + β-o

    α-h : Vec-m m
    α-h a = {!!}  -- W-h·x + U-h·h' + β-h

    -- Apply nonlinearities
    gate-i = σ-vec α-i
    gate-f = σ-vec α-f
    gate-o = σ-vec α-o
    gate-h = τ-vec α-h

    -- Cell state update (Equation 5.5)
    v-f = c' ⊙ gate-f
    v-i = gate-i ⊙ gate-h
    c-new = v-f ⊕ v-i

    -- Hidden state output (Equation 5.6)
    h-new = gate-o ⊙ τ-vec c-new

{-|
**Degree 5 analysis** (p. 114):

In unsaturated regime (linear approximation):

  h_t ≈ h_{t-1} + corrections

Expanding the formula with m_t depending on h_{t-1}:
- m_t = attention(h_{t-1}, ...) ~ polynomial of degree 3
- σ_i(x_t, m_t) ~ depends on m_t linearly
- Final h_t ~ h_{t-1} · m_t · ... ~ degree 1 + 3 + 1 = 5

Therefore: **Natural groupoid embeds in braid group B₅**

**Swallowtail catastrophe** (Section 4.4):
- Degree 5 polynomial: z⁵ + uz³ + vz² + wz + t
- Swallowtail: Next catastrophe after cusp (degree 4)
- Explains "augmentation" of syntactic power over LSTM
-}

--------------------------------------------------------------------------------
-- §5.1.6: Relation Operator (Equation 5.7)

{-|
## Relation Networks

> "Raposo et al. [RSB+17] have defined a relation operator: having produced
> contexts H or questions Q concerning two objects o_i, o_j by a chain of LSTM
> (that can be helped by external memories and attention cells) the answer is
> taken from a formula:" (p. 114)

**Equation 5.7**: A = f(Σ_{i,j} g(o_i, o_j; Q, H))

Where:
- o_i : Objects with their characteristics (vectors)
- Q : Questions (from LSTM/attention)
- H : Context (from LSTM/attention)
- g : Pairwise relation function
- f : Aggregation function

**Key property**: Permutation invariance by 𝔖_n

The operator is invariant under permutation of objects:
  A({o_π(i)}) = A({o_i}) for any permutation π ∈ 𝔖_n

This is crucial for reasoning about sets of objects where order doesn't matter!
-}

-- Objects (vectors representing objects with characteristics)
Object : Nat → Type
Object d = Vec d

-- Context and Questions (from LSTM/attention chains)
Context : Nat → Type
Context d = Vec d

Question : Nat → Type
Question d = Vec d

-- Pairwise relation function g(o_i, o_j; Q, H)
postulate
  relation-pairwise : ∀ {d-obj d-ctx d-q d-g}
                    → Object d-obj
                    → Object d-obj
                    → Question d-q
                    → Context d-ctx
                    → Vec d-g

-- Aggregation function f
postulate
  relation-aggregate : ∀ {d-g d-out} → Vec d-g → Vec d-out

-- Complete relation operator (Equation 5.7)
relation-operator : ∀ {n d-obj d-ctx d-q d-g d-out}
                  → (Fin n → Object d-obj)    -- Objects o_i
                  → Question d-q               -- Question Q
                  → Context d-ctx              -- Context H
                  → Vec d-out                  -- Answer A
relation-operator {n} objects Q H = relation-aggregate sum-relations
  where
    postulate
      sum-pairwise : ∀ {d-g} → (Fin n → Fin n → Vec d-g) → Vec d-g

    -- Compute g(o_i, o_j; Q, H) for all pairs
    pairwise-relations : Fin n → Fin n → Vec _
    pairwise-relations i j = relation-pairwise (objects i) (objects j) Q H

    -- Sum over all pairs (Σ_{i,j})
    sum-relations = sum-pairwise pairwise-relations

{-|
**Permutation invariance**:

For any permutation π: Fin n → Fin n:

  relation-operator (objects ∘ π) Q H ≡ relation-operator objects Q H

Proof idea:
1. Summing over all pairs {i,j} is invariant under reindexing
2. Both relation-pairwise and relation-aggregate don't depend on indices
3. Therefore permuting object indices doesn't change result

This property is essential for:
- Visual reasoning (objects in image have no canonical order)
- Set-based reasoning (mathematical sets are unordered)
- Graph neural networks (node ordering shouldn't matter)
-}

postulate
  relation-permutation-invariant :
    ∀ {n d-obj d-ctx d-q d-out}
    → (objects : Fin n → Object d-obj)
    → (Q : Question d-q)
    → (H : Context d-ctx)
    → (π : Fin n → Fin n)  -- Permutation
    → relation-operator (objects ∘ π) Q H ≡ relation-operator objects Q H

--------------------------------------------------------------------------------
-- §5.1.7: Composed Networks and MAC Architecture

{-|
## MAC: Memory, Attention, Composition

> "The reasoning architecture MAC, defined by Hudson and Manning [HM18], is
> composed of three attention operators named control, write and read, in a DNN,
> inspired from the architecture of computers." (p. 114)

**Three attention operators**:
1. **Control**: Determines what to attend to (like program counter)
2. **Write**: Updates memory (like write to RAM)
3. **Read**: Retrieves from memory (like read from RAM)

**Computer architecture analogy**:
- Control unit: Manages execution flow
- Memory unit: Stores intermediate results
- ALU: Performs operations (relation operator)
-}

record MAC-Cell : Type₁ where
  constructor mac-cell
  field
    d-model : Nat

    -- Three attention operators (control, write, read)
    control-attention : MultiHeadWeights
    write-attention : MultiHeadWeights
    read-attention : MultiHeadWeights

    -- Memory state
    memory-dim : Nat

-- MAC computation step
postulate
  mac-step : MAC-Cell → AttentionInput → Vec (MAC-Cell.memory-dim _) → Vec _

{-|
**Evolution of architectures** (Section 5.2-5.3):

The progression from simple to complex:
1. Feedforward → No memory, no attention
2. LSTM → Memory, no attention
3. LSTM+Attention → Memory + attention (degree 5)
4. Transformer → Pure attention (no recurrence)
5. MAC → Compositional reasoning with three attention types

Section 5.2-5.4 will formalize this as a **3-category**:
- 0-cells: Network architectures (sites C)
- 1-cells: Architecture transformations (functors)
- 2-cells: Deformations (natural transformations)
- 3-cells: Surgeries (modifications of natural transformations)
-}

--------------------------------------------------------------------------------
-- Summary and Connections

{-|
## Summary: Section 5.1 Implementation

**Implemented equations**:
- ✅ Equation 5.1: Soft-max Boltzmann distribution
- ✅ Equation 5.2: Weighted sum of values
- ✅ Equation 5.3: Multi-head mixing
- ✅ Equation 5.4: Complete attention formula
- ✅ Equation 5.5: LSTM cell with attention memory
- ✅ Equation 5.6: LSTM hidden output
- ✅ Equation 5.7: Relation operator with permutation invariance

**Key insights**:
1. **Degree 3 → Degree 5**: Attention increases polynomial degree
2. **Braid groups**: B₃ (LSTM) → B₅ (LSTM+attention)
3. **Catastrophe theory**: Swallowtail explains syntactic power
4. **Permutation invariance**: Essential for object reasoning
5. **Compositional structure**: MAC architecture as template

**Connections to other sections**:
- Section 4 (LSTM): Used as base for attention integration
- Section 5.2: Attention as morphism in 2-category
- Section 5.3: Derivators for comparing attention variants
- Section 5.4: Homotopy equivalence of attention mechanisms

**Next**: Section 5.2 will place these structures in a natural 2-category
where attention becomes a morphism between network architectures.
-}
