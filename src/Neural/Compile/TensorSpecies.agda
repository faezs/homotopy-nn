{-# OPTIONS --no-import-sorts #-}
{-|
# Tensor Species: Einsum-Based Categorical IR

**Reference**: Andrew Dudzik's "Tensor Species" talk (Topos Institute)
**Paper**: Bergomi & Vertechi (2022) - Parametric Spans

## Core Insight

Neural networks are **tensor species** - functors from finite sets to vector spaces:

```
Species = Functor core(FinSet)^d → FinVect
```

Where:
- d = number of index variables (I, J, K, ...)
- Operations are **einsums** (polynomial functors)
- Composition is automatic from functoriality!
- Gradients are einsums (permute the feet!)

## Connection to Topos Theory

Our sheaf F: Fork-Category^op → Set is a tensor species:

1. **Objects** = Index variables (I, J, K)
2. **Morphisms** = Einsum operations
3. **Fork vertices** = Learnable commutative monoids
4. **Sheaf condition** F(A★) ≅ ∏ F(incoming) = Multi-input einsum

## Why This Is Better Than Generic IR

**OLD (Extract.agda)**: Generic operations with manual composition
```agda
data ComputeOp = op-linear | op-conv | op-fork | ...
```

**NEW (TensorSpecies.agda)**: Einsums with automatic composition
```agda
data EinsumOp = einsum String (List IndexVar)
-- "ij,jk->ik" means matmul (composition automatic!)
```

## Compilation Pipeline

```
Topos Theory (Fork-Category sheaves)
    ↓ extract-species
Tensor Species (einsum operations)
    ↓ serialize-species
JSON (einsum strings + index shapes)
    ↓ Python SpeciesCompiler
JAX (jnp.einsum + learnable monoids)
```

-}

module Neural.Compile.TensorSpecies where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Instances.Sets

open import Data.Nat.Base using (Nat; zero; suc)
open import Data.List.Base using (List; []; _∷_; _++_; length)
open import Data.String.Base using (String)
open import Data.Bool.Base using (Bool; true; false)

-- Use map from 1Lab.Prelude (already imported)
open import 1Lab.Prelude using (map) public

-- Import actual topos types (defined early in Architecture, before errors)
open import Neural.Topos.Architecture
  using (OrientedGraph; ForkVertex; Fork-Category; is-fork-star)
  renaming (module ForkConstruction to ForkMod)

private variable
  o ℓ : Level

{-|
## Index Variables

In einsum notation, these are the indices like i, j, k in "ij,jk->ik".
Each index variable has a shape (dimension).

Example: For MLP 784→256→10:
- I with shape 784 (input)
- J with shape 256 (hidden)
- K with shape 10 (output)
-}

record IndexVar : Type where
  constructor idx
  field
    name : String     -- "I", "J", "K", etc.
    shape : Nat       -- Dimension

open IndexVar public

{-|
## Einsum Operations

An einsum operation is specified by:
1. The einsum string (e.g., "ij,jk->ik")
2. Input index variables
3. Output index variables

Example: Matrix multiplication
```
einsum-matmul : EinsumOp
einsum-matmul = einsum "ij,jk->ik"
                  [ idx "I" 784 , idx "J" 256 , idx "J" 256 , idx "K" 10 ]
                  [ idx "I" 784 , idx "K" 10 ]
```

This compiles to: `jnp.einsum('ij,jk->ik', W1, W2)`
-}

data EinsumOp : Type where
  einsum : (spec : String)           -- Einsum specification string
         → (inputs : List IndexVar)  -- Input indices
         → (outputs : List IndexVar) -- Output indices
         → EinsumOp

  -- Special case: identity (no computation)
  identity : IndexVar → EinsumOp

  -- Elementwise operations (applied to each tensor element)
  elementwise : String → List IndexVar → EinsumOp  -- "relu", "sigmoid", etc.

{-|
## Parametric Spans (Bergomi & Vertechi 2022)

An einsum can be represented as a span diagram:

```
      I × J
     ↙     ↘
    I       J        (trace: I → 1)

    I × J × K
   ↙   ↓    ↘
  I    J    K        (outer: I×J → I×J)

    I × J × K
   ↙   ↓    ↘
 I×J  J×K  I×K       (matmul: I×J, J×K → I×K)
```

We encode these as span data structures for categorical interpretation.
-}

record ParametricSpan : Type where
  constructor span
  field
    apex : List IndexVar         -- Top of span (e.g., I×J×K)
    left-foot : List IndexVar    -- Left projection
    right-foot : List IndexVar   -- Right projection

{-|
## Learnable Commutative Monoids

From "Learning Algebraic Structure" (Ong & Veličković, 2022):

> "A well-behaved aggregator for a GNN is a commutative monoid over its
> latent space... we construct an aggregator of O(log V) depth, yielding
> exponential improvements for both parallelism and dependency length."

For fork vertices A★ with sheaf condition F(A★) ≅ ∏ F(incoming), we use
learnable binary operators instead of hardcoded concat/sum.

Example: Instead of `concat([x1, x2, x3])`, learn `Φ: (x, y) ↦ αx + βy + γ(x⊙y)`
and reduce via `Φ(Φ(x1, x2), x3)` in O(log n) depth.
-}

data MonoidType : Type where
  -- Fixed monoids
  sum-monoid : MonoidType      -- x ⊕ y = x + y
  max-monoid : MonoidType      -- x ⊕ y = max(x, y)
  concat-monoid : MonoidType   -- x ⊕ y = concat(x, y)

  -- Learnable monoid (MLP-parameterized)
  learnable-mlp : (depth : Nat) → MonoidType

record LearnableMonoid : Type where
  constructor monoid
  field
    input-arities : List Nat  -- Dimensions of incoming edges
    output-dim : Nat          -- Dimension of output
    monoid-type : MonoidType  -- Which aggregation to use

    -- For learnable monoids: regularization to enforce commutativity
    commutative-reg : Bool    -- λ * ||Φ(x,y) - Φ(y,x)||² penalty

open LearnableMonoid public

{-|
## Tensor Species

A tensor species is a functor F: core(FinSet)^d → FinVect, represented as:

1. **Index shapes**: Assignment of dimensions to index variables
2. **Operations**: Einsum operations (polynomial functors)
3. **Symmetries**: Permutation equivariance constraints
4. **Monoids**: Aggregators for fork vertices

Example: Simple MLP as tensor species
```
Indices: I=784, J=256, K=10
Operations:
  1. einsum("ij->j", [I], [J])      -- Linear layer 1
  2. elementwise("relu", [J])        -- ReLU activation
  3. einsum("jk->k", [J], [K])      -- Linear layer 2
```
-}

-- Symmetry constraints: σ ∈ Sₙ acts on index n
record SymmetryConstraint : Type where
  constructor symmetry-constraint
  field
    index : String           -- Which index this acts on
    permutation-group : Nat  -- n for Sₙ

record TensorSpecies : Type where
  constructor species
  field
    name : String
    dimension : Nat  -- Number of index variables (d in core(FinSet)^d)

    -- F₀: Index variables → dimensions
    index-shapes : List IndexVar

    -- F₁: Morphisms as einsum operations
    operations : List EinsumOp

    -- Parametric spans for categorical interpretation
    spans : List ParametricSpan

    -- Learnable monoids for fork vertices
    monoids : List LearnableMonoid

    -- Symmetric group actions (permutation equivariance)
    symmetries : List SymmetryConstraint

    -- Input/output indices
    inputs : List String   -- Index variable names for inputs
    outputs : List String  -- Index variable names for outputs

open TensorSpecies public

{-|
## Conversion from Sheaves to Einsum Operations

Given a sheaf F: Fork-Category^op → Set, extract tensor species:

1. **Objects → Index variables**: Each object c becomes an index with shape
2. **Morphisms → Einsums**: Each F₁(f) becomes an einsum operation
3. **Fork vertices → Monoids**: F(A★) ≅ ∏ F(incoming) becomes aggregation
4. **Sheaf gluing → Composition**: Automatic from einsum composition!

Key insight: Sheaf condition IS the einsum constraint!
-}

-- Extract tensor species from a sheaf on Fork-Category
module ExtractSpecies {o ℓ : Level} (Γ : OrientedGraph o ℓ) where
  open ForkMod Γ public

  private
    C = Fork-Category Γ
    module C = Precategory C

  -- Convert sheaf morphism to einsum
  morphism-to-einsum : {F : Functor (C ^op) (Sets o)} →
                       {x y : C.Ob} →
                       (f : C.Hom x y) →
                       EinsumOp
  morphism-to-einsum f = {!!}  -- TODO: pattern match on morphism structure

  -- Extract aggregator from fork vertex
  fork-to-monoid : ForkVertex → LearnableMonoid
  fork-to-monoid (original _) = monoid [] 0 sum-monoid false  -- No aggregation
  fork-to-monoid (fork-star _ _) = monoid [] 0 (learnable-mlp 3) true  -- Learnable!
  fork-to-monoid (fork-tang _ _) = monoid [] 0 sum-monoid false

  -- Main extraction function
  extract-species : (F : Functor (C ^op) (Sets o)) → TensorSpecies
  extract-species F = {!!}  -- TODO: traverse functor structure

  -- TODO: Implement by:
  -- 1. Enumerate objects in C → create IndexVar for each
  -- 2. For each morphism f, convert F₁(f) to einsum via morphism-to-einsum
  -- 3. For fork vertices, extract learnable monoid via fork-to-monoid
  -- 4. Use functoriality F-∘ to derive composition (automatic in einsum!)

{-|
## Gradient Duality

Key theorem from Dudzik's talk:

> "The gradient flow through an einsum is an einsum. Hint: Permute the feet!"

For einsum 'ij,jk->ik' (matmul), the gradient is:
- ∂/∂(input1): einsum 'ik,jk->ij' (transpose second input)
- ∂/∂(input2): einsum 'ij,ik->jk' (transpose first input)

This means backpropagation is FREE - it's just einsum dualization!
-}

-- Gradient einsum by permuting the span feet
gradient-einsum : EinsumOp → EinsumOp
gradient-einsum (einsum spec ins outs) =
  einsum (grad-spec spec) outs ins  -- Swap inputs/outputs!
  where postulate grad-spec : String → String  -- TODO: implement permutation
gradient-einsum (identity i) = identity i
gradient-einsum (elementwise nm is) = elementwise (grad-name nm) is
  where postulate grad-name : String → String  -- e.g., "relu" → "relu_grad"

{-|
## Examples

### Example 1: Simple MLP (784 → 256 → 10)

```agda
mlp-species : TensorSpecies
mlp-species = species
  "SimpleMLP"
  3  -- 3 index variables: I, J, K
  [ idx "I" 784 , idx "J" 256 , idx "K" 10 ]
  [ einsum "ij->j" [ idx "I" 784 ] [ idx "J" 256 ]
  , elementwise "relu" [ idx "J" 256 ]
  , einsum "jk->k" [ idx "J" 256 ] [ idx "K" 10 ]
  ]
  []  -- No explicit spans (derived from einsums)
  []  -- No fork vertices
  []  -- No symmetry constraints
  [ "I" ]
  [ "K" ]
```

### Example 2: Attention Mechanism

From the slides, attention is:
```
Q: n×k, K: m×k, V: m×l
Match: einsum('nk,mk->nm', Q, K)        -- Query-key similarity
Select: softmax(nm) → nm                 -- Probability distribution
Sample: einsum('nm,ml->nl', probs, V)    -- Aggregate values
```

As tensor species:
```agda
attention-species : TensorSpecies
attention-species = species
  "Attention"
  4  -- n, k, m, l
  [ idx "n" 1 , idx "k" 64 , idx "m" 10 , idx "l" 64 ]
  [ einsum "nk,mk->nm" [ idx "n" 1 , idx "k" 64 , idx "m" 10 , idx "k" 64 ]
                       [ idx "n" 1 , idx "m" 10 ]
  , elementwise "softmax" [ idx "n" 1 , idx "m" 10 ]
  , einsum "nm,ml->nl" [ idx "n" 1 , idx "m" 10 , idx "m" 10 , idx "l" 64 ]
                       [ idx "n" 1 , idx "l" 64 ]
  ]
  []
  []
  []
  [ "n" , "k" , "m" , "l" ]
  [ "n" , "l" ]
```

### Example 3: ResNet with Fork (Learnable Monoid)

```agda
resnet-block-species : TensorSpecies
resnet-block-species = species
  "ResNetBlock"
  3  -- Indices: B (batch), H (height), C (channels)
  [ idx "B" 32 , idx "H" 64 , idx "C" 64 ]
  [ einsum "bhc,cxy,dxy->bhd" ...  -- Conv1
  , elementwise "relu" ...
  , einsum "bhd,dxy,cxy->bhc" ...  -- Conv2
  -- Fork: Add residual (learnable monoid instead of fixed addition!)
  ]
  []
  [ monoid [ 64 , 64 ] 64 (learnable-mlp 3) true ]  -- Learnable aggregator
  []
  [ "B" , "H" , "C" ]
  [ "B" , "H" , "C" ]
```
-}

-- Example instances (postulated for now)
postulate
  mlp-species : TensorSpecies
  attention-species : TensorSpecies
  resnet-block-species : TensorSpecies

{-|
## Serialization to JSON

Export tensor species to JSON for Python compilation.

Format:
```json
{
  "name": "SimpleMLP",
  "dimension": 3,
  "index_shapes": [
    {"name": "I", "shape": 784},
    {"name": "J", "shape": 256},
    {"name": "K", "shape": 10}
  ],
  "operations": [
    {"type": "einsum", "spec": "ij->j", "inputs": ["I"], "outputs": ["J"]},
    {"type": "elementwise", "name": "relu", "indices": ["J"]},
    {"type": "einsum", "spec": "jk->k", "inputs": ["J"], "outputs": ["K"]}
  ],
  "monoids": [],
  "inputs": ["I"],
  "outputs": ["K"]
}
```
-}

-- JSON type (simplified)
data JSON : Type where
  json-string : String → JSON
  json-number : Nat → JSON
  json-bool : Bool → JSON
  json-array : List JSON → JSON
  json-object : List (String × JSON) → JSON

-- Serialize index variable
serialize-index : IndexVar → JSON
serialize-index (idx n s) =
  json-object (("name" , json-string n) ∷ ("shape" , json-number s) ∷ [])

-- Serialize einsum operation
serialize-einsum : EinsumOp → JSON
serialize-einsum (einsum spec ins outs) =
  json-object
    (("type" , json-string "einsum") ∷
     ("spec" , json-string spec) ∷
     ("inputs" , json-array (map (λ i → json-string (name i)) ins)) ∷
     ("outputs" , json-array (map (λ o → json-string (name o)) outs)) ∷
     [])
serialize-einsum (identity i) =
  json-object
    (("type" , json-string "identity") ∷
     ("index" , json-string (name i)) ∷
     [])
serialize-einsum (elementwise nm idxs) =
  json-object
    (("type" , json-string "elementwise") ∷
     ("name" , json-string nm) ∷
     ("indices" , json-array (map (λ i → json-string (name i)) idxs)) ∷
     [])

-- Serialize monoid
serialize-monoid : LearnableMonoid → JSON
serialize-monoid (monoid arities out-dim mtype comm-reg) =
  json-object
    (("input_arities" , json-array (map json-number arities)) ∷
     ("output_dim" , json-number out-dim) ∷
     ("monoid_type" , serialize-monoid-type mtype) ∷
     ("commutative_reg" , json-bool comm-reg) ∷
     [])
  where
    serialize-monoid-type : MonoidType → JSON
    serialize-monoid-type sum-monoid = json-string "sum"
    serialize-monoid-type max-monoid = json-string "max"
    serialize-monoid-type concat-monoid = json-string "concat"
    serialize-monoid-type (learnable-mlp d) =
      json-object (("type" , json-string "learnable") ∷ ("depth" , json-number d) ∷ [])

-- Serialize complete tensor species
serialize-species : TensorSpecies → JSON
serialize-species (species nm dim idxs ops spans mons syms ins outs) =
  json-object
    (("name" , json-string nm) ∷
     ("dimension" , json-number dim) ∷
     ("index_shapes" , json-array (map serialize-index idxs)) ∷
     ("operations" , json-array (map serialize-einsum ops)) ∷
     ("monoids" , json-array (map serialize-monoid mons)) ∷
     ("inputs" , json-array (map json-string ins)) ∷
     ("outputs" , json-array (map json-string outs)) ∷
     [])

-- Export to JSON string (postulated for now - would need JSON printer)
postulate
  json-to-string : JSON → String
  export-species : TensorSpecies → String
  export-to-file : String → TensorSpecies → String  -- IO in real impl

{-|
## Notes

### Advantages Over Generic IR

1. **Theoretically grounded**: Species theory is the correct abstraction
2. **Compositional**: Einsum composition is automatic from functoriality
3. **Efficient**: Direct compilation to `jnp.einsum` (highly optimized)
4. **Gradient-friendly**: "Gradient of einsum is einsum" (proven theorem!)
5. **Expressive**: Learnable monoids for aggregation (O(log n) depth)

### Connection to Research

- **Dudzik (2024)**: Tensor species formalism
- **Bergomi & Vertechi (2022)**: Parametric spans for einsums
- **Ong & Veličković (2022)**: Learnable commutative monoids for GNNs
- **Belfiore & Bennequin (2022)**: Topos theory for DNNs (our foundation)

### Next Steps

1. Implement `extract-species` from Fork-Category sheaves
2. Python `SpeciesCompiler` that interprets einsums directly
3. `LearnableMonoidAggregator` with commutativity regularization
4. Extract ConvNet from Topos.Examples and compile to JAX
5. Benchmark against PyTorch

-}
