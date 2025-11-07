{-# OPTIONS --cubical --allow-unsolved-metas #-}

{-|
# JAX Compilation Bridge for Attention Mechanisms

Compiles the 3-categorical attention formalization to JAX operations.

The key insight: Attention is a degree-3 polynomial, which decomposes as:
  Attention = Linear × Softmax(Bilinear) × Linear

This module serializes to JSON for JAX runtime execution.
-}

module Neural.Attention.JAX where

open import 1Lab.Prelude
open import 1Lab.Path
open import 1Lab.Type

-- Basic types
open import Data.Bool using (Bool; true; false)
open import Data.Nat using (Nat; _+_; _*_; zero; suc)
open import Data.Fin using (Fin; fzero; fsuc)
open import Data.String.Base using (String)

-- List type
data List (A : Type) : Type where
  [] : List A
  _∷_ : A → List A → List A

infixr 5 _∷_

-- Import attention 3-category
open import Neural.Attention.Tricategory hiding (JSON; _×_; List; _∷_; []; String)
  using (NeuralSemiring; ℝ⁺-semiring; TensorSpace; mk-space; SmoothMap; LinearProjection;
         BilinearForm; SoftmaxFunctor; AttentionHead; MultiHeadAttention)

-- Import real numbers
open import Neural.Smooth.Base using (ℝ)

-- JSON type (simplified for now)
data JSON : Type where
  json-null : JSON
  json-bool : Bool → JSON
  json-nat : Nat → JSON
  json-string : String → JSON
  json-array : List JSON → JSON
  json-object : List (String × JSON) → JSON

--------------------------------------------------------------------------------
-- § 1: JAX Operation AST
--------------------------------------------------------------------------------

-- Helper functions
postulate
  list-map : ∀ {A B : Type} → (A → B) → List A → List B
  json-real : ℝ → JSON

-- Note: String literals used directly in code below (no need to postulate)

-- JAX tensor operation types
data JAXOp : Type where
  -- Tensor operations
  jax-einsum : String → JAXOp               -- Einstein summation
  jax-matmul : JAXOp                        -- Matrix multiplication
  jax-add : JAXOp                            -- Element-wise addition
  jax-mul : JAXOp                            -- Element-wise multiplication
  jax-transpose : List Nat → JAXOp          -- Axis permutation
  jax-reshape : List Nat → JAXOp            -- Reshape tensor
  jax-broadcast : List Nat → JAXOp          -- Broadcasting

  -- Activation functions
  jax-softmax : Nat → JAXOp                 -- Softmax along axis
  jax-relu : JAXOp                           -- ReLU activation
  jax-tanh : JAXOp                           -- Tanh activation
  jax-gelu : JAXOp                           -- GELU activation

  -- Neural operations
  jax-linear : Nat → Nat → JAXOp            -- Linear layer (in_dim, out_dim)
  jax-layer-norm : List Nat → JAXOp         -- Layer normalization
  jax-dropout : ℝ → JAXOp                   -- Dropout with rate

-- JAX expression tree
data JAXExpr : Type where
  -- Variables
  jax-var : String → JAXExpr                -- Variable reference
  jax-param : String → List Nat → JAXExpr   -- Parameter with shape

  -- Operations
  jax-apply : JAXOp → List JAXExpr → JAXExpr -- Apply operation
  jax-let : String → JAXExpr → JAXExpr → JAXExpr -- Let binding
  jax-tuple : List JAXExpr → JAXExpr        -- Tuple of expressions

--------------------------------------------------------------------------------
-- § 2: Attention to JAX Compilation
--------------------------------------------------------------------------------

-- Compile linear projection to JAX
compile-linear : ∀ {S : NeuralSemiring} {d-in d-out : Nat} →
                 LinearProjection S d-in d-out → JAXExpr
compile-linear {S} {d-in} {d-out} proj =
  jax-apply (jax-linear d-in d-out)
            (jax-param "W" (d-out ∷ d-in ∷ []) ∷
             jax-param "b" (d-out ∷ []) ∷ [])

-- Compile bilinear form to JAX (scaled dot product)
compile-bilinear : ∀ {S : NeuralSemiring} {d : Nat} →
                   BilinearForm S d → JAXExpr
compile-bilinear {S} {d} bilinear =
  let scale-expr = jax-var "scale"  -- 1/√d
      q = jax-var "Q"
      k = jax-var "K"
  in jax-apply (jax-mul)
               (scale-expr ∷
                jax-apply (jax-einsum "bij,bij->bi") (q ∷ k ∷ []) ∷ [])

-- Compile softmax functor
compile-softmax : ∀ {S : NeuralSemiring} {d : Nat} →
                  SoftmaxFunctor S → JAXExpr
compile-softmax {S} {d} softmax =
  jax-apply (jax-softmax 1) (jax-var "scores" ∷ [])  -- Softmax over sequence dim

-- Compile single attention head
compile-head : ∀ {S : NeuralSemiring} {d-model d-k d-v : Nat} →
               AttentionHead S d-model d-k d-v → JAXExpr
compile-head {S} {d-model} {d-k} {d-v} head =
  jax-let "X" (jax-var "input") $
  jax-let "Q" (jax-apply (jax-linear d-model d-k)
                         (jax-param "W_Q" (d-k ∷ d-model ∷ []) ∷
                          jax-var "X" ∷ [])) $
  jax-let "K" (jax-apply (jax-linear d-model d-k)
                         (jax-param "W_K" (d-k ∷ d-model ∷ []) ∷
                          jax-var "X" ∷ [])) $
  jax-let "V" (jax-apply (jax-linear d-model d-v)
                         (jax-param "W_V" (d-v ∷ d-model ∷ []) ∷
                          jax-var "X" ∷ [])) $
  jax-let "scores" (jax-apply (jax-einsum "bqd,bkd->bqk")
                               (jax-var "Q" ∷ jax-var "K" ∷ [])) $
  jax-let "scale" (jax-param "scale" (1 ∷ [])) $  -- 1/√d_k
  jax-let "scores_scaled" (jax-apply jax-mul
                                     (jax-var "scores" ∷ jax-var "scale" ∷ [])) $
  jax-let "weights" (jax-apply (jax-softmax 2) (jax-var "scores_scaled" ∷ [])) $
  jax-apply (jax-einsum "bqk,bkd->bqd")
            (jax-var "weights" ∷ jax-var "V" ∷ [])

-- Helper: build list from function
build-list : ∀ {A : Type} n → (Fin n → A) → List A
build-list zero f = []
build-list (suc n) f = f fzero ∷ build-list n (λ i → f (fsuc i))

-- Compile multi-head attention
compile-mha : ∀ {S : NeuralSemiring} {n-heads d-model : Nat} →
              MultiHeadAttention S n-heads d-model → JAXExpr
compile-mha {S} {n-heads} {d-model} mha =
  let d-head = MultiHeadAttention.head-d-v mha
      compile-single-head : Fin n-heads → JAXExpr
      compile-single-head i = compile-head (MultiHeadAttention.heads mha i)

      -- Create list of head outputs
      head-outputs : List JAXExpr
      head-outputs = build-list n-heads compile-single-head

  in jax-let "heads" (jax-tuple head-outputs) $
     jax-let "concat" (jax-apply (jax-reshape (d-model ∷ []))
                                 (jax-var "heads" ∷ [])) $
     jax-apply (jax-linear d-model d-model)
               (jax-param "W_O" (d-model ∷ d-model ∷ []) ∷
                jax-var "concat" ∷ [])

--------------------------------------------------------------------------------
-- § 3: JSON Serialization
--------------------------------------------------------------------------------

-- Convert JAX operation to JSON
jaxop-to-json : JAXOp → JSON
jaxop-to-json (jax-einsum eq) = json-object (("op", json-string "einsum") ∷
                                              ("equation", json-string eq) ∷ [])
jaxop-to-json jax-matmul = json-object (("op", json-string "matmul") ∷ [])
jaxop-to-json jax-add = json-object (("op", json-string "add") ∷ [])
jaxop-to-json jax-mul = json-object (("op", json-string "mul") ∷ [])
jaxop-to-json (jax-transpose axes) = json-object (("op", json-string "transpose") ∷
                                                  ("axes", json-array (list-map json-nat axes)) ∷ [])
jaxop-to-json (jax-reshape shape) = json-object (("op", json-string "reshape") ∷
                                                 ("shape", json-array (list-map json-nat shape)) ∷ [])
jaxop-to-json (jax-broadcast shape) = json-object (("op", json-string "broadcast") ∷
                                                   ("shape", json-array (list-map json-nat shape)) ∷ [])
jaxop-to-json (jax-softmax axis) = json-object (("op", json-string "softmax") ∷
                                                ("axis", json-nat axis) ∷ [])
jaxop-to-json jax-relu = json-object (("op", json-string "relu") ∷ [])
jaxop-to-json jax-tanh = json-object (("op", json-string "tanh") ∷ [])
jaxop-to-json jax-gelu = json-object (("op", json-string "gelu") ∷ [])
jaxop-to-json (jax-linear in-dim out-dim) = json-object (("op", json-string "linear") ∷
                                                         ("in_features", json-nat in-dim) ∷
                                                         ("out_features", json-nat out-dim) ∷ [])
jaxop-to-json (jax-layer-norm axes) = json-object (("op", json-string "layer_norm") ∷
                                                   ("normalized_shape", json-array (list-map json-nat axes)) ∷ [])
jaxop-to-json (jax-dropout rate) = json-object (("op", json-string "dropout") ∷
                                                ("rate", json-real rate) ∷ [])

-- Convert JAX expression to JSON
{-# TERMINATING #-}
jaxexpr-to-json : JAXExpr → JSON
jaxexpr-to-json (jax-var name) = json-object (("type", json-string "var") ∷
                                              ("name", json-string name) ∷ [])
jaxexpr-to-json (jax-param name shape) = json-object (("type", json-string "param") ∷
                                                      ("name", json-string name) ∷
                                                      ("shape", json-array (list-map json-nat shape)) ∷ [])
jaxexpr-to-json (jax-apply op args) = json-object (("type", json-string "apply") ∷
                                                   ("op", jaxop-to-json op) ∷
                                                   ("args", json-array (list-map jaxexpr-to-json args)) ∷ [])
jaxexpr-to-json (jax-let var expr body) = json-object (("type", json-string "let") ∷
                                                       ("var", json-string var) ∷
                                                       ("expr", jaxexpr-to-json expr) ∷
                                                       ("body", jaxexpr-to-json body) ∷ [])
jaxexpr-to-json (jax-tuple exprs) = json-object (("type", json-string "tuple") ∷
                                                 ("exprs", json-array (list-map jaxexpr-to-json exprs)) ∷ [])

--------------------------------------------------------------------------------
-- § 4: Complete Attention Compilation
--------------------------------------------------------------------------------

-- Main compilation function: Attention 3-category to JAX JSON
compile-attention-to-jax : ∀ {n-heads d-model : Nat} →
                          MultiHeadAttention ℝ⁺-semiring n-heads d-model →
                          JSON
compile-attention-to-jax {n-heads} {d-model} mha =
  let jax-expr = compile-mha mha
      json-output = jaxexpr-to-json jax-expr
  in json-object (("type", json-string "multi_head_attention") ∷
                  ("n_heads", json-nat n-heads) ∷
                  ("d_model", json-nat d-model) ∷
                  ("implementation", json-output) ∷ [])

-- Export Python-compatible JAX code generation
postulate
  to-python-jax : JSON → String  -- Convert JSON to Python JAX code

-- Example: Generate JAX implementation
generate-jax-attention : ∀ {n-heads d-model : Nat} →
                        MultiHeadAttention ℝ⁺-semiring n-heads d-model →
                        String
generate-jax-attention mha = to-python-jax (compile-attention-to-jax mha)
