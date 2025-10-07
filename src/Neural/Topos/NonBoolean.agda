{-# OPTIONS --rewriting --guardedness --cubical --allow-unsolved-metas #-}

{-|
# Appendix D: Non-Boolean Information Functions

This module implements Appendix D from Belfiore & Bennequin (2022), covering:
- Chain sites S_n (posets 0 → 1 → ... → n)
- Injective presheaves and Heyting algebras Ω_E
- Internal implication (Lemma D.1)
- Measure functions and ψ_δ (Definition D.2, Equation 44)
- Concavity (Definition D.3, Equation 45)
- Proposition D.1: Concavity under hypothesis on δ
- Logarithmic precision function ψ = ln ψ_δ

## Key Results

1. **Lemma D.1**: Inductive definition of Q⇒T in Heyting algebra
2. **Lemma D.2**: ψ_δ is strictly increasing
3. **Proposition D.1**: ψ_δ is concave when δ_k > δ_{k+1} + ... + δ_n
4. **Fundamental precision**: ψ = ln ψ_δ is strictly concave

## DNN Interpretation

The chain site models temporal/layered structure in DNNs. The concave precision
function ψ provides a measure of information quality that:
- Increases with larger subobjects (more information)
- Is concave (diminishing returns)
- Extends to rooted trees and DNN posets

-}

module Neural.Topos.NonBoolean where

open import 1Lab.Prelude
open import 1Lab.Path
open import 1Lab.HLevel
open import Data.Sum

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Instances.Functor
open import Cat.Diagram.Limit.Base
open import Cat.Diagram.Colimit.Base

open import Data.Nat.Base using (Nat; suc; zero)
open import Data.Fin.Base using (Fin; fzero; fsuc)
open import Data.Float.Base

open import Order.Base
open import Order.Cat

open import Neural.Memory.LSTM using (ℝ; _+ℝ_; _*ℝ_)

private variable
  o ℓ o' ℓ' : Level

--------------------------------------------------------------------------------
-- Chain Site S_n

{-|
## Chain Site

The site S_n is the poset 0 → 1 → ... → n with the discrete topology.

This models:
- Temporal sequences (time steps 0, 1, ..., n)
- Layer hierarchies (input → hidden₁ → ... → output)
- Information refinement (coarse → ... → fine)
-}

-- Indices for chain
ChainIndex : Nat → Type
ChainIndex n = Fin (suc n)

-- Ordering on chain (using Fin's ordering)
postulate
  _≤-chain_ : ∀ {n} → ChainIndex n → ChainIndex n → Type
  ≤-chain-refl : ∀ {n} {i : ChainIndex n} → i ≤-chain i
  ≤-chain-trans : ∀ {n} {i j k : ChainIndex n} → i ≤-chain j → j ≤-chain k → i ≤-chain k
  ≤-chain-antisym : ∀ {n} {i j : ChainIndex n} → i ≤-chain j → j ≤-chain i → i ≡ j

-- Chain poset as a category (postulated for simplicity)
postulate
  Chain : Nat → Precategory lzero lzero

--------------------------------------------------------------------------------
-- Presheaf Topos Ŝ_n

{-|
## Presheaf Topos

Objects: Functors (Chain n)^op → Sets
- E_0 ← E_1 ← ... ← E_n (contravariant functor)
- E_i are sets, maps are functions

Morphisms: Natural transformations
-}

-- Presheaf on chain (contravariant functor to sets)
record ChainPresheaf (n : Nat) : Type₁ where
  constructor chain-presheaf
  field
    -- Objects at each index
    obj : ChainIndex n → Type
    obj-is-set : ∀ i → is-set (obj i)

    -- Restriction maps (contravariant: j ≤ i gives E_i → E_j)
    restrict : ∀ {i j} → j ≤-chain i → obj i → obj j

    -- Functoriality
    restrict-id : ∀ {i} (x : obj i) → restrict {i} {i} ≤-chain-refl x ≡ x
    restrict-comp : ∀ {i j k} (p : k ≤-chain j) (q : j ≤-chain i) (x : obj i)
                  → restrict p (restrict q x) ≡ restrict (≤-chain-trans p q) x

-- Morphism of presheaves (natural transformation)
record ChainPresheafMorphism {n : Nat} (E F : ChainPresheaf n) : Type where
  constructor presheaf-morphism
  module E = ChainPresheaf E
  module F = ChainPresheaf F
  field
    component : ∀ (i : ChainIndex n) → E.obj i → F.obj i

    -- Naturality: commutes with restrictions
    naturality : ∀ {i j} (p : j ≤-chain i) (x : E.obj i)
               → component j (E.restrict p x) ≡ F.restrict p (component i x)

--------------------------------------------------------------------------------
-- Injective Presheaves

{-|
## Injective Presheaves

A presheaf E is **injective** if all restriction maps E_i → E_{i-1} are injections.

This ensures:
- Information is refined, not lost, as we go up the chain
- Subobjects form a well-behaved Heyting algebra
- Complements are defined via set difference
-}

record is-injective-presheaf {n : Nat} (E : ChainPresheaf n) : Type where
  module E = ChainPresheaf E
  field
    restrict-injective : ∀ {i j} (p : j ≤-chain i) → injective (E.restrict p)

-- Example: Constant presheaf (all E_i equal, identity restrictions)
constant-presheaf : ∀ {n} (A : Type) → is-set A → ChainPresheaf n
constant-presheaf A A-set = chain-presheaf
  (λ i → A)
  (λ i → A-set)
  (λ p x → x)
  (λ x → refl)
  (λ p q x → refl)

constant-presheaf-injective : ∀ {n} (A : Type) (A-set : is-set A)
                             → is-injective-presheaf (constant-presheaf {n} A A-set)
constant-presheaf-injective A A-set .is-injective-presheaf.restrict-injective p {x} {y} q = q

--------------------------------------------------------------------------------
-- Heyting Algebra Ω_E

{-|
## Heyting Algebra of Subobjects

For an injective presheaf E, the subobjects form a Heyting algebra Ω_E:
- Objects: T_n ⊂ T_{n-1} ⊂ ... ⊂ T_0 with T_i ⊆ E_i
- Order: Pointwise inclusion
- ∧: Pointwise intersection
- ∨: Pointwise union
- ⇒: Internal implication (Lemma D.1)
- ¬: Negation Q⇒∅
-}

-- Subobject of E (decreasing sequence of subsets)
record Subobject {n : Nat} (E : ChainPresheaf n) : Type₁ where
  constructor subobj
  module E = ChainPresheaf E
  field
    -- Subset at each level
    subset : ∀ (i : ChainIndex n) → (E.obj i → Type)
    subset-prop : ∀ i x → is-prop (subset i x)

    -- Decreasing: T_i restricts to T_j
    decreasing : ∀ {i j} (p : j ≤-chain i) (x : E.obj i)
               → subset i x → subset j (E.restrict p x)

-- Inclusion order on subobjects
_⊆-sub_ : ∀ {n} {E : ChainPresheaf n} → Subobject E → Subobject E → Type
T ⊆-sub S = ∀ i x → T .Subobject.subset i x → S .Subobject.subset i x

-- Intersection (∧)
_∩-sub_ : ∀ {n} {E : ChainPresheaf n} → Subobject E → Subobject E → Subobject E
(T ∩-sub S) .Subobject.subset i x = T .Subobject.subset i x × S .Subobject.subset i x
(T ∩-sub S) .Subobject.subset-prop i x = ×-is-hlevel 1 (T .Subobject.subset-prop i x) (S .Subobject.subset-prop i x)
(T ∩-sub S) .Subobject.decreasing p x (t , s) = T .Subobject.decreasing p x t , S .Subobject.decreasing p x s

-- Union (∨)
_∪-sub_ : ∀ {n} {E : ChainPresheaf n} → Subobject E → Subobject E → Subobject E
(T ∪-sub S) .Subobject.subset i x = ∥ T .Subobject.subset i x ⊎ S .Subobject.subset i x ∥
(T ∪-sub S) .Subobject.subset-prop i x = squash
(T ∪-sub S) .Subobject.decreasing p x t∪s = {!!}  -- Uses eliminator for ∥_∥

-- Complement (relative to E_i)
complement-at : ∀ {n} {E : ChainPresheaf n} (Q : Subobject E) (i : ChainIndex n)
              → (E .ChainPresheaf.obj i → Type)
complement-at {E = E} Q i x = ¬ (Q .Subobject.subset i x)

--------------------------------------------------------------------------------
-- Internal Implication (Lemma D.1)

{-|
## Lemma D.1: Internal Implication

For T, Q ∈ Ω_E, the implication U = (Q⇒T) is defined inductively:
- U_0 = T_0 ∨ (E_0 \ Q_0)
- U_k = U_{k-1} ∧ (T_k ∨ (E_k \ Q_k))  for k ≥ 1

**Proof**: By induction on the chain length. The base case (n=0) is the
Boolean formula. For n=N, U_N must:
1. Belong to U_{N-1} (by decreasing property)
2. Be the union of all V ⊆ E_N ∩ U_{N-1} such that V ∧ Q_N ⊆ T_N

This gives U_N = (T_N ∩ U_{N-1}) ∪ ((E_N \ Q_N) ∩ U_{N-1}).
-}

-- Step 0: U_0 = T_0 ∨ (E_0 \ Q_0)
implication-base : ∀ {n} {E : ChainPresheaf n} (Q T : Subobject E)
                 → (E .ChainPresheaf.obj fzero → Type)
implication-base {E = E} Q T x =
  ∥ T .Subobject.subset fzero x ⊎ (¬ Q .Subobject.subset fzero x) ∥

-- Inductive step: U_k = U_{k-1} ∧ (T_k ∨ (E_k \ Q_k))
implication-step : ∀ {n} {E : ChainPresheaf n} (Q T : Subobject E)
                 → (k : ChainIndex n)
                 → (E .ChainPresheaf.obj k → Type)  -- U_{k-1} (previous level)
                 → (E .ChainPresheaf.obj k → Type)  -- U_k (current level)
implication-step {E = E} Q T k U-prev x =
  U-prev x × ∥ T .Subobject.subset k x ⊎ (¬ Q .Subobject.subset k x) ∥

-- Full implication (Q⇒T) defined inductively
postulate
  internal-implication : ∀ {n} {E : ChainPresheaf n} → Subobject E → Subobject E → Subobject E

-- Notation: ¬Q = Q⇒∅
postulate
  negation : ∀ {n} {E : ChainPresheaf n} → Subobject E → Subobject E

-- Equation 43: ¬Q is the sequence (E_k \ Q_k) ⊂ ... ⊂ (E_0 \ Q_0)
postulate
  negation-formula : ∀ {n} {E : ChainPresheaf n} (Q : Subobject E) (i : ChainIndex n)
                   → (x : E .ChainPresheaf.obj i)
                   → negation Q .Subobject.subset i x ≡ (¬ Q .Subobject.subset i x)

--------------------------------------------------------------------------------
-- Measure Function μ (Definition D.1)

{-|
## Definition D.1: Measure Function

A strictly positive function μ on E_0 assigns weights to elements.

Common choices:
- μ(x) = 1 (counting measure)
- μ(x) = |F|⁻¹ (uniform probability)

For a subset F ⊆ E_0, μ(F) = Σ_{x∈F} μ(x)
-}

postulate
  zeroℝ : ℝ
  oneℝ : ℝ
  _<ℝ_ : ℝ → ℝ → Type

  -- Strictly positive
  is-positive : ℝ → Type
  positive-add : ∀ {x y} → is-positive x → is-positive y → is-positive (x +ℝ y)

-- Measure function on E_0
record MeasureFunction {n : Nat} (E : ChainPresheaf n) : Type where
  constructor measure-fn
  field
    μ : E .ChainPresheaf.obj fzero → ℝ
    μ-positive : ∀ x → is-positive (μ x)

  -- Sum over a subset F ⊆ E_0
  postulate
    μ-sum : (F : E .ChainPresheaf.obj fzero → Type) → ℝ
    μ-sum-additive : ∀ (F G : E .ChainPresheaf.obj fzero → Type)
                   → {!!}  -- Disjoint → μ(F ∪ G) = μ(F) + μ(G)

-- Example: Counting measure
counting-measure : ∀ {n} (E : ChainPresheaf n) → MeasureFunction E
counting-measure E .MeasureFunction.μ x = oneℝ
counting-measure E .MeasureFunction.μ-positive x = {!!}

--------------------------------------------------------------------------------
-- Function ψ_δ (Definition D.2, Equation 44)

{-|
## Definition D.2: Weighted Sum Function

Given a strictly decreasing sequence δ = [δ_0 > δ_1 > ... > δ_n],
define ψ_δ : Ω_E → ℝ by:

  ψ_δ(T_n ⊂ ... ⊂ T_0) = Σ_{k=0}^n δ_k μ(T_k)     (Equation 44)

**Properties**:
- ψ_δ is strictly increasing (Lemma D.2)
- ψ_δ is concave under hypothesis (Proposition D.1)
-}

-- Decreasing sequence of weights
record WeightSequence (n : Nat) : Type where
  constructor weights
  field
    δ : ChainIndex n → ℝ
    δ-positive : ∀ i → is-positive (δ i)
    δ-decreasing : ∀ {i j} → i ≤-chain j → (δ j <ℝ δ i)  -- Strictly decreasing

-- Hypothesis for concavity: δ_k > δ_{k+1} + ... + δ_n
record ConcavityHypothesis {n : Nat} (δ-seq : WeightSequence n) : Type where
  module W = WeightSequence δ-seq
  field
    hypothesis : ∀ (k : ChainIndex n)
               → {!!}  -- δ_k > Σ_{j=k+1}^n δ_j

-- Example: δ_k = 1/2^k satisfies hypothesis
geometric-weights : ∀ (n : Nat) → WeightSequence n
geometric-weights n .WeightSequence.δ i = {!!}  -- 1/2^i
geometric-weights n .WeightSequence.δ-positive i = {!!}
geometric-weights n .WeightSequence.δ-decreasing p = {!!}

-- Function ψ_δ (Equation 44)
ψ-δ : ∀ {n} {E : ChainPresheaf n}
    → WeightSequence n
    → MeasureFunction E
    → Subobject E
    → ℝ
ψ-δ {n} {E} δ-seq μ-fn T = {!!}  -- Σ_{k=0}^n δ_k μ(T_k)
  where
    module W = WeightSequence δ-seq
    module M = MeasureFunction μ-fn

--------------------------------------------------------------------------------
-- Concavity (Definition D.3)

{-|
## Definition D.3: Concavity

A function φ : Ω_E → ℝ is **concave** if for any T ≤ T' and proposition Q:

  Δφ(Q; T, T') = φ(Q⇒T) - φ(T) - φ(Q⇒T') + φ(T') ≥ 0     (Equation 45)

**Strictly concave** if the inequality is strict when T ≠ T'.

**Intuition**: Information gain from refining T to T' is non-increasing.
-}

-- Concavity difference (Equation 45)
postulate
  Δ-concave : ∀ {n} {E : ChainPresheaf n}
            → (Subobject E → ℝ)  -- Function φ
            → Subobject E         -- Q
            → Subobject E         -- T
            → Subobject E         -- T' (with T ⊆ T')
            → ℝ
  -- Δ-concave φ Q T T' = φ(Q⇒T) - φ(T) - φ(Q⇒T') + φ(T')

-- Concavity predicate
is-concave : ∀ {n} {E : ChainPresheaf n} → (Subobject E → ℝ) → Type₁
is-concave φ = ∀ Q T T' → (T ⊆-sub T') → {!!}  -- Δ-concave φ Q T T' ≥ 0

is-strictly-concave : ∀ {n} {E : ChainPresheaf n} → (Subobject E → ℝ) → Type₁
is-strictly-concave φ = ∀ Q T T' → (T ⊆-sub T') → {!!}  -- Δ-concave φ Q T T' > 0

--------------------------------------------------------------------------------
-- Lemma D.2: ψ_δ is Strictly Increasing

{-|
## Lemma D.2: Monotonicity

For T ⊆ T', we have ψ_δ(T) < ψ_δ(T').

**Proof**: Index by index, T'_k contains T_k, so each term δ_k μ(T'_k) ≥ δ_k μ(T_k).
Since δ_k > 0 and μ is positive, strict inclusion gives strict inequality.
-}

postulate
  ψ-δ-increasing : ∀ {n} {E : ChainPresheaf n}
                 → (δ-seq : WeightSequence n)
                 → (μ-fn : MeasureFunction E)
                 → ∀ T T' → T ⊆-sub T'
                 → ψ-δ δ-seq μ-fn T <ℝ ψ-δ δ-seq μ-fn T'

--------------------------------------------------------------------------------
-- Proposition D.1: Concavity of ψ_δ

{-|
## Proposition D.1: Concavity

Under the hypothesis δ_k > δ_{k+1} + ... + δ_n for all k, the function ψ_δ is concave.

**Proof Strategy**:
1. Define sequence T^{(k)} interpolating from T to T' by enlarging T_k to T'_k index by index
2. Show Δψ_δ(Q; T^{(k-1)}, T^{(k)}) ≥ 0 for each k
3. Use telescoping cancellation: Δψ_δ(Q; T, T') = Σ_k Δψ_δ(Q; T^{(k-1)}, T^{(k)})

**Key Insight**:
- Contribution from index 0: δ_0 μ((E_0 \ Q_0) ∩ (T'_0 \ T_0))
- Contribution from index k ≥ 1: -δ_k μ((T'_0 \ T_0) ∩ (E_0 \ Q_0) ∩ W_k)
- Hypothesis ensures Σ_{k≥1} δ_k μ(...) ≤ δ_0 μ(...)
-}

-- Interpolating sequence T^{(k)} from T to T'
postulate
  interpolate : ∀ {n} {E : ChainPresheaf n}
              → (T T' : Subobject E)
              → (k : ChainIndex n)
              → Subobject E

  interpolate-base : ∀ {n} {E : ChainPresheaf n} (T T' : Subobject E)
                   → interpolate T T' fzero ≡ T

  interpolate-step : ∀ {n} {E : ChainPresheaf n} (T T' : Subobject E) (k : ChainIndex n)
                   → {!!}  -- T^{(k)} enlarges T_k to T'_k

-- Main theorem
postulate
  ψ-δ-concave : ∀ {n} {E : ChainPresheaf n}
              → (δ-seq : WeightSequence n)
              → (μ-fn : MeasureFunction E)
              → ConcavityHypothesis δ-seq
              → is-concave (ψ-δ δ-seq μ-fn)

--------------------------------------------------------------------------------
-- Logarithmic Precision Function (Equation 46)

{-|
## Fundamental Precision Function

Taking ψ = ln ψ_δ gives a **strictly concave** function.

**Why logarithm**:
- ln transforms concave → strictly concave
- Second derivative: (ln φ)'' = (φ φ'' - (φ')²) / φ² < 0  (Equation 46)
- Normalization: 0 < ψ_δ ≤ 1, so -∞ < ψ ≤ 0

**DNN Interpretation**:
- ψ measures precision/information content
- Concavity captures diminishing returns
- Logarithmic scale matches information-theoretic intuition
-}

postulate
  ln : ℝ → ℝ
  ln-derivative : ℝ → ℝ
  ln-second-derivative : ℝ → ℝ

  -- Equation 46: (ln φ)'' = (φ φ'' - (φ')²) / φ²
  ln-second-derivative-formula : ∀ φ φ' φ''
                               → {!!}  -- ln''(φ) = (φ φ'' - (φ')²) / φ²

-- Fundamental precision function
ψ : ∀ {n} {E : ChainPresheaf n}
  → WeightSequence n
  → MeasureFunction E
  → Subobject E
  → ℝ
ψ δ-seq μ-fn T = ln (ψ-δ δ-seq μ-fn T)

-- Strict concavity of ψ
postulate
  ψ-strictly-concave : ∀ {n} {E : ChainPresheaf n}
                     → (δ-seq : WeightSequence n)
                     → (μ-fn : MeasureFunction E)
                     → ConcavityHypothesis δ-seq
                     → is-strictly-concave (ψ δ-seq μ-fn)

--------------------------------------------------------------------------------
-- Extension to Rooted Trees

{-|
## Extension to Rooted Trees

The construction extends from chains to rooted inverse trees:
- Multiple initial vertices (inputs)
- Unique terminal vertex (output)
- Branches are chains from initial to terminal

**DNN Connection**:
DNN posets are obtained by gluing such trees on initial vertices:
- Maximal points: Input layers
- Minimal points: Output layers
- Forks: Convergent layers

The existence of concave precision function ψ extends to any DNN site.
-}

-- Rooted tree (multiple roots, single sink)
record RootedTree : Type₁ where
  field
    Vertex : Type
    _≤-tree_ : Vertex → Vertex → Type

    -- Poset structure
    ≤-refl-tree : ∀ {v} → v ≤-tree v
    ≤-trans-tree : ∀ {u v w} → u ≤-tree v → v ≤-tree w → u ≤-tree w
    ≤-antisym-tree : ∀ {u v} → u ≤-tree v → v ≤-tree u → u ≡ v

    -- Unique terminal vertex
    terminal : Vertex
    terminal-maximal : ∀ v → v ≤-tree terminal

    -- Multiple initial vertices
    is-initial : Vertex → Type
    has-initials : ∥ Σ Vertex is-initial ∥

-- DNN poset (gluing of rooted trees)
postulate
  DNN-Poset : Type₁

  -- Hypothesis on δ extends to branches
  branch-hypothesis : ∀ (dnn : DNN-Poset) → {!!}

  -- Existence of ψ for DNN sites
  ψ-exists-for-DNN : ∀ (dnn : DNN-Poset)
                   → {!!}  -- ∃ ψ: Ω_E → ℝ concave

--------------------------------------------------------------------------------
-- Examples

-- Example 1: 3-layer network (chain S_2)
module ThreeLayerExample where
  -- Chain 0 → 1 → 2 (input → hidden → output)
  E : ChainPresheaf 2
  E = {!!}

  -- Geometric weights δ_k = 1/2^k
  δ : WeightSequence 2
  δ = geometric-weights 2

  -- Counting measure
  μ : MeasureFunction E
  μ = counting-measure E

  -- Precision function
  precision : Subobject E → ℝ
  precision = ψ δ μ

  -- Example subobject: T_0 = {x₁, x₂}, T_1 = {y₁}, T_2 = {z₁}
  postulate
    example-subobject : Subobject E

  -- Compute precision
  example-precision : ℝ
  example-precision = precision example-subobject

-- Example 2: Binary tree (Y-shaped fork)
module BinaryTreeExample where
  postulate
    binary-tree : RootedTree

  -- Two branches from inputs to output
  -- ψ extends to this tree structure
  postulate
    tree-precision : {!!}  -- Subobject → ℝ

--------------------------------------------------------------------------------
-- Summary

{-|
## Summary

This module provides the mathematical foundation for **non-Boolean information
measures** in neural networks:

1. **Chain sites**: Model temporal/layered structure
2. **Injective presheaves**: Preserve information along chains
3. **Heyting algebra**: Internal logic with implication
4. **Precision function ψ**: Concave measure of information quality
5. **Extension to DNNs**: Generalizes to arbitrary DNN posets

**Key Properties**:
- ψ is strictly increasing (more information → higher precision)
- ψ is strictly concave (diminishing returns)
- ψ extends from chains to trees to DNN posets

**Applications**:
- Information flow analysis in DNNs
- Precision-aware optimization
- Multi-scale representation quality
-}
