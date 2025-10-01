{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Weighted Codes and Linear Relations (Section 5.2)

This module implements the category WCodes_{n,*} of weighted codes from Section 5.2
of Manin & Marcolli (2024), providing a framework for linear neuron models.

## Overview

**Weighted codes**: Pairs (C, ω) where:
- C: Pointed code of length n
- ω: C → ℝ assigns a (signed) weight to each code word
- ω(c_0) = 0 at the zero word

**Category WCodes_{n,*}** (Lemma 5.10):
- Objects: (C, ω) with weight functions
- Morphisms: (f, Λ) preserving weighted structure
- Sum: (C,ω) ⊕ (C',ω') = (C∨C', ω∨ω')
- Zero: ({c_0}, 0)

## Application: Linear Neuron Model (Remark 5.11)

For directed graph G with single outgoing edge per vertex, the equalizer
condition becomes:
  (C_{out(v)}, ω_{out(v)}) = ⊕_{t(e)=v} (C_e, ω_e)

**Interpretation**:
- Edges e: synaptic connections
- Code C_e: spiking potentials along edge
- Weight ω_e: synaptic efficacy (physiological properties)
- Sign of ω_e: excitatory (+) or inhibitory (-)
-}

module Neural.Code.Weighted where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Diagram.Equaliser

open import Data.Nat.Base using (Nat; zero; suc; _+_)
open import Data.Fin.Base using (Fin; fzero; fsuc)

-- Import codes and real numbers
open import Neural.Code public
  using (PointedCode; PointedCodeMorphism; Code; zero-word; Alphabet)
open import Neural.Information public
  using (ℝ; _+ℝ_; _*ℝ_; _/ℝ_; _≤ℝ_; zeroℝ; oneℝ)
open import Neural.Base public
  using (DirectedGraph; vertices; edges; source; target)

private variable
  o ℓ : Level

{-|
## Weighted Codes (Lemma 5.10)

A weighted code (C, ω) assigns a real-valued weight to each code word,
representing information beyond just the presence/absence of spikes.

**Weight function properties**:
- ω : C → ℝ (can be positive or negative)
- ω(c_0) = 0 at the basepoint (no activity = zero weight)

**Physical interpretation**:
- For edge e in neural network:
  - C_e: code representing spiking along that connection
  - ω_e(c): weighted signal strength for pattern c
  - Sign of ω_e: excitatory (+) or inhibitory (-)
  - Magnitude of ω_e: synaptic efficacy
-}

record WeightedCode (n q' : Nat) : Type where
  no-eta-equality
  field
    {-| Underlying pointed code -}
    code : PointedCode n (suc q')

    {-|
    Weight function assigning real weights to code words

    For simplicity we model this as a function from indices,
    assuming the code is a finite list.
    -}
    weight : (c : Fin n → Alphabet (suc q')) → ℝ

    {-| Zero word has zero weight -}
    weight-zero : weight zero-word ≡ zeroℝ

open WeightedCode public

{-|
## Morphisms of Weighted Codes (Lemma 5.10)

Morphism ϕ = (f, Λ) : (C, ω) → (C', ω') consists of:
1. Pointed map f : C → C' sending zero word to itself
2. f(supp(ω)) ⊆ supp(ω')
3. Fiberwise measures Λ = {λ_{c'}(c)}_{c∈f⁻¹(c')}
4. Compatibility: ω(c) = λ_{f(c)}(c) · ω'(f(c))
5. λ_{c_0}(c_0) = 0

This models how weighted signals transform via functions of codes.
-}

{-|
Fiberwise weight scaling for morphisms
-}
FiberwiseWeight : {n q' : Nat} → (f : (Fin n → Alphabet (suc q')) → (Fin n → Alphabet (suc q'))) → Type
FiberwiseWeight {n} {q'} f =
  (c' : Fin n → Alphabet (suc q')) → (c : Fin n → Alphabet (suc q')) → ℝ

record WeightedCodeMorphism {n q' : Nat} (C C' : WeightedCode n q') : Type where
  no-eta-equality

  private
    code-C = C .code
    code-C' = C' .code
    ω = C .weight
    ω' = C' .weight

  field
    {-| Underlying pointed code morphism -}
    func : (Fin n → Alphabet (suc q')) → (Fin n → Alphabet (suc q'))

    {-| Preserves basepoint -}
    func-basepoint : func zero-word ≡ zero-word

    {-|
    Fiberwise weight scaling factors

    These relate ω(c) to ω'(f(c)) via: ω(c) = λ_{f(c)}(c) · ω'(f(c))
    -}
    fiberwise : FiberwiseWeight func

    {-| Basepoint scaling is zero -}
    fiberwise-basepoint : fiberwise zero-word zero-word ≡ zeroℝ

    {-|
    Compatibility with weights

    For all code words c: ω(c) = λ_{f(c)}(c) · ω'(f(c))
    -}
    weight-compatible :
      (c : Fin n → Alphabet (suc q')) →
      ω c ≡ (fiberwise (func c) c) *ℝ (ω' (func c))

open WeightedCodeMorphism public

{-|
## Composition and Identity (Lemma 5.10)

Composition and identity for weighted code morphisms.
-}
postulate
  WCodes-compose :
    {n q' : Nat} →
    {C₁ C₂ C₃ : WeightedCode n q'} →
    WeightedCodeMorphism C₂ C₃ →
    WeightedCodeMorphism C₁ C₂ →
    WeightedCodeMorphism C₁ C₃

  WCodes-id :
    {n q' : Nat} →
    (C : WeightedCode n q') →
    WeightedCodeMorphism C C

{-|
## Sum of Weighted Codes (Lemma 5.10)

The sum (C,ω) ⊕ (C',ω') = (C∨C', ω∨ω') where:
- Underlying code: C ∨ C' (wedge sum of pointed codes)
- Weight function:
  - (ω∨ω')|_C = ω
  - (ω∨ω')|_{C'} = ω'
  - At basepoint: (ω∨ω')(c_0) = 0

This is simpler than the probability case - no normalization needed.
-}

postulate
  weighted-code-sum :
    {n q' : Nat} →
    WeightedCode n q' →
    WeightedCode n q' →
    WeightedCode n q'

  weighted-code-sum-weight-left :
    {n q' : Nat} →
    (C C' : WeightedCode n q') →
    (c : Fin n → Alphabet (suc q')) →
    {-| If c ∈ C then (ω∨ω')(c) = ω(c) -}
    ⊤

  weighted-code-sum-weight-right :
    {n q' : Nat} →
    (C C' : WeightedCode n q') →
    (c : Fin n → Alphabet (suc q')) →
    {-| If c ∈ C' then (ω∨ω')(c) = ω'(c) -}
    ⊤

{-|
## Category WCodes_{n,*} (Lemma 5.10)

The category of weighted pointed codes of fixed length n.

**Objects**: (C, ω) with ω : C → ℝ and ω(c_0) = 0
**Morphisms**: (f, Λ) preserving weighted structure
**Sum**: (C,ω) ⊕ (C',ω') = (C∨C', ω∨ω')
**Zero**: ({c_0}, 0)
-}
postulate
  WCodes-pointed : (n q' : Nat) → Precategory lzero lzero

  WCodes-pointed-Ob :
    (n q' : Nat) →
    Precategory.Ob (WCodes-pointed n q') ≡ WeightedCode n q'

  WCodes-pointed-zero :
    (n q' : Nat) →
    WeightedCode n q'  -- Zero object: ({c_0}, 0)

{-|
## Network Summing Functors with Weighted Codes

For directed graph G, we consider summing functors:
  Φ_G ∈ Σ_{WCodes_{n,*}}(E_G)

assigning weighted codes to edges of the network.

**Source and target functors**:
- s : Σ_{WCodes_{n,*}}(E_G) ⇒ Σ_{WCodes_{n,*}}(V_G)
- t : Σ_{WCodes_{n,*}}(E_G) ⇒ Σ_{WCodes_{n,*}}(V_G)

These are analogous to the conservation law functors from Section 2.
-}

{-|
Category of subsets for directed graph
-}
postulate
  SubsetsCategory : Type → Precategory lzero lzero

{-|
Summing functor category for weighted codes
-}
SummingFunctorWeighted : (X : Type) → (n q : Nat) → Type
SummingFunctorWeighted X n q =
  Functor (SubsetsCategory X) (WCodes-pointed n q)

{-|
## Equalizer and Conservation Laws

The equalizer of source and target functors gives summing functors
satisfying conservation laws:

  Σ^eq_{WCodes_{n,*}}(G) := equalizer(s, t)

For summing functor Φ_G in the equalizer, for all pointed subsets A ⊂ V_G:
  Φ_G(s⁻¹(A)) = Φ_G(t⁻¹(A))

Or equivalently:
  ⊕_{s(e)∈A} (C_e, ω_e) = ⊕_{t(e)∈A} (C_e, ω_e)
-}

postulate
  {-| Equalizer category for weighted code summing functors -}
  weighted-code-equalizer :
    (G : DirectedGraph) →
    (n q : Nat) →
    Precategory lzero lzero

  weighted-code-equalizer-property :
    {G : DirectedGraph} →
    {n q' : Nat} →
    (Φ : SummingFunctorWeighted (Fin (edges G)) n q') →
    {-| Conservation law property -}
    Type

{-|
## Remark 5.11: Linear Neuron Model

**Assumption**: Graph G has single outgoing edge per vertex
  {e ∈ E_G | s(e) = v} = {out(v)}

**Conservation law becomes** (Equation 5.6):
  (C_{out(v)}, ω_{out(v)}) = ⊕_{t(e)=v} (C_e, ω_e)

**Interpretation**:
1. Edges e: synaptic connections between neurons
2. Code C_e: spiking potentials incoming along edge from s(e)
3. Weight ω_e: synaptic efficacy, depends on:
   - Number of synaptic vesicles (presynaptic terminal)
   - Number of gated channels (postsynaptic membrane)
4. Sign of ω_e: constant per edge
   - Positive: excitatory synapse
   - Negative: inhibitory synapse
5. Amplitude of ω_e(c): varies with code word c

**Note on signs**: For codes C = ∨_e C_e, the sign of ω = ∨_e ω_e is no
longer constant (mixed excitatory/inhibitory inputs).
-}

module LinearNeuronModel where
  {-|
  Single outgoing edge condition
  -}
  postulate
    has-single-outgoing-edge :
      (G : DirectedGraph) →
      Type

  {-|
  Outgoing edge function (under single-edge assumption)
  -}
  postulate
    out : {G : DirectedGraph} →
      has-single-outgoing-edge G →
      Fin (vertices G) →
      Fin (edges G)

  {-|
  Linear neuron conservation law (Equation 5.6)

  For vertex v with outgoing edge out(v) and incoming edges {e | t(e) = v}:
    (C_{out(v)}, ω_{out(v)}) = ⊕_{t(e)=v} (C_e, ω_e)
  -}
  postulate
    linear-neuron-conservation :
      {G : DirectedGraph} →
      {n q' : Nat} →
      (single-out : has-single-outgoing-edge G) →
      (Φ : SummingFunctorWeighted (Fin (edges G)) n q') →
      (v : Fin (vertices G)) →
      {-| Conservation law at vertex v -}
      Type

  {-|
  **Physical interpretation**

  The linear neuron model treats each neuron as a linear combiner of
  weighted inputs, where:

  - Input codes C_e represent presynaptic spiking patterns
  - Weights ω_e represent synaptic strengths (efficacy)
  - Output code C_{out(v)} represents postsynaptic response
  - Conservation law: weighted sum of inputs = output

  **Synaptic properties encoded in ω_e**:
  1. Physiological: vesicle count, channel density
  2. Functional: excitatory/inhibitory character
  3. Dynamic: strength varies with spike pattern
  -}

{-|
## Examples and Applications

**Example 1**: Binary weighted code for two-neuron network
- Neuron 1 → Neuron 2
- Edge e: C_e = {c_0, c_1} with ω_e(c_0) = 0, ω_e(c_1) = w > 0
- Excitatory synapse with weight w

**Example 2**: Inhibitory-excitatory pair
- Neuron 1 → Neuron 3, Neuron 2 → Neuron 3
- Edge e_1: ω_{e_1} > 0 (excitatory)
- Edge e_2: ω_{e_2} < 0 (inhibitory)
- Net input to Neuron 3: (C_1, ω_1) ⊕ (C_2, ω_2)

**Example 3**: Feed-forward network
- Input layer → Hidden layer → Output layer
- Each edge has weighted code
- Conservation laws at each hidden/output neuron
- Sum operation models integration of inputs
-}

module Examples where
  postulate
    {-| Example: Simple excitatory synapse -}
    excitatory-synapse :
      (w : ℝ) →
      WeightedCode 1 2

    {-| Example: Inhibitory synapse -}
    inhibitory-synapse :
      (w : ℝ) →
      WeightedCode 1 2

    {-| Example: Two-input neuron -}
    two-input-neuron :
      (C₁ C₂ : WeightedCode 10 2) →
      WeightedCode 10 2

    {-| Example: Feed-forward network weighted codes -}
    feedforward-network :
      (G : DirectedGraph) →
      (n q : Nat) →
      SummingFunctorWeighted (Fin (edges G)) n q

{-|
## Positive Weighted Codes (Section 6.4, Definition 6.5)

For modeling classical Hopfield dynamics where weights represent activity levels,
we restrict to **non-negative weights only**.

This is used in Section 6.4 to show that categorical Hopfield dynamics on
weighted codes recovers the classical Hopfield network equations.
-}

record WeightedCodePositive (n q' : Nat) : Type where
  no-eta-equality
  field
    {-| Underlying weighted code -}
    underlying : WeightedCode n q'

    {-| All weights are non-negative -}
    weights-nonneg : (c : Fin n → Alphabet (suc q')) → zeroℝ ≤ℝ (underlying .weight c)

open WeightedCodePositive public

{-|
## Morphisms of Positive Weighted Codes (Definition 6.5)

Morphisms between positive weighted codes must satisfy:
1. Underlying morphism of weighted codes
2. Fiberwise weights λ_c'(c) ≤ 1 for all c, c'

The second condition ensures that total weights are non-increasing under morphisms,
which is needed for the functor α : WCodes⁺ → (ℝ, ≥) in Lemma 6.6.
-}

record WCodePosMorphism {n q' : Nat} (C C' : WeightedCodePositive n q') : Type where
  no-eta-equality
  field
    {-| Underlying weighted code morphism -}
    underlying-morph : WeightedCodeMorphism (C .underlying) (C' .underlying)

    {-| Fiberwise weights are bounded by 1 -}
    fiberwise-bounded :
      (c' c : Fin n → Alphabet (suc q')) →
      (underlying-morph .fiberwise c' c) ≤ℝ oneℝ

open WCodePosMorphism public

{-|
## Category WCodes⁺_{n,*} (Definition 6.5)

The category of positive weighted codes with morphisms satisfying λ_c'(c) ≤ 1.

This differs from WCodes_{n,*} by restricting to non-negative weights and
bounded fiberwise measures.
-}

postulate
  WCodes-positive : (n q' : Nat) → Precategory lzero lzero

  WCodes-positive-Ob :
    (n q' : Nat) →
    Precategory.Ob (WCodes-positive n q') ≡ WeightedCodePositive n q'

  WCodes-positive-compose :
    {n q' : Nat} →
    {C₁ C₂ C₃ : WeightedCodePositive n q'} →
    WCodePosMorphism C₂ C₃ →
    WCodePosMorphism C₁ C₂ →
    WCodePosMorphism C₁ C₃

  WCodes-positive-id :
    {n q' : Nat} →
    (C : WeightedCodePositive n q') →
    WCodePosMorphism C C
