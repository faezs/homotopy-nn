{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Random Graphs and Gamma Networks (Section 8.6)

This module implements graph information structures, random graphs, and the
connection between probability functors and Gamma networks.

## Overview

**Graph information structures** extend finite information structures to handle
network data by using functors M : S → G where G = Func(2, F) is the category
of directed graphs.

**Random graphs** are graphs GX where vertices and edges are random variables.

**Key Results**:
1. **Definition 8.5**: Graph information structures (S, M)
2. **Lemma 8.7**: Gamma networks EQ_C = ΓC ∘ Q from probability functors
3. **Proposition 8.8**: Joint distributions on Kronecker products
4. **Proposition 8.9**: Composition with Gamma-space increases II by Shannon entropy

## References

- Manin & Marcolli (2024), Section 8.6
- Bradley, "Information cohomology" [107]

-}

module Neural.Information.RandomGraphs where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Functor.Compose

open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin)

open import Neural.Base
open import Neural.Information using (ℝ; _+ℝ_; _*ℝ_; zeroℝ)
open import Neural.Information.Geometry
open import Neural.Information.Shannon
open import Neural.Homotopy.Simplicial
open import Neural.Homotopy.GammaSpaces
open import Neural.Homotopy.GammaNetworks using (Cat-2)

private variable
  o ℓ : Level

{-|
## Graph Category

The category **G** of directed graphs is Func(2, F) where:
- 2 is the category with two objects and one non-identity arrow
- F is the category of finite sets

Objects: Functors F : 2 → F (i.e., pairs of finite sets with s,t maps)
Morphisms: Natural transformations
-}

postulate
  -- Category of directed graphs
  GraphCat : Precategory (lsuc lzero) lzero

  GraphCat-as-functor :
    {-| GraphCat is isomorphic to Func(2, FinSets) -}
    Precategory.Ob GraphCat ≡ Functor Cat-2 (Sets lzero)

{-|
## Thin Categories and Random Variables (from §5.4.1)

A **thin category** S consists of:
- Objects: Random variables X
- Morphisms: "Coarsening" relations X → Y meaning Y factors through X

This captures the idea that Y contains less information than X.
-}

record ThinCategory : Type (lsuc lzero) where
  field
    -- Underlying category
    cat : Precategory lzero lzero

    -- At most one morphism between any two objects (thin)
    is-thin :
      (X Y : Precategory.Ob cat) →
      (f g : Precategory.Hom cat X Y) →
      f ≡ g

open ThinCategory public

{-|
## Graph Information Structures (Definition 8.5)

A **graph information structure** consists of:
1. A thin category S of random variables
2. A functor M : S → GraphCat assigning to each random variable a graph

For XE, XV ∈ Obj(S), the functor M assigns:
- MXE : Set of possible edge values
- MXV : Set of possible vertex values
- Source and target maps s,t : MXE → MXV

This determines a random graph GX.
-}

record GraphInformationStructure : Type (lsuc lzero) where
  field
    -- Thin category of random variables
    S : ThinCategory

    -- Functor to graphs
    M : Functor (ThinCategory.cat S) GraphCat

open GraphInformationStructure public

-- Extract the random graph from a pair of random variables
postulate
  RandomGraph :
    (GIS : GraphInformationStructure) →
    (XE XV : Precategory.Ob (ThinCategory.cat (S GIS))) →
    DirectedGraph

  RandomGraph-vertices :
    (GIS : GraphInformationStructure) →
    (XE XV : Precategory.Ob (ThinCategory.cat (S GIS))) →
    {-| Vertices are values of XV -}
    Nat

  RandomGraph-edges :
    (GIS : GraphInformationStructure) →
    (XE XV : Precategory.Ob (ThinCategory.cat (S GIS))) →
    {-| Edges are values of XE -}
    Nat

{-|
## Probability Functors on Graph Information Structures

A **probability functor** Q : (S×2, M) → ∆ assigns to each pair (XE, XV)
of random variables a simplicial set QXE,XV of probability distributions over
the vertex and edge sets MXE, MXV.

The simplicial sets have source and target maps respecting the graph structure.
-}

postulate
  -- Probability functor on graph information structures
  GraphProbabilityFunctor :
    GraphInformationStructure →
    Type (lsuc lzero)

  -- Evaluation at a pair of random variables gives simplicial set
  eval-graph-prob :
    {GIS : GraphInformationStructure} →
    GraphProbabilityFunctor GIS →
    (XE XV : Precategory.Ob (ThinCategory.cat (S GIS))) →
    PSSet

  -- The simplicial set is a space of probability distributions
  prob-simplex :
    {GIS : GraphInformationStructure} →
    (Q : GraphProbabilityFunctor GIS) →
    (XE XV : Precategory.Ob (ThinCategory.cat (S GIS))) →
    {-| eval-graph-prob Q XE XV is subsimplicial set of
        full simplex over MXE × MXV -}
    PSSet

{-|
## Joint Distributions and Kronecker Products (Proposition 8.8)

For a pair of independent random variables (X,Y), the corresponding graphs
satisfy:
  G(X,Y) = GX × GY  (Kronecker product)

For dependent (X,Y), the joint distribution determines a **subgraph**:
  G(X,Y) ⊂ GX × GY

**Kronecker product** of graphs G₁ × G₂:
- Vertices: V(G₁) × V(G₂)
- Edges: E(G₁) × E(G₂)
- Sources: s(e₁,e₂) = (s₁(e₁), s₂(e₂))
- Targets: t(e₁,e₂) = (t₁(e₁), t₂(e₂))
-}

-- Kronecker product of directed graphs
postulate
  _×G_ : DirectedGraph → DirectedGraph → DirectedGraph

  kronecker-vertices :
    (G₁ G₂ : DirectedGraph) →
    vertices (G₁ ×G G₂) ≡ vertices G₁ * vertices G₂

  kronecker-edges :
    (G₁ G₂ : DirectedGraph) →
    edges (G₁ ×G G₂) ≡ edges G₁ * edges G₂

  kronecker-source :
    (G₁ G₂ : DirectedGraph) →
    (e₁ : Fin (edges G₁)) →
    (e₂ : Fin (edges G₂)) →
    {-| source (G₁ ×G G₂) (e₁,e₂) = (source G₁ e₁, source G₂ e₂) -}
    ⊤  -- TODO: Need proper pairing of Fin indices

  kronecker-target :
    (G₁ G₂ : DirectedGraph) →
    (e₁ : Fin (edges G₁)) →
    (e₂ : Fin (edges G₂)) →
    {-| target (G₁ ×G G₂) (e₁,e₂) = (target G₁ e₁, target G₂ e₂) -}
    ⊤  -- TODO: Need proper pairing of Fin indices

-- Joint distribution determines subgraph
record JointGraphDistribution
  (GIS₁ GIS₂ : GraphInformationStructure)
  (X : Precategory.Ob (ThinCategory.cat (S GIS₁)))
  (Y : Precategory.Ob (ThinCategory.cat (S GIS₂))) : Type (lsuc lzero) where
  field
    -- The subgraph of the Kronecker product
    joint-graph : DirectedGraph

    -- Embedding into Kronecker product
    is-subgraph :
      {-| joint-graph embeds into
          RandomGraph GIS₁ X X ×G RandomGraph GIS₂ Y Y -}
      ⊤  -- TODO: Need proper subgraph relation

open JointGraphDistribution public

{-|
## Cohomological Integrated Information for Random Graphs (Proposition 8.8)

Given a probability functor Q : G(S,M)×(S',M') → ∆, we can compute the
cohomological integrated information IIH*(Q) measuring irreducibility.

**Key insight**: The simplicial set Q(X,Y) of probabilities on the joint
graph G(X,Y) contains information that cannot be reduced to independent
factors.
-}

postulate
  -- Cohomological II for graph probability functors
  graph-cohom-II :
    (GIS₁ GIS₂ : GraphInformationStructure) →
    (Q : GraphProbabilityFunctor GIS₁) →
    (X : Precategory.Ob (ThinCategory.cat (S GIS₁))) →
    (Y : Precategory.Ob (ThinCategory.cat (S GIS₂))) →
    {-| Cohomology H•(C•(M_α(Q, Q*_α), δ)) -}
    (n : Nat) →
    Type

  -- II measures non-reducibility
  graph-II-irreducibility :
    (GIS₁ GIS₂ : GraphInformationStructure) →
    (Q : GraphProbabilityFunctor GIS₁) →
    (X : Precategory.Ob (ThinCategory.cat (S GIS₁))) →
    (Y : Precategory.Ob (ThinCategory.cat (S GIS₂))) →
    {-| graph-cohom-II measures information in Q(X,Y) that is
        not reducible to independent subsystems -}
    ⊤  -- TODO: Formalize reducibility condition

{-|
## Gamma Networks from Probability Functors (Lemma 8.7)

Given:
- A category C of resources
- A Gamma-space ΓC : F* → PSSet
- A probability functor Q : G(S,M) → ∆*

We obtain a **Gamma network**:

  EQ_C = ΓC ∘ Q : G(S,M) → ∆*

This composes the simplicial set QGX with the Gamma-space to produce a new
simplicial set that incorporates resource structure.

**Physical interpretation**: EQ_C encodes both the probabilistic structure
of the random graph (via Q) and the resource distribution structure (via ΓC).
-}

postulate
  -- Pointed probability functor
  GraphProbabilityFunctor* :
    GraphInformationStructure →
    Type (lsuc lzero)

  -- Evaluation gives pointed simplicial sets
  eval-graph-prob* :
    {GIS : GraphInformationStructure} →
    GraphProbabilityFunctor* GIS →
    (XE XV : Precategory.Ob (ThinCategory.cat (S GIS))) →
    PSSet

-- Gamma network from composition (Lemma 8.7)
record GammaNetworkFromProb (GIS : GraphInformationStructure) : Type (lsuc lzero) where
  field
    -- Underlying probability functor
    Q : GraphProbabilityFunctor* GIS

    -- Gamma-space of resource category
    ΓC : GammaSpace

    -- Composed Gamma network
    EQ_C :
      (XE XV : Precategory.Ob (ThinCategory.cat (S GIS))) →
      PSSet

    -- Composition property
    composition :
      (XE XV : Precategory.Ob (ThinCategory.cat (S GIS))) →
      EQ_C XE XV ≡ eval-Gamma ΓC {-| applied to eval-graph-prob* Q XE XV -} zero  -- TODO

open GammaNetworkFromProb public

{-|
## Composition Increases II by Shannon Entropy (Proposition 8.9)

**Theorem (Proposition 8.9)**: For the Gamma network EQ_C = ΓC ∘ Q, the
Kullback-Leibler divergence satisfies:

  KL(P(X,Y) || Q*(X,Y)) = KL(P'(X,Y) || Q*(X,Y)) + S(P'')

where:
- P ∈ EQ_C(G(X,Y)) is a probability in the composed simplicial set
- P = P' · P'' where P' ∈ QG(X,Y) and P'' is a probability in ΓC([n])
- S(P'') is the Shannon entropy of P''

**Proof sketch**:
1. EQ_C is obtained as coend: QG(X,Y)_n ∧ ΓC([n])
2. Probabilities in the coend factor as P_{σ,τ} = P'_σ · P''_τ
3. Chain rule for KL divergence:
   KL(P_{σ,τ} || Q_{σ,τ}) = P''_τ · KL(P' || Q_τ) + S(P'')
4. By convexity: Σ_τ P''_τ KL(P' || Q_τ) ≥ KL(P' || Σ_τ P''_τ Q_τ)
5. Minimizer Q* of KL(P' || Q') also minimizes the weighted sum

**Physical interpretation**: Composing with a Gamma-space ΓC adds the Shannon
entropy S(P'') as a measure of the additional information from the resource
structure.

**Consequence**: Γ-networks INCREASE integrated information compared to plain
probability functors.
-}

-- Factorization of probabilities in composed Gamma network
record ProbabilityFactorization
  {GIS : GraphInformationStructure}
  (GN : GammaNetworkFromProb GIS)
  (XE XV : Precategory.Ob (ThinCategory.cat (S GIS)))
  (n : Nat) : Type (lsuc lzero) where
  field
    -- Probability in the graph simplicial set
    P-graph : {-| Probability in eval-graph-prob* (Q GN) XE XV -} ℝ

    -- Probability in the Gamma-space simplicial set
    P-gamma : {-| Probability in ΓC([n]) for some n -} ℝ

    -- Combined probability
    P-combined : {-| Probability in EQ_C GN XE XV -} ℝ

    -- Factorization property
    factors :
      {-| P-combined ≡ P-graph * P-gamma -}
      ⊤  -- TODO: Need proper probability product

open ProbabilityFactorization public

-- Chain rule for KL divergence (Proposition 8.9)
postulate
  KL-chain-rule-gamma :
    {GIS : GraphInformationStructure} →
    (GN : GammaNetworkFromProb GIS) →
    (XE XV : Precategory.Ob (ThinCategory.cat (S GIS))) →
    (n : Nat) →
    (factorization : ProbabilityFactorization GN XE XV n) →
    {-| KL divergence decomposes as:
        KL(P-combined || Q*) = KL(P-graph || Q*) + Shannon(P-gamma) -}
    ℝ

  -- Convexity of KL divergence
  KL-convexity :
    (p : ℝ) →
    (q₁ q₂ : ℝ) →
    {-| KL(p || λ·q₁ + (1-λ)·q₂) ≤ λ·KL(p || q₁) + (1-λ)·KL(p || q₂) -}
    ⊤  -- TODO: Need proper inequality type

  -- Composition increases integrated information
  gamma-increases-II :
    {GIS : GraphInformationStructure} →
    (GN : GammaNetworkFromProb GIS) →
    (XE XV : Precategory.Ob (ThinCategory.cat (S GIS))) →
    {-| II(EQ_C) ≥ II(Q) + Shannon(ΓC) in appropriate sense -}
    ⊤  -- TODO: Need proper II comparison

{-|
## Physical Interpretation

**Random graphs** model neural networks where connectivity is stochastic:
- Vertices = Neurons (with probabilities of being active)
- Edges = Synapses (with probabilities of existing/firing)

**Gamma networks** EQ_C = ΓC ∘ Q combine:
- Probabilistic graph structure (Q)
- Resource distribution structure (ΓC)

**Increased integrated information**: The composition theorem (Proposition 8.9)
shows that adding resource structure via Gamma-spaces INCREASES the integrated
information by the Shannon entropy of the Gamma-space.

**Consciousness implication**: Systems with richer resource structure (higher
dimensional Gamma-spaces) have higher integrated information, suggesting a
connection between resource complexity and consciousness.

**Network analysis**: Random graphs provide a bridge between:
- Classical random graph models (Erdős-Rényi, configuration models)
- Homotopy-theoretic neural models (Gamma-spaces, spectra)
- Information-theoretic measures (integrated information, Shannon entropy)
-}
