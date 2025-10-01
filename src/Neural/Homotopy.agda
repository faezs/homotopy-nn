{-# OPTIONS --no-import-sorts --allow-unsolved-metas #-}
{-|
# Homotopy-Theoretic Models of Neural Networks (Section 7)

This module re-exports all homotopy-theoretic structures for neural networks
from Section 7 of Manin & Marcolli (2024).

## Organization

### §7.1: Neural.Homotopy.Simplicial
- Pointed simplicial sets PSSet
- Smash product ∧s
- Suspension Susp and spheres Sⁿ
- Weak homotopy equivalences
- n-connectivity
- Homotopy groups πn

### §7.2: Neural.Homotopy.GammaSpaces
- Segal's Gamma-spaces Γ : F* → PSSet
- Special Gamma-spaces
- Connective spectra
- Infinite loop spaces
- Stable homotopy category

### §7.3: Neural.Homotopy.CliqueComplex
- Clique complexes K(G) from graphs
- Connectivity of K(G)
- Nerve construction N(ΣC(X))
- Betti numbers and Euler characteristic

### §7.4: Neural.Homotopy.GammaNetworks
- Gamma networks E : Func(2, F*) → PSSet
- Composition and operadic structure
- Stable homotopy invariants
- Applications to deep learning

## Key Concepts

**Gamma-spaces** provide homotopy-theoretic models for resource distribution
in neural networks, generalizing symmetric monoidal categories to stable
homotopy theory.

**Clique complexes** encode network topology as simplicial complexes, enabling
topological analysis of neural connectivity.

**Gamma networks** generalize Gamma-spaces to handle networks with multiple
inputs and outputs, modeling deep learning architectures.

## References

Manin, Y. I., & Marcolli, M. (2024). Homotopy-theoretic and categorical
models of neural information networks. *arXiv preprint* arXiv:2406.01228,
Section 7.
-}

module Neural.Homotopy where

-- §7.1: Foundational simplicial homotopy theory
open import Neural.Homotopy.Simplicial public
  using ( -- Pointed sets
          Sets*
        ; U*
          -- Pointed simplicial sets
        ; PSSet
        ; PSSet-cat
        ; PSSet-Hom
          -- Smash product
        ; _∧s_
        ; smash-assoc
        ; smash-point
          -- Suspension and spheres
        ; S¹
        ; Susp
        ; Susp-as-smash
        ; Suspⁿ
        ; Sⁿ
        ; Sⁿ-suc
          -- Weak homotopy equivalences
        ; is-weak-equiv
        ; preserves-weak-equiv
          -- Connectivity
        ; is-n-connected
        ; suspension-connectivity
        ; sphere-connectivity
          -- Homotopy groups
        ; π
        ; πn-group
        ; πn-abelian
          -- Geometric realization
        ; geometric-realization
        ; weak-equiv-realizes
          -- Skeletal filtration
        ; Skₙ
        ; skeleton-filtration
        )

-- §7.2: Segal's Gamma-spaces
open import Neural.Homotopy.GammaSpaces public
  using ( -- Pointed finite sets category
          F*
        ; F*-Hom
        ; _∨F*_
          -- Gamma-spaces
        ; GammaSpace
        ; eval-Gamma
        ; is-special
          -- Connective spectra
        ; ConnectiveSpectrum
        ; spectrum-from-gamma
        ; Segal-equivalence
          -- Infinite loop spaces
        ; is-infinite-loop-space
        ; gamma-infinite-loop
        ; LoopSpace
        ; LoopSpaceⁿ
          -- Very special Gamma-spaces
        ; is-very-special
        ; is-grouplike-E∞
          -- Stable homotopy category
        ; StableHomotopyCategory
        ; gamma-to-SHC
        )

-- §7.3: Clique complexes and network topology
open import Neural.Homotopy.CliqueComplex public
  using ( -- Clique complex
          K
        ; K-connectivity
          -- Nerve construction
        ; Nerve
        ; N-SummingFunctor
          -- Homotopy groups
        ; pi-K
          -- Betti numbers
        ; betti
        ; euler-char
          -- Vietoris-Rips complexes
        ; VR
        ; persistent-homology
        )

-- §7.4: Gamma networks
open import Neural.Homotopy.GammaNetworks public
  using ( -- Functor category
          Cat-2
        ; Func-2-F*
          -- Gamma networks
        ; GammaNetwork
        ; eval-GammaNetwork
        ; is-special-network
          -- Relationship to Gamma-spaces
        ; GammaSpace-to-Network
        ; GammaNetwork-to-Space
          -- Composition
        ; compose-networks
        ; identity-network
          -- Stable homotopy invariants
        ; stable-pi
        ; stability-theorem
        ; stable-equiv
          -- Applications
        ; network-from-summing
        ; deep-network
        )
