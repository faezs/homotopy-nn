{-# OPTIONS --no-import-sorts #-}
{-|
# Comonoids in a Monoidal Category

This module defines comonoid objects in a monoidal category, which are the
categorical dual of monoid objects.

A **comonoid in a monoidal category** $(\mathcal{C}, \otimes, 1)$ consists of:
- An object $M$
- A comultiplication (copy) morphism $\delta : M \to M \otimes M$
- A counit (delete) morphism $\varepsilon : M \to 1$

These must satisfy coassociativity and counit laws, which are the duals of
the associativity and unit laws for monoids.

## Motivation

Comonoids are essential for Markov categories (Fritz 2020), where every object
carries a comonoid structure representing copying and discarding of random
variables. The copy operation $\delta$ duplicates a value, while the counit
$\varepsilon$ discards it.

## References

- Fritz, T. (2020). "A synthetic approach to Markov kernels, conditional
  independence and theorems on sufficient statistics." arXiv:1908.07021v8
- Cho, K., & Jacobs, B. (2019). "Disintegration and Bayesian inversion via
  string diagrams." Mathematical Structures in Computer Science, 29(7).
-}

module Cat.Monoidal.Diagram.Comonoid where

open import Cat.Monoidal.Braided
open import Cat.Monoidal.Base
open import Cat.Prelude

import Cat.Reasoning as Cr

{-|
## Comonoid Objects

A comonoid object in a monoidal category $(\mathcal{C}, \otimes, 1)$ is
an object $M$ equipped with:
- Comultiplication $\delta : M \to M \otimes M$ (copy)
- Counit $\varepsilon : M \to 1$ (delete)

These satisfy coassociativity and counit laws dual to monoid axioms.
-}
module _ {o ℓ} {C : Precategory o ℓ} (M : Monoidal-category C) where
  private module C where
    open Cr C public
    open Monoidal-category M public

  record Comonoid-on (X : C.Ob) : Type ℓ where
    no-eta-equality
    field
      {-| Counit (delete): discards the value -}
      ε : C.Hom X C.Unit

      {-| Comultiplication (copy): duplicates the value -}
      δ : C.Hom X (X C.⊗ X)

      {-|
      Left counit law: deleting the left copy is the right unitor

      Diagram:
      ```
      X ---δ---> X ⊗ X
      |          |
      |        ε⊗id
      |          ↓
      |      Unit ⊗ X
      |          |
      |          λ→
      └----------→
      ```
      -}
      δ-counitl : (ε C.⊗₁ C.id) C.∘ δ ≡ C.λ→

      {-|
      Right counit law: deleting the right copy is the left unitor

      Diagram:
      ```
      X ---δ---> X ⊗ X
      |          |
      |        id⊗ε
      |          ↓
      |      X ⊗ Unit
      |          |
      |          ρ→
      └----------→
      ```
      -}
      δ-counitr : (C.id C.⊗₁ ε) C.∘ δ ≡ C.ρ→

      {-|
      Coassociativity: copying twice is associative

      Diagram:
      ```
      X ---δ---> X ⊗ X --id⊗δ--> X ⊗ (X ⊗ X)
      |                           |
      |                           |
      δ                           ≡
      |                           |
      ↓                           ↓
      X ⊗ X --δ⊗id--> (X ⊗ X) ⊗ X --α→--> X ⊗ (X ⊗ X)
      ```
      -}
      δ-coassoc : (C.id C.⊗₁ δ) C.∘ δ ≡ C.α→ X X X C.∘ (δ C.⊗₁ C.id) C.∘ δ

  open Comonoid-on public

  {-|
  ## Commutative Comonoids

  A comonoid is **commutative** if copying commutes with swapping.
  In a symmetric monoidal category, this means the braiding $\beta$ makes
  copying order-independent.
  -}
  module _ {Cˢ : Symmetric-monoidal M} where
    private module Cˢ = Symmetric-monoidal Cˢ

    record Commutative-comonoid-on (X : C.Ob) : Type ℓ where
      no-eta-equality

      field
        comonoid : Comonoid-on X

      open Comonoid-on comonoid public

      field
        {-|
        Commutativity: copying commutes with swapping

        Diagram:
        ```
        X ---δ---> X ⊗ X
        |          |
        |          β→
        |          ↓
        δ      X ⊗ X
        |
        ↓
        X ⊗ X -----≡
        ```

        In string diagram notation: copying in either order gives the same result.
        -}
        δ-commutative : Cˢ.β→ C.∘ comonoid .δ ≡ comonoid .δ

  open Commutative-comonoid-on public

  {-|
  ## Comonoid Morphisms

  A morphism $f : X \to Y$ between comonoids is a **comonoid morphism** if
  it preserves the comonoid structure.
  -}
  record is-comonoid-morphism
    {X Y : C.Ob}
    (CX : Comonoid-on X)
    (CY : Comonoid-on Y)
    (f : C.Hom X Y) : Type ℓ where
    no-eta-equality

    private
      module CX = Comonoid-on CX
      module CY = Comonoid-on CY

    field
      {-|
      Preserves counit: $\varepsilon_Y \circ f = \varepsilon_X$

      Deleting after applying $f$ is the same as deleting before.
      -}
      preserves-ε : CY.ε C.∘ f ≡ CX.ε

      {-|
      Preserves comultiplication: $(f \otimes f) \circ \delta_X = \delta_Y \circ f$

      Diagram:
      ```
      X ---f---> Y
      |          |
      δX         δY
      ↓          ↓
      X⊗X -f⊗f-> Y⊗Y
      ```

      Copying after applying $f$ is the same as applying $f$ then copying.
      -}
      preserves-δ : (f C.⊗₁ f) C.∘ CX.δ ≡ CY.δ C.∘ f

  open is-comonoid-morphism public

{-|
## Properties

### Identity is a Comonoid Morphism
-}
module _ {o ℓ} {C : Precategory o ℓ} {M : Monoidal-category C} where
  private module C where
    open Cr C public
    open Monoidal-category M public

  id-comonoid-morphism :
    {X : C.Ob}
    (CX : Comonoid-on M X) →
    is-comonoid-morphism M CX CX C.id
  id-comonoid-morphism CX .preserves-ε =
    CX.ε C.∘ C.id ≡⟨ C.idr _ ⟩
    CX.ε          ∎
    where module CX = Comonoid-on CX
  id-comonoid-morphism CX .preserves-δ =
    (C.id C.⊗₁ C.id) C.∘ CX.δ ≡⟨ ap (C._∘ CX.δ) C.⊗.F-id ⟩
    C.id C.∘ CX.δ              ≡⟨ C.idl _ ⟩
    CX.δ                       ≡˘⟨ C.idr _ ⟩
    CX.δ C.∘ C.id              ∎
    where module CX = Comonoid-on CX

{-|
### Composition of Comonoid Morphisms

TODO: Complete this proof. The statement is correct but the proof needs careful
handling of associativity.
-}
  postulate
    compose-comonoid-morphism :
      {X Y Z : C.Ob}
      {CX : Comonoid-on M X}
      {CY : Comonoid-on M Y}
      {CZ : Comonoid-on M Z}
      {f : C.Hom X Y} {g : C.Hom Y Z} →
      is-comonoid-morphism M CX CY f →
      is-comonoid-morphism M CY CZ g →
      is-comonoid-morphism M CX CZ (g C.∘ f)
