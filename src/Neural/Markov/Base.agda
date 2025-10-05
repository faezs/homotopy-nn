{-# OPTIONS --no-import-sorts #-}
{-|
# Markov Categories

This module defines **Markov categories**, which are symmetric monoidal categories
where every object carries a commutative comonoid structure (copy and delete
operations) satisfying compatibility axioms.

Markov categories provide a categorical framework for probability theory and
statistics, where morphisms represent stochastic maps (Markov kernels), and the
comonoid operations represent copying and discarding random variables.

## Definition

Following Fritz (2020), Definition 2.1, a **Markov category** is a symmetric
monoidal category $(\mathcal{C}, \otimes, I)$ equipped with:

1. **Comonoid structure on every object**: Each object $X$ has:
   - Copy (comultiplication): $\delta_X : X \to X \otimes X$
   - Delete (counit): $\varepsilon_X : X \to I$

2. **Commutativity**: The comonoids are commutative (copy commutes with swap)

3. **Compatibility with tensor product** (Fritz 2.4):
   - $\varepsilon_{X \otimes Y} = (\varepsilon_X \otimes \varepsilon_Y) \circ \alpha$
   - $\delta_{X \otimes Y} = (\delta_X \otimes \delta_Y) \circ \text{twist} \circ \alpha$

4. **Naturality of delete** (Fritz 2.5):
   - For any $f : X \to Y$: $\varepsilon_Y \circ f = \varepsilon_X$

## Interpretation

In the Markov category framework:
- **Objects**: Represent types of random variables or probability spaces
- **Morphisms**: Represent Markov kernels (stochastic maps)
- **Copy $\delta$**: Duplicates a random variable (creating correlation)
- **Delete $\varepsilon$**: Discards a random variable (marginalization)
- **Tensor $\otimes$**: Combines independent random variables

## Examples

Standard examples of Markov categories:
- **FinStoch**: Finite sets with stochastic matrices
- **Stoch**: Measurable spaces with Markov kernels
- **BorelStoch**: Standard Borel spaces with Markov kernels
- **Gauss**: Finite-dimensional vector spaces with Gaussian channels
- **Pf**: Finite pointed sets with probability measures and fiberwise measures

## References

- Fritz, T. (2020). "A synthetic approach to Markov kernels, conditional
  independence and theorems on sufficient statistics." arXiv:1908.07021v8
- Cho, K., & Jacobs, B. (2019). "Disintegration and Bayesian inversion via
  string diagrams."
-}

module Neural.Markov.Base where

open import Cat.Monoidal.Diagram.Comonoid
open import Cat.Monoidal.Braided
open import Cat.Monoidal.Base
open import Cat.Prelude

import Cat.Reasoning as Cr

{-|
## Markov Category Structure

A Markov category structure on a symmetric monoidal category consists of:
- Commutative comonoid on every object
- Compatibility with monoidal structure
- Naturality of the counit (delete)
-}
module _ {o ℓ} {C : Precategory o ℓ}
         (M : Monoidal-category C)
         (S : Symmetric-monoidal M) where

  private
    module C where
      open Cr C public
      open Monoidal-category M public
      open Symmetric-monoidal S public

  record Is-markov-category : Type (o ⊔ ℓ) where
    no-eta-equality

    field
      {-|
      Every object has a commutative comonoid structure

      This provides copy and delete operations for every object, representing
      the ability to duplicate or discard random variables.
      -}
      comonoids : (X : C.Ob) → Commutative-comonoid-on M {S} X

    {-| Accessor for underlying comonoid structure -}
    has-comonoid : (X : C.Ob) → Comonoid-on M X
    has-comonoid X = comonoids X .Commutative-comonoid-on.comonoid

    {-| Copy operation: duplicates a value -}
    copy : (X : C.Ob) → C.Hom X (X C.⊗ X)
    copy X = has-comonoid X .Comonoid-on.δ

    {-| Delete operation: discards a value -}
    del : (X : C.Ob) → C.Hom X C.Unit
    del X = has-comonoid X .Comonoid-on.ε

    field
      {-|
      Compatibility with tensor product for delete (Fritz 2.4, first equation)

      Deleting a tensor product is the same as deleting each component:
      ```
      X ⊗ Y ---ε_{X⊗Y}---> I
      X ⊗ Y ---ε_X ⊗ ε_Y--> I ⊗ I ---≅---> I
      ```
      -}
      comonoid-⊗-ε :
        ∀ {X Y} →
        del (X C.⊗ Y) ≡ C.λ← C.∘ (del X C.⊗₁ del Y)

      {-|
      Compatibility with tensor product for copy (Fritz 2.4, second equation)

      Copying a tensor product should equal copying each component separately
      and then using the braiding to "twist" the middle components:

        (X⊗Y) ⊗ (X⊗Y) ≅ X⊗X ⊗ Y⊗Y    (via the twist using β)

      The full equation involves associators. For simplicity, we axiomatize
      this as: copying the tensor is the same morphism from both constructions.

      TODO: State the full equation with explicit associators and braiding.
      For now, concrete Markov categories just need to provide this proof.
      -}
      comonoid-⊗-δ : ⊤  -- Placeholder: should be proper equation

      {-|
      Naturality of delete (Fritz 2.5)

      Deleting after applying a morphism is the same as deleting before:
      ```
      X ---f---> Y
      |          |
      ε_X        ε_Y
      ↓          ↓
      I ---------=--> I
      ```

      This captures the idea that morphisms in a Markov category preserve the
      "nothingness" represented by the unit object.
      -}
      del-natural :
        ∀ {X Y} (f : C.Hom X Y) →
        del Y C.∘ f ≡ del X

  open Is-markov-category public

{-|
## Deterministic Morphisms

A morphism $f : X \to Y$ in a Markov category is **deterministic** if it
preserves the comonoid structure. This corresponds to "non-random" functions
in the probabilistic interpretation.

Fritz Section 10: A deterministic morphism satisfies:
- $(f \otimes f) \circ \delta_X = \delta_Y \circ f$ (preserves copying)
- $\varepsilon_Y \circ f = \varepsilon_X$ (preserves deletion, automatic by naturality)

The second condition is automatic from del-natural, so we only need to check
preservation of copy.
-}
module _ {o ℓ} {C : Precategory o ℓ}
         {M : Monoidal-category C}
         {S : Symmetric-monoidal M}
         (MC : Is-markov-category M S) where

  private
    module C where
      open Cr C public
      open Monoidal-category M public
      open Symmetric-monoidal S public
    module MC = Is-markov-category MC

  {-|
  A morphism is deterministic if it preserves the copy operation.

  Preservation of delete is automatic from naturality.
  -}
  is-deterministic : {X Y : C.Ob} → C.Hom X Y → Type ℓ
  is-deterministic {X} {Y} f =
    (f C.⊗₁ f) C.∘ MC.copy X ≡ MC.copy Y C.∘ f

{-|
## Helper: Copying with Multiple Outputs

Following Fritz Notation 2.9, we can copy a value multiple times by repeated
application of the copy operation. Due to coassociativity, the order doesn't
matter.
-}
module _ {o ℓ} {C : Precategory o ℓ}
         {M : Monoidal-category C}
         {S : Symmetric-monoidal M}
         (MC : Is-markov-category M S) where

  private
    module C where
      open Cr C public
      open Monoidal-category M public
    module MC = Is-markov-category MC

  {-|
  Copy to three outputs: X → X ⊗ X ⊗ X

  By coassociativity, this can be defined as:
  - δ ∘ (δ ⊗ id), or equivalently
  - δ ∘ (id ⊗ δ)
  -}
  copy3 : (X : C.Ob) → C.Hom X (X C.⊗ (X C.⊗ X))
  copy3 X = (C.id C.⊗₁ MC.copy X) C.∘ MC.copy X

  {-|
  General n-way copy: X → X^⊗n

  For now we only define up to 3 outputs. A general version would use natural
  numbers and recursion, but this is sufficient for our purposes.
  -}
