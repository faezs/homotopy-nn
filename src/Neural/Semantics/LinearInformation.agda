{-# OPTIONS --rewriting --guardedness --cubical --allow-unsolved-metas #-}

{-|
# Appendix E.7: Linear Information Theory

This module implements **theories** as comonads on the **bar-complex**, connecting:
- Linear logic with information theory
- Kolmogorov complexity with compression
- Tri-simplicial structure for compositional semantics

## Key Concepts

1. **Theory T**: Comonad structure representing "possible models" or "theories"
   - Objects: Propositions/types
   - T(A): Set of theories about A
   - ε: T(A) → A (extract/evaluate)
   - δ: T(A) → T(T(A)) (duplicate/generate sub-theories)

2. **Bar-Complex Bar(T)**: Simplicial object encoding compositional structure
   - Bar₀(A) = A
   - Bar₁(A) = T(A)
   - Bar₂(A) = T(T(A))
   - Face maps: ε (counit) and δ (comultiplication)

3. **Tri-Simplicial Sets**: Three simplicial directions
   - Horizontal: Tensor products (A ⊗ B)
   - Vertical: Theories (T^n(A))
   - Depth: Semantic composition

4. **Compression Ratio F/K**:
   - F: Shannon entropy (statistical information)
   - K: Kolmogorov complexity (algorithmic information)
   - Connection to resource theory from Section 3.2

5. **Information Comonad**: Theories as compression schemes
   - T(A) = compressed representations of A
   - ε: decompression
   - δ: iterative compression

## References

- [BB22] Belfiore & Bennequin (2022), Appendix E
- [Gir87] Girard (1987): Linear logic
- [Kol65] Kolmogorov (1965): Three approaches to information
- [Bar10] Barendregt & Barendsen (2010): Typed lambda calculi

-}

module Neural.Semantics.LinearInformation where

open import 1Lab.Prelude hiding (id; _∘_)
open import 1Lab.Path
open import 1Lab.HLevel
open import 1Lab.Equiv

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Monoidal.Base
open import Cat.Diagram.Limit.Base

open import Neural.Semantics.ClosedMonoidal
open import Neural.Semantics.LinearExponential
open import Neural.Semantics.TensorialNegation

private variable
  o ℓ o' ℓ' : Level

--------------------------------------------------------------------------------
-- Theories as Comonads

{-|
## Definition: Theory

A **theory** T is a comonad on a closed monoidal category C, interpreted as:
- T(A): Set of "possible theories" or "compressed descriptions" of A
- ε: T(A) → A: Evaluation/decompression
- δ: T(A) → T(T(A)): Sub-theory generation/iterative compression

**Intuition**:
- In logic: T(A) = theorems provable about A
- In computation: T(A) = programs computing A
- In information: T(A) = compressed encodings of A
-}

record Theory {o ℓ} (C : Precategory o ℓ)
              (M : Monoidal-category C) : Type (o ⊔ ℓ) where
  open Precategory C
  open Monoidal-category M

  field
    T : Ob → Ob
    T₁ : ∀ {A B} → Hom A B → Hom (T A) (T B)

    -- Comonad structure
    ε : ∀ {A} → Hom (T A) A
    δ : ∀ {A} → Hom (T A) (T (T A))

    -- Comonad laws
    ε-natural : ∀ {A B} (f : Hom A B) → f ∘ ε ≡ ε ∘ T₁ f
    δ-natural : ∀ {A B} (f : Hom A B) → T₁ (T₁ f) ∘ δ ≡ δ ∘ T₁ f

    ε-δ : ∀ {A} → ε ∘ δ ≡ id
    Tε-δ : ∀ {A} → T₁ ε ∘ δ ≡ id
    δ-coassoc : ∀ {A} → T₁ δ ∘ δ ≡ δ ∘ δ

  -- Kleisli category for theories (coalgebras)
  -- Objects: same as C
  -- Morphisms A → B: T(A) → B (coalgebraic maps)

--------------------------------------------------------------------------------
-- Bar-Complex

{-|
## Definition: Bar-Complex Bar(T)

The **bar-complex** is a simplicial object encoding the compositional structure
of a theory T:

```
Bar₀(A) ←ε← Bar₁(A) ←d₀,d₁← Bar₂(A) ←...
  A     ←ε←   T(A)   ←δ,ε←  T(T(A)) ←...
```

**Face maps**:
- d₀: T^(n+1)(A) → T^n(A) via ε (counit)
- d₁: T^(n+1)(A) → T^n(A) via T^n(ε) (apply ε at level n)
- d₂: T^(n+1)(A) → T^n(A) via T^(n-1)(ε) (apply ε at level n-1)

**Degeneracy maps**:
- s₀: T^n(A) → T^(n+1)(A) via δ (comultiplication)

**Interpretation**: The bar-complex encodes:
- Level 0: Raw data A
- Level 1: One-step theories T(A)
- Level 2: Meta-theories T(T(A))
- Level n: n-fold iterated theories
-}

module BarComplex {o ℓ} {C : Precategory o ℓ}
                  {M : Monoidal-category C}
                  (Th : Theory C M) where
  open Precategory C
  open Theory Th

  -- Bar-complex at level n
  Bar : (n : Nat) → Ob → Ob
  Bar zero A = A
  Bar (suc n) A = T (Bar n A)

  -- Face map d₀: Apply counit at top level
  d₀ : ∀ {n A} → Hom (Bar (suc n) A) (Bar n A)
  d₀ {n} = ε

  -- Face map d₁: Apply counit one level down
  d₁ : ∀ {n A} → Hom (Bar (suc (suc n)) A) (Bar (suc n) A)
  d₁ {n} = T₁ ε

  -- Degeneracy map s₀: Duplicate via comultiplication
  postulate
    unit-map : ∀ {A} → Hom A (T A)  -- Not part of comonad, needs additional structure

  s₀ : ∀ {n A} → Hom (Bar n A) (Bar (suc n) A)
  s₀ {zero} {A} = unit-map  -- From A to T(A)
  s₀ {suc n} = δ

  -- Simplicial identities
  postulate
    d₀-s₀ : ∀ {n A} → d₀ {n} {A} ∘ s₀ ≡ id
    d₁-s₀ : ∀ {n A} → d₁ {n} {A} ∘ s₀ ≡ id

--------------------------------------------------------------------------------
-- Tri-Simplicial Structure

{-|
## Tri-Simplicial Sets

The bar-complex extends to a **tri-simplicial set** with three directions:

1. **Horizontal (⊗)**: Tensor products for parallel composition
   - Bar_n(A ⊗ B) relates to Bar_n(A) ⊗ Bar_n(B)

2. **Vertical (T)**: Theory iteration for meta-levels
   - Bar_n(A) = T^n(A)

3. **Depth**: Semantic nesting for compositional structure
   - Contexts and environments

**Lax monoidal structure**: T(A ⊗ B) relates to T(A) ⊗ T(B)
-}

module TriSimplicial {o ℓ} {C : Precategory o ℓ}
                     {M : Monoidal-category C}
                     (Th : Theory C M) where
  open Precategory C
  open Monoidal-category M
  open Theory Th
  open BarComplex Th

  -- Lax monoidal structure: T(A ⊗ B) → T(A) ⊗ T(B)
  postulate
    lax-mult : ∀ {A B} → Hom (T (A ⊗ B)) (T A ⊗ T B)
    lax-unit : Hom (T Unit) Unit

  -- Horizontal composition: Bar_n(A) ⊗ Bar_n(B) → Bar_n(A ⊗ B)
  horizontal-comp : ∀ {n A B} → Hom (Bar n A ⊗ Bar n B) (Bar n (A ⊗ B))
  horizontal-comp {zero} = id
  horizontal-comp {suc n} = {!!}  -- Inductively using lax-mult

  -- Coherence for tri-simplicial structure
  postulate
    tri-coherence : ∀ {n A B} → Type ℓ  -- Complex coherence conditions

--------------------------------------------------------------------------------
-- Kolmogorov Complexity

{-|
## Kolmogorov Complexity K

The **Kolmogorov complexity** K(x) is the length of the shortest program that
outputs x. In categorical terms:

- K: Ob → ℝ₊ (or ℕ for discrete case)
- K(A) = min {|p| : p computes A}

**Connection to theories**:
- Programs are morphisms in Kleisli category
- Compression = finding minimal theory T(A) → A
-}

-- Real numbers (imported from ClosedMonoidal)
open Neural.Semantics.ClosedMonoidal using (ℝ; _+ℝ_) public

-- Additional real number operations (postulated)
postulate
  ℝ₊ : Type  -- Non-negative reals
  _*ℝ_ : ℝ → ℝ → ℝ
  _/ℝ_ : ℝ → ℝ → ℝ
  _≤ℝ_ : ℝ → ℝ → Type
  zero-real : ℝ₊
  one-real : ℝ₊
  to-real : ℝ₊ → ℝ  -- Coercion from non-negative to all reals

module KolmogorovComplexity {o ℓ} {C : Precategory o ℓ}
                            {M : Monoidal-category C}
                            (Th : Theory C M) where
  open Precategory C
  open Theory Th

  -- Program length (size of morphism encoding)
  postulate
    length : ∀ {A B} → Hom A B → ℝ₊

  -- Kolmogorov complexity: minimum program length
  postulate
    K : Ob → ℝ₊
    K-minimal : ∀ {A} (f : Hom (T A) A)
              → to-real (K A) ≤ℝ to-real (length f)

  -- Conditional complexity K(A|B)
  postulate
    K-cond : Ob → Ob → ℝ₊
    K-cond-def : ∀ {A B} → to-real (K-cond A B) ≤ℝ to-real (K A)

  -- Compression: Finding minimal theory
  -- Goal: Find f: T(A) → A such that length(f) ≈ K(A)
  postulate
    compress : ∀ (A : Ob) → Σ[ TA ∈ Ob ] (Hom TA A)

--------------------------------------------------------------------------------
-- Shannon Entropy F

{-|
## Shannon Entropy F

The **Shannon entropy** F(A) is the expected information content of A under
a probability distribution:

F(A) = -Σᵢ pᵢ log(pᵢ)

where pᵢ is the probability of outcome i.

**Connection to Kolmogorov complexity**:
- F ≤ K (Shannon entropy bounded by Kolmogorov complexity)
- F/K: Compression ratio
-}

module ShannonEntropy where
  -- Probability distribution on finite set
  postulate
    Dist : Nat → Type  -- Distribution on Fin n
    prob : ∀ {n} → Dist n → Nat → ℝ₊  -- Probability function

  -- Shannon entropy
  postulate
    H : ∀ {n} → Dist n → ℝ
    log : ℝ → ℝ

  -- Entropy definition: H(p) = -Σᵢ pᵢ log(pᵢ)
  postulate
    entropy-def : ∀ {n} (p : Dist n)
                → Type  -- H(p) = sum over finite set

  -- Conditional entropy
  postulate
    H-cond : ∀ {n m} → Dist (n * m) → ℝ

--------------------------------------------------------------------------------
-- Compression Ratio F/K

{-|
## Compression Ratio F/K

The **compression ratio** F/K compares statistical (Shannon) and algorithmic
(Kolmogorov) information:

- F/K = 1: Maximum compressibility (highly regular data)
- F/K ≈ 0: Random data (incompressible)

**Interpretation**:
- High F/K: Data has patterns (good for neural learning)
- Low F/K: Data is random (hard to learn)

**Connection to resource theory (Section 3.2)**:
- F: Statistical resource measure
- K: Algorithmic resource measure
- F/K: Conversion rate between resources
-}

module CompressionRatio {o ℓ} {C : Precategory o ℓ}
                        {M : Monoidal-category C}
                        (Th : Theory C M) where
  open Precategory C
  open KolmogorovComplexity Th
  open ShannonEntropy

  -- Compression ratio for object A with distribution p
  postulate
    compression-ratio : ∀ (A : Ob) {n : Nat} (p : Dist n) → ℝ₊

  -- Bounds: 0 ≤ F/K ≤ 1
  postulate
    ratio-lower-bound : ∀ {A n p} → to-real zero-real ≤ℝ to-real (compression-ratio A {n} p)
    ratio-upper-bound : ∀ {A n p} → to-real (compression-ratio A {n} p) ≤ℝ to-real one-real

  -- Connection to resource convertibility (Section 3.2)
  -- If F/K ≈ 1, then resource A is highly convertible to B
  postulate
    convertibility : ∀ {A B : Ob} {n m} {pA : Dist n} {pB : Dist m}
                   → ℝ₊  -- Conversion rate ρA→B
    convertibility-bound : ∀ {A B n m pA pB}
                         → to-real (convertibility {A} {B} {n} {m} {pA} {pB})
                           ≤ℝ to-real (compression-ratio A pA)

--------------------------------------------------------------------------------
-- Information Comonad

{-|
## Information as Comonad

The theory comonad T can be interpreted as an **information comonad**:

- T(A): Compressed/encoded representations of A
- ε: T(A) → A: Decompression/decoding
- δ: T(A) → T(T(A)): Iterative compression

**Key insight**: Theories are compression schemes
- Lossy compression: ε ∘ δ ≠ id
- Lossless compression: ε ∘ δ = id (comonad law)

**Connection to neural networks**:
- Hidden layers = compression T(A)
- Decoder = counit ε
- Encoder = coalgebra structure
-}

module InformationComonad {o ℓ} {C : Precategory o ℓ}
                          {M : Monoidal-category C}
                          (Th : Theory C M) where
  open Precategory C
  open Theory Th
  open KolmogorovComplexity Th

  -- Compression quality: How well does T(A) represent A?
  postulate
    compression-quality : ∀ (A : Ob) → ℝ₊
    quality-def : ∀ {A} → to-real (compression-quality A) ≤ℝ to-real (K A)

  -- Rate-distortion tradeoff
  -- Smaller T(A) → Higher distortion
  postulate
    rate : ∀ (A : Ob) → ℝ₊  -- Size of compressed representation
    distortion : ∀ (A : Ob) → ℝ₊  -- Loss in compression

  postulate
    rate-distortion-tradeoff : ∀ {A} → Type  -- R(D) function

  -- Neural network interpretation
  -- Hidden layer H = T(Input)
  -- Encoder: Input → T(Input)
  -- Decoder: T(Input) → Output
  postulate
    neural-encoder : ∀ (Input Output : Ob)
                   → Hom Input (T Input)
    neural-decoder : ∀ (Input Output : Ob)
                   → Hom (T Input) Output

--------------------------------------------------------------------------------
-- Linear Logic Connection

{-|
## Linear Logic and Information

Linear logic provides a resource-sensitive logic where:
- A ⊗ B: Simultaneous resources (tensor)
- A ⊸ B: Linear implication (consume A to produce B)
- !A: Unlimited copies of A (exponential modality)

**Connection to information theory**:
- !A: Compressible/regular data (can duplicate)
- A without !: Random data (can't duplicate)
- Compression = finding !-structure

**Bar-complex for !**:
- Bar(!A) encodes the algebraic structure of duplication
- Kleisli category for ! = theories about duplicable data
-}

module LinearLogicInformation {o ℓ} {C : Precategory o ℓ}
                              {M : Monoidal-category C}
                              (E : has-exponential-comonad C) where
  open Precategory C
  open has-exponential-comonad E

  -- ! comonad as information comonad
  info-theory : Theory C M
  info-theory = {!!}  -- Construct from exponential comonad

  -- Duplicable data has high F/K ratio
  postulate
    duplicable-compressible : ∀ {A : Ob}
                            → Type  -- ! A implies high F/K

  -- Linear types (without !) are incompressible
  postulate
    linear-incompressible : ∀ {A : Ob}
                          → Type  -- No ! means F/K ≈ 0

--------------------------------------------------------------------------------
-- Summary

{-|
## Summary

This module connects linear logic with information theory via:

1. **Theories as comonads**: T(A) = compressed representations
2. **Bar-complex**: Simplicial structure Bar_n(A) = T^n(A)
3. **Tri-simplicial sets**: Three dimensions (⊗, T, depth)
4. **Kolmogorov complexity K**: Algorithmic information
5. **Shannon entropy F**: Statistical information
6. **Compression ratio F/K**: Regularity measure
7. **Information comonad**: Neural compression via T
8. **Linear logic**: ! modality = duplicable = compressible

**Key equations**:
- Comonad laws: ε ∘ δ = id, T(ε) ∘ δ = id, δ ∘ δ = T(δ) ∘ δ
- Bar-complex: Bar_n(A) = T^n(A)
- Compression: 0 ≤ F/K ≤ 1

**Applications**:
- Neural autoencoders as comonads
- Regularization as ! structure
- Semantic compression in language models
- Information bottleneck principle
-}
