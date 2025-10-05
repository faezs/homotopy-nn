{-|
# Neural Codes and Code Categories (Section 5.1)

This module implements neural codes as binary/q-ary codes representing spike
trains, along with categorical structures on codes from Section 5.1 of
Manin & Marcolli (2024).

## Overview

**Neural codes**: Binary codes C ⊂ F_2^n recording spike trains
- n = T/Δt time intervals
- Code word c = (c_1, ..., c_n) with c_i = 1 if spike detected, 0 otherwise
- Shannon Random Code Ensemble (SRCE) with Bernoulli measure μ_P

**Code categories**:
1. **Codes** (Lemma 5.4): Decomposable codes with direct sum ⊕
2. **Codes_{n,*}** (Lemma 5.6): Pointed codes with wedge sum ∨

**Probability distributions** (Definition 5.2, 5.8):
- P_C = (p, 1-p) with p = Σ_c b(c)/(n·#C)
- Refined: P_C(c) = b(c)/(n·(#C-1)) for c ≠ c_0

**Functor** (Lemma 5.9): P : Codes_{n,*} → Pf assigns probabilities to codes
-}

module Neural.Code where

open import Neural.Base
open import Neural.Information public
  using (ℝ; _+ℝ_; _*ℝ_; _/ℝ_; _≤ℝ_; zeroℝ; oneℝ; _+ℝ'_; _*ℝ'_; _-ℝ'_; 1ℝ; 0ℝ; -ℝ'_)

open import 1Lab.Prelude hiding (_∈_)
open import 1Lab.HLevel
open import 1Lab.Path
open import 1Lab.Type

open import Cat.Base
open import Cat.Functor.Base

open import Data.Bool.Base
open import Data.Fin.Base
open import Data.Finset.Base
open import Data.List.Base hiding (length)
open import Data.Nat.Base using (Nat; zero; suc; _+_; _*_; ≤-trans; s≤s; 0≤x) renaming (_≤_ to _≤ℕ_)
open import Data.Nat.Order using (≤-refl; ≤-refl')
open import Data.Nat.Properties using (+-associative; +-commutative; +-preserves-≤)

open import Data.Power
open import Data.Sum.Base using (_⊎_; inl; inr)

private variable
  o ℓ : Level

-- TODO: Generate neural code from neural network responses
-- Collects all possible activation patterns across stimuli
-- Requires: NeuralNetwork type and neural-response function
-- generate-neural-code : {X : StimulusSpace} (N : NeuralNetwork X)
--                      → (stimuli : List X)
--                      → NeuralCode (vertices (NeuralNetwork.graph N))
-- generate-neural-code N stimuli = map (neural-response N) stimuli

{-|
## Code Words and Basic Definitions

Code words represent individual observations of spike trains over n time intervals.
-}

-- | Code words: individual binary response patterns
CodeWord : (n : Nat) → Type
CodeWord n = Fin n → Bool

-- | Active neurons for a given code word
active-neurons : {n : Nat} → CodeWord n → List (Fin n)
active-neurons {n} cw = Data.List.Base.filter cw (fin-list n)
  where
    fin-list : (k : Nat) → List (Fin k)
    fin-list zero = []
    fin-list (suc k) = fzero ∷ map fsuc (fin-list k)

-- | Support of a neural code: all neurons that appear in some code word
code-support : {n : Nat} → NeuralCode n → List (Fin n)
code-support code = Data.List.Base.concat (map active-neurons code)

-- | Overlap pattern: neurons that fire together
-- This captures the combinatorial structure needed for homotopy reconstruction
record OverlapPattern (n : Nat) : Type where
  field
    neurons : List (Fin n)
    occurs-in-code : CodeWord n → Bool

{-|
## Code Parameters [n,k,d]_q

An **[n,k,d]_q-code** is a q-ary code C ⊂ A^n where:
- n: length (number of symbols per code word)
- #C = q^k: cardinality (number of code words)
- d: minimum distance min{d_H(c,c') | c ≠ c' ∈ C}
- q = #A: alphabet size

For neural codes, typically q=2 (binary) but we allow general q for extensions
like spike directivity encoding.
-}

record CodeParameters : Type where
  no-eta-equality
  field
    {-| Length of code words -}
    length : Nat

    {-| Alphabet size -}
    alphabet-size : Nat

    {-| Number of code words (as exponent: #C = q^k) -}
    dimension : Nat

    {-| Minimum Hamming distance -}
    min-distance : Nat

open CodeParameters public

{-|
Binary code: special case with q=2
-}
BinaryCodeParameters : Type
BinaryCodeParameters =
  Σ CodeParameters (λ params → params .alphabet-size ≡ 2)


module AbstractCodes where
{-|
## Abstract Codes

A code C is a collection of code words over an alphabet.
For categorical purposes, we model codes as finite sets of code words.
-}

{-| Alphabet as finite set -}
Alphabet : Nat → Type
Alphabet q = Fin q

{-|
## Hamming Distance

The Hamming distance d_H(c, c') counts the number of positions where code
words c and c' differ.

**Properties**:
- d_H(c, c') = #{i ∈ {1,...,n} | c_i ≠ c'_i}
- d_H(c, c) = 0
- d_H(c, c') = d_H(c', c)
- d_H(c, c'') ≤ d_H(c, c') + d_H(c', c'') (triangle inequality)
-}

δ : {q : Nat} → Alphabet q → Alphabet q → Nat
δ w₀ w₁ = if (w₀ .lower == w₁ .lower) then 0 else 1

sum : {n : Nat} → (Fin n → Nat) -> Nat
sum {zero} f = 0
sum {suc n} f = f fzero + sum {n} ((f ∘ fsuc))

{-| Hamming distance between code words over any alphabet -}
hamming-distance : {n q : Nat} → (Fin n → Alphabet q) → (Fin n → Alphabet q) → Nat
hamming-distance {zero} w1 w2 = 0
hamming-distance {suc n} w1 w2 = δ (w1 fzero) (w2 fzero) + hamming-distance (w1 ∘ fsuc) (w2 ∘ fsuc) 

==-sym : ∀ {n m} → (n == m) ≡ (m == n)
==-sym {zero} {zero} = refl
==-sym {zero} {suc m} = refl
==-sym {suc n} {zero} = refl
==-sym {suc n} {suc m} = ==-sym {n} {m}

==-refl : ∀ {n} → (n == n) ≡ true
==-refl {zero} = refl
==-refl {suc n} = ==-refl {n}

==-trans : ∀ {x y z} → (x == y) ≡ true → (y == z) ≡ true → (x == z) ≡ true
==-trans {zero} {zero} {zero} p q = refl
==-trans {zero} {zero} {suc z} p q = q
==-trans {zero} {suc y} {z} p q = absurd (true≠false (sym p))
==-trans {suc x} {zero} {z} p q = absurd (true≠false (sym p))
==-trans {suc x} {suc y} {zero} p q = absurd (true≠false (sym q))
==-trans {suc x} {suc y} {suc z} p q = ==-trans {x} {y} {z} p q


δ-sym : {q : Nat} (x y : Alphabet q) → δ x y ≡ δ y x
δ-sym x y = ap (λ b → if b then 0 else 1) (==-sym {x .lower} {y .lower})

hamming-distance-sym :
    {n q : Nat} →
    (c c' : Fin n → Alphabet q) → hamming-distance c c' ≡ hamming-distance c' c
hamming-distance-sym {zero} c c' = refl
hamming-distance-sym {suc n} c c' =
  ap₂ _+_ (δ-sym (c fzero) (c' fzero)) (hamming-distance-sym (c ∘ fsuc) (c' ∘ fsuc))

δ-refl : {q : Nat} (x : Alphabet q) → δ x x ≡ 0
δ-refl x = ap (λ b → if b then 0 else 1) (==-refl {x .lower})

hamming-distance-zero :
    {n q : Nat} →
    (c : Fin n → Alphabet q) → hamming-distance c c ≡ 0
hamming-distance-zero {zero} {zero} c = refl
hamming-distance-zero {zero} {suc q} c = refl
hamming-distance-zero {suc n} {zero} c = absurd (Fin-absurd (c fzero))
hamming-distance-zero {suc n} {suc q} c =
  ap₂ _+_ (δ-refl (c fzero)) (hamming-distance-zero (c ∘ fsuc))

-- Helper for the contradiction case in triangle inequality
δ-contra : {q : Nat} (x y z : Alphabet q) →
           (x .lower == z .lower) ≡ false →
           (x .lower == y .lower) ≡ true →
           (y .lower == z .lower) ≡ true →
           ⊥
δ-contra x y z xz-false xy-true yz-true =
  true≠false (sym (==-trans {x .lower} {y .lower} {z .lower} xy-true yz-true) ∙ xz-false)

-- Computation lemma: hamming-distance on suc n reduces
hamming-distance-suc : {n q : Nat} (c c' : Fin (suc n) → Alphabet q) →
  hamming-distance c c' ≡ δ (c fzero) (c' fzero) + hamming-distance (c ∘ fsuc) (c' ∘ fsuc)
hamming-distance-suc c c' = refl

δ-triangle : {q : Nat} (x y z : Alphabet q) → δ x z ≤ℕ δ x y + δ y z
δ-triangle x y z = go (x .lower == z .lower) (x .lower == y .lower) (y .lower == z .lower) refl refl refl
  where
    go : (xz xy yz : Bool) →
         (x .lower == z .lower) ≡ xz →
         (x .lower == y .lower) ≡ xy →
         (y .lower == z .lower) ≡ yz →
         δ x z ≤ℕ δ x y + δ y z
    go true _ _ p _ _ =
      transport (λ i → (if (sym p i) then 0 else 1) ≤ℕ δ x y + δ y z) 0≤x
    go false true false xz-eq xy-eq yz-eq =
      transport (λ i → (if (sym xz-eq i) then 0 else 1) ≤ℕ (if (sym xy-eq i) then 0 else 1) + (if (sym yz-eq i) then 0 else 1)) (s≤s 0≤x)
    go false true true xz-eq xy-eq yz-eq = absurd (δ-contra x y z xz-eq xy-eq yz-eq)
    go false false _ xz-eq xy-eq _ =
      transport (λ i → (if (sym xz-eq i) then 0 else 1) ≤ℕ (if (sym xy-eq i) then 0 else 1) + δ y z) (s≤s 0≤x)

hamming-distance-triangle :
    {n q : Nat} →
    (c c' c'' : Fin n → Alphabet q) →
    hamming-distance c c'' ≤ℕ hamming-distance c c' + hamming-distance c' c''
hamming-distance-triangle {zero} c c' c'' = 0≤x
hamming-distance-triangle {suc n} c c' c'' =
  ≤-trans step1 step2
  where
    a = δ (c fzero) (c'' fzero)
    b = δ (c fzero) (c' fzero) + δ (c' fzero) (c'' fzero)
    a' = hamming-distance (c ∘ fsuc) (c'' ∘ fsuc)
    b' = hamming-distance (c ∘ fsuc) (c' ∘ fsuc) + hamming-distance (c' ∘ fsuc) (c'' ∘ fsuc)

    -- Step 1: Use triangle inequality for first position and IH for rest
    step1 : (a + a') ≤ℕ (b + b')
    step1 = +-preserves-≤ a b a' b'
              (δ-triangle (c fzero) (c' fzero) (c'' fzero))
              (hamming-distance-triangle (c ∘ fsuc) (c' ∘ fsuc) (c'' ∘ fsuc))

    -- Step 2: Rearrange (a+b)+(c+d) to (a+c)+(b+d) using associativity and commutativity
    +-rearrange : ∀ a b c d → (a + b) + (c + d) ≡ (a + c) + (b + d)
    +-rearrange a b c d =
      (a + b) + (c + d)       ≡⟨ sym (+-associative a b (c + d)) ⟩
      a + (b + (c + d))       ≡⟨ ap (a +_) (+-associative b c d) ⟩
      a + ((b + c) + d)       ≡⟨ ap (a +_) (ap (_+ d) (+-commutative b c)) ⟩
      a + ((c + b) + d)       ≡⟨ ap (a +_) (sym (+-associative c b d)) ⟩
      a + (c + (b + d))       ≡⟨ +-associative a c (b + d) ⟩
      (a + c) + (b + d)       ∎

    step2 : (b + b') ≤ℕ (hamming-distance c c' + hamming-distance c' c'')
    step2 = ≤-refl' (+-rearrange (δ (c fzero) (c' fzero)) (δ (c' fzero) (c'' fzero))
                                  (hamming-distance (c ∘ fsuc) (c' ∘ fsuc)) (hamming-distance (c' ∘ fsuc) (c'' ∘ fsuc)))

{-|
Code as finite set of words

For simplicity, we represent a code of length n over q-ary alphabet
as a list of code words (allowing duplicates for categorical reasons).
-}
Code : (n q : Nat) → Type
Code n q = List (Fin n → Alphabet q)

{-|
Zero word: constant word with all digits zero

Note: q = suc q' ensures non-empty alphabet
-}
zero-word : {n q' : Nat} → (Fin n → Alphabet (suc q'))
zero-word = λ _ → fzero

{-|
One word: constant word with all digits one (for binary codes)
-}
one-word : {n : Nat} → (Fin n → Fin 2)
one-word = λ _ → fsuc fzero

{-|
Number of occurrences of symbol a in code word c
-}

count-symbol : {n q : Nat} → (Alphabet q) → (Fin n → Alphabet q) → Nat
count-symbol {n} {q} a c = sum (λ i → δ a (c i))

{-|
Number of ones in binary code word (b(c) in paper)
-}
count-ones : {n : Nat} → (Fin n → Fin 2) → Nat
count-ones c = count-symbol (fsuc fzero) c

{-|
Number of zeros in binary code word (a(c) in paper)
-}
count-zeros : {n : Nat} → (Fin n → Fin 2) → Nat
count-zeros c = count-symbol fzero c

{-|
## Definition 5.2: Probability Distribution of a Code

For binary code C, the probability distribution P_C = (p, 1-p) is given by:
  p = Σ_{c∈C} b(c)/(n·#C)

where b(c) is the number of 1's in code word c.
-}

module ProbabilityDistributionBasic where
  {-|
  Probability p for binary code (Definition 5.2)
  -}
  postulate
    code-probability-p :
      {n : Nat} →
      (C : Code n 2) →
      ℝ

    code-probability-formula :
      {n : Nat} →
      (C : Code n 2) →
      code-probability-p C ≡ {-| Σ_{c∈C} count-ones(c)/(n·#C) -} oneℝ
      -- Simplified for now

{-|
## Lemma 5.3: Hamming Distance Preserving Maps

Let f : C → C' be a surjective map sending zero word to itself such that
d_H(f(c), f(c')) ≤ d_H(c, c') for all c,c' ∈ C.

Then: p(f(c)) = λ(c) · p(c) where λ(c) = b(f(c))/b(c) ≤ 1
-}

postulate
  hamming-preserving-probability :
    {n : Nat} →
    (C C' : Code n 2) →
    (f : (Fin n → Fin 2) → (Fin n → Fin 2)) →
    {-| Properties: f(zero-word) = zero-word, Hamming distance decreasing -}
    {-| Then: p(f(c)) = λ(c) · p(c) -}
    Type

{-|
## Category of Codes (Lemma 5.4)

**Construction 1**: Decomposable codes

Objects: [n,k,d]_q-codes over alphabet A containing the zero word
Morphisms: ϕ : C → C' with d_H(ϕ(c₁), ϕ(c₂)) ≤ d_H(c₁, c₂)
Sum: C ⊕ C' = {(c,c') ∈ A^{n+n'} | c∈C, c'∈C'}  (direct sum)
Zero: C = {c₀} (just the zero word of length n)

Note: This formulation allows codes of different lengths.
-}

{-|
Code containing zero word
-}


open import Data.List.Membership using (here; there; map-member; ++-memberₗ; ++-memberᵣ)
  renaming (_∈ₗ_ to _∈_)
open import Data.Id.Base using (_≡ᵢ_; reflᵢ; Id≃path)
open import 1Lab.Membership using (_∉_)

record CodeWithZero (n q : Nat) : Type where
  no-eta-equality
  field
    code : Code n (suc q)
    contains-zero : zero-word ∈ᶠˢ (from-list code)

open CodeWithZero public

{-|
Direct sum of codes (Equation 5.2)
-}

-- Concatenate two code words
concat-words : {n n' q : Nat} → (Fin n → Alphabet q) → (Fin n' → Alphabet q) → (Fin (n + n') → Alphabet q)
concat-words {n} {n'} {q} w w' i with split-+ i
... | inl j = w j
... | inr k = w' k

-- Cartesian product for Finset
cartesian-productᶠˢ : {A B : Type} → Finset A → Finset B → Finset (A × B)
cartesian-productᶠˢ [] _ = []
cartesian-productᶠˢ (a ∷ as) bs = union (mapᶠˢ (a ,_) bs) (cartesian-productᶠˢ as bs)
cartesian-productᶠˢ (∷-dup x xs i) bs = union-dup (mapᶠˢ (x ,_) bs) (cartesian-productᶠˢ xs bs) i
cartesian-productᶠˢ (∷-swap x y xs i) bs = union-swap (mapᶠˢ (x ,_) bs) (mapᶠˢ (y ,_) bs) (cartesian-productᶠˢ xs bs) i
cartesian-productᶠˢ (squash x y p q i j) bs = squash
  (cartesian-productᶠˢ x bs) (cartesian-productᶠˢ y bs)
  (λ i → cartesian-productᶠˢ (p i) bs) (λ i → cartesian-productᶠˢ (q i) bs) i j

postulate
  code-direct-sum :
    {n n' q : Nat} →
    CodeWithZero n q →
    CodeWithZero n' q →
    CodeWithZero (n + n') q
-- code-direct-sum {n} {n'} {q} C C' = record
--   { code = mapᶠˢ (λ pair → concat-words (pair .fst) (pair .snd)) (cartesian-productᶠˢ (C .code) (C' .code))
--   ; contains-zero = {!!}
--   }
{-|
Morphism of codes (Hamming distance preserving)

Note: Morphisms are between codes over the same alphabet (q = q')
but potentially different lengths (n, n').
-}
CodeMorphism : {n n' q : Nat} (C : CodeWithZero n q) (C' : CodeWithZero n' q) → Type
CodeMorphism {n} {n'} {q} C C' =
  Σ ((Fin n → Alphabet q) → (Fin n' → Alphabet q))
    (λ func → (c₁ c₂ : Fin n → Alphabet q) → hamming-distance (func c₁) (func c₂) ≤ℕ hamming-distance c₁ c₂)

{-|
Category of codes (Lemma 5.4)
-}

CodeMorphism-is-set : {n n' q : Nat} (C : CodeWithZero n q) (C' : CodeWithZero n' q) → is-set (CodeMorphism C C')
CodeMorphism-is-set {n} {n'} {q} C C' =
  Σ-is-hlevel 2 (hlevel 2) λ f →
    is-prop→is-set (Π-is-hlevel 1 λ c₁ → Π-is-hlevel 1 λ c₂ → hlevel 1)

CodeMorphism-idr : {n n' q : Nat} {C : CodeWithZero n q} {C' : CodeWithZero n' q} (f : CodeMorphism C C') →
  ((λ x → f .fst x) , (λ c₁ c₂ → ≤-trans (f .snd c₁ c₂) ≤-refl)) ≡ f
CodeMorphism-idr f = Σ-pathp refl (is-prop→pathp (λ i → Π-is-hlevel 1 λ c₁ → Π-is-hlevel 1 λ c₂ → hlevel 1) _ _)

CodeMorphism-idl : {n n' q : Nat} {C : CodeWithZero n q} {C' : CodeWithZero n' q} (f : CodeMorphism C C') →
  ((λ x → f .fst x) , (λ c₁ c₂ → ≤-trans ≤-refl (f .snd c₁ c₂))) ≡ f
CodeMorphism-idl f = Σ-pathp refl (is-prop→pathp (λ i → Π-is-hlevel 1 λ c₁ → Π-is-hlevel 1 λ c₂ → hlevel 1) _ _)

Codes : (q : Nat) → Precategory lzero lzero
Codes q = record
  { Ob = Σ Nat (λ n → CodeWithZero n q)
  ; Hom = λ (n , C) (n' , C') → CodeMorphism C C'
  ; Hom-set = λ (n , C) (n' , C') → CodeMorphism-is-set C C'
  ; id = λ {(n , C)} → (λ x → x) , (λ c₁ c₂ → ≤-refl)
  ; _∘_ = λ {(n , C)} {(n' , C')} {(n'' , C'')} g f →
      (λ x → g .fst (f .fst x)) ,
      (λ c₁ c₂ → ≤-trans (g .snd (f .fst c₁) (f .fst c₂)) (f .snd c₁ c₂))
  ; idr = λ {(n , C)} {(n' , C')} f → CodeMorphism-idr {n} {n'} {q} {C} {C'} f
  ; idl = λ {(n , C)} {(n' , C')} f → CodeMorphism-idl {n} {n'} {q} {C} {C'} f
  ; assoc = λ {(n , C)} {(n' , C')} {(n'' , C'')} {(n''' , C''')} f g h →
      Σ-pathp refl (is-prop→pathp (λ i → Π-is-hlevel 1 λ c₁ → Π-is-hlevel 1 λ c₂ → hlevel 1) _ _)
  }

Codes-Ob : (q : Nat) → Type
Codes-Ob q = Precategory.Ob (Codes q)

Codes-Ob-is :
    (q : Nat) →
    Codes-Ob q ≡ (Σ Nat (λ n → CodeWithZero n q))
Codes-Ob-is q = refl

Codes-zero : (q n : Nat) → Precategory.Ob (Codes q)
Codes-zero q n = n , record
  { code = zero-word ∷ []
  ; contains-zero = hereₛ' reflᵢ
  }

{-|
## Category of Pointed Codes (Lemma 5.6)

**Construction 2**: Pointed codes with wedge sum

Objects: [n,k,d]_q-codes of fixed length n containing zero word c₀,
  excluding the code C = {c₀, c₁} (only constant words)
Morphisms: f : C → C' sending c₀ to c'₀
Sum: C ⊕ C' = C ∨ C' = (C ⊔ C')/(c₀ ~ c'₀) (wedge sum)
Zero: C = {c₀}

This is the symmetric monoidal category of pointed sets.
-}

{-|
Pointed code: code with distinguished basepoint (zero word)
-}
record PointedCode (n q : Nat) : Type where
  no-eta-equality
  field
    codeₙ : CodeWithZero n q

    {-| Excludes trivial code {c₀, c₁} with only constant words
        Only applies to binary codes (q = 1, alphabet = Fin 2) -}
    non-trivial : {-| For q = 1: ¬(code ≡ {c₀, c₁}) -} ⊤

open PointedCode public

{-|
Wedge sum of pointed codes (Equation 5.4)
-}

postulate
  pointed-code-wedge-sum :
    {n q : Nat} →
    PointedCode n q →
    PointedCode n q →
    PointedCode n q
  -- Implementation: C ∨ C' = (C ⊔ C')/(c₀ ~ c'₀)
  -- Should be the union of codes with zero words identified

{-|
Morphism of pointed codes

Note: q = suc q' to ensure non-empty alphabet (needed for zero-word)
-}
record PointedCodeMorphism {n q' : Nat} (C C' : PointedCode n (suc q')) : Type where
  no-eta-equality
  field
    func : (Fin n → Alphabet (suc q')) → (Fin n → Alphabet (suc q'))

    {-| Preserves basepoint -}
    preserves-zero : func zero-word ≡ zero-word

open PointedCodeMorphism public

{-|
Category of pointed codes of fixed length n (Lemma 5.6)
-}

Codes-pointed : (n q : Nat) → Precategory lzero lzero
Codes-pointed n q .Precategory.Ob = PointedCode n (suc q)
Codes-pointed n q .Precategory.Hom = λ a b → PointedCodeMorphism a b
Codes-pointed n q .Precategory.Hom-set C C' = Σ-is-hlevel 2 (hlevel 2) λ f →
  is-prop→is-set (hlevel 1)
Codes-pointed n q .Precategory.id {C} = record
  { func = λ x → x
  ; preserves-zero = refl
  }
Codes-pointed n q .Precategory._∘_ {C} {C'} {C''} g f = record
  { func = λ x → g .func (f .func x)
  ; preserves-zero = ap (g .func) (f .preserves-zero) ∙ g .preserves-zero
  }
Codes-pointed n q .Precategory.idr f = Σ-pathp refl (is-prop→pathp (λ i → hlevel 1) _ _)
Codes-pointed n q .Precategory.idl f = Σ-pathp refl (is-prop→pathp (λ i → hlevel 1) _ _)
Codes-pointed n q .Precategory.assoc f g h = Σ-pathp refl (is-prop→pathp (λ i → hlevel 1) _ _)


open import Cat.Monoidal.Base
Codes-pointed-is-monoidal :
    (n q : Nat) →
    {-| Symmetric monoidal structure with wedge sum -}
    Monoidal-category (Codes-pointed n q)
Codes-pointed-is-monoidal = {!!}


{-|
## Definition 5.8: Refined Probability Space for Codes

The probability space P_C associated to a binary code C is given by:

  P_C(c) = b(c)/(n·(#C-1))              for c ≠ c₀
  P_C(c₀) = 1 - Σ_{c≠c₀} b(c)/(n·(#C-1))  for c = c₀

where b(c) is the number of 1's in code word c.

This refines the binary probability (p, 1-p) from Definition 5.2.
-}

module RefinedProbabilitySpace where
  postulate
    {-| Refined probability for code word -}
    code-word-probability :
      {n : Nat} →
      (C : PointedCode n 2) →
      (c : Fin n → Fin 2) →
      ℝ

    {-| Cardinality of a code (number of code words) -}
    code-cardinality :
      {n : Nat} →
      (C : PointedCode n 2) →
      ℝ

    {-| Number of ones in a code word c -}
    ones-count :
      {n : Nat} →
      (c : Fin n → Fin 2) →
      ℝ

    code-word-probability-nonzero :
      {n : Nat} →
      (C : PointedCode n 2) →
      (c : Fin n → Fin 2) →
      ¬ (c ≡ zero-word) →
      {-| P_C(c) = b(c)/(n·(#C-1)) for c ≠ c₀ -}
      ∃[ n-ℝ ∈ ℝ ] (code-word-probability C c ≡ (ones-count c /ℝ (n-ℝ *ℝ' (code-cardinality C -ℝ' 1ℝ))))

    {-| Sum of probabilities over non-zero code words -}
    sum-nonzero-probabilities :
      {n : Nat} →
      (C : PointedCode n 2) →
      ℝ

    code-word-probability-zero :
      {n : Nat} →
      (C : PointedCode n 2) →
      code-word-probability C zero-word ≡ (1ℝ -ℝ' (sum-nonzero-probabilities C))

{-|
## Lemma 5.9: Functor P : Codes_{n,*} → Pf

The assignment C ↦ P_C determines a functor from pointed codes to
finite probabilities, compatible with sums and zero objects.

**Key properties**:
1. Code morphism f : C → C' induces (f, Λ) : (C, P_C) → (C', P_C')
2. Fiberwise measures: λ_{f(c)}(c) = P_C(c)/P_C'(f(c))
3. Wedge sum probabilities:
   - P_{C₁∨C₂}(c) = (N/(N+N')) · P_{C₁}(c) for c ∈ C₁ \ {c₀}
   - P_{C₁∨C₂}(c) = (N'/(N+N')) · P_{C₂}(c) for c ∈ C₂ \ {c₀}
   where N = #C₁ - 1, N' = #C₂ - 1
-}

-- Import Pf from Neural.Code.Probabilities
open import Neural.Code.Probabilities public
  using (PfObject; PfMorphism; Pf)

postulate
  {-|
  Functor from pointed codes to finite probabilities (Lemma 5.9)
  -}
  probability-functor :
    (n q : Nat) →
    Functor (Codes-pointed n q) Pf

  probability-functor-on-objects :
    {n q' : Nat} →
    (C : PointedCode n (suc q')) →
    {-| F₀(C) = (C, P_C) -}
    PfObject

  probability-functor-on-morphisms :
    {n q' : Nat} →
    {C C' : PointedCode n (suc q')} →
    (f : PointedCodeMorphism C C') →
    {-| F₁(f) = (f, Λ) with λ_{f(c)}(c) = P_C(c)/P_C'(f(c)) -}
    PfMorphism (probability-functor-on-objects C) (probability-functor-on-objects C')

  probability-functor-preserves-sum :
    {n q' : Nat} →
    (C C' : PointedCode n (suc q')) →
    {-| F(C ∨ C') ≅ F(C) ⊕ F(C') -}
    Type

{-|
## Shannon Random Code Ensemble (SRCE)

Neural codes are typically in the Shannon Random Code Ensemble, generated by
Bernoulli processes on shift spaces Σ₊_q with Bernoulli measure μ_P.

**Shift space**: Σ₊_q = A^ℕ with shift map σ(a₀a₁a₂...) = a₁a₂a₃...

**Bernoulli measure**: μ_P(Σ₊_q(w₁...wₙ)) = p^{aₙ(w)} (1-p)^{bₙ(w)}
where aₙ(w) = number of 1's, bₙ(w) = number of 0's

**Lemma 5.1**: For large n, neural code C is in SRCE with firing rate y = p/Δt

We postulate the shift space infrastructure for now.
-}

postulate
  {-| Shift space Σ₊_q = A^ℕ -}
  ShiftSpace : (q : Nat) → Type

  {-| Bernoulli measure on shift space -}
  BernoulliMeasure : (q : Nat) → (P : List ℝ) → Type

  {-|
  Cylinder set Σ₊_q(w) for word w
  -}
  cylinder-set :
    {q : Nat} →
    (w : List (Alphabet q)) →
    ShiftSpace q → Type

  {-|
  Lemma 5.1: Neural codes are in SRCE with firing rate y = p/Δt
  -}
  neural-code-in-SRCE :
    {n : Nat} →
    (C : Code n 2) →
    (p : ℝ) →
    {-| C is in SRCE_{(p,1-p)} with probability one -}
    Type
