{-# OPTIONS --cubical #-}

{-|
# Einsum Index System (Concrete Implementation)

**Zero Postulates Policy**: Everything is concrete and executable.

## Key Design

**Finite Index Enumeration**: Idx is a finite datatype, not strings
- ✅ Decidable equality via pattern matching
- ✅ Exhaustive case analysis
- ✅ No axioms needed

-}

module Neural.Compile.Einsum.Index where

open import 1Lab.Prelude
open import 1Lab.Path

-- Use 1Lab's List
open import Data.List using (List; []; _∷_; _++_; length; filter)
open import Data.Nat.Base using (Nat; zero; suc; _+_; _*_; Discrete-Nat)
open import Prim.Data.Nat using (_==_)  -- Nat equality
-- Use 1Lab's Dec and Discrete infrastructure
open import Data.Dec.Base using (Dec; yes; no; Discrete; Discrete-inj; _≡?_)
-- Use 1Lab's Bool
open import Data.Bool.Base using (Bool; true; false; if_then_else_; not)

-- Define missing list functions concretely (with local names to avoid conflicts)
private
  list-map : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} → (A → B) → List A → List B
  list-map f [] = []
  list-map f (x ∷ xs) = f x ∷ list-map f xs

  list-any : ∀ {ℓ} {A : Type ℓ} → (A → Bool) → List A → Bool
  list-any p [] = false
  list-any p (x ∷ xs) = if p x then true else list-any p xs

  list-all : ∀ {ℓ} {A : Type ℓ} → (A → Bool) → List A → Bool
  list-all p [] = true
  list-all p (x ∷ xs) = if p x then list-all p xs else false

--------------------------------------------------------------------------------
-- § 1: Index Names (Finite Enumeration)
--------------------------------------------------------------------------------

{-|
## Dimension Indices

Finite set of dimension labels for tensors.

**Design choice**: Finite enumeration instead of strings
- Decidable equality for free
- Exhaustive pattern matching
- No postulates needed
-}

data Idx : Type where
  -- Generic dimensions
  i j k l m n : Idx

  -- Batch/sequential
  b : Idx    -- batch
  t : Idx    -- time
  s : Idx    -- sequence

  -- Attention-specific
  q : Idx    -- query sequence
  kk : Idx   -- key sequence (doubled to avoid clash with k)
  v : Idx    -- value sequence
  h : Idx    -- heads
  e : Idx    -- embedding dimension per head
  d : Idx    -- model dimension (d_model)

  -- Convolution-specific
  c : Idx    -- channels (generic)
  ic : Idx   -- input channels
  oc : Idx   -- output channels
  ks : Idx   -- kernel size
  w : Idx    -- window position

  -- Output dimension
  o : Idx

{-|
## Decidable Equality

Implemented via discriminator function - maps each constructor to unique Nat.

**Technique**: Define `idx-code : Idx → Nat` that assigns distinct codes,
then prove equality by comparing codes.
-}

-- Discriminator: map each Idx to a unique natural number
idx-code : Idx → Nat
idx-code i = 0
idx-code j = 1
idx-code k = 2
idx-code l = 3
idx-code m = 4
idx-code n = 5
idx-code b = 6
idx-code t = 7
idx-code s = 8
idx-code q = 9
idx-code kk = 10
idx-code v = 11
idx-code h = 12
idx-code e = 13
idx-code d = 14
idx-code c = 15
idx-code ic = 16
idx-code oc = 17
idx-code ks = 18
idx-code w = 19
idx-code o = 20

-- Injectivity: decode from Nat
-- Since idx-code is surjective on its image, we can decode
decode-idx : Nat → Idx
decode-idx 0 = i
decode-idx 1 = j
decode-idx 2 = k
decode-idx 3 = l
decode-idx 4 = m
decode-idx 5 = n
decode-idx 6 = b
decode-idx 7 = t
decode-idx 8 = s
decode-idx 9 = q
decode-idx 10 = kk
decode-idx 11 = v
decode-idx 12 = h
decode-idx 13 = e
decode-idx 14 = d
decode-idx 15 = c
decode-idx 16 = ic
decode-idx 17 = oc
decode-idx 18 = ks
decode-idx 19 = w
decode-idx 20 = o
decode-idx _ = i  -- Default (unreachable for valid codes)

-- Round-trip: decode ∘ idx-code = id
decode-idx-code : ∀ x → decode-idx (idx-code x) ≡ x
decode-idx-code i = refl
decode-idx-code j = refl
decode-idx-code k = refl
decode-idx-code l = refl
decode-idx-code m = refl
decode-idx-code n = refl
decode-idx-code b = refl
decode-idx-code t = refl
decode-idx-code s = refl
decode-idx-code q = refl
decode-idx-code kk = refl
decode-idx-code v = refl
decode-idx-code h = refl
decode-idx-code e = refl
decode-idx-code d = refl
decode-idx-code c = refl
decode-idx-code ic = refl
decode-idx-code oc = refl
decode-idx-code ks = refl
decode-idx-code w = refl
decode-idx-code o = refl

-- Injectivity via round-trip
idx-code-injective : ∀ {x y} → idx-code x ≡ idx-code y → x ≡ y
idx-code-injective {x} {y} p =
  x                      ≡⟨ sym (decode-idx-code x) ⟩
  decode-idx (idx-code x) ≡⟨ ap decode-idx p ⟩
  decode-idx (idx-code y) ≡⟨ decode-idx-code y ⟩
  y                      ∎

-- Automatically derive Discrete instance from injection into Nat!
Discrete-Idx : Discrete Idx
Discrete-Idx = Discrete-inj idx-code idx-code-injective Discrete-Nat

-- Decidable equality: derived from Discrete instance
Idx-eq? : (x y : Idx) → Dec (x ≡ y)
Idx-eq? = Discrete.decide Discrete-Idx

--------------------------------------------------------------------------------
-- § 2: Index Contexts
--------------------------------------------------------------------------------

{-|
## Index Context (List of Indices)

Open product type - extensible dimension list.
-}

IndexCtx : Type
IndexCtx = List Idx

-- Empty context (scalar)
∅ : IndexCtx
∅ = []

--------------------------------------------------------------------------------
-- § 3: Context Operations (All Concrete!)
--------------------------------------------------------------------------------

{-|
## Membership (Decidable)

Check if index appears in context - returns Bool for computability.
-}

infix 4 _∈ᵢ?_
_∈ᵢ?_ : Idx → IndexCtx → Bool
x ∈ᵢ? [] = false
x ∈ᵢ? (y ∷ ctx) with Idx-eq? x y
... | yes _ = true
... | no  _ = x ∈ᵢ? ctx

{-|
## Difference (Remove Indices)

Filter out indices from context.
-}

infixl 6 _\\_
_\\_ : IndexCtx → List Idx → IndexCtx
ctx \\ [] = ctx
ctx \\ (x ∷ remove) = filter (λ idx → not (idx ∈ᵢ? (x ∷ remove))) ctx

{-|
## Disjoint Union

Concatenate contexts (user must ensure disjointness).
-}

infixr 5 _⊎ᵢ_
_⊎ᵢ_ : IndexCtx → IndexCtx → IndexCtx
_⊎ᵢ_ = _++_

{-|
## Append
-}

infixr 5 _++ᵢ_
_++ᵢ_ : IndexCtx → IndexCtx → IndexCtx
_++ᵢ_ = _++_

{-|
## Length
-}

dim : IndexCtx → Nat
dim = length

{-|
## Count Occurrences

How many times does an index appear?
-}

count : Idx → IndexCtx → Nat
count x [] = 0
count x (y ∷ ctx) with Idx-eq? x y
... | yes _ = suc (count x ctx)
... | no  _ = count x ctx

--------------------------------------------------------------------------------
-- § 4: Context Properties
--------------------------------------------------------------------------------

{-|
## Disjointness (Decidable)
-}

disjoint? : IndexCtx → IndexCtx → Bool
disjoint? ctx₁ ctx₂ = list-all (λ idx → not (idx ∈ᵢ? ctx₂)) ctx₁

{-|
## Permutation (Concrete Definition)

Two contexts are permutations if they have the same element counts.
-}

is-permutation? : IndexCtx → IndexCtx → Bool
is-permutation? ctx₁ ctx₂ =
  (length ctx₁ == length ctx₂) &&
  list-all (λ idx → count idx ctx₁ == count idx ctx₂) (ctx₁ ++ ctx₂)
  where
    _&&_ : Bool → Bool → Bool
    true && true = true
    _ && _ = false

record Permutation (ctx : IndexCtx) : Type where
  constructor mkPerm
  field
    reordered : IndexCtx
    is-perm : is-permutation? ctx reordered ≡ true

permute-ctx : {ctx : IndexCtx} → Permutation ctx → IndexCtx
permute-ctx perm = Permutation.reordered perm

--------------------------------------------------------------------------------
-- § 5: Examples
--------------------------------------------------------------------------------

module Examples where
  -- Matrix: [i, j]
  matrix-ctx : IndexCtx
  matrix-ctx = i ∷ j ∷ []

  -- Batch of vectors: [b, i]
  batch-vector-ctx : IndexCtx
  batch-vector-ctx = b ∷ i ∷ []

  -- Attention scores: [b, h, q, kk]
  attention-scores-ctx : IndexCtx
  attention-scores-ctx = b ∷ h ∷ q ∷ kk ∷ []

  -- Matrix multiply output: [i, k] (after contracting j)
  matmul-output : IndexCtx
  matmul-output = i ∷ k ∷ []

  -- Test membership
  test-membership : Bool
  test-membership = j ∈ᵢ? matrix-ctx  -- true

  -- Test difference
  test-diff : IndexCtx
  test-diff = (i ∷ j ∷ k ∷ []) \\ (j ∷ [])  -- Should be [i, k]
