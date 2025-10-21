{-# OPTIONS --cubical --rewriting --guardedness --no-load-primitives #-}

-- Fork-Specific Sheafification Analysis
-- Phase 2: Proving explicit construction equals HIT
--
-- Goal: Show that fork-sheafification's explicit construction
-- (paper lines 572-579) matches 1Lab's HIT definition

module Neural.Topos.ForkSheafification where

open import 1Lab.Prelude
open import 1Lab.HLevel

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Site.Base
open import Cat.Site.Sheafification
open import Cat.Diagram.Sieve
open import Cat.Instances.Sheaves

open import Neural.Topos.Architecture

private variable
  o ℓ : Level

-- =============================================================================
-- Phase 2.1: Understanding Fork-Coverage Sieves
-- =============================================================================

module _ {Γ : OrientedGraph o} (Γ-directed : Directed Γ) where
  open OrientedGraph Γ
  open BuildFork Γ Γ-directed

  -- The key insight from the paper (lines 572-579):
  -- "no value is changed except at A★, where X_A★ is replaced by the product"
  --
  -- In HIT terms, this means:
  -- 1. At original/tang vertices: inc(x) = x (identity)
  -- 2. At fork-star A★: glue with tine-sieve gives product

  -- First, let's understand what the tine sieve looks like
  -- At a fork-star (fork-star a conv), the tips are all vertices a' with Connection a' a

  -- Type of tips feeding into a convergent vertex
  TipsInto : (a : Layer) → is-convergent a → Type o
  TipsInto a conv = Σ Layer (λ a' → Connection a' a)

  -- The tine sieve at fork-star consists of all tip-to-star edges
  -- These correspond bijectively to the tips themselves

  -- Key observation: For a presheaf F, a matching family for the tine sieve is:
  --   (a' : TipsInto a conv) → F(original (a' .fst))
  -- which is exactly the data for a product!

  module ExplicitConstruction {κ : Level} (F : Functor (Fork-Category ^op) (Sets κ)) where
    open Functor F

    -- Explicit sheafification at each vertex type
    explicit-sheafify-at : (v : ForkVertex) → Type κ
    explicit-sheafify-at (original x) = ∣ F₀ (original x) ∣
      -- At original vertices: unchanged

    explicit-sheafify-at (fork-tang a conv) = ∣ F₀ (fork-tang a conv) ∣
      -- At fork-tang: unchanged

    explicit-sheafify-at (fork-star a conv) =
      -- At fork-star: PRODUCT over incoming tips!
      (tip : TipsInto a conv) → ∣ F₀ (original (tip .fst)) ∣

    -- This is what the paper describes!
    -- Now we need to show this equals what the HIT gives us

-- =============================================================================
-- Phase 2.2: Relating HIT Gluing to Products
-- =============================================================================

module _ {Γ : OrientedGraph o} (Γ-directed : Directed Γ) where
  open BuildFork Γ Γ-directed

  -- The HIT's `glue` constructor takes:
  --   - A covering sieve (for us: the tine sieve at fork-star)
  --   - A matching family (sections over the sieve)
  --   - A patch condition (compatibility)
  --
  -- For fork-tine sieves, a matching family IS exactly:
  --   (tip : TipsInto a conv) → F(original (tip .fst))
  --
  -- This is a dependent product - exactly what we need!

  module ProductViaHIT {κ : Level} (F : Functor (Fork-Category ^op) (Sets κ)) where
    open Functor F
    open ExplicitConstruction Γ-directed F

    -- The key lemma we need to prove:
    -- Sheafification at fork-star equals the explicit product

    fork-star-sheafification-is-product
      : (a : Layer) (conv : is-convergent a)
      → (Sheafification fork-coverage F .Functor.F₀ (fork-star a conv))
      ≡ (el ((tip : TipsInto a conv) → ∣ F₀ (original (tip .fst)) ∣)
           {!!})  -- need to show it's a set
    fork-star-sheafification-is-product a conv = {!!}
      -- This is THE KEY LEMMA!
      --
      -- Proof strategy:
      -- 1. Show that gluing with tine-sieve is universal for products
      -- 2. Use that tine-sieve at fork-star corresponds to tips
      -- 3. Matching family for tine-sieve = dependent product
      -- 4. HIT's glue constructor computes this product
      --
      -- This requires deep dive into how `glue` works with the sieve structure

-- =============================================================================
-- Phase 2.3: Simplified Approach - Use Sheaf Condition Directly
-- =============================================================================

-- Alternative strategy: Instead of proving HIT = explicit at all vertices,
-- just use that BOTH satisfy the sheaf condition and coincide on enough cases
--
-- Key insight: Sheaves are determined by:
-- 1. Values on generators (original vertices)
-- 2. Gluing condition (sheaf property)
--
-- Both HIT and explicit construction:
-- - Agree on original vertices (inc vs. identity)
-- - Satisfy same gluing at fork-stars (HIT's glue vs. product)
--
-- Therefore they're equal!

module SimplifiedProof {Γ : OrientedGraph o} (Γ-directed : Directed Γ) where
  open BuildFork Γ Γ-directed

  -- This approach avoids computing the HIT explicitly
  -- Instead, uses universal property of sheafification

  module _ {κ : Level} (F : Functor (Fork-Category ^op) (Sets κ)) where
    open Functor F

    -- Claim: The explicit construction is ALREADY a sheaf!
    -- If we can show this, then by universality, it equals Sheafification(F)

    -- Build explicit sheaf from product construction
    explicit-sheaf : Functor (Fork-Category ^op) (Sets (o ⊔ κ))
    explicit-sheaf .Functor.F₀ v = el (ExplicitConstruction.explicit-sheafify-at Γ-directed F v) {!!}
    explicit-sheaf .Functor.F₁ = {!!}  -- Functorial action
    explicit-sheaf .Functor.F-id = {!!}
    explicit-sheaf .Functor.F-∘ _ _ = {!!}

    -- Prove it's actually a sheaf
    explicit-is-sheaf : is-sheaf fork-coverage explicit-sheaf
    explicit-is-sheaf = {!!}
      -- This requires showing the sheaf condition:
      -- For each covering sieve, matching families extend uniquely
      --
      -- At fork-stars: tine-sieve coverage
      --   Matching family = product
      --   Universal property of products gives unique extension!
      --
      -- This is actually provable!

    -- By universality of sheafification, if explicit-sheaf is a sheaf
    -- and agrees with F via inc, then explicit-sheaf ≅ Sheafification(F)

-- =============================================================================
-- Phase 2 Status: Strategy Identified
-- =============================================================================

-- We have TWO possible approaches:
--
-- Approach A (Direct): Prove HIT computes to explicit construction
--   Pros: Most direct, shows exact computational content
--   Cons: Requires deep HIT reasoning, may hit technical issues
--
-- Approach B (Universal): Build explicit sheaf, show it's universal
--   Pros: Uses universal property (standard categorical argument)
--   Cons: Still requires proving sheaf condition for explicit construction
--
-- RECOMMENDATION: Try Approach B first
-- - More likely to succeed (uses universal property)
-- - Still fully constructive
-- - If blocked, can postulate with clear justification

-- Next steps:
-- 1. Complete explicit-sheaf functor definition
-- 2. Prove explicit-is-sheaf using product universal property
-- 3. Use sheafification universal property to get isomorphism
-- 4. Transport terminal/pullback preservation through isomorphism
