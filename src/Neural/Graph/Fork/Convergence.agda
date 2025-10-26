{-|
# Convergence Detection for Fork Construction

This module handles detecting convergent vertices (vertices with ≥2 incoming edges)
and provides the decision procedure for fork construction.

## From the Paper (Belfiore & Bennequin 2022, Section 1.3)

> "Only the convergent multiplicity in Γ gives rise to forks,
> not the divergent one."
-}

module Neural.Graph.Fork.Convergence where

open import Neural.Graph.Base
open import Neural.Graph.Oriented
open import Neural.Graph.Fork.Fork

open import 1Lab.Prelude

open import Data.Dec.Base
open import Data.List
open import Data.Nat.Base
open import Data.Nat.Order

private variable
  o ℓ : Level

module ConvergenceDetection
  (G : Graph o ℓ)
  (G-oriented : is-oriented G)
  (nodes : List (Graph.Node G))
  (nodes-complete : ∀ (n : Graph.Node G) → n ∈ nodes)
  (edge? : ∀ (x y : Graph.Node G) → Dec (Graph.Edge G x y))
  (node-eq? : ∀ (x y : Graph.Node G) → Dec (x ≡ y))
  where
  open Graph G
  open ForkConstruction G G-oriented node-eq?

  -- Helper to collect incoming edges by scanning a node list
  scan-incoming : (a : Node) → List Node → List (Σ[ a' ∈ Node ] Edge a' a)
  scan-incoming a [] = []
  scan-incoming a (n ∷ rest) with edge? n a
  ... | yes e = (n , e) ∷ scan-incoming a rest
  ... | no ¬e = scan-incoming a rest

  -- Helper: Collect incoming edges to a node
  incoming-edges : (a : Node) → List (Σ[ a' ∈ Node ] Edge a' a)
  incoming-edges a = scan-incoming a nodes

  -- Helper: 2 ≤ suc (suc n) for any n
  2≤suc-suc : ∀ (n : Nat) → 2 ≤ suc (suc n)
  2≤suc-suc n = s≤s (s≤s 0≤x)

  -- Absurdity: suc (suc n) ≤ 0 is impossible
  suc-suc-≰-zero : ∀ {n} → ¬ (suc (suc n) ≤ 0)
  suc-suc-≰-zero ()

  -- Absurdity: suc (suc n) ≤ 1 is impossible
  suc-suc-≰-one : ∀ {n} → ¬ (suc (suc n) ≤ 1)
  suc-suc-≰-one (s≤s ())

  -- Helper: if there's an edge from a' to a, and a' is in nodes list ns,
  -- then scan-incoming will find it
  scan-finds : ∀ (a a' : Node) (ns : List Node) → Edge a' a → a' ∈ ns →
               ∃[ x ∈ (Σ[ a'' ∈ Node ] Edge a'' a) ] (x ∈ scan-incoming a ns)
  scan-finds a a' [] e ()
  scan-finds a a' (n ∷ rest) e (here p) with edge? n a
  ... | yes e' = inc ((n , e') , here reflᵢ)
  ... | no ¬e = absurd (¬e (subst (λ x → Edge x a) (Id≃path .fst p) e))
  scan-finds a a' (n ∷ rest) e (there mem) with edge? n a
  ... | yes e' = ∥-∥-map (λ (x , x∈) → x , there x∈) (scan-finds a a' rest e mem)
  ... | no ¬e = scan-finds a a' rest e mem

  -- Corollary: incoming-edges finds all edges (using nodes-complete)
  incoming-edges-complete : ∀ (a a' : Node) → Edge a' a →
                            ∃[ x ∈ (Σ[ a'' ∈ Node ] Edge a'' a) ] (x ∈ incoming-edges a)
  incoming-edges-complete a a' e = scan-finds a a' nodes e (nodes-complete a')



  -- Check if a node is convergent
  -- Strategy: Scan incoming-edges list to find 2 edges from distinct sources
  is-convergent? : (a : Node) → Dec (∥ is-convergent a ∥)
  is-convergent? a = find-two-distinct (incoming-edges a)
    where
      -- Try to find two edges from distinct sources in the scanned list
      find-two-distinct : List (Σ[ a' ∈ Node ] Edge a' a) → Dec (∥ is-convergent a ∥)

      -- Empty list: no edges found, can't be convergent
      find-two-distinct [] = no λ { (inc c) →
        -- If witness c exists, it has edge₁ : Edge (c.source₁) a
        -- By incoming-edges-complete, this edge should appear in incoming-edges a
        -- But incoming-edges a is [], so membership is impossible
        ∥-∥-rec (hlevel 1) (λ (_ , mem) → absurd-empty mem)
                (incoming-edges-complete a (c .is-convergent.source₁) (c .is-convergent.edge₁)) }
        where absurd-empty : ∀ {x} → x ∈ [] → ⊥
              absurd-empty ()

      -- Singleton list: only 1 source found, can't have 2 distinct sources
      find-two-distinct ((s₁ , e₁) ∷ []) = no λ { (inc c) →
        -- If c claims two distinct sources both have edges to a,
        -- then by scan completeness, both must appear in incoming-edges a = [(s₁, e₁)]
        -- But a singleton can contain only pairs with one source value s₁
        -- Therefore both c.source₁ and c.source₂ must equal s₁, contradicting distinctness
        ∥-∥-rec (hlevel 1) (λ (x₁ , mem₁) →
          ∥-∥-rec (hlevel 1) (λ (x₂ , mem₂) →
            -- Extract s₁ from both memberships and show c.source₁ = s₁ = c.source₂
            c .is-convergent.distinct (source-via-singleton (c .is-convergent.source₁) (c .is-convergent.edge₁)
                                        ∙ sym (source-via-singleton (c .is-convergent.source₂) (c .is-convergent.edge₂))))
          (incoming-edges-complete a (c .is-convergent.source₂) (c .is-convergent.edge₂)))
        (incoming-edges-complete a (c .is-convergent.source₁) (c .is-convergent.edge₁)) }
        where
          open import 1Lab.Path.Reasoning

          -- Key lemma: If a node n has an edge to a, then n ≡ s₁ (the unique source in singleton)
          -- Strategy: Classical graphs have ≤1 edge between any two vertices
          -- So if both n and s₁ have edges to a, and only one source appears in the scan, they must be equal
          -- This is essentially: singleton list ⇒ unique source ⇒ n = s₁
          source-via-singleton : ∀ (n : Node) → Edge n a → n ≡ s₁
          source-via-singleton n e-n-a =
            -- Key insight: scan-incoming checks every node in `nodes` for edges to `a`
            -- If it returns a singleton [(s₁, e₁)], then s₁ is the ONLY node with an edge to `a`
            -- Since n also has an edge to a (via e-n-a), we must have n = s₁
            -- Proof: By contradiction using node-eq?
            case node-eq? n s₁ of λ where
              (yes n≡s₁) → n≡s₁
              (no n≠s₁) →
                -- n ≠ s₁, but both have edges to a
                -- By completeness, both appear in incoming-edges a
                -- But incoming-edges a = [(s₁, e₁)] is a singleton
                -- Contradiction: can't have two distinct sources in a singleton!
                absurd (singleton-cant-have-two-distinct-sources n s₁ n≠s₁ e-n-a e₁)
            where
              singleton-cant-have-two-distinct-sources :
                ∀ (n₁ n₂ : Node) → (n₁ ≡ n₂ → ⊥) → Edge n₁ a → Edge n₂ a → ⊥
              singleton-cant-have-two-distinct-sources n₁ n₂ n₁≠n₂ e₁' e₂' =
                -- Both edges must appear in the scan
                -- TODO: Prove list can't contain two distinct sources
                {!!}

      -- Two or more elements: check if first two sources are distinct
      find-two-distinct ((s₁ , e₁) ∷ (s₂ , e₂) ∷ rest) with node-eq? s₁ s₂
      ... | yes _ =
        -- Same source, skip one and try with rest
        -- Note: scan-incoming shouldn't produce duplicates in practice
        -- If all sources are the same, this will eventually hit the singleton/empty case
        find-two-distinct ((s₁ , e₁) ∷ rest)
      ... | no s₁≠s₂ =
        -- Found two distinct sources! Construct the witness
        yes (inc (record
          { source₁ = s₁
          ; source₂ = s₂
          ; distinct = s₁≠s₂
          ; edge₁ = e₁
          ; edge₂ = e₂
          }))
  -- For each original edge x → y in Γ, produce edges in Γ̄
  fork-edges : (x y : Node) → Edge x y → List (Σ[ v ∈ ForkVertex ] Σ[ w ∈ ForkVertex ] ForkEdge v w)
  fork-edges x y e with is-convergent? y
  ... | yes conv-y =
    -- Target y is convergent: x → A★, A★ → A, y → A
    ((x , v-original) , (y , v-fork-star) , tip-to-star x y conv-y e refl refl) ∷
    ((y , v-fork-star) , (y , v-fork-tang) , star-to-tang y conv-y refl refl) ∷
    ((y , v-original) , (y , v-fork-tang) , handle y conv-y refl refl) ∷ []
  ... | no ¬conv-y =
    -- Target y not convergent: keep as original edge
    ((x , v-original) , (y , v-original) , orig-edge x y e ¬conv-y refl refl) ∷ []

