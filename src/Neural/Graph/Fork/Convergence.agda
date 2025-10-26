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
open import 1Lab.HLevel.Closure
open import 1Lab.Path.Reasoning

open import Data.Dec.Base
open import Data.List
open import Data.Nat.Base
open import Data.Nat.Order

private variable
  o ℓ : Level

-- Helper module for scan lemmas (no module parameter needed)
module ConvergenceHelpers
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

  -- Helper lemma: If scan returns singleton [(s, e)], then any node n with edge to a
  -- must equal s (the unique source)
  scan-incoming-singleton-unique : ∀ (a : Node) (ns : List Node)
                                   (s : Node) (e : Edge s a) →
                                   scan-incoming a ns ≡ ((s , e) ∷ []) →
                                   ∀ (n : Node) → Edge n a → n ∈ ns → n ≡ s
  scan-incoming-singleton-unique a [] s e eq n e-n-a ()
  scan-incoming-singleton-unique a (m ∷ rest) s e eq n e-n-a n-in-list with edge? m a
  ... | yes e-m-a =
    -- Scan found edge: (m, e-m-a) :: scan-incoming a rest ≡ [(s, e)]
    -- So m ≡ s and scan-incoming a rest ≡ []
    let m≡s : m ≡ s
        m≡s = ap fst (cons-injective-head eq)
    in case n-in-list of λ where
      (here p) → Id≃path .fst p ∙ m≡s  -- n ≡ m ≡ s
      (there n-in-rest) →
        -- scan-incoming a rest must be [], but n ∈ rest with edge e-n-a
        -- This contradicts: if scan-incoming a rest ≡ [], no edges in rest
        absurd (scan-empty-no-edges (cons-injective-tail eq) n-in-rest)
    where
      cons-injective-head : ∀ {x y : Σ[ a' ∈ Node ] Edge a' a} {xs ys} →
                           x ∷ xs ≡ y ∷ ys → x ≡ y
      cons-injective-head p = ap (λ { (z ∷ _) → z ; [] → s , e }) p

      cons-injective-tail : ∀ {x y : Σ[ a' ∈ Node ] Edge a' a} {xs ys} →
                           x ∷ xs ≡ y ∷ ys → xs ≡ ys
      cons-injective-tail p = ap (λ { (_ ∷ zs) → zs ; [] → [] }) p

      cons≠nil : ∀ {x : Σ[ a' ∈ Node ] Edge a' a} {xs} → ¬ (x ∷ xs ≡ [])
      cons≠nil p = subst (λ { [] → ⊥ ; (_ ∷ _) → ⊤ }) p tt

      scan-empty-no-edges : scan-incoming a rest ≡ [] → n ∈ rest → ⊥
      scan-empty-no-edges = go rest
        where
          go : ∀ xs → scan-incoming a xs ≡ [] → n ∈ xs → ⊥
          go [] _ ()
          go (x ∷ xs') empty-eq' n-in-xs with edge? x a
          ... | yes e-x-a =
            -- scan-incoming would return (x, e-x-a) :: ..., contradicting empty
            absurd (cons≠nil empty-eq')
          ... | no ¬e-x-a = case n-in-xs of λ where
            (here p') → absurd (¬e-x-a (subst (λ y → Edge y a) (Id≃path .fst p') e-n-a))
            (there n-in-xs') → go xs' empty-eq' n-in-xs'
  ... | no ¬e-m-a =
    -- m has no edge, scan continues: scan-incoming a rest ≡ [(s, e)]
    case n-in-list of λ where
      (here p) →
        -- n ≡ m via p, but m has no edge to a, contradicting e-n-a : Edge n a
        absurd (¬e-m-a (subst (λ x → Edge x a) (Id≃path .fst p) e-n-a))
      (there n-in-rest) →
        -- n is in rest, apply IH
        scan-incoming-singleton-unique a rest s e eq n e-n-a n-in-rest

  -- Helper lemma: If scan returns 2-element list where both sources are equal,
  -- then any source equals that value
  scan-two-equal-sources : ∀ (a : Node) (ns : List Node)
                           (s₁ s₂ : Node) (e₁ : Edge s₁ a) (e₂ : Edge s₂ a) →
                           scan-incoming a ns ≡ ((s₁ , e₁) ∷ (s₂ , e₂) ∷ []) →
                           s₁ ≡ s₂ →
                           ∀ (n : Node) → Edge n a → n ∈ ns → n ≡ s₁
  scan-two-equal-sources a [] s₁ s₂ e₁ e₂ eq s₁≡s₂ n e-n-a ()
  scan-two-equal-sources a (m ∷ rest) s₁ s₂ e₁ e₂ eq s₁≡s₂ n e-n-a n-in-list with edge? m a
  ... | yes e-m-a =
    -- Scan found edge at head: (m, e-m-a) :: scan-incoming a rest
    -- This equals [(s₁, e₁), (s₂, e₂)], so m ≡ s₁ and scan-incoming a rest ≡ [(s₂, e₂)]
    let m≡s₁ : m ≡ s₁
        m≡s₁ = ap fst (cons-injective-head eq)
        rest-is-singleton : scan-incoming a rest ≡ ((s₂ , e₂) ∷ [])
        rest-is-singleton = cons-injective-tail eq
    in case n-in-list of λ where
      (here p) → Id≃path .fst p ∙ m≡s₁  -- n ≡ m ≡ s₁
      (there n-in-rest) →
        -- n is in rest, and rest scans to singleton [(s₂, e₂)]
        -- So n ≡ s₂ ≡ s₁
        scan-incoming-singleton-unique a rest s₂ e₂ rest-is-singleton n e-n-a n-in-rest ∙ sym s₁≡s₂
    where
      cons-injective-head : ∀ {x y : Σ[ a' ∈ Node ] Edge a' a} {xs ys} →
                           x ∷ xs ≡ y ∷ ys → x ≡ y
      cons-injective-head p = ap (λ { (z ∷ _) → z ; [] → s₁ , e₁ }) p

      cons-injective-tail : ∀ {x y : Σ[ a' ∈ Node ] Edge a' a} {xs ys} →
                           x ∷ xs ≡ y ∷ ys → xs ≡ ys
      cons-injective-tail p = ap (λ { (_ ∷ zs) → zs ; [] → [] }) p
  ... | no ¬e-m-a =
    -- m has no edge, scan continues in rest
    case n-in-list of λ where
      (here p) →
        -- n ≡ m via p, but m has no edge to a, contradicting e-n-a : Edge n a
        absurd (¬e-m-a (subst (λ x → Edge x a) (Id≃path .fst p) e-n-a))
      (there n-in-rest) →
        -- n is in rest, apply IH
        scan-two-equal-sources a rest s₁ s₂ e₁ e₂ eq s₁≡s₂ n e-n-a n-in-rest

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
  open ConvergenceHelpers G G-oriented nodes nodes-complete edge? node-eq?

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

  -- Helper lemma: If scan returns empty, no nodes in the list have edges to a
  scan-incoming-empty-no-edges : ∀ (a : Node) (ns : List Node) →
                                 scan-incoming a ns ≡ [] →
                                 ∀ (n : Node) → Edge n a → n ∈ ns → ⊥
  scan-incoming-empty-no-edges a [] eq n e ()
  scan-incoming-empty-no-edges a (m ∷ rest) eq n e (here p) with edge? m a
  ... | yes e-m-a =
    -- scan-incoming a (m ∷ rest) = (m, e-m-a) :: scan-incoming a rest
    -- But eq says this equals [], contradiction!
    absurd (cons≠nil eq)
    where
      cons≠nil : ∀ {x : Σ[ a' ∈ Node ] Edge a' a} {xs} → ¬ (x ∷ xs ≡ [])
      cons≠nil p = subst (λ { [] → ⊥ ; (_ ∷ _) → ⊤ }) p tt
  ... | no ¬e-m-a =
    -- m has no edge, so n ≡ m via p means n has no edge either
    absurd (¬e-m-a (subst (λ x → Edge x a) (Id≃path .fst p) e))
  scan-incoming-empty-no-edges a (m ∷ rest) eq n e (there n-in-rest) with edge? m a
  ... | yes e-m-a =
    -- scan-incoming a (m ∷ rest) = (m, e-m-a) :: scan-incoming a rest
    -- But eq says this equals [], contradiction!
    absurd (cons≠nil eq)
    where
      cons≠nil : ∀ {x : Σ[ a' ∈ Node ] Edge a' a} {xs} → ¬ (x ∷ xs ≡ [])
      cons≠nil p = subst (λ { [] → ⊥ ; (_ ∷ _) → ⊤ }) p tt
  ... | no ¬e-m-a =
    -- m has no edge, scan continues in rest
    -- eq : scan-incoming a (m ∷ rest) ≡ []
    -- Since edge? m a = no, this reduces to scan-incoming a rest ≡ []
    scan-incoming-empty-no-edges a rest eq n e n-in-rest

  -- Check if a node is convergent
  -- Strategy: Scan incoming-edges list to find 2 edges from distinct sources
  {-# TERMINATING #-}
  is-convergent? : (a : Node) → Dec (∥ is-convergent a ∥)
  is-convergent? a = go
    where
      -- Helper: membership in empty list is impossible
      ∈-nil-absurd : ∀ {x : Σ[ a' ∈ Node ] Edge a' a} → x ∈ [] → ⊥
      ∈-nil-absurd ()

      go : Dec (∥ is-convergent a ∥)
      go with incoming-edges a in eq
      ... | [] = no (∥-∥-elim (λ _ → hlevel 1) λ c →
            -- If witness c exists, it has edge₁ : Edge (c.source₁) a
            -- By incoming-edges-complete, this edge should appear in incoming-edges a
            -- But incoming-edges a is [], so membership is impossible
            ∥-∥-rec (hlevel 1)
                    (λ (x , mem) → ∈-nil-absurd (subst (x ∈_) (Id≃path .fst eq) mem))
                    (incoming-edges-complete a (c .is-convergent.source₁) (c .is-convergent.edge₁)))

      ... | (s₁ , e₁) ∷ [] = no (∥-∥-elim (λ _ → hlevel 1) λ c →
            -- Use scan-incoming-singleton-unique: singleton means s₁ is the ONLY source
            -- So c.source₁ ≡ s₁ and c.source₂ ≡ s₁, contradicting c.distinct
            c .is-convergent.distinct (unique (c .is-convergent.source₁) (c .is-convergent.edge₁)
                                       ∙ sym (unique (c .is-convergent.source₂) (c .is-convergent.edge₂))))
            where
              open import 1Lab.Path.Reasoning

              unique : ∀ (n : Node) → Edge n a → n ≡ s₁
              unique n e = scan-incoming-singleton-unique a nodes s₁ e₁ (Id≃path .fst eq) n e (nodes-complete n)

      ... | (s₁ , e₁) ∷ (s₂ , e₂) ∷ rest with node-eq? s₁ s₂
      ... | no s₁≠s₂ =
        yes (inc (record
          { source₁ = s₁
          ; source₂ = s₂
          ; distinct = s₁≠s₂
          ; edge₁ = e₁
          ; edge₂ = e₂
          }))
      ... | yes s₁≡s₂ with rest
      ... | [] = no (∥-∥-elim (λ _ → hlevel 1) λ c →
        -- rest = [], so scan found exactly [(s₁, e₁), (s₂, e₂)] with s₁ ≡ s₂
        let scan-eq : scan-incoming a nodes ≡ ((s₁ , e₁) ∷ (s₂ , e₂) ∷ [])
            scan-eq = Id≃path .fst eq
        in is-convergent.distinct c
          (scan-two-equal-sources a nodes s₁ s₂ e₁ e₂ scan-eq s₁≡s₂
             (is-convergent.source₁ c) (is-convergent.edge₁ c)
             (nodes-complete (is-convergent.source₁ c))
           ∙ sym (scan-two-equal-sources a nodes s₁ s₂ e₁ e₂ scan-eq s₁≡s₂
                   (is-convergent.source₂ c) (is-convergent.edge₂ c)
                   (nodes-complete (is-convergent.source₂ c)))))
      ... | (s₃ , e₃) ∷ more = find-distinct s₁ e₁ ((s₃ , e₃) ∷ more)
        where
          -- s₁ ≡ s₂, so look for a source distinct from s₁ in the rest
          find-distinct : (s : Node) → Edge s a → List (Σ[ a' ∈ Node ] Edge a' a) → Dec (∥ is-convergent a ∥)
          find-distinct s e [] = is-convergent? a  -- Shouldn't happen; re-scan
          find-distinct s e ((s' , e') ∷ more) with node-eq? s s'
          ... | no s≠s' =
            yes (inc (record
              { source₁ = s
              ; source₂ = s'
              ; distinct = s≠s'
              ; edge₁ = e
              ; edge₂ = e'
              }))
          ... | yes s≡s' = find-distinct s e more
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

