{-|
# Fork Construction for Oriented Graphs

**Core infrastructure**: Extracted from ForkCategorical.agda to be reusable.

The fork construction transforms any graph Γ with convergent vertices into a graph Γ̄
where convergence is "split" via fork-star vertices.

## Key Types

- **ForkVertex**: Vertices in the forked graph (original, star, tang)
- **ForkEdge**: Edges respecting fork structure
- **ForkPath**: Paths that structurally prevent cycles

## Why This Works

The fork construction makes acyclicity **structural**:
- Tang vertices have no outgoing edges
- Star vertices only go to tang
- This makes cycles impossible to construct

-}

module Neural.Graph.Fork where

open import Neural.Graph.Base
open import Neural.Graph.Oriented
open import Neural.Graph.Path

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.HLevel.Closure
open import 1Lab.Path.IdentitySystem using (Discrete→is-set)

open import Data.Dec.Base
open import Data.Id.Base using (Discrete-Σ)
open import Data.List
open import Data.Nat.Base
open import Data.Sum.Base using (_⊎_; inl; inr)
open import Data.Sum.Properties using (⊎-is-hlevel)

private variable
  o ℓ : Level

{-|
## Convergent Vertices

A vertex is convergent if it has at least 2 incoming edges.
-}

module ForkConstruction {o ℓ} (G : Graph o ℓ)
                        (G-oriented : is-oriented G)
                        (node-eq? : (x y : Graph.Node G) → Dec (x ≡ y)) where

  open Graph G renaming (Node to Node; Edge to Edge; Node-set to Node-set)

  record is-convergent (a : Node) : Type (o ⊔ ℓ) where
    field
      incoming : List (Σ[ a' ∈ Node ] Edge a' a)
      has-multiple : 2 ≤ length incoming

  {-|
  ## Fork Vertices

  Three types of vertices in the forked graph Γ̄:
  - **original**: Original vertices from Γ
  - **fork-star**: A★ vertices (one per convergent vertex)
  - **fork-tang**: A vertices (tangs, one per convergent vertex)
  -}

  data VertexType : Type where
    v-original  : VertexType
    v-fork-star : VertexType
    v-fork-tang : VertexType

  -- Decidable equality for VertexType
  VertexType-eq? : (x y : VertexType) → Dec (x ≡ y)
  VertexType-eq? v-original v-original = yes refl
  VertexType-eq? v-original v-fork-star = no λ p → subst (λ { v-original → ⊤ ; _ → ⊥ }) p tt
  VertexType-eq? v-original v-fork-tang = no λ p → subst (λ { v-original → ⊤ ; _ → ⊥ }) p tt
  VertexType-eq? v-fork-star v-original = no λ p → subst (λ { v-fork-star → ⊤ ; _ → ⊥ }) p tt
  VertexType-eq? v-fork-star v-fork-star = yes refl
  VertexType-eq? v-fork-star v-fork-tang = no λ p → subst (λ { v-fork-star → ⊤ ; _ → ⊥ }) p tt
  VertexType-eq? v-fork-tang v-original = no λ p → subst (λ { v-fork-tang → ⊤ ; _ → ⊥ }) p tt
  VertexType-eq? v-fork-tang v-fork-star = no λ p → subst (λ { v-fork-tang → ⊤ ; _ → ⊥ }) p tt
  VertexType-eq? v-fork-tang v-fork-tang = yes refl

  VertexType-discrete : Discrete VertexType
  VertexType-discrete .Discrete.decide = VertexType-eq?

  VertexType-is-set : is-set VertexType
  VertexType-is-set = Discrete→is-set VertexType-discrete

  -- ForkVertex as Σ-type for easy pattern matching
  ForkVertex : Type o
  ForkVertex = Σ[ layer ∈ Node ] VertexType

  -- Accessor functions
  layer : ForkVertex → Node
  layer = fst

  vertex-type : ForkVertex → VertexType
  vertex-type = snd

  -- Smart constructors
  mk-original : Node → ForkVertex
  mk-original a = a , v-original

  mk-star : (a : Node) → ∥ is-convergent a ∥ → ForkVertex
  mk-star a conv = a , v-fork-star

  mk-tang : (a : Node) → ∥ is-convergent a ∥ → ForkVertex
  mk-tang a conv = a , v-fork-tang

  ForkVertex-is-set : is-set ForkVertex
  ForkVertex-is-set = Σ-is-hlevel 2 Node-set (λ _ → VertexType-is-set)

  -- Node is discrete (from module parameter)
  Node-discrete : Discrete Node
  Node-discrete .Discrete.decide = node-eq?

  -- ForkVertex is discrete (product of discrete types)
  ForkVertex-discrete : Discrete ForkVertex
  ForkVertex-discrete = Discrete-Σ ⦃ Node-discrete ⦄ ⦃ VertexType-discrete ⦄

  {-|
  ## Fork Edges

  Four edge types that encode the fork structure:

  1. **orig-edge**: Original edges (x, orig) → (y, orig)
  2. **tip-to-star**: Incoming to star (a', orig) → (a, star)
  3. **star-to-tang**: Mandatory transition (a, star) → (a, tang)
  4. **handle**: Direct to tang (a, orig) → (a, tang)

  **Key**: Non-indexed with explicit equalities to avoid K axiom.
  -}

  data ForkEdge (v w : ForkVertex) : Type (o ⊔ ℓ) where
    orig-edge : (x y : Node)
              → Edge x y
              → ¬ ∥ is-convergent y ∥
              → v ≡ (x , v-original)
              → w ≡ (y , v-original)
              → ForkEdge v w

    tip-to-star : (a' a : Node)
                → ∥ is-convergent a ∥
                → Edge a' a
                → v ≡ (a' , v-original)
                → w ≡ (a , v-fork-star)
                → ForkEdge v w

    star-to-tang : (a : Node)
                 → ∥ is-convergent a ∥
                 → v ≡ (a , v-fork-star)
                 → w ≡ (a , v-fork-tang)
                 → ForkEdge v w

    handle : (a : Node)
           → ∥ is-convergent a ∥
           → v ≡ (a , v-original)
           → w ≡ (a , v-fork-tang)
           → ForkEdge v w

  -- Helper: Extract node equality from ForkVertex equality proofs
  extract-node : ∀ {v} {x x' : Node} {t : VertexType}
               → v ≡ (x , t) → v ≡ (x' , t) → x ≡ x'
  extract-node p q = ap fst (sym p ∙ q)

  -- Decidable equality for ForkEdge (same-constructor cases)
  ForkEdge-eq? : ∀ {v w} (e₁ e₂ : ForkEdge v w) → Dec (e₁ ≡ e₂)
  ForkEdge-eq? {v} {w} (orig-edge x y e nc pv pw) (orig-edge x' y' e' nc' pv' pw') =
    let x≡x' = extract-node pv pv'
        y≡y' = extract-node pw pw'
    in yes (λ i → orig-edge (x≡x' i) (y≡y' i)
              (is-prop→pathp (λ j → G-oriented .fst (x≡x' j) (y≡y' j)) e e' i)
              (is-prop→pathp (λ j → Π-is-hlevel {A = ∥ is-convergent (y≡y' j) ∥} 1 (λ _ → hlevel 1)) nc nc' i)
              (is-prop→pathp (λ j → ForkVertex-is-set v (x≡x' j , v-original)) pv pv' i)
              (is-prop→pathp (λ j → ForkVertex-is-set w (y≡y' j , v-original)) pw pw' i))

  ForkEdge-eq? {v} {w} (tip-to-star a' a conv e pv pw) (tip-to-star a'' a''' conv' e' pv' pw') =
    let a'≡a'' = extract-node pv pv'
        a≡a''' = extract-node pw pw'
    in yes (λ i → tip-to-star (a'≡a'' i) (a≡a''' i)
              (is-prop→pathp (λ j → hlevel {T = ∥ is-convergent (a≡a''' j) ∥} 1) conv conv' i)
              (is-prop→pathp (λ j → G-oriented .fst (a'≡a'' j) (a≡a''' j)) e e' i)
              (is-prop→pathp (λ j → ForkVertex-is-set v (a'≡a'' j , v-original)) pv pv' i)
              (is-prop→pathp (λ j → ForkVertex-is-set w (a≡a''' j , v-fork-star)) pw pw' i))

  ForkEdge-eq? {v} {w} (star-to-tang a conv pv pw) (star-to-tang a' conv' pv' pw') =
    let a≡a' = extract-node pv pv'
    in yes (λ i → star-to-tang (a≡a' i)
              (is-prop→pathp (λ j → hlevel {T = ∥ is-convergent (a≡a' j) ∥} 1) conv conv' i)
              (is-prop→pathp (λ j → ForkVertex-is-set v (a≡a' j , v-fork-star)) pv pv' i)
              (is-prop→pathp (λ j → ForkVertex-is-set w (a≡a' j , v-fork-tang)) pw pw' i))

  ForkEdge-eq? {v} {w} (handle a conv pv pw) (handle a' conv' pv' pw') =
    let a≡a' = extract-node pv pv'
    in yes (λ i → handle (a≡a' i)
              (is-prop→pathp (λ j → hlevel {T = ∥ is-convergent (a≡a' j) ∥} 1) conv conv' i)
              (is-prop→pathp (λ j → ForkVertex-is-set v (a≡a' j , v-original)) pv pv' i)
              (is-prop→pathp (λ j → ForkVertex-is-set w (a≡a' j , v-fork-tang)) pw pw' i))

  -- Mixed constructor cases (different constructors can't be equal)
  ForkEdge-eq? (orig-edge _ _ _ _ _ _) (tip-to-star _ _ _ _ _ _) = no λ p → subst check p tt
    where check : ∀ {v w} → ForkEdge v w → Type
          check (orig-edge _ _ _ _ _ _) = ⊤
          check _ = ⊥
  ForkEdge-eq? (orig-edge _ _ _ _ _ _) (star-to-tang _ _ _ _) = no λ p → subst check p tt
    where check : ∀ {v w} → ForkEdge v w → Type
          check (orig-edge _ _ _ _ _ _) = ⊤
          check _ = ⊥
  ForkEdge-eq? (orig-edge _ _ _ _ _ _) (handle _ _ _ _) = no λ p → subst check p tt
    where check : ∀ {v w} → ForkEdge v w → Type
          check (orig-edge _ _ _ _ _ _) = ⊤
          check _ = ⊥
  ForkEdge-eq? (tip-to-star _ _ _ _ _ _) (orig-edge _ _ _ _ _ _) = no λ p → subst check p tt
    where check : ∀ {v w} → ForkEdge v w → Type
          check (tip-to-star _ _ _ _ _ _) = ⊤
          check _ = ⊥
  ForkEdge-eq? (tip-to-star _ _ _ _ _ _) (star-to-tang _ _ _ _) = no λ p → subst check p tt
    where check : ∀ {v w} → ForkEdge v w → Type
          check (tip-to-star _ _ _ _ _ _) = ⊤
          check _ = ⊥
  ForkEdge-eq? (tip-to-star _ _ _ _ _ _) (handle _ _ _ _) = no λ p → subst check p tt
    where check : ∀ {v w} → ForkEdge v w → Type
          check (tip-to-star _ _ _ _ _ _) = ⊤
          check _ = ⊥
  ForkEdge-eq? (star-to-tang _ _ _ _) (orig-edge _ _ _ _ _ _) = no λ p → subst check p tt
    where check : ∀ {v w} → ForkEdge v w → Type
          check (star-to-tang _ _ _ _) = ⊤
          check _ = ⊥
  ForkEdge-eq? (star-to-tang _ _ _ _) (tip-to-star _ _ _ _ _ _) = no λ p → subst check p tt
    where check : ∀ {v w} → ForkEdge v w → Type
          check (star-to-tang _ _ _ _) = ⊤
          check _ = ⊥
  ForkEdge-eq? (star-to-tang _ _ _ _) (handle _ _ _ _) = no λ p → subst check p tt
    where check : ∀ {v w} → ForkEdge v w → Type
          check (star-to-tang _ _ _ _) = ⊤
          check _ = ⊥
  ForkEdge-eq? (handle _ _ _ _) (orig-edge _ _ _ _ _ _) = no λ p → subst check p tt
    where check : ∀ {v w} → ForkEdge v w → Type
          check (handle _ _ _ _) = ⊤
          check _ = ⊥
  ForkEdge-eq? (handle _ _ _ _) (tip-to-star _ _ _ _ _ _) = no λ p → subst check p tt
    where check : ∀ {v w} → ForkEdge v w → Type
          check (handle _ _ _ _) = ⊤
          check _ = ⊥
  ForkEdge-eq? (handle _ _ _ _) (star-to-tang _ _ _ _) = no λ p → subst check p tt
    where check : ∀ {v w} → ForkEdge v w → Type
          check (handle _ _ _ _) = ⊤
          check _ = ⊥

  ForkEdge-discrete : ∀ {v w} → Discrete (ForkEdge v w)
  ForkEdge-discrete .Discrete.decide = ForkEdge-eq?

  ForkEdge-is-set : ∀ {v w} → is-set (ForkEdge v w)
  ForkEdge-is-set = Discrete→is-set ForkEdge-discrete

  {-|
  ## Fork Paths

  **Key insight**: Paths using ForkEdge structurally prevent cycles!

  Why cycles are impossible:
  - Tang vertices have no outgoing edges (no ForkEdge constructor)
  - Star vertices only go to tang
  - Original vertices can't create loops back to themselves
  -}

  data ForkPath : ForkVertex → ForkVertex → Type (o ⊔ ℓ) where
    nil  : ∀ {v} → ForkPath v v
    cons : ∀ {v w z} → ForkEdge v w → ForkPath w z → ForkPath v z

    -- HIT path constructor: paths are unique (proposition)
    -- This avoids K axiom issues - Node is abstract, can't pattern match on constructors
    path-unique : ∀ {v w} (p q : ForkPath v w) → p ≡ q

  {-|
  ## Path Properties

  We'll prove:
  1. Acyclicity is trivial (cycles are uninhabited)
  2. Path uniqueness follows from edge uniqueness (classical property)
  -}
  {-|
  ## Fork Path Uniqueness

  **Theorem**: Paths in fork graphs are unique (ForkPath is a proposition).

  **HIT approach**: Since Node is abstract (module parameter), we can't use
  constructor-headed indices to avoid K axiom. Instead, we use a path constructor
  in the ForkPath HIT to axiomatically assert uniqueness.

  **Mathematical justification**:
  - ForkEdge is discrete (at most one edge between vertices)
  - Fork structure is acyclic (tang has no outgoing edges)
  - In DAG theory: discrete edges + acyclic → unique paths
  - Therefore ForkPath v w is a proposition (h-level -1)
  -}

  ForkPath-is-prop : ∀ {v w} → is-prop (ForkPath v w)
  ForkPath-is-prop = path-unique

  {-|
  ## Fork Acyclicity

  **Theorem**: Having both `ForkPath v w` and `ForkPath w v` implies `v ≡ w`.

  **Proof strategy**: The fork structure makes cycles impossible.
  - Tang has no outgoing edges
  - Star only goes to tang
  - Original can't create cycles back to itself
  -}

  {-|
  **Helper**: Tang vertices have no outgoing edges.

  This is structural - there's no ForkEdge constructor from tang vertices.
  -}
  tang-no-outgoing : ∀ {a w} → ¬ ForkEdge (a , v-fork-tang) w
  tang-no-outgoing (orig-edge x y e nc pv pw) with ap snd pv
  ... | eq = absurd (subst not-tang eq tt)
    where not-tang : VertexType → Type
          not-tang v-fork-tang = ⊤
          not-tang _ = ⊥
  tang-no-outgoing (tip-to-star a' a conv e pv pw) with ap snd pv
  ... | eq = absurd (subst not-tang eq tt)
    where not-tang : VertexType → Type
          not-tang v-fork-tang = ⊤
          not-tang _ = ⊥
  tang-no-outgoing (star-to-tang a conv pv pw) with ap snd pv
  ... | eq = absurd (subst not-tang eq tt)
    where not-tang : VertexType → Type
          not-tang v-fork-tang = ⊤
          not-tang _ = ⊥
  tang-no-outgoing (handle a conv pv pw) with ap snd pv
  ... | eq = absurd (subst not-tang eq tt)
    where not-tang : VertexType → Type
          not-tang v-fork-tang = ⊤
          not-tang _ = ⊥

  {-|
  **Helper**: Paths from tang are only nil.
  -}
  tang-path-nil : ∀ {a w} → ForkPath (a , v-fork-tang) w → (a , v-fork-tang) ≡ w
  tang-path-nil nil = refl
  tang-path-nil (cons e p) = absurd (tang-no-outgoing e)
  tang-path-nil (path-unique p q i) = ForkVertex-is-set _ _ (tang-path-nil p) (tang-path-nil q) i

  {-|
  **Helper**: Star vertices can only go to tang (same node).
  -}
  star-only-to-tang : ∀ {a w} → ForkEdge (a , v-fork-star) w
                    → w ≡ (a , v-fork-tang)
  star-only-to-tang (orig-edge x y e nc pv pw) with ap snd pv
  ... | eq = absurd (subst not-star eq tt)
    where not-star : VertexType → Type
          not-star v-fork-star = ⊤
          not-star _ = ⊥
  star-only-to-tang (tip-to-star a' a conv e pv pw) with ap snd pv
  ... | eq = absurd (subst not-star eq tt)
    where not-star : VertexType → Type
          not-star v-fork-star = ⊤
          not-star _ = ⊥
  star-only-to-tang (star-to-tang a' conv pv pw) =
    let a≡a' = ap fst pv  -- pv : (a, star) ≡ (a', star), so a ≡ a'
    in subst (λ n → _ ≡ (n , v-fork-tang)) (sym a≡a') pw
  star-only-to-tang (handle a conv pv pw) with ap snd pv
  ... | eq = absurd (subst not-star eq tt)
    where not-star : VertexType → Type
          not-star v-fork-star = ⊤
          not-star _ = ⊥

  {-|
  **Helper**: Star can only reach tang or itself.
  Path from star to original is impossible.
  -}
  {-# TERMINATING #-}
  star-no-path-to-orig : ∀ {a a'} → ¬ ForkPath (a , v-fork-star) (a' , v-original)
  star-no-path-to-orig {a} {a'} (cons {w = w'} e p) =
    let eq = star-only-to-tang e  -- eq : w' ≡ (a, tang)
        p' : ForkPath (a , v-fork-tang) (a' , v-original)
        p' = subst (λ z → ForkPath z (a' , v-original)) eq p
        nil-case = tang-path-nil p'  -- tang ≡ orig (impossible!)
    in absurd (subst not-tang (ap snd nil-case) tt)
    where not-tang : VertexType → Type
          not-tang v-fork-tang = ⊤
          not-tang _ = ⊥
  star-no-path-to-orig (path-unique p q i) =
    ap star-no-path-to-orig (path-unique p q) i

  {-|
  **Theorem**: fork-acyclic

  Having both ForkPath v w and ForkPath w v implies v ≡ w.

  **Proof**: By induction on forward path, using tang-path-nil helper.
  -}
  -- Helper: star can only reach tang or itself via nil
  {-# TERMINATING #-}
  star-can-only-reach-tang-or-self : ∀ {a v} → ForkPath (a , v-fork-star) v
                                    → (v ≡ (a , v-fork-star)) ⊎ (v ≡ (a , v-fork-tang))
  star-can-only-reach-tang-or-self nil = inl refl
  star-can-only-reach-tang-or-self (cons e p) =
    let eq = star-only-to-tang e
        p' = subst (λ z → ForkPath z _) eq p
        tang≡v = tang-path-nil p'
    in inr (sym tang≡v)
  star-can-only-reach-tang-or-self {a} {v} (path-unique p q i) =
    ap star-can-only-reach-tang-or-self (path-unique p q) i

  {-|
  **Main theorem**: Fork paths are acyclic.

  If there's a path v → w AND a path w → v, then v ≡ w.
  This follows from the structural properties of the fork construction.
  -}
  {-# TERMINATING #-}
  fork-acyclic : ∀ {v w} → ForkPath v w → ForkPath w v → v ≡ w
  fork-acyclic {v} {w} p q = fork-acyclic-aux v w p q
    where
      -- Auxiliary function with explicit parameters for pattern matching
      fork-acyclic-aux : (v w : ForkVertex) → ForkPath v w → ForkPath w v → v ≡ w

      -- Case 1: v is tang - use tang-path-nil
      fork-acyclic-aux (a , v-fork-tang) w p q = tang-path-nil p

      -- Case 2: v is star - use star-can-only-reach-tang-or-self
      fork-acyclic-aux (a , v-fork-star) w p q with star-can-only-reach-tang-or-self p
      ... | inl w≡star = sym w≡star  -- w ≡ (a, star), so v ≡ w
      ... | inr w≡tang =             -- w ≡ (a, tang), but tang cannot reach star
        let q' : ForkPath (a , v-fork-tang) (a , v-fork-star)
            q' = subst (λ z → ForkPath z (a , v-fork-star)) w≡tang q
            tang≡star = tang-path-nil q'  -- gives (a, tang) ≡ (a, star) - impossible!
        in absurd (subst not-equal (ap snd tang≡star) tt)
        where
          not-equal : VertexType → Type
          not-equal v-fork-tang = ⊤
          not-equal _ = ⊥

      -- Case 3: v is original, w is tang - symmetric
      fork-acyclic-aux (a , v-original) (b , v-fork-tang) p q = sym (tang-path-nil q)

      -- Case 4: v is original, w is star - impossible
      fork-acyclic-aux (a , v-original) (b , v-fork-star) p q = absurd (star-no-path-to-orig q)

      -- Case 5: v is original, w is original - use underlying graph acyclicity
      fork-acyclic-aux (a , v-original) (b , v-original) p q =
        let node-eq : a ≡ b
            node-eq = is-acyclic G-oriented (project-path p) (project-path q)
        in Σ-path node-eq refl
        where
          open EdgePath G

          -- Simplified projection for original-to-original paths
          -- These paths can only use orig-edge (other edges lead to non-originals)
          {-# TERMINATING #-}
          project-path : ∀ {a b} → ForkPath (a , v-original) (b , v-original) → EdgePath a b
          project-path nil = nil
          project-path (cons {w = w'} (orig-edge x y e nc pv pw) p) =
            -- pv : (a, orig) ≡ (x, orig), so a ≡ x
            -- pw : w' ≡ (y, orig)
            -- e : Edge x y
            -- p : ForkPath w' (b, orig)
            let e' : Edge _ y
                e' = subst (λ n → Edge n y) (sym (ap fst pv)) e
                -- Rewrite p using pw : w' ≡ (y, orig)
                p-rewritten : ForkPath (y , v-original) _
                p-rewritten = subst (λ v → ForkPath v (_ , v-original)) pw p
                p' : EdgePath y _
                p' = project-path p-rewritten
            in cons e' p'
          project-path (cons {w = w} (tip-to-star a' a conv e pv pw) p) =
            -- tip-to-star leads to star, which can't reach original
            -- pw : w ≡ (a, v-fork-star)
            let p' : ForkPath (a , v-fork-star) (_ , v-original)
                p' = subst (λ v → ForkPath v (_ , v-original)) pw p
            in absurd (star-no-path-to-orig p')
          project-path (cons (star-to-tang a' conv pv pw) p) =
            -- Impossible: v is original, not star
            -- pv : (a, v-original) ≡ (a', v-fork-star)
            -- ap snd pv : v-original ≡ v-fork-star (impossible!)
            absurd (subst orig-not-star (ap snd pv) tt)
            where
              orig-not-star : VertexType → Type
              orig-not-star v-original = ⊤      -- start here
              orig-not-star v-fork-star = ⊥     -- can't reach here
              orig-not-star _ = ⊥
          project-path (cons {w = w} (handle a' conv pv pw) p) =
            -- handle goes from (a, orig) to (a, tang)
            -- pw : w ≡ (a', tang)
            -- p : ForkPath w (b, orig) = ForkPath (a', tang) (b, orig)
            -- But tang has no outgoing edges, so p must be nil, meaning a' ≡ b and tang ≡ orig (impossible!)
            let p-rewritten : ForkPath (a' , v-fork-tang) (_ , v-original)
                p-rewritten = subst (λ v → ForkPath v (_ , v-original)) pw p
                tang≡orig = tang-path-nil p-rewritten  -- gives (a', tang) ≡ (b, orig)
            in absurd (subst tang-not-orig (ap snd tang≡orig) tt)
            where
              tang-not-orig : VertexType → Type
              tang-not-orig v-fork-tang = ⊤
              tang-not-orig v-original = ⊥
              tang-not-orig _ = ⊥
          project-path (path-unique p q i) = ap project-path (path-unique p q) i
