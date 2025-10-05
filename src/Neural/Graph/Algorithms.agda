{-# OPTIONS --no-import-sorts #-}

module Neural.Graph.Algorithms where

open import 1Lab.Prelude

open import Data.Bool.Base using (Bool; true; false; if_then_else_)
open import Data.Maybe.Base using (Maybe; just; nothing)
open import Data.Nat.Base using (Nat; zero; suc; _+_; _*_)
open import Data.Fin.Base using (Fin; fzero; fsuc)
open import Data.List.Base using (List; []; _∷_; filter; concat; _++_; reverse)
open import Data.Sum.Base using (_⊎_; inl; inr)
open import Data.Dec.Base using (Dec; yes; no)

open import Neural.Base using (DirectedGraph; vertices; edges; source; target)

private variable
  n m : Nat

-- Basic utilities ------------------------------------------------------------

-- Finite enumeration of Fin n
fin-list : (k : Nat) → List (Fin k)
fin-list zero = []
fin-list (suc k) = fzero ∷ map fsuc (fin-list k)

-- Boolean negation
not : Bool → Bool
not true = false
not false = true

-- List length (local, to avoid clashes)
length : ∀ {A : Type} → List A → Nat
length [] = 0
length (_ ∷ xs) = suc (length xs)

-- Any predicate holds in list
any : ∀ {A : Type} → (A → Bool) → List A → Bool
any p [] = false
any p (x ∷ xs) = if p x then true else any p xs

-- Remove elements satisfying predicate
removeIf : ∀ {A : Type} → (A → Bool) → List A → List A
removeIf p [] = []
removeIf p (x ∷ xs) = if p x then removeIf p xs else x ∷ removeIf p xs

-- Membership with decidable equality, returning Bool
memberBy : ∀ {A : Type} → (A → A → Bool) → A → List A → Bool
memberBy _≟_ x [] = false
memberBy _≟_ x (y ∷ ys) = if _≟_ x y then true else memberBy _≟_ x ys

-- Equality on Fin as Bool via Discrete-Fin
postulate
  eqFin? : ∀ {k : Nat} → Fin k → Fin k → Bool

-- Graph views ---------------------------------------------------------------

-- List all vertices of a directed graph
verticesList : (G : DirectedGraph) → List (Fin (vertices G))
verticesList G = fin-list (vertices G)

-- List all edges of a directed graph
edgesList : (G : DirectedGraph) → List (Fin (edges G))
edgesList G = fin-list (edges G)

-- Out-neighbors of a vertex
neighborsOut : (G : DirectedGraph) → Fin (vertices G) → List (Fin (vertices G))
neighborsOut G v =
  map (λ e → target G e)
      (filter (λ e → eqFin? (source G e) v) (edgesList G))

-- In-neighbors of a vertex
neighborsIn : (G : DirectedGraph) → Fin (vertices G) → List (Fin (vertices G))
neighborsIn G v =
  map (λ e → source G e)
      (filter (λ e → eqFin? (target G e) v) (edgesList G))

-- Reachability --------------------------------------------------------------

-- Depth-first search collecting all vertices reachable from a seed
-- Uses a fuel argument for structural termination.
reachableFrom : (G : DirectedGraph) → (start : Fin (vertices G)) → List (Fin (vertices G))
reachableFrom G start =
  let vs = verticesList G in
  dfs (length vs) (start ∷ []) []
  where
    dfs : (fuel : Nat)
        → List (Fin (vertices G))  -- stack
        → List (Fin (vertices G))  -- visited (accumulator)
        → List (Fin (vertices G))
    dfs zero       stack visited = visited
    dfs (suc fuel) []    visited = visited
    dfs (suc fuel) (x ∷ xs) visited =
      if memberBy eqFin? x visited then
        dfs fuel xs visited
      else
        let nbrs = neighborsOut G x in
        dfs fuel (nbrs ++ xs) (x ∷ visited)

-- Reverse reachability (in the transpose graph)
coReachableTo : (G : DirectedGraph) → (goal : Fin (vertices G)) → List (Fin (vertices G))
coReachableTo G goal =
  let vs = verticesList G in
  dfs (length vs) (goal ∷ []) []
  where
    dfs : (fuel : Nat)
        → List (Fin (vertices G))  -- stack
        → List (Fin (vertices G))  -- visited
        → List (Fin (vertices G))
    dfs zero       stack visited = visited
    dfs (suc fuel) []    visited = visited
    dfs (suc fuel) (x ∷ xs) visited =
      if memberBy eqFin? x visited then
        dfs fuel xs visited
      else
        let nbrs = neighborsIn G x in
        dfs fuel (nbrs ++ xs) (x ∷ visited)

-- Topological sort (Kahn’s algorithm) --------------------------------------

-- Decide if a vertex has zero indegree with respect to a set of remaining edges
hasZeroInDeg : (G : DirectedGraph)
             → (remE : List (Fin (edges G)))
             → (v : Fin (vertices G))
             → Bool
hasZeroInDeg G remE v =
  not (any (λ e → eqFin? (target G e) v) remE)

-- Remove all outgoing edges of a vertex from the remaining edge set
removeOutEdges : (G : DirectedGraph)
               → (v : Fin (vertices G))
               → (remE : List (Fin (edges G)))
               → List (Fin (edges G))
removeOutEdges G v remE =
  removeIf (λ e → eqFin? (source G e) v) remE

-- Topological sort result: a list of vertices or nothing if a cycle exists
topoSort : (G : DirectedGraph) → Maybe (List (Fin (vertices G)))
topoSort G =
  let vs = verticesList G
      es = edgesList G
  in go (length vs) vs es []
  where
    -- Try to pick a zero indegree vertex from the list (define first for visibility)
    findZero : List (Fin (edges G)) → List (Fin (vertices G)) → Maybe (Fin (vertices G))
    findZero remE [] = nothing
    findZero remE (v ∷ vs) = if hasZeroInDeg G remE v then just v else findZero remE vs

    -- fuel decreases by one each time we remove a vertex
    go : (fuel : Nat)
       → List (Fin (vertices G))  -- remaining vertices
       → List (Fin (edges G))     -- remaining edges
       → List (Fin (vertices G))  -- output in reverse order
       → Maybe (List (Fin (vertices G)))
    go fuel []    remE acc = just (reverse acc)
    go zero (_ ∷ _) remE acc = nothing
    go (suc fuel) remV remE acc with findZero remE remV
    ... | just v  =
      let remV' = removeIf (λ u → eqFin? u v) remV
          remE' = removeOutEdges G v remE
      in go fuel remV' remE' (v ∷ acc)
    ... | nothing = nothing

-- Strongly connected components (Kosaraju) ----------------------------------

-- Intersection of two lists of Fin (as sets)
intersectFin : ∀ {k} → List (Fin k) → List (Fin k) → List (Fin k)
intersectFin []       ys = []
intersectFin (x ∷ xs) ys =
  if memberBy eqFin? x ys
  then x ∷ intersectFin xs ys
  else intersectFin xs ys

-- Stack items to compute finishing order
Stage : Type
Stage = Bool  -- false = pre, true = post

-- DFS to compute finishing order (decreasing by cons)
dfsFinish : (G : DirectedGraph)
         → (fuel : Nat)
         → List (Fin (vertices G))  -- seeds
         → List (Fin (vertices G))  -- visited
         → List (Fin (vertices G))  -- order (accumulator)
         → List (Fin (vertices G))
dfsFinish G zero      _           visited order = order
dfsFinish G (suc f) []           visited order = order
dfsFinish G (suc f) (v ∷ seeds)  visited order =
  if memberBy eqFin? v visited then
    dfsFinish G f seeds visited order
  else
    let vis' , ord' = walk G f ((v , false) ∷ []) visited order in
    dfsFinish G f seeds vis' ord'
  where
    walk : (G : DirectedGraph)
         → (fuel : Nat)
         → List (Fin (vertices G) × Stage)
         → List (Fin (vertices G))
         → List (Fin (vertices G))
         → Σ (List (Fin (vertices G))) (λ vis → List (Fin (vertices G)))
    walk G zero      _             vis ord = vis , ord
    walk G (suc f) []             vis ord = vis , ord
    walk G (suc f) ((v , pre) ∷ st) vis ord =
      if pre then
        walk G f st vis (v ∷ ord)
      else
        if memberBy eqFin? v vis then
          walk G f st vis ord
        else
          let nbrs = neighborsOut G v in
          walk G f (map (λ w → (w , false)) nbrs ++ ((v , true) ∷ st)) (v ∷ vis) ord

-- Compute SCCs via Kosaraju: two-pass DFS
sccs : (G : DirectedGraph) → List (List (Fin (vertices G)))
sccs G =
  let vs = verticesList G
      order = dfsFinish G (length vs * 4) vs [] []  -- ample fuel
  in secondPass (length vs * 4) order [] []
  where
    -- DFS on transpose to collect a component starting from v
    collect : (fuel : Nat)
           → List (Fin (vertices G))  -- stack
           → List (Fin (vertices G))  -- visited2
           → List (Fin (vertices G))  -- acc component
           → Σ (List (Fin (vertices G))) (λ vis → List (Fin (vertices G)))
    collect zero      stack vis acc = vis , acc
    collect (suc f) []    vis acc = vis , acc
    collect (suc f) (v ∷ st) vis acc =
      if memberBy eqFin? v vis then
        collect f st vis acc
      else
        let nbrs = neighborsIn G v in
        collect f (nbrs ++ st) (v ∷ vis) (v ∷ acc)

    -- Second pass over finishing order to build components
    secondPass : (fuel : Nat)
              → List (Fin (vertices G))          -- vertices in decreasing finish time
              → List (Fin (vertices G))          -- visited2
              → List (List (Fin (vertices G)))   -- acc comps
              → List (List (Fin (vertices G)))
    secondPass zero      _        vis comps = reverse comps
    secondPass (suc f) []       vis comps = reverse comps
    secondPass (suc f) (v ∷ vs) vis comps =
      if memberBy eqFin? v vis then
        secondPass f vs vis comps
      else
        let vis' , comp = collect f (v ∷ []) vis [] in
        secondPass f vs vis' (comp ∷ comps)

-- Build a Functor ·⇉· → FinSets from counts and endpoints -------------------

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Instances.FinSets
open import Cat.Instances.Shape.Parallel

-- Bounded-to-Fin conversion (axiomatized here to keep code lightweight)
postulate
  unsafeNatToFin : (k n : Nat) → Fin n

mkGraph : (v e : Nat)
        → (src tgt : Fin e → Fin v)
        → Functor ·⇉· FinSets
mkGraph v e src tgt .Functor.F₀ true  = v
mkGraph v e src tgt .Functor.F₀ false = e
mkGraph v e src tgt .Functor.F₁ {false} {false} tt = (λ x → x)
mkGraph v e src tgt .Functor.F₁ {true}  {true}  tt = (λ x → x)
mkGraph v e src tgt .Functor.F₁ {false} {true}  true  = src
mkGraph v e src tgt .Functor.F₁ {false} {true}  false = tgt
mkGraph v e src tgt .Functor.F-id {false} = funext λ x → refl
mkGraph v e src tgt .Functor.F-id {true}  = funext λ x → refl
mkGraph v e src tgt .Functor.F-∘ {false} {false} {false} tt tt = funext λ x → refl
mkGraph v e src tgt .Functor.F-∘ {true}  {true}  {true}  tt tt = funext λ x → refl
mkGraph v e src tgt .Functor.F-∘ {false} {false} {true}  b  tt = funext λ x → refl
mkGraph v e src tgt .Functor.F-∘ {false} {true}  {true}  tt b  = funext λ x → refl

postulate
  nth : ∀ {A : Type} → (xs : List A) → Fin (length xs) → A

-- Condensation graph: SCCs as vertices; edges between distinct components
condensationDirected : (G : DirectedGraph) → DirectedGraph
condensationDirected G = mkGraph vC eC src' tgt'
  where
    comps = sccs G
    vC    = length comps

    -- map original vertex to its component index as Fin vC (via Nat index → Fin)
    index-of-Fin : Fin (vertices G) → Fin vC
    index-of-Fin v = unsafeNatToFin (indexNat comps 0) vC
      where
        indexNat : List (List (Fin (vertices G))) → Nat → Nat
        indexNat []       k = k
        indexNat (c ∷ cs) k = if memberBy eqFin? v c then k else indexNat cs (suc k)

    -- edges crossing components (skip self-loops)
    crossEdges : List (Fin (edges G))
    crossEdges =
      let es = edgesList G in
      filter (λ e → not (eqFin? (index-of-Fin (source G e))
                               (index-of-Fin (target G e)))) es

    eC = length crossEdges

    -- use filtered list as index space for condensation edges
    edgeAt : Fin eC → Fin (edges G)
    edgeAt = nth crossEdges

    src' : Fin eC → Fin vC
    src' i = index-of-Fin (source G (edgeAt i))

    tgt' : Fin eC → Fin vC
    tgt' i = index-of-Fin (target G (edgeAt i))

infixl 0 _|>_
_|>_ : ∀ {A B : Type} → A → (A → B) → B
x |> f = f x
