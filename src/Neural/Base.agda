{-# OPTIONS --no-import-sorts #-}
module Neural.Base where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path
open import 1Lab.Path.Reasoning
open import 1Lab.Type
open import 1Lab.Type.Pointed
open import 1Lab.Reflection.Marker

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Instances.Functor
open import Cat.Instances.Shape.Parallel
open import Cat.Instances.FinSets

open import Data.Bool.Base
open import Data.Dec.Base
open import Data.Nat.Base using (Nat; zero; suc)
open import Data.Fin.Base using (Fin; fzero; fsuc; fin-view; Fin-elim; Fin-cases)
open Data.Fin.Base using (Fin-view)
open import Data.List.Base
open import Data.Sum.Base

-- | Neural codes as finite binary vectors
-- A neural code is a collection of binary response patterns
NeuralCode : (n : Nat) → Type
NeuralCode n = List (Fin n → Bool)

-- | Stimulus space as a type
StimulusSpace : Type₁
StimulusSpace = Type

-- | Place fields: regions of stimulus space that activate neurons
-- Each neuron has an associated place field (subset of stimulus space)
PlaceField : StimulusSpace → Type₁
PlaceField X = X → Type

-- | Directed finite graph (Definition 2.6)
-- A functor G: 2 → FinSets where 2 is the parallel arrows category ·⇉·
DirectedGraph : Type
DirectedGraph = Functor ·⇉· FinSets

-- | Extract vertices from a directed graph (G applied to true)
vertices : DirectedGraph → Nat
vertices G = G .Functor.F₀ true

-- | Extract edges from a directed graph (G applied to false)
edges : DirectedGraph → Nat
edges G = G .Functor.F₀ false

-- | Source function: edges → vertices (G applied to true morphism)
source : (G : DirectedGraph) → Fin (edges G) → Fin (vertices G)
source G = G .Functor.F₁ {false} {true} true

-- | Target function: edges → vertices (G applied to false morphism)
target : (G : DirectedGraph) → Fin (edges G) → Fin (vertices G)
target G = G .Functor.F₁ {false} {true} false

-- -- | Category of pointed finite sets F*
-- -- Objects are suc n (non-empty finite sets) with basepoint always at fzero
-- PointedFinSets : Precategory lzero lzero
-- PointedFinSets .Precategory.Ob = Nat  -- Will use suc n to ensure non-emptiness
-- PointedFinSets .Precategory.Hom m n = Σ (Fin (suc m) → Fin (suc n)) (λ f → f fzero ≡ fzero)  -- Basepoint-preserving maps
-- PointedFinSets .Precategory.Hom-set _ _ = hlevel 2
-- PointedFinSets .Precategory.id = (λ x → x) , refl
-- PointedFinSets .Precategory._∘_ (f , pf) (g , pg) = (f ∘ g) , (ap f pg ∙ pf)
-- PointedFinSets .Precategory.idr f = Σ-pathp refl (∙-idl (f .snd))
-- PointedFinSets .Precategory.idl f = Σ-pathp refl (∙-idr (f .snd))
-- PointedFinSets .Precategory.assoc f g h = Σ-pathp refl (
--   ⌜ ap (f .fst) (ap (g .fst) (h .snd) ∙ g .snd) ⌝ ∙ f .snd ≡⟨ ap! (ap-∙ (f .fst) _ _) ⟩
--   (ap (f .fst ∘ g .fst) (h .snd) ∙ ap (f .fst) (g .snd)) ∙ f .snd ≡⟨ sym (∙-assoc _ _ _) ⟩
--   ap (f .fst ∘ g .fst) (h .snd) ∙ ap (f .fst) (g .snd) ∙ f .snd ∎)


-- -- | Pointed directed finite graph (Definition 2.7)
-- -- A functor G: 2 → F*
-- PointedDirectedGraph : Type
-- PointedDirectedGraph = Functor ·⇉· PointedFinSets


-- -- | Extract vertices from a directed graph (G applied to true)
-- verticesP : PointedDirectedGraph → Nat
-- verticesP G = G .Functor.F₀ true 

-- -- | Extract edges from a directed graph (G applied to false)
-- edgesP : PointedDirectedGraph → Nat
-- edgesP G = G .Functor.F₀ false

-- -- | Source function: edges → vertices (G applied to true morphism)
-- sourceP : (G : PointedDirectedGraph) → Fin (edgesP G) → Fin (verticesP G)
-- sourceP G =  G .Functor.F₁ {false} {true} true

-- -- | Target function: edges → vertices (G applied to false morphism)
-- targetP : (G : PointedDirectedGraph) → Fin (edgesP G) → Fin (verticesP G)
-- targetP G = {!!}




-- | Construct pointed graph G* from ordinary graph G (Lemma 2.8)
-- Adds a disjoint basepoint vertex and looping edge
-- to-pointed-graph : DirectedGraph → PointedDirectedGraph
-- to-pointed-graph G .Functor.F₀ true = suc (vertices G)  -- Add one vertex (basepoint at fzero)
-- to-pointed-graph G .Functor.F₀ false = suc (edges G)   -- Add one edge (looping edge at fzero)
-- to-pointed-graph G .Functor.F₁ {false} {true} true = {!source!} , {!!}
-- to-pointed-graph G .Functor.F₁ {false} {true} false =
--   Fin-cases fzero (λ e → {!!}) , refl
-- to-pointed-graph G .Functor.F₁ {false} {false} tt = (λ x → x) , refl  -- identity on edges
-- to-pointed-graph G .Functor.F₁ {true} {true} tt = (λ x → x) , refl    -- identity on vertices
-- to-pointed-graph G .Functor.F-id {false} = Σ-pathp refl (∙-idl refl)
-- to-pointed-graph G .Functor.F-id {true} = Σ-pathp refl (∙-idl refl)
-- to-pointed-graph G .Functor.F-∘ {false} {false} {false} f g = {!!}
-- to-pointed-graph G .Functor.F-∘ {false} {false} {true} f g = {!!}
-- to-pointed-graph G .Functor.F-∘ {false} {true} {true} f g = {!!}
-- to-pointed-graph G .Functor.F-∘ {true} {true} {true} f g = {!!}


{-
-- | Basepoint vertex in a pointed graph (always fzero)
basepoint-vertex : (G : PointedDirectedGraph) → Fin (G .Functor.F₀ true)
basepoint-vertex G = fzero

-- | Looping edge in a pointed graph (always fzero)
looping-edge : (G : PointedDirectedGraph) → Fin (G .Functor.F₀ false)
looping-edge G = fzero
-}
