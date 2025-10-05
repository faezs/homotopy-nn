## Detailed Postulate List

### src/Neural/Dynamics/IntegratedInformation.agda

58:postulate
59-  -- Integrated information for network state
60-  Φ-hopfield :
61-    {G : DirectedGraph} →
--
154:postulate
155-  -- State space partition indexed by input values
156-  StatePartition :
157-    (G : DirectedGraph) →
--
227:postulate
228-  -- Change in Φ per time step
229-  ΔΦ :
230-    {G : DirectedGraph} →
--
273:postulate
274-  -- Attractor detection
275-  is-attractor :
276-    {G : DirectedGraph} →
--
321:postulate
322-  -- Cohomology at time t
323-  cohomology-at-time :
324-    {G : DirectedGraph} →
--
367:postulate
368-  -- Phase transition in Φ
369-  Φ-phase-transition :
370-    {G : DirectedGraph} →
--
407:postulate
408-  -- Attractor manifold
409-  AttractorManifold :
410-    {G : DirectedGraph} →
--
459:postulate
460-  -- Global workspace = High Φ
461-  global-workspace-Φ :
462-    {G : DirectedGraph} →
--
497:postulate
498-  -- Experimental correlates
499-  fMRI-Φ-correlation :
500-    {-| Φ correlates with BOLD signal synchrony -}

### src/Neural/Information/Shannon.agda

79:postulate
80-  {-|
81-  Shannon entropy of a probability measure
82-  -}
--
118:postulate
119-  {-|
120-  Extensivity of Shannon entropy
121-
--
151:postulate
152-  ℝ-thin-category : Precategory lzero lzero
153-
154-  ℝ-thin-Ob : Precategory.Ob ℝ-thin-category ≡ ℝ
--
178:postulate
179-  is-surjection :
180-    {X Y : FinitePointedSet} →
181-    (f : underlying-set X → underlying-set Y) →
--
189:postulate
190-  is-fiberwise-probability :
191-    {X Y : FinitePointedSet} →
192-    (f : underlying-set X → underlying-set Y) →
--
216:postulate
217-  Pf-surjective : Precategory lzero lzero
218-
219-  Pf-surjective-Ob : Precategory.Ob Pf-surjective ≡ PfObject
--
238:postulate
239-  {-|
240-  Shannon entropy functor (Lemma 5.13)
241-  -}
--
286:postulate
287-  {-|
288-  Entropy relation for embeddings
289-
--
326:postulate
327-  {-|
328-  Entropy bounds for summing functors (Lemma 5.14)
329-  -}
--
374:postulate
375-  {-| Simplex category -}
376-  SimplexCategory : Precategory lzero lzero
377-

### src/Neural/Computational/TransitionSystems.agda

142:postulate
143-  is-reachable : (τ : TransitionSystem) → (s : τ .States) → Type
144-
145-  is-reachable-prop : (τ : TransitionSystem) → (s : τ .States) → is-prop (is-reachable τ s)
--
241:postulate
242-  TransitionSystems-is-category : Precategory (lsuc lzero) (lsuc lzero)
243-
244--- Extract structure
--
251:postulate
252-  TS-id : (τ : TransitionSystem) → TransitionSystemMorphism τ τ
253-
254-{-|
--
259:postulate
260-  TS-∘ :
261-    {τ₁ τ₂ τ₃ : TransitionSystem} →
262-    TransitionSystemMorphism τ₂ τ₃ →
--
287:postulate
288-  {-| Coproduct of two transition systems -}
289-  TS-coproduct :
290-    (τ₁ τ₂ : TransitionSystem) →
--
329:postulate
330-  {-| Product of two transition systems -}
331-  TS-product :
332-    (τ₁ τ₂ : TransitionSystem) →
--
406:postulate
407-  {-|
408-  General grafting: connect state s ∈ S₁ to state s' ∈ S₂
409-
--
437:postulate
438-  {-| Topological ordering on vertices of a graph -}
439-  TopologicalOrdering : DirectedGraph → Type
440-
--
547:postulate
548-  {-| Strongly connected components of a graph -}
549-  StronglyConnectedComponents : DirectedGraph → Type
550-
--
645:postulate
646-  {-| Input degree: number of incoming external edges -}
647-  deg-in : TransitionSystem' → Nat
648-
--
682:postulate
683-  Subgraphs : DirectedGraph → Precategory lzero lzero
684-
685-module ProperadFunctor where
--
838:postulate
839-  {-| Input degree for time-delay systems -}
840-  td-deg-in : TimeDelayTransitionSystem' → Nat
841-
--
963:postulate
964-  {-| Condensation graph of distributed structure -}
965-  condensation-distributed :
966-    (G : DirectedGraph) →
--
1001:postulate
1002-  induced-subgraph :
1003-    (G : DirectedGraph) →
1004-    (m : DistributedStructure G) →
--
1059:postulate
1060-  Gdist : Precategory (lsuc lzero) (lsuc lzero)
1061-
1062-  {-| Objects are distributed graphs -}
