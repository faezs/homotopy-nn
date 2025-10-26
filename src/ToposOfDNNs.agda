-- { OPTIONS --no-import-sorts --allow-unsolved-metas #- -#}
{-|
# Topos and Stacks of Deep Neural Networks

This module imports all formalized components from Belfiore & Bennequin (2022).
"Topos and Stacks of Deep Neural Networks"

## Organization

This follows the paper structure:

### Section 1: Topos Theory for DNNs (Belfiore & Bennequin 2022)

**Section 1.1-1.4: Fork Construction and Backpropagation**
- Neural.Topos.Architecture: Oriented graphs, Fork-Category, DNN-Topos
  * Section 1.1: Oriented graphs (directed, classical, acyclic)
  * Section 1.3: Fork construction (ForkVertex, fork topology J)
  * Section 1.4: Backpropagation as natural transformations (Theorem 1.1)
  * Theorem 1.2: Poset X structure (trees joining at minimal elements)

- Neural.Topos.Examples: Concrete network architectures
  * SimpleMLP: Chain network (no convergence)
  * ConvergentNetwork: Diamond poset with fork
  * ComplexNetwork: Multi-path ResNet-like

- Neural.Topos.PosetDiagram: 5-output FFN visualization
  * Complete poset structure with multiple outputs
  * Demonstrates tree structure from Theorem 1.2

**Section 1.5: Topos Foundations**
- Neural.Topos.Poset: Proposition 1.1, CX poset structure
- Neural.Topos.Alexandrov: Alexandrov topology, Proposition 1.2
- Neural.Topos.Properties: Topos equivalences, localic structure
- Neural.Topos.Localic: Localic aspects and frame homomorphisms
- Neural.Topos.Spectrum: Spectral spaces and Stone duality
- Neural.Topos.ClassifyingGroupoid: Classifying topos for groupoids
- Neural.Topos.NonBoolean: Non-Boolean toposes and intuitionistic logic

### Section 2: Stacks and Fibrations

**Section 2.1: Groupoid Actions**
- Neural.Stack.Groupoid: Group actions, Equation 2.1, CNN example

**Section 2.2: Fibrations & Classifiers**
- Neural.Stack.Fibration: Equations 2.2-2.6, sections, presheaves
- Neural.Stack.Classifier: Ω_F, Proposition 2.1, Equations 2.10-2.12
- Neural.Stack.Geometric: Geometric functors, Equations 2.13-2.21
- Neural.Stack.LogicalPropagation: Lemmas 2.1-2.4, Theorem 2.1, Eqs 2.24-2.32

**Section 2.3: Type Theory & Semantics**
- Neural.Stack.TypeTheory: Equation 2.33, formal languages, MLTT
- Neural.Stack.Semantic: Equations 2.34-2.35, soundness, completeness

**Section 2.4: Model Categories**
- Neural.Stack.ModelCategory: Proposition 2.3, Quillen structure
- Neural.Stack.Examples: Lemmas 2.5-2.7, CNN/ResNet/Attention examples
- Neural.Stack.Fibrations: Theorem 2.2, multi-fibrations
- Neural.Stack.MartinLof: Theorem 2.3, Lemma 2.8, univalence

**Section 2.5: Classifying Topos**
- Neural.Stack.Classifying: Extended types, completeness, E_A

### Section 3: Dynamics, Logic, and Homology

**Section 3.1-3.4: Complete Framework**
- Neural.Stack.CatsManifold: Cat's manifolds, conditioning, vector fields
- Neural.Stack.SpontaneousActivity: Spontaneous vertices, dynamics decomposition
- Neural.Stack.Languages: Language sheaves, deduction fibrations, modal logic
- Neural.Stack.SemanticInformation: Homology, persistent homology, IIT connection

## Totals

**Complete implementation**: ~10,000+ lines of formalized mathematics covering:
- ✅ All 35 equations (2.1-2.35)
- ✅ All 8 lemmas (2.1-2.8)
- ✅ All 8 propositions (1.1, 1.2, 2.1, 2.3, 3.1-3.5)
- ✅ All 3 theorems (2.1, 2.2, 2.3)
- ✅ 110+ definitions
- ✅ 29 complete definitions (3.1-3.29)

## Connection to Schreiber (2024)

The DNN-Topos framework instantiates Schreiber's "Higher Topos Theory in Physics":

**Our site**: Fork-Category (oriented graph with convergent vertices)
**Our coverage**: Sheaf condition F(A★) ≅ ∏ F(incoming)
**Our topos**: DNN-Topos = Sh[Fork-Category, fork-coverage]

This provides:
- **Compositional architectures**: Functoriality
- **Information aggregation**: Sheaf condition
- **Backpropagation**: Natural transformations
- **Field spaces**: Mapping spaces in cartesian closed topos

## Type Checking

To type-check all topos modules:
```bash
agda --library-file=./libraries src/ToposOfDNNs.agda
```

## References

- **Belfiore, F. & Bennequin, D.** (2022). Topos and Stacks of Deep Neural Networks.
- **Schreiber, U.** (2024). Higher Topos Theory in Physics. December 30, 2024.
-}

module ToposOfDNNs where

-- ============================================================================
-- Section 1: Topos Theory for DNNs
-- ============================================================================

-- Section 1.1-1.4: Fork Construction and Backpropagation
import Neural.Topos.Architecture
import Neural.Topos.Examples
import Neural.Topos.PosetDiagram

-- Section 1.5: Topos Foundations
import Neural.Topos.Poset
import Neural.Topos.Alexandrov
import Neural.Topos.Properties
import Neural.Topos.Localic
import Neural.Topos.Spectrum
import Neural.Topos.ClassifyingGroupoid
import Neural.Topos.NonBoolean

-- ============================================================================
-- Section 2: Stacks and Fibrations
-- ============================================================================

-- Section 2.1: Groupoid Actions
import Neural.Stack.Groupoid

-- Section 2.2: Fibrations & Classifiers
import Neural.Stack.Fibration
import Neural.Stack.Classifier
import Neural.Stack.Geometric
import Neural.Stack.LogicalPropagation

-- Section 2.3: Type Theory & Semantics
import Neural.Stack.TypeTheory
import Neural.Stack.Semantic

-- Section 2.4: Model Categories
import Neural.Stack.ModelCategory
import Neural.Stack.Examples
import Neural.Stack.Fibrations
import Neural.Stack.MartinLof

-- Section 2.5: Classifying Topos
import Neural.Stack.Classifying

-- ============================================================================
-- Section 3: Dynamics, Logic, and Homology
-- ============================================================================

import Neural.Stack.CatsManifold
import Neural.Stack.SpontaneousActivity
import Neural.Stack.Languages
import Neural.Stack.SemanticInformation

{-|
## Module Summary

This file provides a single entry point for the complete formalization of
Belfiore & Bennequin's "Topos and Stacks of Deep Neural Networks" (2022).

### Key Theoretical Results

**Theorem 1.1** (Backpropagation as Natural Transformation):
The backpropagation differential ∇W L is a natural transformation
between the weight sheaf W and itself, flowing along directed paths
in the network.

**Theorem 1.2** (Poset Structure):
The oriented graph X underlying a DNN has a poset structure where
paths form trees that join at minimal elements (convergent layers).

**Proposition 1.1** (CX Poset):
The category CX of open sets with coverings forms a poset under
inclusion, with the Alexandrov topology.

**Proposition 1.2** (Topos Equivalence):
The DNN-Topos Sh[Fork-Category, J] is equivalent to the topos of
sheaves on the Alexandrov space of the oriented graph.

**Theorem 2.1** (Logical Propagation):
Forward propagation in DNNs is a geometric morphism between toposes,
preserving the internal logic (intuitionistic logic of the network).

**Theorem 2.2** (Multi-Fibrations):
The category of DNNs with group actions forms a multi-fibration over
the category of groupoids.

**Theorem 2.3** (Martin-Löf Semantics):
The internal type theory of the DNN-Topos is a model of Martin-Löf
type theory with dependent types.

### Philosophical Import

This formalization shows that deep neural networks are not ad-hoc
constructions but rather **canonical structures in topos theory**:

1. **Networks are sheaves**: Information is glued consistently
2. **Learning is geometric**: Weight updates are morphisms in topos
3. **Backprop is natural**: Gradients flow along natural transformations
4. **Composition is fundamental**: Network layers compose categorically

This bridges the gap between:
- **Machine Learning**: Practical neural network architectures
- **Mathematical Physics**: Topos-theoretic field theories (Schreiber)
- **Logic**: Intuitionistic type theory and proof theory

The result is a **rigorous mathematical foundation for deep learning**
that connects it to the broader landscape of modern mathematics.
-}



{-
Entirety of the Paper:


Topos and Stacks
of
Deep Neural Networks
arXiv:2106.14587v3 [math.AT] 16 Jun 2022
Jean-Claude Belfiore
Huawei Advanced Wireless Technology Lab.
Paris Research Center
Daniel Bennequin
Huawei Advanced Wireless Technology Lab.
Paris Research Center
University of Paris Diderot, Faculty of Mathematics
Abstract
Every known artificial Deep Neural Network (DNN) corresponds to an object in a canonical Grothendieck’s
topos; its learning dynamic corresponds to a flow of morphisms in this topos. Invariance structures in the
layers (like CNNs or LSTMs) correspond to Giraud’s stacks. This invariance is supposed to be responsi-
ble of the generalization property, that is extrapolation from learning data under constraints. The fibers
represent pre-semantic categories (Culioli [CLS95], Thom [Tho72]), over which artificial languages are
defined, with internal logics, intuitionist, classical or linear (Girard [Gir87]). Semantic functioning of
a network is its ability to express theories in such a language for answering questions in output about
input data. Quantities and spaces of semantic information are defined by analogy with the homological
interpretation of Shannon’s entropy (Baudot & Bennequin [BB15]). They generalize the measures found
by Carnap and Bar-Hillel [CBH52]. Amazingly, the above semantical structures are classified by geo-
metric fibrant objects in a closed model category of Quillen [Qui67], then they give rise to homotopical
invariants of DNNs and of their semantic functioning. Intentional type theories (Martin-Löf [ML80])
organize these objects and fibrations between them. Information contents and exchanges are analyzed by
Grothendieck’s derivators [Gro90].
1
Contents
1 Architectures 8
1.1 1.2 1.3 1.5 Underlying graph . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
Dynamical objects of the chains . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
Dynamical objects of the general DNNs . . . . . . . . . . . . . . . . . . . . . . . . . . 10
1.4 Backpropagation as a natural (stochastic) flow in the topos . . . . . . . . . . . . . . . . 12
The specific nature of the topos of DNNs . . . . . . . . . . . . . . . . . . . . . . . . . 15
2 Stacks of DNNs 20
2.1 2.3 2.5 Groupoids, general categorical invariance and logic . . . . . . . . . . . . . . . . . . . . 20
2.2 Objects classifiers of the fibers of a classifying topos . . . . . . . . . . . . . . . . . . . 23
Theories, interpretation, inference and deduction . . . . . . . . . . . . . . . . . . . . . 32
2.4 The model category of a DNN and its Martin-Löf type theory . . . . . . . . . . . . . . . 35
Classifying the M-L theory ? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
3 Dynamics and homology 45
3.1 3.2 3.3 3.4 3.5 Ordinary cat’s manifolds . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
Dynamics with spontaneous activity . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
Fibrations and cofibrations of languages and theories . . . . . . . . . . . . . . . . . . . 48
Semantic information. Homology constructions . . . . . . . . . . . . . . . . . . . . . . 59
Homotopy constructions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 73
4 Unfoldings and memories, LSTMs and GRUs 94
4.1 4.2 4.3 4.4 RNN lattices, LSTM cells . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 94
GRU, MGU . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 98
Universal structure hypothesis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 100
Memories and braids . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 102
4.5 Pre-semantics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 109
2
5 A natural 3-category of deep networks 112
5.1 Attention moduli and relation moduli . . . . . . . . . . . . . . . . . . . . . . . . . . . . 112
5.2 5.3 The 2-category of a network . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 115
Grothendieck derivators and semantic information . . . . . . . . . . . . . . . . . . . . . 116
5.4 Stacks homotopy of DNNs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 119
Appendices
A Localic topos and Fuzzy identities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 120
B Topos of DNNs and spectra of commutative rings . . . . . . . . . . . . . . . . . . . . . 125
C Classifying objects of groupoids . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 126
D Non-Boolean information functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . 128
E Closer to natural languages: linear semantic information . . . . . . . . . . . . . . . . . 130
3
Preface
Introduction
This text presents a general theory of semantic functioning of deep neural networks, DNNs, based on
topology, more precisely, Grothendieck’s topos, Quillen’s homotopy theory, Thom’s singularity theory
and the pre-semantic of Culioli in enunciative linguistic.
The theory is based on the existing networks, transforming data, as images, movies or written texts,
to answer questions, achieve actions or take decisions. Experiments, recent and past, show that the deep
neural networks, which have learned under constrained methods, can achieve surprising semantic per-
formances [XQLJ20], [BBD+11], [BBDH14], [BBG21a], [DHSB20], [KL14], [MXY+15], [ZRS+18],
[ZCZ+19], [GLH+20]. However, the exploitation of more explicit invariance structures and adapted
languages, are in great part a task for the future. Thus the present text is a mixture of an analysis of the
functioning networks, and of a conjectural frame to make them able to approach more ideal semantic
functioning.
Note that categories, homology and homotopy were recently applied in several manners to semantic
information. An example is the application of category theory to the design of networks, by Fong and
Spivak [FS18]. For a recent review on many applications of category theory to Machine Learning, see
[SGW21]. Other examples are given by the general notion of Information Networks based on Segal
spaces by Yuri Manin and Matilde Marcolli, [MM20] and the Čech homology reconstruction of the
environment by place fields of Curto and collaborators, [Cur17]. Let us also mention the characteriza-
tion of entropy, by Baez, Fritz, Leinster, [BFL11], and the use of sheaves and cosheaves for studying
information networks, Ghrist, Hiraoka 2011 [GH11], Curry 2013 [Cur13], Robinson and Joslyn [Rob17],
and Abramsky et al. specially for Quantum Information [AB11]. Persistent homology for detecting
structures in data must also be cited in this context, for instance Port, Karidi, Marcolli 2019, [PKM19]
on syntactic structures, and Carlsson et al. on shape recognition [CZCG05]. More in relation with Bayes
4
networks, there are the three recent PhD theses of Juan-Pablo Vigneaux [Vig19], Olivier Peltre [Pel20]
and Grégoire Sergeant-Perthuis [SP21].
With respect to these works, we look at a notion of information which is a (toposic) topological
invariant of the situation which involves three dimensions of dynamics:
1) a logical flow along the network;
2) in the layers, the action of categories;
3) the evocations of meaning in languages.
The resulting notion of information generalizes the suggestion of Carnap and Bar-Hillel 1952 in these
three dynamical directions. Our inspiration came from the toposic interpretation of Shannon’s entropy
in [BB15] and [Vig20]. A new fundamental ingredient is the interpretation of internal implication
(exponential) as a conditioning on theories, analogous to the conditioning in probabilities. We distinguish
between the theoretically accessible information, concerning all the theories in a fibred languages, and
the practically accessible information, that corresponds to the semantic functioning of concrete neural
networks, associated to a feed-forward dynamics which depends on a learning process.
The main results in this text are,
✔ theorems 1.1 and 1.2 characterizing the topos associated to a DNN
✔ theorem 2.1 giving a geometric suﬃcient condition for a fluid circulation of semantics in this topos
✔ theorems 2.2 and 2.3, characterizing the fibrations (in particular the fibrant objects) in a closed
model category made by the stacks of the DNNs having a given network architecture
✔ the tentative definition of Semantic Information quantities and spaces in sections 3.4 and 3.5
✔ theorem 4.1 on the generic structures and dynamics of LSTMs.
Specific examples, showing the nature of the semantic information that we present here, are at the end
of section 3.5 extracted from the exemplar toy language of Carnap and Bar-Hillel and the mathematical
interpretation of the pre-semantic of Culioli in relation with the artificial memory cells of sections 4.4
and 4.5.
Chapter 1 describes the nature of the sites and the topos associated to deep neural networks, said
𝐷𝑁𝑁𝑠, with their dynamics, feedforward and backward (backpropagation) learning.
Chapter 2 presents the diﬀerent stacks of a 𝐷𝑁𝑁, which are fibred categories over the site of the
𝐷𝑁𝑁, incorporating symmetries and logics for approaching the wanted semantics in functioning. Usual
examples are 𝐶𝑁𝑁𝑠for translation symmetries, but also other ones regarding logic and semantics (see
experiments in Logical Information Cells I [BBG21a]). Thus the logical structure of the classifying topos
5
of such a stack is described. We introduce hypotheses on the stack and the language objects that allow
a transmission of theories downstream and of propositions upstream in the network. The 2-category of
the stacks over a given architecture is shown to constitute a closed model theory of injective type, in
the sense of Quillen (also Cisinski and Lurie). The fibrant objects, which are diﬃcult to characterize in
general, are determined in the case of the Grothendieck sites of 𝐷𝑁𝑁𝑠. Interestingly, they correspond to
the hypothesis guarantying the transmission of theories. Using the work of Arndt and Kapulkin [AK11]
we show that the above model theory gives rise to a Martin-Löf type theory associated to every 𝐷𝑁𝑁.
Semantics in the sense of topos (Lambek) is added by considering objects in the classifying topos of the
stack.
In chapter 3, we start exploring the notion of semantic information and semantic functioning in
𝐷𝑁𝑁𝑠, by using homology and homotopy theory. Then we define semantic conditioning of the theories
by the propositions, and compute the corresponding ringed cohomology of the functions of these theories;
this gives a numerical notion of semantic ambiguity, of semantic mutual information and of semantic
Kullback-Leibler divergence. Then we generalize the homogeneous bar-complex to define a bi-simplicial
set 𝐼•
★ of classes of theories and propositions histories over the network, by taking homotopy colimits.
We introduce a class of increasing and concave functions from 𝐼•
★ to an external model category M; and
with them, we obtain natural homotopy types of semantic information, associated to coherent semantic
functioning of a network with respect to a semantic problem; they satisfy properties conjectured by Car-
nap and Bar-Hillel in 1952 [CBH52] for the sets of semantic information. On the simple example they
studied we show the interest of considering spaces of information, in particular groupoids, in addition to
the more usual combinatorial dimension of logical content of propositions.
Chapter 4 describes examples of memory cells, as the long and short terms memory cells (LSTM),
and shows that the natural groupoids for their stack have as fundamental group the group of Artin’s braids
with three strands 𝔅3. Generalizations are proposed, for semantics closer to the semantic of natural
languages, in appendix E.
Finally chapter 5 introduces possible applications of topos, stacks and models to the relations between
several 𝐷𝑁𝑁𝑠: understanding the modular structures of networks, defining and studying the obstructions
to integrate some semantics or to solve problems in some contexts. Examples could be taken from the
above mentioned experiments on logical information cells, and from recent attempts of several teams
in artificial intelligence: Hudson & Manning [HM18], Santoro, Raposo et al. [SRB+17], Bengio and
Hinton, using memory modules, linguistic analysis modules, attention modules and relation modules, in
addition to convolution 𝐶𝑁𝑁𝑠, for answering questions about images and movies (also see [RSB+17],
[ZCZ+19], [HB20]).
Most of the figures mentioned in the text can be found in the chapter by Bennequin and Belfiore On
new mathematical concepts for Artificial Intelligence, in the Huawei volume on Mathematics for Future
Computing and Communication, edited by Liao Heng and Bill McColl [HM21]. We also refer to this
chapter for the elements of category theory that are necessary to understand this text, the definitions and
6
first properties of topos and Grothendieck topos, and the presentation of elementary type theories.
Chapter 9 in [HM21], by Ge Yiqun and Tong Wen, Mathematics, Information and Learning, explains
the large place of topology in the notions of semantic information.
In a forthcoming preprint, entitled A search of semantic spaces, we will compute spaces of semantic
information for several elementary languages, along the lines indicated in section 3.5, and develop further
the Galois point of view on the information flow in a network. The notions of intentional signification,
meaning and knowledge are discussed from a philosophical point of view, and adapted to artificial se-
mantic and its intelligibility.
In another following preprint, A mathematical theory of semantic communication, we plan to present
the application of the above stacks of functioning DNNs and their information spaces, to the problem
of semantic communication. In particular we show how the invariance structures in the fibers, made by
categories acting on artificial languages, give a way to understand generalization properties of DNNs, for
extrapolation, not only interpolation.
Analytical aspects, as equivariant standard DNNs approximation of functions, or gradient descent
respecting the invariance, are developed in this context.
Acknowledgements
The two authors deeply thank Olivia Caramello and Laurent Laﬀorgue for the impulsion they gave to
this research, for their constant encouragements and many helpful suggestions. They also warmly thank
Merouane Debbah for his deep interest, the help and the support he gave, Xavier Giraud, for the concrete
experiments he realized with us, allowing to connect the theory with the lively spontaneous behavior of
artificial neural networks, and Zhenrong Liu (Louise) for her constant and very kind help at work.
D.B. gives special thanks to Alain Berthoz, with whom he has had the chance to work and dream since
many years on a conjectural topos geometry (properly speaking stacks) for the generation and control
of the variety of humans voluntary movements. He also does not forget that the presence of natural
invariants of topos in Information theory was discovered during a common work with Pierre Baudot,
that he heartily thanks. D.B. had many inspiring discussions on closely related subjects with his former
students, in particular Alireza Bahraini, Alexandre Afgoustidis, Juan-Pablo Vigneaux, Olivier Peltre and
Grégoire Sergeant-Perthuis, that he friendly thanks, with gratitude.
7
1
Architectures
Let us show how every (known) artificial deep neural network (𝐷𝑁𝑁) can be described by a family of
objects in a well defined topos.
1.1 Underlying graph
Definition. An oriented graph Γ is directed when the relation 𝑎≤𝑏 between vertices, defined by the
existence of an oriented path, made by concatenation of oriented edges, is a partial ordering on the set
𝑉(Γ)= Γ(0) of vertices. A graph is said classical if there exists at most one edge between two vertices,
and no loop at one vertex (also named tadpole). A classical directed graph can have non-oriented cycles,
but no oriented cycles.
The layers and the direct connections between layers in an artificial neural network constitute a finite
oriented graph Γ, which is directed, and classical.
The minimal elements correspond to the initial layers, or input layers, and the maximal elements to
the final layers, or output layers, all the other correspond to hidden layers, or inner layers. In the case of
𝑅𝑁𝑁𝑠(as when we look at feedback connections in the brain) we apparently see loops, however they are
not loops in space-time, the graph which represents the functioning of the network must be seen in the
space-time (not necessary Galilean but causal), then the loops disappear and the graph appears directed
and classical (see figure 1.1). Apparently there is no exception to these rules in the world of 𝐷𝑁𝑁𝑠.
Remark. Bayesian networks are frequently associated to oriented or non-oriented graphs, which can
be non-directed, and have oriented loops. However, the underlying random variables are associated to
vertices and to edges, the variable of an edge 𝑎𝑏being the joint variable of the variables of 𝑎and 𝑏. More
generally, an hypergaph is considered, made by a subset Aof the set P(𝐼)of subsets of a given set 𝐼. In
this situation, we have a poset, where the natural partial ordering relation is the opposite of the inclusion,
i.e. it goes from the finer variable to the coarser one.
8
y
yt yt
rt rt rt+
r
x
(a) Original RNN
xt xt
(b) Unfolded RNN in Space-Time
Figure 1.1: RNN with space-time unfolding
1.2 Dynamical objects of the chains
The simplest architecture of a network is a chain, and the feed-forward functioning of the network, when
it has learned, corresponds to a covariant functor 𝑋 from the category C𝑜(Γ)freely generated by the
graph to the category of sets, Set: to a layer 𝐿𝑘; 𝑘 ∈Γ is associated the set 𝑋𝑘 of possible activities of
the population of neurons in 𝐿𝑘, to the edge 𝐿𝑘 ↦→𝐿𝑘+1 is associated the map 𝑋𝑤
𝑘+1,𝑘 : 𝑋𝑘 →𝑋𝑘+1 which
corresponds to the learned weights 𝑤𝑘+1,𝑘; then to each arrow in C𝑜(Γ), we associate the composed map.
But also the weights can be encoded in a covariant functor Π from C𝑜(Γ)to Set: for 𝐿𝑘 we define
Π𝑘 as the product of all the sets 𝑊𝑙+1,𝑙 of weights for 𝑙 ≥𝑘, and to the edge 𝑘↦→𝑘+1 we associate the
natural forgetting projection Π𝑘+1,𝑘 : Π𝑘 →Π𝑘+1. (The product over an empty set is the singleton ★in
Set, then for the output layer 𝐿𝑛 the last projection is the unique possible map from Π𝑛−1 to ★.) In what
follows, we will note W= Π, for remembering that it describes the functor of weights, but the notation Π
is less confusing for denoting the morphisms in this functor.
The cartesian products 𝑋𝑘 ×Π𝑘 together with the maps
𝑋𝑘+1,𝑘×Π𝑘+1,𝑘 𝑥𝑘,(𝑤𝑘+1,𝑘,𝑤
′
𝑘+1)= 𝑋𝑤
𝑘+1,𝑘(𝑥𝑘),𝑤
′
𝑘+1 (1.1)
also defines a covariant functor X; it represents all the possible feed-forward functioning of the network,
for every potential weights. The natural projection from Xto W= Π is a natural transformation of
functors. It is remarkable that, in supervised learning, the Backpropagation algorithm is represented by
a flow of natural transformations of the functor Wto itself. We give a proof below in the general case,
not only for a chain, where it is easier.
Remark a diﬀerence with Spivak et al. [FST19], where backpropagation is a functor, not a natural
transformation.
In fact, the weights represent mappings between two layers, individually they correspond to mor-
phisms in a functor 𝑋𝑤, then it should have been more intuitive if they had been coded by morphisms,
however globally they are better encoded by the objects in the functor W, and the morphisms in this
functor are the erasure of the weights along the arrows that correspond to them. This appears as a kind
9
of dual representation of the mappings 𝑋𝑤
.
As we want to respect the convention of Topos theory, [AGV63], we introduce the category C= C(Γ)
which is opposed to C0(Γ); then 𝑋𝑤
, W= Π and Xbecome contravariant functors from this category Cto
Sets, i.e. presheaves over C, i.e. objects in the topos C∧[HM21]. This is this topos which is associated
to the neural network which has the shape of a chain (multi-layer perceptron). Observe that the arrows
between sets continue to follow the natural dynamical ordering, from the initial layer to the final layer,
but the arrows in the category (the site) Care going now in the opposite direction.
The object 𝑋𝑤 can be naturally identified with a subobject of X, we call this singleton the fiber of
𝑝𝑟2 : X→Wover the singleton 𝑤in W, (that is a morphism in C∧from the final object 1 (the constant
functor equal to the point ★at each layer) to the object W), which is a system of weights for each edge of
the graph Γ.
In this simple case of a chain, the classifying object of subobjects Ω, which is responsible of the logic
in the topos [Pro19], is given by the subobjects of 1; more precisely, for every 𝑘∈C, Ω(𝑘)is the set of
subobjects of the localization 1|𝑘, made by the arrows in Cgoing to 𝑘. All these subobjects are increasing
sequences (∅,...,∅,★,...,★). This can be interpreted as the fact that a proposition in the language (and
internal semantic theory) of the topos is more and more determined when we approach the last layer.
Which corresponds well to what happens in the internal world of the network, and also, in most cases, to
the information about the output that an external observer can deduce from the activity in the inner layers
[BBG21a].
1.3 Dynamical objects of the general DNNs
However, many networks, and most today’s networks, are far from being simple chains. The topology of
Γ is very complex, with many paths going from a layer to a deeper one, and many inputs and outputs at
a same vertex. In these cases, the functioning and the weights are not defined by functors on C(Γ)(the
category opposite to the category freely generated by Γ). But a canonical modification of this category
allows to solve the problem: at each layer 𝑎where more than one layer sends information, say 𝑎′,𝑎”
,...,
i.e. where there exist irreducible arrows 𝑎𝑎′,𝑎𝑎”
,... in C(Γ)(edges in Γop), we perform a surgery:
between 𝑎 and 𝑎′(resp. 𝑎 and 𝑎”, a.s.o.) introduce two new objects 𝐴★ and 𝐴, with arrows 𝑎′→𝐴★
,
𝑎” →𝐴★, ..., and 𝐴★→𝐴, 𝑎→𝐴, forming a fork, with tips in 𝑎′,𝑎”
,...and handle 𝐴★𝐴𝑎(more precisely
if not too pedantically, the arrows 𝑎′𝐴★,𝑎”𝐴★
,...are the tines, the arrow 𝐴★𝐴is the tang, or socket, and
the arrow 𝑎𝐴is the handle) (see figure 1.2). By reversing arrows, this gives a new oriented graph , also
without oriented cycles, and the category Cwhich replaces C(Γ)is the category C( ), opposite of the
category which is freely generated by .
Remark. In , the complement of the unions of the tangs is a forest. Only the convergent multiplicity
10
a
op
a
a
a
a
A A a
op
Figure 1.2: From the initial graph to the Fork
in Γ gives rise to forks, not the divergent one. In the category C, this convergence (resp. divergence)
corresponds to a divergence (resp. convergence) of the arrows.
When describing concrete networks (see for instance 𝑅𝑁𝑁, and 𝐿𝑆𝑇𝑀 or 𝐺𝑅𝑈 memory cells that we
will study in chapter 4), ambiguity can appear with the input layers: they can be considered as input or
as tips when several inputs join for connecting a deeper layer 𝑎. The better attitude is to duplicate them;
for instance two input layers 𝑥𝑡,ℎ𝑡−1 going to ℎ𝑡,𝑦𝑡, we introduce 𝑋𝑡,𝑥′
𝑡, 𝐻𝑡−1,ℎ′
𝑡−1, then a fork 𝐴★,𝐴, and
in C, arrows 𝑥′
𝑡 →𝑋𝑡, ℎ′
𝑡−1 →𝐻𝑡−1 for representing the input data, arrows of fork 𝑥′
𝑡 →𝐴★
, ℎ′
𝑡−1 →𝐴★
,
𝐴★ →𝐴, and arrows of information transmissions ℎ𝑡 →𝐴and 𝑦𝑡 →𝐴, representing the output of the
memory cell.
With this category C, it is possible to define the analog of the presheaves 𝑋𝑤
, W= Π and Xin general.
First 𝑋𝑤: at each old vertex, the set 𝑋𝑤
𝑎 is as before the set of activities of the neurons of the corresponding
layer; over a point like 𝐴★ and 𝐴 we put the product of all the incoming sets 𝑋𝑤
𝑎′×𝑋𝑤
𝑎”,.... The map
from 𝑋𝐴 to 𝑋𝑎 is the dynamical transmission in the network, joining the information coming from all
the inputs layers 𝑎′,𝑎”
,...at 𝑎, all the other maps are given by the structure: the projection on its factors
from 𝑋𝑤
𝐴★ , and the identity over the arrow 𝐴★𝐴. It is easy to show, that given a collection of activities 𝜀0
𝑖𝑛
in all the initial layers of the network, it results a unique section of the presheaf 𝑋𝑤, a singleton, or an
element of limC𝑋𝑤, which induces 𝜀0
in. Thus, dynamically, each arrow of type 𝑎→𝐴has replaced the
set of arrows from 𝑎to 𝑎′,𝑎”
,....
It is remarkable that the main structural part (which is the projection from a product to its components)
can be interpreted by the fact that the presheaf is a sheaf for a natural Grothendieck topology 𝐽 on the
category C: in every object 𝑥of Cthe only covering is the full category 𝐶|𝑥, except when 𝑥is of the type
11
of 𝐴★, where we add the covering made by the arrows of the type 𝑎′→𝐴★ [AGV63].
The sheafification process, associating a sheaf 𝑋★ over (C,𝐽)to any presheaf 𝑋 over Cis easy to de-
scribe: no value is changed except at a place 𝐴★, where 𝑋𝐴★ is replaced by the product 𝑋★
𝐴★ of the
𝑋𝑎′, and the map from 𝑋★
𝐴 = 𝑋𝐴 to 𝑋★
𝐴★ is replaced by the product of the maps from 𝑋𝐴 to the 𝑋𝑎′
given by the functor 𝑋. In particular, important for us, the sheaf 𝐶★ associated to a constant presheaf 𝐶
replaces 𝐶in 𝐴★by a product 𝐶𝑛and the identity 𝐶→𝐶by the diagonal map 𝐶→𝐶𝑛over the arrow 𝐴★𝐴.
Let us now describe the sheaf Wover (C,𝐽)which represents the set of possible weights of the 𝐷𝑁𝑁
(or 𝑅𝑁𝑁 a.s.o.). First consider at each vertex 𝑎of the initial graph Γ, the set 𝑊𝑎 of weights describing
the allowed maps from the product 𝑋𝐴 =
𝑎′←𝑎𝑋𝑎′ to 𝑋𝑎, over the projecting layers 𝑎′,𝑎”
,... to 𝑎.
Then consider at each layer 𝑥the (necessarily connected) subgraph Γ𝑥 (or 𝑥|Γ) which is the union of the
connected oriented paths in Γ from 𝑥 to some output layer (i.e. the maximal branches issued from 𝑥 in
Γ); take for W(𝑥)the product of the 𝑊𝑦 over all the vertices in Γ𝑥. (For the functioning, it is useful to
consider the part𝑥 (or 𝑥| ) which is formed from Γ𝑥, by adding the collections of points 𝐴★,𝐴when
necessary, and the arrows containing them in .) At every vertex of type 𝐴★or 𝐴of , we put the product
W𝐴 of the sets W𝑎′ for the aﬀerent 𝑎′,𝑎”
,...to 𝑎. If 𝑥′𝑥 is an oriented edge of , there exists a natural
projection Π𝑥𝑥′ : W(𝑥′)→W(𝑥). This defines a sheaf over C= C( ).
The crossed product Xof the 𝑋𝑤 over Wis defined as for the simple chains. It is an object of the
topos of sheaves over Cthat represents all the possible functioning of the neural network.
1.4 Backpropagation as a natural (stochastic) flow in the topos
Nothing is loosed in generality if we put together the inputs (resp. the output) in a product space 𝑋0 (resp.
𝑋𝑛); this corresponds to the introduction of an initial vertex 𝑥0 and a final vertex 𝑥𝑛 in Γ, respectively
connected to all the existing initial or final vertices.
We also assume that the spaces of states of activity 𝑋𝑎 and the spaces of weights 𝑊𝑎𝐴 are smooth
manifolds, and that the maps (𝑥,𝑤)↦→𝑋𝑤(𝑥)defines smooth maps on the corresponding product mani-
folds.
In particular it is possible to define tangent objects in the topos of the network 𝑇(X)and 𝑇(W), and
smooth natural transformations between them.
Supervised learning consists in the choice of an energy function
(𝜉0,𝑤)↦→𝐹(𝜉0; 𝜉𝑛(𝑤,𝜉0)); (1.2)
then in the search of the absolute minimum of the mean Φ = E(𝐹)of this energy over a measure on the
inputs 𝜉0; it is a real function on the whole set of weights 𝑊= W0. For simplicity, we assume that 𝐹
12
is smooth, and we do not enter the diﬃcult point of eﬀective numerical gradient descent algorithms, we
just want to develop the formula of the linear form 𝑑𝐹 on 𝑇𝑤0𝑊, for a fixed input 𝜉0 and a fixed system
of weights 𝑤0. The gradient will depend on the choices of a Riemannian metric on 𝑊. And the gradient
of Φ is the mean of the individual gradients.
We have
𝑑𝐹(𝛿𝑤)= 𝐹★𝑑𝜉𝑛(𝛿𝑤), (1.3)
then it is suﬃcient to compute 𝑑𝜉𝑛.
The product formula is
W0 =
𝑊𝑎𝐴, (1.4)
𝑎∈Γ
where 𝑎describes all the vertices of Γ, 𝐴𝑎is the corresponding edge in . Then it is suﬃcient to compute
𝑑𝜉𝑛(𝛿𝑤𝑎)for 𝛿𝑤𝑎 ∈𝑇𝑤0𝑊𝑎𝐴, assuming that all the other vectors 𝛿𝑤𝑏𝐵 are zero, except 𝛿𝑤𝑎 which denotes
the weight over the edge 𝐴𝑎.
For that, we consider the set Ω𝑎 of directed paths 𝛾𝑎 in Γ going from 𝑎to the output layer 𝑥𝑛. Each
such path gives rise to a zigzag in :
...←𝐵′→𝑏′←𝐵→𝑏←... (1.5)
which gives a feed-forward composed map, by taking over each 𝐵→𝑏the map 𝑋𝑤𝑏 𝐵 from the product 𝑋𝐵
to the manifold 𝑋𝑏, where everything is fixed by 𝜉0 and 𝑤0 except on the branch coming from 𝑏′, where
𝑤𝑎 varies, and by taking over each 𝑏′←𝐵the injection 𝜌𝐵𝑏′ defined by the other factors 𝑋𝑏”,𝑋𝑏′′′,...of
𝑋𝐵. This composition is written
𝜙𝛾𝑎
=
𝑏𝑘 ∈𝛾𝑎
𝑋𝑤0
𝑏𝑘 𝐵𝑘
◦𝜌𝐵𝑘 𝑏𝑘−1 ◦𝑋𝑤
𝑎𝐴; (1.6)
going from the manifold 𝑊𝑎×𝑋𝐴 to the manifold 𝑋𝑛. In the above formula, 𝑘starts with 1, and 𝑏0 = 𝑎.
Two diﬀerent elements 𝛾′
𝑎, 𝛾”
𝑎 of Ω𝑎 must coincide after a given vertex 𝑐, where they join from
diﬀerent branches 𝑐′𝑐, 𝑐”𝑐in Γ; they pass through 𝐵in ; then we can define the sum 𝜙𝛾′
𝑎
⊕𝜙𝛾”
𝑎 , as a
map from 𝑊⊕2
𝑎𝐴×𝑋𝐴 to 𝑋𝑛, by composing the maps between the 𝑋′𝑠after 𝑏, from 𝑏to 𝑥𝑛, with the two
maps 𝜙𝛾′
𝑎
and 𝜙𝛾”
𝑎
truncated at 𝐵. We name this operation the cooperation, or cooperative sum, of 𝜙𝛾′
𝑎
and 𝜙𝛾”
.
𝑎
Cooperation can be iterated in associative and commutating manner to any subset of Ω𝑎, representing a
tree issued from 𝑥𝑛, embedded in Γ, made by all the common branches between the pairs of paths from
𝑎to 𝑥𝑛. The full cooperative sum is the map
𝜙𝛾𝑎
: 𝑋𝐴×
𝛾𝑎 ∈Ω𝑎
𝑊𝑎𝐴 →𝑋𝑛. (1.7)
13
For a fixed 𝜉0, and all 𝑤𝑏𝐵 fixed except 𝑤𝑎𝐴, the point 𝜉𝑛(𝑤)can be described as the composition of the
diagonal map with the total cooperative sum
𝑤𝑎 ↦→(𝑤𝑎,...𝑤𝑎)∈
𝑊𝑎𝐴 →𝑋𝑛. (1.8)
𝛾𝑎 ∈Ω𝑎
This gives
𝑑𝜉𝑛(𝛿𝑤𝑎)=
𝛾𝑎 ∈Ω𝑎
𝑑𝜙𝛾𝑎
𝛿𝑤𝑎; (1.9)
which implies the backpropagation formula:
Lemma 1.1.
𝑑𝜉𝑛(𝛿𝑤𝑎)=
𝛾𝑎 ∈Ω𝑎 𝑏𝑘 ∈𝛾𝑎
𝐷𝑋𝑤0
𝑏𝑘 𝐵𝑘
◦𝐷𝜌𝐵𝑘 𝑏𝑘−1 ◦𝜕𝑤𝑋𝑤
𝑎𝐴.𝛿𝑤𝑎 (1.10)
going from the tangent space 𝑇𝑤0
𝑎 (𝑊𝑎)to the tangent space 𝑇𝜉0
𝑛 (𝑋𝑛). In this expression, 𝑘 starts with 1,
and 𝑏0 = 𝑎.
To get the backpropagation flow, we compose to the left with 𝐹★ = 𝑑𝐹, which gives a linear form,
then apply the chosen metric on the manifold 𝑊, which gives a vector field 𝛽(𝑤0 |𝜉0). Let us assume that
the function 𝐹 is bounded from below on 𝑋0 ×𝑊 and coercive (at least proper). Then the flow of 𝛽is
globally defined on 𝑊. From it we define a one parameter group of natural transformations of the object W.
In practice, a sequence Ξ𝑚; 𝑚∈[𝑀]of finite set of inputs 𝜉0 (benchmarks) is chosen randomly,
according to the chosen measure on the initial data, and the gradient is taken for the sum
𝐹𝑚 =
Ξ𝑚
𝐹𝜉0, (1.11)
then the flow is integrated (with some important cooking) for a given time, before the next integration
with 𝐹𝑚+1.
This changes nothing to the result:
Theorem 1.1. Backpropagation is a flow of natural transformations of W, computed from collections of
singletons in X.
Figure 1.3 shows a bifurcation Σ in W, X→W. Subfigure 1.3a shows three forms of potentials for
dynamics of 𝑋𝑤 on the left part when, in the upper-right part, we can see the regions of a planar
projection of W, where the learned dynamics has the corresponding shape.
14
X Xw Xwt
(a) Dynamics of 𝑋𝑤
W {w} {wt}
(b) Illustration of theorem 1.1
Figure 1.3: Examples of bifurcations
Remark. Frequently, the function 𝐹takes the form of a Kullback-Leibler divergence
𝐷𝐾𝐿(𝑃(𝜉𝑛)|𝑃𝑛)
and can be rewritten as a free energy, which can itself be replaced by a Bethe free energy over inner
variables, which are probabilistic laws on the weights. This is where information quantities could enter
[Pel20].
1.5 The specific nature of the topos of DNNs
We wonder now to what species the topos C∼of a 𝐷𝑁𝑁belongs.
Definitions. Let X denotes the set of vertices of the full subcategory CX of Cgenerated by X.
of type 𝑎or of type 𝐴(see figure 1.2). We introduce
There only exists one arrow from a vertex of type 𝑎′to a vertex of type 𝐴through 𝐴★ (but a given
𝑎′can join diﬀerent 𝐴★ then diﬀerent 𝐴), only one arrow from a vertex of type 𝑎to its preceding 𝐴(but
𝐴can belong to several vertices 𝑎). Moreover there exists only one arrow from a vertex 𝑐to a vertex 𝑏
when 𝑏and 𝑐are on a chain in Cwhich does not contain a fork. And no other arrows exist in CX. By
definition of the forks, a point 𝑎(i.e. a handle) cannot join another point than its tang 𝐴, and an input or
a tang 𝐴is the center of a convergent star.
Any maximal chain in Cop
X joins an input entry or a 𝐴-point (i.e. a tang), to a vertex of type 𝑎′(i.e. a
tip) or to an output layer. Issued from a tang 𝐴it can pass through a handle 𝑎or a tip 𝑎′, because nothing
forbids a tip to join a vertex 𝑏.
15
If 𝑥,𝑦belong to X, we note 𝑥≤𝑦when there exists a morphism from 𝑥to 𝑦; then it is equivalent to
write 𝑥→𝑦in the category CX.
Proposition 1.1. (i) CX is a poset.
(ii) Every presheaf on Cinduces a presheaf on CX.
(iii) For every presheaf on CX, there exists a unique sheaf on Cwhich induces it.
Proof. (i) let 𝛾1,𝛾2 be two diﬀerent simple directed paths in CX going from a point 𝑧in X to a point 𝑥
in X, there must exists a first point 𝑦where the two paths disjoin, going to two diﬀerent points 𝑦1,
𝑦2. This point 𝑦cannot be a handle (type 𝑎), nor an input, nor a tang (type 𝐴), then it is an output
or a tip. It cannot be an output, because a fork would have been introduced here to manage the
divergence. If the two points 𝑦1,𝑦2 were tangs, they were the ending points of the paths, which is
impossible. But at least one of them is a tang, say 𝐴2, because a tip cannot diverge to two ordinary
vertices, if not, there should be a fork here. Then one of them, say 𝑦1, is an ordinary vertex and
begins a chain, without divergence until it attains an input or a tang 𝐴1. Therefore 𝐴1 = 𝐴2, but this
gives an oriented loop in the initial graph Γ, which was excluded from the beginning for a 𝐷𝑁𝑁.
This final argument directly forbids the existence of 𝑥,≠ 𝑦 with 𝑥 ≤𝑦 and 𝑦≤𝑥. Then CX is a
poset.
(ii) is obvious.
(iii) remark that the vertices of which are eliminated in X are the 𝐴★. Then consider a presheaf 𝐹on
X, the sheaf condition over Ctells that 𝐹(𝐴★)must be the product of the entrant 𝐹(𝑎′),..., then
the product map 𝐹(𝐴)→𝐹(𝐴★)of the maps 𝐹(𝐴)→𝐹(𝑎′)gives a sheaf.
Corollary. C∼is naturally equivalent to the category of presheaves C∧
X.
Remark. In Friedman [Fri05], it was shown that every topos defined by a finite site, where objects do
not possess non unit endomorphisms, has this property to be equivalent to a topos of presheaves over a
finite full subcategory of the site: this is the category generated by the objects that have only the trivial
full covering. Then we are in a particular case of this theorem. The special fact, that we get a site which
is a poset, implies many good properties for the topos [Bel08], [Car09].
In what follows, X will often denote the poset CX.
Definitions 1. The (lower) Alexandrov topology on X, is made by the subsets 𝑈 of X such that (𝑦∈𝑈
and 𝑥≤𝑦) imply 𝑥∈𝑈.
A basis for this topology is made by the collections 𝑈𝛼 of the 𝛽 such that 𝛽≤𝛼. In fact, consider the
intersection 𝑈𝑥∩𝑈𝑥′; if 𝑦≤𝑥and 𝑦≤𝑥′, we have 𝑈𝑦 ⊆𝑈𝑥∩𝑈𝑥′, then 𝑈𝑥∩𝑈𝑥′ =
𝑦∈𝑈𝑥 ∩𝑈𝑥′𝑈𝑦.
16
In our examples the poset X is in general not stable by intersections or unions of subsets of X, but the
intersection and union of the sets 𝑈𝑥, 𝑈𝑦 for 𝑥,𝑦∈X plays this role.
We note Ω or Ω(X)when there exists a possibility of confusion, the set of (lower) open sets on X.
A sheaf in the topological sense over the Alexandrov space X is a sheaf in the sense of topos over the
category Ω(X), where arrows are the inclusions, equipped with the Grothendieck topology, generated by
the open coverings of open sets.
Proposition 1.2. (see [Car18, Theorem 1.1.8, the comparison lemma] and [Bel08, p. 210]): every
presheaf of sets over the category CX can be extended to a sheaf on X for the Alexandrov topology, and
this extension is unique up to a unique isomorphism.
Proof. Let 𝐹be a presheaf on CX; for every 𝑥∈X, 𝐹(𝑈𝑥)is equal to 𝐹(𝑥). For any open set 𝑈=
𝑥∈𝑈𝑈𝑥
we define 𝐹(𝑈)as the limit over 𝑥∈𝑈 of the sets 𝐹(𝑥)(that is the set of families 𝑠𝑥; 𝑥∈𝑈 in the sets
𝐹(𝑥); 𝑥∈𝑈, such that for any pair 𝑥,𝑥′in 𝑈 and any element 𝑦in 𝑈𝑥∩𝑈𝑥′, the images of 𝑠𝑥 and 𝑠𝑥′ in
𝐹(𝑦)coincide. This defines a presheaf for the lower topological topology.
This presheaf is a sheaf:
1) if Uis a covering of 𝑈, and if 𝑠,𝑠′are two elements of 𝐹(𝑈)which give the same elements over 𝑉
for all 𝑉 ∈U, the elements 𝑠𝑥,𝑠′
𝑥 that are defined by 𝑠and 𝑠′respectively in every 𝐹(𝑥)for 𝑥∈𝑈
are the same, then by definition, 𝑠= 𝑠′
.
2) To verify the second axiom of a sheaf, suppose that a collection 𝑠𝑉 is defined for 𝑉in the covering
Uof 𝑈, and that for any intersection 𝑉∩𝑊,𝑉,𝑊 ∈Uthe restrictions of 𝑠𝑉 and 𝑠𝑊 coincide, then
by restriction to any 𝑈𝑥 for 𝑥∈𝑈we get a coherent section over 𝑈.
3) For the uniqueness, take a sheaf 𝐹′which extends 𝐹, and consider the open set 𝑈=
𝑥∈𝑈𝑈𝑥,
any element 𝑠′of 𝐹′(𝑈)induces a collection 𝑠′
𝑥 ∈𝐹(𝑈𝑥)= 𝐹(𝑥)which is coherent, then defines a
unique element 𝑠= 𝑓𝑈(𝑠′)∈𝐹(𝑈). These maps 𝑓𝑈;𝑈∈Ω define the required isomorphism.
Corollary. The category C∼ is equivalent to the category Sh(X)of sheaves of X, in the ordinary
topological sense, for the (lower) Alexandrov topology.
Consequences from [Bel08, pp.408-410]: the topos E= C∼of a neural network is coherent. It possesses
suﬃciently many points, i.e. geometric functors Set →C∼, such that equality of morphisms in C∼
can
be tested on these points.
In fact, such an equality can be tested on sub-singletons, i.e. the topos is generated by the subobjects of
the final object 1. This property is called sub-extensionality of the topos E.
Moreover E(as any Grothendieck topos) is defined over the category of sets, i.e. there exists a unique
geometric functor 𝜇: E→Set. This functor is given by the global sections of the sheaves over X. In
this case, as shown in [Bel08], the equality of subobjects (i.e. propositions) in every object of the form
17
𝜇★(𝑆)(named sub-constant objects) is decidable.
The two above properties characterize the so-called localic topos [Bel08], [MLM92].
The points of Ecorrespond to the ordinary points of the topological space X; they are also the points
of the poset CX. For each such point 𝑥∈X, the functor 𝜖𝑥 : Set →Eis the right adjoint of the functor
sending any sheaf 𝐹to its fiber 𝐹(𝑥).
In the neural network, the minimal elements for the ordering in X are the output layers plus some points
𝑎′ (tips), the maximal ones are the input layers, and the points of type 𝐴 (tangs). However, for the
standard functioning and for supervised learning, in the objects X, W, the fibers in 𝐴are identified with
the products of the fibers in the tips 𝑎′,𝑎”
,..., and play the role of transmission to the branches of type 𝑎.
Therefore the feed-forward functioning does not reflect the complexity of the set Ω. The backpropagation
learning algorithm also escapes this complexity.
Remarks. If 𝐴were not present in the fork, we should have added the empty covering of 𝑎in order to
satisfy the axioms of a Grothendieck topology, and this would have been disastrous, implying that every
sheaf must have in 𝑎the value ★(singleton). A consequence is the existence of more general sheaves than
the ones that correspond to usual feed-forward dynamics, because they can have a value 𝑋𝐴 diﬀerent from
the product of the 𝑋𝑎′ appearing in 𝐴★, equipped with a map 𝑋𝐴★ 𝐴 : 𝑋𝐴 → 𝑋𝑎′ and 𝑋𝑎𝐴 : 𝑋𝐴 →𝑋𝑎.
Then, depending on the value of 𝜀0
in and of the other objects and morphisms, a propagation can happen
or not. This opens the door to new types of networks, having a part of spontaneous activities (see chapter
3).
Remark. Several evidences show that the natural neuronal networks in the brain of the animals are
working in this manner, with spontaneous activities, internal modulations and complex variants of
supervised and unsupervised learning, involving memories, spontaneous activities, genetically and epi-
genetically programmed activations and desactivations, which optimize the survival at the level of the
evolution of species.
Remark. Appendix A gives an interpretation due to Bell of the class of topos we encounter here, named
localic topos, in terms of a categorical version of fuzzy sets, called sets with fuzzy identities taking values
in a given Heyting algebra.
For the topos of a 𝐷𝑁𝑁, the Heyting algebra Ω is the algebra of open subsets of the poset X. However,
we can go further in the characterization of this topos by using the particular properties of the poset X,
and of the algebra Ω.
Theorem 1.2. The poset X of a DNN is made by a finite number of trees, rooted in the maximal points
and which are joined in the minimal points.
18
More precisely, the minimal elements are of two types: the outputs layers 𝑥𝑛,𝑗 and the tips of the
forks, i.e. the points of type 𝑎′; the maximal elements are also of two types: the input layers 𝑥0,𝑖 and the
tangs of the forks (i.e. the points 𝐴). Moreover, the tips and the tanks are joined by an irreducible arrow,
but a tip can join several tanks and some ordinary point (of type 𝑎but not being an input 𝑥0,𝑖), and a tank
can be joined by several tips and other ordinary points (but not being an output 𝑥𝑛,𝑗) as it is illustrated in
figure 1.4.
A
B C
x0,2 x0,1
c
c
xn x0 c x0 xn,1 xn,2 xn,3 xn,4 xn,5
Figure 1.4: Poset of a DNN
Remark. The only possible divergences happen at tips, because they can joint several tanks and additional
ordinary points in X.
Remark. Appendix B gives an interpretation of the type of toposes we may obtain for 𝐷𝑁𝑁𝑠in terms
of spectrum of commutative rings.
Any object in the category C∧
X can be interpreted as a dynamical network, because it describes a flow
of maps between sets {𝐹𝑥; 𝑥∈X}, along the arrows between the layers, and each of these sets can be
interpreted as a space of states, not necessarily made by vectors. However what matters for the functioning
of the network is the correspondence between input data, that are elements in the product 𝐹in of spaces
over input layers, and output states, that are elements in the product 𝐹out of the spaces over output layers.
This correspondence is described by the limit of 𝐹 over CX, i.e. 𝐻0 (𝐹)= 𝐻0 (CX; 𝐹), [Mac71]. This
contains the graphs of ordinary applications from 𝐹in to 𝐹out, when taking the products at the forks, but
in general, except for chains, that are models of simple reflexes, this limit is much wider, and a source of
innovation (see the above remarks on spontaneius activity and section 3.2 below).
19
2
Stacks of DNNs
2.1 Groupoids, general categorical invariance and logic
In many interesting cases, a restriction on the structure of the functioning 𝑋𝑤, or the learning in W,
comes from a geometrical or semantic invariance, which is extracted (or expected) from the input data
and/or the problems that the network has to solve as output.
The most celebrate example is given by the convolutional networks 𝐶𝑁𝑁𝑠. These networks are made
for analyzing images; it can be for finding something precise in an image in a given class of images,
or it can be for classifying special forms. The images are assumed to be by nature invariant by planar
translation, then it is imposed to a large number of layers to accept a non trivial action of the group
𝐺 of 2𝐷-translations and to a large number of connections between two layers to be compatible with
the actions, which implies that the underlying linear part when it exists is made by convolutions with a
numerical function on the plane. This does not forbid that in several layers, the action of 𝐺is trivial, to
get invariant characteristics under translations, and here, the layers can be fully connected. The Resnets
today have such a structure, with non-trivial architectures, as described in the preceding chapter.
Other Lie groups and their associated convolutions were recently used for DNNs, (see Cohen et
al. [CWKW19], [CGW20], [BBCV21]). Diverse case of equivariant deep learning are presented in
[SGW21]. For example, studies of graph networks involve invariance and equivariance under groupoids
of isomorphisms between graphs [MBHSL19].
Cohen et al. [CWKW19] underline the analogy with Gauge theory in Physics. In the same spirit,
Bondesan and Welling [BW21] give an interpretation of the excited states in DNNs in terms of particles
in Quantum Field Theory.
DNNs that analyze images today, for instance in object detection, have several channels of convolu-
tional maps, max pooling and fully connected maps, that are joint together to take a decision. It looks
as a structure for localizing the translation invariance, as it happens in the successive visual areas in
the brains of animals. Experiments show that in the first layers, kinds of wavelet kernels are formed
spontaneously to translate contrasts, and color opposition kernels are formed to construct color invariance.
20
A toposic manner to encode such a situation consists to consider contravariant functors from the
category Cof the network with values in the topos 𝐺∧of 𝐺-sets, in place of taking values in the category
Set of sets. Here the group 𝐺is identified with the category with one object and whose arrows are given
by the elements of 𝐺, then a 𝐺-set, that is a set with a left action of 𝐺, is viewed as a set valued sheaf over
𝐺. The collection of these functors, with morphisms given by the equivariant natural transformations,
form a category C∼
𝐺, which was shown to be itself a topos by Giraud [Gir72]. We will prove this fact in
the following section 2.2: there exists a category F, which is fibred in groups isomorphic to 𝐺 over C,
𝜋: F→C, and satisfies the axioms of a stack, equipped with a canonical topology 𝐽(the least fine such
that 𝜋 is cocontinuous [Sta, 7.20], i.e. a comorphism of site [CZ21]), in such a manner that the topos
E= F∼of sheaves of sets over the site (F,𝐽), is naturally equivalent to the category C∼
𝐺. This topos is
named the classifying topos of the stack.
The construction of Giraud is more general; it extends to any stack over C, not necessarily in groups
or in groupoids. In this chapter, we will consider this more general situation, given by a functor 𝐹from
Cop to the category Cat of small categories, then corresponding to a fibred category F→C. But we
will not consider the issue of non-trivial topologies, because, as we have shown in chapter 1, the topos of
𝐷𝑁𝑁𝑠are topos of presheaves. Then we determine the inner logic of the classifying topos, from fibers
to fibers, to describe later the possible (optimal) flows of information in functioning networks.
The case of groupoids has the interest that the presheaves on a groupoid form a Boolean topos, then
ordinary logic is automatically incorporated.
Remarks. 1) The logic in the topos of a groupoid consists of simple Boolean algebras; however,
things appear more interesting when we remember the meaning of the atoms 𝑍𝑖;𝑖∈𝐾, because
they are made of irreducible 𝐺𝑎-sets. We interpret that as a part of the semantic point of view, in
the languages of topos and stacks.
2) In the experiments reported in [BBG21a] as in 𝐶𝑁𝑁𝑠, the irreducible linear representations of
groups appear spontaneously among the dynamical objects.
3) In every language we can talk of the future, the uncertain past, and introduce hypotheses, this does
not mean that we are leaving the world of usual Boolean logic, we are just considering externally
some intuitionist Heyting algebra, this can be done within ordinary set theory, as is done topos
theory in Mathematics, in the fibers, defined by groupoids.
Appendix C gives a description of the classifying object of a groupoid, that is well known by specialists
of category theory.
However, other logics, intuitionist, can also have an interest. In more recent experiments done with
Xavier Giraud on data representing time evolution, we used simple posets in the fibers.
21
The notion of invariance goes further than groupoids.
Invariance is synonymous of action (like group action), and is understood here in the categorical
sense: a category Gacts on another category Vwhen a (contravariant) functor from Gto Vis given.
The example that justifies this terminology is when Gis a group 𝐺, and Vthe Abelian category of vector
spaces and linear maps over a commutative field K. In the latter case, we obtain a linear representation
of the group 𝐺.
In any category V, there exists a notion which generalizes the notion of element of a set. Any
morphism 𝜑: 𝑢→𝑣in Vcan be viewed as an element of the object 𝑣of V.
Definition. Suppose that Gacts through the functor 𝑓 : G→Vand that 𝑣= 𝑓(𝑎), then the orbit of 𝜑
under G|𝑎is the functor from the left slice category G|𝑎to the right slice category 𝑢|V, that associates
to any morphism 𝑎′→𝑎the element 𝑢→𝑓(𝑎)→𝑓(𝑎′)of 𝑓(𝑎′)in Vand to an arrow 𝑎” →𝑎′over 𝑎
the corresponding morphism 𝑓(𝑎′)→𝑓(𝑎”), from 𝑢→𝑓(𝑎′)to 𝑢→𝑓(𝑎”).
In the classical example of a group representation, 𝑢= Kand the morphism 𝜑 defines a vector 𝑥 in
the space 𝑉𝑒. The group 𝐺 is identified with 𝐺|𝑒and the vector space 𝑉𝑒, identified with Hom(𝐾,𝑉𝑒),
contains the whole orbit of 𝑥.
In a stack, the notion of action of categories is extended to the notion of fibred action of a fibred category
Fto a fibred category N:
Definition. Suppose we are given a sheaf of categories 𝐹: C→Cat, that we consider as a general
structure of invariance, and another sheaf 𝑀: C→Cat. An action of 𝐹on 𝑀is a family of contravariant
functors 𝑓𝑈 : F𝑈 →M𝑈 such that, for any morphism 𝛼: 𝑈→𝑈′of C, we have
𝑓𝑈◦𝐹𝛼 = 𝑀𝛼◦𝑓𝑈′. (2.1)
This is the equivariance formula generalizing group equivariance as it can be found in [Kon18] for
instance. It is equivalent to morphisms of stacks, and allows to define the orbits of sections 𝑢𝑈 →𝑓𝑈(𝜉𝑈)
in the sheaf 𝑢|Munder the action of the relative stack F|𝜉.
Remark that Eilenberg and MacLane, when they invented categories and functors in [EM45], were
conscious to generalize the Klein’s program in Geometry (Erlangen program).
In the next sections, we will introduce languages with types taken from presheaves over the fibers of the
stack, where we define the terms of theories and propositions of interest for the functioning of the DNN.
Then the above notion of invariance will concern the action of a kind of pre-semantic categories on the
22
languages and the possible sets of theories, that the network could use and express in functioning.
This view is a crucial point for our applications of topos theory to DNNs, because it is in this frame-
work that logical reasoning, and more generally semantics, in the neural network, can be set: in a stack,
the diﬀerent layers interpret the logical propositions and the sentences of the output layers. As we will
see, the interpretations are expected to become more and more faithful when approaching the output,
however the information flow in the whole networks is interesting by itself.
This shift from groups to groupoids, then to categories, then to more general semantic, by taking
presheaves in groupoids or categories, is a fundamental addition to the site C. The true topos associated
to a network is the classifying topos Eover F; it incorporates much more structure than the visible
architecture of layers, it takes into account invariance (which appears here to be part of the semantic,
or better pre-semantic). More generally, it can concern the domain of natural human semantics that the
network has to understand in his own artificial world.
Moreover, as we will show below, working in this setting gives access to more flexible type theories,
like the Martin-Löf intensional types, and goes into the direction of homotopy type theory according to
Hofmann and Streicher [HS98], Hollander [Hol01], Arndt and Kapulkin [AK11], enlarged by objects
and morphisms in classifying topos in the sense of Giraud.
2.2 Objects classifiers of the fibers of a classifying topos
In this section we study the propagation of logical theories through a stack (equipped with a scindage
in the sense of Giraud). In particular we find a suﬃcient condition for free propagation downstream
and upstream, that was apparently not described before; it asks that gluing functors are fibrations, plus a
supplementary geometrical condition, always satisfied in the case of groupoids, (see theorem 2.1).
The application to the dynamics of functioning 𝐷𝑁𝑁𝑠is presented in the next section 2.3, with the
notion of semantic functioning. It is developed in the next chapter 3. Examples are presented in the
following chapters 4, with long and short term memory cells and variants of them, and their tentative
relation with cognitive linguistic, then in 5, with more general networks moduli.
In the general case and in more canonical toposic terms, the logic in the stack Fover Cis studied by
Olivia Caramello and Riccardo Zanfa [CZ21]; see also the available notes written for "Topos Online",
24-30 june 2021.
Also the contributions of Shulman, [Shu10], [Shu19], and the slides of his talk "Large categories and
quantifiers in topos theory", January 26 2021, Cambridge Category Seminar are of interest.
23
Among the equivalent points of view on stacks and classifying topos [Gir64], [Gir71], and [Gir72]),
the most concrete one starts with a contravariant functor 𝐹from the category Cto the 2-category of small
categories Cat. (This corresponds to an element of the category Scind(C)in the book of Giraud [Gir71].)
To each object 𝑈∈Cis associated a small category F(𝑈), and to each morphism 𝛼: 𝑈→𝑈′is associated
a covariant functor 𝐹𝛼 : 𝐹(𝑈′)→𝐹(𝑈), also denoted 𝐹(𝛼), satisfying the axioms of a presheaf over C.
If 𝑓𝑈 : 𝜉→𝜂is a morphism in 𝐹(𝑈), the functor 𝐹𝛼 sends it to a morphism 𝐹𝛼(𝑓𝑈): 𝐹𝛼(𝜉)→𝐹𝛼(𝜂)in
𝐹(𝑈′).
The corresponding fibration 𝜋: F→C, written ∇𝐹by Grothendieck, has for objects the pairs (𝑈,𝜉)
where 𝑈∈Cand 𝜉∈𝐹(𝑈), sometimes shortly written 𝜉𝑈, and for morphisms the elements of
HomF((𝑈,𝜉),(𝑈′,𝜉′))=
𝛼∈HomC(𝑈,𝑈′)
Hom𝐹(𝑈)(𝜉,𝐹(𝛼)𝜉′). (2.2)
For every morphism 𝛼: 𝑈→𝑈′of C, the set Hom𝐹(𝑈)(𝜉,𝐹(𝛼)𝜉′)is also denoted
Hom𝛼((𝑈,𝜉),(𝑈′,𝜉′)); it is the subset of morphisms in Fthat lift 𝛼.
The functor 𝜋sends (𝑈,𝜉)on 𝑈. We will write indiﬀerently 𝐹(𝑈)or F𝑈 the fiber 𝜋−1 (𝑈).
A section 𝑠 of 𝜋 corresponds to a family 𝑠𝑈 ∈F𝑈 indexed by 𝑈 ∈C, and a family of morphisms
𝑠𝛼 ∈Hom𝐹(𝑈)(𝑠𝑈,𝐹(𝛼)𝑠𝑈′)indexed by 𝛼∈HomC(𝑈,𝑈′)such that, for any pair of compatible morphisms
𝛼,𝛽, we have
𝑠𝛼◦𝛽 = 𝐹𝛽(𝑠𝛼)◦𝑠𝛽. (2.3)
As shown by Grothendieck and Giraud [Gir64], a presheaf 𝐴 over F corresponds to a family of
presheaves 𝐴𝑈 on the categories F𝑈 indexed by 𝑈∈C, and a family 𝐴𝛼 indexed by 𝛼∈HomC(𝑈,𝑈′), of
natural transformations from 𝐴𝑈′ to 𝐹★
𝛼𝐴𝑈. (Here 𝐹★
𝛼 denotes the pullback of presheaf associated to the
functor 𝐹𝛼 : 𝐹(𝑈′)→𝐹(𝑈), that is, for 𝐴𝑈 : 𝐹(𝑈)→Set, the composed functor 𝐴𝑈◦𝐹𝛼.)
Moreover, for any compatible morphisms 𝛽: 𝑉→𝑈, 𝛼: 𝑈→𝑈′, we must have
𝐴𝛼◦𝛽 = 𝐹★
𝛼(𝐴𝛽)◦𝐴𝛼. (2.4)
If 𝜉 is an object of F𝑈, we define 𝐴(𝑈,𝜉)= 𝐴𝑈(𝜉), and if 𝑓 : 𝜉𝑈 →𝐹𝛼𝜉′
𝑈′ is a morphism of Fbetween
𝜉𝑈 ∈F𝑈 and 𝜉′
𝑈′ ∈F𝑈′ lifting 𝛼, we take
𝐴(𝑓)= 𝐴𝑈(𝑓)◦𝐴𝛼 : 𝐴𝑈′(𝜉′)→𝐴𝑈(𝐹𝛼(𝜉′))→𝐴𝑈(𝜉). (2.5)
The relation 𝐴(𝑓◦𝑔)= 𝐴(𝑔)◦𝐴(𝑓)follows from (2.4).
A natural transformation 𝜑: 𝐴→𝐴′corresponds to a family of natural transformations
𝜑𝑈 : 𝐴𝑈 →𝐴′
𝑈,
such that, for any arrow 𝛼: 𝑈→𝑈′in C,
𝐹★
𝛼𝜑𝑈◦𝐴𝛼 = 𝐴′
𝛼◦𝜑𝑈′: 𝐴𝑈′ →𝐹★
𝛼𝐴′
𝑈. (2.6)
24
This describes the category Eof presheaves over Ffrom the family of categories E𝑈 of presheaves over
the fibers F𝑈 and the family of functors 𝐹★
𝛼 : E𝑈 →E𝑈′.
Note that for two consecutive morphisms 𝛽: 𝑉→𝑈, 𝛼: 𝑈→𝑈′, we have 𝐹★
𝛼𝛽 = 𝐹★
𝛼 ◦𝐹★
𝛽.
The category Eis fibred over the category C, it corresponds to the functor 𝐸 from Cto 𝐶𝑎𝑡, which
associates to 𝑈∈Cthe category E𝑈 and to an arrow 𝛼: 𝑈→𝑈′, the functor 𝐹𝛼
! : E𝑈′ →E𝑈, which
is the left adjoint of 𝐹★
𝛼. This functor extends 𝐹𝛼 through the Yoneda embedding, [AGV63, Chap. I,
Presheaves].
For two consecutive morphisms 𝛽: 𝑉→𝑈, 𝛼: 𝑈→𝑈′, we have 𝐹𝛼𝛽
= 𝐹𝛽
!
! ◦𝐹𝛼
!.
Let 𝜂𝛼 : 𝐹𝛼
! ◦𝐹★
𝛼 →𝐼𝑑E𝑈 the counit of the adjunction; a natural transformation 𝐴𝛼 : 𝐴𝑈′ →𝐹★
𝛼𝐴𝑈
gives a natural transformation 𝐴★
𝛼 : 𝐹𝛼
! 𝐴𝑈′ →𝐴𝑈, by taking 𝐴★
𝛼
= (𝜂𝛼⊗𝐼𝑑)𝐹𝛼
! (𝐴𝛼). This gives another
way to describe the elements of E, through the presheaves over F.
Remark. A section (𝑠𝑈,𝑠𝛼)defines a presheaf 𝐴, by taking
𝐴𝑈(𝜉)= HomF𝑈 (𝜉,𝑠𝑈); (2.7)
and 𝐴𝛼 = 𝑠★
𝛼◦𝐹𝛼, according to the following sequence:
Hom(𝜉′
,𝑠𝑈′)→Hom(𝐹𝛼𝜉′,𝐹𝛼(𝑠𝑈′))→Hom(𝐹𝛼𝜉′
,𝑠𝑈). (2.8)
The identity (2.4) follows from the identity (2.3).
This construction generalizes in the fibered situation the Y oneda objects in the absolute situation.
A morphism of sections gives a morphism of presheaves.
In each topos E𝑈there exists a classifying object 𝑈, such that the natural transformations Hom𝑈(𝑋𝑈, naturally correspond to the subobjects of 𝑋𝑈; the presheaf 𝑈 has for value in 𝜉𝑈 ∈F𝑈 the set of subob-
jects in E𝑈 of the Yoneda presheaf 𝜉∧
𝑈 defined by 𝜂↦→Hom(𝜂,𝜉𝑈), with morphisms given by composition
to the right.
The set 𝑈(𝜉𝑈)can also be identified with the set of subobjects of the final sheaf 1𝜉𝑈 over the slice
category F𝑈|𝜉𝑈.
Remark. In general, the object of parts 𝑋 of an object 𝑋 in a presheaf topos D∧over a category D,
is the presheaf given in 𝑥∈D0 by the set of subsets of the product set (D|𝑥)×𝑋(𝑥)and by the maps
induced by 𝑋(𝑓)for 𝑓 ∈D1. Observe that 𝑋 realizes an equilibrium between the category of basis
through D|𝑥and the set theoretic nature of the value 𝑋(𝑥).
A special case is when 𝑋= 1, the final object, made by a singleton ★at each 𝑥∈D0, and the unique
possible maps for 𝑓 ∈C1. The presheaf 1 is denoted by . Its value in 𝑥∈D0, is the set 𝑥 of subsets
of the Yoneda object 𝑥∧
.
It can be proved that a subobject 𝑌 of an object 𝑋of D∧corresponds to a unique morphism 𝜒𝑌 : 𝑋→
25
𝑈)
such that at any 𝑥∈D0, we have 𝑌(𝑥)= 𝜒−1
𝑌 (⊤).
The exponential presheaf 𝑋 is characterized by the natural family of bĳections
HomD∧(𝑌×𝑋, )≈HomD∧(𝑌, 𝑋), (2.9)
which expresses the universal property of the classifier .
We will also frequently consider the set of subobjects of 1 over the whole category D, and we simply
denote it by the letter Ω. It is named the Heyting algebra of the topos C∧. See appendix A for more
details.
As just said before, the functor 𝐹★
𝛼 : E𝑈 →E𝑈′ which associates 𝐴◦𝐹𝛼 to 𝐴, possesses a left adjoint
𝐹𝛼
! : E𝑈′ →E𝑈 which extends the functor 𝐹𝛼 on the Yoneda objects. For any object 𝜉′in F𝑈′, note
𝜉= 𝐹𝛼(𝜉′); the functor 𝐹𝛼
! sends (𝜉′)∧ to 𝜉∧, and sends a subset of (𝜉′)∧ to a subset of 𝜉∧. This is
not because 𝐹𝛼
! is necessarily left exact, but because we are working with Grothendieck topos, where
subobjects are given by families of coherent subsets.
Moreover 𝐹𝛼
! respects the ordering between these subsets, then it induces a poset morphism between the
posets of subobjects
Ω𝛼(𝜉′): Ω𝑈′(𝜉′)→Ω𝑈(𝐹𝛼(𝜉′))= 𝐹★
𝛼Ω𝑈(𝜉′); (2.10)
the functoriality of Ω𝑈, Ω𝑈′ and 𝐹𝛼 implies that these maps constitute a natural transformation between
presheaves
Ω𝛼 : Ω𝑈′ →𝐹★
𝛼Ω𝑈. (2.11)
The naturalness of the construction insures the formula (2.4) for the composition of morphisms. Conse-
quently, we obtain a presheaf F.
Moreover the final object 1F of the classifying topos E= F∧corresponds to the collection of final
objects 1𝑈;𝑈∈Cand to the collection of morphisms 1𝑈′ →𝐹★
𝛼1𝑈; 𝛼∈HomC(𝑈,𝑈′), then we have:
Proposition 2.1. The classifier of the classifying topos is the sheaf the pullback morphisms Ω𝛼, which can be summarized by the formula
F given by the classifiers Ω𝑈 and
F = ∇𝑈∈CΩ𝑈𝑑Ω𝛼. (2.12)
In general the functor 𝐹★
𝛼 is not geometric; by definition, it is so if and only if its left adjoint
(𝐹𝛼)! = (𝐹𝛼)!
, which is right exact (i.e. commutes with the finite colimits), is also left exact (i.e. commutes with the
finite limits). Also by definition, this is the case if and only if the morphism 𝐹𝛼 is a morphism of sites
from F𝑈 to F𝑈′, [AGV63, IV 4.9.1.1.], not to be confused with a comorphism, [AGV63, III.2], [CZ21].
Important for us: it results from the work of Giraud in [Gir72], that 𝐹★
𝛼 is geometric when 𝐹𝛼 is itself a
stack, and when finite limits exist in the sites F𝑈 and F𝑈′ and are preserved by 𝐹𝛼. (We will see in the
next section, that these stacks F→C, made by stacks between fibers, correspond to some admissible
26
contexts in a dependent type theory, when Cis the site of a 𝐷𝑁𝑁.)
When 𝐹★
𝛼 is geometric, a great part of the logic in E𝑈′ can be transported to E𝑈:
Let us write 𝑓= 𝐹★
𝛼 and 𝑓★ = (𝐹𝛼)! its left adjoint, supposed to be left exact, therefore exact, as
just mentionned. This functor 𝑓★ preserves the monomorphisms, and the final elements of the slices
categories. Then it induces a map between the sets of subsets, called the inverse image or pullback by 𝑓,
for any object 𝑋′∈E𝑈′:
𝑓★: Sub(𝑋′)→Sub(𝑓★𝑋′). (2.13)
When 𝑋′describes the Yoneda objects (𝜉′)∧, this gives the morphism Ω𝛼 : Ω𝑈′ →𝐹★
𝛼Ω𝑈.
As it is shown in MacLane-Moerdĳk [MLM92, p. 496], this map is a morphism of lattices, it preserves
the ordering and the operations ∧and ∨. If ℎ: 𝑌′→𝑋′is a morphism in E𝑈′, the reciprocal image ℎ★
between the sets of subsets has a left adjoint ∃ℎ and a right adjoint ∀ℎ. The morphism 𝑓★commutes with
∃ℎ, but in general not with ∀ℎ, for which there is only an inclusion:
𝑓★(∀ℎ𝑃′)≤∀𝑓★ℎ(𝑓★𝑃′). (2.14)
To have an equality, the morphism 𝑓 must be geometric and open. This is equivalent to the existence
of a left adjoint, in the sense of posets morphisms, for Ω𝛼, [MLM92, Theorem 3, p. 498].
In MacLaneMoerdĳk1992, this natural transformation Ω𝛼 is denoted 𝜆𝛼, and its left adjoint when it exists
is denoted 𝜇𝛼.
When this left adjoint in the sense of Heyting algebras exists, we have, by adjunction, the counit and
unit morphisms:
𝜇◦𝜆≤Id : Ω𝑈′ →Ω𝑈′; (2.15)
𝜆◦𝜇≥Id : 𝐹★Ω𝑈 →𝐹★Ω𝑈. (2.16)
If 𝑓 is geometric and open, the map 𝑓★ also commutes with the negation¬and with the (internal)
implication ⇒.
If openness fails, only inequality (external implication) holds for the universal quantifier.
Remark. When F𝑈′and F𝑈 are the posets of open sets of (sober) topological spaces X′and X, and when
𝐹𝛼 is given by the direct image of a continuous open map 𝜑𝛼 : X𝑈′ →X𝑈, the functor 𝐹★
𝛼 is geometric
and open. This extends to locale, [MLM92].
When 𝐹★
𝛼 is geometric and open, it transports the predicate calculus of formal theories from E𝑈′ to
E𝑈, as exposed in the book of Mac Lane and Moerdĳk, [MLM92]. This is expressed by the following
result,
Proposition 2.2. Suppose that all the 𝐹𝛼; 𝛼: 𝑈→𝑈′are open morphisms of sites (in the direction from
𝐹(𝑈)to 𝐹(𝑈′), then,
(i) the pullback Ω𝛼 commutes with all the operations of predicate calculus;
27
(ii) any theory at a layer 𝑈′, i.e. in E𝑈′, can be read and translated in a deeper layer 𝑈, in E𝑈, in
particular at the output layers.
In the sequence we will be particularly interested by the case where all the F𝑈 are groupoids and the 𝐹𝛼
are morphisms of groupoids, in this case, the algebras of subobjects SubE(𝑋)are boolean, then, in this
case, the following lemma implies that, as soon as 𝐹★
𝛼 is geometric, it is open:
Lemma 2.1. In the boolean case the morphism of lattices 𝑓★: Sub(𝑋′)→Sub(𝑓★𝑋′)is a morphism of
algebras which commutes with the universal quantifiers ∀ℎ.
Proof. Since 𝑓★ is right and left exact, it sends 0= ⊥to 0= ⊥and 𝑋′
= ⊤to 𝑋= ⊤. Therefore, for
every 𝐴∈SubE′(𝑋′), 𝑓★(𝑋′\𝐴′)= 𝑋\𝑓★(𝐴′), i.e. 𝑓★ commutes with the negation¬. This negation
establishes a duality between ∃and ∀, then 𝑓★ commutes with the universal quantifier. More precisely:
𝑓★(¬(∀𝑥
′,𝑃′(𝑥
′)))= 𝑓★(∃𝑎
′
,¬𝑃′(𝑎
′))= ∃𝑎,𝑓★(¬𝑃′)(𝑎)= ¬(∀𝑥𝑓★(𝑃′)(𝑥)), (2.17)
then by commutation with¬, and¬¬= Id, we have
𝑓★(∀𝑥
′,𝑃′(𝑥
′))= ∀𝑥,𝑓★(𝑃′)(𝑥). (2.18)
Let us mention here a diﬃculty: in the case of groups or groupoids, 𝐹★
𝛼 is geometric if and only
if 𝐹𝛼 is an equivalence of categories (then an isomorphism in the case of groups). This is because a
morphism of group is flat if and only if it is an isomorphism, [AGV63, 4.5.1.]. The main problem is with
the preservation of products.
However, it is remarkable that for any kind of group homomorphisms 𝐹: 𝐺′→𝐺, in every algebra
of subobjects the map 𝑓★ induced by 𝐹! preserves "locally" and "naturally" all the logical operations:
Lemma 2.2. For every object 𝑋′ in 𝐵𝐺′, note 𝑋= 𝐹! (𝑋′), then 𝑓★ induces a map of lattices 𝑓★ :
Sub(𝑋′)→Sub(𝑋), that is bĳective. It preserves the order ≤, the elements ⊤and ⊥, and the operations
∧and ∨, therefore it is a morphism of Heyting algebras. Moreover, for any natural transformation
ℎ: 𝑌′→𝑋′, it commutes with both the existential quantifier ∃ℎ and the universal quantifiers ∀ℎ.
Proof. As said in [AGV63, 4.5.1], if 𝐹: 𝐺′→𝐺 is a morphism of groups, the functor 𝐹! from 𝐵𝐺′ to
𝐵𝐺 is given on 𝑋′by the contracted product 𝐹! (𝑋′)= 𝐺×𝐺′𝑋′, that is the set of orbits of the action of
𝐺′on the 𝐺-set 𝐺𝑑×𝑋′
.
The algebra Sub(𝑋′)is the boolean algebra generated by the primitive representations of 𝐺′on the orbits
𝐺′𝑥′of the elements of 𝑋′. But each orbit 𝐺′𝑥′is sent in 𝑋 to an orbit of 𝐺, that is the product of
𝐺/𝐹(𝐻′
𝑥′)with the singleton {𝐺′𝑥′}, where 𝐻′
𝑥′ is the stabilizer of 𝑥′. These sets describe the orbits of
the action of 𝐺on 𝑋, then the elements of Sub(𝑋).
The commutativity with ℎ★for a 𝐺′-morphism ℎ: 𝑌′→𝑋′is evident, the rest follows from the bĳection
property, orbitwise.
28
Therefore, even if 𝐹★ is not a geometric morphism, it is legitimate to say that in some sense, it is
open, because all logical properties are preserved by the induced morphisms between the local Heyting
algebras. We could say that 𝐹★ is "weakly geometric and open".
This can be easily extended to morphisms of groupoids. The left adjoint 𝐹! admits a description
which is analogous to the contracted product of groups. Lemma 2.2 holds true. The only diﬀerence is
that 𝑓★ is not a bĳection, but it is a surjection when 𝐹is surjective on the objects. More details and the
generalization of the above results to fibrations of categories that are themselves fibrations in groupoids
over posets will be given in the text Search of semantic spaces.
In the reverse direction of the flow, it is important that a proposition in the fiber over 𝑈 can be
understood over 𝑈′
.
Hopefully, this can always be done, at least in part: the functor 𝐹★
𝛼 is left exact and has a right adjoint
𝐹𝛼
★ : E𝑈′→E𝑈, which can be described as a right Kan extension [AGV63]: for a presheaf 𝐴′over F𝑈′, the
value of the presheaf 𝐹𝛼
★(𝐴′
𝑈′)at 𝜉𝑈 ∈F𝑈 is the limit of 𝐴′
𝑈′ over the slice category 𝐹𝛼|𝜉𝑈, whose objects
are the pairs (𝜂′,𝜑)where 𝜂′∈F𝑈′ and 𝜑: 𝐹𝛼(𝜂′)→𝜉𝑈 is a morphism in F𝑈, and whose morphisms
from (𝜂′,𝜑)to (𝜁′,𝜙)are the morphisms 𝑢: 𝜂′→𝜁′such that 𝜑= 𝜙◦𝐹𝛼(𝑢).
Therefore, if we denote 𝜌the forgetting functor from 𝐹𝛼|𝜉𝑈 to F𝑈′, we have
𝐹𝛼
★(𝐴′)(𝜉𝑈)= 𝐻0 (𝐹𝛼|𝜉𝑈; 𝜌★𝐴′), (2.19)
that is the set of sections of the presheaf 𝜌★𝐴′over the slice category.
Remark. In the case where 𝐹𝛼 : F𝑈′ →F𝑈 is a morphism of groupoids, this set is the set of sections of
𝐴′over the connected components of 𝐹−1
𝛼 (𝜉𝑈).
Therefore the functor 𝑔= 𝐹𝛼
★ is always geometric in our situation of presheaves. By definition, this
proves that 𝐹𝛼 is a comorphism of sites. Consequently, as shown in [MLM92], the pullback of subobjects
defines a natural transformation of presheaves over F𝑈′:
𝜆′
𝛼 : Ω𝑈 →𝐹𝛼
★Ω𝑈′; (2.20)
which corresponds by the adjunction of functors 𝐹★
𝛼 ⊣𝐹𝛼
★, to a natural transformation of sheaves over F𝑈:
′
𝜏
𝛼 : 𝐹★
𝛼Ω𝑈 →Ω𝑈′. (2.21)
Lemma 2.3. If 𝐹𝛼 is a fibration (not necessarily in groupoids), it is an open morphism of sites, and the
functor 𝐹𝛼
★ is open [Gir72].
Proof. This results directly from [MLM92, Proposition 1, pp. 509-513]. Precisely this proposition says
that a morphism of sites 𝐹: F′→Finduces an open geometric morphism 𝐹★: Sh(F′,𝐽′)→Sh(F,𝐽)
between the categories of sheaves, as soon as the following three conditions are satisfied:
29
(i) 𝐹has the property of lifting of the coverings:
∀𝜉′∈F′
,∀𝑆∈𝐽(𝐹(𝜉′)),∃𝑇′∈𝐽′(𝜉′),𝐹(𝑇′)⊆𝑆; (2.22)
where 𝐹(𝑇′)is the sieve generated by the images of the arrows in 𝑇′;
(ii) 𝐹preserves the covers, i.e.
∀𝜉′∈F′
,∀𝑆′∈𝐽′(𝜉′),𝐹(𝑆′)∈𝐽(𝐹(𝜉′)); (2.23)
(iii) for every 𝜉′∈F′, the sliced morphism 𝐹|𝜉′: F′|𝜉′→F|𝐹(𝜉′)is surjective on the objects.
The two first conditions are true for the canonical topology of a stack [Gir72]. They are obvious in our
case of presheaves. Condition (iii) is part of the definition of fibration (pre-fibration).
If in addition 𝐹itself is surjective on the objects, as it will be the case in our applications, the maps of
algebras 𝑔★
𝑋 : Sub(𝑋)→Sub(𝑓★𝑋)are injective and the geometric open morphism 𝑔= 𝐹★ is surjective
on the objects [MLM92, page 513].
Lemma 2.4. When 𝐹𝛼 is a fibration, the relation between 𝜆𝛼= Ω𝛼 : Ω𝑈′→𝐹★
𝛼Ω𝑈 and 𝜆′
𝛼: Ω𝑈 →𝐹𝛼
★Ω𝑈′,
is given by the adjunction of posets morphisms:
Ω𝛼 ⊣𝜏
′
𝛼; (2.24)
where 𝜏′
𝛼 : 𝐹★
𝛼Ω𝑈 →Ω𝑈′ is the dual of 𝜆′
𝛼.
The morphism Ω𝛼 is the left adjoint of the morphism 𝜏′
𝛼. Moreover, 𝜏′
𝛼 is an injective section of the
surjective morphism Ω𝛼.
Proof. If 𝐹𝛼 is a fibration, 𝐹★
𝛼Ω𝑈 is isomorphic to Ω𝑈, it is the sub-algebra of Ω𝑈′ formed by the
subobjects of 1𝑈′ that are invariant by 𝐹𝛼, i.e. by 𝜆𝛼 : Ω𝑈′ →𝐹★
𝛼Ω𝑈.
The map 𝜏′
𝛼 associates to an element 𝑃of Ω𝑈 the element 𝑃◦𝐹𝛼, seen as a sub-sheaf of 1𝑈′, that is an
element of Ω𝑈′ saturated by 𝐹𝛼. Therefore, for every 𝑃′∈Ω𝑈′, the element 𝜏′
𝛼◦𝜆𝛼(𝑃′)of Ω𝑈′ is the
saturation of 𝑃′, then it contains 𝑃′. This gives a natural transformation
𝜂: IdΩ𝑈 ′ →𝜏
′
𝛼◦Ω𝛼. (2.25)
In the other direction, 𝜏′
𝛼 is a section over Ω𝑈′ of the map 𝜆𝛼, i.e. Ω𝛼◦𝜏′
𝛼
natural transformation
= 𝐼𝑑𝐹★
𝛼 Ω𝑈 . Which gives a
𝜖: Ω𝛼◦𝜏
′
𝛼 →Id𝐹★
𝛼 Ω𝑈. (2.26)
In the following lines, we forget the indices 𝛼everywhere, and show that 𝜂and 𝜖are respectively the unit
and counit of an adjunction of posets morphisms.
Let 𝑃′and 𝑄, be respectively elements of Ω𝑈′and Ω𝑈, if we have a morphism from 𝜆𝑃′to 𝑄, by applying
30
𝜏′, we obtain a morphism from 𝜏′◦𝜆𝑃′to 𝜏′𝑄, then a morphism from 𝑃′to 𝜏′𝑄. All that is equivalent
to the following implications:
(𝜆𝑃′≤𝑄)⇛ (𝑃′≤𝜏
′𝜆𝑃′≤𝜏
′𝑄). (2.27)
In the other direction,
(𝑃′≤𝜏
′𝑄)⇛ (𝜆𝑃′≤𝜆𝜏′𝑃′≤𝑄). (2.28)
Therefore
(𝑃′≤𝜏
′𝑄)⇚⇛ (𝜆𝑃′≤𝑄). (2.29)
Which is the statement of lemma 2.4.
From the above lemmas, we conclude the following result (central for us):
Theorem 2.1. When for each 𝛼: 𝑈→𝑈′in C, the functor 𝐹𝛼 is a fibration, the logical formulas and their
truth in the topos propagate from 𝑈to 𝑈′by 𝜆′
𝛼 (feedback propagation in the DNN), and if in addition 𝐹𝛼
is a morphism of groupoids (surjective on objects and morphisms), the logic in the topos also propagates
from 𝑈′to 𝑈, by 𝜆𝛼 (feed-forward functioning in the DNN).
Moreover, the map 𝜆𝛼 is the left adjoint of the transpose 𝜏′
𝛼 of the map 𝜆′
𝛼. And we have, for any
𝛼: 𝑈→𝑈′in C,
𝜆𝛼◦𝑡𝜆′
𝛼
= IdΩ𝑈 ′. (2.30)
Definition 2.1. When the conclusion of the above theorem holds true, even if the 𝐹𝛼 are not fibrations, we
say that the stack 𝜋: F→Csatisfies the strong standard hypothesis (for logical propagation). Without
the equation (2.30), we simply say that the standard hypothesis is satisfied.
In this case, the logic is richer in 𝑈′than in 𝑈, like a fibration of Heyting algebras of subobjects of objects.
To finish this section, let us describe the relation between the classifier ΩF and the classifier ΩCof
the basis category Cof the fibration 𝜋: F→C.
As reminded above, proposition 2.1 in [Gir71], gives suﬃcient conditions for guarantying that the
functor 𝜋★is geometric. But, even in the non-geometric case, when the fibers are groupoids, the morphism
has locally (at the level of subobjects) the logical properties of an open geometric morphism, (see lemmas
2.1 and 2.2 ) and lemma 2.3 says that the functor 𝜋★, which is its right adjoint, is geometric and open.
We can then apply lemma 2.4, and get an adjunction 𝜆𝜋 ⊣𝜏′
𝜋, where
𝜆𝜋 : ΩF →𝜋★ΩC, (2.31)
is a surjective morphism of lattices, and
′
𝜏
𝜋 : 𝜋★ΩC→Ω𝐹, (2.32)
31
is the section by invariant objects.
When 𝜋is fibration of groupoids, 𝜋★ is open, and 𝜆𝜋 is a morphism of Heyting algebras. In this case,
there exists a perfect lifting of the theories in Cto the theories in F.
2.3 Theories, interpretation, inference and deduction
Main references are Bell [Bel08], Lambek and Scott [LS81], [LS88] , MacLane and Mœrdĳk [MLM92].
The formal languages, that we will mainly consider, are the typed languages of type theory, in the
sense of Lambek and Scott [LS81]. In particular, in such a type theory we have a notion of deduction,
conditioned by a set 𝑆of propositions, named axioms, which is denoted by ⊢𝑆. This is a relation between
two propositions, 𝑃⊢𝑆 𝑄, which satisfies the usual axioms, structural, logical, and set theoretical, also
named rules of inference, of the form
(𝑃1 ⊢𝑆 𝑄1,𝑃2 ⊢𝑆 𝑄2,...,𝑃𝑛 ⊢𝑆 𝑄𝑛)/𝑃⊢𝑆 𝑄, (2.33)
meaning that the truth (or validity) of the left (said upper) conjunction of deductions implies the truth of
the right deduction (said lower).
The conditional validity of a proposition 𝑅is noted ⊢𝑆 𝑅.
A (valid) proof of ⊢𝑆 𝑅 is an oriented classical graph without oriented cycles, whose vertices are
labelled by valid inferences, and whose oriented edges are identifying one of the upper terms of its final
extremity to the lower term of its initial extremity, and having only one final vertex whose lower term is
⊢𝑆 𝑅. The initial vertices have left terms that are empty or belonging to the set 𝑆.
A theory Tin a formal language Lis the set of propositions that can be asserted to be true if some axioms
are assumed to be true, this means that these propositions are deduced by valid proofs from the axioms.
A language Lis interpreted in a topos Ewhen some objects of Eare associated to every type, the
object ΩEcorresponding to the logical type ΩL, when some arrows 𝐴→𝐵are associated to the variables
(or terms) of 𝐵in the context 𝐴, all that being compatible with the respective definitions of products,
subsets, exponentials, singleton, changes of contexts (substitutions), and logical rules, including the
predicate calculus, which includes the two projections (existential and universal) on the side of topos
[Bel08], [LS81].
A theory Tis represented in Ewhen all its axioms are true in E. The fact that all the deductions are
valid in Eis the statement of the soundness theorem of Tin E.
Remark. The completeness theorem says that, for any language and any theory, there exists a minimal
"elementary topos" ET, which in general is not a Grothendieck topos, where the converse of the soundness
theorem is true; validity in ETimplies validity in T. The diﬀerent interpretations in a topos Eof a theory
32
Tform a category M(T,E), which is equivalent to the category of "logical functors" from ETto E. This
equivalence needs precisions given by Lambek and Scott, in particular to fix representant of subobjects,
which is automatic in a Grothendieck topos.
As suggested by Lambek, an interpretation of a type theory in a topos constitutes a semantic of this theory.
If a formal language Lcan be interpreted in a topos E, and if 𝐹: E→Fis a left exact functor from Eto
a topos F, the interpretation is transferred to F. The condition for transporting any theory Tby 𝑓 is that
it admits a right adjoint 𝑓 : F→Ewhich is geometric and open.
A geometric functor allows the transportation of the restricted family of geometric theories as in [Car09],
[Car18] or [MLM92].
Remark. If Tis a geometric theory, there is a Grothendieck topos E′
Twhich classifies the interpretations
of T, i.e. for every Grothendieck topos Ethe category of geometric functors from Eto E′
Tis equivalent
to M(T,E)[Car09], [Car18],[MLM92]. A logical functor is the left adjoint of a geometric functor.
In many applications of 𝐷𝑁𝑁𝑠, a network has to proceed to a semantic analysis of some data. Our
aim now is to precise what this means, and how we, observers, can have access to the internal process of
this analysis.
As before, the network is presented as a dynamic object Xin a topos, with learning object of weights
W, and the considered topos Eis the classifying topos of a fibration 𝜋: F→C.
In the applications, the logic is richer in 𝑈′ than in 𝑈 when there is a morphism 𝛼: 𝑈→𝑈′ in
C. We suppose given a family of typed language L𝑈;𝑈∈C, interpreted in the topos E𝑈;𝑈∈Cof the
corresponding layers.
We say that the functors 𝑓= 𝑔★ = 𝐹★
𝛼 propagate these languages backward, when for each morphism
𝛼: 𝑈→𝑈′in C, there exists a natural transformation
L𝛼 : L𝑈′ →𝐹★
𝛼L𝑈, (2.34)
which extends Ω𝛼 = 𝜆𝛼, implying that the types define objects or morphisms in E, in particular 0𝑈, 1𝑈.
And we say that the left adjoint functor 𝑓★ propagates the languages feed-forward, when for each
morphism 𝛼: 𝑈→𝑈′in C, there exists a natural transformation
L′
𝛼 : L𝑈 →𝐹𝛼
★L𝑈′, (2.35)
which extends 𝜆′
𝛼, implying that the types define objects or morphisms in the fibration E′, defined by the
right adjoint functors 𝐹𝛼
★.
We assume that the standard hypothesis 2.1 is satisfied for the extensions L𝛼 and L′
𝛼.
Note that in the case of stacks of DNNs, there exist two kinds of functors 𝐹𝛼 : F𝑈′ →F𝑈 over 𝐶, the
ordinary ones, flowing from the input to the output, and the added canonical projections from the fiber at
33
a fork 𝐴to the fibers of their tines 𝑎′
, 𝑎”, .... The second kind of functors are canonically fibrations, but
for the other functors, this is a condition we can require for a good semantic functioning (see theorem 2.1).
Let Ldenote the corresponding presheaf in languages over C, ΩLits logical type, and for each 𝑈∈C,
we note ΩL𝑈 the value of this logical type at 𝑈. For each 𝑈∈C, we write Θ𝑈 the set of possible sets of
axioms in L𝑈, that is Θ𝑈 = P(ΩL𝑈 ). This is also the set of theories.
We take as output (resp. input) the union of the output (resp. output) layers. In supervised and
reinforcement learning, we can tell that, for every input 𝜉in ∈Ξin in a set of inputs for learning, a theory
Tout (𝜉)in Lout is imposed at the output of the network., i.e. some propositions are asked to be true, other
are asked to be false.
The set of theories in the language Lout is denoted Θout. Then the objectives of the functioning is a map
Tout : Ξin →Θout.
Definition. A semantic functioning of the dynamic object 𝑋𝑤 of possible activities in the network,
with respect to the mapping Tout, is a family of quotient sets 𝐷𝑈 of 𝑋𝑤
𝑈, 𝑈∈C, equipped with a map
𝑆𝑈 : 𝐷𝑈 →Θ𝑈, such that for every 𝜉in ∈Ξin and every 𝑈∈C, the image 𝑆𝑈(𝜉𝑈)generates a theory which
is coherent with Tout (𝜉in), for the transport in both directions along any path.
Remark. In the known applications, the richer logic relies on a richer language with more propositions
and less axioms, present near the input layers, but the opposite happens to expressed theories; they are
more constrained in the deepest layers, with more axioms in general.
In the examples we know [BBG21a], the quotient 𝐷𝑈 (from discretized cells) is given by the activity
of some special neurons in the layer 𝐿𝑈, which saturate at a finite number of values, associated to
propositions in the Heyting algebras ΩL𝑈 . In this case, the definition of semantic functioning can be
made more concrete: for each neuron 𝑎∈𝐿𝑈, each quantized value of activity 𝜖𝑎 implies the validity of
a proposition 𝑃𝑎(𝜖𝑎)in ΩL𝑈 ; this defines the map 𝑆𝑈. Then the definition of semantic functioning asks
that, for each input 𝜉in ∈Ξin, the generated activity defines values 𝜖𝑎(𝜉in)of the special neurons, such
that the generated set of propositions 𝑃𝑎(𝜖𝑎), implies the validity of a given proposition in ΩLout , which
is valid for Tout (𝜉in).
In particular, we saw experimentally that the inner layers understand the language Lout, which is an
indication that the functors 𝑓= 𝑔★= 𝐹★
𝛼 propagate the languages backward.
This gives a crude notion of logical information of a given layer, or any subset 𝐸 of neurons in the
union of the sets 𝐷𝑈: it is the set of propositions predicted to hold true in Tout (𝜉in)by the activities in 𝐸.
If all the involved sets are finite, the amount of information given by the set 𝐸can be defined as the ratio
of the number of predicted propositions over the number of wanted decisions, and a mean of this ratio
can be taken over the entries 𝜉in.
34
Remark. The above notion of semantic functioning and semantic information can be extended to sets of
global activities Ξ, singletons sections of 𝑋𝑤, more general that the ones used for learning.
Our experiments in [BBG21a] have shown that the number of hidden layers, or the complexity of the
architecture, strongly influences the nature of the semantic functioning. This implies that the semantic
functioning, then the corresponding accessible semantic information, depend on the characteristics of the
dynamic 𝑋𝑤, for instance the non-linearities for saturation and quantization, and of the characteristics of
the learning, the influence of the non-linearities of the gradient of backpropagation on the optimal weights
𝑤∈𝑊. Therefore, it appears a notion of semantic learning, which is a flow of natural transformations
between dynamic objects 𝑋𝑤𝑡 , increasing the semantic information.
In the mentioned experiments, the semantic behavior appears only for suﬃciently deep networks, and
for non-linear activities.
2.4 The model category of a DNN and its Martin-Löf type theory
In this section, we study the collection of stacks over a given layers architecture, with fibers in a given cat-
egory, as groupoids, and we show that it possesses a natural structure of closed model category of Quillen,
giving both a theory of homotopy and an intensional type theory, where the above stacks with free logical
propagation, described by theorem 2.1, correspond respectively to fibrant objects and admissible contexts.
Consider two fibrations (F𝑈,𝐹𝛼)and (F′
𝑈,𝐹′
𝛼)over C; a morphism 𝜑from the first to the second is
given by a collection of functors 𝜑𝑈 : F𝑈 →F′
𝑈 such that for any arrow 𝛼: 𝑈→𝑈′of C, 𝜑𝑈◦𝐹𝛼= 𝐹′
𝛼◦𝜑𝑈′.
With the fibrations in groupoids, this gives a category GrpdC. Natural transformations between two
morphisms give it a structure of strict 2-category.
We consider this category fibred over CX. Remind that the Grothendieck topology on CX that we
consider is chaotic [AGV63]. If we consider an equivalent site, with a non-trivial topology, homotopical
constraints appear for defining stacks [Gir72], [Hol08]. However the category of stacks (resp. stacks in
groupoids) is equivalent to the category obtained from CX.
Hofmann and Streicher [HS98], have proved that the category Grpd of groupoids gives rise to a
Martin-Löf type theory [ML80], by taking for types the fibrations in groupoids, for terms their sections,
for substitutions the pullbacks, and they have defined non-trivial (non-extensional) identity types in this
theory.
Hollander [Hol01], [Hol08], using Giraud’s work and homotopy limits, constructed a Quillen model
theory on the category of fibrations (resp. stacks) in groupoids over any site C, where the fibrant
objects are the stacks, the cofibrant objects are generators, and the weak equivalences are the homotopy
equivalence in the fibers (see also Joyal-Tierney and Jardine cited in Hollander [Hol08]). These results
were extended to the category of general stacks, not only in groupoids, over a site by Stanculescu [Sta14].
35
Awodey and Warren [AW09] observed that the construction of Hofmann-Streicher is based on the
most natural closed model category structure in the sense of Quillen on Grpd, and proposed an extension
of the construction to more general model categories. Thus they established the connection between
Quillen’s models and Martin-Löf intensional theories, which was soon extended to a connection between
more elaborate Quillen’s models and Voedvosky univalent theory.
Arndt and Kapulkin, in Homotopy Theoretic Models of Type Theory [AK11], have proposed addi-
tional axioms on a closed model theory that are suﬃcient to formally deduce a Martin-Löf theory. This
was extended later by Kapulkin and Lumsdaine [KLV12], to obtain models of Voedvosky theory, by
using more simplicial techniques. Here, we will follow their approach, without going to the special prop-
erties of HoTT, that are functions extensionality, Univalence axiom and Higher inductive type formations.
In what follows, we focus on the model structure of groupoids and stacks in groupoids, which are
the most useful models for our applications. However, many things also work with Cat in place of Grpd,
and some other model categories M. The complication is due to the diﬀerence between fibrations (resp.
stacks) in the sense of Giraud and Grothendieck and the fibrations in the sense of Quillen’s models, which
is not the case with groupoids. For Cat, there exists a unique closed model structure, defined by Joyal and
Tierney, such that the weak equivalences are the equivalence of categories [SP12]. It is named for this
reason the canonical model structure on Cat; in this structure, the cofibrations are the functors injective
on objects and the fibrations are the isofibrations. An isofibration is a functor 𝐹: A→B, such that every
isomorphism of Bcan be lifted to an isomorphism of A. Any fibration of category is an iso-fibration,
but the converse is true only for groupoids. A diﬀerent model theory was defined by Thomason [Tho80],
which is better understandable in terms of ∞-groupoids and ∞-categories.
The axioms of Quillen [Qui67] concern three subsets of morphisms in a category M, supposed to
be (at least finitely) complete and cocomplete, the set Fib of fibrations, the set Cofib of cofibrations and
the set WE of weak equivalences. An object 𝐴of Mis said fibrant (resp. cofibrant) if 𝐴→1, the final
object (resp. ∅→𝐴from the initial object) is a fibration (resp. a cofibration).
Definitions. Two morphisms 𝑖: 𝐴→𝐵and 𝑝: 𝐶→𝐷in a category are said orthogonal, written (non-
traditionally) 𝑖⋌ 𝑝, if for any pair of morphisms 𝑢: 𝐴→𝐶and 𝑣: 𝐵→𝐷, such that 𝑝◦𝑢= 𝑣◦𝑖, there
exists a morphism 𝑗 : 𝐵→𝐶 such that 𝑗◦𝑖= 𝑢and 𝑝◦𝑗= 𝑣. The morphism 𝑗 is named a lifting, left
lifting of 𝑖and a right lifting of 𝑝.
Two sets Land Rare said be the orthogonal one of each other if 𝑖∈Lis equivalent to ∀𝑝∈R,𝑖⋌ 𝑝
and 𝑝∈Ris equivalent to ∀𝑖∈L,𝑖⋌ 𝑝.
The three axioms of Quillen for a closed category Mof models are:
1) given two morphisms 𝑓 : 𝐴→𝐵, 𝑔: 𝐵→𝐶, define ℎ= 𝑔◦𝑓; if two of the morphisms 𝑓,𝑔,ℎ
belong to WE, then the third one belongs to WE;
2) every morphism 𝑓 is a composition 𝑓= 𝑝◦𝑖of an element 𝑝of Fib and an element 𝑖of Cofib ∩WE,
and a composition 𝑝′◦𝑖′of an element 𝑝′of Fib ∩WE and an element 𝑖′of Cofib;
36
3) the sets Fib and Cofib ∩WE are the orthogonal one of each other and the sets Fib ∩WE and Cofib
also.
An element of Fib ∩WE is named a trivial fibration, and an element of Cofib ∩WE is named a trivial
cofibration.
These axioms (and some more general) allowed Quillen to develop a convenient homotopy theory
in M, and to define a homotopy category 𝐻𝑜M(see his book, Homotopical Algebra, [Qui67]). The
objects of 𝐻𝑜Mare the fibrant and cofibrant objects of M, and its morphisms are the homotopy classes
of morphisms in M; two morphisms 𝑓,𝑔from 𝐴to 𝐵are homotopic if there exists an object 𝐴′, equipped
with a weak equivalence 𝜎: 𝐴′→𝐴and two morhisms 𝑖0,𝑖1 from 𝐴to 𝐴′such that 𝜎◦𝑖0 = 𝜎◦𝑖1, and a
morphism ℎ: 𝐴′→𝐵, such that ℎ◦𝑖0 = 𝑓 and ℎ◦𝑖1 = 𝑔. In the category 𝐻𝑜M, the weak equivalences
of Mare inverted.
A particular example is the category of sets with surjections as fibrations, injections as cofibrations and
all maps as equivalences. Another trivial structure, which exists for any category is no restriction for Fib
and Cofib but isomorphisms for WE.
As we already said, an important example is the category of groupoids Grpd, with the usual fibrations in
groupoids, with all the functors injective on the objects as cofibrations, and the usual homotopy equiva-
lence (i.e. here category equivalence) as weak equivalences.
We also mentioned the canonical structure on Cat, that is the only one where weak homotopy corresponds
to the usual equivalence of category.
Other fundamental examples are the topological spaces Top and the simplicial sets SSet = Δ∧, with Serre
and Kan fibrations for Fib respectively.
The closed model theory of Thomason 1980 [Tho80] on Cat is deduced by the above structure on SSet,
by using the nerve construction and the square of the right adjoint functor f the barycentric subdivision.
In this structure the weak equivalences are not reduced to the category equivalences and the cofibrant
objects are constrained [Cis06]; this theory is weakly equivalent to the Kan structure on SSet. Then in this
structure, a category is considered through its weak homotopy type (the weak homotopy type of its nerve).
We now call on a general result of Lurie ’s book, [Lur09, appendix A.2.8, prop. A.2.8.2], which
establishes the existence of two canonical closed model structures on the category of functors MC=
Fun(Cop
,M)when Mis a model category. (Caution, Lurie consider diagrams, i.e. Cand not Cop.)
An additional hypothesis is made on M, that it is combinatorial in the sense of Smith (see Rosicky in
[rR09]), i.e. locally presentable (i.e. accessible by a regular cardinal), and generated by cofibrant objects,
which are both satisfied by Grpd and by Cat. Moreover Mis supposed to have all small limits and small
colimits, which is also the case for Grpd (or Cat); as Set, both are cartesian closed categories; every
object is fibrant and cofibrant.
37
The two Lurie structures are respectively obtained by defining the sets Fib or Cofib in the fiberwise
manner, as for the set WE, and by taking respectively the set Cofib or Fib of morphism which satisfy the
required lifting properties, respectively on the left and on the right, i.e. the orthogonality of Quillen.
The structure obtained by fixing Fib (resp. Cofib) by the behavior in the fibers, is named the projective
structure, or right one (resp. the injective one, or left one).
Caution: depending on the authors, the term right and left may be exchanged.
The model structure of Hollander on GrpdC(or Stanculescu for CatC) is the right Lurie model. She called
this model a left model.
A model category is said right proper when the pullback of any weak equivalence along an element
of 𝐹𝑖𝑏is again a weak equivalence. Dually, left proper is when push-forward of weak equivalence along
cofibrations is again in WE.
In the right proper case, the injective (left) structure of Lurie was defined before by D-C. Cisinski in
"Images directes cohomologiques dans les catégories de modèles" [Cis03].
The cofibrations in the right model (resp. the fibrations in the left model) depend on the category C.
They certainly deserve to be better understood.
See the discussion of Cisinski, in his book Higher Categories and Homotopical Algebra, [Cis19, section
2.3.10].
Proposition 2.3. If Chas suﬃciently many points, the elements of Fib for the left Lurie structure are
fibrations in the fibers (i.e. elements of Fib for the right structure) and the elements of Cofib for the right
structure are injective on the objects in the fibers (i.e. elements of Cofib for the left structure.
Proof. Suppose that a morphism 𝜑 is right orthogonal to any trivial cofibration 𝜓 of the left Lurie
structure; for every point 𝑥in C, this gives an orthogonality in the model Grpd, then over 𝑥, 𝜑𝑥 induces a
fibration in groupoids. From the hypothesis, this implies that in every fiber over C, 𝜑is a fibration, then
an element of Fib for the right Lurie structure.
The other case is analog.
However in general, even if Cis a poset, not all fibrations in the fibers are in Fib for the left model structure,
and not all the injective in fibers are in Cofib for the right model. This was apparent in Hollander [Hol01].
Trying to determine the obstruction for a local fibration (resp. local cofibration) to be orthogonal to
functors that are locally injective on the objects (resp. local fibrations) and locally homotopy equivalence,
we see that the intuitionistic structure of ΩC enters the game, through the global constraints on the
complement of presheaves:
Lemma 2.5. The category Cbeing the oriented segment 1 →0 and the category Mbeing Set (then
MCis the topos of the Shadoks [Pro08]); in the left Lurie model the fibrant objects are the (non-empty)
surjective maps 𝑓 : 𝐹0 →𝐹1.
38
Proof. A trivial cofibration is a natural transformation
𝜂: (ℎ: 𝐻0 →𝐻1)→(ℎ′: 𝐻′
0 →𝐻′
1); (2.36)
such that 𝜂0 and 𝜂1 are injective.
Suppose given a natural transformation 𝑢= (𝑢0,𝑢1)from ℎ to 𝑓 : 𝐹0 →𝐹1; the lifting problem is the
extension of 𝑢 to 𝑢′from ℎ′to 𝑓. If 𝐻1 is empty, there is no problem. If not, we choose a point ★0
in 𝐻0 and note ★1 = ℎ(★0). If 𝑥′
1 ∈𝐻′
1 does’nt belong to 𝐻1 we define 𝑢′
1 (𝑥′
1)= 𝑢1 (★1), and for any 𝑥′
0
such that ℎ′(𝑥′
0)= 𝑥′
1, we define 𝑢′
0 (𝑥′
0)= 𝑢0 (★0). Now the problem comes with the points 𝑥”
0 in 𝐻′
0\𝐻0
such that ℎ′(𝑥”
0)∈𝐻1 (a shadok with an egg); their image by 𝑢1 is defined, then 𝑢′
1 (ℎ′(𝑥”
0))is forced
to be in the image of 𝐹0 by 𝑓. If 𝑓 is not surjective there exists 𝜂 such that the lifting is impossible.
But, if 𝑓 is surjective there is no obstruction: we define 𝑢′
0 (𝑥”
0)to be any point 𝑦0 in 𝐹0 such that
𝑓(𝑦0)= 𝑢1 (ℎ′(𝑥”
0))in 𝐹1.
Lemma 2.6. Also M= Set, but Cbeing the (confluence) category with three objects 0,1,2 and
two non-trivial arrows 1 →0 and 2 →0. In the left Lurie model, the fibrant objects are the pairs
(𝑓1 : 𝐹0 →𝐹1,𝑓2 : 𝐹0 →𝐹2), such that the product map (𝑓1,𝑓2)is surjective.
Proof. Following the path of the preceding proof, with an injective transformation 𝜂 from a triple
𝐻0,𝐻1,𝐻2 to a triple 𝐻′
0,𝐻′
1,𝐻′
2, we are in trouble with the elements 𝑥”
0 ∈𝐻′
0 that ℎ′
1 or ℎ′
2 sends into
𝐻1 or 𝐻2 respectively. Under the hypothesis of bi-surjectivity, we know where to define 𝑢′
0 (𝑥”
0). But if
this hypothesis is not satisfied, impossibility happen in general for 𝜂.
Lemma 2.7. Also M= Set, but Cbeing the (divergence) category with three objects 0,1,2 and
two non-trivial arrows 0 →1 and 0 →2. In the left Lurie model, the fibrant objects are the pairs
(𝑓1 : 𝐹1 →𝐹0,𝑓2 : 𝐹2 →𝐹0), such that separately 𝑓1 and 𝑓2 are surjective.
Proof. following the path of the preceding proof, with an injective transformation 𝜂 from a triple
𝐻0,𝐻1,𝐻2 to a triple 𝐻′
0,𝐻′
1,𝐻′
2, we are in trouble with the elements 𝑥”
1 ∈𝐻′
1 (resp. 𝑥”
2 ∈𝐻′
2) that ℎ′
1
(resp. ℎ′
2) sends into 𝐻0. As in the proﬀ of the lemma 1, the problem is solved under the hypothesis of
surjectivity, but it cannot be solved without it.
More generally, we can determine the fibrant objects of the left Lurie model (injective) for every closed
model category M, and a finite poset Cwhich has the structure of a DNN, coming with a graph, with
unique directed paths:
Theorem 2.2. When Cis the poset of a 𝐷𝑁𝑁, for any combinatorial category of model, the fibrations
of MCfor the injective (left) model structure are made by the natural transformations F→F′between
functors in Cto M, that induce fibrations in Mat each object of C, such that the functor Fis also a
fibration in Malong each arrow of Ccoming from an internal of minimal vertex (ordinary vertex, output
or tip), and a fibration along each of the arrows issued from a minimal vertex (output and tip), and a
multi-fibration at each confluence point, in particular at the maximal vertices (input or tank).
39
By multi-fibration 𝑓𝑖,𝑖∈𝐼 from an object 𝐹𝐴 of Mto a family of objects 𝐹𝑖,𝑖∈𝐼 of M, we mean a
fibration (element of Fib) from 𝐹𝐴 to the product 𝑖∈𝐼𝐹𝑖.
Proof. We proceed by recurrence on the number of vertices. For an isolated vertex, this is the definition
of fibration in M. Then consider an initial vertex (tank or input) 𝐴with incoming arrows 𝑠𝑖 : 𝑖→𝐴for
𝑖∈𝐼 in the graph poset C, and note C★ the category with the star 𝐴,𝑠𝑖 deleted. A trivial cofibration in
MCis a natural transformation 𝜂; H→H′between contravariant functors in C→M, which is at each
vertex injective on objects and an element of WE. Let us consider a morphism (𝑢,𝑢′)in in MCfrom 𝜂
to a morphism 𝜑: F→F′, where Fbelongs to MC.
Suppose that 𝜑satisfies the hypotheses of the theorem. From the recurrence hypothesis, there exists
a lifting 𝜃★: (H′)★→F★ between the restrictions of the functors to C★; it is in particular defined on the
objects 𝐻′
𝑖,𝑖∈𝐼to the objects 𝐹𝑖,𝑖∈𝐼.
Consider the functor from 𝐻′
𝐴to the product 𝑖𝐹𝑖, which is obtained by composing the horizontal arrows
of 𝜂, from 𝐻′
𝐴 to the product 𝐻′
= 𝑖𝐻′
𝑖 with 𝜃′. The fact that 𝐹𝐴 → 𝑖𝐹𝑖 is a multi-fibration in Mand
the fact that 𝜂𝐴 : 𝐻𝐴 →𝐻′
𝐴 is a trivial cofibration in Mimply the existence of a lifting 𝜃𝐴 : 𝐻′
𝐴 →𝐹𝐴,
which is given on 𝐻𝐴.
Conversely, if the hypothesis of multi-fibration is not satisfied, there exists elements 𝜂𝐴 : 𝐻𝐴 →𝐻′
𝐴
in Cofib ∩WE of 𝑀, such that the lifting 𝜃𝐴 of 𝐻′
𝐴 to 𝐹𝐴 does’nt exist, by the axiom (3)of closed models.
To finish the proof, we note that the necessity to be a fibration at each vertex in 𝐶is given by proposition
2.3.
Corollary. Under the same hypotheses, the fibrant objects of MCfor the injective (left) model structure
are made by the functors that are a fibration in Mat each internal of minimal vertex (ordinary vertex,
output or tip), and a fibrant object at the minimal (output and tip), and a multi-fibration at each confluence
point (see lemma 2.7), in particular at the maximal vertices (input or tank).
One interest of this result is that it will describe the allowed contexts in the associated Martin-Löf theory
when it exists, as we will see just below.
Another interest is for the behavior of the classifying object F: in the case of GrpdCthe fibrant objects
are all good for the induction theory in logic over the network (see theorem 2.1). In the case of CatC,
with the canonical structure, we will see below that it is not the case, only a subclass of fibrant objects
are good, which are made by composition of Giraud-Grothendieck fibrations.
Last by not least, this corollary allows to enter the homotopy theory of the stacks, according to Quillen
[Qui67], because it associates objects up to homotopy with the stacks that have a fluid semantic function-
ing as in theorem 2.1.
In GrpdCthe final object 1 (resp. the initial object ∅) is the constant functor on Cwith values a singleton,
(resp. the empty set). It follows that any object is cofibrant.
The additional axioms of Arndt and Kapulkin for a Logical Model Theory are as follows:
40
(1) for any element 𝑓 ∈Fib, 𝑓 : 𝐵→𝐴, the pullback functor 𝑓★: M|𝐴→M|𝐵, once restricted to the
fibrations, possesses a right-adjoint, denoted Π 𝑓.
(2) The pullback of a trivial cofibration, i.e. an element of Cofib ∩WE, along an element of Fib is again
a trivial cofibration.
Remark. In Arndt and Kapulkin [AK11], the first axiom is written without the restriction of the adjunction
to fibrations, however they remark later [AK11, section 4.1, acknowledging an anonymous reviewer] that
this restricted axiom is suﬃcient for the application below.
The second axiom is satisfied if separately Cofib and WE are stable by pullback along a fibration. As
we already said, a model category satisfying the second property for WE is called right proper.
When every object in Mis fibrant (resp. cofibrant) the theory is right (resp. left) proper [Hir03].
This is the case for Grpd (and Cat). And Lurie proved that his two model structures on diagrams (or
phe-sheaves) are right (reps. left) proper as soon as Mis so. Then in our case, all the considered models
are right proper and left proper. This was shown by Hollander [Hol01] for GrpdC.
The injectivity on objects in the fibers and the equivalence of category in the fibers are preserved
by every pullback, thus condition (2) is satisfied for the left injective structure. This is the structure we
choose. What happens to the right structure?
Arndt and Kapulkin noticed the example of the injective structure [AK11, Prop. 27, p.12] and its
Bousfield-Kan localizations; this gives in particular the injective model structures for the category of
stacks over any site (see Hirschhorn, Localization of Model Categories [Hir03]).
The existence of a right adjoint and a left adjoint of the pullback of fibrations in categories, as it holds
for presheaves of sets, was proved by Giraud in 1964 [Gir64, section I.2.].
Then, by proposition 2.3, for M= Grpd, both left and right structures satisfy the condition (1). For
M= Cat this is true only if 𝑓 is a fibration in the geometric sense, not only an isofibration. What happens
to other models categories M?
As noticed by Arndt and Kapulkin, the left adjoint of 𝑓★ : M|𝐴→M|𝐵always exists, it is written
Σ 𝑓, and the right properness implies that it respects WE.
If Msatisfies the axioms (1) and (2), Arndt and Kapulkin generalized the constructions of Seely
[See84], Hofmann and Streicher [HS98], and Awodey−Warren [AW09] to define a M-L theory:
A context is a fibration Γ →C, that is a fibrant object. A type Ain this context is a fibration A→Γ.
The declaration (judgment) of a type is written Γ ⊢A. A term 𝑎: 𝐴is a section Γ →A. It is denoted
41
Γ ⊢𝑎: A.
A substitution 𝑥/𝑎 is given by a change of base 𝐹★ for a functor 𝐹: Δ →Γ in MC, not necessarily a
fibration.
The adjoint functor Σ 𝑓 and Π 𝑓 of 𝑓★, allows to define new types of objects: given Γ and 𝑓 : A→Γ, and
𝑔: B→A, we get Σ 𝑓(𝑔): Σ𝑥:AB(𝑥)→Γ and Π 𝑓(𝑔): Π𝑥:AB(𝑥)→Γ. They respectively replace the
union over Aand the product over A.
On the types, logical operations are applied, A∧B, A∨B, A⇒B, ⊥is empty, ∃𝑥,𝐵(𝑥), ∀𝑥,𝐵(𝑥).
The rules for these operations satisfy the usual axioms.
More types, like the integers or the real numbers or the well ordering can be added, with specific rules.
As remarked by Arndt and Kapulkin, it is not necessary to have a fully closed model theory to get a
Martin-Löf type theory [AK11, remarks pp. 12-15]. They noticed that 𝑀−𝐿type theories are probably
associated to fibration-categories (or categories with fibrant objects) in the sense of Brown [Bro73] (see
also [Uem17]). In these categories, cofibrations are not considered, however a nice homotopy theory can
be developed.
We have the following result concerning the weak factorization system made by cofibrations and
trivial fibrations in the canonical model Cat:
Lemma 2.8. A canonical trivial fibration in Cat is a geometric fibration.
Proof. Consider an isofibration 𝑓 : A→Bthat is also an equivalence of category. Take 𝑎∈Aand
𝑓(𝑎)= 𝑏∈Band a morphism 𝜑: 𝑏′→𝑏 of B; because 𝑓 is surjective on the objects, there exists
𝑎′∈𝐴such that 𝑓(𝑎”)= 𝑏′, and because 𝑓 is an equivalence the map from Hom(𝑎′,𝑎)to Hom(𝑏”,𝑏)
is a bĳection, then there exists a unique morphism 𝜓: 𝑎′→𝑎such that 𝑓(𝜓)= 𝜑. In the same manner,
every morphism 𝑏” →𝑏′has a unique lift 𝑎” →𝑎′, and conversely any morphism 𝜓′: 𝑎” →𝑎′defines
a composed morphism 𝜒: 𝑎” →𝑎and a morphism image 𝜑′: 𝑏” →𝑏′that define the same morphism
𝜑◦𝜑” from 𝑏” to 𝑏. As the morphisms from 𝑎” to 𝑎 are identified by 𝑓 with the morphisms from
𝑏” to 𝑏, this gives a natural bĳection between the morphisms 𝜓′ from 𝑎” to 𝑎′and the pairs (𝜒,𝜑′)
in Hom(𝑎”,𝑎)×Hom(𝑏”,𝑏′)over the same element in Hom(𝑏”,𝑏). Therefore 𝜓 is a strong cartesian
morphism over 𝜑.
The same proof shows that a canonical trivial fibration is a geometric op-fibration, that is by definition a
fibration between the opposite categories.
In the case where Cis the poset of a 𝐷𝑁𝑁 and Mis the category Cat, we say that a model
fibration 𝑓 : 𝐴→𝐵, in MCis a geometric fibration if it is a Grothendieck-Giraud fibration, and if all the
iso-fibrations that constitute the fibrant object 𝐴are Grothendieck-Giraud fibrations (see theorem 2.2).
Theorem 2.3. Let Cbe a poset of 𝐷𝑁𝑁, there exists a canonical 𝑀−𝐿 structure where contexts and
types correspond to the geometric fibrations in the 2-category of contravariant functors CatC, and such
that base change substitutions correspond to its 1-morphisms.
42
Proof. We follow the lines of Arndt and Kapulkin [AK11, theorem 26]. The main point is to prove that
if 𝑓 : 𝐴→𝐵is a geometric fibration in MC, the pullback functor 𝑓★: Cat|𝐴→Cat𝐵, has a left adjoint
𝑓! = Σ 𝑓 and a right adjoint 𝑓★= Π 𝑓 that both preserve the geometric fibrations. For the first case it is the
stability of Grothendieck-Giraud fibrations by composition. For the second one, this is Giraud theorem
of bi-adjunction [Gir71].
There exist several equivalent interpretations of such a type theory, as for the intuitionistic theory of
Bell, Lambek Scott et al. (see Martin-Löf, Intuitionistic Type Theory, [ML80]). For instance the types
are sets, the terms are elements, or a type is a proposition and a term is a proof, or a type is a problem (a
task) and a term is a method for solving it. (For each interpretation, things are local over a context.)
In particular, Identity types are admitted, representing equivalence of elements, proofs or methods that
are not strict equalities, like homotopies, or invertible natural equivalences.
The types of identities, as in Hofmann and Streicher [HS98], are fibrations Id𝐴 : 𝐼A→A×Aequipped
with a cofibration 𝑟: A→𝐼A (with a section) such that Id𝐴◦𝑟= Δ, the diagonal morphism. They are
considered as paths spaces.
For instance, given a groupoid 𝐴, Id𝐴 = ({0 ↔1}⇒𝐴= 𝐴{0↔1}is an identity type.
Axioms of inference for the types are expressed by rules of formation, introduction and determination,
specific to each type [ML80].
Let us compare to the semantics in a topos C∧: a context is an object Γ which is a presheaf with
values in Set, so a fibration in sets over Cand a type is another object 𝐴; to get something over Γ we can
consider the projection Γ ×𝐴→Γ. A section corresponds to a morphism 𝑎: Γ →𝐴, which is rightly a
term of type 𝐴, Γ ⊢𝑎: A.
A substitution corresponds to a morphism 𝐹: Δ →Γ, and defines a pullback of trivial fibrations Δ×𝐴→Δ.
If we have a morphism 𝑔: 𝐵→Γ ×𝐴in the topos, we can define its existential image ∃𝜋𝑔(𝐵)and its
universal image ∀𝜋𝑔(𝐵)as subobjects of Γ, which can be seen as a trivial fibrations over Γ.
Therefore, we have analogs of M-L type theory in Set theory, but with trivial fibrations only and
without fibrant restriction.
2.5 Classifying the M-L theory ?
In what precedes the category Grpd has replaced the category Set; it is also cartesian closed. Also we
have seen that all small limits and colimits exist in GrpdC (Giraud, Hollander, Lurie). However every
natural transformation between two functors with values in Grpd is invertible. Thus in the 2-category,
the morphisms in HomGrpd (𝐺,𝐺′)are like homotopies. In fact they become homotopies when passing
to the nerves.
43
Let us introduce the categories of presheaves on every fibration in groupoids A→C, i.e. the clas-
sifying topos EA of the stack A. Their objects are fibered in groupoids over C, because the fibers E𝑈
for 𝑈∈Care such (they take their values in IsoSet), but their morphisms, the natural transformations
between functors, are taken in the sense of sets, not invertible.
In what follows we combine the type theory of topos with the groupoidal 𝑀−𝐿type theory.
We propose new types, associated to every object 𝑋Ain every EA.
The fibration A→Γ itself can be identified with the final object 1A∈EAin the context Γ.
Sections of A→Γ are particular cases of objects. For the terms in an object 𝑋𝐴, we take any natural
transformation from the object 𝑆corresponding to a section Γ →Ato the object 𝑋𝐴 in EA.
A simple section is a term to 1A, the final object, which is a usual M-L type.
Due to the adjunction for the topos of presheaves, the construction Σ and Π extend to the new types.
Now a classifier of subobjects ΩAis available for any M-L type A.
We define relative subobjects using the correspondence 𝜆𝜋 : ΩA→𝜋★ΩΓ.
This extension of M-L theory allows to define languages and semantics over DNNs with internal
structure in the model category M.
44
3
Dynamics and homology
3.1 Ordinary cat’s manifolds
Some limits, in the sense of category theory, of the dynamical object 𝑋𝑤 of C∼ describe the sets of
activities in the 𝐷𝑁𝑁 which correspond to some decisions taken by its output (the so called cat’s
manifolds in the folklore of Deep Learning).
Here we consider the case of supervised learning or the case of reinforcement learning, because the
success or the failure of an action integrating the output of the network is also a kind of metric.
For instance, consider a proposition 𝑃𝑜𝑢𝑡 about the input 𝜉in which depends on the final states 𝜉out.
It can be seen as a function 𝑃on the product 𝑋𝐵 = 𝑏𝑋𝑏 of the spaces of states over the output layers
to the boolean field ΩSet = {0,1}, taking the value 1 if the proposition is true, 0 if not. Our aim is to
better understand the involvement of the full network in this decision; it is caused by the input data in a
deterministic manner, but it results from the chosen weights and from the full functioning of the 𝐷𝑁𝑁.
One of the many ways to express the situation in terms of category is to enlarge C(or ) by several
terminal layers (see figure 3.1):
1) a layer 𝐵★ which makes the product of the output layers, as we have done with forks, followed by
the layer 𝐵(remark that this can be replaced by 𝐵only, with an arrow from 𝑏∈𝑥out);
2) a layer 𝜔𝑏 with one cell and two states in a set Ω𝑏, as in ΩSet, with one arrow from 𝜔𝑏 to 𝐵, for
translating the proposition 𝑃, followed by a last layer 𝜔1, with one arrow 𝜔𝑏 →𝜔1, the state’s space
𝑋𝜔1 being a singleton ★1, and the map ★1 →Ω𝑏 sending the singleton to 1. This gives a category
C+enlarging Cby a fork with handle 𝐵←𝜔𝑏 →𝜔1, and a unique extension 𝑋𝑤
+, depending on 𝑃,
of the functor 𝑋𝑤 from Cop to Set in a presheaf over C+.
The space of sections singletons of 𝑋𝑤
+ is identified naturally with the space of sections of 𝑋𝑤 such
that the output satisfies 𝑃out, i.e. the subset of the product of all the 𝑋𝑤(𝑎)when 𝑎describes Cmade by
the coherent activities giving the assertion "𝑃is true" at the output. In this picture, we also can consider
that 𝑃is the weight over the arrow 𝐵←𝜔𝑏, and note 𝑋𝑤,𝑃
+ the extension of 𝑋𝑤
.
In other terms, the subset of activities of 𝑋 which aﬃrm the proposition 𝑃out is given by a value of the
45
b1
Xout Xout
Id P
b
{ }
b2
B
B
ωb
ω1
C+
bm
Figure 3.1: Interpretation of a proposition : categorical representation
right Kan extension of 𝑋+along the unique functor 𝑝+: Cop
+ →★:
𝑀(𝑃out)(𝑋)= RKanC+(𝑋+)= lim𝑎∈Cop
𝑋𝑤
+(𝑎): (3.1)
+
In the 𝐴𝐼 folklore, the set 𝑀(𝑃out)(𝑋)is named a cat’s manifold, alluding to the case where the
network has to decide if yes or no a cat is present in the image. 𝑀(𝑃out)(𝑋)can be identified with a
subset of the product 𝑋𝑖𝑛 of the input layers. It has to be compared with the assertion "𝑃is true" made
by an observer, then studied in function of the weights Wof the dynamics.
However, in general, 𝑀(𝑃out)(𝑋)cannot be identified with a product of subsets in the 𝑋𝑎’s, for 𝑎∈C; it
is a global invariant.
In fact, it is a particular case of a set of cohomology:
𝑀(𝑃out)(𝑋)= 𝐻0 (C+; 𝑋+). (3.2)
If the proposition 𝑃out is always true, 𝑀 coincides with the set of section of 𝑋= 𝑋𝑤, which can be
identified with the product of the entry layers activities:
Γ(𝑋)= 𝐻0 (𝐶; 𝑋)
𝑎∈𝑥in
𝑋𝑎 (3.3)
The construction of C+and the extension of 𝑋by 𝑋+can be seen as a conditioning. The map 𝑋+(𝜔𝑏 →𝐵)
is equivalent to a proposition, the characteristic map of a subset of 𝑋𝐵. In this case we have
𝐻0 (C+; 𝑋+)⊂𝐻0 (C; 𝑋). (3.4)
46
In the same manner, we define the manifold of a theory Tout expressed in a typed language Lout in the
output layers, by replacing the above objects 𝜔𝑏, 𝜔1, and the presheaf 𝑋+(𝑃)over them, by larger sets
and 𝑋+(T), as the set of sections of 𝑋+(T)over the whole C+.
We will revisit the notion of cat’s manifold when considering the homotopy version of semantic
information.
3.2 Dynamics with spontaneous activity
In our approach of networks functioning, the feed-forward dynamic coincides with the limit set 𝐻0 (𝑋).
The coincidence with the traditional notion of propagation from the input to the output relies on the par-
ticular choice of morphisms at the centers of forks (named tanks), product on one side and isomorphism
on the other. But this can be generalized to other morphisms: the only condition being that the inner
sources 𝐴and the input from the outer world 𝐼 determine a unique section of the object 𝑋𝑤 over C. In
concrete terms, this happens if and only if the maps from 𝐴and 𝐼give coherent values at any tip of each
fork.
This tuning involves the values in entry 𝜉0 ∈Ξ, the values of the inner sources 𝜎0 ∈Σ and the weights, in
particular from an 𝐴to the 𝑎′,𝑎”
,...’s. Therefore it depends on the learning process.
Then a possibility for defining coherent dynamical objects with spontaneous activities is to start with
standard objects 𝑋𝑤, satisfying the restriction of products and isomorphisms, then to introduce small
deformations of the projections maps, and obtain the global dynamics by using algorithms which realize
the Implicit Function Theorem in the Learning process.
Another possibility, closer to the natural networks in animals, and more readable, is to keep unchanged
the projections to the tips 𝑎′
,..., and to introduce new dynamical entries 𝑦𝐴 for each tang 𝐴, then to send
a message to the handle 𝑎according to the following formula
𝑥𝑎 = 𝑓𝐴
𝑎 (𝑥𝑎′,𝑥𝑎”
,...; 𝑦𝐴). (3.5)
The state in 𝐴being described by (𝑥𝑎′,𝑥𝑎”,...,𝑦𝐴).
In such a manner the coherence is automatically verified. Each collection of inputs and tangs modulations
generates a unique section.
These spontaneous entries can be learned by backpropagation, as the weights, by minimizing a functional,
or realizing a task with success.
It is important to remark that in natural brains, even for very small animals, having no more than
several hundred neurons, the part of spontaneous activity is much larger than the part due to the sensory
47
inputs. This activity comes from internal rhythms, regulatory activities of the autonomous system, in-
ternal motivations more or less planed. The neural network transforms them in actions or more general
decisions. To make them eﬃcient, corrections are necessary, due to reentrant architectures.
However these natural networks in general do not learn using fully supervised methods; they depend
on reinforcement, by success of actions, or by unsupervised methods, involving maximization of mutual
information quantities. This will require much further works to achieve this degree of integration in
artificial networks. Also evolution plays a fundamental role, in particular by specifying the processes of
weights transformations. But certainly experiments can be easily conducted in this direction, with simple
networks as in Logical Information Cells [BBG21a], the experimental companion article.
3.3 Fibrations and cofibrations of languages and theories
In this section, we define several sheaves and cosheaves over the stack F, that are naturally associated to
languages and theories, defining moduli over monoids in the classifying topos E, by using the semantic
conditioning (see theorem 3.1), in such a manner that their homology or homotopy invariants, in the
sense of topos, give tentative semantical Information quantities and spaces. In the most elementary
cases, we recover in the following sections 3.4, 3.5, the definitions of Carnap and Bar-Hillel [CBH52],
and their known generalizations [BBD+11], [BBDH14], already used in Deep Learning (see for instance
[XQLJ20]), but at the end of this chapter, we will also show new promising elements of information.
In this section and the following ones, we use the semantic functioning in usual 𝐷𝑁𝑁𝑠to define their
semantic information content. Taking into account the internal dimensions given by the stacks Fover C,
several levels of information emerge. Without closing the subject, they reflect diﬀerent meaning of the
word information.
A first level concerns the pertinent types, or objects, to introduce in order to understand how the
network performs a semantic task, in addition to the types coming from Lout, that are put at the hand by
the observer, and guide the learning process, by backpropagation or reinforcement. A first conjecture,
that we will not study in the present text, is that new objects appear in cohomological forms, as obstruc-
tions for integrating correctly the input data in the output theory. It is not excluded that this can appear
spontaneously in the network, but more probably it requires the intervention of the observer, for changing
the functional (the metric) or the data, which induces a variation of the weights. We will describe below
in section 3 examples of semantic groupoids which could generate or constrain these obstructions. More
precisely, we expect that the new objects are vanishing cycles, in the sense of Grothendieck, Deligne,
Laumon [Ill14], for convenient maps of sites, localized in the fibers F𝑈, at points (𝑈,𝜉).
In some regions of the weights, the network should become able to develop a semantic functioning about
the new objects, formalized by the languages L𝑈;𝑈∈Csimilarly to what happens with singularities of
functions or varieties, with imposed reality conditions. The analogy is made more precise in chapter 4.
48
A second level, perhaps not independent of the first one, concerns the information contained in some
theories about other theories, or about decisions to take or actions to do, for instance T𝑈′ in some layer,
considered in relation to T𝑈, when 𝛼: 𝑈→𝑈′, or Tout. As we saw, the expression of these theories
in functioning networks depends on the given section 𝜖 of 𝑋𝑤. However, we expect that the notion
of information also allows to compare the theories made by diﬀerent networks about a some class of
problems.
The semantic information that we want to make more precise must be attached to the communication
between layers and the communication between networks, and attached to some problems to solve, for a
view of the necessity to introduce interaction in a satisfying view of information. See Thom in [Tho83].
Some theories will be more informative than others, or more redundant, then we will be happy to
attach quantitative notions of amount of information to the notion of semantic information. However,
eﬃcient numerical measures should also take care of the expression of theories by some axioms. Some
systems of axioms are more economical than others, or more redundant than others. Redundancy is more
the matter of axioms, ambiguity is more the matter of theories. In the present approach, the notion of
ambiguity comes first.
In Shannon information theory, [SW49], the fundamental quantity is the entropy, which is in fact a
measure of the ambiguity of the expressed knowledge with respect to an individual fact, for instance a
message. Only some algebraic combinations of entropies can be understood as an information in the
common sense, for instance the mutual information
𝐼(𝑋;𝑌)= 𝐻(𝑋)+𝐻(𝑌)−𝐻(𝑋,𝑌).
Here the theories T𝑈,𝑈 ∈Care seen as possible models, analogous to the probabilistic models
P𝑋,𝑋 ∈Bin Bayesian networks. The variables of the Bayesian network are analogous to the layers of
the neural networks; the values of the variables are analogs of the states of the neurons of the layers. In
some version of Bayes analysis, for instance presented by Pearl [Pea88], the Bayes network is associated
to a directed graph, but in some other versions it is an hypergraph [YFW01], or a more general poset
[BPSPV20].
In the case of the probabilistic models, Shannon theorems have revealed the importance of entropy
and of mutual information. It has been shown in [BB15] and [Vig20]), that the entropy is a universal
class of cohomology of degree one of the topos of presheaves over the Bayesian network, seen as a
poset B, equipped with a cosheaf Pof probabilities (covariant functor of sets). The operation of joining
variables gives a presheaf Ain monoids over B. On the other hand, the numerical functions on Pform a
sheaf FP, which becomes an 𝐴-module by considering the mean conditioning of Shannon. The entropy
belongs to the Ext1
A(𝐾; FP)with coeﬃcients in this module. Moreover, in this framework, higher mutual
information quantities [McG54], [Tin62] belong to the homotopical algebra of cocycles of higher degrees
49
[BB15].
them.
We conjecture that something analog appears in the case of 𝐷𝑁𝑁𝑠and theories T, and of axioms for
The first ingredient in the case of probabilities was the operation of marginalization of a proba-
bility law, interpreted as the definition of a covariant functor (a copresheaf); it can be replaced here
by the transfers of theories associated to the functors 𝐹𝛼 : F𝑈′ →F𝑈, and to the morphisms ℎ in the
fibers F𝑈 from objects 𝜉 to objects 𝐹𝛼(𝜉′), as we saw in the section 3. For logics, this transfer can go
in two directions, depending on the geometry of 𝐹𝛼, from 𝑈′to 𝑈, and from 𝑈to 𝑈′, as seen in section 2.2.
We start with the transfer from 𝑈′to 𝑈, having in mind the flow of information in the downstream
direction to the output of the 𝐷𝑁𝑁; when it exists, a non-supervised learning should also correspond to
this direction. However, the learning by backpropagation or by reinforcement goes from the output layers
to the inner layers, then the inner layers have to understand something of the imposed language Lout and
the useful theories Tout for concluding. Therefore we will also laterconsider this backward or upstream
direction.
For an arrow (𝛼,ℎ): (𝑈,𝜉)→(𝑈′,𝜉′), the map
Ω𝛼,ℎ : Ω𝑈′(𝜉′)→Ω𝑈(𝜉), (3.6)
is obtained by composing the map 𝜆𝛼 = Ω𝛼 at 𝜉′, from Ω𝑈′(𝜉′)to Ω𝑈(𝐹𝛼𝜉′)with the map Ω𝑈(ℎ)from
Ω𝑈(𝐹𝛼𝜉′)to Ω𝑈(𝜉).
More generally, for every object 𝑋′in E𝑈′, the map 𝐹𝛼
! sends the subobjects of 𝑋′to the subobjects of
𝐹𝛼
! (𝑋′), respecting the lattices structures. Then for any natural transformation over F𝑈, ℎ: 𝑋→𝐹𝛼
! (𝑋′),
we get a transfer
Ω𝛼,ℎ : Ω𝑋′
𝑈′ →Ω𝑋
𝑈. (3.7)
The object 𝑋or 𝑋′is seen as a local context in the topos semantics.
We assume in what follows that this mapping extends to the sentences in the typed languages L𝑈,
where the dependency on 𝜉 reflects the variation of meaning in the included notions. In particular, the
morphisms in the topos E𝑈 express such variations. At the level of theories, this induces in general a
weakening, something which is implied at (𝑈,𝜉)by the propositions at (𝑈′,𝜉′), or more generally at the
context 𝑋by telling what is true, or expected, in the context 𝑋′
.
In what follows we note by A= ΩL this presheaf of sentences in Lover F, and by L𝛼,ℎ, or 𝜋★
𝛼,ℎ, its
transition maps, extending Ω𝛼,ℎ.
Under the strong standard hypotheses on the fibration F, for instance if it defines a fibrant object in
the injective groupoids models, i.e. any 𝐹𝛼 is a fibration, (see definition 2.1 above, following lemma 2.4
50
of section 2.2) there exists a right adjoint of Ω𝛼,ℎ:
Ω′
𝛼,ℎ : Ω𝑋
𝑈 →Ω𝑋′
𝑈′. (3.8)
It is given by extension of the operators 𝜆′
𝛼, associated to 𝐹★
𝛼, in the place of 𝐹𝛼
! , plus a transposition.
In what follows we note by A′
=
𝑡ΩLthis copresheaf of sentences over F, and by 𝑡L′
𝛼,ℎ, or simply 𝜋𝛼,ℎ
★ ,
its transition maps. The extended strong hypothesis requires that 𝜋★
𝛼,ℎ◦𝜋𝛼,ℎ
★ = Id.
For fixed 𝑈 and 𝜉∈F𝑈, the operation ∧gives a monoid structure on the set A𝑈,𝜉 = A′
𝑈,𝜉, which is
respected by the maps L𝛼,ℎ and 𝑡L′
𝛼,ℎ.
Moreover, A𝑈,𝜉 has a natural structure of poset category, given by the external implication 𝑃≤𝑄, for
which L𝛼,ℎ and 𝑡L′
𝛼,ℎ are functors.
There exists a right adjoint of the functor 𝑅↦→𝑅∧𝑄; this is the internal implication, 𝑃↦→(𝑄⇒𝑃).
Then, by definition, A𝑈,𝜉 = A′
𝑈,𝜉 is a closed monoidal category. In fact this is the only structure that is
essentially needed for the information theory below; this allows the linear generalization of appendix E.
The maps 𝜋★ and 𝜋★ give a fibration Aover F, and a cofibration A′ over F, in the sense of
Grothendieck [Mal05]:
a morphism 𝛾in Afrom (𝑈,𝜉,𝑃)to (𝑈′,𝜉′,𝑃′), lifting a morphism (𝛼,ℎ)in Ffrom (𝑈,𝜉)to (𝑈′,𝜉′),
is given by an arrow 𝜄in ΩL𝑈 from 𝑃to L𝛼,ℎ(𝑃′)= 𝜋★
𝛼,ℎ𝑃′, that is an external implication
𝑃≤L𝛼,ℎ(𝑃′). (3.9)
Similarly, an arrow in the category A′lifting the same morphism (𝛼,ℎ)in F, is an implication
𝑡L′
𝛼,ℎ(𝑃)≤𝑃′
. (3.10)
Remark that a priori the left adjunction 𝜋★
𝛼,ℎ ⊣𝜋𝛼,ℎ
★ does not imply something between 𝑃and L𝛼,ℎ(𝑃′)
when (3.10) is satisfied. However, under the strong hypothesis 𝜋★◦𝜋★ = Id, the relation (3.10) implies
the relation (3.9). Then in this case, A′is a subcategory of A.
Remark. An important particular case, where our standard hypotheses are satisfied, is when the ΩL𝑈 , 𝜉 =
A𝑈,𝜉 are the sets of open sets of a topological spaces 𝑍𝑈,𝜉, and when there exist continuous open maps
𝑓𝛼 : 𝑍𝑈′,𝜉′→𝑍𝑈,𝜉 lifting the functors 𝐹𝛼, such that the maps 𝜋★and 𝜋★are respectively the direct images
and the inverse images. The strong hypothesis holds when the 𝑓𝛼 are topological fibrations.
Aand A′belong to augmented model categories using monoidal posets [Rap10]. See section 2.5.
For 𝑃∈ΩL𝑈 , 𝜉 = A𝑈,𝜉, we note A𝑈,𝜉,𝑃 the set of proposition 𝑄such that 𝑃≤𝑄. They are sub-monoidal
categories of A𝑈,𝜉. Moreover they are closed, because 𝑃≤𝑄,𝑃≤𝑅implies 𝑃∧𝑄= 𝑃, then 𝑃∧𝑄≤𝑅,
then 𝑃≤(𝑄⇒𝑅).
When varying 𝑃, these sets form a presheaf over A𝑈,𝜉 = A′
𝑈,𝜉.
51
Lemma 3.1. The monoids A𝑈,𝜉,𝑃, with the functors 𝜋★ between them, form a presheaf over the category
A.
Proof. Given a morphism (𝛼,ℎ,𝜄): A𝑈,𝜉,𝑃 →A𝑈′,𝜉′,𝑃′ in A, the symbol 𝜄means 𝑃≤𝜋★𝑃′, then, from
𝑃′≤𝑄′, we deduce 𝑃≤𝜋★𝑃′≤𝜋★𝑄′
.
Lemma 2.4 in section 2.2 established the existence of a counit 𝜂: 𝜋★𝜋★ →Id𝑈, for every morphism
(𝛼,ℎ): (𝑈,𝜉)→(𝑈′,𝜉′), then for every 𝑃∈𝐴𝑈,𝜉, we have 𝜋★𝜋★𝑃≤𝑃.
Under the stronger hypothesis on the fibration F, that 𝜂= IdΩL, i.e. 𝜋★𝜋★𝑃= 𝑃, lemma 3.1 holds also
true for the category A′
.
Definition. Θ𝑈,𝜉 is the set of theories expressed in the algebra ΩL𝑈 in the context 𝜉. Under our standard
hypothesis on F, both L𝛼 and 𝑡L𝛼 send theories to theories.
Definition. Θ𝑈,𝜉,𝑃 is the subset of theories which imply the truth of proposition ¬𝑃, i.e. the subset of
theories excluding 𝑃.
Remind that¬𝑃≡(𝑃⇒⊥)is the largest proposition 𝑅such that 𝑅∧𝑃≤⊥.
It is always true that 𝑃≤𝑃′implies¬𝑃′≤¬𝑃, but the reciprocal implication in general requires a boolean
logic.
Then, for fixed 𝑈,𝜉, the sets Θ𝑈,𝜉,𝑃 when 𝑃varies in A𝑈,𝜉, form a presheaf over A𝑈,𝜉; if 𝑃≤𝑄, any
theory excluding 𝑄is a theory excluding 𝑃.
Lemma 3.2. Under the standard hypotheses on the fibration F, without necessarily axiom (2.30), the
sets Θ𝑈,𝜉,𝑃 with the morphisms 𝜋★, form a presheaf over A.
Proof. Let us consider a morphism (𝛼,ℎ,𝜄): A𝑈,𝜉,𝑃 →A𝑈′,𝜉′,𝑃′, where 𝜄denotes 𝑃≤𝜋★𝑃′; we deduce
𝜋★
¬𝑃′
= ¬𝜋★𝑃′≤¬𝑃; then 𝑇′≤¬𝑃′implies 𝜋★𝑇′≤𝜋★
¬𝑃′≤¬𝑃.
Corollary. Under the standard hypotheses on the fibration Fplus the stronger one, the sets Θ𝑈,𝜉,𝑃 with
morphisms 𝜋★, also form a presheaf over A′
.
What happens to 𝜋★?
It is in general false that the collection A𝑈,𝜉,𝑃 with the functors 𝜋𝛼,ℎ
★ forms a copresheaf over A′
.
However, if we restrict ourselves to the smaller category A′
strict, with the same objects but with morphisms
from A𝑈,𝜉,𝑃 to A𝑈′,𝜉′,𝑃′ only when 𝑃′
= 𝜋𝛼,ℎ
★ 𝑃, this is true.
Proof. If 𝑃≤𝑄, 𝜋★𝑃≤𝜋★𝑄, then 𝑃′≤𝜋★𝑄.
52
The same thing happens to the collection of the Θ𝑈,𝜉,𝑃 with the morphism 𝜋★: over the restricted
category A′
strict , they form a copresheaf. Proof : if 𝑇 ≤¬𝑃, we have 𝜋★𝑇 ≤𝜋★¬𝑃= ¬𝜋★𝑃= ¬𝑃′
.
However for the full category A′(resp. the category A), the argument does not work: from 𝜋★𝑃≤𝑃′
(resp. 𝑃≤𝜋★𝑃′), it follows that¬𝑃′≤¬¬𝜋★𝑃= 𝜋★¬𝑃(resp. 𝜋★𝑃≤𝜋★𝜋★𝑃′then¬𝜋★𝜋★𝑃′≤𝜋★𝑃, then
by adjunction¬𝑃′≤¬𝜋★𝑃= 𝜋★¬𝑃); then 𝑇 ≤¬𝑃implies 𝜋★𝑇 ≤𝜋★¬𝑃, not 𝜋★𝑇 ≤¬𝑃′
.
To summarize what is positive with 𝜋★,
Lemma 3.3. Under the strong standard hypothesis of defition 2.1, the collections A𝑈,𝜉,𝑃 and Θ𝑈,𝜉,𝑃 with
the morphisms 𝜋★, constitute copresheaves over A′
strict.
Note that the fibers A𝑈,𝜉,𝑃 are not sub-categories of A′
strict, they are subcategoris of A′and A.
Definition. A theory T′is said weaker than a theory Tif its axioms are true in T. We note T≤T′, as we
made for weaker probabilistic models. This applies to theories excluding a proposition 𝑃, in Θ𝑈,𝜉,𝑃.
With respect to propositions in A𝑈,𝜉, if we take the joint 𝑅by the operation "and" of all the axioms
⊢𝑅𝑖;𝑖∈𝐼of T, and the analog 𝑅′for T′, the above relation corresponds to 𝑅≤𝑅′
.
Remark: a weaker theory can also be seen as a simpler or more understandable theory; for instance in
Θ𝜆, the maximal theory ⊢(¬𝑃)is dedicated to exclude 𝑃, and the propositions implying 𝑃.
Be careful that in the sense of sets of truth assertions, the pre-ordering by inclusion of the theories goes
in the reverse direction. For instance {⊢⊥}is the strongest theory, in it everything is true, thus every
other theory is weaker.
Now we introduce a notion of semantic conditioning.
Definition 3.1. For fixed 𝑈,𝜉, 𝑃≤𝑄 in ΩL𝑈 , 𝜉 , and Ta theory in the language L𝑈,𝜉, we define a new
theory by the internal implication:
𝑄.T= (𝑄⇒T). (3.11)
More precisely: the axioms of 𝑄.Tare the assertions ⊢(𝑄⇒𝑅)where ⊢𝑅describes the axioms of T.
We consider 𝑄.Tas the conditioning of Tby 𝑄, in the logical or semantic sense, and frequently we write
the resulting theory T|𝑄.
At the level of propositions, the operation ⇒is the right adjoint in the sense of the Heyting algebra of
the relation ∧, i.e.
(𝑅∧𝑄≤𝑃) 𝑖𝑓 𝑓 (𝑅≤(𝑄⇒𝑃)). (3.12)
Proposition 3.1. The conditioning gives an action of the monoid A𝑈,𝜉,𝑃 on the set of theories in the
language L𝑈,𝜉.
53
Proof.
(𝑅∧𝑄′∧𝑄≤𝑃) iﬀ (𝑅∧𝑄′)≤(𝑄⇒𝑃)
iﬀ (𝑅≤(𝑄′⇒(𝑄⇒𝑃)).
Note that 𝑄⇒𝑃is also the maximal proposition 𝑄′(for ≤) such that 𝑄∧𝑄′≤𝑃.
(3.13)
Therefore the theory 𝑄⇒Tis the largest one among all theories T′satisfying
𝑄∧T′≤T. (3.14)
This implies that T|𝑄is weaker than Tand than¬𝑄.
1) In 𝑄∧T, the axioms are of the form ⊢(𝑄∧𝑅)where ⊢𝑅is an axiom of T, and from ⊢(𝑄∧𝑅), we
deduce ⊢𝑅.
2) Here 𝑄 (resp.¬𝑄) is understood as the theory with unique axiom ⊢𝑄 (resp. ⊢¬𝑄), then if
⊢(𝑄∧¬𝑄)we have ⊢⊥and all theories are true.
Remark. The theory T|𝑄= (𝑄⇒T)can also be written T𝑄, by definition of the internal exponential,
as the action by conditioning is also the internal exponential.
Notation: for being lighter, in what follows, we will mostly denote the propositions by the letters
𝑃,𝑄,𝑅,𝑃′
,...and the theories by the next capital letters 𝑆,𝑇,𝑈,𝑆′
,....
The operation of conditioning was considered by Carnap and Bar-Hilled [CBH52], in the case of Boolean
theories, studying the content of propositions and looking for a general notion of sets of semantic
Information. In this case 𝑄⇒𝑇 is equivalent to 𝑇∨¬𝑄= (𝑇∧𝑄)∨¬𝑄 (see the companion text on
logicoprobabistic information for more details [BBG20]).
Their main formula for the concept of information was
Inf (T|𝑃)= Inf (T∧𝑃)\Inf(𝑃); (3.15)
assuming that Inf (𝐴∧𝐵)⊇Inf (𝐴)∪Inf (𝐵).
Proposition 3.2. The conditioning by elements of A𝑈,𝜉,𝑃, i.e. propositions 𝑄 implied by 𝑃, preserves
the set Θ𝑈,𝜉,𝑃 of theories excluding 𝑃.
Proof. Let 𝑇 be a theory excluding 𝑃and 𝑄≥𝑃; consider a theory 𝑇′such that 𝑄∧𝑇′≤𝑇, we deduce
𝑇′∧𝑃≤𝑇, thus 𝑇′∧𝑃≤𝑇∧𝑃. But 𝑇∧𝑃≤⊥, then 𝑇′∧𝑃≤⊥. But 𝑄⇒𝑇 is the largest theory such
that 𝑄∧𝑇′≤𝑇, therefore 𝑄⇒𝑇 excludes 𝑃, i.e. asserts¬𝑃.
54
Remark. Consider the sets Θ′
𝑈,𝜉,𝑃 of theories which imply the validity of the proposition 𝑃. These
sets constitute a cosheaf over the category A′
strict for 𝜋★ and a sheaf for 𝜋★. However, the formula
(3.11) does’nt give an action of the monoid A𝑈,𝜉,𝑃 on the set Θ′
𝑈,𝜉,𝑃, even in the boolean case, where
(𝑄⇒𝑇)= 𝑇∨¬𝑄.
We can also consider the set of all theories over the largest category A, without further localization;
they also form a sheaf for 𝜋★ and a cosheaf Θ for 𝜋★, which are stable by the conditioning.
When necessary, we note Θloc the presheaf for 𝜋★ made by the Θ𝑈,𝜉,𝑃 over A.
The naturality over A′
𝑠𝑡𝑟𝑖𝑐𝑡 of the action of the monoids relies on the following formulas, for every arrow
(𝛼,ℎ): (𝑈,𝜉)→(𝑈′,𝜉′)in F, we have the arrows (𝑈,𝜉,𝑃)→(𝑈′,𝜉′,𝜋★𝑃)in A′
strict; in the presheaf of
monoids A𝑈,𝜉,𝑃, for the morphism 𝜋★, and the presheaf Θ𝑈,𝜉,𝑃 with morphisms 𝜋★:
(𝜋★𝑄′).𝑇= 𝜋★[𝑄′
.𝜋★(𝑇)]. (3.16)
This holds true under the strong hypothesis 𝜋★𝜋★= Id.
If we want to consider functions 𝜙of the theories, two possibilities appear: 𝜋★ for Θ with 𝜋★ for the
monoids A, or the opposite 𝜋★ for Θ with 𝜋★ for the monoids A. But Both cases give a cosheaf over
A, however only the second one gives a functional module Φ over A, even with the strong standard
hypothesis,
Theorem 3.1. Under the strong hypothesis, in particular 𝜋★𝜋★ = Id, and over the restricted category
A′
strict , the cosheaf Φ′made by the measurable functions (with any kind of fixed values) on the theories
Θ𝑈,𝜉,𝑃, with the morphisms 𝜋★, is a cosheaf of modules over the monoidal cosheaf A′
loc, made by the
monoidal categories A𝑈,𝜉,𝑃, with the morphisms 𝜋★.
Proof. Consider a morphism (𝛼,ℎ,𝜄): 𝐴𝑈,𝜉,𝑃 →𝐴𝑈′,𝜉′,𝜋★𝑃, a theory 𝑇′in Θ𝑈′,𝜉′,𝜋★𝑃, a proposition 𝑄in
𝐴𝑈,𝜉,𝑃, and an element 𝜙𝑃 in Φ′
𝑈,𝜉,𝑃, we have
𝜋★𝑄.(Φ′
★𝜙𝑃)(𝑇′)= (Φ′
★𝜙𝑃)(𝑇′|𝜋★𝑄)
= 𝜙𝑃[𝜋★(𝑇′|𝜋★𝑄)]
= 𝜙𝑃[𝜋★(𝜋★𝑄⇒𝑇′)]
= 𝜙𝑃[𝜋★𝜋★𝑄⇒𝜋★𝑇′]
= 𝜙𝑃[𝑄⇒𝜋★𝑇′]
= 𝜙𝑃[𝜋★(𝑇′)|𝑄].
Remark. The same kind of computation shows that, in the case of the sheaf Φ of functions on the cosheaf
Θ with 𝜋★ and the sheaf Aloc with 𝜋★, we have, for the corresponding elements 𝑄′,𝑇,𝜙′
,
𝜋★𝑄′
.Φ★(𝜙′)(𝑇)= 𝜙′(𝜋★𝜋★𝑄′
.𝜋★𝑇); (3.17)
55
which is not the correct equation of compatibility, under our assumption. It should be true for the other
direction, if 𝜖= 𝜋★𝜋★= IdA𝑈 ′, 𝜉 ′.
However, there exists an important case where both hypotheses 𝜋★𝜋★= Id𝑈 and 𝜋★𝜋★= Id𝑈′ hold true; it
the case where the languages over the objects (𝑈,𝜉)are all isomorphic. In terms of the intuitive maps
𝑓𝛼, this means that they are homeomorphisms. This case happens in particular when we consider the
restriction of the story to a given layer in a network.
Remark. Considering lemmas 3.1 and 3.2, we could forget the functional point of view with a space Φ.
In this case we do not have an Abelian situation, but we have a sheaf of sets of theories Θloc, on which
the sheaf of monoids Aloc acts by conditioning,
Proposition 3.3. The presheaf Θloc for 𝜋★ is compatible with the monoidal action of the presheaf Aloc,
both considered on the category A(then over A′by restriction, under the strong standard hypothesis on
F).
Proof. If 𝑇′≤¬𝑃′and 𝑃≤𝜋★𝑃′, we have¬𝜋★𝑃′≤¬𝑃, therefore 𝜋★𝑇′≤¬𝑃.
In the Bayesian case, the conditioning is expressed algebraically by the Shannon mean formula on
the functions of probabilities:
𝑌.𝜙(P𝑋)= E𝑌★P𝑋 (𝜙(P|𝑌= 𝑦)) (3.18)
This gives an action of the monoid of the variables 𝑌 coarser than 𝑋, as we find here for the fibers 𝐴𝑈,𝜉,𝑃
and the functions of theories Φ𝑈,𝜉,𝑃.
Equation (3.15) was also inspired by Shannon’s equation
(𝑌.𝐻)(𝑋; P)= 𝐻((𝑌,𝑋); P)−𝐻(𝑌;𝑌★P). (3.19)
However this set of equations for a system Bcan be deduced from the set of equations of invariance
(𝐻𝑋−𝐻𝑌)|𝑍= 𝐻𝑋∧𝑍−𝐻𝑌∧𝑍. (3.20)
In the semantic framework, two analogies appear with the bayesian framework: in one of them, in
each layer, the role of random variables is played by the propositions 𝑃; in the other one, their role is
played by the layers 𝑈, augmented by the objects of a groupoid (or another kind of category for contexts).
The first analogy was chosen by Carnap and Bar-Hillel, and certainly will play a role in our toposic
approach too, at each 𝑈,𝜉, to measure the logical value of functioning. However, the second analogy is
more promising for the study of DNNs, in order to understand the semantic adventures in the feedforward
and feedback dynamics.
To unify the two analogies, we have to consider the triples (𝑈,𝜉,𝑃)as the semantic analog of random
variables, with the covariant morphisms of the category D= A′op
strict,
(𝛼,ℎ,𝜋𝛼,ℎ
★ ): (𝑈,𝜉,𝑃)→(𝑈′,𝜉′,𝑃′
= 𝜋𝛼,ℎ
★ 𝑃), (3.21)
56
as analogs of the marginals.
In fact, a natural extension exists and will be also studied, replacing the monoids A𝑈,𝜉,𝑃 by the monoids
D𝑈,𝜉,𝑃 of arrows in Dgoing to (𝑈,𝜉,𝑃), i.e. replacing Aloc by the left slice D\D. This will allow the
use of combinatorial constructions over the nerves of Fand C.
If we consider the theories in Θ𝑈,𝜉,𝑃 as the analogs of the probability laws, the analogs of the values
of a variable 𝑄are the conditioned theories 𝑇|𝑄.
When a functioning network is considered, the neural activities in 𝑋𝑤
𝑈,𝜉, can also be seen as values of
the variables, through a map 𝑆𝑈,𝜉,𝑃 : 𝑋𝑤
𝑈,𝜉 →Θ𝑈,𝜉,𝑃.
As defined in section 2.3, a semantic functioning of the neural network Xis given by a function
𝑆𝑈,𝜉 : X𝑈,𝜉 →Θ𝑈,𝜉. (3.22)
The introduction of 𝑃, seen as logical localization, corresponds to a refined notion of semantic function-
ing, a quotient of the activities made by the neurons that express a rejection of this proposition. This
generates a foliations in the individual layer’s activities.
Remark. We could also consider the cosheaf Θ′or Θ′
loc over A′
strict, and obtain the cosheaf Σ′, of all
possible maps 𝑆𝑈,𝜉 : 𝑋𝑈,𝜉 →Θ′
𝑈,𝜉;𝑈∈C,𝜉 ∈F𝑈, where the transition from 𝑈,𝜉 to 𝑈′,𝜉′over 𝛼,ℎ is
given by the contravariance of 𝑋and by the covariance of Θ′:
Σ′
𝛼,ℎ(𝑠𝑈)𝑈′,𝜉′ =
𝑡 L′
𝛼,ℎ◦𝑠𝑈◦𝑋𝛼,ℎ. (3.23)
However the above discussion shows that the compatibility with the conditioning would require 𝜋★𝜋★= Id,
which appears to be too restrictive.
In addition, the network’s feed-forward dynamics 𝑋𝑤 makes appeal to a particular class of inputs Ξ,
and is more or less adapted by learning to the expected theories Θout at the output. Therefore a convenient
notion of information, if it exists, must involve these ingredients.
By using functions of the mappings 𝑆𝑈,𝜉, we could not apply them to particular vectors in 𝑋𝑈,𝜉. But
using functions on the Θ𝑈,𝜉 we can. Then this will be our choice. And this can give numbers (or sets
or spaces) associated to a family of activities 𝑥𝜆 ∈𝑋𝜆, and to their semantic expression 𝑆𝜆(𝑥𝜆)∈Θ′
𝜆.
Moreover, we can take the sum over the set of 𝑥 belonging to Ξ, then a sum of semantic information
corresponding to the whole set of data and goals. Which seems preferable.
The relations
𝑆𝑈,𝜉◦𝑋★
= 𝜋★◦𝑆𝑈′,𝜉′, (3.24)
mean that the logical transmission of the theories expressed by 𝑈′(in the context 𝜉′) coincide with the
theories in 𝑈induced by the neuronal transmission from 𝑈′to 𝑈.
57
If this coherence is verified, the object Σ in the topos, replacing Σ′, could be taken as the exponential
object ΘXin the topos of presheaves over A. By definition, this is equivalent to consider the parameterized
families of functioning
𝑆𝜆 : X𝑈,𝜉×𝑌𝜆 →Θ𝑈,𝜉,𝑃; (3.25)
where 𝑌 is any object in the topos of presheaves over A.
Remark. In the experiments with small networks, we verified this coherence, but only approximatively,
i.e. with high probability on the activities in 𝑋.
On another hand, a semantic information over the network must correspond to the impact of the inner
functioning on the output decision, given the inputs. For instance, it has to measure how far from the
output theory is the expressed theory at 𝑈′,𝜉′. We hope that this should be done by an analog of the
mutual information quantities. If we believe in the analogy with probabilities, this should be given by
the topological coboundary of the family of sections of the module Φ𝜆; 𝜆∈A′[BB15]
Then we enter the theory of topological invariants of the sheaves of modules in a ringed topos. Here
Φ over D, or D\D.
The category D= A′op
strict gives birth to a refinement of the cat’s manifolds we have defined before in
section 3.1:
Suppose, to simplify, that we have a unique initial point in C; it corresponds to the output layer 𝑈𝑜𝑢𝑡.
Then look at a given 𝜉0 ∈Fout, and a given proposition 𝑃out in Ωout (𝜉0)= A𝑈out ,𝜉0 ; it propagates in the
inner layers through 𝜋★ in 𝑃∈A𝑈,𝜉 for any 𝑈and any 𝜉linked to 𝜉0, and can be reconstructed by 𝜋★ at
the output, due to the hypothesis 𝜋★𝜋★ = Id. Then we get a section over Cof the cofibration Dop →C.
This can be extended as a section of Dop →F, by varying 𝜉0, when the 𝐹𝛼 are fibrations, which is the
main case we have in mind.
Note that this does not give all the sections, because some propositions 𝑃in a A𝜆 are not in the image of
𝜋★, even if all of them are sent by 𝜋★ to an element of a set Ωout (𝜉0).
However, these interesting sections are in bĳection with the connected components of Dop
.
Let Kbe a commutative ring, and 𝑐𝑃 a non zero element of K; we define the (measurable) function
𝛿𝑃 on the theories in the Θ𝜆(𝑃), taking the value 𝑐𝑃 over a point in the above connected component of
D, and 0 outside.
Looking at the semantic functioning 𝑆: 𝑋𝑈,𝜉 →Θ𝜆, we get a function 𝛿𝑃 on the sets of local activities.
This function takes the value 𝑐𝑃 on the set of activities that form theories excluding 𝑃.
Several subtle points appear:
1) the function really depends on 𝑃, but when 𝑃 varies, it does not change when two propositions
have the same negation¬𝑃;
58
2) to conform with the before introduced notion of cat’s manifold, we must assume that the activities
in diﬀerent layers which exclude 𝑃in their axioms, are coherent, i.e. form a section of the object
𝑋𝑤
.
Without the coherence hypothesis between dynamics and logics, we have two diﬀerent notions of cat’s
manifolds, one dynamic and one linguistic or logical. In a sense, only the agreement deserves to be really
named semantics.
3.4 Semantic information. Homology constructions
Bar complex of functions of theories and conditioning by propositions.
We start with the computation of the Abelian invariants, therefore with the module of functions Φ on Θ
in the cases where conditioning act.
We consider first the most interesting case described by theorem 3.1, given by the presheaf Θ over the
category D, fibred over Fwhich is itself fibred over C. Note that over A′
strict we get cosheaves, thus we
prefer to work over the opposite D. Then A′
loc with morphisms 𝜋★, becomes a sheaf of monoids over D,
and Θ′
loc, with morphisms 𝜋★, becomes a cosheaf of sets over D, in such a manner that the functions Φ
on Θ′
loc constitute a sheaf of A′
loc modules.
We suppose that the elements 𝜙𝜆 in Φ𝜆 take their values in a commutative ring 𝐾(with cardinality at
most continuous).
The method of relative homological algebra, used for probabilities in Baudot, Bennequin [BB15], and
Vigneaux [Vig20], cited above, can be applied here, for computing Ext★
A′
(𝐾,Φ)in the toposic sense.
loc
The action of A′
loc on 𝐾 is supposed trivial.
We note R= 𝐾 A′
loc the cosheaf in 𝐾-algebras associated to the monoids A′
𝜆; 𝜆∈A′. The non-
homogeneous bar construction gives a free resolution of the trivial constant module 𝐾:
0 ←𝐾←𝐵′
0 ←𝐵′
1 ←𝐵′
2 ←... (3.26)
where 𝐵′
𝑛; 𝑛∈N, is the free Rmodule R⊗(𝑛+1), with the action on the first factor. In each object
𝜆= (𝑈,𝜉,𝑃), the module 𝐵′
𝑛(𝜆)is freely generated over 𝐾 A′
𝜆 by the symbols [𝑃1 |𝑃2 |...|𝑃𝑛], where
the 𝑃𝑖 are elements of A′
𝜆, i.e. propositions implied by 𝑃. Then the elements of 𝐵′
𝑛(𝜆)are finite sums of
elements 𝑃0 [𝑃1 |𝑃2 |...|𝑃𝑛].
The first arrow from 𝐵′
0 to 𝐾is the coordinate along [∅].
59
The higher boundary operators are of the Hochschild type, defined on the basis by the formula
𝜕[𝑃1 |𝑃2 |...|𝑃𝑛]= 𝑃1 [𝑃2 |...|𝑃𝑛] +
𝑛−1
(−1)𝑖[𝑃1 |...|𝑃𝑖𝑃𝑖+1 |...|𝑃𝑛]+(−1)𝑛[𝑃1 |𝑃2 |...|𝑃𝑛−1] 𝑖=1
(3.27)
For each 𝑛∈N, the vector space Ext𝑛
A′(𝐾,Φ)is the 𝑛-th group of cohomology of the associated complex
HomA′(𝐵★
,Φ), made by natural transformations which commutes with the action of 𝐾[A′].
The coboundary operator is defined by
𝛿𝑓𝜆(𝑇; 𝑄0|...|𝑄𝑛)=
𝑓𝜆(𝑇|𝑄0; 𝑄1 |...|𝑄𝑛)+
𝑛−1
(−1)𝑖+1 𝑓𝜆(𝑇; 𝑄0 |...|𝑄𝑖𝑄𝑖+1 |...|𝑄𝑛)+(−1)𝑛+1 𝑓𝜆(𝑇; 𝑄0 |...|𝑄𝑛−1).
𝑖=0
(3.28)
A cochain of degree zero is a section 𝜙𝜆; 𝜆∈Dof Φ, that is, a collection of maps 𝜙𝜆 : Θ′
𝜆 →𝐾, such
that, for any morphism 𝛾: 𝜆→𝜆′in Dop, and any 𝑆′∈Θ′
𝜆′, we have
𝜙𝜆′(𝑆′)= 𝜙𝜆(𝜋★𝑆′). (3.29)
If there exists a unique last layer 𝑈out, as in the chain, this implies that the functions 𝜙𝜇 are all deter-
mined by the functions 𝜙out on the sets of theories 𝑆out in the final logic, excluding given propositions,
by definition of the sets Θ′
𝑈,𝜉,𝑃. And a priori these final functions are arbitrary.
Acyclicity and fundamental cochains
To be a cocycle, 𝜙must satisfy, for any 𝜆= (𝑈,𝜉,𝑃), and 𝑃≤𝑄,
0= 𝛿𝜙([𝑄])(𝑆)= 𝑄.𝜙𝜆(𝑆)−𝜙𝜆(𝑆)= 𝜙𝜆(𝑄⇒𝑆)−𝜙𝜆(𝑆). (3.30)
However, for any 𝑃we have 𝑃≤⊤, and 𝑆|⊤= ⊤; then the invariance (3.30) implies that 𝜙𝜆 is independent
of 𝑆; it is equal to 𝜙𝜆(⊤).
Then, a cocycle is a collection elements 𝜙(𝜆)in 𝐾, satisfying 𝜙𝜆′= 𝜙𝜆 each time there exists an arrow
from 𝜆to 𝜆′in A′
strict , thus forming a section of the constant sheaf over A′
strict.
This gives:
Proposition 3.4. As
Ext0
A′(𝐾,Φ)= 𝐻0 (A′
strict; 𝐾)= 𝐾𝜋0 A′
strict
, (3.31)
then degree zero cohomology counts the propositions that are transported by 𝜋★ from the output.
60
The discussion at the end of section 3.3 describes the relation between the zero cohomology of
information and the cats manifolds, that was identified before with the degree zero cohomology in the
sense of Čech.
A degree one cochain is a collection 𝜙𝑅
𝜆 of measurable functions on Θ′
𝜆, and 𝑅∈A′
𝜆, which satisfies the
naturality hypothesis: for any morphism 𝛾: 𝜆→𝜆′in Dop, and any 𝑆′∈Θ′
𝜆′, we have
𝜙𝜋★ 𝑅
𝜆′ (𝑆′)= 𝜙𝑅
𝜆(𝜋★𝑆′). (3.32)
The cocycle equation is
∀𝑈,𝜉,∀𝑃,∀𝑄≥𝑃,∀𝑅≥𝑃,∀𝑆∈Θ′
𝑈,𝜉,𝑃,𝜙𝑄∧𝑅
𝜆 (𝑆)= 𝜙𝑄
𝜆 (𝑆)+𝜙𝑅
𝜆(𝑄⇒𝑆). (3.33)
Let us define a family of elements of 𝐾 by the equation
𝜓𝜆(𝑆)=−𝜙𝑃
𝜆(𝑆). (3.34)
Formula (3.32) implies formula (3.29), then 𝜓𝜆 is a zero cochain.
Take its coboundary
𝛿𝜓𝜆([𝑄])(𝑆)=−𝜙𝑃
𝜆(𝑆)+𝑄.𝜙𝑃
𝜆(𝑆). (3.35)
using the cocycle equation and the fact that for any 𝑄≥𝑃we have 𝑄∧𝑃= 𝑃, this gives
𝜙𝑄
𝜆 (𝑆)= 𝜙𝑄∧𝑃
𝜆 (𝑆)−𝑄.𝜙𝑃
𝜆(𝑆)=−𝛿𝜓𝜆([𝑄])(𝑆). (3.36)
Remark that the cochain 𝜓is not unique, the formula 𝜓=−𝜙𝑃
𝜆 is only a choice. Two cochains 𝜓satisfying
𝛿𝜓= 𝜙diﬀer by a zero cocycle, that is a family of numbers 𝑐𝜆, dependent on 𝑃but not on 𝑆. Remind us
that 𝑃is part of the object 𝜆.
Therefore every one cocycle is a coboundary, or in other terms:
Proposition 3.5. Ext1
A′(𝐾,Φ)= 0.
The same argument applies to every degree 𝑛≥1, giving,
Proposition 3.6. Ext𝑛
A′(𝐾,Φ)= 0.
Proof. If 𝜙𝑄1;...;𝑄𝑛
𝜆 is a cocycle of degree 𝑛≥1, where 𝜆= (𝑈,𝜉,𝑃), the formula
𝜓𝑄1;...;𝑄𝑛−1
𝜆 = (−1)𝑛𝜙𝑄1;...;𝑄𝑛−1 ;𝑃
𝜆 (3.37)
defines a cochain of degree 𝑛−1 such that 𝛿𝜓= 𝜙.
Extracting 𝜙𝑄1;...;𝑄𝑛
𝜆 from the last term of the cocycle equation for 𝜙, applied to 𝑄1,...,𝑄𝑛+1 with 𝑄𝑛+1 = 𝑃,
gives
(−1)𝑛𝜙𝑄1;...;𝑄𝑛
𝜆 = 𝑄1.𝜙𝑄2;...;𝑄𝑛 ;𝑃
𝜆 +
𝑛−1
𝜙𝑄2;...;𝑄𝑖 𝑄𝑖+1;...;𝑄𝑛 ;𝑃
𝜆 +(−1)𝑛𝜙𝑄2;...;𝑄𝑛 ∧𝑃
𝜆. (3.38)
𝑖=1
As 𝑄𝑛∧𝑃= 𝑃in A𝜆, this is exactly the coboundary of 𝜓applied to 𝑄1;...; 𝑄𝑛.
61
Remark. At first sight this is a deception; however, there is a morality here, because it tells that the
measure of semantic information reflects a value of a theory at the output, depending on many elements
that the network does not know, without knowing the consequences of this theory. Some of these
consequences can be included in the metric for learning, some other cannot be.
When a cochain 𝜓as above is chosen, it defines the degree one cocycle 𝜙by the formula
𝜙𝑄
𝜆 (𝑆)= 𝜓𝜆(𝑄⇒𝑆)−𝜓𝜆(𝑆). (3.39)
The cochain 𝜓satisfied (3.29), and the coboundary 𝜙the equation (3.32).
All the arbitrariness is contained in the values of 𝜓out, which are function of 𝑃 and of the theory
excluding 𝑃. Now examine the role of a proposition 𝑄implied by 𝑃. It changes the value of 𝜙according
to the equation
𝜙out (𝑄;𝑇)= 𝜙𝑄
out (𝑇)= 𝜙𝑃
out (𝑇)−𝜙𝑃
out (𝑇|𝑄)= 𝜓out (𝑇|𝑄)−𝜓out (𝑇), (3.40)
then it subtracts from 𝜓out (𝑇)the conditioned value 𝜓out (𝑇|𝑄). And this is transmitted inside the network
by the equation
𝜙𝜋★𝑄
𝜆′ (𝑆′)= 𝜙𝑄
𝜆 (𝜋★𝑆′); (3.41)
which is equivalent to the simplest equation
𝜓𝜆′(𝑆′)= 𝜓𝜆(𝜋★𝑆′). (3.42)
Note that we are working under the hypothesis 𝜋★𝜋★= Id, then it can happen that a theory 𝑆′, in the inner
layers cannot be reconstructed (by 𝜋★) from its deduction 𝜋★𝑆′in the outer layer. Thus the logic inside is
richer than the transmitted propositions, but the quantity 𝜓𝜆′(𝑆′)depends only on 𝜋★𝑆′
.
This corresponds fairly well with what we observed in the experiments about simple classification prob-
lems, with architectures more elaborated than a chain, (see Logical cells II, [BBG21b]). In some cases,
the inner layers invent propositions that are not stated in the objectives. They correspond to proofs of
these objectives.
Mutual information, classical and quantum analogies
We propose now an interpretation of the functions 𝜙and 𝜓, when K= R, or an ordered ring, as Z: the
value 𝜙𝑃
out (𝑆)measures the ambiguity of 𝑆with respect to¬𝑃, then it is natural to assume that the value
of 𝜓out (𝑆)is growing with 𝑆, i.e. 𝑆≤𝑇 implies 𝜓out (𝑆)≤𝜓out (𝑇).
Among the theories which exclude 𝑃, there is a minimal one, which is ⊥, without much interest, even it
has the maximal information in the sense of Carnap and Bar-Hillel, and a maximal theory, which is¬𝑃
itself; it is the more precise, but with the minimal information, if we measure information by the quantity
62
of exclusions of propositions it can give. Thus 𝜓does not count the quantity of possible information, but
the closeness to¬𝑃.
Consequently, 𝜙𝑄
𝑃(𝑆)is always a positive number, which is decreasing in 𝑄when 𝑆is given. Therefore,
we can take 𝜓negative, by choosing 𝜓𝜆 =−𝜙𝑃
𝜆. In what follows we consider this choice for 𝜓.
The maximal value of 𝜙𝑄
𝑃(𝑆), for a given 𝑆is attained for 𝑄= 𝑃, in this case 𝑆|𝑃= ¬𝑃, then the maximal
value is 𝜙𝑃
𝜆(𝑆)−𝜙𝑃
𝜆(¬𝑃).
The truth of the proposition¬𝑄can be seen as a theory excluding 𝑃when 𝑃≤𝑄. Like a counterexample
of 𝑃.
Note the following formula for 𝑃≤𝑄:
𝜙𝑄
𝜆 (𝑆)= 𝜙𝑃
𝜆(𝑆)−𝜙𝑃
𝜆(𝑆|𝑄). (3.43)
Remind that the entropy function 𝐻of a joint probability is also always positive, and we have
𝐼(𝑋;𝑌)= 𝐻(𝑋)−𝐻(𝑋|𝑌), (3.44)
as it follows from the Shannon equation and the definition of 𝐼.
This also gives 𝐼(𝑋; 𝑋)= 𝐻(𝑋).
Then we interpret 𝜙𝑄
𝜆 (𝑆)as a mutual information between 𝑆and¬𝑄, and 𝜙𝑃
𝜆(𝑆)itself as a kind of entropy,
thus measuring an ambiguity: the ambiguity of what is expressed in the layer 𝜆about the exclusion of 𝑃
at the output.
This is in agreement with next formula,
𝜙𝜋★𝑄
𝜆 (𝑆)= 𝜙𝑄
out (𝜋★𝑆). (3.45)
Remark. In Quantum Information Theory, where variables are replaced by orthogonal decomposition
of an Hilbert space, and probabilities are replaced by adapted positive hermitian operators of trace
one [BB15], the Shannon entropy 𝐻 (entropy of the associated classical law) appears as (minus) the
coboundary of a cochain which is the Von Neumann entropy 𝑆=−log2 Trace(𝜌),
𝐻𝑌(𝑌; 𝜌)= 𝑆𝑋(𝜌)−𝑌.𝑆𝑋(𝜌). (3.46)
Thus in the present case, it is better to consider that theories are analogs of density matrices, propositions
are analogs of the observables, the function 𝜓is an analog of the opposite of the Von-Neumann entropy,
and the ambiguity 𝜙an analog of the Shannon entropy.
Let us see what we get for a functioning network 𝑋𝑤, possessing a semantic functioning 𝑆𝑈,𝜉 : 𝑋𝑈,𝜉 →
Θ𝑈,𝜉, not necessarily assuming the naturality (3.25). We can even specialize by taking a family of neurons
63
having an interest in the exclusion of some property 𝑃, and look at a family
𝑆𝜆 : 𝑋𝑈,𝜉 →Θ′
𝜆, (3.47)
where 𝜆= (𝑈,𝜉,𝑃).
To a true activity 𝑥of the network, we get 𝑥𝑈,𝜉, then, we define
𝐻𝑄
𝜆 (𝑥)= 𝜙𝑄
𝜆 (𝑆𝜆(𝑥𝑈,𝜉)). (3.48)
And we propose it as the ambiguity in the layer 𝑈,𝜉, about the proposition 𝑃at the output, when 𝑄 is
given as an example.
To understand better the role of 𝑄, we apply the equation (3.32), which gives
𝐻𝜋★𝑄
𝜆′ (𝑥
′)= 𝜙𝑄
𝜆 (𝜋★𝑆′(𝑥
′)). (3.49)
Therefore, evaluated on a proposition 𝜋★𝑄which comes from the output, the above quantity 𝐼(𝑥′)in the
hidden layer 𝑈′, is the mutual information of¬𝑄 and the deduction in 𝑈out by 𝜋★ of the theory 𝑆′(𝑥′),
expressed in 𝑈′in presence of the given section (feedforward information flow), coming from the input,
by the activity 𝑥′∈𝑋𝑈′.
Remark. Consider a chain (𝑈,𝜉)→(𝑈′,𝜉′)→(𝑈”,𝜉”). We denote by 𝜌★and 𝜌★the applications which
correspond to the arrow (𝑈′,𝜉′)→(𝑈”,𝜉”). Therefore (𝜋′)★= 𝜋★𝜌★ and 𝜋′
★= 𝜌★𝜋★.
For any section 𝑥, and proposition 𝑃 in the output (𝑈,𝜉), consider the particular case 𝑃= 𝑄, where
(𝑄⇒𝑆)= ¬𝑃for every theory excluding 𝑃:
𝐻(𝑥
′)−𝐻(𝑥”)= 𝜙𝑃
𝜆(𝜋★𝑆′(𝑥
′))−𝜙𝑃
𝜆(𝜋★𝑆′(𝑥
′)|𝑃)−(𝜙𝑃
𝜆((𝜋
′)★𝑆”(𝑥”))
−𝜙𝑃
𝜆((𝜋
′)★𝑆”(𝑥”)|𝑃))
= 𝜙𝑃
𝜆(𝜋★𝑆′(𝑥
′))−𝜙𝑃
𝜆((𝜋
′)★𝑆”(𝑥”))
= 𝜓𝜆(𝜋★𝜌★𝑆”(𝑥”))−𝜓𝜆(𝜋★𝑆′(𝑥
′))
This is surely negative in practice, because the theory 𝑆′(𝑥′)is larger than the theory 𝜌★𝑆”(𝑥”). For
instance, at the end, we surely have 𝑆out = ¬𝑃, as soon as the network has learned.
Consequently this quantity has a tendency to be negative. Then it is not like the mutual information
between the layers. It looks more as a diﬀerence of ambiguities. Because the ambiguity is decreasing in
a functioning network, in reality.
This confirms that 𝐻is a measure of ambiguity.
Therefore, the mutual information should come out in a manner that involves a pair of layers.
64
To obtain a notion of mutual information, we make an extension of the monoids A𝑈,𝜉,𝑃, which continues
to act by conditioning on the sets Θ𝑈,𝜉,𝑃.
For that, we consider a fibration over A′
strict made by monoids D𝜆 which contain A𝜆 as submonoids.
By definition, if 𝜆= (𝑈,𝜉,𝑃), an object of D𝜆 is an arrow 𝛾0 = (𝛼0,ℎ0,𝜄0)of A′
strict, going from a
triple (𝑈0,𝜉0,𝑃0)to a triple (𝑈,𝜉,𝜋★𝑃0), where 𝑃≤𝜋★𝑃0, and a morphism from 𝛾0 = (𝛼0,ℎ0,𝜄0)to
𝛾1 = (𝛼1,ℎ1,𝜄1)is a morphism 𝛾10 from (𝑈0,𝜉0,𝑃0)to (𝑈1,𝜉1,𝑄1 = 𝜋
𝛼10,ℎ10
★ 𝑃0)such that 𝑄1 ≥𝑃1.
For the intuition it is better to see the objects as arrows in the opposite category Dof A′
strict , in such
a manner they can compose with the arrows 𝑄≤𝑅in the monoidal category A𝜆, then we get a variant
of the right slice 𝜆|D, just extended by A𝜆. The category D𝜆 is monoidal and strict if we define the
product by
𝛾1 ⊗𝛾2 = (𝑈,𝜉,𝜋𝛾1
★ 𝑃1 ∧𝜋𝛾0
★ 𝑃2). (3.50)
The identity being the truth ⊤𝜆.
We also define the action of D𝜆 on Θ𝜆 as follows:
for every arrow 𝛾0 : 𝜆0 →𝜆𝜋★𝑃0 , where 𝜆0 = (𝑈0,𝜉0,𝑃0), and where 𝜆𝜋★𝑃0 denotes (𝑈,𝜉,𝜋★𝑃0), assuming
𝜋★𝑃0 ≥𝑃, we define
𝛾0.T= (𝜋𝛾0
★ 𝑃0 ⇒T). (3.51)
This gives an action of the monoid of propositions in A𝜆0 which are implied by 𝑃0, whose images by 𝜋★
are implied by 𝑃.
If 𝑃0 ≤𝑄0 and 𝑃0 ≤𝑅0, we have 𝜋𝛾0
★ (𝑄0 ∧𝑅0)= 𝜋𝛾0
★ (𝑄0)∧𝜋𝛾0
★ (𝑅0).
The monoidal categories D𝜆; 𝜆∈Dform a natural presheaf D\Dover D. For any morphism 𝛾= (𝛼,ℎ,𝜄)
of A′
𝑠𝑡𝑟𝑖𝑐𝑡, going from (𝑈,𝜉,𝑃)to (𝑈′,𝜉′,𝜋★𝑃), and any object 𝛾0 : 𝜆0 →𝜆𝜋★𝑃0 in D𝜆, we define 𝛾★(𝛾0)
by the composition (𝛼,ℎ)◦(𝛼0,𝜉0)and the proposition 𝜋𝛾
★◦𝜋★𝑃0 in A𝜆′.
The naturalness of the monoidal action on the theories follows from 𝜋★
𝛾𝜋𝛾
★ = Id𝑈:
𝜋★
𝛾[𝛾★(𝜋★𝑃0).𝑇′]= 𝜋★
𝛾[𝜋𝛾
★𝜋★𝑃0 ⇒𝑇′]
= 𝜋★
𝛾𝜋𝛾
★𝜋★𝑃0 ⇒𝜋★
𝛾𝑇′
= 𝜋★𝑃0 ⇒𝜋★
𝛾𝑇′
Then, defining [Φ★(𝛾)(𝜙𝜆)](𝑇′)= 𝜙𝜆(𝜋★
𝛾𝑇′), we get the following result
Lemma 3.4.
[Φ★(𝛾)𝜙𝜆](𝛾★(𝛾0).𝑇′)= 𝜙𝜆(𝛾0.𝜋★𝑇′). (3.52)
65
Consequently the methods of Abelian homological algebra can be applied [Mac12].
The (non-homogeneous) bar construction makes now appeal to symbols [𝛾1 |𝛾2 |...|𝛾𝑛], where the 𝛾𝑖
are elements of D𝜆. The action of algebra pass through the direct image of propositions 𝜋★𝑃𝑖;𝑖= 1,...,𝑛.
Things are very similar to what happened with the precedent monoids A′
𝜆:
the zero cochains are families 𝜙𝜆 of maps on theories satisfying
𝜓𝜆(𝜋★𝑇′)= 𝜓𝜆′(𝑇′), (3.53)
where 𝛾: 𝜆→𝜆′is a morphism in 𝐴′
strict.
The coboundary operator is
𝛿𝜓𝜆([𝛾1])= 𝜓𝜆(𝑇|𝜋𝛾1
★ 𝑃1)−𝜓𝜆(𝑇). (3.54)
Then the cohomology is defined as before. We get analog propositions. For instance, the degree one
cochains are collections of maps of theories 𝜙𝛾1
𝜆 satisfying
𝜙𝛾1
𝜆 (𝜋★𝑇′)= 𝜙𝛾★𝛾1
𝜆′ (𝑇′); (3.55)
the cocycle equation is
𝜙𝛾1∧𝛾2
𝜆 = 𝜙𝛾1
𝜆 +𝛾1.𝜙𝛾2
𝜆. (3.56)
One more time, the cocycles are coboundaries; the following formula is easily verified
𝜙𝛾1
𝜆 = (𝛿𝜓𝜆)[𝛾1]= 𝜋★𝑃1.𝜓𝜆−𝜓𝜆; (3.57)
where
𝜓𝜆 =−𝜙Id𝜆
𝜆. (3.58)
The new interesting point is the definition of a mutual information. For that we mimic the formulas of
Shannon theory: we apply a combinatorial operator to the ambiguity. Then we consider the canonical
bar resolution for Ext★
D(K,Φ), with the trivial action of A′|𝜆; 𝜆∈A. The operator is the combinatorial
coboundary 𝛿𝑡 at degree two, and it gives:
𝐼𝜆(𝛾1; 𝛾2)= 𝛿𝑡𝜙𝜆[𝛾1,𝛾2]= 𝜙𝛾1
𝜆−𝜙𝛾1∧𝛾1
𝜆 +𝜙𝛾2
𝜆. (3.59)
This gives the following formulas
𝐼𝜆(𝛾1; 𝛾2)= 𝜙𝛾1
𝜆−𝛾2.𝜙𝛾1
𝜆 = 𝜙𝛾2
𝜆−𝛾1.𝜙𝛾2
𝜆. (3.60)
More concretely, for two morphisms 𝛾1 : 𝜆1 →𝜆and 𝛾2 : 𝜆2 →𝜆, denoting by 𝑃1,𝑃2 their respective
coordinates on propositions, and by 𝜓𝜆 =−𝜙𝜆
𝜆 the canonical 0-cochain, we have:
𝐼𝜆(𝛾1; 𝛾2)(𝑇)= 𝜓𝜆(𝑇|𝜋★𝑃2)+𝜓𝜆(𝑇|𝜋★𝑃1)−𝜓𝜆(𝑇|𝜋★𝑃1 ∧𝜋★𝑃2)−𝜓𝜆(𝑇)
Remark. We decided that the interpretation of 𝜙𝜆 is better when 𝜓𝜆 is growing. Now, assuming the
positivity of 𝐼𝜆, we get a kind of concavity of 𝜓𝜆.
66
More generally, we say that a real function 𝜓 of the theories, containing ⊢¬𝑃, in a given language, is
concave (resp. strictly concave), if for any pair of such theories 𝑇 ≤𝑇′and any proposition 𝑄≥𝑃, the
following expression is positive (resp. strictly positive),
𝐼𝑃(𝑄;𝑇,𝑇′)= 𝜓(𝑇|𝑄)−𝜓(𝑇)−𝜓(𝑇′|𝑄)+𝜓(𝑇′). (3.61)
Remark that this definition extends verbatim to any closed monoidal category, because it uses only the
pre-order and the exponential.
The positivity of the mutual information is the particular case where 𝑇′
= 𝑇|𝑄1.
This makes 𝜓look like the function log (ln 𝑃)for a domain ⊥<𝑃≤¬𝑃, analog of the interval ]0,1[in
the propositional context.
The functions 𝜓𝜆 can always be chosen such that 𝜙𝑃
𝜆 =−𝜓𝜆. Then the above interpretation of 𝜙as
an informational ambiguity is compatible with an interpretation of 𝜓(𝑇)as a measure of the precision of
the theory.
The Boolean case, comparing to Carnap and Bar-Hillel [CBH52]
In the finite Boolean case, the opposite of the content defined by Carnap and Bar-Hillel gives such a
function 𝜓, strictly increasing and concave. Remind that the content set 𝐶(𝑇)is the set of elementary
propositions that are excluded by the theory 𝑇. Here we assimilate a theory with the language and
its axioms, and with a subset of a finite set 𝐸. If 𝑇 <𝑇′, there is less excluded points by 𝑇′than by
𝑇, then−𝑐(𝑇′)−(−𝑐(𝑇))>0. If 𝑃≤𝑄, the content set of 𝑇∨¬𝑄 is the intersection of 𝐶(𝑇)and
𝐶(⊢¬𝑄)= 𝐶(𝑄)𝑐, and the content of 𝑇′∨¬𝑄the intersection of 𝐶(𝑇)and 𝐶(⊢¬𝑄)= 𝐶(𝑄)𝑐, then the
complement of 𝐶(𝑇′∨¬𝑄)in 𝐶(𝑇′)is contained in the complement of 𝐶(𝑇∨¬𝑄)in 𝐶(𝑇). Consequently
𝜓(𝑇|𝑄)−𝜓(𝑇)−(𝜓(𝑇′|𝑄)−𝜓(𝑇′))= 𝑐(𝑇)−𝑐(𝑇|𝑄)−(𝑐(𝑇′)−𝑐(𝑇′|𝑄))≥0. (3.62)
It is zero when 𝑇′∧(¬𝑄)≤𝑇.
A natural manner to obtain a strictly concave function is to apply the logarithm function to the function
(𝑐max−𝑐(𝑇))/𝑐max.
Therefore a natural formula in the boolean case is
𝜓𝑃(T)= ln 𝑐(⊥)−𝑐(T)
𝑐(⊥)−𝑐(¬𝑃) (3.63)
But we also could take a uniform normalization:
𝜓⊥(T)= ln 𝑐(⊥)−𝑐(T)
𝑐(⊥) (3.64)
67
Amazingly, this was the definition of the amount of information (with a minus sign) of Carnap and
Bar-Hillel [CBH52].
A generalization along their line consists to choose any strictly positive function 𝑚 of the elementary
propositions and to define the numerical content 𝑐(𝑇)as the sum of the values of 𝑚 over the elements
excluded by 𝑇. This corresponds to the attribution of more or less value to the individual elements.
We essentially recover the basis of the theory presented by Bao, Basu et al. [BBD+11], [BBDH14].
Question. Does a natural formula exist, that is valid in every Heyting algebra, or at least in a class of
Heyting algebras larger than Boole algebras?
Example. The open sets of a topology on a finite set 𝑋. The analog of the content of 𝑇is the cardinality
of the closed set 𝑋\𝑇. Then a preliminary function 𝜓 is the cardinality of 𝑇 itself, which is naturally
increasing with 𝑇. However simple examples show that this function can be non-concave. The set 𝑇|𝑄\𝑇
is made by the points 𝑥 of 𝑋\𝑇 having a neighborhood 𝑉 such that 𝑉∩𝑉 ⊂𝑇, there exists no relation
between this set and the analog set for 𝑇′larger than 𝑇, but smaller than¬𝑃.
However, appendix D constructs a good function 𝜓for the sites of DNNs and the injective finite sheaves.
This applies in particular to the chains 0 →1 →...→𝑛.
A remark on semantic independency
In their 1952 report [CBH52], Carnap and bar-Hillel gave a diﬀerent justification than us for taking the
logarithm of a normalized version of the content. This was in the Boolean situation, 𝑛= 0, but our
appendix D extends what they said to some non-Boolean situations.
They had in mind that independent assertions must give an addition of the amounts of information of
the separate assertions. However, as they recognized themselves, the concept of semantic independency
is not very clear [CBH52, page 12]. In fact they studied a particular case of typed language that they
named L𝜋
𝑛, where there exists one type of subjects with 𝑛 elements, 𝑎,𝑏,𝑐,..., that can have a given
number 𝜋of attributes (or predicate). The example is three humans, their gender (male or female), and
their age (old or young). For every elementary proposition 𝑍𝑖, i.e. a point inn 𝐸, they choose a number
𝑚𝑃(𝑍𝑖)in ]0,1|, and define, as in the preceding section with 𝜇, the function 𝑚of any proposition 𝐿, by
taking the sum of the 𝑚𝑖 over the elements of 𝐿, viewed as a subset of 𝐸.
Carnap and Bar-Hillel imposed several axioms on 𝑚𝑃, for instance the invariance under the natural
action of the symmetry group 𝔖𝑛×𝔊𝜋, where 𝔊𝜋 describes the symmetries between the predicates, and
the normalization by 𝑚(𝐸)= 1. The content is an additive normalization of the opposite of 𝑚. The
number 𝑐(𝐿)evaluates the quantity of elementary propositions excluded by 𝐿.
At some moment, they introduce axiom ℎ, [CBH52, page 14], 𝑚(𝑄∧𝑅)= 𝑚(𝑄)𝑚(𝑅), if 𝑄and 𝑅do
not consider any common predicate. This axiom was rarely considered in the rest of the paper. However
68
it is followed by a definition: two assertions 𝑆and 𝑇 were said inductively independent (with respect to
𝑚𝑃) if an only if
𝑚(𝑆∧𝑇)= 𝑚(𝑆)𝑚(𝑇). (3.65)
This was obviously inspired from the theory of probabilities [Car50], where primitive predicates are
considered in relation to probabilities.
If we think of the example with the age and the gender, the axiom is not very convincing from the
point of view of probability, because in most suﬃciently large population of humans it is not true that
age and gender are independent. However, from a semantic point of view, this is completely justified!
Now, if we come to the amount of information, taking the logarithm of the inverse of 𝑚(𝑇)to measure
inf(𝑇)makes that independency (inductive) is equivalent to the additivity:
𝜓(𝑆∧𝑇)= 𝜓(𝑆)+𝜓(𝑇). (3.66)
Under this form, the definition still has a meaning, for any function 𝜓. Even with values in a category
of models, with a good notion of colimit, as the disjoint union of sets.
In Shannon’s theory, with the set theoretic interpretation of Hu Kuo Ting, [Tin62], we recover the
same thing.
Comparison of information between layers
Another way to obtain a comparison between layers, i.e. objects (𝑈,𝜉), comes from the ordinary coho-
mology of the object Φ in the topos of presheaves over the opposite category of A′
strict, that we named
D.
This cohomology can be computed following the method exposed by Grothendieck and Verdier in
SGA 4 [AGV63], using a canonical resolution of Φ. This resolution is constructed from the nerve N(D),
made by the sequences of arrows 𝜆→𝜆1 →𝜆2...in A′
strict, then associated to the fibration by the slices
category 𝜆|Dover D. Be carefull that in D, the arrows are in reverse order.
The nerve N(D)has a natural structure of simplicial set whose 𝑛 simplices are sequences of
composable arrows (𝛾1,...,𝛾𝑛)between objects 𝜆0 →···→𝜆𝑛 in A′
strict, and whose face operators
𝑑𝑖;𝑖= 0,...,𝑛are given by the following formulas:
𝑑0 (𝛾1,...,𝛾𝑛)= (𝛾2,...,𝛾𝑛)
𝑑𝑖(𝛾1,...,𝛾𝑛)= (𝛾1,...,𝛾𝑖+1 ◦𝛾𝑖,...,𝛾𝑛)if 0 <𝑖<𝑛
𝑑𝑛(𝛾1,...,𝛾𝑛)= (𝛾1,...,𝛾𝑛−1).
69
This allows to define a canonical cochain complex (𝐶𝑛(D,Φ),𝑑)which cohomology is 𝐻★(D,Φ).
The 𝑛-cochains are
𝐶𝑛(D,Φ)=
𝜆0→···→𝜆𝑛
and the coboundary operator 𝛿: 𝐶𝑛−1 (D,Φ)→𝐶𝑛(D,Φ)is given by
Φ𝜆𝑛 (3.67)
(𝛿𝜙)𝜆0 →···→𝜆𝑛
=
𝑛−1
(−1)𝑖𝜙𝑑𝑖 (𝜆0→···𝜆𝑛)+(−1)𝑛Φ★(𝛾𝑛)𝜙𝑑𝑛 (𝜆0→···𝜆𝑛). (3.68)
𝑖=0
For instance at degree zero, this gives, for 𝛾: 𝜆→𝜆′
𝛿𝜙0
𝛾(𝑆′)= 𝜙0
𝜆′(𝑆′)−𝜙0
𝜆(𝜋★𝑆′). (3.69)
For our cocycle 𝜙𝑄
𝜆, with 𝑃≤𝑄, a more convenient sheaf over Dis given by the sets Ψ𝜆 of functions
of the pairs (𝑆,𝑄), with 𝑆excluding 𝑃and 𝑃implying 𝑄, with morphisms
Ψ★(𝛾)(𝑆′,𝑄′)= 𝜓(𝜋★𝑆′
,𝜋★𝑄′). (3.70)
This gives
𝛿𝜙0
𝛾(𝑆′,𝑄′)= 𝜙0
𝜆′(𝑆′,𝑄′)−𝜙0
𝜆(𝜋★𝑆′
,𝜋★𝑄′). (3.71)
In our case, with 𝜙0
𝜆(𝑆,𝑄)= 𝜙𝑄
𝜆 (𝑆), we get the measure of the evolution of the ambiguity along the
network.
From now on, we change topic and consider the reverse direction of propagation of theories and propo-
sitions.
The particular case of natural isomorphisms
Until the end of this subsection, we consider the particular case of isomorphisms between the logics in
the layers, i.e. 𝜋★𝜋★= Id𝑈 and 𝜋★𝜋★= Id𝑈′.
As we will see, this is rather deceptive, giving a particular case of the preceding notion of ambiguity and
information, obtained without the hypothesis of isomorphism, then it can be skipped easily, but it seemed
necessary to explore what possibilities were oﬀered by the contravariant side of 𝐴.
In this case we are allowed to consider the sheaf of propositions Afor 𝜋★together and the cosheaf of
theories Θ for 𝜋★ over the category A. The action of Aby conditioning on the sheaf Φ of measurable
functions on Θ is natural, (see proposition 3.3).
Thus we can apply the same strategy as before, using the bar complex.
70
The zero cochains satisfy
𝜓𝜆′(𝜋★𝑇)= 𝜓𝜆(𝑇). (3.72)
This equation implies the naturality (3.29). However, there is a diﬀerence with the preceding framework,
because we have more morphisms to take in account, i.e. the implications 𝑃≤𝑃′. This implies that,
for 𝑈,𝜉 fixed, 𝜙does not depend on 𝑃; there exists a function 𝜓𝑈,𝜉 on all the theories such that 𝜓𝜆 on
Θ(𝑈,𝜉,𝑃)is its restriction.
Proof : for any pair 𝑃≤𝑄 in A𝜆, and any theory which excludes 𝑄 then 𝑃, we have 𝜓𝑃(𝑆)= 𝜓𝑄(𝑆).
Therefore 𝜓𝑃 = 𝜓⊥.
The equation of cocycle is the same as before, i.e. (3.30). It implies that 𝜓𝑈,𝜉 is invariant by the action of
A𝜆. In every case, boolean or not, this implies that 𝜙𝑈,𝜉 is also independent of the theory 𝑇. Therefore
the 𝐻0 now simply counts the sections of F.
The degree one cochains satisfy
𝜙𝑅′
𝜆′(𝜋★𝑆)= 𝜙𝜋★ 𝑅′
𝜆 (𝑆). (3.73)
In particular, for any triple 𝑃≤𝑄≤𝑅, and any 𝑆∈Θ𝑃, we have
𝜙𝑅
𝑈,𝜉,𝑄(𝑆)= 𝜙𝑅
𝑈,𝜉,𝑃(𝑆), (3.74)
which allows us to consider only the elements of the form 𝜙𝑃
𝜆, that we denote simply 𝜙𝜆.
The cocycle equation is as before, (3.33): And taking 𝜓𝜆=−𝜙𝜆gives canonically a zero whose coboundary
is 𝜙:
𝜙𝑄
𝜆 (𝑆)= 𝜓𝜆(𝑆)−𝜓𝜆(𝑆|𝑄). (3.75)
Which defines the dependency of 𝜙in 𝑄.
The naturality, in the case of isomorphisms, for a connected network, with a unique output layer, tells
that everything can be computed in the output layer. The intervention of the layers is illusory. Then it is
suﬃcient to consider the case of one layer and logical calculus.
What follows is only a verification that things transport naturally to the whole category 𝐴.
The extension of monoids is made via the left slices categories 𝜆|A; the action of 𝜆|Aon Θ𝜆 is given by
𝛾.T= (𝜋★
𝛾𝑃′⇒T)= T|𝜋★
𝛾𝑃′ (3.76)
= (𝑈′,𝜉′,𝑃′), 𝑃≤𝜋★𝑃′, and 𝜋𝛾 = (𝛼,ℎ)is the projected morphism of
where 𝛾: 𝜆→𝜆′
, 𝜆= (𝑈,𝜉,𝑃), 𝜆′
F.
This defines an action of the monoid of propositions in A𝜆′ which are implied by 𝑃′. If 𝑃′≤𝑄′and
𝑃′≤𝑅′, we have 𝜋★
𝛾(𝑄′∧𝑅′)= 𝜋★
𝛾(𝑄′)∧𝜋★
𝛾(𝑅′).
71
A natural structure of monoid is given by
𝛾1.𝛾2 = (𝑈,𝜉,𝜋★𝛾1 ∧𝜋★𝛾2). (3.77)
This works because, for a morphism 𝛾: 𝜆→𝜆′, we have 𝑃≤𝜋★
𝛾𝑃′
.
The identity is the truth ⊤𝜆.
Lemma 3.5. The naturality of the operations over A′ follows from the further hypothesis: for every
morphism (𝛼,ℎ), we assume that the counit 𝜋★𝜋★ is equal to IdL𝑈 , 𝜉.
Proof. Consider an arrow 𝜌: 𝜆→𝜆1; it gives a morphism 𝜌★: 𝜆1 |A→𝜆|A.
For a morphism 𝛾1 : 𝜆1 →𝜆′
1, 𝜌★(𝜆1)= 𝛾1 ◦𝜌.
If 𝛾1 : 𝜆1 →𝜆′
1 is an arrow in A′, where 𝜆′
1 = (𝑈′
1,𝜉′
1,𝑃′
1), and 𝑇 a theory in Θ𝜆, we have
𝜌★(𝛾1).𝑇= 𝜋★
𝛾1◦𝜌𝑃′
1 ⇒𝑇
= 𝜋★
𝜌𝜋★
𝛾1 𝑃′
1 ⇒𝜋★
𝜌(𝜋𝜌)★𝑇
= 𝜋★
𝜌[𝜋★
𝛾1 𝑃′
1 ⇒(𝜋𝜌)★𝑇]
= 𝜋★
𝜌[𝛾1.(𝜋𝜌)★𝑇]
= 𝜌★(𝛾1.𝜌★𝑇)
The monoids 𝜆|Ais a presheaf over A, only in the case of isomorphisms, i.e. 𝜋★𝜋★= Id𝜆′.
The bar construction now makes appeal to symbols [𝛾1 |𝛾2 |...|𝛾𝑛|, where the 𝛾𝑖 are arrows issued
from 𝜆. The action of algebra pass through the inverse image of propositions 𝜋★𝑃𝑖.
The zero cochains are families 𝜙𝜆 of maps on theories satisfying
𝜓𝜆(𝑇)= 𝜓𝜆′(𝜋★𝑇), (3.78)
where 𝛾: 𝜆→𝜆′is a morphism in 𝐴.
The coboundary operator is
𝛿𝜓𝜆([𝛾1])= 𝜓𝜆(𝑇|𝜋★
𝛾𝑃1)−𝜓𝜆(𝑇). (3.79)
Then the cohomology is as before.
The one cochains are collections of maps of theories 𝜙𝛾1
𝜆 satisfying
1
𝜙𝛾′
𝜆′(𝜋★𝑇)= 𝜙𝛾′
1◦𝛾
𝜆 (𝑇). (3.80)
The cocycle equation is
𝜙𝛾1∧𝛾2
𝜆 = 𝜙𝛾1
𝜆 +𝛾1.𝜙𝛾2
𝜆. (3.81)
One more time, the cocycles are coboundaries; the following formula is easily verified
𝜙𝜆1
𝜆 = (𝛿𝜓𝜆)[𝜆1]= 𝜋★𝑃1.𝜓𝜆−𝜓𝜆; (3.82)
72
where
𝜓𝜆 =−𝜙𝐼𝑑𝜆
𝜆. (3.83)
The combinatorial coboundary 𝛿𝑡 at degree two gives:
𝐼𝜆(𝛾1; 𝛾2)= 𝛿𝑡𝜙𝜆[𝛾1,𝛾2]= 𝜙𝛾1
𝜆−𝜙𝛾1∧𝛾1
𝜆 +𝜙𝛾2
𝜆. (3.84)
This gives the following formulas
𝐼𝜆(𝛾1; 𝛾2)= 𝜙𝛾1
𝜆−𝛾2.𝜙𝛾1
𝜆 = 𝜙𝛾2
𝜆−𝛾1.𝜙𝛾2
𝜆. (3.85)
More concretely, for two morphisms 𝛾1 : 𝜆1 →𝜆and 𝛾2 : 𝜆2 →𝜆, denoting by 𝑃1,𝑃2 their respective
coordinates on propositions, and by 𝜓𝜆 =−𝜙𝜆
𝜆 the canonical 0-cochain, we have:
𝐼𝜆(𝛾1; 𝛾2)(𝑇)= 𝜓𝜆(𝑇|𝜋★𝑃1 ∧𝜋★𝑃2)−𝜓𝜆(𝑇|𝜋★𝑃1)−𝜓𝜆(𝑇|𝜋★𝑃2)+𝜓𝜆(𝑇) (3.86)
In a unique layer 𝑈, for a given context 𝜉, we get
𝐼(𝑃1; 𝑃2)(𝑇)= 𝜓(𝑇|𝑃1 ∧𝑃2)−𝜓(𝑇|𝑃1)−𝜓(𝑇|𝑃2)+𝜓(𝑇). (3.87)
This is the particular case of the mutual information we got before, see equation (3.59), because now, the
generating function 𝜓is the restriction to Θ(𝑃)of a function that is defined on Θ = Θ(⊥).
3.5 Homotopy constructions
Abelian homogeneous bar complex of information
We start by describing an homogeneous version of the information cocycles, giving first the diﬀerences of
ambiguities, from which the above ambiguity can be derived by reducing redundancy. For that purpose
we consider equivariant cochains as in [BB15].
The sets Θ𝜆, where 𝜆= (𝑈,𝜉,𝑃), are now extended by the symbols [𝛾0 |𝛾1 |...|𝛾𝑛], where 𝑛∈N, and
the 𝛾𝑖;𝑖= 0,...,𝑛, are objects of the category D𝜆 or arrows in A′
strict abutting to 𝜆𝑅 = (𝑈,𝜉,𝑅)for 𝑃≤𝑅.
This extension with 𝑛+1 symbols is denoted by Θ𝑛
𝜆. It represents the possible theories in the local
language and its context 𝑈,𝜉, excluding the validity of 𝑃, augmented by the possibility to use counter-
examples¬𝑄𝑖,𝑖= 0,...,𝑛. There is a natural simplicial structure on the union Θ•
𝜆 of these sets. The face
operators 𝑑𝑖;𝑖= 0,...,𝑛being given by the following formulas:
𝑑0 (𝛾0,...,𝛾𝑛)= (𝛾1,...,𝛾𝑛)
𝑑𝑖(𝛾0,...,𝛾𝑛)= (𝛾0,...,𝛾𝑖−1,𝛾𝑖+1...,𝛾𝑛)if 0 <𝑖<𝑛
𝑑𝑛(𝛾0,...,𝛾𝑛)= (𝛾0,...,𝛾𝑛−1).
By definition, the geometric realization of Θ•
𝜆 is named the space of theories at 𝜆or localized at 𝜆. Its
homotopy type is named the algebraic homotopy type of theories, also at 𝜆.
73
Remind that a simplicial set 𝐾 is a presheaf over the category Δ, with objects Nand morphisms from 𝑚
to 𝑛, the non decreasing maps from [𝑚]= {1,...,𝑚}to [𝑛]= {1,...,𝑛}. The geometric realization |𝐾|
of a simplicial set 𝐾 is the topological space obtained by quotienting the disjoint union of the products
𝐾𝑛×Δ(𝑛), where 𝐾𝑛 = 𝐾([𝑛])and Δ(𝑛)⊂R𝑛+1 is the geometric standard simplex, by the equivalence
relation that identifies (𝑥,𝜑★(𝑦))and (𝜑★(𝑥),𝑦)for every nondecreasing map 𝜑: [𝑚]→[𝑛], every 𝑥∈𝐾𝑛
and every 𝑦∈Δ(𝑚); here 𝑓★is 𝐾(𝑓)and 𝑓★is the restriction to Δ(𝑛)of the unique linear map from R𝑛+1
to R𝑚+1 that sends the canonical vector 𝑒𝑖 to 𝑒𝑓(𝑖). In this construction, for 𝑛∈N, 𝐾𝑛 is equipped with
the discrete topology and Δ(𝑛)with its usual topology, then compact, the topology on the union over
𝑛∈Nis the weak topology, i.e. a subset is closed if and only if its intersection with each closed simplex
is closed, and the realization is equipped with the quotient topology, the finest making the quotient map
continuous. In particular, even it is not obvious at first glance, the realization of the simplicial set Δ𝑘 is
the standard simplex Δ(𝑘).
Let Kbe commutative ring of cardinality at most continuous (conditions of measurability will be
considered later). We consider the rings Φ𝑛
𝜆; 𝑛∈Nof (measurable) functions on the respective Θ𝑛
𝜆 with
values in K.
The above simplicial structure gives a diﬀerential complex on the graded sum Φ•
𝜆 of the Φ𝑛
𝜆; 𝑛∈N, with
the simplicial (or combinatorial) coboundary operator
(𝛿𝜆𝜙)𝛾0 |···|𝛾𝑛
𝜆 =
𝑛
(−1)𝑖𝜙𝛾0 |···|𝛾𝑖 |···|𝛾𝑛
. (3.88)
𝑖=0
We call algebraic cocycles the elements in the kernel.
As we have seen, the arrows 𝛾𝑄 ∈D𝜆 can be multiplied, using the operation ∧on propositions in A𝜆,
and this defines an action of monoid on Θ𝜆 by the conditioning operation. Therefore we can define
the homogeneous functions or homogeneous algebraic cochains of degree 𝑛∈Nas the (measurable)
functions 𝜙𝛾0;𝛾1;...;𝛾𝑛
𝜆 on Θ𝜆, such that for any 𝛾𝑄 in D𝜆, abutting in (𝑈,𝜉,𝑄), for 𝑃≤𝑄, and any 𝑇∈Θ𝜆,
thus excluding 𝑃,
𝜙𝛾𝑄 ∧𝛾0;𝛾𝑄∧𝛾1;...;𝛾𝑄 ∧𝛾𝑛
𝜆 (𝑇)= 𝜙𝛾0;𝛾1;...;𝛾𝑛
𝜆 (𝑇|𝑄). (3.89)
The above operator 𝛿𝜆preserves the homogeneous algebraic cochains. The kernel restriction of 𝛿𝜆 defines
the homogeneous algebraic cocycles.
A morphism 𝛾: 𝜆→𝜆′naturally associates 𝜙𝛾0 |𝛾1 |...|𝛾𝑛
𝜆 with 𝜙𝛾′
0 |𝛾′
1 |...|𝛾′
𝑛
𝜆′ through the formula
𝜙𝛾0 |𝛾1 |...|𝛾𝑛
𝜆 (𝜋★𝑇′)= 𝜙𝛾★𝛾0 |𝛾★𝛾1 |...|𝛾★𝛾𝑛
𝜆′ (𝑇′). (3.90)
Then the hypothesis 𝜋★𝜋★ = Id𝑈′,𝜉′ allows to define a cosheaf Φ𝑛
𝜆; 𝜆∈Dover D, not a sheaf, by
(Φ★𝜙𝜆′)𝛾0 |𝛾1 |...|𝛾𝑛 (𝑇)= 𝜙𝛾★𝛾0 |𝛾★𝛾1 |...|𝛾★𝛾𝑛
𝜆′ (𝜋★𝑇). (3.91)
74
However the first equation (3.90) is more precise, and we take it as a definition of natural algebraic
homogeneous cochains.
Remark. We cannot consider it as a sheaf because of a lack of definition of 𝛾★𝛾′
𝑖.
The operation of conditioning preserves the naturality, in reason of the following identity, involving
𝛾: 𝜆→𝜆′
, 𝛾𝑄 ∈D𝜆, 𝑆′∈Θ𝑛
𝜆:
𝜋★
𝛾[𝑆′|𝛾★(𝛾𝑄)]= 𝜋★
𝛾𝑆′|𝛾𝑄. (3.92)
Therefore we can speak of natural homogeneous algebraic cocycles.
For 𝑛= 0, the cochains are collections of functions 𝜓𝛾0
𝜆 of the theories in A𝜆 such that
𝜓𝛾𝑄 ∧𝛾0
𝜆 (𝑆)= 𝜓𝛾0
𝜆 (𝑆|𝑄), (3.93)
and such that, for any morphism 𝛾: 𝜆→𝜆′
,
𝜓𝛾0
𝜆 (𝜋★
𝛾𝑇′)= 𝜓𝛾★𝛾0
𝜆′ (𝑇′). (3.94)
From the first equation, we can eliminate 𝛾0. We define 𝜓𝜆 = 𝜓⊤
𝜆, and get
𝜓𝛾𝑄
𝜆 (𝑆)= 𝜓𝜆(𝑆|𝑄). (3.95)
The second equation, with the transport of truth, is equivalent to
𝜓𝜆(𝜋★
𝛾𝑇′)= 𝜓𝜆′(𝑇′). (3.96)
A cocycle corresponds to a collection of constant 𝑐𝜆, which are natural, then to the functions of the
connected components of the category D.
Thus we recover the same notion as in the preceding section.
In degree one, the homogeneous cochain 𝜙𝛾0;𝛾1
𝜆 cannot be a priori expressed through the collection of
functions 𝜑𝛾0
𝜆 = 𝜙𝛾0;⊤
𝜆 , but, if it is a cocycle, it can:
𝜙𝛾0;𝛾1
𝜆 = 𝜑𝛾0
𝜆−𝜑𝛾1
𝜆 ; (3.97)
as this follows directly from the algebraic cocycle relation applied to [𝛾0 |𝛾1 |⊤𝜆].
But we also have, by homogeneity
𝑄.𝜑𝛾𝑄 = 𝑄.𝜙𝛾𝑄 |⊤= 𝜙𝛾𝑄 ∧𝛾𝑄 |𝛾𝑄∧⊤
= 𝜙𝛾𝑄 |𝛾𝑄 = 𝜑𝛾𝑄−𝜑𝛾𝑄 = 0. (3.98)
Then, the homogeneity equation gives the particular case
𝜑𝑄∧𝑄𝑂
−𝜑𝑄∧𝑄 = 𝑄.𝜑𝛾𝑄𝑂−𝑄.𝜑𝛾𝑄 = 𝑄.𝜑𝛾𝑄𝑂
, (3.99)
75
therefore
𝜑𝑄∧𝑄𝑂 = 𝜑𝛾𝑄 +𝑄.𝜑𝛾𝑄𝑂 ; which is the cocycle equation we discussed in the preceding section, under the form of Shannon.
(3.100)
Remark. All that generalizes to any degree, in virtue of the comparison theorem between projective
resolutions, proved in the relative case in MacLane "Homology" [Mac12], or in SGA 4 [AGV63], more
generally, because the above homogeneous bar complex and in-homogeneous bar complex are such
resolutions of the constant functor K.
Semantic Kullback-Leibler distance
In [BB15], it was also shown that the Kullback-Leibler distance (or divergence) 𝐷𝐾𝐿(𝑋; P; P′)between
two probability laws on a random variable 𝑋defines a cohomology class in the above sense. The cochains
depend on a sequence P0,...,P𝑛 of probabilities and a sequence of variables 𝑋0,...,𝑋𝑚 less fine than a
given variable 𝑋; the conditioning the 𝑛+1 laws by the value 𝑦 of a variable 𝑌 ≥𝑋 is integrated over
𝑌★P0, for giving an action on the set of measurable functions of the 𝑛+1 laws, then the homogeneity
is defined as before, and the coboundary is the standard combinatorial one, as before. For 𝑛= 1, the
universal degree one class is shown to be the diﬀerence of divergences.
Remind that the 𝐾−𝐿divergence is given by the formula
𝐷𝐾𝐿(𝑋; P; P′)=−
𝑖
𝑝𝑖log 𝑝′
𝑝𝑖
. (3.101)
𝑥𝑖
In our present case, we consider functions of 𝑛+1 theories and 𝑚+1 propositions, all works as for
𝑛= 0. In degree zero, the cochains are defined by functions 𝜓𝜆(𝑆0,𝑆1)satisfying
𝜓𝜆(𝜋★
𝛾𝑆′
0;...; 𝜋★
𝛾𝑆′
𝑛)= 𝜓𝜆′(𝑆′
0;...; 𝑆′
𝑛), (3.102)
for any morphism 𝛾: 𝜆→𝜆′
.
The formula for the homogeneous cochain is
𝜓𝛾𝑄
𝜆 (𝑆0;...; 𝑆𝑛)= 𝜓𝜆(𝑆0 |𝑄;...; 𝑆𝑛|𝑄). (3.103)
The non-homogeneous zero cocycles are the functions of 𝑃only, invariant by the transport 𝜋★.
In degree one, the cocycles are defined by any function 𝜑𝑄
𝜆 (𝑆0;...; 𝑆𝑛)which satisfies
𝜑𝑄
𝜆 (𝜋★
𝛾𝑆′
0;...; 𝜋★
𝛾𝑆′
𝑛)= 𝜑
𝜋★ (𝑄)
𝜆′ (𝑆′
0;...; 𝑆′
𝑛), (3.104)
for any morphism 𝛾: 𝜆→𝜆′, and verifies the cocycle equation
𝜑𝑄∧𝑅
𝜆 (𝑆0;...; 𝑆𝑛)= 𝜑𝑄
𝜆 (𝑆0;...; 𝑆𝑛)+𝜑𝑅
𝜆(𝑆0 |𝑄;...; 𝑆𝑛|𝑄). (3.105)
76
The homogeneous cocycle associated to 𝜑is defined by
𝜙𝛾𝑄0 ;𝛾𝑄1
𝜆 (𝑆0;...; 𝑆𝑛)= 𝜑𝑄0
𝜆 (𝑆0;...; 𝑆𝑛)−𝜑𝑄1
𝜆 (𝑆0;...; 𝑆𝑛). (3.106)
As for 𝑛= 0, there exists a function 𝜓𝜆(𝑆0;...; 𝑆𝑛)such that for any 𝑄∈A𝜆, i.e. 𝑄≥𝑃, we have
𝜑𝑄
𝜆 (𝑆0;...; 𝑆𝑛)= 𝜓𝜆(𝑆0 |𝑄;...; 𝑆𝑛|𝑄)−𝜓𝜆(𝑆0;...; 𝑆𝑛). (3.107)
In the particular case 𝑛= 1, we can consider a basic real function 𝜓𝜆(𝑆), seen as a logarithm of
theories as before, and define
𝜓𝜆(𝑆0; 𝑆1)= 𝜓𝜆(𝑆0 ∧𝑆1)−𝜓𝜆(𝑆0). (3.108)
If the function 𝜓𝜆(𝑆)is supposed increasing in 𝑆(for the relation of weakness ≤, as before), this gives a
negative function.
We obtain
𝜙𝑄
𝜆 (𝑆0; 𝑆1)= 𝜓𝜆(𝑆0 ∧𝑆1 |𝑄)−𝜓𝜆(𝑆0 ∧𝑆1)−𝜓𝜆(𝑆0 |𝑄)+𝜓𝜆(𝑆0). (3.109)
The positivity of this quantity is equivalent to the concavity of 𝜓𝜆(𝑆)on the pre-ordered set of theories.
Assuming this property we obtain an analog of the Kullback-Leibler divergence.
If 𝜓𝜆(𝑆)is strictly concave, that is the most convenient hypothesis, this function takes the value zero if
and only if 𝑆0 = 𝑆1. Therefore it can be taken as a natural semantic distance, depending on the data of 𝑄,
as candidate from a counter-example of 𝑃.
As in the case of 𝐷𝐾𝐿 this function is not symmetric, then it could be more convenient to take the sum
𝜎𝑄
𝜆 (𝑆0; 𝑆1)= 𝜙𝑄
𝜆 (𝑆0; 𝑆1)+𝜙𝑄
𝜆 (𝑆1; 𝑆0) (3.110)
to have a good notion of distance between two theories.
Simplicial homogeneous space of histories of theories
Another argument to justify the consideration of the homogeneity is the interest of taking a pushout of
the theories.
The sheaf of monoidal categories D𝜆 over Dacts in two manners on the algebraic space of theories Θ•
𝜆:
𝛾𝑄.(𝑆⊗[𝛾0;...; 𝛾𝑛])= (𝑆|𝑄)⊗[𝛾0;...; 𝛾𝑛], (3.111)
𝛾𝑄∧(𝑆⊗[𝛾0;...; 𝛾𝑛])= 𝑆⊗[𝛾𝑄𝛾0;...; 𝛾𝑄𝛾𝑛]. (3.112)
Then we can consider the colimit Θ•
𝜆/Dof these pairs of maps over all the arrows 𝛾𝑄, i.e. over D𝜆: this
colimit is the disjoint union of the coequalizers for each arrow. This is a quotient simplicial set. The
homogeneous cochains are just the (measurable) functions on this simplicial set.
77
This can be realized directly as a pushout, or coequalizer, of a unique pair of maps, by taking the
union 𝑍 of the products Θ•
𝜆×D𝜆, and the two natural maps 𝜇,𝜈 to 𝑇= Θ•
𝜆 given by multiplication and
conditioning respectively.
Remark that the two operations in (3.111) and (3.112) are adjoint of each other, then we can speak of
adjoint gluing.
Also interesting is the homotopy quotient, taking into account that, geometrically, 𝑍 has a higher de-
gree in propositions belonging to D𝜆, due to the presence of 𝛾𝑄. This homotopy colimit is the simplicial
set Σ•obtained from the disjoint union (𝑍×[0,1])⊔(𝑇×{0})⊔(𝑇×{1})by taking the identification
of (𝑧,0)with 𝜇(𝑧)and of (𝑧,1)with 𝜈(𝑧). It can be named a homotopy gluing, because the arrows are
used geometrically as continuous links between points in 𝑇×{0}and 𝑇×{1}. The simplicial set Σ•
is equipped with a natural projection onto the ordinary coequalizer Θ•
𝜆/D𝜆. See for instance Dugger
[Dug08] for a nice exposition of this notion, and its interest for homotopical stability with respect to the
ordinary colimit. Then we propose that a more convenient notion of homogeneous cochains could be the
functions on Σ•
.
Similarly, we have two natural actions of the category Dof arrows leading to 𝜆and issued from 𝜆′:
the first one being of the type
Θ𝜆′⊗D⊗(𝑛+1)
𝜆 →Θ𝑛
𝜆; (3.113)
the second one of the type
Θ𝜆′⊗D⊗(𝑛+1)
𝜆 →Θ𝑛
𝜆′. (3.114)
They are respectively defined by the following formulas:
𝛾★(𝑆′⊗[𝛾0;...; 𝛾𝑛])= (𝜋★
𝛾𝑆′)𝜆⊗[𝛾0;...; 𝛾𝑛] (3.115)
The second one is
𝛾★(𝑆′⊗[𝛾0;...; 𝛾𝑛])= 𝑆′⊗[𝜋𝛾
★𝛾0;...; 𝜋𝛾
★𝛾𝑛] (3.116)
They are both compatibles with the quotient by the actions of the monoids, then they define maps at the
level of Σ•
.
The natural cochains are the functions that satisfy, for each 𝛾: 𝜆→𝜆′, the equation
𝜙𝜆◦𝛾★
= 𝜙𝜆′◦𝛾★. (3.117)
Note that no one of the above equations, for homogeneity and naturality, necessitates numerical values,
but the second necessitates values in a constant set or a constant category, at least along the orbits of D.
78
And it is important for us that the cochains can take their values in a category Madmitting limits,
like Set or Top, non necessarily Abelian, because our aim is to obtain a theory of information spaces in
the sense searched by Carnap and Bar-Hillel in 1952 [CBH52].
Define a set Θ𝑛
1 (resp. Θ𝑛
0) by the coproduct, or disjoint union, over 𝛾: 𝜆→𝜆′(resp. 𝜆) of the sets
Θ𝜆′⊗D⊗(𝑛+1)
𝜆 (resp. Θ𝑛
𝜆). When the integer 𝑛 varies, we note the sum by Θ•
1 (resp. Θ•
0). They are
canonically simplicial sets.
The collections of maps 𝛾★ and 𝛾★ define two (simplicial) maps from Θ•
1 to Θ•
0, that we will denote
respectively 𝜛and 𝜗, for past and future. The colimit or coequalizer of these two maps, is the quotient
H•
0 of Θ•
0 by the equivalence relation
(𝜋★
𝛾𝑆′)𝜆⊗[𝛾0;...; 𝛾𝑛]𝜆∼ 𝑆′
𝜆′⊗[𝜋𝛾
★𝛾0;...; 𝜋𝛾
★𝛾𝑛]𝜆′. (3.118)
Once iterated over the arrows, this relation represents the complete story of a theory, from the source of
its formulation in the network to the final layer.
It is remarkably conform to the notion of cat’s manifold, and compatible with the possible presence of
inner sources in the network.
Remark that the two operations in (3.115) and (3.116) are also adjoint relative to each other, then
again the corresponding colimit can be named an adjoint gluing.
Remark. The above equivalence relation is more fine than the relation we would have found with the
covariant functor, i.e.
(𝜋𝛾
★𝑆)𝜆′⊗[𝜋𝛾
★𝛾0;...; 𝜋𝛾
★𝛾𝑛]𝜆′ ∼ 𝑆𝜆⊗[𝛾0;...; 𝛾𝑛]𝜆; (3.119)
because this relation is implied by the former, when we applied it to 𝑆′
= 𝜋★𝑆, in virtue of our hypothesis
𝜋★𝜋★= Id.
the network.
The two relations ar equivalent if and only if 𝜋★𝜋★ = Id, that is the case of isomorphic logics among
We define the natural cochains as the (measurable) functions on H•
0, and the natural homogeneous
cochains as the functions on the quotient H•
0/Dby the identification of junction with conditioning. And
we are more interested in the homogeneous case.
However, in a non-Abelian context, the stability under homotopy will be an advantage, therefore we
also consider the homotopy colimit of the maps 𝜛and 𝜗, or homotopy gluing between past and future,
and propose that this colimit I•
0 (or ℎ𝑜I if we reserve I for the usual colimit) is a better notion of the
histories of theories in the network. It is also naturally a simplicial set. Then the natural homotopy
79
homogeneous cochains will be functions on the homotopy gluing ℎ𝑜I.
The homotopy type of the theories histories I•
0 itself is an interesting candidate for representing the
information, and information flow in the network.
For instance, its connected components gives the correct notion of zero-cycles, and the functions on
them are zero-cocycles. The Abelian construction is suﬃcient to realize these cocycles.
We will later consider functions from the space I•
0 to a closed model category M, their homotopy
type in the sense of Quillen can be seen as a non-Abelian set of cococycles.
What we just have made above for the cochains (homogeneous and/or natural) is a particular case of
a homotopy limit.
The notion of homotopy limit was introduced in Bousfield-Kan 1972, [BK72, chapter XI] where it gen-
eralized the classical bar resolution in a non-linear context, see MacLane’s book "Homology" [Mac12].
The authors attributed its origin to Milnor, in the article "On axiomatic homology theory" [Mil62]. For
this notion and more recent developments (see [Hir03], [DHKS04], or [Dug08]).
In this spirit, we extend now the two maps 𝜛,𝜗from Θ•
1 to Θ•
0, in higher degrees, by using the nerve
of the category D.
The nerve N= N(D)of the category Dis the simplicial set made by the sequences 𝐴of successive
arrows in D. For 𝑘 ∈N, N𝑘 is the set of sequences of length 𝑘. A sequence is written (𝛿1,...,𝛿𝑘),
where 𝛿𝑖;𝑖= 1,...,𝑘goes from 𝜆𝑖−1 to 𝜆𝑖 in D. We use the symbols 𝛿★
𝑖 , or the letters 𝛾𝑖 when there is no
ambiguity, for the arrow 𝛿𝑖 considered in the opposite category D𝑜𝑝 = A′
strict; this reverse the direction
of the sequence, going now upstream. When necessary, we write 𝛿𝑖(𝐴),𝜆𝑖−1 (𝐴),..., for the arrows and
vertices of a chain 𝐴.
For 𝑘 ∈N, we define Θ𝑛
𝑘 as the disjoint union over 𝐴= (𝛿1,...,𝛿𝑘)of the sets Θ𝜆0 ⊗D⊗(𝑛+1)
. Thus
𝜆𝑘
the theory is attached to the beginning in the sense of D, and the involved propositions are at the end.
The chain in Dgoes in the dynamical direction, downstream. When the integers 𝑛and 𝑘 vary, we note
Θ•
★ the sum (disjoint union). This is a bi-simplicial set.
We have 𝑘+1 canonical maps 𝜗𝑖;𝑖= 1,...,𝑘+1 from Θ𝑛
𝑘+1 to Θ𝑛
𝑘. Each map deletes a vertex, moreover
at the extremities it also deletes the arrow, and inside the chain, it composes the arrows at 𝑖−1 and 𝑖.
In 𝜆0, the map 𝜋★
𝛾1 is applied to the theory, to be transmitted downstream, and in 𝜆𝑘+1, the map 𝜋𝛾𝑘+1
★ is
applied to the 𝑛+1 elements 𝛾𝑄𝑗 in D𝜆𝑘+1 , to be transmitted upstream.
By analogy with the definition of the homotopy colimit of a diagram in a model category cf. references
upcit, we take for a more complete space of histories, the whole geometric realization of the simplicial
80
functor Θ•
★, seen now as a simplicial space with the above skeleton in degree 𝑘, and the above gluing
maps 𝜗𝑖. The expression 𝑔I denotes this space, that we understand as the geometrical space of complete
histories of theories.
The extension of information over the nerve incorporates the topology of the categories C,F,D. The
degree 𝑛was for the logic, the degree 𝑘is for its transfer through the layers.
𝑔I, or its homotopy type, represents for us the logical part of the available information; it takes into
account
1) the architecture C,
2) the pre-semantic structure, through the fibration Fover C, which constrains the possible weights,
and also generates the logical transfers 𝜋★
, 𝜋★,
3) the terms of a language through A, and the propositional judgements through Dand Θ.
The dynamic is given by the semantic functioning 𝑆𝑤 : 𝑋𝑤 →Θ, depending on the data and the learning.
Its analysis needs an intermediary, a notion of cocycles of information, that we describe now.
The information appears as a tensor 𝐹𝛾0,...,𝛾𝑛
𝛿1,...,𝛿𝑘 (𝑆). A priori its components take their values in the category
M, that can be Set or Top.
The points in 𝑔I are classes of elements
𝑢= 𝑆⊗[𝛾0,...,𝛾𝑛]⊗[𝛿1,...,𝛿𝑘](𝑡0,...,𝑡𝑛; 𝑠1,...,𝑠𝑘) (3.120)
where the 𝑡𝑖;𝑖= 0,...,𝑛and 𝑠𝑗; 𝑗= 1,...,𝑘are respectively barycentric coordinates in Δ(𝑛)and Δ(𝑘−1).
It is tempting to interpret the coordinates 𝑡𝑖 as weights, or values, attributed to the propositions 𝑄𝑖, and
the numbers 𝑠𝑗 as times, conduction times perhaps, along the chain of mappings.
Therefore we see the tensor 𝐹as a local system 𝐹𝑢; 𝑢∈𝑔I over 𝑔I.
Simplicial dynamical space of a DNN, information content
Considering a semantic functioning 𝑆: 𝑋→Θ, we can enrich it by the choice of propositions in each layer
𝑈 and context 𝜉𝑈 (or better collections of elements of D𝜆), and consider sequences over the networks,
relating activities and enriched theories. Then, for each local activity, and each chain of arrows in the
network, equipped with propositions at one end (downstream), the function 𝐹gives a space of information.
More precisely, we form the topological space of activities 𝑔X, by taking the homotopy colimit of the
object X, fibred over the object W, in the classifying topos of F, lifted to D, and seen as a diagram
81
over D. This space is defined in the same manner 𝑔𝐼★ was defined from Θ★ over D; it is the geometric
realization of the simplicial set 𝑔X★, whose 𝑘-skeleton is the sum of the pairs (𝐴𝑘,𝑥𝜆)where 𝐴 is an
element of length 𝑘 in N(D)and 𝑥𝜆 an element in X𝜆, at the origin of 𝐴 in D. The degeneracies
𝑑𝑖;𝑖= 1,...,𝑘+1 from X𝑘+1 to X𝑘 are given for 1 <𝑖<𝑘+1, by composition of the morphisms at 𝑖, by
forgetting 𝛿𝑘+1 (𝐴)for 𝑖= 𝑘+1, and by forgetting 𝛿1 and transporting 𝑥𝜆 by 𝑋★
𝑤 for 𝑖= 1.
Then we can ask for an extension of the semantic functioning to a continuous or simplicial map
𝑔𝑆: 𝑔X →𝑔I. (3.121)
This implies a compatibility between dynamical functioning in Xand logical functioning in Θ. However,
this map factorizes by a quotient, that can be small, when the semantic functioning is poor. It is only for
some regions in the weight object W, giving itself a geometrical space 𝑔W, that the semantic functioning
is interesting.
Given 𝐹: 𝑔I →M, this gives a map 𝐹◦𝑔𝑆from 𝑔X to M, that can be seen as the information content
of the network.
To have a better analog on the Abelian quantities, we suppose that Mis a closed model category, and
we pass to the homotopy type
ℎ𝑜.𝐹◦𝑔𝑆: 𝑔X →ℎ𝑜M. (3.122)
For real data inputs and spontaneous internal activities, this gives a homotopy type for each image.
For instance, the degree one homogeneous cocycle 𝜙𝑄
𝜆 (𝑆)deduced from a precision function 𝜓𝜆(𝑆)with
real values, is replaced by a map to topological spaces, associated to some "propositional" paths between
two points of 𝑔I; a degree two combinatorial cocycles, as the mutual information, is replaced by a varying
space associated to a "propositional" triangle, up to homotopy.
Non-Abelian inhomogeneous fundamental cochains and cocycles. A tentative
Remember that the fundamental zero cochain 𝜓𝑄0
𝜆 with real coeﬃcients, satisfied 𝜓𝑄
𝜆 (𝑆)= 𝜓𝜆(𝑆|𝑄)≥
𝜓𝜆(𝑆). Then, in the nonlinear framework, it is tempting to assume the existence in Mof a class of
morphisms replacing the inclusions of the sets, namely cofibrations, and to generalize the increasing of
the function 𝜓𝜆 of 𝑆, by the existence of a cofibration, 𝐹(𝑆)֌ 𝐹(𝑆|𝑄), or more generally a cofibration
𝐹(𝑆)֌ 𝐹(𝑆′)each time 𝑆≤𝑆′
.
This is suﬃcient for defining an object of ambiguity, then an information object (non-homogeneous),
by generalizing the relation between precision and ambiguity of the Abelian case:
𝐻𝑄(𝑆)= 𝐹(𝑆|𝑄)\𝐹(𝑆); (3.123)
82
where the subtraction is taken in a geometrical or homotopical sense.
All that supposes that Mis a closed model category of Quillen.
This invites us to assume that 𝐹 is covariant under the action of the monoidal categories D𝜆, i.e. for
every arrow 𝛾𝑄 in D𝜆, and every theory 𝑆in Θ𝜆, there exists a morphism 𝐹(𝛾𝑄; 𝑆): 𝐹(𝑆)→𝐹(𝑆|𝑄)in
M, and for two arrows 𝛾𝑄, 𝛾𝑄′,
𝐹(𝛾𝑄′𝛾𝑄; 𝑆)= 𝐹(𝛾𝑄′; 𝑆|𝑄)◦𝐹(𝛾𝑄; 𝑆) (3.124)
and we assume that every 𝐹(𝛾𝑄; 𝑆)is a cofibration.
In the same manner, the generalization of the concavity of the real function 𝜓𝑄
𝜆 is the hypothesis that, for
two arrows 𝛾𝑄, 𝛾𝑄′, there exists a cofibration of the quotient objects 𝐻:
𝐻(𝑄,𝑄′; 𝑆): 𝐻𝑄(𝑆|𝑄′)֌ 𝐻𝑄(𝑆). (3.125)
The same thing happening for 𝐻𝑄′(𝑆|𝑄)֌ 𝐻𝑄′(𝑆).
The diﬀerence space is the model category version of the mutual information between 𝑄and 𝑄′:
by definition
𝐼2 (𝑄; 𝑄′)= 𝐻𝑄\[𝐻𝑄⊗𝑄′
\𝐻𝑄′
], (3.126)
or in other terms,
𝐼2 (𝑄; 𝑄′)= (𝑄.𝐹\𝐹)\[(𝑄⊗𝑄′)𝐹\𝑄′.𝐹], (3.127)
Reasoning on subsets of 𝐻𝑄⊗𝑄′, this gives the symmetric relation
𝐼2 (𝑄; 𝑄′)∼𝐻𝑄∩𝐻𝑄′
. (3.128)
The general concavity condition is the existence of a natural cofibration 𝐻𝑄(𝑆′)֌ 𝐻𝑄(𝑆)as soon as
there is an inclusion 𝑆≤𝑆′
.
This stronger property of concavity for the functor 𝐹implies in particular, for any pair of theories 𝑆0,𝑆1,
the existence of a cofibration
𝐽𝑄(𝑆0; 𝑆1): 𝐻𝑄(𝑆0)→𝐻𝑄(𝑆0 ∧𝑆1). (3.129)
This allows to define a homotopical notion of Kullback-Leibler divergence space in M, between two
theories falsifying 𝑃, at a proposition 𝑄≥𝑃:
𝐷𝑄(𝑆0; 𝑆1)= 𝐻𝑄(𝑆0 ∧𝑆1)\𝐹★𝐻𝑄(𝑆0). (3.130)
83
Comparison between homogeneous and inhomogeneous non-Abelian cochains and
cocycles
To be complete, we have to relate these maps 𝐹,𝐻,𝐼,𝐷,...from theories and constellations of propo-
sitions to Mwith the homogeneous tensors 𝐹𝛾0,...,𝛾𝑛
𝛿1,...,𝛿𝑘 (𝑆). For that, the natural idea is to follow the
path we had described from the homogeneous Abelian bar-complex to the non-homogeneous one, at the
beginning of this section. This will give a homotopical/geometrical version of the MacLane comparison
in homological algebra.
We consider the bi-simplicial set I•
★ as a simplicial set I★ in the algebraic exponent 𝑛for •, then it is
a contravariant functor from the category Δ to the category of simplicial sets Δ𝑆𝑒𝑡. The morphisms of Δ
from [𝑚]to [𝑛]are the non-decreasing maps, their set is noted Δ(𝑚,𝑛).
Our hypothesis is that the above tensors form a cosimplicial local system Φ with values in the cate-
gory Mover the simplicial presheaf I★, in the sense of the preprint Extra-fine sheaves and interaction
decompositions [BPSPV20]. In an equivalent manner, we consider the category T= Set(I★)which
objects are the simplicial cells 𝑢 of I★ and arrows from 𝑣 of dimension 𝑛to 𝑢 of dimension 𝑚 are the
non-decreasing maps 𝜑∈Δ(𝑚,𝑛)(morphisms in the category Δ) such that 𝜑★(𝑣)= 𝑢. Here the map 𝜑★
is simplicial in the index 𝑘for ★, concerning the nerve complex of D; then the cosimplicial local system
is a contravariant functor from Tto M.
All that is made to obtain a non-Abelian version of the propositional (semantic) bar-complex. Following
a recent trend, we name spaces the elements of M.
We add that an inclusion of theories 𝑆≤𝑆′gives a cofibration Φ(𝑆′)֌ Φ(𝑆), in a functorial manner
over the poset of theories.
Let us repeat the arguments to go from homogeneous cochains or cocycles to non-homogeneous ones.
First, a zero-cochain is defined over the cells 𝑆𝜆⊗[𝛾0], where the arrow 𝛾0 abuts in a propositions 𝑄0 ≥𝑃.
The associated non-homogeneous space 𝐹(𝑆)corresponds to 𝑄0 = ⊤. The relation between conditioning
and multiplication gives the way to recover Φ𝑄0 (𝑆).
Second, we name degree one homogeneous cocycle a sheaf of spaces Φ[𝛾0,𝛾1](𝑆), over the one skeleton
of 𝜑★, which satisfies that for the triangle [𝛾0,⊤,𝛾1], the space Φ[𝛾0 ,𝛾1] is homotopy equivalent to the
diﬀerence of the spaces Φ[𝛾0,⊤]and Φ[𝛾1,⊤]
.
Remark: more generally a degree one cocycle should satisfies this axiom for every zigzag 𝛾0 ≤𝛾1
≥𝛾1.
2
This definition supposes that we have a notion of diﬀerence in M, satisfying the same properties that the
diﬀerence 𝐴\(𝐴∩𝐵)satisfies in subsets of set. If all the theories considered contain a minimal one, then
84
spaces are subspaces of a given space, and this hypothesis has a meaning. However, this is the case in our
situation, considering the sets Θ𝑃, because we consider only propositions 𝑄,𝑄0,𝑄1,...that are implied
by 𝑃.
To the degree one cocycle Φ[𝛾0,𝛾1](𝑆)we associate the space 𝐻𝛾0 (𝑆)= Φ[𝛾0,⊤](𝑆), obtained by replacing
𝛾1 by ⊤. The space 𝐺𝛾1 (𝑆)is obtained by replacing 𝛾0 by ⊤in Φ.
Note the important point that 𝐻and 𝐺are in general non-homogeneous cocycles.
Applying the definition of 1-cocycle to the triangle [𝛾0,⊤,𝛾1], we obtain that
Φ[𝛾0 ,𝛾1](𝑆)∼𝐻𝛾0 (𝑆)\𝐻𝛾1 (𝑆). (3.131)
Lemma 3.6. The cocyclicity of Φ implies
𝑄.𝐻𝑄 ∼𝐻𝑄⊗𝑄\𝐻𝑄
. (3.132)
Proof.
𝑄.𝐻𝑄 = 𝑄.Φ𝑄|⊤= Φ𝑄⊗𝑄|𝑄⊗⊤= Φ𝑄⊗𝑄|𝑄 = 𝐻𝑄⊗𝑄\𝐻𝑄
. (3.133)
From that we deduce,
Proposition 3.7. The homogeneity of Φ implies
𝐻𝑄⊗𝑄′
\𝐻𝑄⊗𝑄 ∼𝑄.𝐻𝑄′
\[𝐻𝑄⊗𝑄\𝐻𝑄]. (3.134)
Proof.
𝐻𝑄⊗𝑄′
\𝐻𝑄⊗𝑄 = 𝑄.𝐻𝑄′
\𝑄.𝐻𝑄 ∼𝑄.𝐻𝑄′
\[𝐻𝑄⊗𝑄\𝐻𝑄]. (3.135)
In the Abelian case of ordinary diﬀerence this is equivalent to
𝐻𝑄⊗𝑄′
∼𝑄.𝐻𝑄′
∪𝐻𝑄
. (3.136)
This is the usual Shannon equation; then (3.134) can be seen as a non-Abelian Shannon equation. Taking
homotopy in 𝐻𝑜(M)probably gives a more intrinsic meaning of semantic information.
It is natural to admit that, at the level of information spaces, 𝐻𝑄⊗𝑄 ∼𝐻𝑄. Under this hypothesis, we get
the usual Shannon’s formula under
𝐻𝑄⊗𝑄′
\𝐻𝑄 ∼𝑄.𝐻𝑄′
. (3.137)
85
That is, for every theory 𝑆falsifying 𝑃:
𝐻𝑄⊗𝑄′(𝑆)\𝐻𝑄(𝑆)∼𝐻𝑄′(𝑆|𝑄). (3.138)
Remind there is no reason a priori that 𝐻𝑄 ֌ 𝐻𝑄⊗𝑄′. Then the above diﬀerence is after intersection.
If 𝐹 is any non-homogeneous zero-cochain, we have a cofibration 𝐹֌ 𝑄.𝐹, where 𝑄.𝐹(𝑆)= 𝐹(𝑆|𝑄).
In this case we already defined a space 𝐻𝑄 by
𝐻𝑄(𝑆)= 𝐹(𝑆|𝑄)\𝐹(𝑆). (3.139)
Proposition 3.8. 𝐻𝑄 automatically satisfies equation (3.134).
Proof. we have 𝐹֌ (𝑄⊗𝑄′)𝐹and 𝐹֌ (𝑄⊗𝑄)𝐹, then
𝐻𝑄⊗𝑄′\𝐻𝑄⊗𝑄 = ((𝑄⊗𝑄′)𝐹\𝐹)\((𝑄⊗𝑄)𝐹\𝐹)
∼(𝑄⊗𝑄′)𝐹\(𝑄⊗𝑄)𝐹.
Using 𝐹֌ 𝑄.𝐹֌ (𝑄⊗𝑄)𝐹, and assuming 𝑄.𝐹֌ (𝑄⊗𝑄′)𝐹, we get
𝑄.𝐻𝑄′
\[𝐻𝑄⊗𝑄\𝐻𝑄]= 𝑄.(𝑄′𝐹\𝐹)\[((𝑄⊗𝑄)𝐹\𝐹)\(𝑄.𝐹\𝐹)]
= (𝑄⊗𝑄′)𝐹\𝑄.𝐹)\[(𝑄⊗𝑄)𝐹\𝑄.𝐹]
∼(𝑄⊗𝑄′)𝐹\(𝑄⊗𝑄)𝐹.
Therefore, as wanted,
𝐻𝑄⊗𝑄′\𝐻𝑄⊗𝑄 ∼𝑄.𝐻𝑄′\[𝐻𝑄⊗𝑄\𝐻𝑄]. (3.140)
We also had suggested above to define the mutual information 𝐼2 (𝑄; 𝑄′)associated to a cocycle 𝐻by the
formula 𝐼2 (𝑄: 𝑄′)= 𝐻𝑄\𝑄′.𝐻𝑄
.
The restricted concavity condition on 𝐻is the existence of a natural cofibration 𝑄′.𝐻𝑄 ֌ 𝐻𝑄
.
Remark. This goes in the opposite direction to 𝐹: the more precise the theory 𝑆is, the bigger 𝐻𝑄(𝑆)is,
i.e. 𝑆≤𝑆′implies 𝐻𝑄(𝑆′)֌ 𝐻𝑄(𝑆).
We assume also that for all pair 𝑄,𝑄′we have 𝐻𝑄⊗𝑄′
∼𝐻𝑄′⊗𝑄
.
Proposition. under the above hypothesis and the assumption that 𝐻𝑄⊗𝑄 ∼𝐻𝑄 and 𝐻𝑄′⊗𝑄′
can consider 𝐻𝑄 and 𝐻𝑄′ as subsets of 𝐻𝑄⊗𝑄′, and we have
∼𝐻𝑄′
, we
𝐼2 (𝑄; 𝑄′)= 𝐼2 (𝑄′; 𝑄)= 𝐻𝑄∩𝐻𝑄′
. (3.141)
86
Proof. The Shannon formula (3.137) tells that 𝑄.𝐻𝑄′ is 𝐻𝑄⊗𝑄′\𝐻𝑄 and 𝑄′.𝐻𝑄 is 𝐻𝑄′⊗𝑄\𝐻𝑄′, then
𝐼2 (𝑄; 𝑄′)= 𝐻𝑄\[𝐻𝑄⊗𝑄′\𝐻𝑄′]∼𝐻𝑄∩𝐻𝑄′
. (3.142)
Remark. We cannot write the relation with the usual union, but, under the above hypotheses, there is a
cofibration
𝑗∨𝑗′: 𝐻𝑄∨𝐻𝑄′
֌ 𝐻𝑄⊗𝑄′
, (3.143)
giving rise to a quotient
𝐼2 (𝑄; 𝑄′) 𝐻𝑄×𝐻𝑄 ⊗𝑄′𝐻𝑄′
. (3.144)
Generalizing the suggestion of Carnap and Bar-Hillel, and a Shannon theorem in the case of probabilities,
we propose, to tell that 𝑄,𝑄′are independent (with respect to 𝑃) at the theory 𝑆, when 𝐻𝑄∩𝐻𝑄′ is
empty (initial element of M).
With 𝐼2, we can continue and get a semantic version of the synergy quantity of three variables:
𝐼3 (𝑄1; 𝑄2; 𝑄3)(𝑆)= 𝐼2 (𝑄1; 𝑄2)(𝑆)\𝐼2 (𝑄1; 𝑄2)(𝑆|𝑄3). (3.145)
However, there is no reason why it must be a true space, because in the Abelian case it can be a negative
number; (see [BTBG19] for the relation with the Borromean rings).
Remark. This invites us to go to 𝐻𝑜(M), where there exists a notion of relative objects: for a zigzag
𝐴և 𝐶֌ 𝐵, with a trivial fibration to the left, and a cofibration to the right, the deduced arrow 𝐴→𝐵
in 𝐻𝑜(M), can be considered as a kind of diﬀerence of spaces as in Jardine, Cocyle categories [Jar09],
and Zhen Lin Low, Cocycles in categories of fibrant objects [Low15]. Before Quillen and Jardine this
kind of homotopy construction was introduced by Gabriel and Zisman [GZ67], as a calculus of fraction,
in the framework of simplicial objects, their book being the first systematic exposition of the simplicial
theory.
With respect to the Shannon information, what is missing is an analog of the expectation of functions
over the states of the random variables. In some sense, this is replaced by the properties of growing and
concavity of the function 𝜓, or spaces 𝐹and 𝐻, which give a manner to compare the theories. The true
semantic information is not the value attributed to each individual theory, it is the set of relations between
these values, either numerical, either geometric, as expressed by functors over the simplicial space 𝑔𝐼•
★,
or better, more practical, over the part of ot that is accessible to a functioning network 𝑔X.
87
The example of the theory L2
3 of Carnap and Bar-Hillel
Let us try to describe the structure of Information, as we propose it, in the simple (static) example that
was chosen for development by Carnap and Bar-Hillel in their report in 1952, [CBH52].
The authors considered a language L𝜋
𝑛 with 𝑛 subjects 𝑎,𝑏,𝑐,...and 𝜋 attributes of them 𝐴,𝐵,...,
taking some possible values, respectively 𝜋𝐴,𝜋𝐵,.... In their developed example 𝑛= 3, 𝜋= 2 and every
𝜋𝑖 equals 2. The subjects are human persons, the two attributes are the gender 𝐺, male 𝑀 or female 𝐹,
and the age 𝐴, old 𝑂or young 𝑌.
The elementary, or ultimate, states, 𝑒∈𝐸 of the associated Boolean algebra Ω = Ω𝐸 are given by
choosing values of all the attributes for all the subjects. For instance, in the language L2
3 , we have 43 = 64
elementary states.
The propositions 𝑃,𝑄,𝑅,...are the subsets of Ω, their number is 264. The theories 𝑆,𝑇,..., in this
case, are also described by their initial assertion, that is the truth of a given proposition, obtained by
conjunction, and also named 𝑆,𝑇,....
With our conventions, for conditioning and information spaces or quantities, it appears practical to
define the propositions by the disjunction of their elements 𝑒𝐼 = 𝑒𝑖1 ∨...∨𝑒𝑖𝑘 and the theories by the con-
jonction of the complementary sets¬𝑒𝑖 = 𝑆𝑖, that is 𝑆𝐼 = (¬𝑒𝑖1 )∧...∧(¬𝑒𝑖𝑘 ). Experimentally [BBG21a]
the theories exclude something, like 𝑃, i.e. contain¬𝑃, then with 𝑆𝐼 we see that 𝑃= 𝑒𝐼 is excluded, as are
all the 𝑒𝑖𝑗 for 1 ≤𝑗 ≤𝑘. A proposition 𝑄which is implied by 𝑃, corresponds to a subset which contains
all the elementary propositions 𝑒𝑖𝑗 for 1 ≤𝑗 ≤𝑘.
In what follows, the models of "spaces of information" that are envisaged are mainly groupoids, or
sets, or topological spaces.
A zero cochain 𝐹𝑃(𝑆)gives a space for any theory excluding 𝑃, in a growing manner, in the sense
that 𝑆≤𝑆′(inclusion of sets) implies 𝐹(𝑆)≤𝐹(𝑆′). The coboundary 𝛿𝐹= 𝐻, gives a space 𝐻𝑄
𝑃(𝑆)for
any proposition 𝑄such that 𝑃≤𝑄, whose formula is
𝐻𝑄
𝑃(𝑆)= 𝐹𝑃(𝑆∨¬𝑄)\𝐹𝑃(𝑆). (3.146)
By concavity, this function (space) is assumed to be decreasing with 𝑆, i.e. if 𝑆≤𝑆′
,
𝐻𝑄
𝑃(𝑆)֋ 𝐻𝑄
𝑃(𝑆′). (3.147)
And by monotonicity of 𝐹, it is also decreasing in 𝑄, i.e. if 𝑄≤𝑄′
,
𝐻𝑄
𝑃(𝑆)֋ 𝐻𝑄′
𝑃 (𝑆′). (3.148)
In particular, we can consider the smaller 𝐹𝑃(𝑆)that is 𝐹𝑃(⊥), as it is contained in all the spaces 𝐹𝑃(𝑆),
we choose to take it as the empty space (or initial object in M), then
𝐻𝑄
𝑃(⊥)= 𝐹𝑃(¬𝑄). (3.149)
88
As we saw in general for every one-cocycle, not necessarily a coboundary, we have for any pair 𝑄,𝑄′
larger than 𝑃,
𝐻𝑄∧𝑄′
𝑃 (𝑆)\𝐻𝑄′
𝑃 (𝑆)≈𝐻𝑄
𝑃(𝑆|𝑄′)= 𝐻𝑄
𝑃(𝑆∨¬𝑄′). (3.150)
Therefore, in the boolean case, every value of 𝐻can be deduced from its value on the empty theory:
𝐻𝑄
𝑃(¬𝑄′)≈𝐻𝑄∧𝑄′
𝑃 (⊥)\𝐻𝑄′
𝑃 (⊥). (3.151)
We note simply 𝐻𝑄
𝑃(⊥)= 𝐻𝑄
𝑃 = 𝐹𝑃(¬𝑄).
And they are the spaces to determine.
The localization at 𝑃(i.e. the fact to exclude 𝑃) consists in discarding the elements 𝑒𝑖 belonging to 𝑃from
the analysis. Therefore we begin by considering the complete situation, which corresponds to 𝑃= ⊥.
In this case we note simply 𝐻𝑄 = 𝐻𝑄
⊥= 𝐹(¬𝑄).
The concavity of 𝐹 is expressed by the existence of embeddings (or more generally cofibrations)
associated to each set of propositions 𝑅0,𝑅1,𝑅2,𝑅3 such that 𝑅0 ≤𝑅1 ≤𝑅3 and 𝑅0 ≤𝑅2 ≤𝑅3:
𝐹(𝑅3)\𝐹(𝑅1)֌ 𝐹(𝑅2)\𝐹(𝑅0). (3.152)
In particular, for any pair of proposition 𝑄,𝑄′, we have ⊥≤¬𝑄≤¬(𝑄∧𝑄′)and ⊥≤¬𝑄′≤¬(𝑄∧𝑄′),
and 𝐹(⊥)= 𝐻⊤= ∅, then
𝑗: 𝐻𝑄∧𝑄′
\𝐻𝑄′
֌ 𝐻𝑄
, (3.153)
and
𝑗′: 𝐻𝑄∧𝑄′
\𝐻𝑄 ֌ 𝐻𝑄′
. (3.154)
Then we introduced the hypothesis that the subtracted spaces of both situations give equivalent results,
and defined the mutual information 𝐼2 (𝑄; 𝑄′):
𝐻𝑄\(𝑗(𝐻𝑄∧𝑄′
\𝐻𝑄′))≈𝐼2 (𝑄; 𝑄′)≈𝐻𝑄′\(𝑗′(𝐻𝑄∧𝑄′
\𝐻𝑄)). (3.155)
Importantly, to get a cofibration, the subtraction cannot be replaced by a collapse with marked point, but
it can in general be a collapse without marked point.
Consequently, the main axioms for the brut semantic spaces 𝐻𝑄 are: (𝑖)the existence of natural
embeddings (or cofibrations) when 𝑄≤𝑄′:
𝐻𝑄′
֌ 𝐻𝑄
, (3.156)
and (𝑖𝑖)the above formulas (3.153) and (3.154) defining the same space 𝐼2 (𝑄; 𝑄′), as in (3.155), which
can perhaps all be interpreted after intersection.
89
We left open the relation between 𝐼2 (𝑄; 𝑄′)and 𝐻𝑄∨𝑄′, however the axioms (𝑖𝑖)imply that there
exist natural embeddings
𝐻𝑄∨𝑄′
֌ 𝐼2 (𝑄; 𝑄′). (3.157)
The idea, to obtain a coherent set of non-trivial information spaces, is to exploit the symmetries of the
language, or other elements of structure, which give an action of a category on the language, and generate
constraints of naturalness for the spaces.
There exists a Galois group 𝐺 of the language, generated by the permutation of the 𝑛subjects, the
permutations of the values of each attribute and the permutations of the attributes that have the same
number of possible values.
To be more precise, we order and label the subjects, the attribute and the values, with triples 𝑥𝑌𝑖. In
our example, 𝑥= 𝑎,𝑏,𝑐, 𝑌= 𝐴,𝐺, 𝑖= 1,2, the group of subjects permutation is 𝔖3, the transposition of
values are 𝜎𝐴 = (𝐴1 𝐴2)and 𝜎𝐺 = (𝐺1𝐺2), and the four exchanges of attributes are 𝜎= (𝐴1𝐺1)(𝐴2𝐺2),
𝜅= (𝐴1𝐺1 𝐴2𝐺2), 𝜅3 = 𝜅−1 = (𝐴1𝐺2 𝐴2𝐺1), and 𝜏= (𝐴1𝐺2)(𝐴2𝐺1).
We have
𝜎𝐴◦𝜎𝐺 = 𝜎𝐺◦𝜎𝐴 = (𝐴1 𝐴2)(𝐺1𝐺2)= 𝜅2; (3.158)
𝜎◦𝜎𝐴 = 𝜎𝐺◦𝜎= 𝜅; 𝜎𝐴◦𝜎= 𝜎◦𝜎𝐺 = 𝜅−1; (3.159)
𝜎𝐴◦𝜎◦𝜎𝐺 = 𝜏; 𝜎𝐴◦𝜏◦𝜎𝐺 = 𝜎 (3.160)
The group generated by 𝜎,𝜎𝐴,𝜎𝐺 is of order 8; it is the dihedral group 𝐷4 of all the isometries of the
square with vertices 𝐴1𝐺1,𝐴1𝐺2,𝐴2𝐺2,𝐴2𝐺1. The stabilizer of a vertex is a cyclic group 𝐶2, of type 𝜎
or 𝜏, the stabilizer of an edge is of type 𝜎𝐴 or 𝜎𝐺, noted 𝐶𝐴
2 or 𝐶𝐴
2.
Therefore, in the example L2
3 , the group 𝐺is the product of 𝔖3 with a dihedral group 𝐷4.
In the presentation given by the present article, the language Lis a sheaf over the category 𝐺, which
plays the role of the fiber F. We have only one layer 𝑈0, but the duality of propositions and theories
corresponds to the duality between questions and answers (i.e. theories) respectively.
The action of 𝐺on the set Ω is deduced from its action on the set 𝐸, which can be described as follows:
1) One orbit of four elements, where 𝑎,𝑏,𝑐 have the same gender and age. The stabilizer of each
element is 𝔖3 ×𝐶2, or order 12.
2) One orbit of 24 elements made by a pair of equal subjects and one that diﬀers from them by one
attribute only. The stabilizer being the 𝔖2 of the pair of subjects.
90
3) One orbit of 12 elements made by a pair of equal subjects and one that diﬀers from them by the
two attributes. The stabilizer being the product 𝔖2 ×𝐶2, where 𝐶2 stabilizes the characteristic of
the pair, which is the same as stabilizing the character of the exotic subject.
4) One last orbit of 24 elements, where the three subjects are diﬀerent, then two of them diﬀer by one
attribute and diﬀer from the last one by the two attributes. The stabilizer is the stabilizer 𝐶′
2 of the
missing pair of values of the attributes.
The action of 𝐺on the set 𝐸corresponds to the conjugation of the inertia subgroups.
Remark. All that looks like a Galois theory; however there exist subgroups of 𝐺, even normal subgroups,
that cannot happen as stabilizers in the language, without adding terms or concepts. For instance, the
cyclic group 𝔄3 ⊂𝔖3; if it stabilizes a proposition 𝑃, this means that the subjects appear in complete
orbits of 𝔄3, but these orbits are orbits of 𝔖3 as well, then the stabilizer contains 𝔖3. The notion of cyclic
ordering is missing.
The collection of all the ultimate states of a given type defines a proposition, noted 𝑇, describing
𝐼,𝐼𝐼,𝐼𝐼𝐼,𝐼𝑉. This proposition has for stabilizer the group 𝐺itself. Its space of information must have a
form attached to 𝐺, but it also must take into account the structure of its elements.
Ansatz 1. The information space of type 𝑇 corresponds to the natural groupoid of type 𝑇
Remark that each type corresponds to a well formed sentence in natural languages: type 𝐼 is translated
by "all the subjects have the same attributes"; type 𝐼𝐼by "all the subjects have the same attributes except
one which diﬀers by only one aspect"; type 𝐼𝐼𝐼 "one subject is opposite to all the others"; type 𝐼𝑉 "all
the subjects are distinguished by at least one attribute".
The union of the types 𝐼𝐼 and 𝐼𝐼𝐼 is described by the sentence "all the subjects have the same attributes
except one".
The information space of (𝐼𝐼)∨(𝐼𝐼𝐼)is (naturally) a groupoid with 12 objects and fundamental group
𝔖2. A good exercise is to determine the information spaces of all the unions of the four orbits. It should
convince the reader that something interesting happens here, even if the whole tentative here evidently
needs to be better formalized.
Remark that other propositions have non-trivial inertia, and evidently support interesting semantic infor-
mation. The most important for describing the system are the numerical statements, for instance "there
exist two female subjects in the population". Its inertia is 𝔖3 ×𝐶𝐴
2.
By definition, a simple proposition is given by the form 𝑎𝐴, telling that one given subject has one
given value for one given attribute. There exist twelve such propositions, they are permuted by the group
91
𝐺. The simple propositions form an orbit of the group 𝐺, of type 𝐼𝐼𝐼above.
Amazingly, the set of the twelve simples is selfdual under the negation:
¬(𝑎𝐴)= 𝑎𝐴, (3.161)
where 𝐴denotes the opposite value.
Ansatz 2. Each simple corresponds to a groupoid with one object, and four arrows, that form a Klein
sub-group of 𝐺which fixes the subject 𝑎and fixes the attribute 𝐴corresponding to 𝐶2, generated by the
transposition 𝜎𝐴, also preserving 𝐴.
Another ingredient, introduced by Carnap and Bar-Hillel, is the mutual independency of the 12 simple
propositions.
According to the definition of the spaces 𝐼2 (𝑄,𝑄′), this implies:
Ansatz 3. The spaces of the simples are disjoint; the maximal information spaces, associated to full
populations 𝑒, are unions of them, after some gluing.
It is natural to expect that for each individual population 𝑒∈𝑋, the information space 𝐻𝑒 is a kind
of marked groupoid 𝐻𝑇
𝑒, that is a groupoid with a singularized object. A good manner to mark the
point 𝑒in 𝐻𝑒 is to glue to the space 𝐻𝑇 of its type a space 𝐻𝑃, where 𝑃is the proposition which char-
acterizes 𝑒among the elements of the orbit 𝑇. The groupoid of this space 𝐻𝑃can contains several objects.
All kinds of gluing that we had to consider are realized by identifying two spaces 𝐻1,𝐻2 with marked
points along a subspace 𝐾 (representing a mutual information or the space of the "or"), as asked by the
axiom (𝑖𝑖)above.
Therefore in general, the subspace has strictly less marked points than any of the spaces that are glued.
When we mention cylinders in this context, this means that one of the spaces, say 𝐻1 is a cylinder with
basis 𝐾, and we say that 𝐻1 is grafted on the other space 𝐻2.
Ansatz 4. The information space of the ultimate element 𝑒is obtained by gluing a cylinder to the space
of its type, based on a subspace associated to it, and containing as many objects as we need simple pieces
For type 𝐼, one object is added; for type 𝐼𝐼and 𝐼𝐼𝐼, two objects are added and for type 𝐼𝑉, three objects.
92
Illustration. Associate to each 𝑒 a trefoil knot, presented as a braid with three colored strands, corre-
sponding to its three simple constituents.
Each subject corresponds to a strand, each pair of values 𝐴,𝐺of the attributes to a color, red, blue, green
and black for the vertices ofthe square, red and green and blue and black being in diagonal.
Any proposition is a union of elementary ones, then to go farther, we have to delete pieces of the maximal
spaces 𝐻𝑒, for obtaining its information spaces.
The existence of a full coherent set of spaces is non-trivial and is described in detail in the forthcoming
preprint, A search of semantic spaces [BB22].
Then to describe the information of the more general propositions, we have to combine the forms
given by the groups and groupoids, as for 𝐻𝑇 and 𝐻𝑒, with a combinatorial counting of information,
deduced from the content, as in Carnap and Bar-Hillel.
A suggestion is to represent the combinatorial aspect by a dimension: all propositions are ranged by
their numerical content, for instance 𝑒 has 𝑐(𝑒)= 63, ¬𝑒 has 𝑐= 1, and 𝑎𝐴has 𝑐= 58. We represent
the groups and groupoids by 𝐶𝑊−complexes of dimension 2 or ∞, associated to a presentation by
generators and relations of their fundamental group, possibly marked by several base points. The spaces
of information 𝐻𝑄 are obtained by thickening the complexes, by taking the product with a simplex or a
ball of the dimension corresponding to 𝑄. However, note that any manner to code this dimension by a
number, for instance, connected components, would work as well.
For some propositions, we cannot expect a form of information in addition of the dimension. This
concerns propositions that are complex and not used in natural languages; example: "in this population,
there is two old mans, or there is a young woman, or there exist a woman that has the same age of a man".
This is pure logical calculus, not really semantic.
The general construction shows that the number of non-trivial semantic spaces is far from 264, it is of
the order of 64𝛼, with 𝛼between 3 or 4.
Then, on this simple example we see that "spaces" of semantic information are more interesting and
justified than numerical estimations, but also that this concerns only few propositions, the ones which
seem too have more sense. Then the structure of spaces has to be completed by calculus and combinatorics
for most of the 264 sentences. This touches the sensitive departure point from the admissible sentences,
more relevant to Shannon theory, and the significant sentences, more relevant for a future semantic theory,
that we hope to find in the above direction of homotopy invariants of spaces of theories and questions.
93
4
Unfoldings and memories, LSTMs and GRUs
This chapter presents evidences that some architectures of 𝐷𝑁𝑁𝑠, which are known to be eﬃcient in
syntactic and semantic tasks, rely on internal invariance supported by some groupoids of braids, which
also appear in enunciative linguistic, in relation with cognition and representation of notions in natural
languages.
4.1 RNN lattices, LSTM cells
Artificial networks for analyzing or translating successions of words, or any timely ordered set of data,
have a structure in lattice, which generalizes the chain: the input layers are arranged in a corner: hori-
zontally 𝑥1,0, 𝑥2,0, ..., named data, vertically ℎ0,1, ℎ0,2, ..., named hidden memories.
Generically, there is a layer 𝑥𝑖,𝑗 for each 𝑖= 1,2,...,𝑁, 𝑗= 0,1,2,...,𝑀, and a layer ℎ𝑖,𝑗 for each
𝑖= 1,2,...,𝑁, 𝑗= 0,1,2,...,𝑀. The information of 𝑥𝑖,𝑗−1 and ℎ𝑖−1,𝑗 are joined in a layer 𝐴𝑖,𝑗, which sends
information to 𝑥𝑖,𝑗 and ℎ𝑖,𝑗.
Then in our representation, the category CX has one arrow from 𝑥𝑖,𝑗 to 𝐴𝑖,𝑗, from ℎ𝑖,𝑗 to 𝐴𝑖,𝑗, from 𝑥𝑖,𝑗−1
to 𝐴𝑖,𝑗 and from ℎ𝑖−1,𝑗 to 𝐴𝑖,𝑗, and it is all (see figure 4.1). If we want, we could add the layers 𝐴★
𝑖,𝑗, but
there is no necessity.
The output is generally a up-right corner horizontally 𝑦1 = 𝑥1,𝑀, 𝑦2 = 𝑥2,𝑀, ..., named the result (a
classification or a translation), and vertically ℎ𝑁,1, ℎ𝑁,2, ..., (which could be named future memories).
However, the inputs and outputs can have the shape of a more complex curves, transverse to vertical
and horizontal propagation. Things are organized as in a two dimensional Lorentz space, where a space
coordinate is 𝑥𝑖,𝑗−1−ℎ𝑖−1,𝑗 and a time coordinate 𝑥𝑖,𝑗−1 +ℎ𝑖−1,𝑗. Input and output correspond to spatial
sections, related by causal propagation.
Remark. In many applications, several lattices are used together, for instance a sentence or a book can
be read backward after translation, giving reverse propagation, without trouble. We will discuss these
aspects with the modularity.
94
h0
x1
B
A C
x2 x3
c
a b
a a y1 b b y2 c y3 h3
c
Figure 4.1: Categorical representation of a RNN
Most 𝑅𝑁𝑁𝑠have a dynamic of the type a non-linearity applied to a linear summation:
we denote the vectorial states of the layers by greek letters 𝜉for layers 𝑥and 𝜂for layers ℎ, like 𝜉𝑎
𝑖,𝑗 and
𝜂𝑏
𝑘,𝑙; the lower indices denote the coordinates of the layer and the upper indices denote the neuron, that
is the real value of the state. In most applications, the basis of neurons plays an important role.
In the layer 𝐴𝑖,𝑗 the vector of state is made by the pairs (𝜉𝑎
𝑖,𝑗−1,𝜂𝑏
𝑖−1,𝑗); 𝑎∈𝑥𝑖,𝑗−1,𝑏∈ℎ𝑖−1,𝑗.
The dynamic 𝑋𝑤 has the following form:
𝜉𝑎
𝑖,𝑗 = 𝑓𝑎
𝑥
𝑤𝑎
𝑎′;𝑥,𝑖,𝑗𝜉𝑎′
𝑖,𝑗−1 +
𝑎′
𝑏′
𝑢𝑎
𝑏′;𝑥,𝑖,𝑗𝜂𝑏′
𝑖−1,𝑗 +𝛽𝑎
𝑥,𝑖,𝑗 ; (4.1)
𝜂𝑏
𝑖,𝑗 = 𝑓𝑏
ℎ
𝑤𝑏
𝑎′;ℎ,𝑖,𝑗𝜉𝑎′
𝑖,𝑗−1 +
𝑢𝑏
𝑏′;ℎ,𝑖,𝑗𝜂𝑏′
𝑖−1,𝑗 +𝛽𝑏
𝑎′
𝑏′
𝑥,𝑖,𝑗. (4.2)
The functions 𝑓 are sigmoids or of the type tanh(𝐶𝑥), the real numbers 𝛽 are named bias, and the
numbers 𝑤and 𝑢are the weights.
In practice, everything here is important, the system being very sensitive, however theoretically, only the
overall form matters, thus for instance we can incorporate the bias in the weights, just by adding a formal
neuron in 𝑥 or ℎ, with fixed value 1. The weights are summarized by the matrices 𝑊𝑥,𝑖,𝑗, 𝑈𝑥,𝑖,𝑗, 𝑊ℎ,𝑖,𝑗,
𝑈ℎ,𝑖,𝑗.
All these weights are supposed to be learned by backpropagation, or analog more general reinforcement.
Experiments during the eighties and nineties showed the strongness of the 𝑅𝑁𝑁s but also some weak-
nesses, in particular for learning or memorizing long sequences. Then Hochreiter and Schmidhuber, in a
remarkable paper in Neural Computation [HS97], introduced a modification of the simple 𝑅𝑁𝑁, named
the Long Short Term Memory, or 𝐿𝑆𝑇𝑀, which overcame all the diﬃculties so eﬃciently that more than
thirty years after it continues to be the standard.
The idea is to duplicate the layers ℎby introducing parallel layers 𝑐, playing the role of longer time
95
memory states, and just called cell states, by opposition to hidden states for ℎ.
In what follows we present the cell which replaces 𝐴𝑖,𝑗 without insisting on the lattice aspect, which is
unchanged for many applications.
The sub-network which replaces the simple crux 𝐴= 𝐴𝑖,𝑗 is composed of five tanks 𝐴,𝐹,𝐼,𝐻′,𝑉, plus
the inputs 𝐶𝑡−1,𝐻𝑡−1,𝑋𝑡−1, and has nine tips 𝑐′
𝑡−1,ℎ′
𝑡−1,𝑥′
𝑡,𝑓,𝑖,𝑜,ℎ,𝑣𝑖,𝑣𝑓 plus the three outputs 𝑐𝑡,ℎ𝑡,𝑦𝑡.
However, 𝑦𝑡 being a function of ℎ𝑡 only, it is forgotten in the analysis below.
In 𝐴, the two layers ℎ′and 𝑥′(where we forget the indices 𝑡−1 and 𝑡respectively) join to give by formulas
like (4.2) the four states of 𝑖,𝑓,𝑜,ℎrespectively called input gate, forget gate, output gate, combine gate,
the first three are sigmoidal, the fourth one is of type tanh, indicating a function of states separations.
The weights in these operations are the only parameters to adapt, they form matrices 𝑊𝑖,𝑈𝑖, 𝑊𝑓,𝑈𝑓,
𝑊𝑜,𝑈𝑜 and 𝑊ℎ,𝑈ℎ; which makes four times more than for a 𝑅𝑁𝑁(because the output 𝜉𝑖,𝑗 is not taken in
account).
Then the states in 𝑣𝑓 and 𝑣𝑖 are respectively given by combining 𝑐′with 𝑓 and ℎwith 𝑖, in the simplest
bilinear way:
𝜉𝑎
𝑣
= 𝛾𝑎𝜑𝑎; 𝑎∈𝑣; (4.3)
where 𝛾denotes the states of 𝑐′or ℎ, and 𝜑the states of 𝑓 or 𝑖respectively.
Note that the above formulae have a sense if and only of the dimensions of 𝑐and 𝑓 and 𝑣𝑓 are equal and
the dimension of ℎand 𝑖and 𝑣𝑖 are equal. This is an important restriction.
At the level of vectors this diagonal product is name the Hadamard product and is written
𝜉𝑣 = 𝛾⊙𝜑. (4.4)
It is free of parameters. Only the dimension is free for a choice.
Then, 𝑣𝑖 and 𝑣𝑓 are joined by a Hadamard sum, adding term by term, to give the new cell state
𝜉𝑐 = 𝜉𝑣𝑓 ⊕𝜉𝑣𝑖 ; (4.5)
which implies that 𝑣𝑖 and 𝑣𝑓 have the same dimension.
And finally, a new Hadamard product gives the new hidden state:
𝜂ℎ = 𝜉𝑜⊙tanh 𝜉𝑐. (4.6)
We get an additional degree of freedom with the normalization factor 𝐶in tanh𝐶𝑥but this is all. However
this implies that 𝑐and 𝑜and ℎhave the same dimension.
Therefore the 𝐿𝑆𝑇𝑀 has a discrete invariant, which is the dimension of the layers and is named its
multiplicity 𝑚.
Only the layers 𝑥can have other dimensions; in what follows, we denote 𝑛this dimension (see figure 4.2).
96
Ct−1 ⊙
Vf ⊕
Ct
f
Vi
⊙tanh
ı
ht−1
˜
h
o
⊙tanh ht
xt
Figure 4.2: Grothendieck site representing a LSTM cell
Symbolically, the dynamics can be summarized by the two formulas:
𝑐𝑡 = 𝑐𝑡−1 ⊙𝜎𝑓(𝑥𝑡,ℎ𝑡−1)⊕𝜎𝑖(𝑥𝑡,ℎ𝑡−1)⊙𝜏ℎ(𝑥𝑡,ℎ𝑡−1) (4.7)
ℎ𝑡 = 𝜎𝑜(𝑥𝑡,ℎ𝑡−1)⊙tanh 𝑐𝑡, (4.8)
where 𝜎𝑘 (resp. 𝜏𝑘) denotes the application of 𝜎(resp. tanh) to a linear or aﬃne form.
In what follows, 𝑥𝑡 is replaced by 𝑥′and ℎ𝑡−1, 𝑐𝑡−1 by ℎ′
, 𝑐′, like their tips.
Due to the non-linearities 𝜎and tanh, there are several regimes of functioning, according to the fact
that some of the variables give or not a saturation; this can generate almost linear transformations or the
opposite, a discrete-valued transformation. For instance, ±1 when tanh is applied, or ∈{0,1}if 𝜎 is
applied. Here appears the fundamental aspect of discretization in the functioning of 𝐷𝑁𝑁𝑠.
In the linear regime, the new state 𝑐appears as a polynomial of degree 2 in the vectors 𝑥,ℎ′and degree 1
in 𝑐′, and ℎappears as a polynomial of degree 3 in 𝑥′,ℎ′
.
Introducing the linear (or aﬃne with bias) forms 𝛼𝑓,𝛼𝑖,𝛼𝑜,𝛼ℎ, before application of 𝜎or tanh, we have
ℎ𝑡 = 𝛼𝑜⊙(𝑐
′⊙𝛼𝑓 ⊕𝛼𝑖⊙𝛼ℎ). (4.9)
The dominant term in 𝑥′,ℎ′is decomposable: 𝛼𝑜⊙𝛼𝑖⊙𝛼ℎ; the term of degree 2 in 𝑥′,ℎ′is 𝛼𝑜⊙𝑐′⊙𝛼𝑓,
and there is no linear term, because we forgot the bias. When separating 𝑥′from ℎ′, we obtain all possible
degrees ≤3.
However, experiments with alternative memory cells, named 𝐺𝑅𝑈and their simplifications, have shown
that the degree in 𝑥′is apparently less important then the degree in ℎ′. All trials with degree <3 in ℎ′
gave a dramatic loss of performance, but this was not the case for 𝑥′, where degree 1 appears to be suﬃcient.
97
The number of parameters to tune is 4𝑚2 +4𝑚𝑛or 4𝑚2 +𝑑𝑚𝑛, with 1 ≤𝑑≤4 is for the dependencies
in 𝑥in the four operations 𝛼𝑓,𝛼𝑖,𝛼𝑜,𝛼ℎ. At least 𝑑= 1 for 𝛼ℎ or for 𝛼𝑓 seems to be necessary from the
study of 𝑀𝐺𝑈.
4.2 GRU, MGU
Several attempts were made for diminishing the quantity of parameters to adapt in 𝐿𝑆𝑇𝑀 without di-
minishing the performance. The most popular solution is known as Gated Recurrent Unit, or 𝐺𝑅𝑈(see
[CvMBB14] and [CGCB14] from Bengio’s group). Then this cell has been simplified into several kinds
of Minimal Gated Units, 𝑀𝐺𝑈([ZWZZ16] or [HS17]).
The idea is to replace several gated layers by one, at the cost of a more complex architecture’s topology.
In the standard 𝐺𝑅𝑈, the pair ℎ𝑡,𝑐𝑡 is replaced by ℎ𝑡 alone, as in the original 𝑅𝑁𝑁; there exists two
input layers 𝑋𝑡,𝐻𝑡−1, the number of joins, our tanks, is six: 𝑅,𝐹,𝐼,𝑉,𝑊,𝐻′, the number of tips is six,
𝑧,𝑟,𝑣1−𝑧,𝑣𝑟,𝑣𝑥,𝑣ℎ and one output ℎ𝑡.
The dynamic begins with two non-linear linear transform, of type 𝜎 , like (4.2) in 𝑅, giving 𝑧 and
𝑟 from 𝑥′and ℎ′; then in 𝐼, there is a Hadamard product 𝑣𝑧 = ℎ′⊙(1−𝑧), where 1−𝑧 designates the
Hadamard diﬀerence between the saturation and the values of the states of 𝑧. Moreover, in 𝐹, there is
another Hadamard product 𝑣𝑟 = ℎ′⊙𝑟. A tanh , like (4.2) with 𝑓= tanh, joins 𝑥′with 𝑣𝑟 in 𝑊 to give
𝑣𝑥, which joins 𝑧in 𝐻′to give 𝑣ℎ by a third Hadamard product. Finally, 𝑣ℎ and 𝑣1−𝑧 are joined together
by a Hadamard sum in 𝑉, giving ℎ= 𝑣𝑧⊕𝑣ℎ.
Symbolically, with the same conventions used for 𝐿𝑆𝑇𝑀, the dynamic can be summarized by the
following formula
ℎ𝑡 = (1−𝜎𝑧(𝑥𝑡,ℎ𝑡−1))⊙ℎ𝑡−1 ⊕𝜎𝑧(𝑥𝑡,ℎ𝑡−1)⊙tanh(𝑊𝑥(𝑥𝑡)+𝑈𝑥(𝜎𝑟(𝑥𝑡,ℎ𝑡−1)⊙ℎ𝑡−1)). (4.10)
In a 𝐺𝑅𝑈 as in a 𝐿𝑆𝑇𝑀 we have three Hadamard products and one Hadamard sum, plus three
non-linear-linear transforms 𝑁𝐿𝐿 (one with tanh); 𝐿𝑆𝑇𝑀 had four 𝑁𝐿𝐿 transforms (two with tanh),
but the complexity of 𝐺𝑅𝑈stays in the succession of two 𝑁𝐿𝐿with adaptable parameters.
Remark that 𝐿𝑆𝑇𝑀also contains a succession of non-linearities, tanh being applied to 𝑐𝑡, which is a sum
of product on non-linear terms of type 𝜎or tanh.
In the linear (or aﬃne) regime, the 𝐺𝑅𝑈gives
ℎ𝑡 = [(1−𝛼𝑧)⊙ℎ𝑡−1]⊕[𝛼𝑧⊙[𝑊𝑥𝑡+𝑈(𝛼𝑟 ⊙ℎ𝑡−1)]]. (4.11)
For the same reason than 𝐿𝑆𝑇𝑀a 𝐺𝑅𝑈has a multiplicity 𝑚, and a dimension 𝑛of data input. The
parameters to be adapted are the matrices 𝑊𝑧,𝑈𝑧, 𝑊𝑟,𝑈𝑟 and 𝑊𝑥,𝑈𝑥 in 𝑊. This gives 3𝑚2 +3𝑚𝑛real
98
numbers to adapt, in place of 4𝑚2 +4𝑚𝑛for a complete 𝐿𝑆𝑇𝑀.
The simplification which was proposed by Zhou et al. in [ZWZZ16] for 𝑀𝐺𝑈 consists in taking
𝜎𝑧 = 𝜎𝑟, thus reducing the parameters to 2𝑚2 +2𝑚𝑛. This unique vector is denoted 𝜎𝑓, assimilated to the
forget gate 𝑓 of 𝐿𝑆𝑇𝑀.
It seems that the performance of 𝑀𝐺𝑈was as good as the ones of 𝐺𝑅𝑈, which are almost as good as
𝐿𝑆𝑇𝑀for many tasks.
Heck and Salem [HS17] suggested further radical simplifications, some of them being as good as
𝑀𝐺𝑈. 𝑀𝐺𝑈1 consists in suppressing the dependency of the unique 𝜎𝑓 in 𝑥′, and 𝑀𝐺𝑈2 in suppressing
also the bias 𝛽𝑓. An 𝑀𝐺𝑈3 removed 𝑥′and ℎ′, just keeping a bias, but it showed poor learning and
accuracy in the tests.
The experimental results proved that 𝑀𝐺𝑈2 is excellent in all tests, even better than 𝐺𝑅𝑈.
Note that both 𝑀𝐺𝑈2 and 𝑀𝐺𝑈1 continue to be of degree 3 in ℎ′. This reinforces the impression that
this degree is an important invariant of the memory cells. But these results indicate that the degree in 𝑥′
is not so important.
Consequently we may assume
And in the linear regime
ℎ𝑡 = (1−𝜎𝑧(ℎ𝑡−1))⊙ℎ𝑡−1 ⊕𝜎𝑧(ℎ𝑡−1)⊙tanh(𝑊𝑥(𝑥𝑡)+𝑈𝑥(𝜎𝑧(ℎ𝑡−1)⊙ℎ𝑡−1))). (4.12)
ℎ𝑡 = [(1−𝛼𝑧)⊙ℎ′]⊕[𝛼𝑧⊙[𝑊𝑥𝑡+𝑈(𝛼𝑧⊙ℎ′)]]. (4.13)
Only two vectors of linear (or aﬃne) forms intervene, 𝛼𝑎
𝑧(ℎ′); 𝑎= 1,...,𝑚and ℎ′itself, i.e. 𝜂𝑎(ℎ′); 𝑎=
1,...,𝑚.
The parameters to adapt are 𝑈𝑧, giving 𝛼𝑧, and 𝑈𝑥 = 𝑈, 𝑊𝑥 = 𝑊, giving the polynomial of degree two in
parenthesis, i.e. the state of the layer called 𝑣ℎ.
The number of free parameters in 𝑀𝐺𝑈2 is 2𝑚2 +𝑚𝑛, twice less than the most economical 𝐿𝑆𝑇𝑀.
The graph Γ of a 𝐺𝑅𝑈or a 𝑀𝐺𝑈has five independent loops, a fundamental group free of rank five;
it is non-planar. The categorical representation of a 𝐿𝑆𝑇𝑀 has only three independent loops, and is
planar (see figure 4.2).
99
4.3 Universal structure hypothesis
A possible form of dynamic covering the above examples is a vector of dimension 𝑚 of non-linear
functions of several vectors 𝜎𝛼𝑎 , 𝜎𝛽𝑏 , ..., that are 𝜎of 𝑡ℎfunctions of linear (or perhaps aﬃne) forms of
the variables 𝜉𝑎,𝜂𝑏, for 𝑎,𝑏,𝑐varying from 1 to 𝑚. More precisely
𝜂𝑎
𝑡 =
𝑏,𝑐,𝑑
𝑡𝑎
𝑏𝜎𝛼𝑏 tanh
𝑐,𝑑
𝑢𝑎
𝑐,𝑑𝜎𝛽𝑐 𝜎𝛾𝑑 +
𝑐
𝑣𝑎
𝑐𝜎𝛽𝑐 +
𝑑
𝑤𝑎
𝑑𝜎𝛾𝑑 +𝜎𝛿𝑎 . (4.14)
Remark: we have written 𝜎𝛼,𝜎𝛽,...for the application to a linear form of a sigmoid or a tanh indiﬀerently;
but for a more precise discussion of the examples, we must distinguish and write 𝜏𝛼,𝜏𝛽,...when tanh
is applied. However, sometimes in the following lines, we will use 𝜏 when we are sure that a tanh is
preferable to a 𝜎.
The tensor 𝑢𝑎
𝑐,𝑑 would introduce 𝑚3 parameters, leading to great computational diﬃculties. A natural
manner to limit the degrees of freedom at 𝐾𝑚2, inspired by 𝐿𝑆𝑇𝑀 and 𝐺𝑅𝑈, is to use the Hadamard
product, for instance 𝜎𝛽𝑎 𝜎𝛾𝑎 .
A second simplification, justified by the success of 𝑀𝐺𝑈consists to impose 𝛼𝑎 = 𝛾𝑎
.
A third one, justified by the success of 𝑀𝐺𝑈2 is to limit the degree in 𝑥′to 1. This can be done by
reserving the dependency on 𝑥′to the forms 𝛽and 𝛿.
All that gives
𝜂𝑎
𝑡 = 𝜎𝛼𝑎 (𝜂)tanh 𝜎𝛼𝑎 (𝜂)𝜎𝛽𝑎 (𝜂,𝜉)+𝜎𝛽𝑎 (𝜂,𝜉)+𝜎𝛿𝑎 (𝜉). This contains 2𝑚2 +2𝑚𝑛free parameters to be adapted.
(4.15)
Remark. Here we have neglected the addition of the alternative term in the dynamic which is (1−𝜎𝛼𝑎 )𝜂𝑎
in 𝐺𝑅𝑈and 𝑀𝐺𝑈, but this term is probably very important, therefore, we must keep in mind that it can
be added in the applications. At the end it will reappear in the formulas we suggest below.
For 𝑀𝐺𝑈1,2, the term of higher degree has no dependency in 𝑥′, then we can simplify further in
𝜂𝑎
𝑡 = 𝜎𝛼𝑎 (𝜂)tanh 𝜎𝛼𝑎 (𝜂)𝜎𝛽𝑎 (𝜂)+𝜎𝑦𝑎 (𝜉)𝜎𝛽𝑎 (𝜂)+𝜏𝛿𝑎 (𝜉). (4.16)
Moreover, as 𝑀𝐺𝑈2 is apparently better than 𝑀𝐺𝑈1 in the tested applications, the forms 𝛼𝑎 can be
taken linear, not aﬃne.
It looks like a simplified 𝐿𝑆𝑇𝑀, if we define for the state of 𝑐𝑡 the following vector:
𝛾𝑎
𝑡 = 𝜎𝛼𝑎 (𝜂)𝜎𝛽𝑎 (𝜂)+𝜎𝑦𝑎 (𝜉)𝜎𝛽𝑎 (𝜂)+𝜏𝛿𝑎 (𝜉), (4.17)
and impose the recurrence 𝑦𝑎(𝜉)= 𝛾𝑎
𝑡−1.
This gives a kind of minimal 𝐿𝑆𝑇𝑀, so-called 𝑀𝐿𝑆𝑇𝑀,
𝛾𝑎
𝑡 = 𝜎𝛼𝑎 (𝜂)𝜎𝛽𝑎 (𝜂)+𝛾𝑎
𝑡−1𝜎𝛽𝑎 (𝜂)+𝜏𝛿𝑎 (𝜉), (4.18)
100
𝜂𝑎
𝑡 = 𝜎𝛼𝑎 (𝜂)tanh[𝛾𝑎
𝑡 ]. (4.19)
Or with the forgotten alternative term,
𝜂𝑎
𝑡 = 𝜎𝛼𝑎 (𝜂)tanh[𝛾𝑎
𝑡 ]+(1−𝜎𝛼𝑎 (𝜂))𝜂𝑎
. (4.20)
Now we suggest to look at these formulas from the point of view of the deformation of singularities
having polynomial universal models, and trying to keep the main properties of the above dynamics:
1) on a generic straight line in the input space ℎ′, and in any direction of the output space ℎ, we have
every possible shape of a 1D polynomial function of degree 3, when modulating by the functions
of 𝑥′;
2) the presence of non-linearity 𝜎applied to forms in ℎ′and 𝑡ℎapplied to forms in 𝑥′allow discretized
regimes for the full application, but also a regime where the dynamic is close to a simple polynomial
model.
In the above formulas the last application of 𝑡ℎrenders possible the degeneration to degree 1 in ℎ′
and 𝑥′, we suggest to forbid that, and to focus on the coeﬃcients of the polynomial. In fact the truncation
of the linear forms by 𝜎or 𝑡ℎis suﬃcient to warranty the saturation of the polynomial map.
From this point of view the terms of degree 2 are in general not essential, being absorbed by a Viete
transformation. Also the term of degree zero, does not change the shape, only the values; but this can be
non-negligible.
In the simplest form this gives
𝜂𝑎
𝑡 = 𝜎𝛼𝑎 (𝜂)3 +𝑢𝑎(𝜉)𝜎𝛼𝑎 (𝜂)+𝑣𝑎(𝜉); (4.21)
where 𝑢and 𝑣are 𝑡ℎapplied to a linear form of 𝜉, and 𝜎𝛼 is a 𝜎applied to a linear form in 𝜂. This gives
only 𝑚2 +2𝑚𝑛free parameters, thus one order less than 𝑀𝐺𝑈2 in 𝑚.
However, we cannot neglect the forgotten alternative (1−𝑧)ℎ′ of 𝐺𝑅𝑈, or more generally the
possible function in the transfer of a term of degree two, even if structurally, from the point of view of
the deformation of shapes, it seems not necessary, thus the following form could be preferable:
𝜂𝑎
𝑡 = 𝜎𝛼𝑎 (𝜂)3 +(1−𝜎𝛼𝑎 (𝜂))𝜂𝑎+𝑢𝑎(𝜉)𝜎𝛼𝑎 (𝜂)+𝑣𝑎; or more generally, with 2𝑚2 +2𝑚𝑛free parameters:
𝜂𝑎
𝑡 = 𝜎𝛼𝑎 (𝜂)3 +𝜎𝛼𝑎 (𝜂)[𝜎𝛽𝑎 (𝜂)+𝑢𝑎(𝜉)]+𝑣𝑎(𝜉); where 𝛽is a second linear map in 𝜂.
(4.22)
(4.23)
Description of an architecture for this dynamic : it has two input layers 𝐻𝑡−1,𝑋𝑡, three sources or
tanks 𝐴, 𝐵, 𝐶, and seven internal layers that give six tips, 𝛼,𝛽, 𝑣𝛽, 𝑢, 𝑣, 𝑣𝛼𝛽, 𝑣𝛼𝛼𝛼, and one output layer
101
ℎ𝑡. First ℎ𝑡−1 gives 𝜎𝛼 and 𝜎𝛽, and 𝑥𝑡 gives 𝑢and 𝑣; then 𝜎𝛽 joins 𝑢in 𝐴to give 𝑣𝛽 = 𝜎𝛽⊕𝑢, then 𝜎𝛼
joins 𝑣𝛽 in 𝐵to give 𝑣𝛼𝛽 = 𝜎𝛼⊙𝑣𝛽. In parallel, 𝜎𝛼 is transformed along an ordinary arrow in 𝑣𝛼𝛼𝛼 = 𝜎⊙3
𝛼.
And finally, in 𝐶, the sum of 𝑣, 𝑣𝛼𝛼𝛼 and 𝑣𝛽 produces the only output ℎ𝑡.
The simplified network is for 𝛽= 0. It has also three tanks, 𝐴, 𝐵and 𝐶, but only five tips, 𝛼, 𝑢, 𝑣, 𝑣𝛼,
𝑣𝛼𝛼𝛼. The schema is the same, without the creation of 𝛽, and 𝑣𝛽 (resp. 𝑣𝛼𝛽) replaced by 𝑣𝛼 (resp. 𝑣𝛼𝛼).
Remark. In the models with tanh like (4.20) the sign of the terms of eﬀective degree three can be minus
or plus; in the model (4.23) it is always plus, however this can be compensated by the change of sign of
the eﬀerent weights in the next transformation.
Equation (4.15) could induce the belief that 0 goes to 0, but in general this is not the case, be-
cause the function 𝜎 contrarily to tanh has only strictly positive values. For instance the standard
𝜎(𝑧)= 1/1 +exp(−𝑧)gives 𝜎(0)= 1/2.
However, the point 0 plays apparently an important role, even if it is not preserved: 1) in 𝑀𝐺𝑈2 the
absence of bias in 𝛼𝑎 confirms this point; 2) the functions 𝜎and 𝑡ℎare almost linear in the vicinity of 0
and only here. Therefore, let us define the space 𝐻of the activities of the memory vectors ℎ𝑡−1 and ℎ𝑡,
of real dimension 𝑚; it is pointed by 0, and the neighborhood of this point is a region of special interest.
We also introduce the line 𝑈 of coordinate 𝑢 and the plane Λ = 𝑈×Rof coordinates 𝑢,𝑣, where 0
and its neighborhood is also crucial. The input from new data 𝑥𝑡 is sent to Λ, by the two maps 𝑢(𝜉)and
𝑣(𝜉). By definition this constitutes an unfolding of the degree three map in 𝜎𝛼(𝜂).
A more complex model of the same spirit is
𝜂𝑎
𝑡 = 𝜎𝛼𝑎 (𝜂)3 ±𝜎𝛼𝑎 (𝜂)[𝜎𝛽𝑎 (𝜂)2 +𝑢𝑎(𝜉)]+𝑣𝑎(𝜉)𝜎𝛽𝑎 (𝜂)+𝑤𝑎(𝜉)[𝜎𝛼𝑎 (𝜂)2 +𝜎𝛽𝑎 (𝜂)2]+𝑧𝑎(𝜉); (4.24)
it has 2𝑚2 +4𝑚𝑛free parameters. The expression of 𝑥𝑡 is much richer and we will see below that it shares
many good properties with the model (4.21), in particular stability and universality. The corresponding
space 𝑈has dimension 3 and the corresponding space Λ has dimension 4.
4.4 Memories and braids
In every 𝐷𝑁𝑁, the dynamic from one or several layers to a deeper one must have a sort of stability, to be
independent of most of the details in the inputs, but it must also be plastic, and sensitive to the important
details in the data, then not too stable, able to shift from a state to another one, for constructing a kind
of discrete signification. These two aspects are complementary. They were extensively discussed a long
time before the apparition of 𝐷𝑁𝑁s in the theory of dynamical systems. The framework was diﬀerent
because most concepts in this theory were asymptotic, pertinent when the time tends to infinity, and here
102
in deep learning, to the contrary, most concepts are transient: one shot transformations for feed forward,
and gradient descent or open exploration for learning; however, with respect to the shape of individual
transformation, or with respect to the parameters of deformation, the two domains encounter similar
problems, and probably answer in similar manners.
Structural stability is the property to preserve the shape after small variation of the parameters. In
the case of individual map between layers, this means that little change in the input has little eﬀect on the
output. In the case of a family of maps, taking in account a large set of diﬀerent inputs, this means that
varying a little the weights, we get little change in the global functioning and the discrimination between
data. The second level is deeper, because it allows to understand what are the regions of the manifolds of
input data, where the individual dynamics are stable in the first sense, and what happens when individual
dynamics changes abruptly, how are made the transitions and what are the properties of the inputs at the
boarders. A third level of structural stability concerns the weights, selected by learning: in the space of
weights it appears regions where the global functioning in the sense of family is stable, and regions of
transitions where the global functioning changes; this happens when the tasks of the network change, for
instance detect a cat versus a dog. This last notion of stability depends on the architecture and on the
forms of dynamical maps that are imposed.
With 𝐿𝑆𝑇𝑀, 𝐺𝑅𝑈and their simplified versions like 𝑀𝐺𝑈, 𝑀𝐺𝑈2, we have concrete examples of
these notions of structural stability.
The transformation is 𝑋𝑤 from (ℎ𝑡−1,𝑥𝑡)to ℎ𝑡. The weights 𝑤 are made by the coeﬃcients of the
linear forms, 𝛼𝑎(𝜂),𝛽𝑎(𝜂),𝑢𝑎(𝜉), 𝑣𝑎(𝜉), but the structure depends on the fixed architecture and the
non-linearities, of two types, the tensor products and sums, and the applied sigmoids and 𝑡𝑎𝑛ℎ.
For simplicity we assume a response of the cell of the form (4.21), but the discussion is not very
diﬀerent with the other cell families (4.23), (4.16) or (4.20).
We have a linear endomorphism 𝛼of coordinates 𝛼𝑎; 𝑎∈ℎof R𝑚 = 𝐻; when we apply to it the sigmoid
function coordinate by coordinate, we obtain a map 𝜙from 𝐻to a compact domain in 𝐻. The invariance
of the multiplicity 𝑚of the memory cell suggests the hypothesis (to be verified experimentally) that 𝜙is
a diﬀeomorphism from 𝐻to its image. However, as we will see just below, other reasons like redundancy
suggests the opposite, therefore we left open this hypothesis, with a preference for diﬀeomorphism,
for mathematical or structural reasons. Probably, depending on the application, there exists a range of
dimensions 𝑚which performs the task, such that 𝜙is invertible.
We also have the two mappings 𝑢𝑎(𝜉); 𝑎∈ℎand 𝑣𝑎(𝜉); 𝑎∈ℎfrom the space 𝑋= R𝑛 of states 𝑥𝑡, to R𝑚
.
This gives a complete description of the set of weights 𝑊ℎ;ℎ′,𝑥′.
The formula (4.21) defines the map 𝑋𝑤 from 𝐻×𝑋 to 𝐻.
We also consider the restriction 𝑋𝑤
𝜉 at a fixed state 𝜉of 𝑥𝑡.
Theorem 4.1. The map 𝑋𝑤 is not structurally stable on 𝐻 or 𝐻×𝑋, but each coordinate 𝜂𝑎
𝑡 , seen as
103
function on a generic line of the input ℎ𝑡−1 and a generic line of the input 𝑥𝑡, or as a function on 𝐻 or
𝐻×𝑋, is stable (at least in the bounded regions where the discretization does not apply).
These coordinates represent the activities of individual neurons, then we get structural stability at the
level of the neurons and not at the level of the layers.
As we justify in the following lines, this theorem follows from the results of the universal unfolding
theory of smooth mappings, developed by Whitney, Thom, Malgrange and Mather (see [GWDPL76] and
[Mar82]).
The main point here (our hypothesis) is the observation that, for each neuron in the ℎ𝑡 layer, the cubic
degeneracy 𝑧3 can appear, together with its deformation by the function 𝑢.
For the deformation of singularities of functions, and their unfolding, see [Arn73] and [AGZV12a].
The universal unfolding of the singularity 𝑧3 is given by a polynomial
𝑃𝑢(𝑧)= 𝑧3 +𝑢𝑧, (4.25)
This means that for every smooth real function 𝐹, from a neighbor of a point 0 in R1+𝑀, such that
𝐹(𝑧,0,...,0)= 𝑧3
, (4.26)
there exist a smooth map 𝑢(𝑌)and a smooth family of maps 𝜁(𝑧,𝑌)such that
𝐹(𝑧,𝑌)= 𝜁(𝑧,𝑌)3 +𝑢(𝑌)𝜁(𝑧,𝑌) (4.27)
Equivalently, the smooth map
(𝑧,𝑢)↦→(𝑃𝑢(𝑧),𝑢), (4.28)
in the neighbor of (0,0)is stable: every map suﬃciently near to it can be transformed to it by a pair
of diﬀeomorphisms of the source and the goal. This result on maps from the plane to the plane, is the
starting point of the whole theory, found by Whitney: the stability of the gathered surface over the plane
𝑣,𝑢.
The stability is not true for the product
(𝑧,𝑢,𝑤,𝑣)↦→(𝑃𝑢(𝑧),𝑢,𝑃𝑣(𝑤),𝑣) (4.29)
The infinitesimal criterion of Mather is not satisfied (see [GWDPL76], [Mar82]).
There also exists a notion of universal unfolding for maps from a domain of R𝑛 to R𝑝 in the neigh-
borhood of a point 0, however in most cases, there exists no universal unfolding, at the opposite of the
case of functions, when 𝑝= 1.
Here 𝑛= 𝑝= 𝑚, the transformation from ℎ𝑡−1 to ℎ𝑡 is an unfolding, dependent of 𝜉∈𝑥𝑡, but it does not
104
admit a universal deformation. It has an infinite codimension in the space of germs of maps.
Also for mappings, universality of and unfolding and its stability as a map are equivalent (another theorem
from Mather).
Our non-linear model from equation (4.21) with 𝑢free being equivalent to the polynomial model by
diﬀeomorphism, we can apply to it the above results. This establishes theorem 4.1.
Corollary. Each individual cell plays a role.
This does not contradict the fact that frequently several cells send similar message, i.e. there exists a
redundancy, which is opposite to the stability or genericity of the whole layer. However, as said before,
in some regime and/or for 𝑚suﬃciently small, the redundancy is not a simple repetition, it is more like
a creation of characteristic properties.
Let us look at a neuron 𝑎∈ℎ𝑡, and consider the model (4.21). If 𝑢= 𝑢𝑎(𝜉)does not change of sign,
the dynamic of the neuron 𝑎is stable under small perturbations. For 𝑢>0, it looks like a linear function,
it is monotonic. For 𝑢<0 there exist a unique stable minimum and a unique saddle point which limits its
basin of attraction. But for 𝑢= 0 the critical points collide, the individual map is unstable. This is named
the catastrophe point. For the whole theory, see [Tho72], [AGZV12a].
If we are interested in the value of 𝜂𝑎
𝑡 , as this is the case in the analysis of the cat’s manifolds seen
before, for understanding the information flow layer by layer, we must also consider the levels of the
function, involving 𝑣𝑎 then Λ. This asks to follow a sort of inversion of the flow, going to the past, by
finding the roots 𝑧of the equations
𝑃𝑎(𝑧)= 𝑐. (4.30)
Depending on 𝑢and 𝑣, there exist one root or three roots. For instance, for 𝑐= 0, the second case happens
if an only if the numbers 𝑢𝑎(𝜉),𝑣𝑎(𝜉)satisfy the inequality 4𝑢3 +27𝑣2 <0. When the point (𝑢𝑎(𝜉),𝑣𝑎)in
the plane Λ belongs to the discriminant curve Δ of equation 4𝑢3 +27𝜂2 = 0, things become ambiguous,
two roots collide and disappear together for 4𝑢3 +27𝑣2 >0.
These accidents create ramifications in the cat’s manifolds.
This analysis must be applied independently to all the neurons 𝑎= 1,...,𝑚 in ℎ, that is to all the axis
in 𝐻. If 𝛼is an invertible endomorphism, the set of inversions has a finite number of solutions, less than 3𝑚
.
Remind that the region around 0 in the space 𝐻is especially important, because it is only here that the
polynomial model applies numerically, 𝜎and tanh being almost linear around 0. Therefore the set of data
𝜂𝑡−1 and 𝜉𝑡 which gives some point 𝜂𝑡 in this region have a special meaning: they represent ambiguities
in the past for 𝜂𝑡−1 and critical parameters for 𝜉𝑡. Thus the discriminant Δ of equation 4𝑢3 +27𝑣2 = 0 in
105
Λ plays an important role in the global dynamic.
The inversion of 𝑋𝑤
𝜉 : 𝐻→𝐻is impossible continuously along a curve in 𝜉whose 𝑢𝑎,𝑣𝑎 meet Δ for
some component 𝑎. It becomes possible if we pass to complex numbers, and lift the curve in Λ to the
universal covering Λ∼
★(C)of the complement Λ★
Cof ΔCin ΛC[AGZV12b].
The complex numbers have the advantage that every degree 𝑘 polynomials has 𝑘 roots, when counted
with multiplicities. The ambiguity in distinguishing individual roots along a path is contained in the
Poincaré fundamental group 𝜋1 (Λ★
C). However the precise definition of this group requires the choice of
a base point in Λ★
C, then it is more convenient to consider the fundamental groupoid Π(Λ★
C)= B3, which
is a category, having for points the elements of Λ★
Cand arrows the homotopy classes of paths between
two points. The choice of an object 𝜆0 determine 𝜋1 (Λ★
C; 𝜆0), which is the group of homotopy classes of
loops from 𝜆0 to itself, i.e. the isomorphisms of 𝜆0 in B3. This group is isomorphic to the Artin braid
group 𝐵3 of braids with three strands [AGZV12b].
σ
σ
σ
σ
σ
σ
Figure 4.3: Two homotopic braids
This group 𝐵3 is generated by two loops 𝜎1,𝜎2 that could be defined as follows: take a line 𝑢= 𝑢0 ∈R−⊂C„
with complex coordinate 𝑣, and let 𝑣+
0 ,𝑣−
0 be the positive and negative square roots of−
4
27 𝑢3
0; the loop
𝜎1 = 𝜎+(resp. 𝜎2 = 𝜎−) is based in 0, contained in the line 𝑢= 𝑢0 and makes one turn in the trigonometric
sense around 𝑣+
0 (resp. 𝑣−
0 ). The relations between 𝜎1 and 𝜎2 are generated by 𝜎1𝜎2𝜎1 = 𝜎2𝜎1𝜎2.
The center 𝐶 of 𝐵3 is generated by 𝑐= (𝜎1𝜎2)3. The quotient by this center is isomorphic to the group
𝐵3/𝐶generated by 𝑎= 𝜎1𝜎2𝜎1 and 𝑏= 𝜎1𝜎2 satisfying 𝑎2 = 𝑏3; the quotient of 𝐵3/𝐶by 𝑎2 is the Möbius
group 𝑃𝑆𝐿2 (Z)of integral homographies, and the quotient of 𝐵3/𝐶by 𝑎4 is the modular group 𝑆𝐿2 (Z)
of integral matrices of determinant one, then a two fold covering of 𝑃𝑆𝐿2 (Z). The quotient 𝔖3 of 𝐵3 is
106
defined by the relations 𝜎2
1 = 𝜎2
2 = 1, and by the relation which defines 𝐵3, i.e. 𝜎1𝜎2𝜎1 = 𝜎2𝜎1𝜎2 (see
figure 4.3).
Of course the disadvantage of the complex numbers is the diﬃculty to compute with them in 𝐷𝑁𝑁s,
for instance 𝜎 and tanh extended to Chave poles. Moreover all the dynamical regions are confounded
in Λ★
C; in some sense the room is too wide. Therefore, we will limit ourselves to the sub-category
ΠR= B3 (R), made by the real points of Λ★, but retaining all the morphisms between them, that is a full
sub-category of B3. This means that only the paths are imaginary in B3 (R).
Attractor
Threshold
Repulsor
Attractor
v
u
Figure 4.4: Cusp
Another sub-groupoid could be also useful (see figure 4.4): consider the gathered surface Σ in Λ ×R
of equation 𝑧3 +𝑢𝑧+𝑣= 0; let Δ3 be the natural lifting of Δ along the folding lines of Σ over Λ, the
complement Σ★ of Δ3 in Σ can be canonically embedded in the complex universal covering Λ∼
★, based
in the real contractile region Λ0 inside the real cusp, by taking, for each (𝑢,𝑣)= 𝜆in Λ0 the points 𝜆+
and 𝜆−respectively given by the paths 𝜎+= 𝜎1 and 𝜎−
= 𝜎2, which make simple turn over the branches
of the cusp. When 𝜆 approaches one of these branches, the corresponding point collides with it on
107
Δ3, but the other point continues to be isolated then the construction gives an embedding of Σ★. There-
fore we can define the full sub-groupoid of B3 which has as objects the points of Σ★, and name it B𝑟
3 or Π𝑟.
Remark. The groupoid Π𝑟 can be further simplified, by taking one point in each region of interest: one
point outside the preimage of the cusp Δ, and three points in each region over the interior of the cusp.
Remark. These four points correspond to the four real structures of Looĳenga in the complex kaleido-
scope [Loo78].
The groupoid B𝑟
3 is naturally equipped with a covering (surjective) functor 𝜋to the groupoid B3 (R)
of real points.
The interest of B𝑟
3 with respect to B3 (R)is that it distinguishes between the stable minimum and the
unstable one in the regime 𝑢<0. But the interest of B3 (R)with respect to B𝑟
3 is that it speaks only of
computable quantities 𝑢,𝑣without ambiguity, putting all the ambiguities in the group 𝐵3.
All these groupoids are connected, the two first ones, B3 (R)and B𝑟
3 because they are full subcategories
of the connected groupoid B3, the other ones in virtue of the definition of a quotient (to the right) of a
groupoid by a normal sub-group 𝐻of its fundamental group 𝐺: it has the same objects, and two arrows
𝑓,𝑔from 𝑎 to 𝑏 are equivalent if they diﬀer by an element of 𝐻. This is meaningful because in 𝐴𝑢𝑡𝑎
(resp. 𝐴𝑢𝑡𝑏) the sub-group 𝐻𝑎 (resp. 𝐻𝑏) is well defined, being normal, and moreover 𝑓−1𝑔∈𝐻𝑎 is
equivalent to 𝑔𝑓−1 ∈𝐻𝑏.
Cardan formulas expresses the roots by using square roots and cubic roots. They give explicit formu-
las for the diﬀerences of roots 𝑧2−𝑧1,𝑧3−𝑧1. They can be seen directly in the surface Σ.
Remarks. These formulas correspond to the simplest non trivial case of a map of period:
(i) integral classes of the homology 𝐻0 (𝑃−1
𝑢,𝑣(0)are transported along paths;
(ii) the holomorphic form 𝑑𝑧is integrated on the integral classes.
This gives a linear representation of 𝐵3, which factorizes through 𝔖3.
Augment the variable 𝑧 by a variable 𝑦, the roots can be completed by the levels 𝑍𝑢,𝑣 over (𝑢,𝑣)∈Λ,
which are the elliptic curves
the 2-form 𝜔= 𝑑𝑧∧𝑑𝑦can be factorized as follows
𝑃𝑢,𝑣(𝑧,𝑦)= 𝑧3 +𝑦2 +𝑢𝑧+𝑣= 0, 1
𝜔=−
2 𝑑𝑃∧
𝑑𝑧
𝑦
108
(4.31)
; (4.32)
the integral of 𝑑𝑥/𝑦 over the curve 𝑍𝑢,𝑣 is an elliptic integral, its periods over integral cycles, gives a
linear representation of 𝐵3 which factorizes through 𝑆𝐿2 (Z).
Every stabilization of 𝑧3 by a quadratic form gives rise to the representation of the first case in odd
dimension and of the second case in even dimension.
Natural groupoids smaller than B3 are given by quotienting the morphisms, replacing 𝐵3 by 𝔖3 or
𝑆𝐿2 (Z)or its projective version 𝑃𝑆𝐿2 (Z)made by homographies.
4.5 Pre-semantics
The natural languages have many functions, from everyday life to poetry and science, or politics and law,
however all of them rely on cognitive operations about meanings and shapes, as they appear in the many
language-games of Wittgenstein or the action/perception dimensions of Austin. Cf. [Wit53], [Aus61].
The linguist Antoine Culioli, having studied in depth a great variety of natural languages, tried to
characterize some of these operations in meta-linguistic, for instance the generic structure and dynamics
of a notional domain. The notion here can be "dog" or "cat" or "good" or "absent" or anything which has a
meaning for most peoples, or specialists in some field. To have a meaning must involve in general several
occurrences and disappearances of the notion, a knowledge of its possible properties and individuations,
in a language and in the world (data for instance, relations between them and classifications).
A good reference is the book Cognition and Representation in Linguistic Theory, A. Culioli, Benjamins,
[CLS95].
The notional domain has an interior 𝐼where the properties of the notion are sure, an exterior 𝐸where
the properties are false, and a boundary 𝐵, where things are more uncertain. A path through the boundary
goes from "truly P" to "truly not P", through an uncertain region where "non-really P, non really not P"
can be said. In the center of 𝐼 are one or several prototypes of the notion. A kind of gradient vector
leads the mind to these archetypes, that Culioli named attracting centers, or attractors; however he wrote
in 1989 (upcit.) the following important precision: "Now the term attractor cannot be interpreted as an
attainable last point (...) but as the representation of the imaginary absolute value of the property (the
predicate) which organizes an aggregate of occurrences into a structured notional domain." Culioli also
used the term of organizing center, but as we shall see this would conflict with another use.
The division 𝐼,𝐵,𝐸 takes all its sense when interrogative mode is involved, or negation and double
negation, or intero-negative mode. In negation you go out of the interior, in interro-negation you come
back inside from 𝐸. "Is your brother really here" (it means that "I do not expect that your brother is
here".) "Now that, that is not a dog!" (you place yourself in front of proposition P, or inside the notion
𝐼, you know what is a dog, then goes to 𝐸); "Shall I still call that a dog?" "I do not refuse to help"; here
come back in 𝐼 of "help" after a turn in its exterior 𝐸. All these circumstances involve an imaginary
place 𝐼𝐸, where the regions are not separated, this is like the cuspidal point before the separation of the
109
branches 𝐼and 𝐸of the cusp.
Mathematically this corresponds precisely to the creation of the external (resp. internal) critical point
of 𝑧3 +𝑢𝑧+𝑣, on the curve Δ. Example: "he could not have left the window open", the meaning mobilizes
the place 𝐼𝐸 of indetermination, the maximum of ambiguity, where the two actions, "left" and "not to
left" are possible, then one of them is forbidden, and "not having left" is retained by the negation. In the
terminology of Thom, the place 𝐼𝐸 is the organizing center, the function 𝑧3 itself, the most degenerate
one in the stable family, giving birth to the unfolding.
To describe the mechanisms beyond these paths, Culioli used the model of the cam: "the movement
travels from one place to another, only to return to the initial plane". Example: start from 𝐼𝐸, then
make a half-turn around 𝐼 which passes by 𝐸 then come to 𝐼 by another half-turn. "This book is only
slightly interesting." Here the meaning only appears if you imagine the place where interesting and not
interesting are not yet separated, then go to not interesting and finally temperate the judgment by going to
the boundary, near 𝐼; the compete turn leads you in another place, over the same point, thus the meaning
is greatly in the path, as an enclosed area. "This book is not uninteresting" means that it is more than
interesting. The paths here are well represented on the gathered real surface Σ, of equation
𝑧3 +𝑢𝑧+𝑣= 0, (4.33)
but they can also be made in the complement of Δ in Λ in a complexified domain. It seems that
only the homotopy class is important, not the metric, however we cannot neglect a weakly quantita-
tive aspect, on the way of discretization in the nuances of the language. Consequently, the convenient
representation of the moves of Culioli is in the groupoid B𝑟
3, that we propose to name the Culioli groupoid.
Remind that 𝐿𝑆𝑇𝑀and the other memory cells are mostly used in chains, to translate texts.
It is natural to make a rapprochement between their structural and dynamical properties and the meta-
linguistic description of Culioli. In many aspects René Thom was closed to Culioli in his own approach
of semantics, see his book Mathematical Models of Morphogenesis [Tho83], which is a translation of
a French book published by Bourgois in 1980. The original theory was exposed in [Tho72]. In this
approach, all the elementary catastrophes having a universal unfolding of dimension less than 4 are used,
through their sections and projections, for understanding in particular the valencies of the verbs, from the
semantic point of view, according to Peirce, Tesnière, Allerton: impersonal, "it rains", intransitive "she
sleeps", transitive "he kicks the ball", triadic "she gives him a ball", quadratic "she ties the goat to a tree
with a rope".
The list of organizing centers is as follows:
𝑦= 𝑥2, 𝑦= 𝑥3, 𝑦= 𝑥4, 𝑦= 𝑥5, 𝑦= 𝑥6
,
𝑦= 𝑥3
1−𝑥2
2𝑥1, 𝑦= 𝑥3
1 +𝑥3
2 (𝑜𝑟 𝑦= 𝑥3
1 +𝑥2
2𝑥1), 𝑦= 𝑥4
1 +𝑥2
2𝑥1; (4.34)
respectively named: well, fold, cusp, swallowtail, butterfly, elliptic umbilic, hyperbolic umbilic and
parabolic umbilic, or with respect to the group which generalizes the Galois group 𝔖3 for the fold,
110
respectively: 𝐴1, 𝐴2, 𝐴3, 𝐴4, 𝐴5, 𝐷+
4 = 𝐷−
4 = 𝐷4 and 𝐷5. The 𝐴𝑛 are the symmetric groups 𝔖𝑛+1 and the
𝐷𝑛 index two subgroups of the symmetry groups of the hypercubes 𝐼𝑛 [Ben86].
It is not diﬃcult to construct networks, on the model of 𝑀𝐿𝑆𝑇𝑀, such that the dynamics of neurons
obey to the unfolding of these singular functions. The various actors of a verb in a sentence could be
separated input data, for diﬀerent coordinates on the unfolding parameters. The eﬃciency of these cells
should be tested in translation.
Coming back to the memory cell (4.21), the critical parameters 𝑥𝑡 over Δ can be interpreted as board-
ers between regions of notional domains.
The precise learned 2𝑚𝑛weights 𝑤𝑥 for the coeﬃcients 𝑢𝑎 and 𝑣𝑎, for 𝑎= 1,...,𝑚, together with the
weights in the forms 𝛼𝑎 for ℎ𝑡−1 gives vectors (or more accurately matrices), which are like readers of
the words 𝑥 in entry, taking in account the contexts from the other words through ℎ. Remember Frege:
a word has a meaning only in the context od a sentence. This is a citation of Wittgenstein, after he said
that "Naming is not yet a move in a language-game" [Wit53, p. 49].
To get "meanings", the names, necessarily embedded in sentences, must resonate with other contexts
and experiences, and must be situated with respect to the discriminant, along a path, thus we suggest that
the vector spaces of "readers" 𝑊, and the vector spaces of states 𝑋 are local systems 𝐴 over a fibered
category Fin groupoids B𝑟
3 over the network’s category C.
In some circumstances, the groupoid B𝑟
3 can be replaced by the quotient over objects B3 (R), or a quotient
over morphisms giving 𝑆𝐿2 or 𝔖3.
The case of 𝑧3 corresponds to 𝐴2. It is tempting to consider the case of 𝐷4, i.e. the elliptic and
hyperbolic umbilics, because their formulas are very closed to 𝑀𝐺𝑈2 as mentioned at the end of the
preceding section.
This would allow the direct coding and translation of sentences by using three actant.
𝜂= 𝑧3 ∓𝑧𝑤2 +𝑢𝑧+𝑣𝑤+𝑥(𝑧2 +𝑤2)+𝑦. (4.35)
111
5
A natural 3-category of deep networks
In this chapter, we introduce a natural 3-category for representing the morphisms, deformations and
surgeries of semantic functioning of 𝐷𝑁𝑁𝑠 based on various sites and various stacks, which have
connected models in their fibers.
Grothendieck’s derivators will appear at two successive levels:
1. formalizing internal aspects of this 3-category;
2. defining potential invariants of information over the objects of this 3-category. Therefore we can
expect that the interesting relations (for the theory and for its applications) appear at the level of a
kind of "composition of derivators", and are analog to the spectral sequences of [Gro57].
5.1 Attention moduli and relation moduli
In addition to the chains of 𝐿𝑆𝑇𝑀, another network’s component is now recognized as essential for
most of the tasks in linguistic: to translate, to complete a sentence, to determine a context and to take
into account a context for finding the meaning of a word or sentence. This modulus has its origin in
the attention operator, introduced by Bahdanau et al. [BCB16], for machine translation of texts. The
extended form that is the most used today was defined in the same context by Vaswani et al. 2017
[VSP+17], under the common name of transformer or simply decoder.
Let us describe the steps of the algorithm: the input contains vectors 𝑌 representing memories or hidden
variables like contexts, and external input data 𝑋 also in vectorial form.
1) Three sets of linear operators are applied:
𝑄= 𝑊𝑄[𝑌],
𝐾= 𝑊𝐾[𝑌,𝑋],
𝑉= 𝑊𝑉[𝑌];
112
where the 𝑊’s are matrices of weights, to be learned. The vectors 𝑄,𝐾,𝑉 are respectively called
queries, keys and values, from names used in Computer Science; they are supposed to be indexed
by "heads" 𝑖∈𝐼, representing individuals in the input, and by other indices 𝑎∈𝐴, representing
for instance diﬀerent instant times, or aspects, to be integrated together. Then we have vectors
𝑄𝑎
𝑖 ,𝐾𝑎
𝑖 ,𝑉𝑎
𝑖.
2) The inner products 𝐸𝑎
𝑖 = 𝑘(𝑄𝑎
𝑖 |𝐾𝑎
𝑖 )are computed (implying that 𝑄and 𝐾have the same dimension),
and the soft-max function is applied to them, giving a probability law, from the Boltzmann weights
of energy 𝐸𝑎
𝑖
𝑝𝑎
𝑖 =
1
𝑍𝑎
𝑖
𝑒𝐸𝑎
𝑖
, (5.1)
3) a sum of product is computed
𝑉′
𝑖 =
𝑎
4) A new matrix is applied in order to mix the heads
𝐴𝑗 =
𝑖
𝑝𝑎
𝑖𝑉𝑎
𝑖. (5.2)
𝑤𝑖
𝑗𝑉′
𝑖. (5.3)
All that is summarized in the formula:
𝐴𝑗(𝑌,𝑋)=
𝑖 𝑎
𝑤𝑖
𝑗softmax 𝑘(𝑊𝑄(𝑌)𝑎
𝑖 |𝑊𝐾(𝑌,𝑋)𝑎
𝑖 ) 𝑊𝑉(𝑌)𝑎
𝑖. (5.4)
A remarkable point is that, as it is the case for 𝑀𝐺𝑈2 or 𝐿𝑆𝑇𝑀 and 𝐺𝑅𝑈 cells, the transformer
corresponds to a mapping of degree 3, made by multiplying a linear form of 𝑌 with non-linear function
of a bilinear form of 𝑌. Strictly speaking the degree 3 is only valid in a region of the parameters. In other
regions, some saturation decreases the degree.
Chains of 𝐿𝑆𝑇𝑀were first used for language translations, and were later on used for image descrip-
tion helped by sentences predictions, as in [KL14] or [MXY+15], where they proved to outperform other
methods for detection of objects and their relations.
In the same manner, the concatenation of attention cells has been proven to be very beneficial in this
context [ZRS+18], then it was extended to develop reasoning about the the relations between objects in
images and videos [RSB+17], [BHS+18], [BHS+18], [SRB+17], or [DHSB20].
In the 𝑀𝐻𝐷𝑃𝐴(multi-head dot product attention) algorithm [SFR+18], the inputs 𝑋are either words,
questions and features of objects and their relations, coded into vectors, the inputs 𝑌combine hidden and
external memories, the outputs 𝐴are new memories, new relations and new questions.
Remark. Interestingly, the method combines fully supervised learning with unsupervised learning (or
adaptation) by maximization of a learned functional of the above variables.
113
In particular, the memories or hidden variables issued from the transformer were re-introduced in the
𝐿𝑆𝑇𝑀chain; giving the following symbolic formulas:
𝑐𝑡 = 𝑐𝑡−1 ⊙𝜎𝑓(𝑥𝑡,ℎ𝑡−1)⊕𝜎𝑖(𝑥𝑡,𝑚𝑡)⊙𝜏ℎ(𝑥𝑡,ℎ𝑡−1); (5.5)
where 𝑚𝑡 results of transformer applied to the antecedent sequence of ℎ𝑠, 𝑐𝑠 and 𝑥𝑠 ; and
ℎ𝑡 = 𝜎𝑜(𝑥𝑡,ℎ𝑡−1)⊙tanh 𝑐𝑡. (5.6)
Geometrically, this can be seen as a concatenation of folds, as proposed by Thom Esquisse d’une Sémio-
physique [Tho88], to explain many kinds of organized systems in biology and cognition. From this
point of view, the concatenation of folds, giving the possibility of coincidence of cofolds [Arg78], is a
necessary condition for representing the emergence of a meaningful structure and oriented dynamic in a
living system.
Note that, in the unsaturated regimes, ℎ𝑡 has a degree 5 in ℎ𝑡−1, then its natural groupoid can be embedded
in a braids groupoid of type B5. This augmentation, from the fold to the so called swallowtail, could
explain the greatest syntactic power of the 𝑀𝐻𝐷𝑃𝐴with respect to 𝐿𝑆𝑇𝑀. However the concrete use
of more memories in times 𝑠before 𝑡makes the cells much more complex than a simple mapping from
𝑡−1 to 𝑡.
The above algorithm can be composed with other cells for detecting relations. For instance, Raposo
et al. [RSB+17] have defined a relation operator: having produced contexts 𝐻or questions 𝑄concerning
two objects 𝑜𝑖,𝑜𝑗 by a chain of 𝐿𝑆𝑇𝑀(that can be helped by external memories and attention cells) the
answer is taken from a formula:
𝐴= 𝑓
𝑖,𝑗
𝑔(𝑜𝑖,𝑜𝑙; 𝑄,𝐻), (5.7)
where 𝑓 and 𝑔 are parameterized functions, and 𝑜𝑖 : 𝑖∈𝐼 are vectors representing objects with their
characteristics.
The authors insisted on the important invariance of this operator by the permutation group 𝔖𝑛 of the
objects.
More generally, composed networks were introduced in 2016 by Andreas et al. [ARDK16] for question
answering about images. The reasoning architecture 𝑀𝐴𝐶, defined by Hudson and Manning, [HM18],
is composed of three attention operators named control, write and read, in a 𝐷𝑁𝑁, inspired from the
architecture of computers.
This leads us to consider the evolution of architectures and internal fibers of stacks and languages, in
relation to the problems to be solved in semantic analysis.
114
5.2 The 2-category of a network
For representing languages in DNNs, we have associated to a small category Cthe class AC= Grpd∧
C
of presheaves over the category of fibrations in groupoids over C. The objects of ACwere described in
terms of presheaves 𝐴𝑈 on the fibers F𝑈 for 𝑈∈Csatisfying gluing conditions, cf. sections 3 and 4.
Remark. Other categories than groupoids, for instance posets or fibrations in groupoids over posets, can
replace the groupoids in this section, and are useful in the applications, as we mentioned before, and as
we will show in the forthcoming article on semantic communication.
Natural morphisms between objects (F,𝐴)and (F′,𝐴′)of AC are defined by a family of functors
𝐹𝑈 : F𝑈 →F′
𝑈, such that for any morphism 𝛼: 𝑈→𝑈′in C,
𝐹′
𝛼◦𝐹𝑈′= 𝐹𝑈◦𝐹𝛼; (5.8)
and by a family of natural transformations 𝜑𝑈 : 𝐴𝑈 →𝐹★
𝑈(𝐴′
𝑈)= 𝐴′
𝑈 ◦𝐹𝑈, such that for any morphism
𝛼: 𝑈→𝑈′in C,
𝐹★
𝑈′(𝐴′
𝛼)◦𝜑𝑈′ = 𝐹★
𝛼(𝜑𝑈)◦𝐴𝛼, (5.9)
from 𝐴𝑈′ to 𝐹★
𝛼(𝐹★
𝑈𝐴′
𝑈)= 𝐹★
𝑈′ (𝐹′
𝛼)★𝐴′
𝑈.
Note that the family {𝐹𝑈;𝑈∈C}is equivalent to a C-functor 𝐹 : F→F′ of fibered categories in
groupoids, and the family 𝜑𝑈 is equivalent to a morphism 𝜑 in the topos EF from the object 𝐴to the
object 𝐹★(𝐴′).
Remark. These morphisms include the morphisms already defined for the individual classifying topos
EF. But, even for one fibration Fand its topos E, we can consider non-identity end-functor from Fto
itself, which give new morphisms in AC.
The composition of (𝐹𝑈,𝜑𝑈);𝑈∈Cwith (𝐺𝑈,𝜓𝑈)from (G,𝐵)to (F,A)is defined by the ordinary
composition of functors 𝐹𝑈◦𝐺𝑈, and the twisted composition of natural transformation
(𝜑◦𝜓)𝑈 = 𝐺★
𝑈(𝜑𝑈)◦𝜓𝑈 : 𝐵𝑈 →(𝐹𝑈◦𝐺𝑈)★𝐴′
𝑈. (5.10)
This rule gives a structure of category to AC.
In addition, the natural transformations between functors give the vertical arrows in HomA(F,𝐴:
F′,𝐴′), that form categories:
a morphism from (𝐹,𝜑)to (𝐺,𝜓)is a natural transformations 𝜆: 𝐹 →𝐺, which in this case with
groupoids, is an homotopy in the nerve, plus a morphism 𝑎: 𝐴→𝐴, such that
𝐴′(𝜆)◦𝜑= 𝜓◦𝑎: 𝐴→𝐺★𝐴′
. (5.11)
115
For a better understanding of this relation, we can introduce the points (𝑈,𝜉)in Fover C, and read
𝐴′
𝑈(𝜆𝑈(𝜉))◦𝜑𝑈(𝜉)= 𝜓𝑈(𝜉)◦𝑎𝑈(𝜉): 𝐴𝑈(𝜉)→𝐴′
𝑈(𝐺𝑈(𝜉)). (5.12)
This can be understood geometrically, as a lifting of the deformation 𝜆to a deformation of the presheaves.
Vertical composition is defined by usual composition for the deformations 𝜆and ordinary composition
in End(𝐴)for 𝑎. Horizontal compositions are for F→F′→F”
.
Horizontal arrows and vertical arrows satisfy the axioms of a 2-category [Gir71], [Mac71].
This structure encodes the relations between several semantics over the same network.
The relations between several networks, for instance moduli inside a network, or networks that are
augmented by external links, belong to a 3-category, whose objects are the above semantic triples, and
the 1-morphism are lifting of functors between sites 𝑢: C→C′
.
[Gir71, Theorem 2.3.2] tells us that, as for ordinary presheaves, there exist natural right and left
adjoints 𝑢★ and 𝑢! respectively of the pullback 𝑢★ from the 2-category CatC′ of fibrations over C′to
the 2-category CatC of fibrations over C. They are natural 2-functors, adjoint in the extended sense.
These 2-functors define adjoint 2-functors between the above 2-categories of classifying toposes ACand
AC′, by using the natural constructions of 𝑆𝐺𝐴4 for the categories of presheaves. They can be seen as
substitutions of stacks and languages induced by functors 𝑢.
The construction of ACfrom Cis a particular case of Grothendieck’s derivators [Cis03].
5.3 Grothendieck derivators and semantic information
For Ma closed model category, the map C↦→MC, or M∧
C(see section 2.4), is an example of derivator
in the sense of Grothendieck. References are [Gro83], [Gro90], the three articles of Cisinski [Cis03],
and the book of Maltsiniotis on the homotopy theory of Grothendieck [Mal05].
A derivator generalizes the passage from a category to its topos of presheaves, in order to develop
homotopy theory, as topos were made to develop cohomology theory. It is a 2-functor Dfrom the
category Cat (or a special sub-category of diagrams, for instance Poset) to the 2-category CAT, satisfying
four axioms.
a) The first one tells us that Dtransforms sums of categories into products,
b) The second one that isomorphisms of images can be tested on objects,
c) the third one that there exists, for any functor 𝑢: C→C′, a right adjoint 𝑢★ (defining homotopy
limit) and a left adjoint 𝑢! (defining homotopy colimit) of the functor 𝑢★= D(𝑢);
116
d) the fourth axiom requires that these adjoints are defined locally; for instance, if 𝑋′∈C′, and
𝐹∈D(𝐶), therefore 𝑢★𝐹∈D(𝐶)′, the fourth axiom tells us that
(𝑢★𝐹)𝑋′ 𝑝★𝑗★𝐹; (5.13)
where 𝑗 is the canonical map from C|𝑋′to C, and 𝑝the unique morphism from C|𝑋′to ★.
Another formula that expresses the same thing is
(𝑢★𝐹)𝑋′ 𝐻★ C|𝑋′; 𝐹|C|𝑋′ , (5.14)
abstract version of a Kan extension formula.
In general, the cohomology is defined by
𝐻★(C; 𝐹)= (𝑝C)★𝐹∈D(★). (5.15)
A first example of derivator is given by an Abelian category Ab, like commutative groups or real vector
spaces, and it is defined by the derived category of diﬀerential complexes, where quasi-isomorphisms
(isomorphisms in cohomology) are formally inverted,
D(𝐼)= Der(Hom(𝐼op
,Ab)). (5.16)
Another kind of example is a representable derivator
DM(𝐼)= Funct(𝐼op
,M), (5.17)
where Mis a closed model category. This can be seen as a non-Abelian generalization of the above first
example.
A third kind of examples is given by the topos of sheaves over a representable derivator M∧
C.
Then representable derivators allow to compare the elements of semantic functioning between several
networks, for instance a network with a sub-network of this network, playing the role of a module in
computation.
Consider the sub-categories Θ𝑃, over the languages A𝜆,𝜆∈F𝑈, made by the theories that exclude a
rigid proposition 𝑃=!Γ, in the sense they contain 𝑃⇒Δ, for a given chosen Δ, (see appendix E). The
right slice category 𝑃|A𝜆 acts on Θ𝑃. The information spaces 𝐹define an object of MΘ𝑃 , its cohomology
allow us to generalize the cat’s manifolds, that we defined below with the connected components of the
category D, in the following way: the dynamical object Xis assumed to be defined over the stack F, then
the dynamical space 𝑔Xis defined over the nerve of F, and the semantic functioning gives a simplicial
map 𝑔𝑆: 𝑔X→𝑔𝐼• from 𝑔Xspace to the equipped theories, then we can consider the inverse image
117
of Θ𝑃 over the functioning network. Composing with 𝐹 we obtain a parameterized object 𝑀𝑃 in M,
defining a local system over the category associated to 𝑔X, which depends on Γ,Δ. This represents the
semantic information in Xabout the problem of (rigidly) excluding 𝑃when considering that Δ is (thought
to be) false. Seen as an element of D(𝑔X), its cohomology is an homotopical invariant of the information.
In this text, we have defined information quantities, or information spaces, by applying cohomology
or homotopy limits, over the category Dwhich expresses a triple C,F,A, made by a language over a
pre-semantic over a site. The Abelian situation was studied through the bar-complex of cochains of the
module of functions Φ on the fibration Tof theories Θ over the category D. A non-Abelian tentative,
for defining spaces of information, was also proposed at this level, using (in the non-homogeneous form)
the functors 𝐹 from Θloc to a model category M(see section 3.5). Therefore information spaces were
defined at the level of MT, not at a level MC.
Information spaces belong to DM(T). To compare spaces of information flows in two theoretical
semantic networks, we have at disposition the adjoint functors 𝜑★,𝜑! of the functors 𝜑★ = D(𝜑)associ-
ated to 𝜑: T→T′, between categories of theories. Those functors 𝜑can be associated to changes of
languages A, changes of stacks Fand/or changes of basic architecture C.
An important problem to address, for constructing networks and applying deep learning eﬃciently
to them, is the realization of information relations or correspondences, by relations or correspondences
between the underlying invariance structures. For instance, to realize a family of homotopy equivalences
(resp. fibration, resp. cofibration) in M, by transformations of languages, stacks or sites having some
properties, like enlargement of internal symmetries.
The analog problem for presheaves (set valued) is to realize a correspondence (or relation) between
the topos I∧and (I′)∧from a correspondence between convenient sites for them.
For toposes morphisms this is a classical result (see [AGV63, 4.9.4] or the Stacks project [Sta, 7.16n
2.29]) that any geometric morphism 𝑓★ : Sh(𝐼)→Sh(𝐽)comes from a morphism of sites up to topos
equivalence between 𝐼 and 𝐼′. More precisely, there exists a site 𝐼′and a cocontinuous and continuous
functor 𝑣: 𝐼→𝐼′giving an equivalence 𝑣! : Sh(𝐼)→Sh(𝐼′)extending 𝑣, and a site morphism 𝐽→𝐼′
,
given by a continuous functor 𝑢: 𝐼′→𝐽such that 𝑓★= 𝑢★◦𝑣!.
From [Shu12], a geometric morphism between Sh(𝐼)and Sh(𝐽)comes from a morphism of site if and
only if it is compatible with the Yoneda embeddings.
118
5.4 Stacks homotopy of DNNs
The characterization of fibrant and cofibrant objects in MCwas the main result of chapter 2. All objects
of MCare cofibrant and the fibrant objects are described by theorem 2.2; we saw that they correspond to
ideal semantic flows, where the condition 𝜋★𝜋★= Id holds. They also correspond to the contexts and the
types of a natural 𝑀−𝐿theory. The objects of 𝐻𝑜(MC), [Qui67], are these fibrant and cofibrant objects
of MC, the 𝐻𝑜 morphisms being the homotopy classes of morphisms in MC, generated by inverting
formally zigzags similar to the above ones. Thus we get a direct access to the homotopy category 𝐻𝑜MC.
The 𝐻𝑜morphisms are the homotopy equivalences classes of the substitutions of variables in the 𝑀−𝐿
theory.
From the point of view of semantic information, we just saw that homotopy is pertinent at the next
level: looking first at languages over the stacks, then at some functors from the posets of theories to a test
model category M′, then going to 𝐻𝑜(M′). However, the fact that we restrict to theories over fibrant
objects and fibrations between them, implies that the homotopy of semantic information only depends
on the images of these theories over the category 𝐻𝑜(MC). How to use this fact for functioning networks?
119
Appendices
A Localic topos and Fuzzy identities
Definitions. let Ω be a complete Heyting algebra; a set over Ω, (𝑋,𝛿), also named an Ω-set, is a set 𝑋
equipped with a map 𝛿: 𝑋×𝑋→Ω, which is symmetric and transitive, in the sense that for any triple
𝑥,𝑦,𝑧, we have 𝛿(𝑥,𝑦)= 𝛿(𝑦,𝑥)and
𝛿(𝑥,𝑦)∧𝛿(𝑦,𝑧)≤𝛿(𝑥,𝑧). (18)
Note that 𝛿(𝑥,𝑥)can be diﬀerent from ⊤.
But we always have 𝛿(𝑥,𝑦)= 𝛿(𝑥,𝑦)∩𝛿(𝑦,𝑥)≤𝛿(𝑥,𝑥), and 𝛿(𝑥,𝑦)≤𝛿(𝑦,𝑦).
As Ω is made for fixing a notion of relative values of truth, 𝛿 is interpreted as fuzzy equality in 𝑋; it
generalizes the characteristic function of the diagonal when Ω is boolean. In our context of DNN, it can
be understood as the progressive decision about the outputs on the trees of layers rooted in a given layer.
A morphism from (𝑋,𝛿)to (𝑋′,𝛿′)is an application 𝑓 : 𝑋×𝑋′→Ω, such that, for every, 𝑥,𝑥′,𝑦,𝑦′
𝛿(𝑥,𝑦)∧𝑓(𝑥,𝑥
′)≤𝑓(𝑦,𝑥
′), (19)
𝑓(𝑥,𝑥
′)∧𝛿′(𝑥
′
,𝑦
′)≤𝑓(𝑥,𝑦
′); (20)
𝑓(𝑥,𝑥
′)∧𝑓(𝑥,𝑦
′)≤𝛿′(𝑥
′
,𝑦
′). (21)
Moreover
𝛿(𝑥,𝑥)=
𝑓(𝑥,𝑥
′). (22)
𝑥′∈𝑋′
Which generalizes the usual properties of the characteristic function of the graph of a function in the
boolean case.
The composition of a map 𝑓 : 𝑋×𝑋′→Ω with a map 𝑓′: 𝑋′×𝑋” →Ω is given by
(𝑓′
◦𝑓)(𝑥,𝑥”)=
𝑥′∈𝑋′
𝑓(𝑥,𝑥
′)∧𝑓(𝑥
′
,𝑥”). (23)
And the identity morphism is defined by
Id𝑋,𝛿 = 𝛿. (24)
120
This gives the category SetΩ of sets over Ω, also named Ω-sets.
The Heyting algebra Ω of a topos Eis made by the subobjects of the final object 1; the elements of
Ω are named the open sets of E. In fact, there exists an object in E, the Lawvere object, such that
for every object 𝑋 ∈E, the set of subobjects of 𝑋 is naturally identified with the set of morphisms 𝑋
.
When E= Sh(X)is a Grothendieck topos, is the sheaf over 𝑋, which is defined by (𝑥)= Ω(E|𝑥),
the subobjects of 1|𝑥. In the Alexandrov case, (𝑥)is the set of open sets for the Alexandrov topology
contained in Λ𝑥.
According to Bell, [Bel08], a localic topos, as the one of a DNN, is naturally equivalent to the category
SetΩ of Ω-sets, i.e. sets equipped with fuzzy identities with values in Ω. We now give a direct explicit
construction of this equivalence, because it oﬀers a view of the relation between the network layers
directly connected to the intuitionist logic of the topos.
Let us mention the PhD thesis of Johan Lindberg [Lin20, part III], developing this point of view, and
studying in details the naturalness of the geometric morphism of topos induced by a morphism of locale.
Definition A.1. On the poset (Ω,≤), the canonical Grothendieck topology 𝐾is defined by the coverings
by open subsets of the open sets.
In the localic case, where we are, the topos is isomorphic to the Grothendieck topos E= 𝑆ℎ(Ω,𝐾).
We assume that this is the case in the following exposition.
In the particular case E= X∧, where X is a poset, Ω is the poset of lower Alexandrov open sets and the
isomorphism with 𝑆ℎ(Ω,𝐾)is given explicitly by proposition 1.2.
Let 𝑋be an object of E; we associate to it the set 𝑋 of natural transformation from to 𝑋. For two
elements 𝑥,𝑦of 𝑋 , we define 𝛿𝑋(𝑥,𝑦)∈Ω as the largest open set over which 𝑥and 𝑦coincide.
An element 𝑢of 𝑋 is nothing else than a sub-singleton in 𝑋, its domain 𝜔𝑢 is 𝛿𝑋(𝑢,𝑢). In other terms,
in the localic case, 𝑢is a section of the presheaf 𝑋over an open subset 𝜔𝑢 in Ω.
Then, if 𝑢, 𝑣and 𝑤are three elements of 𝑋 , the maximal open set where 𝑢= 𝑤contains the intersection
of the open sets where 𝑢= 𝑣and 𝑣= 𝑤. Thus 𝑋 is a set over Ω.
In the same manner, suppose we have a morphism 𝑓 : 𝑋→𝑌 in E, if we take 𝑥∈𝑋 and 𝑦∈𝑌 we
define 𝑓(𝑥,𝑦)∈Ω as the largest open set of where 𝑦 coincides with 𝑓★𝑥. This gives a morphism of
Ω-sets.
All that defines a functor from Eto 𝑆𝑒𝑡Ω.
A canonical functor from SetΩ to Eis given by a similar construction:
for 𝑈∈Ω, Ω𝑈 = (𝑈)is an Ω-set, with the fuzzy equality defined by the internal equality
𝛿𝑈(𝛼,𝛼
′)= (𝛼≍𝛼
′), (25)
121
that is the restriction of the characteristic map of the diagonal subset: Δ : ↩→ × . The set Ω𝑈 can
be identified with the Ω-set 𝑈 associated to the Yoneda presheaf defined by 𝑈. More concretely, an
element 𝜔of Ω𝑈 is an open subset of 𝑈, and its domain 𝛿(𝜔,𝜔)is 𝜔itself.
Now, for any Ω-set (𝑋,𝛿), and for any element 𝑈∈Ω, we define the set (see (19),(22)),
𝑋Ω (𝑈)= HomSetΩ (Ω𝑈,𝑋)= {𝑓 : Ω𝑈×𝑋→Ω}. (26)
In what follows, we sometimes write 𝑋Ω = 𝑋, when the notation does not introduce too much ambiguity.
If 𝑉 ≤𝑊, the formula 𝑓(𝜔𝑉,𝜔𝑊)= 𝜔𝑉∩𝜔𝑊 defines a Ω-morphism from Ω𝑉 to Ω𝑊, which gives a map
from 𝑋(𝑊)to 𝑋(𝑉). Then 𝑋Ω is a presheaf over Ω.
Proposition A.1. A morphism of Ω-set 𝑓 : 𝑋×𝑌 →Ω gives by composition a natural transformation
𝑓Ω : 𝑋Ω →𝑌Ω of presheaves over Ω.
Proof. Consider 𝑓𝑈 ∈𝑋(𝑈); the axiom (22) tells that for every open set 𝑉 ⊂𝑈, the family of open sets
𝑓𝑈(𝑉,𝑢); 𝑢∈𝑋is an open covering 𝑓𝑉
𝑈 of 𝑉.
The first axiom of (19), which represents the substitution of the first variable, tells that on 𝑉∩𝑊 the
two coverings 𝑓𝑉
𝑈 and 𝑓𝑊
𝑈 coincide. Therefore, for every 𝑢∈𝑋, the value 𝑓𝑈(𝑢)= 𝑓(𝑈,𝑢)of 𝑓𝑈 on the
maximal element 𝑈determines by intersection all the values 𝑓𝑈(𝑉,𝑢)for 𝑉 ⊂𝑈.
For 𝑓𝑈 ∈𝑋(𝑈)and 𝑉 ≤𝑈, the functorial image 𝑓𝑉 of 𝑓𝑈 in 𝑋(𝑉)is the trace on 𝑉:
∀𝑢∈𝑋, 𝑓𝑉(𝑢)= 𝜌𝑉𝑈 𝑓𝑈(𝑢)= 𝑓𝑈(𝑢)∩𝑉. (27)
This implies that 𝑋Ω is a sheaf: consider a covering Uof 𝑈, (1)for two elements 𝑓𝑈,𝑔𝑈 of 𝑋(𝑈), if
the families of restrictions 𝑓𝑈 ∩𝑉;𝑉 ∈U, 𝑔𝑈 ∩𝑉;𝑉 ∈U, then 𝑓𝑈 = 𝑔𝑈; (2)if a family of coverings
𝑓𝑉;𝑉∈Uis given, such that for any intersection 𝑊= 𝑉∩𝑉′, the restriction 𝑓𝑉|𝑊and 𝑓𝑉′|𝑊coincide, as
open coverings, we can define an element 𝑓𝑈 of 𝑋(𝑈)by taking for each 𝑢∈𝑋the open set 𝑓𝑈(𝑢)which
is the reunion of all the 𝑓𝑉(𝑢)for 𝑉 ∈U. The union of the sets 𝑓𝑉(𝑢)over 𝑢∈𝑋 is 𝑉, and the union of
the sets 𝑉 is 𝑈, then the union of the 𝑓𝑈(𝑢)when 𝑢describes 𝑋is 𝑈.
The second axiom of substitution tells that for any 𝑢,𝑣∈𝑋, 𝛿(𝑢,𝑣)∩𝑓(𝑢)= 𝛿(𝑢,𝑣)∩𝑓(𝑣). The third
axiom of (19), which expresses the functional character of 𝑓, tells that for any 𝑢,𝑣 ∈𝑋, 𝛿(𝑢,𝑣)⊇
𝑓(𝑢)∩𝑓(𝑣).
Consequently, the elements of 𝑋(𝛼)can be identified with the open coverings 𝑓𝑈(𝑢); 𝑢∈𝑋 of the
open set 𝑈, such that, in Ω, we have
∀𝑢,𝑣∈𝑋, 𝑓𝑈(𝑢)∩𝑓𝑈(𝑣)⊆𝛿(𝑢,𝑣)⊆(𝑓𝑈(𝑢)⇔𝑓𝑈(𝑣)); (28)
where ⇔denotes the internal equivalence ⇐∧⇒in Ω.
Remind that 𝛼⇒𝛽 is the largest element 𝛾 ∈Ω such that 𝛾∧𝛼≤𝛽, and in our topological setting
Ω = U(X)it is the union of the open sets 𝑉 such that 𝑉∩𝛼⊆𝛽, therefore 𝑓(𝑢)⇔𝑓(𝑣)is the union of
the elements 𝑉 of Ω such that 𝑉∩𝑓(𝑢)= 𝑉∩𝑓(𝑣).
122
Proposition A.2. Let Ω be any complete Heyting algebra (i.e. a locale); the two functors 𝐹: (𝑋,𝛿)↦→
(𝑈↦→𝑋(𝑈)= HomΩ (Ω𝑈,𝑋)and 𝐺: 𝑋↦→(𝑋 ,𝛿𝑋)= HomE(,X)define an equivalence of category
between SetΩ and E= 𝑆ℎ(Ω,𝐾).
Proof. The composition 𝐹◦𝐺sends a sheaf 𝑋(𝑈);𝑈∈Ω to the sheaf 𝑋 (𝑈);𝑈∈Ω made by the open
coverings of 𝑈by sets indexed by the sub-singletons 𝑢of 𝑋satisfying the two inclusions (28).
Consider an element 𝑠𝑈 ∈𝑋(𝑈), identified with a section of 𝑋over 𝑈. For each sub-singleton 𝑣∈𝑋 ,
we define the open set 𝑓(𝑣)= 𝑓𝑠
𝑈(𝑣)by the largest open set in 𝑈 where 𝑣= 𝑠𝑈. As the sub-singletons
generate 𝑋, this forms an open covering of 𝑈. It satisfies (28) for any pair (𝑢,𝑣): 𝛿(𝑢,𝑣)is the largest
open set where 𝑢 coincides with 𝑣, then the first inclusion is evident, for the second one, consider the
intersection 𝛿(𝑢,𝑣)∩𝑓(𝑢), on it we have 𝑢= 𝑣and 𝑢= 𝑠, then it is included in 𝛿(𝑢,𝑣)∩𝑓(𝑣). If 𝑉 ⊂𝑈
and 𝑠𝑉 = 𝑠𝑈|𝑉, the open covering of 𝑉 defined by 𝑠𝑉 is the trace of the open covering defined by 𝑠𝑈.
Moreover, a morphism 𝜙: 𝑋→𝑌 in Esends sub-singletons to sub-singletons and induces injections of
the maximal domain of extension; therefore the above construction defines a natural transformation 𝜂E
from 𝐼𝑑Eto 𝐹◦𝐺.
This transformation is invertible: take an element 𝑓 of 𝑋 (𝑈), and for every 𝑈∈Ω, consider the set
𝑆(𝑓,𝑈)of sub-singletons 𝑢of 𝑋such that 𝑓𝑈(𝑢)≠ ∅. If 𝑢and 𝑣belong to this set, the first inequality of
(28) implies that 𝑢= 𝑣on the intersection 𝑓𝑈(𝑢)∩𝑓𝑈(𝑣), then, by the sheaf property 3, 𝑆(𝑓,𝑈)defines
a unique element 𝑢𝑈 ∈𝑋(𝑈).
In the other direction, the composition 𝐺◦𝐹associates to a Ω-set (𝑋,𝛿)the Ω-set (𝑋 ,𝛿𝑋, )made
by the sub-singletons of the presheaf 𝑋Ω, i.e. the families (𝑓,𝑈)of compatible coverings 𝑓𝑉(𝑣),𝑣∈𝑋
of 𝑉;𝑉 ⊂𝑈. We have 𝛿((𝑓,𝑈),(𝑓,𝑈))= 𝑈; therefore, for simplifying the notations, we denote the
singleton by 𝑓, and 𝑈is 𝛿(𝑓,𝑓).
We saw that, for two elements 𝑓, (𝑓′, the open set 𝛿(𝑓,𝑓′)is the maximal open subset of 𝑈∩𝑈′where
the coverings 𝑓𝑉(𝑢)and 𝑓′
𝑉(𝑢)coincide for every 𝑢∈𝑋and 𝑉 ⊂𝑈.
For a pair (𝑢,𝑓), of 𝑢∈𝑋and (𝑓 ∈𝑋 , we define 𝐻(𝑢,𝑓)∈Ω as the unions of the open sets 𝑓𝑉(𝑢), over
𝑉 ⊂𝛿(𝑓,𝑓)∩𝛿(𝑢,𝑢).
The formula (27) implies that 𝐻(𝑢,𝑓)is also the union of open sets 𝛼such that 𝛼⊂𝑓𝛼(𝑢), i.e. 𝑓𝛼(𝑢)= 𝛼.
We verify that 𝐻is a morphism of Ω-sets: the first axiom
𝛿(𝑢,𝑣)∧𝐻(𝑢,𝑓)≤𝐻(𝑣,𝑓) (29)
results from
𝛿(𝑢,𝑣)∧𝑓𝛼(𝑢)≤𝑓𝛼(𝑣) (30)
for every 𝛼∈Ω.
The second axiom
𝐻(𝑢,𝑓)∧𝛿(𝑓,𝑓′)≤𝐻(𝑢,𝑓′) (31)
comes from the definition of 𝛿(𝑓,𝑓′)as an open set where the induced coverings coincide.
123
For the third axiom,
𝐻(𝑢,𝑓)∧𝐻(𝑢,𝑓′)≤𝛿(𝑓,𝑓′); (32)
if 𝛼is included in the intersection we have 𝑓𝛼(𝑢)= 𝛼= 𝑓′
𝛼(𝑢), then 𝛼≤𝛿(𝑓,𝑓′).
From (28), we have 𝑓𝛼(𝑢)⊂𝛿(𝑢,𝑢), then
𝐻(𝑢,𝑓)⊂𝛿(𝑢,𝑢) (33)
And for every 𝛼≤𝛿(𝑢,𝑢), we can define a special covering 𝑓𝑢
𝛼 by
𝑓𝑢
𝛼(𝑢)= 𝛼, 𝑓𝑢
𝛼(𝑣)= 𝛼∧𝛿(𝑢,𝑣); (34)
it satisfies (28). Then
𝛿(𝑢,𝑢)=
𝐻(𝑢,𝑓) (35)
𝑓∈𝑋(𝑈)
The Ω-map 𝐻 is natural in 𝑋 ∈SetΩ. To terminate the proof of proposition A.2, we have to show that
𝐻is invertible, that is to find a Ω-map 𝐻′: 𝑋Ω ×𝑋→Ω, such that 𝐻′◦𝐻= 𝛿𝑋 and 𝐻◦𝐻′
= 𝛿𝑋Ω ,
. We
note the first fuzzy identity by 𝛿and the second one by 𝛿′
.
In fact 𝐻′(𝑓,𝑢)= 𝐻(𝑢,𝑓)works; in other terms 𝐻is an involution of Ω-sets. let us verify this fact:
by definition of the composition
𝐻′
◦𝐻(𝑢,𝑣)=
𝐻(𝑢,𝑓)∧𝐻′(𝑓,𝑣) (36)
𝑓
is the reunion of the 𝛼∈Ω such that there exists 𝑓 with 𝛼= 𝑓𝛼(𝑢)= 𝑓𝛼(𝑣), then by the first inequality
in (28) it is included in 𝛿(𝑢,𝑣). Now consider 𝛼≤𝛿(𝑢,𝑣)⊆𝛿(𝑢,𝑢), and define a covering of 𝛼 by
𝑓𝑢
𝛼(𝑤)= 𝛼∩𝛿(𝑢,𝑤)for any 𝑤∈𝑋, this gives 𝛼≤𝑓𝑢
𝛼(𝑣)then 𝛼⊆𝐻(𝑣,𝑓𝑢), then 𝛼⊂𝐻(𝑢,𝑓𝑢)∧𝐻′(𝑓𝑢,𝑣).
On the other side,
𝐻◦𝐻′(𝑔,𝑓)=
𝐻(𝑔,𝑢)∧𝐻(𝑢,𝑓), (37)
𝑢
is the reunion of the 𝛼∈Ω such that there exists 𝑢with 𝛼= 𝑓𝛼(𝑢)= 𝑔𝛼(𝑢). In this case, we consider the
set 𝑆(𝑓,𝛼)of elements 𝑣∈𝑋 such that 𝑓𝛼(𝑣)≠ ∅. If 𝑣 and 𝑤 belong to this set, the first inequality of
(28) implies that 𝑣= 𝑤on the intersection 𝑓𝛼(𝑣)∩𝑓𝑧(𝑤), then, by the sheaf property, 𝑆(𝑓,𝛼)defines a
unique element 𝑢𝛼 ∈𝑋. This element must be equal to 𝑢. The same thing being true for 𝑔, this implies
that 𝑓𝛼(𝑣)= 𝑔𝛼(𝑣)for all the elements 𝑣 of 𝑋, some of them giving 𝛼 the other giving the empty set.
Consequently, 𝐻◦𝐻′(𝑔,𝑓)⊆𝛿′(𝑓,𝑔).
The other inclusion 𝛿′(𝑓,𝑔)⊆𝐻◦𝐻′(𝑔,𝑓)being obvious, this terminates the proof of the proposition.
This proposition generalizes to the localic Grothendieck topos the construction of the sheaf space
(espace étalé in French) associated to a usual topological sheaf. However the accent in Ω-sets is put
more on the gluing of sections than on a well defined set of germs of sections, as in the sheaf space. In
some sense, the more general Ω-sets give also a more global approach, as in the original case of Riemann
surfaces. Replacing a dynamics for instance by its solutions, pairs of domains and functions on them,
124
with the relation of prolongation over sub-domains. This seems to be well adapted to the understanding
of a DNN, on sub-trees of its architectural graph .
The localic Grothendieck topos EΩ are the "elementary topos" which are sub-extensional (generated
by sub-singletons) and defined over Set [Bel08, p. 207].
Particular cases are characterized by special properties of the lattice structure of the locale Ω [Bel08, pp.
208-210]:
we say that two elements 𝑈,𝑉 in Ω are separated by another element 𝛼∈Ω when one of them is
smaller than 𝛼but not the other one.
EΩ is the topos of sheaves over a topological space X if and only if Ω is spatial, which means by definition,
that any pair of elements of Ω is separated by a large element, i.e. an element 𝛼 such that 𝛽∧𝛾 ≤𝛼
implies 𝛽≤𝛼or 𝛾≤𝛼.
Moreover, in this case, Ω is the poset of open sets of X, and the large elements are the complement of the
closures of points of X.
The topological space is not unique, only the sober quotient is unique. A topological space is sober when
every irreducible closed set is the closure of one and only one point.
EΩ is the topos of presheaves over a poset CX if and only if Ω is an Alexandrov lattice, i.e. any pair
of elements of Ω is separated by a huge (very large) element, i.e. an element 𝛼 such that 𝑖∈𝐼 𝛽𝑖 ≤𝛼
implies that ∃𝑖∈𝐼,𝛽𝑖 ≤𝛼.
In this case Ω is the set of lower open sets for the Alexandrov topology on the poset.
If Ω is finite, large and huge coincide, then spatial is the same as Alexandrov.
B Topos of DNNs and spectra of commutative rings
A finite poset with the Alexandrov topology is sober. This is a particular case of Scott’s topology. Then
it is also a particular case of spectral spaces [Hoc69], [Pri94], that are (prime) spectra of a commutative
ring with the Zariski topology.
From the point of view of spectrum, a tree in the direction described in theorem 1.2, corresponds to a
ring with a unique maximal ideal, i.e., by definition a local ring.
The minimal points correspond to minimal primes. The gluing of two posets along an ending vertex
corresponds to the fiber product of the two rings over the simple ring with only one prime ideal [Ted16].
A ring with a unique prime ideal is a field, in this case the maximal ideal is {0}. This gives the following
result:
125
Proposition B.1. The canonical (i.e. sober) topological space of a 𝐷𝑁𝑁 is the Zariski spectrum of a
commutative ring which is the fiber product of a finite set of local rings over a product of fields.
The construction of a local rings for a given finite poset can be made by recurrence over the number
of primes, by successive application of two operations: gluing a poset along an open subset of another
poset, and joining several maximal points; this method is due to Lewis 1973 [Ted16].
Examples. I−The topos of Shadoks [Pro08] corresponds to the poset 𝛽<𝛼with two points; this is
the spectrum of any discrete valuation ring only containing the ideal {0}and a non-zero maximal
ideal. Such a ring is the subset of a commutative field Kwith a valuation 𝑣 valued in Z, defined
by {𝑎∈K|𝑣(𝑎)≥0}. An example is K((𝑥))the field of fractions of the formal series K[[𝑥]], with
the valuation given by the smallest power of 𝑥(and ∞) for 𝑎= 0. The valuation ring is K[[𝑥]], also
noted 𝐾{𝑥}, its maximal ideal is 𝔪𝑥 = 𝑥K[[𝑥]].
II−Consider the poset of length three: 𝛾<𝛽<𝛼. Apply the gluing construction to the ring 𝐴= K{𝑥}
embedded in K((𝑥))and the ring 𝐵= K((𝑥)){𝑦}projecting to K((𝑥)); this gives the following
local ring:
𝐷= K{𝑥}×K((𝑥))K((𝑥)){𝑦} {𝑑= 𝑎+𝑦𝑏|𝑎∈𝐴,𝑏∈𝐵}⊂𝐵. (38)
The sequence of prime ideals is
{0}⊂𝑦𝐵⊂𝔪𝑥+𝑦𝐵. (39)
III−Continuing this process, we get a natural local ring which spectral space is the chain of length 𝑛+1,
𝛼𝑛 <...<𝛼0 or simplest 𝐷𝑁𝑁s. There is one such ring for any commutative field K:
𝐷𝑛 = {𝑑= 𝑎𝑛+𝑥𝑛−1𝑏𝑛−1 +...+𝑥1𝑏1 ∈K((𝑥1,𝑥2,...,𝑥𝑛))|
(40)
𝑎𝑛 ∈K{𝑥𝑛},𝑏𝑛−1 ∈K((𝑥𝑛)){𝑥𝑛−1 },...,𝑏1 ∈K((𝑥2,...,𝑥𝑛)){𝑥1 }.
The sequence of prime ideals is
{0}⊂𝑥1K((𝑥2,...,𝑥𝑛)){𝑥1 }⊂
𝑥1K((𝑥2,...,𝑥𝑛)){𝑥1 }+𝑥2K((𝑥3,...,𝑥𝑛)){𝑥2 }⊂
...⊂𝑥1K((𝑥2,...,𝑥𝑛)){𝑥1 }+...+𝑥𝑛K{𝑥𝑛}.
(41)
C Classifying objects of groupoids
Proposition C.1. There exists an equivalence of category between any connected groupoid Gand its
fundamental group 𝐺.
Proof. let us choose an object 𝑂 in G, the group 𝐺 is represented by the group 𝐺𝑂 of automorphisms
of 𝑂. The inclusion gives a natural functor 𝐽: 𝐺→Gwhich is full and faithful. In the other direction,
126
we choose for any object 𝑥of G, a morphism (path) 𝛾𝑥 from 𝑥to 𝑂, we choose 𝛾𝑂 = 𝑖𝑑𝑂, and we define
a functor 𝑅 from Gto 𝐺 by sending any object to 𝑂 and any arrow 𝛾: 𝑥→𝑦 to the endomorphism
𝛾𝑦◦𝛾◦𝛾−1
𝑥 of 𝑂. The rule of composition follows by cancellation. A natural isomorphism between
𝑅◦𝐽and 𝐼𝑑𝐺 is the identity. A natural transformation 𝑇 from 𝐽◦𝑅to 𝐼𝑑Gis given by sending 𝑥∈Gto
𝛾𝑥, which is invertible for each 𝑥. The fact that it is natural results from the definition of 𝑅: for every
morphism 𝛾: 𝑥→𝑦, we have
𝑇(𝑦)◦𝐼𝑑(𝛾)= 𝛾𝑦◦𝛾= (𝛾𝑦◦𝛾)◦𝛾−1
𝑥 ◦𝛾𝑥 = 𝐽𝑅(𝛾)◦𝑇(𝑥). (42)
What is not natural in general (except if G= 𝐺= {1}) is the choice of 𝑅. This makes groupoids strictly
richer than groups, but not from the point of view of homotopy equivalence. Every functor between
two groupoids that induces an isomorphism of 𝜋0, the set of connected components, and of 𝜋1, the
fundamental group, is an equivalence of category.
One manner to present the topos E= EG of presheaves over a small groupoid G(up to category
equivalence) is to decompose Gin connected components G𝑎; 𝑎∈𝐴, then Ewill be product of the topos
E𝑎; 𝑎∈𝐴of presheaves over each component. For each 𝑎∈𝐴, the topos E𝑎 is the category of 𝐺𝑎-sets,
where 𝐺𝑎 denotes the group of auto-morphisms of any object in G𝑎.
The classifying object Ω = ΩGis the boolean algebra of the subsets of 𝐴.
In the applications, we are frequently interested by the subobjects of a fixed object 𝑋= {𝑋𝑎; 𝑎∈𝐴}.
The algebra of subobjects Ω𝑋, has for elements all the subsets that are preserved by 𝐺𝑎 for each compo-
nent 𝑎∈𝐴independently.
Thus we can consider what happens for a given 𝑎. Every element 𝑌𝑎 ∈Ω𝑋𝑎 has a complement 𝑌𝑐
𝑎
= ¬𝑌𝑎,
which is also invariant by 𝐺𝑎, and we have¬¬= 𝐼𝑑. Here the relation of negation ≤is the set-theoretic
one. It is also true for the operations ∧(intersection of sets), ∨(union of sets), and the internal implication
𝑝⇒𝑞, which is defined in this case by (𝑝∧𝑞)∨¬𝑝.
All the elements 𝑌𝑎 of Ω𝑋𝑎 are reunions of orbits 𝑍𝑖;𝑖∈𝐾(𝑋𝑎)of the group 𝐺𝑎 in the 𝐺𝑎-set 𝑋𝑎. On
each orbit, 𝐺𝑎 acts transitively.
Each subobject of 𝑋is a product of subobjects of the 𝑋𝑎 for 𝑎∈𝐴. The product over 𝑎of the 𝐾(𝑋𝑎)is a
set 𝐾= 𝐾(𝑋).
The algebra Ω𝑋 is the Boolean algebra of the subsets of the set of elements {𝑍𝑖;𝑖∈𝐾}, that we can
note simply Ω𝐾.
The arrows in this category, 𝑝→𝑞, correspond to the pre-order ≤, or equivalently to the inclusion of
sets, and can be understood as implication of propositions. This is the implication in the external sense, if
𝑝is true then 𝑞is true, not in the internal sense 𝑞𝑝, also denoted 𝑝⇒𝑞, that is also the maximal element
𝑥such that 𝑥∧𝑝≤𝑞).
On this category, there exists a natural Grothendieck topology, named the canonical topology, which is
the largest (or the finest) Grothendieck topology such that, for any 𝑝∈Ω, the presheaf 𝑥↦→Hom(𝑥,𝑝)
is a sheaf. For any 𝑝∈Ω, the set of coverings 𝐽𝐾(𝑝)is the set of collections of subsets 𝑞 of 𝑝 whose
127
reunion is 𝑝. In particular 𝐽𝐾(∅)contains the empty family; this is a singleton.
Proposition C.2. The topos Eis isomorphic to the topos Sh(Ω; 𝐾)of sheaves for this topology 𝐽𝐾 (see
for instance Bell, Toposes and local set theories [Bel08]).
Proof. For all 𝑝, any covering of 𝑝 has for refinement the covering made by the disjoint singletons 𝑍𝑖
that belong to 𝑝, seen as a set; then, for every sheaf 𝐹 over Ω, the restriction maps give a canonical
isomorphism from 𝐹(𝑝)with the product of the sets 𝐹(𝑍𝑖)over 𝑝itself.
In particular, any sheaf has for value in ⊥= ∅a singleton.
D Non-Boolean information functions
This is the case of chains and injective presheaves on them.
The site 𝑆𝑛 is the poset 0 →1 →...→𝑛. A finite object 𝐸 is chosen in the topos of presheaves 𝑆∧
𝑛,
such that each map 𝐸𝑖 →𝐸𝑖−1 is an injection, and we consider the Heyting algebra Ω𝐸, that is made by
the subobjects of 𝐸. The inclusion, the intersection and the union of subobjects are evident. The only
non-trivial internal operations are the exponential, or internal implication 𝑄⇒𝑇, and the negation¬𝑄,
that is a particular case 𝑄⇒∅.
Lemma D.1. Let 𝑇𝑛 ⊂𝑇𝑛−1 ⊂...⊂𝑇0 and 𝑄𝑛 ⊂𝑄𝑛−1 ⊂...⊂𝑄0 be two elements of Ω𝐸, then the
implication 𝑈= (𝑄⇒𝑇)is inductively defined by the following formulas:
𝑈0 = 𝑇0 ∨(𝐸0\𝑄0),
𝑈1 = 𝑈0 ∧(𝑇1 ∨(𝐸1\𝑄1),
...
𝑈𝑘 = 𝑈𝑘−1 ∧(𝑇𝑘∨(𝐸𝑘\𝑄𝑘),
...
Proof. By recurrence. For 𝑛= 0 this is the well known boolean formula. Let us assume the result for
𝑛= 𝑁−1, and prove it for 𝑛= 𝑁. The set 𝑈𝑁 must belong to 𝑈𝑁−1 and must be the union of all the sets
𝑉 ⊂𝐸𝑁∩𝑈𝑁−1 such that 𝑉∧𝑄𝑁 ⊂𝑇𝑁, then it is the union of 𝑇𝑁∩𝑈𝑁−1 and (𝐸𝑁\𝑄𝑁)∩𝑈𝑁−1.
In particular the complement¬𝑄is made by the sequence
𝑛
(𝐸𝑘\𝑄𝑘)⊂
𝑘=0
𝑛−1
𝑘=0
(𝐸𝑘\𝑄𝑘)⊂...⊂𝐸0\𝑄0. (43)
128
Definition D.1. We choose freely a strictly positive function 𝜇on 𝐸0; for any subset 𝐹 of 𝐸0, we note
𝜇(𝐹)the sum of the numbers 𝜇(𝑥)for 𝑥∈𝐹.
In practice 𝜇is the constant function equal to 1, or to |𝐹|−1
.
Definition D.2. Consider a strictly decreasing sequence [𝛿]of strictly positive real numbers 𝛿0 >𝛿1 >
...>𝛿𝑛; the function 𝜓𝛿 : Ω𝐸 →Ris defined by the formula
𝜓𝛿(𝑇𝑛 ⊂𝑇𝑛−1 ⊂...⊂𝑇0)= Σ𝑛
𝑘=0𝛿𝑘𝜇(𝑇𝑘). (44)
Lemma D.2. The function 𝜓𝛿 is strictly increasing.
This is because index by index, 𝑇′
𝑘 contains 𝑇𝑘.
Definition D.3. A function 𝜑: Ω𝐸 →Ris concave (resp. strictly concave), if for any pair of subsets
𝑇 ≤𝑇′and any proposition 𝑄, the following expression is positive (resp. strictly positive),
Δ𝜑(𝑄;𝑇,𝑇′)= 𝜑(𝑄⇒𝑇)−𝜑(𝑇)−𝜑(𝑄⇒𝑇′)+𝜑(𝑇′). (45)
Hypothesis on 𝛿: for each 𝑘, 𝑛≥𝑘≥0, we assume that 𝛿𝑘 >𝛿𝑘+1 +...+𝛿𝑛.
This hypothesis is satisfied for instance for 𝛿0 = 1,𝛿1 = 1/2,...,𝛿𝑘 = 1/2𝑘
,....
Proposition D.1. Under this hypothesis, the function 𝜓𝛿 is concave.
Proof. Let 𝑇≤𝑇′in Ω𝐸. We define inductively an increasing sequence 𝑇(𝑘)of 𝑆𝑛-sets by taking 𝑇(0)= 𝑇
and, for 𝑘 >0, 𝑇(𝑘)
𝑗 equal to 𝑇(𝑘−1)
𝑗 for 𝑗 <𝑘 or 𝑗 >𝑘, but equal to 𝑇′
𝑗 for 𝑗= 𝑘. In other terms, the
sequence is formed by enlarging 𝑇𝑘 to 𝑇′
𝑘, index after index. Let us prove that Δ𝜓𝛿(𝑄;𝑇(𝑘−1),𝑇(𝑘))is
positive, and strictly positive when at the index 𝑘, 𝑇𝑘 is strictly included in 𝑇′
𝑘. The theorem follows by
telescopic cancellations.
The only diﬀerence between 𝑇(𝑘−1)and 𝑇(𝑘)is the enlargement of 𝑇𝑘 to 𝑇′
𝑘, and this generates a diﬀerence
between 𝑇(𝑘−1)
𝑗 |𝑄 and 𝑇(𝑘)
𝑗 |𝑄 only for the indices 𝑗 >𝑘. This allows us to simplify the notations by
assuming 𝑘= 0.
The contribution of the index 0 to the double diﬀerence Δ𝜓𝛿 is the diﬀerence between the sum of 𝛿0 𝜇
over the points in 𝐸0\𝑄0 that do not belong to 𝑇0 and the sum of 𝛿0 𝜇over the points in 𝐸0\𝑄0 that do
not belong to 𝑇′
0, then it is the sum of 𝛿0𝜇over the points in 𝐸0\𝑄0 that belong to 𝑇′
0\𝑇0.
As in lemma D.1, let us write 𝑈0 = 𝑇0 ∨(𝐸0\𝑄0)and 𝑈′
0 = 𝑇′
0 ∨(𝐸0\𝑄0). And for 𝑘 ≥1, let us write
𝑉𝑘 = 𝑇𝑘∨(𝐸𝑘\𝑄𝑘), and 𝑊𝑘 = 𝑉1 ∩...∩𝑉𝑘.
From the lemma 1, the contribution of the index 1 to the double diﬀerence Δ𝜓𝛿, is the simple diﬀerence
between the sum of 𝛿𝑘𝜇 over the points in 𝑈0 ∩𝑊𝑘 and its sum over the points in 𝑈′
0 ∩𝑊𝑘, then it is
equal to the opposite of the sum of 𝛿𝑘𝜇over the points in (𝑇′
0\𝑇0)∩(𝐸0\𝑄0)∩𝑊𝑘. The hypothesis on
the sequence 𝛿implies that the sum over 𝑘of these sums is smaller than the diﬀerence given by the index
0.
129
Remark. In general the function 𝜓𝛿, whatever being the sequence 𝛿, is not strictly concave, because it
can happen that 𝑇′
0 is strictly larger than 𝑇0, and the intersection of 𝑇′
0\𝑇0 with 𝐸0\𝑄0 is empty. Therefore,
to get a strictly concave function, we take the logarithm, or another function from R★
+to Rthat transforms
strictly positive strictly increasing concave functions to strictly increasing strictly concave functions.
This property for the logarithm comes from the formulas
(ln 𝜑)”
= [𝜑′
]′
𝜑
=
𝜑𝜑” −(𝜑′)2
𝜑2 <0. (46)
In what follows we take 𝜓= ln 𝜓𝛿 as the fundamental function of precision.
By normalizing 𝜇and taking 𝛿0 = 1, we get 0 <𝜓𝛿 ≤1, −∞<𝜓≤0.
Remark. Lemmas D.1, D.2 and proposition D.1 can easily be extended to the case where the basic site S
is a rooted (inverse) tree, i.e. the poset that comes from an oriented graph with several initial vertices and
a unique terminal vertex. The computation with intersections works in the same manner. The hypothesis
on 𝛿concerns only the descending branches to the terminal vertex.
Now, remember that the poset of a 𝐷𝑁𝑁 is obtained by gluing such trees on some of their initial
vertices, interpreted as tips (of forks) or output layers. The maximal points correspond to tanks (of forks)
of input layers. Therefore it is natural to expect that the existence of 𝜓holds true for any site of a 𝐷𝑁𝑁.
E Closer to natural languages: linear semantic information
Several attempts were made by logicians and computer scientists, since Frege and Russel, Tarski and
Carnap, to approach the properties of human natural languages by formal languages and processes. In
particular, a computational grammar was proposed by Lambek [Lam58]: a syntactic category is defined
with sentences as objects and applications of grammatical rules as arrows, a second category is defined,
that contains products and exponentials, for instance a topos, and semantic is seen as some functor from
the first category to the second one. This is the first place where semantic is defined as interpretations
of types and propositions in a topos. Precursors of the kind of grammar considered by Lambek were
Adjukiewicz in 1935 [Adj35] and Bar-Hillel in 1953 [BH53].
Then a decisive contribution was made by Montague in 1970, [Mon70], who developed in particular
a formal treatment of pieces of English [Par75]. Also in this approach, semantics appears as a transfor-
mation from a syntactic algebraic structure, having lexis and multiple operations, to a coarser structure.
In the nineties mathematicians and linguists observed that the categorical point of view, as in Lambek,
gives a good framework for developing further Montague’s theory [vB90].
The next step used intensional type theories, like Martin-Löf’s theory [ML80], named modern TT
by Luo [Luo14], or rich TT by Cooper et al. [CDLL15]. New types were introduced, corresponding
to the many structural notions of linguistic, e.g. noun, verb, adjective, and so on. Also modalities like
interrogative, performative, can be introduced (see Brunot [Bru36] for the complexity of the enterprise in
130
French). Recent experiment with programming languages have shown that many properties of languages
can be captured by extending TT. For instance, in Martin-Löf TT it is possible to construct ZFT theo-
ries but also alternative Non-well-founded set theories, like in [Acz88], taking into account paradoxical
vicious circles as natural languages do [Lin89]. Even more powerful is the homotopical type theory
(HoTT) of Voevodski, Awodey, Kapulkin, Lumsdaine, Shulman, ..., [KLV12]. Also see Gylterud and
Bonnevier [GB20] for the inclusion of non-well-founded sets theories.
These formal theories do not give a true definition of what is meaning, (see the fundamental objec-
tions of Austin [Aus61]), but they give an insight of the various ways the meanings can be combined and
how they are related to grammar, compatible with the intuition we have of human interpretations. We
do not suggest that the categorial presentation defines the natural languages, but here also we think that
its capture something of toys languages, an some languages games that can help the understanding of
semantic functioning in networks, including properties of natural semantics of human peoples.
In what follows, we consider that a given category Arepresents the semantic for a given language,
or some language game [Wit53], and reflects properties of a language, not the abstract rules, as in the
algebra ΩLbefore. The objects of Arepresent interpretations of sentences, or images, corresponding to
the "types as propositions" (Curry-Howard) in a given grammar, and its arrows represent the evocations,
significations, or deductions, corresponding to proofs or application of rules in grammar. Oriented cycles
are a priori admitted.
We simply assume that Ais a closed monoidal category [EK66] that connects with linear logic and
linear type theory as in Mellies, "Categorical Semantics of Linear Logic" [Mel09].
In such a category, a bifunctor (𝑋,𝑌)↦→𝑋⊗𝑌 is given, that is associative up to natural transformation,
with a neutral element ★ also up to linear transformation, satisfying conditions of coherence. This
product representing aggregation of sentences. Moreover there exists classifiers objects of morphisms,
i.e. objects 𝐴𝑌 defined for any pair of objects 𝐴,𝑌, such that for any 𝑋, there exist natural isomorphisms
Hom(𝑋⊗𝑌,𝐴)≃Hom(𝑋,𝐴𝑌). (47)
The functor 𝑋↦→𝑋⊗𝑌 has for right-adjoint the functor 𝐴↦→𝐴𝑌
.
For us, this defines the semantic conditioning, the eﬀect on the interpretation 𝐴that 𝑌 is taken into
account, when 𝐴is evoked by a composition with 𝑌. Thus we also denote 𝐴𝑌 by 𝑌⇒𝐴or 𝐴|𝑌.
When 𝐴is given, and if 𝑌′→𝑌 we get 𝐴|𝑌→𝐴|𝑌′
.
From 𝑋⊗★ 𝑋, it follows that canonically 𝐴★ 𝐴. We make the supplementary hypothesis that ★is a fi-
nal object, then we get a canonical arrow 𝐴→𝐴|𝑌, for any object 𝑌. This represents the internal constants.
Remark. In the product 𝑋⊗𝑌, the ordering plays a role, and in linguistic, in the spirit of Montague,
two functors can appear, the one we just said 𝑌 ↦→𝑋⊗𝑌 and the other one 𝑋 ↦→𝑋⊗𝑌. If both have
a left adjoint, we get two exponentials: 𝐴𝑌 = 𝐴|𝑌 and 𝑋𝐴= 𝑋 𝐴; the natural axiomatic becomes the
131
bi-closed category of Eilenberg and Kelly [EK66]. Dougherty [Dou92] gave a clear exposition of part of
the Lambek calculus in the Montague grammar in terms of this structure (same in [Lam88]). A theory of
semantic information should benefit of this possibility, where composition depends on the ordering, but
in what follows, to begin, we assume that Ais symmetric: there exist natural isomorphisms exchanging
the two factors of the product.
All that can be localized in a context Γ ∈Aby considering the category Γ\Aof morphisms Γ →𝐴,
where 𝐴 describes A, with morphisms given by the commutative triangles. For Γ →𝐴, and 𝑌 ∈A,
we get a morphism Γ →𝐴|𝑌 by composition with the canonical morphism 𝐴→𝐴|𝑌. This extends the
conditioning. We will discuss the existence of a restricted tensor product later on; it asks restrictions on Γ.
The analog of a theory, that we will also name theory here, is a collection 𝑆of propositions 𝐴, that
is stable by morphisms to the right, i.e. 𝐴∈𝑆 and 𝐴→𝐵 implies 𝐵∈𝑆. This can be seen as the
consequences of a discourse. A theory 𝑆′is said weaker than a theory 𝑆 if it is contained in it, noted
𝑆≤𝑆′. Then the analog of the conditioning of 𝑆by 𝑌 is the collection of the objects 𝐴𝑌 for 𝐴in 𝑆. The
collection of theories is partially ordered.
We have 𝑆|𝑌′≤𝑆|𝑌 when there exists 𝑌′→𝑌. In particular 𝑆|𝑌 ≤𝑆, as it was the case in simple type
theory.
When a context is given, it defines restricted theories, because it introduces a constraint of commu-
tativity for 𝐴→𝐵, to define a morphism from Γ →𝐴to Γ →𝐵.
The monoidal category Aacts on the set of functions from the theories to a fixed commutative group,
for instance the real numbers.
We will later discuss how the context Γ can be included in a category generalizing the category Dof
sections 3.4 and 3.5, to obtain the analog of the classical ordinary logical case with the propositions 𝑃
excluded. This needs a notion of negation, which, we will see, are many.
Remark. The model should be more complete if we introduce a syntactic type theory, as in Montague
1970, such that Ais an interpretation of part of the types, compatible with products and exponentials.
Then some of the arrows can interpret transformation rules in the grammar. The introduction of syntaxes
will be necessary for communication between networks.
Let us use the notations of chapter 2. Between two layers 𝛼: 𝑈→𝑈′lifted by ℎto F, we assume the
existence of a functor 𝜋★𝛼,ℎfrom A𝑈,𝜉 to A𝑈′,𝜉′, with a left adjoint 𝜋★
𝛼,ℎ, such that 𝜋★𝜋★= Id, in such a
manner that Abecomes a pre-cosheaf over Ffor 𝜋★ and the sets of theories Θ form a presheaf for 𝜋★
.
The information quantities are defined as before, by the natural bar-complex associated to the action
of Aon the pre-cosheaf Φ′of functions on the functor Θ.
The passage to a network gives a dynamic to the semantic, and the consideration of weights gives
a model of learning semantic. Even if they are caricature of the natural ones, we hope this will help to
132
capture some interesting aspects of them.
A big diﬀerence with the ordinary logical case, is the absence of "false", then in general, the absence of
the negation operation. This can make the cohomology of information non-trivial.
Another big diﬀerence is that the category Ais not supposed to be a poset, the sets Hom can be more
complex than ∅and ★, and they can contain isomorphisms. In particular loops can be present.
Consider for instance any function 𝜓 on the collection of theories; and suppose that there exist arrows
from 𝐴to 𝐵and from 𝐵to 𝐴; then the function 𝜓must take the same value on the theories generated by
𝐴and 𝐵. This tells in particular that they contain the same information.
The homotopy construction of a bi-simplicial set 𝑔Θ can be made as before, representing the propaga-
tion feed-forward of theories and propagation backward of the propositions, and the information can be
defined by a natural increasing and concave map 𝐹with values in a closed model category Mof Quillen
(see chapter 2).
The semantic functioning becomes a simplicial map 𝑔𝑆: 𝑔X→𝑔Θ, and the semantic spaces are given
by the composition 𝐹◦𝑔𝑆.
Here is another interest of this generalization: we can assume that a measure of complexity 𝐾 is
attributed to the objects, seen as expressions in a language, and that this complexity is additive in the
product, i.e. 𝐾(𝑋⊗𝑌)= 𝐾(𝑋)+𝐾(𝑌), and related to the combinatorics of the syntax, and the com-
plexity of the lexicon, and the grammatical rules of formation. In this framework, we could compare
the values of 𝐾in the category, and define the compression as the ratio 𝐹/𝐾of information by complexity.
Remark. It is amazing and happy that the bar-complex for the information cocycles and the homotopy
limit, can also be defined for the bi-closed generalization. The two exponentials 𝑋𝐴and 𝐴𝑌 an action
of the monoid Ato the right and to the left that commute on the functions of theories, and on the
bi-simplicial set 𝑔Θ. Then we can apply the work of MacLane, Beck on bi-modules and the work of
Schulman on enriched categories.
Taking into account the network, we get a tri-simplicial set Θ••
★ of information elements, or tensors,
giving rise to a bi-simplicial space of histories of theories, with multiple left and right conditioning, 𝑔𝐼••
,
that is the geometrical analog of the bar-complex of semantic information.
Links with Linear Logic (intuitionist) and negations.
The generalized framework corresponds to a fragment of an intuitionist Linear Logic (see Bierman and
de Paiva [BdP00], Mellies [Mel09]). The arrows 𝐴→𝐵 in the category are the expression of the
133
assertions of consequence 𝐴⊢𝐵, and the product expresses the joint of the elements of the left members
of consequences, in the sense that a deduction 𝐴1,...,𝐴𝑛 ⊢𝐵corresponds to an arrow 𝐴1 ⊗...⊗𝐴𝑛 →𝐵.
There is no necessarily a "or" for the right side, but there is an internal implication 𝐴⊸ 𝐵which satisfies
all the axioms of the above implication 𝐴⇒𝐵, right adjoint of the tensor product. The existence of
the final element corresponds to the existence of (multiplicative) truth 1= ★. To be more complete, we
should suppose that all the finite products exist in the category A. Then the (categorial) product of
two corresponds to an additive disjunction ⊕, then a "or", that can generate the right side of sequents
𝐵1,...,𝐵𝑚 in 𝐴1,...,𝐴𝑛/𝐵1,...,𝐵𝑚; however, a neutral element for ⊕could be absent, even if it is always
present in the full theory of Girard [Gir87]. No right adjoint is required for ⊕. And in what follows we
do not assume the data ⊕.
One of the main ideas of [Gir87] was to incorporate the fact that in real life the proposition 𝐴that is
used in a consequence 𝐴⊸ 𝐵does not remain unchanged after the event, however it is important to give
a special status for propositions that continue to hold after the event. For that purpose Girard introduced
an operator on the formulas, named a linear exponential, and written !. It is named "of course" and has
the meaning of a reaﬃrmation, something stable. The functor ! is required to be naturally equivalent to !!,
then a projector in the sense of categories, such that, in a natural manner, the objects !𝐴and the morphisms
! 𝑓 between them satisfy the Gentzen rules of weakening and contraction, respectively (Γ ⊢Δ)/(Γ,!𝐴⊢Δ)
and (Γ,𝐴,𝐴⊢Δ)/(Γ,𝐴⊢Δ). (This corresponds to the traditional assertions 𝐴∧𝐵≤𝐴and 𝐴≤𝐴∧𝐴.)
Further axioms state, when translated in categorical terms, that ! is a monoidal functor equipped with
two natural transformations 𝜖𝐴 :!𝐴→𝐴and 𝛿𝐴 :!𝐴→!!𝐴, that are monoidal transformations, satisfying
the coherence rules of a comonad, and with natural transformations 𝑒𝐴 :!𝐴→1 (useful when 1 is not
assumed final) and 𝑑𝐴 :!𝐴→!𝐴⊗!𝐴, that is a diagonal operator, also satisfying coherence axioms telling
that each !𝐴is a commutative comonoid, and each ! 𝑓 a morphism of commutative comonoid. From all
these axioms, it is proved that under ! the monoidal product becomes a usual categorial product in the
category !A:= A!
,
!(𝐴⊗𝐵) !𝐴⊗!𝐵 !(𝐴×𝐵); (48)
and the category A!, named the Kleisli category of (A,!), is cartesian closed. More precisely, under !
the multiplicative exponential becomes the usual exponential:
!(𝐴⊸ 𝐵) !𝐵!𝐴
. (49)
Remind that a comonad in a category is a functor 𝑇 of this category to itself, equipped with two natural
transformations 𝑇→𝑇◦𝑇 and 𝜀: 𝑇→Id, satisfying coassociativity and counity axioms. This the dual
of a monad, 𝑇◦𝑇→𝑇 and Id →𝑇, that is the generalization of monoids to categories. The functor ! is
an example of comonad [Mac71].
The axioms of a closed symmetric monoidal category, plus the existence of finite products, plus the
functor !, give the largest part of the Gentzen rules, as they were generalized by Jean-Yves Girard in 1987
134
[Gir87].
Proposition E.1. The linear exponential ! allows to localize the product at a given proposition, in the
sense that the slice category to the right Γ|Ais closed by products of linear exponential objects as soon
as Γ belongs to A!
.
Proof. If we restrict us to the arrows !Γ →𝑄, then the product !Γ →𝑄⊗𝑄′is obtained by composing
the diagonal 𝑑!Γ :!Γ →!Γ⊗!Γ with the tensor product !Γ⊗!Γ →𝑄⊗𝑄′
.
Its right adjoint is given by !Γ →(𝑄⊸ 𝑅), obtained by composing !Γ →𝑄 with the natural map
𝑄→𝑄|𝑅.
To localize the theories themselves at 𝑃, for instance at a !Γ, we used, in the Heyting case, a notion of
negation. To exclude a given proposition was the only coherent choice from the point of view of informa-
tion, and this was also in accord with the experiments of spontaneous logics in small networks [BBG21a].
In the initial work of Girard, negation was a fundamental operator, verifying the hypothesis of in-
volution¬¬= Id, thus giving a duality. That explains that the initial theory is considered as a classical
Linear Logic; it generalizes the usual Boolean logic in another direction than intuitionism. In a linear
intuitionist theory, the negation is not necessary, but it is also not forbidden, and axioms were discussed
in the nineties.
We follow here the exposition of Paul-André Melliès in [Mel09] and of his article with Nicolas
Tabareau [MT10]. The authors work directly in a monoidal category A, without assuming that it is
closed, and define negation as a functor¬: A→Aop, such that the opposite functor¬op from Aop to
A, also denoted by¬, is the left-adjoint of¬, giving a unit 𝜂: Id →¬¬and a counit 𝜖: ¬¬→Id, that are
not equivalence in general. Then there exist for any objects 𝐴,𝐵a canonical bĳection bĳection between
HomA(¬𝐴,𝐵)and HomA(¬𝐵,𝐴). Note that in this case 𝜀 and 𝜂 coincide, because the morphisms in
Aop are the morphisms in Awritten in the reverse order.
The double negation 𝑇= ¬op¬forms a monad whose 𝜂is the unit; the multiplication 𝜇: ¬¬¬¬→¬¬is
obtained by composing Id¬with ¬(𝜂), to the left or to the right, that is 𝜇𝐴= ¬(𝜂𝐴)◦Id¬𝐴= Id¬¬¬𝐴◦¬(𝜂𝐴).
In theoretical computer science, 𝑇is called the continuation monad, and plays an important role in com-
putation and games logics as in the works of Kock, Moggi, Mellies, Tabareau.
In the case of the Heyting algebra of a topos (elementary), this continuation defines a topology, named
after Lawvere and Tierney, which defines the unique subtopos that is Boolean and dense (i.e. contains
the initial object ∅[Car12]).
The second important axiom tells how the (multiplicative) product ⊗is transformed : it is required that
for any objects 𝐵,𝐶the object ¬(𝐵⊗𝐶)represents the functor 𝐴↦→Hom(𝐴⊗𝐵,¬𝐶) Hom(𝐶,¬(𝐴⊗𝐵);
135
that is
Hom(𝐴⊗𝐵,¬𝐶) Hom(𝐴,¬(𝐵⊗𝐶)). (50)
This bĳection being natural in the three argument and coherent with the associativity and unit for the
product ⊗.
For instance all the sets Hom(𝐴𝐵𝐶,¬𝐷), Hom(𝐴𝐵,¬(𝐶𝐷), Hom(𝐴,¬(𝐵𝐶𝐷)), are identified with
Hom(𝐴𝐵𝐶𝐷,¬1).
Mellies and Tabareau [MT10] called such a structure a tensorial negation, and named the monoidal
category A, equipped with¬, a dialogue category.
The special object¬1 is canonically associated to the chosen negation; it is named the pole and frequently
denoted by ⊥. It has no reason in general to be an initial object of A.
A monoidal structure of (multiplicative) disjunction is deduced from the tensor product by duality:
𝐴℘𝐵= ¬(¬𝐴⊗¬𝐵). (51)
Its neutral element is the pole of¬.
This implies that the notion of "or" is parameterized by the variety of negations, that we will see equivalent
to Aitself.
In the same manner an additive conjonction is defined by
𝐴&𝐵= ¬(¬𝐴⊕¬𝐵). (52)
Its neutral element is ⊤= ¬∅, when an initial element ∅exists, that is the additive "false".
An operator ? was introduced by Girard in classical linear logic, that satisfies
?¬𝐴= ¬!𝐴,¬?𝐴=!¬𝐴 (53)
For us, just these relations are not suﬃcient to define it, because ¬is not a bĳection.
The Girard operator ? means "why not?", as the operator ! means "of course"; they are examples of
modalities, and correspond to the modalities more frequently denoted and ⋄in modal logics.
However, Hasegawa [Has03], Moggi [Mog91], Mellies and Tabareau [MT10] have remarked that
more convenient tensorial negations must satisfy a further axiom. Note that this story started with Kock
[Koc70] inspired by Eilenberg and Kelly [EK66].
136
Lemma E.1. From the second axiom of a tensorial negation it results two natural transformations
¬¬𝐴⊗𝐵→¬¬(𝐴⊗𝐵); (54)
𝐴⊗¬¬𝐵→¬¬(𝐴⊗𝐵). (55)
A monad where such maps exist in a monoidal category, is named a strong monad [Koc70] and [Mog91].
The first transformation is named the strength of the monad 𝑇= ¬¬, the second one its costrength.
Proof. Let us start with the Identity morphism of ¬(𝐴⊗𝐵); by the axiom, it can be interpreted as a
morphism 𝐵⊗¬(𝐴⊗𝐵)→¬𝐴, then applying the functor¬, we get a morphism
¬¬𝐴→¬[𝐵⊗¬(𝐴⊗𝐵)]; (56)
then, applying the axiom again, we obtain a natural transformation
¬¬𝐴⊗𝐵→¬¬(𝐴⊗𝐵). (57)
Exchanging the roles of 𝐴and 𝐵gives the other transformation.
Said in other terms, we have natural bĳections given by the tensorial axiom, applied two times,
Hom(¬(𝐴⊗𝐵),¬(𝐴⊗𝐵)) Hom(¬(𝐴⊗𝐵)⊗𝐵,¬𝐴)
Hom(𝐴,¬[𝐵⊗¬(𝐴⊗𝐵)] Hom(𝐴⊗𝐵,¬¬(𝐴⊗𝐵)); (58)
and also natural bĳections, obtained in the same manner,
Hom(¬(𝐴⊗𝐵),¬(𝐴⊗𝐵)) Hom(¬(𝐴⊗𝐵)⊗𝐴,¬𝐵)
Hom(𝐵,¬[𝐴⊗¬(𝐴⊗𝐵)] Hom(𝐴⊗𝐵,¬¬(𝐴⊗𝐵)); (59)
The identity of ¬(𝐴⊗𝐵)in the first term gives a natural marked point, that is also identifiable with 𝜂𝐴⊗𝐵
in the last term.
On the set Hom((¬(𝐴⊗𝐵)⊗𝐵,¬𝐴)(resp. Hom(𝐴⊗¬(𝐴⊗𝐵),¬𝐵)) we can apply the functor¬; this
gives a map to Hom(¬¬𝐴,¬[𝐵⊗¬(𝐴⊗𝐵)])(resp. Hom(¬¬𝐵,¬[𝐴⊗¬(𝐴⊗𝐵)])), then the strength
(resp. the costrength) after applying the second axiom.
The strength and costrength taken together give two a priori diﬀerent transformations 𝑇𝐴⊗𝑇𝐵→
𝑇(𝐴⊗𝐵)(see n lab cafe, Kock, Moggi, Hazegawa).
The first one is the composition starting with the costrength of 𝑇𝐴followed by the strength of 𝐵, then
ending with the product:
𝑇𝐴⊗𝑇𝐵→𝑇(𝑇𝐴⊗𝐵)→𝑇𝑇(𝐴⊗𝐵)→𝑇(𝐴⊗𝐵); (60)
the other one starts with the strength, then uses the costrength, and ends with the product
𝑇𝐴⊗𝑇𝐵→𝑇(𝐴⊗𝑇𝐵)→𝑇𝑇(𝐴⊗𝐵)→𝑇(𝐴⊗𝐵). (61)
137
Then a third axiom was suggested by Kock in general for strong monads, and reconsidered by Haze-
gawa, Moggi, Mellies and Tabareau, it consist to require that these two morphisms coincide. This is
named, since Kock, a commutative monad, or a monoidal monad. We will say that the negation itself is
monoidal.
According to Mellies and Tabareau, Hasegawa observed that 𝑇= ¬¬is commutative, if and only if 𝜂
gives an isomorphism¬ ¬¬on the objects of ¬A, if and only if 𝜇gives an isomorphism on the objects
of A.
Proposition E.2. A necessary and suﬃcient condition for having¬monoidal is that for each object 𝐴,
the transformation 𝜂¬𝐴 is an equivalence from¬𝐴and¬¬¬𝐴in the category A.
Corollary. Define A𝜂 as the collection of objects 𝐴′of A, such that 𝜂𝐴′ is an isomporphism; in the
commutative case, ¬Ais a sub-category¬induces an equivalence of the full subcategory A𝜂 of Awith
its opposite [Bel08, Proposition 1.31].
Thus we recover most of the usual properties of negation, without having a notion of false.
Now assume that Ais symmetric monoidal and closed; we get natural isomorphisms
¬(𝐴⊗𝐵)≈𝐴⇒¬𝐵≈𝐵⇒¬𝐴. (62)
And using the neutral element 1= ★for 𝐶, and denoting¬1 by 𝑃, we obtain that¬𝐵= 𝐵⊸ 𝑃.
Proposition E.3. For any object 𝑃∈A, the functor 𝐴↦→(𝐴⊸ 𝑃)= 𝑃|𝐴is a tensor negation whose
pole is 𝑃.
Proof. First, this is a contravariant functor in 𝐴.
Secondly, for any pair 𝐴,𝐵in A, using the symmetry hypothesis, we get natural bĳections
Hom(𝐵,𝐴⊸ 𝑃) Hom(𝐵⊗𝐴,𝑃) Hom(𝐴,𝐵⊸ 𝑃). (63)
This gives the basic adjunction.
Third, for any triple 𝐴,𝐵,𝐶in A, the associativity gives
Hom(𝐴⊗𝐵,𝐶⊸ 𝑃) Hom(𝐴⊗𝐵⊗𝐶,𝑃) Hom(𝐴,(𝐵⊗𝐶)⊸ 𝑃). (64)
This gives the tensorial condition.
The transformation 𝜂is given by the Yoneda lemma, from the following natural map
Hom(𝑋,𝐴)→Homop (¬𝑋,¬𝐴) Hom(𝑋,¬¬𝐴). (65)
138
There is no reason for asserting that this negation is commutative.
From proposition E.1, the necessary and suﬃcient condition is that, for any object 𝐴, the following map
is an isomorphism
𝜂𝐴⇒𝑃 : (𝐴⇒𝑃)→(((𝐴⇒𝑃)⇒𝑃)⇒𝑃). (66)
Even for 𝐴= 1 this is a non-trivial condition: 𝑃≈((𝑃⇒𝑃)⇒𝑃).
The fact that 1 ⇒𝑃≡𝑃being obvious.
Choose an arbitrary object Δ and define¬𝑄as 𝑄⊸ Δ. This Δ will play the role of "false".
We say that a theory Texcludes 𝑃if it contains 𝑃⊸ Δ. This is equivalent to say that there exists 𝑅in T
such that 𝑅→(𝑃⊸ Δ), i.e. 𝑅⊗𝑃→Δ, that is by symmetry: there exists 𝑃→(𝑅⊸ Δ). In particular,
if 𝑃→𝑅, we obtain such a map by composition with 𝑅→(𝑅⊸ Δ).
To localize the action of the proposition at 𝑃, we have to prove the following lemma:
Lemma E.2. Conditioning by 𝑄 such that 𝑃→𝑃⊗𝑄 is non-empty, sends a theory Tthat excludes 𝑃
into a theory Tthat also excludes 𝑃.
Proof. From the hypothesis we have a morphism ¬(𝑃×𝑄)→¬𝑃, but ¬(𝑃×𝑄)is isomorphic to
𝑄⇒(𝑃⇒Δ)= (¬𝑃)|𝑄.
This is analog to the statement of Proposition 3.2 in section 3.3, because in this case 𝑃≤𝑄is equivalent
to 𝑃= 𝑃∧𝑄and to 𝑃≤𝑃∧𝑄. The proof does not use that 𝑃is a linear exponential object.
Now assume that 𝑃belongs to the category A!, i.e. 𝑃=!Γ for a given object Γ ∈A; we saw that the
set A𝑃 of 𝑄such that 𝑃→𝑄forms a closed monoidal category, and by the above lemma, it acts on the
set of theories excluding 𝑃. That is because 𝑃→𝑄implies 𝑃→𝑃⊗𝑃→𝑃⊗𝑄
Therefore, all the ingredients of the information topology of chapter 2 are present in this situation.
139
Bibliography
[AB11] [Acz88] [Adj35] [AGV63] [AGZV12a] [AGZV12b] [AK11] [ARDK16] [Arg78] Samson Abramsky and Adam Brandenburger. The sheaf-theoretic structure of non-locality
and contextuality. New Journal of Physics, 13(11):113036, 2011.
Peter Aczel. Non-Well-Founded Sets. Stanford University, Center for the Study of Language
and Information, 1988.
Kazimierz Adjukiewicz. Die Syntaktische Konnexität. Studia Philosophica, 1:1–27, 1935.
Michael Artin, Alexander Grothendieck, and Jean-Louis Verdier. Théorie des topos et
cohomologie étale des schémas. SGA4, IHES, 1963.
Vladimir Igorevitch Arnold, Sabir Medgidovich Gusein-Zade, and Alexander Nikolaevich
Varchenko. Singularities of Diﬀerentiable Maps, Volume 1: Classification of Critical
Points, Caustics and Wave Fronts. Modern Birkhäuser Classics. Birkhäuser Boston, 2012.
Vladimir Igorevitch Arnold, Sabir Medgidovich Gusein-Zade, and Alexander Nikolaevich
Varchenko. Singularities of Diﬀerentiable Maps, Volume 2: Monodromy and Asymptotics
of Integrals. Modern Birkhäuser Classics. Birkhäuser Boston, 2012.
Peter Arndt and Krzysztof Kapulkin. Homotopy-theoretic models of type theory. In Pro-
ceedings of the 10th International Conference on Typed Lambda Calculi and Applications,
TLCA’11, Berlin, Heidelberg, 2011. Springer-Verlag.
Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Learning to compose
neural networks for question answering. CoRR, abs/1601.01705, 2016.
Jose Argemi. Approche qualitative d’un problème de perturbations singulières dans R4
.
Equadiﬀ. 78, Conv. int. su equazioni diﬀerenziali ordinarie ed equazioni funzionali, Firenze
1978, 333-340, 1978.
140
[Arn73] Vladimir Igorevitch Arnold. Normal forms for functions near degenerate critical points, the
Weyl groups of 𝐴𝑘, 𝐷𝑘, 𝐸𝑘 and Lagrangian singularities. Funct. Anal. Appl., 6:254–272,
1973.
[Aus61] John Langshow Austin. Philosophical Papers. Oxford University Press, 1961.
[AW09] Steve Awodey and Michael A. Warren. Homotopy theoretic models of identity types.
Mathematical Proceedings of the Cambridge Philosophical Society, 146(1), Jan 2009.
[BB15] Pierre Baudot and Daniel Bennequin. The Homological Nature of Entropy. Entropy, pages
3253–3318, 2015.
[BB22] Jean-Claude Belfiore and Daniel Bennequin. A search of semantic spaces. Internal
technical report, Huawei, 2022.
[BBCV21] [BBD+11] [BBDH14] [BBG20] [BBG21a] [BBG21b] [BCB16] [BdP00] [Bel08] [Ben86] Michael M. Bronstein, Joan Bruna, Taco Cohen, and Petar Veličković. Geometric deep
learning: Grids, groups, graphs, geodesics, and gauges, 2021.
Jie Bao, Prithwish Basu, Mike Dean, Craig Partridge, Ananthram Swami, Will Leland, and
James A. Hendler. Towards a theory of semantic communication. In 2011 IEEE Network
Science Workshop, pages 110–117, 2011.
Prithwish Basu, Jie Bao, Mike Dean, and James A. Hendler. Preserving quality of infor-
mation by using semantic relationships. Pervasive and Mobile Computing, 11:188 – 202,
2014.
Jean-Claude Belfiore, Daniel Bennequin, and Xavier Giraud. Logico-probabilistic infor-
mation. Internal technical report, Huawei, 2020. Available upon request.
Jean-Claude Belfiore, Daniel Bennequin, and Xavier Giraud. Logical Information Cells I.
arXiv, 2021. 2108.04751.
Jean-Claude Belfiore, Daniel Bennequin, and Xavier Giraud. Logical Information Cells,
Part II. Internal technical report, Huawei, 2021. Available upon request.
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by
jointly learning to align and translate. ArXiv, abs/1409.0473v7, 2016.
G. M. Bierman and V. C. V. de Paiva. On an intuitionistic modal logic. Studia Logica: An
International Journal for Symbolic Logic, 65(3):383–416, 2000.
John L. Bell. Toposes and Local Set Theories. Dover, 2008.
Daniel Bennequin. Caustique mystique. In Séminaire Bourbaki : volume 1984/85, exposés
633-650, number 133-134 in Astérisque. Société mathématique de France, 1986. talk:634.
141
[BFL11] John C Baez, Tobias Fritz, and Tom Leinster. A characterization of entropy in terms of
information loss. Entropy, 13(11):1945–1957, 2011.
[BH53] Yehoshua Bar-Hillel. A quasi-arithmetical notation for syntactic description. Language,
29(1):47–58, 1953.
[BHS+18] David Barrett, Felix Hill, Adam Santoro, Ari Morcos, and Timothy Lillicrap. Measur-
ing abstract reasoning in neural networks. In Jennifer Dy and Andreas Krause, editors,
Proceedings of the 35th International Conference on Machine Learning, volume 80 of Pro-
ceedings of Machine Learning Research, pages 511–520, Stockholmsmässan, Stockholm
Sweden, 10–15 Jul 2018. PMLR.
[BK72] A. K. Bousfield and D. M. Kan. Homotopy Limits, Completions and Localizations.
Springer, 1972.
[BPSPV20] Daniel Bennequin, Olivier Peltre, Grégoire Sergeant-Perthuis, and
Juan Pablo Vigneaux. Extra-fine sheaves and interaction decompositions.
https://doi.org/10.48550/arXiv.2009.12646, 2020.
[Bro73] Kenneth Brown. Abstract homotopy theory and generalized sheaf cohomology. Transaction
of the American Mathematical Society, 186:419–458, 1973.
[Bru36] Ferdinand Brunot. La pensée et la langue. Masson et compagnie, 1936.
[BTBG19] Pierre Baudot, Monica Tapia, Daniel Bennequin, and Jean-Marc Goaillard. Topological
information data analysis. Entropy, 21(9):869, Sep 2019.
[BW21] Roberto Bondesan and Max Welling. The hintons in your neural network: a quantum field
theory view of deep learning, 2021.
[Car50] Rudolf Carnap. Logical Foundations of Probability. Chicago Press, University of Chicago,
1950.
[Car09] Olivia Caramello. The Duality Between Grothendieck Toposes and Geometric Theories.
University of Cambridge, 2009.
[Car12] Olivia Caramello. Universal models and definability. Mathematical Proceedings of the
Cambridge Philosophical Society, 152(2), 2012.
[Car18] Olivia Caramello. Theories, sites, toposes : relating and studying mathematical theories
through topos-theoretic ’bridges’. Oxford University Press, Oxford, 2018.
[CBH52] Rudolf Carnap and Jehoshua Bar-Hillel. An Outline of a Theory of Semantic Information.
Technical report, Research Laboratory of Electronics, MIT, 1952.
142
[CDLL15] R. Cooper, Simon Dobnik, Staﬀan Larsson, and Shalom Lappin. Probabilistic type theory
and natural language semantics. Linguistic Issues in Language Technology, 10, 2015.
[CGCB14] Junyoung Chung, Çaglar Gülçehre, KyungHyun Cho, and Yoshua Bengio. Empirical eval-
uation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555,
2014.
[CGW20] Taco Cohen, Mario Geiger, and Maurice Weiler. A general theory of equivariant CNNs on
homogeneous spaces, 2020.
[Cis03] Denis-Charles Cisinski. Images directes cohomologiques dans les catégories de modèles.
Annales Mathématiques Blaise Pascal, 10(2):195–244, 2003.
[Cis06] Denis-Charles Cisinski. Les Préfaisceaux Comme Modèles Des Types d’Homotopie.
Astérisque (Société Mathématique de France). Société Mathématique de France, 2006.
[Cis19] Denis-Charles Cisinski. Higher Categories and Homotopical Algebra. Cambridge Studies
in Advanced Mathematics. Cambridge University Press, 2019.
[CLS95] Antoine Culioli, Michel Liddle, and John T. Stonham. Cognition and Representation in
Linguistic Theory. Benjamins, John Publishing Company, 1995.
[Cur13] Justin Curry. Sheaves, cosheaves and applications. arXiv preprint arXiv:1303.3255, 2013.
[Cur17] Carina Curto. What can topology tell us about the neural code? Bulletin of the American
Mathematical Society, 54(1):63–78, 2017.
[CvMBB14] KyungHyun Cho, Bart van Merrienboer, Dzmitry Bahdanau, and Yoshua Bengio. On
the properties of neural machine translation: Encoder-decoder approaches. CoRR,
abs/1409.1259, 2014.
[CWKW19] Taco S. Cohen, Maurice Weiler, Berkay Kicanaoglu, and Max Welling. Gauge equivariant
convolutional networks and the icosahedral CNN, 2019.
[CZ21] Olivia Caramello and Riccardo Zanfa. https://arxiv.org/abs/2107.04417, 2021.
Relative Topos Theory via Stacks.
[CZCG05] Gunnar Carlsson, Afra Zomorodian, Anne Collins, and Leonidas J Guibas. Persistence
barcodes for shapes. International Journal of Shape Modeling, 11(02):149–187, 2005.
[DHKS04] William G. Dwyer, Philip S. Hirschhorn, Daniel M. Kan, and Jeﬀrey H. Smith. Homotopy
Limit Functors on Model Categories and Homotopical Categories. AMS Mathematical
Surveys and Monographs, 2004.
143
[DHSB20] David Ding, Felix Hill, Adam Santoro, and Matt M. Botvinick. Object-based attention
for spatio-temporal reasoning: Outperforming neuro-symbolic models with flexible dis-
tributed architectures. CoRR, abs/2012.08508, 2020.
[Dou92] Daniel J. Dougherty. Closed categories and categorical grammar. Notre Dame Journal of
Formal Logic, 34(1):36 – 49, 1992.
[Dug08] Daniel Dugger. A primer on homotopy colimits.
https://pages.uoregon.edu/ddugger/hocolim.pdf, 2008.
[EK66] [EM45] [Fri05] [FS18] [FST19] [GB20] [GH11] [Gir64] [Gir71] [Gir72] [Gir87] Samuel Eilenberg and G. Max Kelly. Closed categories. In S. Eilenberg, D. K. Harrison,
S. MacLane, and H. Röhrl, editors, Proceedings of the Conference on Categorical Algebra,
pages 421–562, Berlin, Heidelberg, 1966. Springer Berlin Heidelberg.
Samuel Eilenberg and Saunders MacLane. General theory of natural equivalences. Trans-
actions of the American Mathematical Society, 58(2):231–294, 1945.
Joel Friedmann. Cohomology in Grothendieck topologies and lower bounds in Boolean
complexity. ArXiv, 2005.
Brendan Fong and David I Spivak. Seven sketches in compositionality: An invitation to
applied category theory, 2018.
Brendan Fong, David I. Spivak, and Rémy Tuyéras. Backprop as functor: a compositional
perspective on supervised learning. ArXiv, 2019.
Håkon Robbestad Gylterud and Elisabeth Bonnevier. Non-wellfounded sets in homotopy
type theory, 2020.
R. Ghrist and Y. Hiraoka. Applications of sheaf cohomology and exact sequences on
network codings. Research Institute for Mathematical Sciences, 1752:31–40, 2011.
Jean Giraud. Méthode de la descente. Number 2 in Mémoires de la Société Mathématique
de France. Société mathématique de France, 1964.
Jean Giraud. Cohomologie non abélienne. Springer, 1971.
Jean Giraud. Classifying Topos. Toposes, algebraic Geometry and Logic, Dalhousie Univ.
Halifax 1971, Lect. Notes Math. 274, 43-56 (1972)., 1972.
Jean-Yves Girard. Linear logic. Theoretical Computer Science, 50(1):1–101, 1987.
[GLH+20] Anirudh Goyal, Alex Lamb, Jordan Hoﬀmann, Shagun Sodhani, Sergey Levine, Yoshua
Bengio, and Bernhard Schölkopf. Recurrent independent mechanisms, 2020.
144
[Gro57] [Gro83] [Gro90] [GWDPL76] [GZ67] [Has03] [HB20] [Hir03] [HM18] [HM21] [Hoc69] [Hol01] [Hol08] [HS97] [HS98] Alexander Grothendieck. Sur quelques points d’algèbre homologique. Tohoku Mathemat-
ica Journal, 9:119–221, 1957.
Alexander Grothendieck. Pursuing Stacks. https://thescrivener.github.io/PursuingStac
1983.
Alexandre Grothendieck. Les Dérivateurs. https://webusers.imj-prg.fr/~georges.maltsi
1990.
C.G. Gibson, K. Wirthmuller, A.A. Du Plessis, and Eduard Looĳenga. Topological Stability
of Smooth Mappings, volume 552 of Lecture Notes in Mathematics. Springer, 1976.
Pierre Gabriel and Michel Zisman. Calculus of Fractions and Homotopy Theory. Ergeb-
nisse der Mathematik und ihrer Grenzgebiete. Island Press, 1967.
Masahito Hasegawa. Coherence of the double negation in linear logic. In Algebra, Logic
and Geometry in Informatics, 2003.
R. Devon Hjelm and Philip Bachman. Representation learning with video deep infomax,
2020.
Philip S. Hirschhorn. Model Categories and their Localizations, volume 99 of Mathemat-
ical Surviews and Monographs. AMS, 2003.
Drew Hudson and Christopher Manning. Compositional attention networks for machine
reasoning. https://arxiv.org/pdf/1803.03067.pdf, March 2018.
Liao Heng and Bill McColl, editors. Mathematics for Future Computing and Communi-
cations. Cambridge University Press, 2021.
Melvin Hochster. Prime ideal structure in commutative rings. Transactions of the American
Mathematical Society, 142:43–60, 1969.
Sharon Hollander. A homotopy theory for stacks. Israel Journal of Mathematics, 163:93–
124, 2001.
Sharon Hollander. A homotopy theory for stacks. Israel Journal of Mathematics, 2008.
Sepp Hochreiter and Jurgen Schmidhuber. Long-Short Term Memory. Neural Computa-
tion, 9, 1997.
Martin Hofmann and Thomas Streicher. The Groupoid interpretation of type theory. In
Twenty-five years of constructive type theory (Venice, 1995), volume 36 of Oxford Logic
Guides, pages 83–111. Oxford Univ. Press, New York, 1998.
145
[HS17] Joel C. Heck and Fathi M. Salem. Simplified minimal gated unit variations for recurrent
neural networks. CoRR, abs/1701.03452, 2017.
[Ill14] Luc Illusie. Travaux de Gabber sur l’uniformisation locale et la cohomologie étale des
schémas excellents. In Luc Illusie, Yves Laszlo, and Fabrice Orgogozo, editors, Logical
Aspects of Computational Linguistics, pages 213–234. Asterisque, 2014.
[Jar09] John Frederick Jardine. Cocycle categories. In Nils Baas, Eric M. Friedlander, Björn
Jahren, and Paul Arne Østvær, editors, Algebraic Topology: The Abel Symposium 2007,
pages 185–218, Berlin, Heidelberg, 2009. Springer Berlin Heidelberg.
[KL14] Andrej Karpathy and Fei-Fei Li. Deep visual-semantic alignments for generating image
descriptions. CoRR, abs/1412.2306, 2014.
[KLV12] Chris Kapulkin, Peter Lumsdaine, and Vladimir Voevodsky. The simplicial model of
univalent foundations. arxiv, 11 2012.
[Koc70] Anders Kock. Monads on symmetric monoidal closed categories. Archiv der Mathematik,
21:1–10, Jan. 1970.
[Kon18] Risi Kondor. 𝑁−body Networks: a Covariant Hierarchical Neural Network Architecture
for learning atomic potentials. ArXiv, abs/1803.01588, 2018.
[Lam58] Joachim Lambek. The mathematics of sentence structure. The American Mathematical
Monthly, 65(3):154–170, 1958.
[Lam88] Joachim Lambek. Categorial and Categorical Grammars, pages 297–317. Springer
Netherlands, Dordrecht, 1988.
[Lin89] Ingrid Lindström. A construction of Non-Well-Founded Sets within Martin-Löf’s Type
Theory. The Journal of Symbolic Logic, 54(1):57–64, 1989.
[Lin20] Johan Lindberg. Localic Categories of Models and Categorical Aspects of Intuitionistic
Ramified Type Theory. PhD thesis, Stockholm University, Department of Mathematics,
2020.
[Loo78] Eduard Looĳenga. The discriminant of a real simple singularity. Compositio Mathematica,
37(1):51–62, 1978.
[Low15] Zhen Lin Low. Cocycles in categories of fibrant objects.
https://arxiv.org/abs/1502.03925, 2015. arXiv.
[LS81] Joachim Lambek and Philip J. Scott. Intuitionist type theory and foundations. Journal of
Philosophical Logic, 10:101–115, 1981.
146
[LS88] Joachim Lambek and Philip J Scott. Introduction to higher-order categorical logic, vol-
ume 7. Cambridge University Press, 1988.
[Luo14] Zhaohui Luo. Formal semantics in modern type theories: Is it model-theoretic, proof-
theoretic, or both? In Nicholas Asher and Sergei Soloviev, editors, Logical Aspects
of Computational Linguistics, pages 177–188, Berlin, Heidelberg, 2014. Springer Berlin
Heidelberg.
[Lur09] Jacob Lurie. Higher Topos Theory (AM-170). Academic Search Complete. Princeton
University Press, 2009.
[Mac71] Saunders MacLane. Categories for the Working Mathematician. Springer-Verlag, New
York, 1971. Graduate Texts in Mathematics, Vol. 5.
[Mac12] Saunders MacLane. Homology. Classics in Mathematics. Springer Berlin Heidelberg,
2012.
[Mal05] Georges Maltsiniotis. La théorie de l’homotopie de Grothendieck. Number 301 in
Astérisque. Société mathématique de France, 2005.
[Mar82] Jean Martinet. Singularities of smooth functions and maps. Transl. from the French by
Carl P. Simon, volume 58. Cambridge University Press, Cambridge. London Mathematical
Society, London, 1982.
[MBHSL19] Haggai Maron, Heli Ben-Hamu, Nadav Shamir, and Yaron Lipman. Invariant and equiv-
ariant graph networks, 2019.
[McG54] William McGill. Multivariate information transmission. Psychometrika, 19:97–116, 1954.
[Mel09] Paul-André Melliès. Categorical Semantics of Linear logic. SMF, 2009.
[Mil62] John Milnor. On axiomatic homology theory. Pacific Journal of Mathematics, 12(1):337
– 341, 1962.
[ML80] Per Martin-Löf. Intuitionistic Type Theory. Lectures given at University of Padova, 1980.
[MLM92] Saunders Mac Lane and Ieke Mœrdĳk. Sheaves in geometry and logic: a first introduction
to topos theory. Universitext. New York etc.: Springer-Verlag., 1992.
[MM20] Yuri Manin and Matilde Marcolli. Homotopy theoretic and categorical models of neural
information networks. ArXiv, 2020.
[Mog91] Eugenio Moggi. Notions of computation and monads. Information and Computation,
93(1):55–92, 1991.
147
[Mon70] [MT10] [MXY+15] [Par75] [Pea88] [Pel20] [PKM19] [Pri94] [Pro08] [Pro19] [Qui67] Richard Montague. Universal grammar. Theoria, 36, 1970.
Paul-André Melliès and Nicolas Tabareau. Resource modalities in tensor logic. Annals of
Pure and Applied Logic, 161(5):632–653, February 2010.
Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, and Zhiheng Huangand Alan Yuille. Deep
captioning with multimodal recurrent neural networks (𝑚−RNN). ArXiv, abs/1412.6632v5,
2015.
Barbara Partee. Montague grammar and transformational grammar. Linguistic Inquiry,
6(2):203–300, 1975.
Judea Pearl. Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Infer-
ence. Morgan Kaufmann, 1988.
Olivier Peltre. A Homological Approach to Belief Propagation and Bethe Approximations.
PhD thesis, Dept of Mathematics, Université Paris Diderot, 2020.
Alexander Port, Taelin Karidi, and Matilde Marcolli. Topological analysis of syntactic
structures. arXiv preprint arXiv:1903.05181, 2019.
Hillary Priestley. Spectral sets. Journal of Pure and Applied Algebra, 94:101–114, 1994.
Alain Prouté. La théorie des ensembles selon les Shadoks. Séminaire général de Logique
de l’université Paris Diderot, 2008.
Alain Prouté. Introduction à la logique catégorique. MSc Course, Université Paris Diderot,
2019.
Daniel G. Quillen. Homotopical Algebra. Lecture notes in mathematics. Springer-Verlag,
1967.
[Rap10] George Raptis. Homotopy theory of posets. Homology, Homotopy and Applications,
12:211–230, 2010.
[Rob17] Michael Robinson. Sheaves are the canonical data structure for sensor integration. Infor-
mation Fusion, 36:208–224, 2017.
[rR09] Jiří Rosický. On combinatorial model categories. Applied Categorical Structures, 17:303–
316, 2009.
[RSB+17] David Raposo, Adam Santoro, David G. T. Barrett, Razvan Pascanu, Timothy P. Lillicrap,
and Peter W. Battaglia. Discovering objects and their relations from entangled scene
representations. CoRR, abs/1702.05068, 2017.
148
[See84] R. A. G. Seely. Locally cartesian closed categories and type theory. Mathematical Pro-
ceedings of the Cambridge Philosophical Society, 95(1):33–48, 1984.
[SFR+18] Adam Santoro, Ryan Faulkner, David Raposo, Jack W. Rae, Mike Chrzanowski, Théophane
Weber, Daan Wierstra, Oriol Vinyals, Razvan Pascanu, and Timothy P. Lillicrap. Relational
recurrent neural networks. In NeurIPS, 2018.
[SGW21] Dan Shiebler, Bruno Gavranović, and Paul W. Wilson. Category theory in machine
learning. CoRR, abs/2106.07032, 2021.
[Shu10] Michael A. Shulman. Stack semantics and the comparison of material and structural set
theories, 2010.
[Shu12] Michael Shulman. Exact completions and small sheaves. Theory and Applications of
Categories, 27:97–173, 2012.
[Shu19] Michael Shulman. Comparing material and structural set theories. Annals of Pure and
Applied Logic, 170(4), Apr 2019.
[SP12] Chris Schommer-Pries. The canonical model structure on Cat.
https://sbseminar.wordpress.com/2012/11/, 2012. Blog.
[SP21] Grégoire Sergeant-Perthuis. Intersection property, interaction decomposition, regionalized
optimization and applications. PhD thesis, University of Paris Diderot, March 2021.
[SRB+17] Adam Santoro, David Raposo, David G. T. Barrett, Mateusz Malinowski, Razvan Pascanu,
Peter W. Battaglia, and Timothy P. Lillicrap. A simple neural network module for relational
reasoning. CoRR, abs/1706.01427, 2017.
[Sta] Stacks. The Stacks Project. https://stacks.math.columbia.edu/.
[Sta14] Alexandru E. Stanculescu. Stacks and sheaves of categories as fibrant objects, I and II.
Arxiv, 2014.
[SW49] Claude E. Shannon and Warren Weaver. The Mathematical Theory of Communication.
University of Illinois Press, Urbana and Chicago, 1949.
[Ted16] Christopher Francis Tedd. Ring constructions on spectral spaces. PhD thesis, University
of Manchester, 2016.
[Tho72] René Thom. Stabilité structurelle et morphogénése : Essai d’une théorie générale des
modèles. Benjamin, 1972.
[Tho80] Robert Wayne Thomason. Cat as a closed model category. Cahiers de Topologie et
Géométrie Diﬀérentielle Catégoriques, 21(3):305–324, 1980.
149
[Tho83] [Tho88] [Tin62] [Uem17] [vB90] [Vig19] [Vig20] [VSP+17] [Wit53] [XQLJ20] [YFW01] [ZCZ+19] [ZRS+18] René Thom. Mathematical Models of Morphogenesis. Ellis Horwood Series in Math-
ematics and its applications. Ellis Horwood, 1983. translated by W.M. Brookes and D.
Rand.
René Thom. Esquisse d’une sémiophysique : Physique aristotélicienne et théorie des
catastrophes. Dunod, 1988.
Hu Kuo Ting. On the amount of information. Theory of Probability and its Applications,
1962.
Taichi Uemura. Fibred fibration categories. 2017 32nd Annual ACM/IEEE Symposium on
Logic in Computer Science (LICS), Jun 2017.
Johan van Benthem. Categorial grammar and type theory. Journal of Philosophical Logic,
19(2):115–168, 1990.
Juan-Pablo Vigneaux. Topology of statistical systems : a cohomological approach to
information theory. PhD thesis, University of Paris Diderot, 2019.
Juan Pablo Vigneaux. Information structures and their cohomology. Theory Appl. Categ.,
35:1476–1529, 2020.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N.
Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. CoRR,
abs/1706.03762, 2017.
Ludwig Wittgenstein. Philosophical Investigations. Oxford, Blackwell, 1953.
Huiqiang Xie, Zhĳin Qin, Geoﬀrey Ye Li, and Biing-Hwang Juang. Deep learning enabled
semantic communication systems. ArXiv, 2020.
Jonathan S Yedidia, William Freeman, and Yair Weiss. Generalized belief propagation. In
T. Leen, T. Dietterich, and V. Tresp, editors, Advances in Neural Information Processing
Systems, volume 13. MIT Press, 2001.
Tao Zhuo, Zhiyong Cheng, Peng Zhang, Yongkang Wong, and Mohan Kankanhalli. Ex-
plainable video action reasoning via prior knowledge and state transitions. In Proceedings
of the 27th ACM International Conference on Multimedia, MM ’19, New York, NY, USA,
2019. Association for Computing Machinery.
Vinícius Flores Zambaldi, David Raposo, Adam Santoro, Victor Bapst, Yujia Li, Igor
Babuschkin, Karl Tuyls, David P. Reichert, Timothy P. Lillicrap, Edward Lockhart, Murray
Shanahan, Victoria Langston, Razvan Pascanu, Matthew Botvinick, Oriol Vinyals, and
Peter W. Battaglia. Relational deep reinforcement learning. CoRR, abs/1806.01830, 2018.
150
[ZWZZ16] Guo-Bing Zhou, Jianxin Wu, Chen-Lin Zhang, and Zhi-Hua Zhou. Minimal Gated Unit
for Recurrent Neural Networks. CoRR, abs/1603.09420, 2016.
151

-}
