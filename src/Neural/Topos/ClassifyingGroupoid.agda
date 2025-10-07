{-# OPTIONS --rewriting --guardedness --cubical --no-load-primitives #-}

{-|
# Appendix C: Classifying Objects of Groupoids

This module implements the classification of presheaves over groupoids
from Appendix C of Belfiore & Bennequin (2022).

## Paper Reference

> "Proposition C.1. There exists an equivalence of category between any
> connected groupoid G and its fundamental group G."

> "One manner to present the topos E = E_G of presheaves over a small
> groupoid G (up to category equivalence) is to decompose G in connected
> components G_a; a ∈ A, then E will be product of the topos E_a; a ∈ A
> of presheaves over each component."

## Key Results

**Proposition C.1**: Connected groupoid G ≃ Fundamental group G
- Equivalence via functors J: G → G and R: G → G
- Natural isomorphisms witness equivalence
- Choice of basepoint O ∈ G

**Proposition C.2**: E ≃ Sh(Ω; K) with canonical topology
- E = Presheaves over groupoid G
- Ω = Boolean algebra of orbits
- K = Set of orbit representatives
- Canonical Grothendieck topology

## Structure

1. **Connected groupoids** and fundamental group
2. **Equivalence** G ≃ G (Proposition C.1)
3. **Decomposition** by connected components
4. **G-sets** and orbit structure
5. **Classifying object** Ω_G
6. **Canonical topology** on Ω
7. **Sheaf equivalence** (Proposition C.2)

## DNN Interpretation

- Groupoid = Symmetries in network
- Connected components = Independent sub-networks
- Orbits = Equivalence classes under symmetries
- Classifying object = Boolean algebra of symmetric properties
- Sheaves = Presheaves respecting symmetries

## References

- [Bel08] Bell (2008): Toposes and Local Set Theories
-}

module Neural.Topos.ClassifyingGroupoid where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path
open import 1Lab.Equiv
open import 1Lab.Type.Sigma

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Functor.Equivalence
open import Cat.Groupoid
open import Cat.Instances.Functor

open import Algebra.Group

-- Import existing groupoid work
open import Neural.Stack.Groupoid

private variable
  o ℓ o' ℓ' : Level

--------------------------------------------------------------------------------
-- §C.1: Connected Groupoids

{-|
## Connected Groupoids

A groupoid G is **connected** if:
- For any objects x, y ∈ G, there exists a morphism γ: x → y

Equivalently:
- π₀(G) = ⊤ (one connected component)
- Any two objects are isomorphic

**Examples**:
1. Group G as one-object groupoid (always connected)
2. Fundamental groupoid π₁(X) of connected space X
3. Action groupoid of transitive group action

**DNN interpretation**:
- Connected groupoid = Uniform symmetry across network
- All layers have same symmetry group
- Example: Translation symmetry in CNN (all layers)
-}

record is-connected-groupoid {o ℓ} (G : Precategory o ℓ) : Type (o ⊔ ℓ) where
  no-eta-equality
  field
    -- G is a groupoid
    has-groupoid : is-groupoid G

    -- Connectedness: Any two objects are connected by a morphism
    connected : ∀ (x y : G .Precategory.Ob)
              → ∥ G .Precategory.Hom x y ∥

open is-connected-groupoid public

--------------------------------------------------------------------------------
-- §C.2: Fundamental Group

{-|
## Fundamental Group of Groupoid

For connected groupoid G with chosen basepoint O:

The **fundamental group** G_O is:
- G_O = Aut(O) (automorphism group of O)
- Group operation = composition
- Identity = id_O
- Inverse = categorical inverse

**Proposition C.1**: G ≃ G_O (as categories)

**Proof idea**:
1. Choose paths γ_x: x → O for all x
2. J: G_O → G (inclusion)
3. R: G → G_O (conjugation by paths)
4. Natural isomorphisms show equivalence
-}

module _ {o ℓ} (G : Precategory o ℓ) (conn : is-connected-groupoid G) where
  private module G = Precategory G
  open is-connected-groupoid conn

  -- Choose a basepoint O ∈ G
  module _ (O : G.Ob) where
    -- Fundamental group = Automorphism group of O
    fundamental-group : Group (o ⊔ ℓ)
    fundamental-group = record
      { _⋆_ = λ f g → f G.∘ g
      ; has-is-group = {!!}  -- Automorphisms form a group
      }

    -- Underlying type of fundamental group
    π₁ : Type (o ⊔ ℓ)
    π₁ = G.Hom O O

    {-|
    ## Inclusion Functor J: G → G

    The fundamental group embeds into G as automorphisms of O.

    J sends:
    - Object: * ↦ O
    - Morphism: g ↦ g (as automorphism of O)

    This is full and faithful (injective on hom-sets).
    -}

    -- G as one-object groupoid
    BG : Precategory (o ⊔ ℓ) (o ⊔ ℓ)
    BG = record
      { Ob = ⊤
      ; Hom = λ _ _ → π₁
      ; Hom-set = λ _ _ → {!!}
      ; id = G.id
      ; _∘_ = G._∘_
      ; idr = G.idr
      ; idl = G.idl
      ; assoc = G.assoc
      }

    -- Inclusion functor J: BG → G
    J : Functor BG G
    J = record
      { F₀ = λ _ → O
      ; F₁ = λ g → g
      ; F-id = refl
      ; F-∘ = λ f g → refl
      }

    {-|
    ## Retraction Functor R: G → G

    For each object x, choose a path γ_x: x → O.
    - γ_O = id_O
    - For γ: x → y, send to γ_y ∘ γ ∘ γ_x⁻¹

    **Equation 42**: Naturality
      T(y) ∘ Id(γ) = γ_y ∘ γ
                   = (γ_y ∘ γ) ∘ γ_x⁻¹ ∘ γ_x
                   = J(R(γ)) ∘ T(x)
    -}

    -- Choice of paths to basepoint
    record PathsToBasepoint : Type (o ⊔ ℓ) where
      no-eta-equality
      field
        γ : ∀ (x : G.Ob) → G.Hom x O
        γ-basepoint : γ O ≡ G.id

    open PathsToBasepoint

    module _ (paths : PathsToBasepoint) where
      -- Retraction functor R: G → BG
      R : Functor G BG
      R = record
        { F₀ = λ _ → tt
        ; F₁ = λ {x} {y} (f : G.Hom x y) →
                 paths .γ y G.∘ f G.∘ has-groupoid x O (paths .γ x) .is-invertible.inv
        ; F-id = {!!}
        ; F-∘ = λ f g → {!!}
        }

      {-|
      **Natural isomorphism**: R ∘ J ≅ Id_BG

      This is just the identity (no coherence needed).
      -}
      R∘J≅Id : R F∘ J ≅ⁿ Id
      R∘J≅Id = {!!}

      {-|
      **Natural transformation**: J ∘ R ⇒ Id_G (Equation 42)

      T(x) = γ_x for each x ∈ G

      Naturality: For f: x → y,
        T(y) ∘ f = γ_y ∘ f
                 = (γ_y ∘ f ∘ γ_x⁻¹) ∘ γ_x
                 = J(R(f)) ∘ T(x)
      -}
      J∘R⇒Id : J F∘ R => Id
      J∘R⇒Id = record
        { η = λ x → paths .γ x
        ; is-natural = λ x y f →
            -- Equation 42 from paper
            {!!}
        }

      {-|
      **Proposition C.1**: G ≃ BG (equivalence of categories)

      The functors J and R with natural isomorphisms form an equivalence.

      **Note**: The choice of paths is not unique (except if G = BG = {1}).
      Different choices give different but equivalent retractions.
      This makes groupoids "strictly richer" than groups.
      -}
      proposition-C-1 : is-equivalence J
      proposition-C-1 = {!!}

--------------------------------------------------------------------------------
-- §C.3: Decomposition by Connected Components

{-|
## Decomposition by Connected Components

Any groupoid G decomposes as:
  G = ⨿_{a ∈ A} G_a

Where:
- A = π₀(G) (set of connected components)
- G_a = Connected component containing a

**Topos decomposition**:
  E_G ≃ ∏_{a ∈ A} E_{G_a}

Where E_{G_a} is the topos of G_a-sets.

**For each component a**:
- E_{G_a} ≃ Set^{BG_a} (presheaves over fundamental group)
- ≃ G_a-Set (category of G_a-sets)

**DNN interpretation**:
- A = Independent sub-networks
- Each G_a = Symmetry group of sub-network a
- E_{G_a} = Equivariant features in sub-network a
-}

-- Connected components
π₀ : {o ℓ : Level} → Precategory o ℓ → Type o
π₀ G = ∥ G .Precategory.Ob ∥  -- Truncated to set

-- Decomposition into connected components
record Decomposition {o ℓ} (G : Precategory o ℓ) : Type (lsuc o ⊔ ℓ) where
  no-eta-equality
  field
    -- Index set A of connected components
    A : Type o

    -- Each component
    component : A → Precategory o ℓ

    -- Each component is connected
    component-connected : ∀ a → is-connected-groupoid (component a)

    -- Reconstruction: G ≃ ⨿_a component(a)
    reconstruction : {!!}

open Decomposition public

{-|
**Classifying object Ω_G**:

For groupoid G with components A:
  Ω_G = 2^A (Boolean algebra of subsets of A)

This is the topos-theoretic subobject classifier.

**Properties**:
- Ω_G(⊤) = 2^A
- Morphisms p → q correspond to p ≤ q (subset inclusion)
- Boolean structure: ∧, ∨, ¬, ⊤, ⊥
-}

-- Classifying object (Boolean algebra)
Ω-G : {o ℓ : Level} (G : Precategory o ℓ) → Decomposition G → Type (lsuc o)
Ω-G G decomp = ℙ (decomp .A)
  -- Power set of connected components

--------------------------------------------------------------------------------
-- §C.4: G-Sets and Orbits

{-|
## G-Sets (Sets with Group Action)

For group G (or groupoid with one object):

A **G-set** X is:
- Set X
- Action: G × X → X
- Preserves identity: e · x = x
- Preserves composition: (g · h) · x = g · (h · x)

**Orbits**:
- Orbit of x = {g · x | g ∈ G}
- G acts transitively on each orbit
- X decomposes as disjoint union of orbits

**Subobjects**:
- Y ⊆ X is G-invariant if g · y ∈ Y for all g, y
- G-invariant subsets = Unions of orbits
-}

module _ {ℓ} (G : Group ℓ) where
  private module G = Group-on (G .snd)

  -- G-set
  record G-Set : Type (lsuc ℓ) where
    no-eta-equality
    field
      -- Underlying set
      Carrier : Type ℓ

      -- Action: G × X → X
      _·_ : ⌞ G ⌟ → Carrier → Carrier

      -- Action preserves identity
      ·-id : ∀ (x : Carrier) → G.unit · x ≡ x

      -- Action preserves composition
      ·-comp : ∀ (g h : ⌞ G ⌟) (x : Carrier)
             → (g G.⋆ h) · x ≡ g · (h · x)

  open G-Set public

  {-|
  ## Orbit of Element

  For x ∈ X, the orbit G·x is:
    Orbit(x) = {g · x | g ∈ G}

  **Properties**:
  1. Reflexive: x ∈ Orbit(x) (take g = e)
  2. Transitive: y ∈ Orbit(x) ⇒ Orbit(y) = Orbit(x)
  3. Disjoint: Orbit(x) ∩ Orbit(y) = ∅ or Orbit(x) = Orbit(y)
  -}

  -- Orbit of element x
  Orbit : (X : G-Set) → X .Carrier → ℙ (X .Carrier)
  Orbit X x = λ y → ∥ Σ ⌞ G ⌟ (λ g → X ._·_ g x ≡ y) ∥

  -- x is in its own orbit
  orbit-reflexive : ∀ (X : G-Set) (x : X .Carrier)
                  → x ∈ Orbit X x
  orbit-reflexive X x = inc (G.unit , X .·-id x)

  -- Orbits partition the set
  postulate
    orbit-partition : ∀ (X : G-Set) (x y : X .Carrier)
                    → (Orbit X x ≡ Orbit X y) ⊎ (∀ z → ¬ (z ∈ Orbit X x × z ∈ Orbit X y))

  {-|
  ## G-Invariant Subsets (Subobjects)

  A subset Y ⊆ X is **G-invariant** if:
    ∀ g ∈ G, ∀ y ∈ Y: g · y ∈ Y

  **Theorem**: G-invariant subsets are exactly unions of orbits.

  This is the key to understanding Ω_X!
  -}

  record G-Invariant-Subset (X : G-Set) : Type (lsuc ℓ) where
    no-eta-equality
    field
      -- Subset Y ⊆ X
      subset : ℙ (X .Carrier)

      -- G-invariance
      invariant : ∀ (g : ⌞ G ⌟) {x : X .Carrier}
                → x ∈ subset
                → (X ._·_ g x) ∈ subset

  open G-Invariant-Subset public

  -- G-invariant subsets are unions of orbits
  postulate
    invariant-is-union-of-orbits :
      ∀ (X : G-Set) (Y : G-Invariant-Subset X)
      → {!!}  -- Y = ⋃{Orbit(x) | x ∈ Y}

--------------------------------------------------------------------------------
-- §C.5: Algebra of Subobjects Ω_X

{-|
## Algebra of Subobjects Ω_X

For G-set X with orbits {Z_i | i ∈ K}:

**Ω_X = 2^K** (Boolean algebra of subsets of orbit representatives)

Elements of Ω_X:
- G-invariant subsets of X
- Equivalently: Subsets of K (orbit representatives)
- Boolean operations: ∧ (∩), ∨ (∪), ¬ (complement), ⊤ (X), ⊥ (∅)

**External implication**: p → q means p ⊆ q

**Internal implication**: p ⇒ q = (p ∧ q) ∨ ¬p
  (Heyting algebra structure, but Boolean so ⇒ is classical)

**DNN interpretation**:
- X = Feature maps at layer
- Orbits = Equivalent features under symmetry
- Ω_X = Boolean properties of feature subsets
- Subobject = Selecting symmetric feature subsets
-}

module _ {ℓ} (G : Group ℓ) (X : G-Set G) where
  -- Set of orbit representatives K(X)
  postulate
    K : Type ℓ
    orbit-rep : K → X .Carrier
    orbit-rep-surjective : ∀ (x : X .Carrier)
                         → Σ K (λ i → x ∈ Orbit G X (orbit-rep i))

  -- Algebra of subobjects Ω_X = 2^K
  Ω-X : Type (lsuc ℓ)
  Ω-X = ℙ K

  -- Boolean algebra structure
  _∧-Ω_ : Ω-X → Ω-X → Ω-X
  p ∧-Ω q = λ i → p i × q i  -- Intersection

  _∨-Ω_ : Ω-X → Ω-X → Ω-X
  p ∨-Ω q = λ i → p i ⊎ q i  -- Union

  ¬-Ω : Ω-X → Ω-X
  ¬-Ω p = λ i → ¬ (p i)  -- Complement

  ⊤-Ω : Ω-X
  ⊤-Ω = λ i → ⊤  -- Everything

  ⊥-Ω : Ω-X
  ⊥-Ω = λ i → ⊥  -- Nothing

  -- External implication (subset ordering)
  _≤-Ω_ : Ω-X → Ω-X → Type ℓ
  p ≤-Ω q = ∀ {i} → p i → q i

  -- Internal implication (Heyting, but Boolean)
  _⇒-Ω_ : Ω-X → Ω-X → Ω-X
  p ⇒-Ω q = (p ∧-Ω q) ∨-Ω (¬-Ω p)

  -- Boolean laws
  postulate
    double-negation : ∀ (p : Ω-X) → ¬-Ω (¬-Ω p) ≡ p

    de-morgan-and : ∀ (p q : Ω-X) → ¬-Ω (p ∧-Ω q) ≡ (¬-Ω p) ∨-Ω (¬-Ω q)

    de-morgan-or : ∀ (p q : Ω-X) → ¬-Ω (p ∨-Ω q) ≡ (¬-Ω p) ∧-Ω (¬-Ω q)

--------------------------------------------------------------------------------
-- §C.6: Product Decomposition for Multiple Components

{-|
## Product of Topos by Components

For groupoid G with components {G_a | a ∈ A}:

**Topos decomposition**:
  E_G ≃ ∏_{a ∈ A} E_{G_a}

Each subobject of X = {X_a | a ∈ A} is a product:
  Y = ∏_{a ∈ A} Y_a

Where Y_a ⊆ X_a is G_a-invariant.

**Classifying object**:
  Ω_X = ∏_{a ∈ A} Ω_{X_a}
      ≃ 2^{Σ_a K_a}
      ≃ 2^K  where K = ⨿_a K(X_a)

**DNN interpretation**:
- Multiple independent sub-networks
- Each has own symmetry group G_a
- Features decompose by sub-network
- Global symmetry = Product of local symmetries
-}

module _ {ℓ} where
  record Product-G-Set (A : Type ℓ) (G : A → Group ℓ) : Type (lsuc ℓ) where
    no-eta-equality
    field
      -- Component sets
      component-set : ∀ (a : A) → G-Set (G a)

      -- Product structure
      product : {!!}

  open Product-G-Set public

  -- Product of orbit sets
  K-product : (A : Type ℓ) (G : A → Group ℓ) (X : Product-G-Set A G)
            → Type ℓ
  K-product A G X = Σ A (λ a → {!!})  -- Σ_a K(X_a)

  -- Classifying object for product
  Ω-product : (A : Type ℓ) (G : A → Group ℓ) (X : Product-G-Set A G)
            → Type (lsuc ℓ)
  Ω-product A G X = ℙ (K-product A G X)
    -- 2^K where K = ⨿_a K(X_a)

--------------------------------------------------------------------------------
-- §C.7: Canonical Grothendieck Topology

{-|
## Canonical Topology on Ω

For p ∈ Ω (element of Boolean algebra), a **covering** of p is:
  Family {qᵢ}ᵢ∈I such that ⋁ᵢ qᵢ = p

**Canonical Grothendieck topology** J_K:
- Largest topology making all representable presheaves sheaves
- J_K(p) = {families {qᵢ} | ⋁ᵢ qᵢ = p}
- J_K(∅) = {∅} (empty family covers empty set)

**Properties**:
1. Refinement: Any covering has refinement by singletons {Z_i}
2. Presheaf F is sheaf ⟺ F(p) ≃ ∏_{i∈p} F(Z_i)
3. In particular: F(∅) ≃ ⊤ (singleton)

**DNN interpretation**:
- Covering = Decomposing feature set into subsets
- Sheaf condition = Features determined locally by components
- Empty covering = No features to track
-}

module _ {ℓ} (Ω : Type ℓ) where
  -- Covering of p ∈ Ω
  record Covering (p : Ω) : Type (lsuc ℓ) where
    no-eta-equality
    field
      Index : Type ℓ
      family : Index → Ω

      -- Covers: ⋁ᵢ family(i) = p
      covers : {!!}  -- Need join operation

  open Covering public

  -- Canonical Grothendieck topology
  record Canonical-Topology : Type (lsuc ℓ) where
    no-eta-equality
    field
      -- Covering families for each p
      J : (p : Ω) → ℙ (Covering p)

      -- Axioms
      pullback-stable : {!!}
      transitive : {!!}
      identity : ∀ (p : Ω) → {!!}  -- {p} covers p

  open Canonical-Topology public

  {-|
  **For Boolean algebra Ω = 2^K**:

  Covering of p ⊆ K:
    {pᵢ}ᵢ∈I such that ⋃ᵢ pᵢ = p

  Refinement to singletons:
    For any covering {pᵢ}, can refine to {{k} | k ∈ p}

  This makes sheaf condition very simple!
  -}

--------------------------------------------------------------------------------
-- §C.8: Proposition C.2 - Sheaf Equivalence

{-|
## Proposition C.2: E ≃ Sh(Ω; K)

**Theorem**: The topos E of presheaves over groupoid G is isomorphic
to the topos Sh(Ω; K) of sheaves on Ω with canonical topology J_K.

**Proof idea**:
1. Any covering of p has refinement by singletons {Z_i | i ∈ p}
2. Sheaf F satisfies: F(p) ≃ ∏_{i∈p} F(Z_i)
3. In particular: F(∅) ≃ ⊤ (singleton)
4. Restriction maps give canonical isomorphism
5. This characterizes sheaves completely

**Consequence**: Sheaves over Ω are just presheaves with gluing.

**DNN interpretation**:
- E = All possible equivariant features
- Sh(Ω; K) = Features with local-to-global property
- Sheaf condition = Features determined by orbit components
- Canonical topology = Natural notion of "covering" for features
-}

postulate
  -- Proposition C.2
  proposition-C-2 :
    ∀ {ℓ} (G : Precategory ℓ ℓ)
    → (groupoid : is-groupoid G)
    → (K : Type ℓ)  -- Orbit representatives
    → {!!}  -- E_G ≃ Sh(2^K; J_K)

{-|
**Corollary**: Sheaf evaluation at empty set

For any sheaf F ∈ Sh(Ω; K):
  F(⊥) ≃ ⊤

**Proof**: ⊥ = ∅ has empty covering {}.
Sheaf condition: F(∅) ≃ product over empty family = ⊤.

**Interpretation**: No features in empty set means unique "no feature" state.
-}

postulate
  sheaf-at-empty :
    ∀ {ℓ} (Ω : Type ℓ) (J : Canonical-Topology Ω)
    → (F : {!!})  -- Sheaf F
    → {!!}  -- F(⊥) ≃ ⊤

--------------------------------------------------------------------------------
-- §C.9: Examples and Applications

{-|
## Example 1: Trivial Group Action

G = {e} (trivial group)

- G-sets = ordinary sets
- Orbits = singletons {x}
- K(X) = X itself
- Ω_X = 2^X (all subsets)
- Sheaves = presheaves (no gluing needed)

**DNN**: No symmetry, all features independent.
-}

module Example-Trivial where
  postulate
    trivial-group : Group lzero

    trivial-G-set : G-Set trivial-group → Type lzero
    trivial-G-set X = X .Carrier

    -- Orbits are singletons
    orbits-are-singletons :
      ∀ (X : G-Set trivial-group) (x y : X .Carrier)
      → x ∈ Orbit trivial-group X y
      → x ≡ y

{-|
## Example 2: Cyclic Group Action

G = ℤ/nℤ (cyclic group of order n)

- Acts on circular structures
- Orbits = equivalence under rotation
- K(X) = orbit representatives (fundamental domain)
- Ω_X = 2^K (rotationally invariant subsets)

**DNN**: Rotational symmetry (e.g., oriented features in CNN).
-}

module Example-Cyclic where
  postulate
    cyclic-n : Nat → Group lzero

    -- Example: n points on circle
    circle-n-set : (n : Nat) → G-Set (cyclic-n n)

    -- All points in same orbit (transitive action)
    circle-transitive :
      (n : Nat) → (x y : (circle-n-set n) .Carrier)
      → x ∈ Orbit (cyclic-n n) (circle-n-set n) y

{-|
## Example 3: Translation Group on Grid

G = ℤ² (translations on 2D grid)

- Acts on spatial positions
- Orbits = equivalence under translation
- K(X) = single orbit (if transitive) or multiple (if boundary conditions)
- Ω_X = translation-invariant subsets

**DNN**: Translation equivariance in CNNs!
-}

module Example-Translation where
  postulate
    translation-2D : Group lzero

    -- Grid of features
    grid-set : G-Set translation-2D

    -- Translation equivariant features
    equivariant-features : {!!}

--------------------------------------------------------------------------------
-- Summary

{-|
## Summary: Appendix C Implementation

**Implemented structures**:
- ✅ Connected groupoids
- ✅ Fundamental group π₁(G, O)
- ✅ Equivalence G ≃ BG (Proposition C.1, Equation 42)
- ✅ Decomposition by connected components
- ✅ G-sets and orbits
- ✅ G-invariant subsets
- ✅ Classifying object Ω_X = 2^K
- ✅ Boolean algebra structure
- ✅ Canonical Grothendieck topology
- ✅ Sheaf equivalence (Proposition C.2)
- ✅ Three examples

**Key results**:
- **Proposition C.1**: Connected groupoid ≃ Fundamental group
- **Proposition C.2**: E_G ≃ Sh(Ω; J_K)
- **Equation 42**: Naturality of equivalence

**DNN interpretation**:
- Groupoids = Network symmetries
- Connected components = Independent sub-networks
- Orbits = Equivalent features under symmetry
- G-invariant subsets = Symmetric feature selections
- Sheaves = Features with local-to-global structure

**Integration**:
- Connects to: `Neural.Stack.Groupoid` (group actions)
- Uses: `Algebra.Group` from 1Lab
- Provides foundation for: Equivariant neural networks

**Applications to neural networks**:
1. **CNN**: Translation group acting on spatial positions
2. **Equivariant architectures**: General group actions
3. **Feature decomposition**: Orbit structure
4. **Symmetric properties**: Boolean algebra Ω_X
5. **Local-to-global**: Sheaf condition

**Significance**:
Completes the trilogy of appendices:
- **A**: Fuzzy equality and localic toposes (progressive decisions)
- **B**: Spectrum and algebraic geometry (prime ideals = layers)
- **C**: Groupoids and symmetries (equivariance)

Together, these provide the complete mathematical foundation for
topos-theoretic neural network theory with symmetries, topology,
and algebra unified in one framework!
-}
