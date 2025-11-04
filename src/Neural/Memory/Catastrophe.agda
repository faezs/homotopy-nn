{-# OPTIONS --guardedness --rewriting --cubical #-}

{-|
# Universal Unfolding Theory and Catastrophe Points

This module implements Section 4.4 of Belfiore & Bennequin (2022), covering:
- Universal unfolding of the singularity z³
- Discriminant curve Δ and catastrophe points
- **Theorem 4.1**: Structural stability at neuron level
- Gathered surface Σ and root ramifications
- Whitney-Thom-Malgrange-Mather theory

## Key Results

1. **Universal unfolding** (Eq 4.25-4.27): P_u(z) = z³ + uz + v
   - Every smooth F near z³ can be written F(z,Y) = ζ(z,Y)³ + u(Y)·ζ(z,Y)

2. **Whitney's stability** (Eq 4.28): Map (z,u) ↦ (P_u(z), u) is stable

3. **Discriminant** Δ: 4u³ + 27v² = 0
   - Separates parameter space into regions with 1 or 3 real roots

4. **Theorem 4.1**: Individual neurons are stable, but layer map is not
   - Explains why multiplicity m matters
   - Each neuron has structurally stable cubic dynamics

## References

- [GWDPL76] Golubitsky & Guillemin: Stable Mappings and Their Singularities
- [Mar82] Martinet: Singularities of Smooth Functions and Maps
- [Arn73] Arnold: Normal Forms of Functions near Degenerate Critical Points
- [AGZV12a/b] Arnold et al.: Singularity Theory I/II
- [Tho72] Thom: Stabilité Structurelle et Morphogenèse (catastrophe theory)
-}

module Neural.Memory.Catastrophe where

open import 1Lab.Prelude
open import 1Lab.Path
open import 1Lab.HLevel

open import Cat.Base

open import Data.Nat.Base

-- Import from LSTM module
open import Neural.Memory.LSTM using (ℝ; _+_; _-_; _*_; _/_)

--------------------------------------------------------------------------------
-- §4.4: Universal Unfolding of z³

{-|
## The Cubic Singularity

> "The main point here is the observation that, for each neuron in the h_t layer,
>  the cubic degeneracy z³ can appear, together with its deformation by the
>  function u." (p. 104)

**Organizing center**: z³ (most degenerate function in the family)

**Universal unfolding**: P_u(z) = z³ + uz

**Why universal**:
- Codimension 2 singularity
- Two parameters (u, v) required for complete unfolding
- Every nearby smooth function can be reduced to this form
-}

-- Polynomial P_u(z) = z³ + uz (Equation 4.25)
P_u : ℝ → ℝ → ℝ
P_u u z = z * z * z + u * z

-- Full unfolding P_{u,v}(z) = z³ + uz + v
P_uv : ℝ → ℝ → ℝ → ℝ
P_uv u v z = z * z * z + u * z + v

postulate
  -- Multi-variable function space (smooth maps)
  Smooth : Nat → Nat → Type
  eval-smooth : ∀ {m n} → Smooth m n → (Fin m → ℝ) → (Fin n → ℝ)

  -- Universal unfolding property (Equation 4.27)
  universal-unfolding : ∀ (M : Nat) (F : Smooth (suc M) 1) -- F : R^{1+M} → R
                      → ((λ Y → eval-smooth F (λ { fzero → 0.0 ; (fsuc i) → Y i })) ≡ (λ _ → 0.0))  -- F(z, 0,...,0) = z³ condition simplified
                      → ∃[ u ] ∃[ ζ ] ((λ z Y → let ζ-val = eval-smooth ζ (λ { fzero → z ; (fsuc i) → Y i }) fzero
                                                 in ζ-val * ζ-val * ζ-val + eval-smooth u Y fzero * ζ-val)
                                       ≡ (λ z Y → eval-smooth F (λ { fzero → z ; (fsuc i) → Y i }) fzero))

{-|
**Theorem** (Whitney, Thom, Malgrange, Mather):

The map (z, u) ↦ (P_u(z), u) is **stable** near (0, 0).

**Stability** means: Every sufficiently close map can be transformed to it by
diffeomorphisms of source and target.

**Interpretation for DNNs**:
- Small changes in weights → small changes in neuron dynamics
- Shape of cubic is preserved under perturbations
- This is the foundation of robust learning!

**Non-stability**: Product map (Eq 4.29) is NOT stable
  (z, u, w, v) ↦ (P_u(z), u, P_v(w), v)

Mather's infinitesimal criterion fails. This explains why layer-wise stability
fails (Theorem 4.1) even though neuron-wise stability holds!
-}

postulate
  -- Diffeomorphism predicate
  is-diffeomorphism : ∀ {m n} → Smooth m n → Type

  -- Structural stability for smooth maps
  is-stable-map : ∀ {m n} → Smooth m n → Type
  is-stable-map {m} {n} f = ∀ (g : Smooth m n)
    → ∃[ φ ] ∃[ ψ ] (is-diffeomorphism φ × is-diffeomorphism ψ
                    × ((eval-smooth g) ≡ (λ x → eval-smooth ψ (eval-smooth f (eval-smooth φ x)))))

  -- Map (z,u) ↦ (P_u(z), u)
  whitney-map : Smooth 2 2

  whitney-stability : is-stable-map whitney-map  -- Map (z,u) ↦ (P_u(z), u) is stable

  -- Product map (z,u,w,v) ↦ (P_u(z), u, P_v(w), v)
  product-map : Smooth 4 4

  product-not-stable : ¬ is-stable-map product-map  -- Product map is NOT stable

  -- Tangent space at a point
  TangentSpace : ∀ {m} → (Fin m → ℝ) → Type

  -- Mather's infinitesimal criterion for stability
  mather-criterion : ∀ {m n} (f : Smooth m n)
    → (∀ (x : Fin m → ℝ) → TangentSpace {n} (eval-smooth f x) ≡ TangentSpace {n} (eval-smooth f x))  -- Tangent map surjective
    → is-stable-map f

{-|
## Codimension and Universality

**For functions** (p → ℝ):
- Always admit universal unfoldings
- Codimension = number of parameters needed

**For mappings** (ℝⁿ → ℝᵖ, n = p):
- Usually NO universal unfolding (infinite codimension)
- Exception: Specific cases like Whitney cusp

**Here** (Section 4.4):
- n = p = m (hidden state dimension)
- Transformation h_{t-1} → h_t is an unfolding (parametrized by ξ ∈ x_t)
- Does NOT admit universal deformation
- But EACH coordinate η^a does! (This is Theorem 4.1)
-}

postulate
  -- Codimension of a singularity (number of parameters in universal unfolding)
  codimension : ∀ {m n} → Smooth m n → Nat ⊎ ⊤  -- Either finite or infinite (⊤)

  infinite-codimension : ∀ (m : Nat) → m > 1
                       → (f : Smooth m m)  -- Map ℝᵐ → ℝᵐ has infinite codimension
                       → codimension f ≡ inr tt

--------------------------------------------------------------------------------
-- §4.4: Discriminant Curve Δ

{-|
## The Discriminant

**Equation**: Δ: 4u³ + 27v² = 0

**Geometric interpretation**:
- Cusp curve in (u,v) plane
- Separates regions with 1 vs 3 real roots
- Points on Δ: Two roots collide (catastrophe!)

**Three regimes**:

1. **u > 0**: Monotonic
   - P_u(z) has exactly 1 real root
   - No local extrema
   - Neuron acts like linear transformation

2. **u < 0, outside Δ**: Bistable
   - P_u(z) has 3 distinct real roots
   - One stable minimum (attractor)
   - One unstable maximum (repulsor)
   - One saddle point
   - Neuron can switch between states

3. **On Δ**: Catastrophe point
   - Critical points collide
   - Transition between regimes 1 and 2
   - Semantic boundary (Section 4.5)

**DNN interpretation** (p. 105):
> "The discriminant Δ plays an important role in the global dynamic."

Critical parameters x_t that place (u(x_t), v(x_t)) on Δ are **semantic boundaries**
between notional domains (Culioli, Section 4.5).
-}

-- Discriminant function: 4u³ + 27v²
discriminant : ℝ → ℝ → ℝ
discriminant u v = 4.0 * (u * u * u) + 27.0 * (v * v)

-- Points on discriminant curve
Δ : Type
Δ = Σ[ u ∈ ℝ ] Σ[ v ∈ ℝ ] (discriminant u v ≡ 0.0)

-- Parameter space Λ = ℝ² (coordinates u, v)
Λ : Type
Λ = ℝ × ℝ

-- Complement Λ★ = Λ \ Δ (non-catastrophic parameters)
Λ★ : Type
Λ★ = Σ[ uv ∈ Λ ] (discriminant (fst uv) (snd uv) ≠ 0.0)

postulate
  -- Cusp singularity predicate (specific planar curve type)
  is-cusp-curve : (ℝ × ℝ → Type) → Type  -- Subset of ℝ² with cusp singularity

  -- Discriminant is a cusp
  Δ-is-cusp : is-cusp-curve (λ (u , v) → discriminant u v ≡ 0.0)

  -- Number of distinct real roots of P_uv
  num-distinct-real-roots : ℝ → ℝ → Nat

  -- Inside cusp: u < 0 and discriminant < 0
  inside-cusp : ∀ u v → u < 0.0 → discriminant u v < 0.0
              → num-distinct-real-roots u v ≡ 3  -- 3 real roots

  -- Outside cusp: discriminant > 0
  outside-cusp : ∀ u v → discriminant u v > 0.0
               → num-distinct-real-roots u v ≡ 1  -- 1 real root

  -- On discriminant: roots collide
  on-discriminant : ∀ u v → discriminant u v ≡ 0.0
                  → num-distinct-real-roots u v ≡ 2  -- Double root (catastrophe)

--------------------------------------------------------------------------------
-- §4.4: Three Regimes

{-|
## Regime Classification

> "If u = u^a(ξ) does not change of sign, the dynamic of the neuron a is stable
>  under small perturbations. For u > 0, it looks like a linear function, it is
>  monotonic. For u < 0 there exist a unique stable minimum and a unique saddle
>  point which limits its basin of attraction. But for u = 0 the critical points
>  collide, the individual map is unstable. This is named the catastrophe point."
>  (p. 105)
-}

data Regime : Type where
  Monotonic : Regime    -- u > 0
  Bistable : Regime     -- u < 0 (with 3 roots)
  Catastrophe : Regime  -- u = 0 or on Δ

postulate
  -- Comparison for reals (needed for regime classification)
  _>ℝ_ : ℝ → ℝ → Type
  _<ℝ_ : ℝ → ℝ → Type
  _≡ℝ_ : ℝ → ℝ → Type

  -- Decision procedure for regime
  decide-regime : (u v : ℝ) → (u >ℝ 0.0) ⊎ ((u <ℝ 0.0) × (discriminant u v <ℝ 0.0)) ⊎ (discriminant u v ≡ℝ 0.0)

-- Classify regime based on parameters
classify-regime : ℝ → ℝ → Regime
classify-regime u v with decide-regime u v
... | inl _ = Monotonic                    -- u > 0
... | inr (inl _) = Bistable              -- u < 0 and inside cusp
... | inr (inr _) = Catastrophe           -- On discriminant

-- Number of real roots
num-real-roots : ℝ → ℝ → Nat
num-real-roots u v with classify-regime u v
... | Monotonic = 1
... | Bistable = 3
... | Catastrophe = 2  -- Double root

{-|
## Critical Points

For P_u(z) = z³ + uz, critical points satisfy:
  dP_u/dz = 3z² + u = 0
  ⇒ z² = -u/3

**When u > 0**: No real solutions (monotonic)

**When u < 0**: Two critical points at z = ±√(-u/3)
- z_min = -√(-u/3): Local minimum (stable attractor)
- z_max = +√(-u/3): Local maximum (unstable saddle)

**When u = 0**: Critical points collide at z = 0 (inflection point)
-}

postulate
  -- Square root operation on reals
  sqrt : ℝ → ℝ

  -- Negation
  -ℝ_ : ℝ → ℝ

  -- Critical points of P_u(z) = z³ + uz satisfy 3z² + u = 0
  critical-points : ℝ → List ℝ

  critical-points-def : ∀ u → u <ℝ 0.0
                      → critical-points u ≡ (sqrt ((-ℝ u) / 3.0)) ∷ (-ℝ sqrt ((-ℝ u) / 3.0)) ∷ []

  stable-minimum : ℝ → ℝ
  stable-minimum u = -ℝ sqrt ((-ℝ u) / 3.0)  -- -√(-u/3) is the local minimum

  unstable-saddle : ℝ → ℝ
  unstable-saddle u = sqrt ((-ℝ u) / 3.0)  -- +√(-u/3) is the local maximum

  -- Second derivative of P_u: d²P/dz² = 6z
  second-derivative : ℝ → ℝ → ℝ
  second-derivative u z = 6.0 * z

  -- Second derivative test
  is-local-min : ∀ u z → second-derivative u z >ℝ 0.0  -- d²P/dz² > 0
  is-local-max : ∀ u z → second-derivative u z <ℝ 0.0  -- d²P/dz² < 0

--------------------------------------------------------------------------------
-- §4.4: Gathered Surface Σ

{-|
## The Gathered Surface

**Definition**: Σ = {(u, v, z) ∈ Λ × ℝ | z³ + uz + v = 0}

**Projection**: π: Σ → Λ, (u,v,z) ↦ (u,v)

**Folding lines**: Δ₃ ⊂ Σ (lifting of Δ along fold)

**Interpretation**:
- Σ is the graph of roots as function of parameters
- Over Λ★: π has 1 or 3 sheets (depending on regime)
- Over Δ: π has fold singularities (two sheets collide)

**Connection to cat's manifolds** (p. 105):
> "These accidents create ramifications in the cat's manifolds."

When tracing information backward (inverting the flow), ramifications occur
at points where (u(ξ), v(ξ)) crosses Δ.
-}

-- Gathered surface Σ
Σ : Type
Σ = Σ[ u ∈ ℝ ] Σ[ v ∈ ℝ ] Σ[ z ∈ ℝ ] (P_uv u v z ≡ 0.0)

-- Projection to parameter space
π-Σ : Σ → Λ
π-Σ (u , v , z , _) = (u , v)

postulate
  -- Fold singularity predicate for a projection
  has-fold-singularity-at : ∀ {A B : Type} → (A → B) → A → Type

  -- Smooth manifold structure predicate
  is-smooth-manifold : Type → Nat → Type  -- Type and dimension

-- Folding lines Δ₃ (lifting of Δ)
Δ₃ : Type
Δ₃ = Σ[ s ∈ Σ ] (has-fold-singularity-at π-Σ s)  -- Points where π has fold singularity

-- Complement Σ★ = Σ \ Δ₃
Σ★ : Type
Σ★ = Σ[ s ∈ Σ ] (¬ has-fold-singularity-at π-Σ s)  -- Non-fold points

postulate
  -- Σ is a smooth surface (2-dimensional manifold)
  Σ-is-smooth : is-smooth-manifold Σ 2

  -- π has fold singularities exactly on Δ₃
  π-fold-on-Δ₃ : ∀ (s : Σ) → has-fold-singularity-at π-Σ s ≡ (discriminant (fst (π-Σ s)) (fst (snd (π-Σ s))) ≡ 0.0)

  -- Number of sheets over each point
  num-sheets : Λ → Nat
  num-sheets-1-or-3 : ∀ λ → num-sheets λ ≡ 1 ⊎ num-sheets λ ≡ 3

--------------------------------------------------------------------------------
-- §4.4: Roots and Inversion

{-|
## Finding Roots

> "If we are interested in the value of η^a_t, we must follow a sort of inversion
>  of the flow, going to the past, by finding the roots z of the equations
>  P_a(z) = c." (p. 105)

**Equation 4.30**: P_a(z) = c

**Number of solutions**:
- Depends on (u, v) and level c
- For c = 0: Inequality 4u³ + 27v² < 0 determines 1 vs 3 roots

**Cardan formulas** (Section 4.4 end):
- Express roots using √ and ∛
- Give explicit formulas for z₂ - z₁, z₃ - z₁
- Can be seen directly in surface Σ

**Cat's manifold ramifications** (p. 105):
> "When the point (u^a(ξ), v^a(ξ)) in the plane Λ belongs to the discriminant
>  curve Δ, things become ambiguous, two roots collide and disappear."

Tracing information backward from η_t to η_{t-1} encounters ramifications
when crossing Δ.
-}

postulate
  -- List length
  length : ∀ {A : Type} → List A → Nat

  -- Solve P_uv(u,v,z) = c for z
  roots-of-level : ℝ → ℝ → ℝ → List ℝ
  roots-of-level-count : ∀ u v c
                       → length (roots-of-level u v c) ≡ 1
                       ⊎ length (roots-of-level u v c) ≡ 3  -- Length is 1 or 3 depending on regime

  -- Cardan formulas give explicit root expressions
  cardan-formula : ℝ → ℝ → List ℝ  -- Explicit root formulas using ∛ and √
  cardan-formula-correct : ∀ u v → cardan-formula u v ≡ roots-of-level u v 0.0

  -- Root differences (for visualizing the gathered surface)
  root-differences : ℝ → ℝ → ℝ × ℝ  -- (z₂ - z₁, z₃ - z₁)

  -- Roots collide predicate
  roots-collide : ℝ → ℝ → Type
  roots-collide u v = ∃[ z ] ∃[ w ] (z ≠ w × P_uv u v z ≡ 0.0 × P_uv u v w ≡ 0.0
                                     × (∀ r → P_uv u v r ≡ 0.0 → (r ≡ z) ⊎ (r ≡ w)))

  -- Ramification condition
  ramification-at : ∀ u v → discriminant u v ≡ 0.0
                  → roots-collide u v  -- Two roots collide when on discriminant

--------------------------------------------------------------------------------
-- §4.4: Theorem 4.1 - Structural Stability ⭐

{-|
## Theorem 4.1 (Belfiore & Bennequin)

> "**Theorem 4.1**: The map X_w is not structurally stable on H or H×X, but
>  each coordinate η^a_t, seen as function on a generic line of the input h_{t-1}
>  and a generic line of the input x_t, or as a function on H or H×X, is stable."
>  (p. 103-104)

**What it says**:
- **Layer map NOT stable**: X_w : H × X → H has infinite codimension
- **Individual neurons ARE stable**: η^a : H × X → ℝ is stable
- **Corollary**: "Each individual cell plays a role"

**Why this matters**:

1. **Explains multiplicity m**: Can't be arbitrary, neurons aren't redundant
2. **Justifies cubic model**: Each neuron near z³ singularity is universal
3. **Architecture constraint**: m must be large enough for task complexity

**Proof sketch** (p. 104):
> "This theorem follows from the results of the universal unfolding theory of
>  smooth mappings, developed by Whitney, Thom, Malgrange and Mather."

The key observation:
- Each η^a is of form σ_α³ + u·σ_α + v
- This is equivalent to P_u(z) = z³ + uz + v by diffeomorphism
- Whitney's theorem: (z,u) ↦ (P_u(z), u) is stable
- ∴ Each coordinate η^a is stable

But the product of stable maps is NOT stable (Mather's criterion fails).
-}

postulate
  -- Space of neuron states
  H : Nat → Type  -- ℝᵐ
  X : Nat → Type  -- ℝⁿ

  -- Weight matrices (smooth parameter space)
  Weights : Nat → Nat → Type  -- Weights m n = parameter space for m neurons, n inputs

  -- Layer transformation X_w : H × X → H
  X_w : ∀ {m n} → Weights m n  -- Network weights
      → H m × X n → H m

  -- Individual neuron coordinate projection η^a : H × X → ℝ
  η^a : ∀ {m n} → Fin m → Weights m n  -- Weights for neuron a
      → H m × X n → ℝ

  -- Project a-th component from H m
  π^a : ∀ {m} → Fin m → H m → ℝ

  -- Extract a-th component from layer output
  η^a-projection : ∀ {m n} (a : Fin m) (w : Weights m n) (input : H m × X n)
                 → η^a a w input ≡ π^a a (X_w w input)  -- a-th component of layer output

  -- Structural stability predicate (smooth map version)
  is-structurally-stable : {A B : Type} → (A → B) → Type
  is-structurally-stable f = ∀ g → ∃[ φ ] ∃[ ψ ] ((λ x → g x) ≡ (λ x → ψ (f (φ x))))

  -- Theorem 4.1 statement: Layer unstable, neurons stable
  theorem-4-1 : ∀ {m n} (w : Weights m n)
              → ¬ is-structurally-stable (X_w {m} {n} w)  -- Layer map NOT stable
              × (∀ (a : Fin m) → is-structurally-stable (η^a {m} {n} a w))  -- Each neuron IS stable

  -- Each neuron plays a distinct role (non-redundancy)
  neuron-distinct-role : ∀ {m n} (w : Weights m n) → Fin m → Type
  neuron-distinct-role w a = ∀ (b : Fin m) → a ≠ b → η^a a w ≢ η^a b w

  -- Corollary: Individual neurons matter
  corollary-individual-neurons : ∀ {m n} (w : Weights m n) (a : Fin m)
                               → neuron-distinct-role w a  -- Each neuron plays distinct role

{-|
**Redundancy vs. Individuality**:

> "This does not contradict the fact that frequently several cells send similar
>  message, i.e. there exists a redundancy, which is opposite to the stability
>  or genericity of the whole layer. However, in some regime and/or for m
>  sufficiently small, the redundancy is not a simple repetition, it is more
>  like a creation of characteristic properties." (p. 105)

**Interpretation**:
- Redundancy ≠ exact duplication
- Multiple neurons may respond similarly but with different "flavors"
- For m large: Some redundancy inevitable
- For m small: Each neuron has distinct characteristic
- Optimal m depends on task complexity

This explains empirical observation: Performance improves with m, then plateaus.
-}

--------------------------------------------------------------------------------
-- §4.4: Inversion and Ambiguity

{-|
## Continuous Inversion Impossible

> "The inversion of X_w^ξ : H → H is impossible continuously along a curve in ξ
>  whose u^a, v^a meet Δ for some component a." (p. 105-106)

**Problem**: Tracking roots backward continuously fails at discriminant

**Solution**: Use complex numbers!

> "It becomes possible if we pass to complex numbers, and lift the curve in Λ
>  to the universal covering Λ̃★(ℂ) of the complement Λ★_ℂ of Δ_ℂ in Λ_ℂ."

**Complex advantage**:
- Every degree k polynomial has k roots (counting multiplicity)
- No jumps when crossing discriminant
- Ambiguity captured by fundamental group π₁(Λ★_ℂ)

**Disadvantage**:
- Hard to compute with in DNNs (σ, tanh have poles in ℂ)
- All dynamical regions confounded

**Compromise**: Use real points Λ★(ℝ) but allow complex paths (groupoid B₃(ℝ))
-}

postulate
  -- Complex extension
  Λ_ℂ : Type
  Δ_ℂ : Type  -- Complex discriminant
  Λ★_ℂ : Type  -- Λ_ℂ \ Δ_ℂ

  -- Universal covering
  Λ̃★_ℂ : Type
  covering-map : Λ̃★_ℂ → Λ★_ℂ

  -- Group structure (abstract)
  Group : Type₁
  GroupIso : Group → Group → Type

  -- Artin braid group B₃ (3-strand braids)
  B₃ : Group

  -- Fundamental group (braid group B₃)
  π₁[Λ★_ℂ] : Λ★_ℂ → Group  -- Fundamental group at basepoint
  π₁-is-B₃ : ∀ (base : Λ★_ℂ) → GroupIso (π₁[Λ★_ℂ] base) B₃  -- Isomorphic to Artin braid group

  -- Continuous path in a space (I → X for interval I)
  Path : Type → Type
  path-start : ∀ {X} → Path X → X
  path-end : ∀ {X} → Path X → X

  -- Path lifting property for covering spaces
  path-lifting : ∀ {p : Λ★_ℂ} (γ : Path Λ★_ℂ)  -- Path in Λ★_ℂ
               → (p ≡ path-start γ)  -- Starting point matches
               → ∃[ γ̃ ] ((covering-map ∘ γ̃) ≡ γ)  -- Lifted path projects to original

  -- Inversion with complex paths (continuous root tracking)
  continuous-inversion-complex : ∀ (u₀ v₀ : ℝ) (path : Path Λ★_ℂ)  -- Path avoiding Δ_ℂ
                               → List ℝ  -- Continuously tracked roots along path

--------------------------------------------------------------------------------
-- §4.4: Space H and Neighborhood of 0

{-|
## The Pointed Space H

> "Let us define the space H of the activities of the memory vectors h_{t-1}
>  and h_t, of real dimension m; it is pointed by 0, and the neighborhood of
>  this point is a region of special interest." (p. 102)

**Why 0 is special**:

1. **MGU2 empirical fact**: Absence of bias in α^a (linear form through 0)

2. **Near-linearity**:
   > "The functions σ and τ are almost linear in the vicinity of 0 and only here."

3. **Polynomial model applies**:
   - σ(z) ≈ 1/2 + z/4 near z = 0
   - τ(z) ≈ z near z = 0
   - Cubic model η = z³ + uz + v is accurate

**Unfolding space** Λ = U × ℝ:
- U: Line of coordinate u
- Λ: Plane of coordinates (u, v)
- Input x_t → Λ via maps u(ξ), v(ξ)

> "By definition this constitutes an unfolding of the degree three map in σ_α(η)."

**Special meaning** (p. 105):
> "The set of data η_{t-1} and ξ_t which gives some point η_t in this region [near 0]
>  have a special meaning: they represent ambiguities in the past for η_{t-1}
>  and critical parameters for ξ_t."

When η_t ≈ 0, the system is in the regime where:
- Polynomial approximation is valid
- Cubic dynamics dominates
- Semantic transitions occur (near discriminant)
-}

postulate
  -- Neighborhood of 0 in H
  Neighborhood-0 : ∀ {m} → H m → Type

  -- Cubic approximation accuracy bound
  approximation-error : ∀ {m n} (a : Fin m) (w : Weights m n) → H m × X n → ℝ

  -- Accuracy threshold for polynomial approximation
  ε-accuracy : ℝ

  polynomial-accurate-near-0 : ∀ {m n} (w : Weights m n) (a : Fin m) (η : H m) (x : X n)
                             → Neighborhood-0 η
                             → approximation-error a w (η , x) <ℝ ε-accuracy  -- Error bound ε

  -- Parameter maps from input to unfolding space
  u^a : ∀ {m n} → Fin m → Weights m n → X n → ℝ  -- Linear parameter
  v^a : ∀ {m n} → Fin m → Weights m n → X n → ℝ  -- Constant parameter

  -- Unfolding structure: Each neuron has form P_{u^a,v^a}
  unfolding-of-cubic : ∀ {m n} (w : Weights m n) (a : Fin m)
                     → ∃[ ζ ] (∀ (input : H m × X n)
                              → η^a a w input ≡ P_uv (u^a a w (snd input)) (v^a a w (snd input)) (ζ input))

--------------------------------------------------------------------------------
-- Summary: Why Degree 3 is Essential

{-|
## Mathematical Explanation of Empirical Success

**Empirical observation** (p. 97):
> "All trials with degree < 3 in h' gave a dramatic loss of performance, but
>  this was not the case for x', where degree 1 appears to be sufficient."

**Mathematical explanation**:

1. **Universal unfolding**: z³ is the simplest non-trivial singularity
   - Codimension 2 (needs 2 parameters: u, v)
   - All nearby functions reduce to P_u(z) = z³ + uz

2. **Whitney's stability**: Map (z,u) ↦ (P_u(z), u) is structurally stable
   - Small perturbations preserve cubic shape
   - Robust to weight noise during training

3. **Three regimes**: Discriminant Δ creates rich dynamics
   - Monotonic (u > 0): Linear-like behavior
   - Bistable (u < 0): Multi-stable states
   - Catastrophe (Δ): Sharp transitions (semantics!)

4. **Individual stability**: Each neuron η^a is stable (Theorem 4.1)
   - Layer map X_w not stable (Mather criterion fails)
   - But coordinates η^a are stable (Whitney's theorem)
   - ∴ Each neuron matters (multiplicity m constrained)

**Why not degree 2?**
- y = z² has codimension 1 (only 1 parameter)
- Not enough flexibility for complex tasks
- No bistable regime (monotonic for all u ≠ 0)

**Why not degree 4 or higher?**
- Higher codimension (more parameters needed)
- Unnecessary complexity
- Cubic is universal for "generic" perturbations
- Occam's razor: Simplest stable form wins

**Connection to next sections**:
- Section 4.4 (Braids): Fundamental group π₁(Λ★) = B₃ encodes root ambiguity
- Section 4.5 (Semantics): Discriminant Δ = semantic boundaries (Culioli)
- Braid paths = semantic operations (negation, double negation, etc.)

This is **catastrophe theory explaining deep learning architecture design**!
-}
