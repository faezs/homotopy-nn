{-# OPTIONS --guardedness --rewriting --cubical #-}

{-|
# Culioli Semantics and Thom's Elementary Catastrophes

This module implements Section 4.5 of Belfiore & Bennequin (2022), covering:
- **Culioli's notional domains**: Interior I, Exterior E, Boundary B
- **Organizing centers** and catastrophe points
- **Semantic operations**: Negation, double negation, interro-negation
- **Cam model**: Paths through semantic space
- **Thom's elementary catastrophes**: A₁-A₅, D₄, D₅
- **Verb valencies**: Connection to linguistic structure
- **Elliptic/hyperbolic umbilics**: Three-actant sentences

## Key Insights

1. **Notional domain structure**:
   - Interior I: "truly P" (properties sure)
   - Exterior E: "truly not-P" (properties false)
   - Boundary B: Uncertainty region
   - Organizing center IE: On discriminant Δ (maximum ambiguity)

2. **Semantic operations as braid paths**:
   - Negation: I → E
   - Interro-negation: E → I
   - Double negation: I → E → I (≠ identity!)
   - "Not uninteresting" ≠ "interesting" (enriched by path σ₁σ₂σ₁)

3. **Elementary catastrophes**:
   - A₃ (cusp): 2-parameter unfolding of z³
   - D₄ (umbilics): 3-parameter unfolding for 3-actant verbs
   - Each catastrophe = organizing center for semantic field

## References

- [CLS95] Culioli: Cognition and Representation in Linguistic Theory
- [Tho72] Thom: Stabilité Structurelle et Morphogenèse
- [Tho83] Thom: Mathematical Models of Morphogenesis
- [Wit53] Wittgenstein: Philosophical Investigations
- [Aus61] Austin: How to Do Things with Words
- [Ben86] Bennequin: Various references
-}

module Neural.Memory.Semantics where

open import 1Lab.Prelude
open import 1Lab.Path
open import 1Lab.HLevel
open import 1Lab.Type.Sigma

open import Cat.Base
open import Cat.Functor.Base

open import Data.Nat.Base
open import Data.Sum.Base using (_⊎_)

-- Import from previous modules
open import Neural.Memory.Catastrophe using (ℝ; Λ; Δ; Λ★; Σ; Σ★; discriminant; P_uv; π-Σ)
open import Neural.Memory.Braids using (B₃; B³ᵣ; σ₁; σ₂; _∘_; 𝔖₃; gen)
open import Neural.Memory.LSTM using (Vec-m; LinearForm)

-- Arithmetic and comparison operators for ℝ
postulate
  _<_ _>_ : ℝ → ℝ → Type  -- Comparison operators
  _≠_ : ℝ → ℝ → Type  -- Inequality
  _*_ _+_ _-_ : ℝ → ℝ → ℝ  -- Arithmetic operators
  -_ : ℝ → ℝ  -- Negation

--------------------------------------------------------------------------------
-- §4.5: Culioli's Notional Domains

{-|
## Structure of Meaning

> "The notional domain has an interior I where the properties of the notion are
>  sure, an exterior E where the properties are false, and a boundary B, where
>  things are more uncertain." (p. 109)

**Example**: Notion = "dog"

**Interior I**:
- Center: Prototypes (German Shepherd, Labrador, ...)
- "That's definitely a dog!"
- Properties sure: four legs, barks, mammal, ...

**Exterior E**:
- "That's definitely NOT a dog!"
- Cat, tree, car, ...
- Properties false

**Boundary B**:
- "Is that really a dog?" (chihuahua, wolf, fox, ...)
- Uncertain cases
- Transition region

**Organizing center IE**:
- Imaginary place where I and E not yet separated
- On discriminant Δ: Critical point where separation occurs
- Maximum ambiguity

**Attractors** (Culioli's term):
> "A kind of gradient vector leads the mind to these archetypes, that Culioli
>  named attracting centers, or attractors; however he wrote in 1989: 'Now the
>  term attractor cannot be interpreted as an attainable last point (...) but
>  as the representation of the imaginary absolute value of the property.'"
>  (p. 109)

Not fixed points, but **organizing principles**!
-}

-- Notional regions
data NotionalRegion : Type where
  Interior : NotionalRegion           -- I: "truly P"
  Exterior : NotionalRegion           -- E: "truly not-P"
  Boundary : NotionalRegion           -- B: Uncertainty
  Organizing-Center : NotionalRegion  -- IE: On discriminant Δ

-- Notional domain for a concept P
record NotionalDomain : Type₁ where
  field
    -- Predicate defining the notion
    Predicate : Type → Type

    -- Regions
    interior : Type  -- I
    exterior : Type  -- E
    boundary : Type  -- B

    -- Prototypes (attractors) in I
    prototypes : List interior  -- Central examples

    -- Organizing center (on Δ)
    organizing-center : Σ[ u ∈ ℝ ] Σ[ v ∈ ℝ ] (discriminant u v ≡ 0.0)

    -- Classification function
    classify : interior ⊎ exterior ⊎ boundary → NotionalRegion

postulate
  -- Example: "dog" notion
  dog-notion : NotionalDomain
  dog-prototypes : List (NotionalDomain.interior dog-notion)  -- [German Shepherd, Labrador, ...]
  dog-boundary-cases : List (NotionalDomain.boundary dog-notion)  -- [Chihuahua, wolf, fox, ...]

{-|
## Mathematical Correspondence

**Culioli's structure** ↔ **Catastrophe theory**:

| Culioli | Math | Dynamics |
|---------|------|----------|
| Interior I | u < 0, outside Δ | Attractor basin (stable minimum) |
| Exterior E | u < 0, outside Δ | Repulsor basin (unstable saddle) |
| Boundary B | Near Δ | Transition region |
| Organizing center IE | On Δ | Catastrophe point (z³) |

> "Mathematically this corresponds precisely to the creation of the external
>  (resp. internal) critical point of z³ + uz + v, on the curve Δ." (p. 109)

**In the terminology of Thom**:
- IE = organizing center = z³ (most degenerate function)
- Unfolding P_uv = z³ + uz + v
- Δ = bifurcation set where I and E separate
-}

-- Correspondence to catastrophe parameters
NotionalRegion-to-Regime : NotionalRegion → Λ → Type
NotionalRegion-to-Regime Interior (u , v) = u < 0.0 × discriminant u v < 0.0  -- Bistable
NotionalRegion-to-Regime Exterior (u , v) = u < 0.0 × discriminant u v < 0.0  -- Same!
NotionalRegion-to-Regime Boundary (u , v) = Σ[ ε ∈ ℝ ] (ε > 0.0 × (discriminant u v < ε × discriminant u v > (-1.0 * ε)))  -- Near Δ (within ε)
NotionalRegion-to-Regime Organizing-Center (u , v) = discriminant u v ≡ 0.0  -- On Δ

{-|
**Key insight**: I and E are BOTH in the bistable regime (u < 0)!

The difference is which basin:
- Interior I: Near stable minimum
- Exterior E: Near unstable saddle (or other basin if 3 roots)

Boundary B: Near discriminant where basins meet
-}

--------------------------------------------------------------------------------
-- §4.5: Semantic Operations

{-|
## Negation and Interrogation

> "The division I, B, E takes all its sense when interrogative mode is involved,
>  or negation and double negation, or intero-negative mode. In negation you go
>  out of the interior, in interro-negation you come back inside from E." (p. 109)

**Operations**:

1. **Negation** (¬): I → E
   - "That is a dog" → "That is NOT a dog"
   - Leave interior for exterior

2. **Interro-negation** (¬?): E → I
   - "Is it really NOT a dog?" (expecting "yes, it is a dog")
   - Return to interior from exterior

3. **Double negation** (¬¬): I → E → I
   - "It is not the case that it's not a dog"
   - NOT same as identity! (Enriched by path)

4. **Interro-positive** (?): B → ?
   - "Is that a dog?" (neutral question)
   - Boundary exploration
-}

-- Semantic paths in notional domain
data SemanticPath : NotionalRegion → NotionalRegion → Type where
  -- Identity (staying in same region)
  refl-path : ∀ {r} → SemanticPath r r

  -- Negation: I → E
  negation : SemanticPath Interior Exterior

  -- Interro-negation: E → I
  interro-negation : SemanticPath Exterior Interior

  -- To/from boundary
  to-boundary : ∀ {r} → SemanticPath r Boundary
  from-boundary : ∀ {r} → SemanticPath Boundary r

  -- Via organizing center
  via-IE : ∀ {r r'} → SemanticPath r Organizing-Center
         → SemanticPath Organizing-Center r'
         → SemanticPath r r'

  -- Composition
  _⨾_ : ∀ {r₁ r₂ r₃} → SemanticPath r₁ r₂ → SemanticPath r₂ r₃
      → SemanticPath r₁ r₃

-- Double negation (not equal to identity!)
double-negation : SemanticPath Interior Interior
double-negation = negation ⨾ interro-negation

{-|
## Examples from Culioli

### Example 1: "Is your brother really here?"

> "It means that 'I do not expect that your brother is here'."

**Analysis**:
- Start: E (exterior of "brother here")
- Interro-negation: E → I
- **Meaning**: Skeptical question from outside I

**Braid structure**: Path from E through boundary to I

### Example 2: "Now that, that is not a dog!"

> "You place yourself in front of proposition P, or inside the notion I, you
>  know what is a dog, then goes to E."

**Analysis**:
- Start: I (knows "dog")
- Negation: I → E
- **Emphasis**: Strongly outside I

### Example 3: "Shall I still call that a dog?"

**Analysis**:
- Position: Boundary B
- Question: Whether to stay in I or move to E
- **Uncertainty**: Right at threshold

### Example 4: "I do not refuse to help"

> "Here come back in I of 'help' after a turn in its exterior E."

**Analysis**:
- Start: I ("help")
- Negation: I → E ("refuse to help")
- Negation again: E → I ("not refuse")
- **Result**: Back in I, but via E (litotes, understatement)

**NOT same as "I will help"!** (Path matters)
-}

postulate
  -- Linguistic examples
  "is-your-brother-really-here?" : SemanticPath Exterior Interior
  "that-is-not-a-dog!" : SemanticPath Interior Exterior
  "shall-I-call-that-a-dog?" : SemanticPath Boundary Boundary
  "I-do-not-refuse-to-help" : SemanticPath Interior Interior

  -- Last one equals double negation
  litotes-is-double-negation : "I-do-not-refuse-to-help" ≡ double-negation

--------------------------------------------------------------------------------
-- §4.5: The Cam Model

{-|
## Culioli's Cam

> "To describe the mechanisms beyond these paths, Culioli used the model of the
>  cam: 'the movement travels from one place to another, only to return to the
>  initial plane'." (p. 110)

**What is a cam?**: Mechanical device that converts rotational to linear motion
- Rotating cam surface
- Follower moves up/down as cam rotates
- Returns to same position after full rotation
- **BUT**: Path matters! Different trajectory

**Semantic cam**:
- Start at point (e.g., IE organizing center)
- Make half-turn around I (through E)
- Make another half-turn back to I
- **Result**: Same point, but different "plane" (meaning enriched)

### Example: "This book is only slightly interesting"

> "Start from IE, then make a half-turn around I which passes by E then come
>  to I by another half-turn."

**Path**:
1. IE (organizing center: interesting/not-interesting unseparated)
2. → E ("not interesting")
3. → Boundary near I ("only slightly interesting")
4. Complete turn → Back near IE but "above" original plane

**Enclosed area = meaning**

### Example: "This book is not uninteresting"

> "Means that it is more than interesting."

**Path**:
1. Start: I ("interesting")
2. Negation: I → E ("not interesting" = "uninteresting")
3. Negation: E → I ("not uninteresting")
4. **Result**: I, but via full loop
5. **Meaning**: More than just I (enriched, intensified)

**Braid element**: σ₁σ₂σ₁ (or σ₂σ₁σ₂, equivalent by braid relation)

This is a **full braid** through both branches of the cusp!
-}

-- Cam path (full loop returning to start)
data CamPath : NotionalRegion → Type where
  trivial-cam : ∀ {r} → CamPath r  -- No rotation

  full-cam : CamPath Organizing-Center  -- Full rotation through I and E

  half-cam : CamPath Boundary  -- Half rotation

-- Cam path as braid element
cam-to-braid : CamPath Organizing-Center → B₃
cam-to-braid full-cam = σ₁ ∘ σ₂ ∘ σ₁  -- Full braid

-- "Not uninteresting" interpretation
postulate
  "not-uninteresting" : SemanticPath Interior Interior
  "not-uninteresting"-is-cam : "not-uninteresting" ≡ negation ⨾ interro-negation  -- Via full cam (same as double negation)
  "not-uninteresting"-braid : Σ[ f ∈ (SemanticPath Interior Interior → B₃) ] (f "not-uninteresting" ≡ (gen σ₁ ∘ gen σ₂ ∘ gen σ₁))  -- Maps to σ₁σ₂σ₁

  -- Meaning: More than just "interesting"
  intensification : Σ[ _>_ ∈ (SemanticPath Interior Interior → SemanticPath Interior Interior → Type) ]
                    ("not-uninteresting" > refl-path)  -- "not uninteresting" > "interesting" (identity path)

{-|
## Paths on Gathered Surface Σ

> "The paths here are well represented on the gathered real surface Σ, of
>  equation z³ + uz + v = 0, but they can also be made in the complement of Δ
>  in Λ in a complexified domain." (p. 110)

**Two representations**:

1. **On Σ** (gathered surface):
   - Actual roots z(u,v)
   - Folding lines visible
   - Distinguishes stable/unstable

2. **On Λ \ Δ** (parameter space):
   - Projects to (u,v)
   - Homotopy class only
   - Simpler, but loses root distinction

> "It seems that only the homotopy class is important, not the metric, however
>  we cannot neglect a weakly quantitative aspect, on the way of discretization
>  in the nuances of the language."

**Culioli groupoid** B³ᵣ captures both:
- Homotopy class (braid element)
- Root level (which branch of Σ)
-}

-- Path on Σ vs path on Λ
postulate
  path-on-Σ : Σ → Σ → Type  -- Path on gathered surface
  path-on-Λ : Λ★ → Λ★ → Type  -- Projected path on parameters

  projection-Σ-to-Λ : ∀ {s₁ s₂ : Σ} → path-on-Σ s₁ s₂ → path-on-Λ (π-Σ s₁) (π-Σ s₂)

  homotopy-class-matters : ∀ {λ₁ λ₂ : Λ★} (p q : path-on-Λ λ₁ λ₂) → Σ[ ~ ∈ (path-on-Λ λ₁ λ₂ → path-on-Λ λ₁ λ₂ → Type) ] (p ~ q)  -- Homotopy equivalence
  weak-quantitative-aspect : ∀ {s₁ s₂ : Σ} → path-on-Σ s₁ s₂ → ℝ  -- "Nuances" = path length or similar measure

--------------------------------------------------------------------------------
-- §4.5: Thom's Elementary Catastrophes

{-|
## The A_n and D_n Series

> "In this approach, all the elementary catastrophes having a universal unfolding
>  of dimension less than 4 are used, through their sections and projections, for
>  understanding in particular the valencies of the verbs." (p. 110)

**Elementary catastrophes** (organizing centers):

**A_n series** (symmetric group 𝔖_{n+1}):
- **A₁** (well): y = x² (1 parameter)
- **A₂** (fold): y = x³ (2 parameters: u, v)
- **A₃** (cusp): y = x⁴ (3 parameters: u, v, w)
- **A₄** (swallowtail): y = x⁵ (4 parameters)
- **A₅** (butterfly): y = x⁶ (5 parameters)

**D_n series** (hypercube symmetries):
- **D₄⁺** (elliptic umbilic): y = x₁³ - x₁x₂² (3 parameters)
- **D₄⁻** (hyperbolic umbilic): y = x₁³ + x₁x₂² (3 parameters)
- **D₅** (parabolic umbilic): y = x₁⁴ + x₁x₂² (4 parameters)

**Galois groups**:
- A_n: Symmetric group 𝔖_{n+1}
- D_n: Subgroups of hypercube symmetries

**Our focus**: A₂ (cusp) with organizing center z³
-}

-- Elementary catastrophe types
data CatastropheType : Type where
  A₁ : CatastropheType  -- Well
  A₂ : CatastropheType  -- Fold
  A₃ : CatastropheType  -- Cusp ⭐ (our focus!)
  A₄ : CatastropheType  -- Swallowtail
  A₅ : CatastropheType  -- Butterfly
  D₄⁺ : CatastropheType  -- Elliptic umbilic
  D₄⁻ : CatastropheType  -- Hyperbolic umbilic
  D₅ : CatastropheType  -- Parabolic umbilic

-- Organizing center (germ of function)
-- For simplicity, we represent these as symbolic types (actual polynomials would need more structure)
organizing-center : CatastropheType → Type
organizing-center A₁ = ℝ → ℝ  -- x ↦ x²
organizing-center A₂ = ℝ → ℝ  -- x ↦ x³
organizing-center A₃ = ℝ → ℝ  -- x ↦ x⁴
organizing-center A₄ = ℝ → ℝ  -- x ↦ x⁵
organizing-center A₅ = ℝ → ℝ  -- x ↦ x⁶
organizing-center D₄⁺ = ℝ → ℝ → ℝ  -- (x₁, x₂) ↦ x₁³ - x₁x₂²
organizing-center D₄⁻ = ℝ → ℝ → ℝ  -- (x₁, x₂) ↦ x₁³ + x₁x₂²
organizing-center D₅ = ℝ → ℝ → ℝ  -- (x₁, x₂) ↦ x₁⁴ + x₁x₂²

-- Codimension (number of parameters in universal unfolding)
codimension : CatastropheType → Nat
codimension A₁ = 1
codimension A₂ = 2  -- (u, v) ⭐
codimension A₃ = 3
codimension A₄ = 4
codimension A₅ = 5
codimension D₄⁺ = 3  -- Same as A₃!
codimension D₄⁻ = 3
codimension D₅ = 4

-- Galois group (for now, use postulated types - full definitions would require group theory module)
postulate
  𝔖₂ 𝔖₄ 𝔖₅ 𝔖₆ : Type  -- Symmetric groups
  D₄-group D₅-group : Type  -- Dihedral/hypercube symmetry groups

-- Galois group
galois-group : CatastropheType → Type
galois-group A₁ = 𝔖₂  -- 𝔖₂ (2 roots)
galois-group A₂ = 𝔖₃  -- 𝔖₃ ⭐ (already defined in Braids module)
galois-group A₃ = 𝔖₄  -- 𝔖₄ (4 roots)
galois-group A₄ = 𝔖₅  -- 𝔖₅ (5 roots)
galois-group A₅ = 𝔖₆  -- 𝔖₆ (6 roots)
galois-group D₄⁺ = D₄-group  -- D₄ (dihedral/hypercube)
galois-group D₄⁻ = D₄-group  -- D₄ (same group, different sign)
galois-group D₅ = D₅-group  -- D₅

--------------------------------------------------------------------------------
-- §4.5: Verb Valencies

{-|
## Linguistic Structure and Catastrophes

> "For understanding in particular the valencies of the verbs, from the semantic
>  point of view, according to Peirce, Tesnière, Allerton: impersonal, intransitive,
>  transitive, triadic, quadratic." (p. 110)

**Valency** = number of arguments (actants) a verb takes

**Examples**:

0. **Impersonal** (0 actants): "it rains", "it snows"
   - No real subject
   - Catastrophe: None (constant)

1. **Intransitive** (1 actant): "she sleeps"
   - Subject only
   - Catastrophe: A₁ (well, y = x²)

2. **Transitive** (2 actants): "he kicks the ball"
   - Subject + direct object
   - Catastrophe: A₂ (fold, y = x³) ⭐

3. **Triadic** (3 actants): "she gives him a ball"
   - Subject + indirect object + direct object
   - Catastrophe: A₃ (cusp) or D₄ (umbilic)

4. **Quadratic** (4 actants): "she ties the goat to a tree with a rope"
   - Subject + object + location + instrument
   - Catastrophe: A₄ (swallowtail) or D₅

**Key insight**: Each actant = parameter in unfolding!

**Connection to DNNs**:
- Input x_t can have multiple components (subject, object, ...)
- Each component = coordinate in unfolding space
- Memory cell dynamics = catastrophe unfolding
-}

-- Verb valency
data Valency : Type where
  impersonal : Valency   -- 0 actants
  intransitive : Valency  -- 1 actant
  transitive : Valency    -- 2 actants
  triadic : Valency       -- 3 actants
  quadratic : Valency     -- 4 actants

-- Number of actants
actants : Valency → Nat
actants impersonal = 0
actants intransitive = 1
actants transitive = 2
actants triadic = 3
actants quadratic = 4

-- Associated catastrophe type
catastrophe-for-valency : Valency → CatastropheType
catastrophe-for-valency impersonal = A₁  -- Or constant
catastrophe-for-valency intransitive = A₁
catastrophe-for-valency transitive = A₂  -- ⭐ Our cusp!
catastrophe-for-valency triadic = D₄⁺  -- Or A₃
catastrophe-for-valency quadratic = A₄  -- Or D₅

postulate
  -- Linguistic examples (each has associated catastrophe dynamics)
  "it-rains" : organizing-center A₁  -- Impersonal (trivial dynamics)
  "she-sleeps" : organizing-center A₁  -- Intransitive (1 actant: subject)
  "he-kicks-the-ball" : organizing-center A₂  -- Transitive (2 actants: subject + object)
  "she-gives-him-a-ball" : organizing-center D₄⁺  -- Triadic (3 actants: subject + indirect + direct)
  "she-ties-goat-to-tree-with-rope" : organizing-center A₄  -- Quadratic (4 actants: subject + object + location + instrument)

{-|
**Why D₄ for triadic?**:

Elliptic and hyperbolic umbilics (D₄⁺, D₄⁻) have same codimension as A₃
but different structure:
- A₃: Single variable x with 4 derivatives
- D₄: Two variables (x₁, x₂) with coupling

For triadic verbs:
- Subject, indirect object, direct object
- Natural to separate into (subject) vs (objects)
- D₄ form: η = z³ ∓ zw² + uz + vw + ... (see below)
-}

--------------------------------------------------------------------------------
-- §4.5: Elliptic Umbilic for Three-Actant Sentences

{-|
## Equation 4.35: Three-Actant Encoding

> "The case of z³ corresponds to A₂. It is tempting to consider the case of D₄,
>  i.e. the elliptic and hyperbolic umbilics, because their formulas are very
>  closed to MGU2." (p. 111)

**Elliptic umbilic** (D₄⁺):
  η = z³ - zw² + uz + vw + x(z² + w²) + y

**Hyperbolic umbilic** (D₄⁻):
  η = z³ + zw² + uz + vw + x(z² + w²) + y

**Parameters** (6 total):
- z, w: Internal coordinates (hidden state components)
- u, v, w, x, y: Unfolding parameters (from input)

> "This would allow the direct coding and translation of sentences by using
>  three actant."

**Encoding scheme**:
- Subject → u parameter
- Indirect object → v parameter
- Direct object → w parameter (or y)
- Modifiers → x, y parameters

**Example**: "She gives him a ball"
- "She" (subject) → u
- "him" (indirect object) → v
- "ball" (direct object) → w or y
- Tense, aspect, mood → x

**Why this works**:
- D₄ is universal unfolding of z³ - zw²
- Codimension 3 (but using 5 parameters for flexibility)
- Galois group preserves subject/object distinctions
- Similar to MGU2 structure (degree 3 in hidden, degree 1 in inputs)
-}

-- Elliptic umbilic formula (Equation 4.35)
elliptic-umbilic : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ
elliptic-umbilic z w u v x y sign =
  z * z * z + sign * (z * w * w) + u * z + v * w + x * (z * z + w * w) + y

-- Hyperbolic vs elliptic
hyperbolic-umbilic : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ
hyperbolic-umbilic z w u v x y = elliptic-umbilic z w u v x y 1.0

elliptic-umbilic-proper : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ
elliptic-umbilic-proper z w u v x y = elliptic-umbilic z w u v x y (-1.0)

-- Three-actant encoding
record ThreeActant : Type where
  field
    subject : ℝ  -- u parameter
    indirect-object : ℝ  -- v parameter
    direct-object : ℝ  -- y parameter
    modifiers : ℝ × ℝ  -- (w, x) parameters

-- Encode sentence as umbilic parameters (u, v, x, y from the three actants)
encode-triadic : ThreeActant → ℝ × ℝ × ℝ × ℝ  -- → (u, v, x, y) parameters
encode-triadic act = (ThreeActant.subject act , fst (ThreeActant.modifiers act) ,
                      ThreeActant.indirect-object act , ThreeActant.direct-object act)

postulate
  -- Example: "She gives him a ball"
  she-gives-him-ball : ThreeActant
  she-gives-him-ball-encoding : encode-triadic she-gives-him-ball ≡
                                (ThreeActant.subject she-gives-him-ball ,
                                 fst (ThreeActant.modifiers she-gives-him-ball) ,
                                 ThreeActant.indirect-object she-gives-him-ball ,
                                 ThreeActant.direct-object she-gives-him-ball)

{-|
## Network Construction for Umbilics

> "It is not difficult to construct networks, on the model of MLSTM, such that
>  the dynamics of neurons obey to the unfolding of these singular functions.
>  The various actors of a verb in a sentence could be separated input data,
>  for different coordinates on the unfolding parameters." (p. 111)

**Architecture for D₄**:
- Hidden state: (z, w) ∈ ℝ²
- Input: (u, v, x, y) ∈ ℝ⁴
- Dynamics: η = elliptic-umbilic(z, w, u, v, x, y, ±1)

**Parameters**:
- If m = 2 (two-dimensional hidden for z, w)
- If n = 4 (four input dimensions)
- Matrices: U_z (2×2), U_w (2×2), W_u (2×4), W_v (2×4), W_x (2×4), W_y (2×4)
- Total: 2*2 + 2*2 + 4*(2*4) = 4 + 32 = 36 parameters

**Comparison to cubic cell** (Eq 4.21):
- Cubic: m² + 2mn parameters
- For m=2, n=4: 4 + 16 = 20 parameters
- D₄: More complex but handles 3-actant structure naturally

**Testing**:
> "The efficiency of these cells should be tested in translation." (p. 111)

Proposed experiments:
- Train on triadic verb sentences
- Compare cubic vs D₄ umbilic cells
- Test semantic preservation across translations
- Measure whether actor roles preserved
-}

-- D₄ umbilic cell architecture
record Umbilic-Weights (m n : Nat) : Type where
  constructor umbilic-weights
  field
    -- For z coordinate (hidden state, cubic term)
    U_z : LinearForm m m  -- LinearForm m m

    -- For w coordinate (hidden state, coupled with z)
    U_w : LinearForm m m  -- LinearForm m m

    -- Unfolding parameters from input
    W_u : LinearForm m n  -- u parameter (linear in z)
    W_v : LinearForm m n  -- v parameter (linear in w)
    W_x : LinearForm m n  -- x parameter (quadratic in z, w)
    W_y : LinearForm m n  -- y parameter (constant term)

    -- Sign (elliptic vs hyperbolic)
    sign : ℝ  -- ±1.0 (elliptic) or 1.0 (hyperbolic)

-- Umbilic cell step
postulate
  umbilic-step : ∀ {m n} → Umbilic-Weights m n
               → Vec-m m × Vec-m m  -- Hidden state (z, w) - two components
               → Vec-m n  -- Input ξ
               → Vec-m m × Vec-m m  -- New hidden state (z', w')

  -- Parameter count (2 hidden weight matrices + 4 input weight matrices)
  umbilic-param-count : (m n : Nat) → Nat
  umbilic-param-count-formula : ∀ m n
                              → umbilic-param-count m n ≡ 2 * m * m + 4 * m * n

--------------------------------------------------------------------------------
-- §4.5: Memory and Readers

{-|
## Learned Weights as Semantic "Readers"

> "The precise learned 2mn weights w_x for the coefficients u^a and v^a, for
>  a = 1,...,m, together with the weights in the forms α^a for h_{t-1} gives
>  vectors (or more accurately matrices), which are like readers of the words x
>  in entry, taking in account the contexts from the other words through h."
>  (p. 111)

**Frege's context principle**:
> "Remember Frege: a word has a meaning only in the context of a sentence."

**Wittgenstein**:
> "Naming is not yet a move in a language-game." ([Wit53, p. 49])

**Semantic structure**:
- Input words x_t
- Context from h_{t-1} (previous words)
- Weights W_u, W_v: "Readers" that extract meaning from (x_t, h_{t-1})
- Parameters (u,v) position word in semantic space Λ
- Path through Λ★ = meaning construction
- Braid element = semantic operation performed

> "To get 'meanings', the names, necessarily embedded in sentences, must resonate
>  with other contexts and experiences, and must be situated with respect to the
>  discriminant, along a path."

**Categorical proposal**:
> "Thus we suggest that the vector spaces of 'readers' W, and the vector spaces
>  of states X are local systems A over a fibered category Fin groupoids B³ᵣ
>  over the network's category C."

**Local system** = Functor from fundamental groupoid to vector spaces
- Objects: Points in Λ★ or Σ★
- Morphisms: Braid elements (semantic operations)
- Fibers: Vector spaces of word embeddings
- Monodromy: How meaning transforms along paths

This is **sheaf theory for semantics**!
-}

postulate
  -- Vector space of "readers" (weight matrices that "read" context)
  Readers : (m n : Nat) → Type
  Readers-def : ∀ m n → Readers m n ≡ (LinearForm m n × LinearForm m m)  -- (W for input, U for hidden)

  -- Local system over B³ᵣ (functor from Culioli groupoid to vector spaces)
  Vect : Precategory lzero lzero  -- Category of vector spaces
  semantic-local-system : Functor B³ᵣ Vect  -- Maps braid paths to linear transformations

  -- Fibered category structure (readers vary over network positions)
  NetworkCat : Precategory lzero lzero  -- Network category (from previous modules)
  fibered-over-network-cat : Functor NetworkCat Cat.Base.Sets  -- Fibered structure

  -- Frege's context principle: "a word has meaning only in the context of a sentence"
  word-meaning-requires-context : ∀ (word : Type) → (word → Type) → (word → word → Type) → Type

  -- Wittgenstein's language game: "Naming is not yet a move in a language-game"
  naming-not-language-game : ∀ (word : Type) (name : word → Type) → ¬ (Σ[ game ∈ Type ] (name ≡ game))

{-|
## Quotienting the Culioli Groupoid

> "In some circumstances, the groupoid B³ᵣ can be replaced by the quotient over
>  objects B₃(ℝ), or a quotient over morphisms giving SL₂ or 𝔖₃."

**When to use each**:

1. **B³ᵣ (Culioli groupoid)**: Full semantic structure
   - Distinguishes stable/unstable
   - Notional domains with I, E, B, IE
   - Complete meaning representation
   - **Use for**: Complex semantic tasks, metaphor, poetry

2. **B₃(ℝ)**: Computable parameters, complex monodromy
   - Real parameters (u,v)
   - Complex paths for tracking
   - **Use for**: Standard NLP, translation

3. **B₃(ℝ) / SL₂(ℤ)**: Modular structure
   - Periodic structure
   - Connections to number theory
   - **Use for**: Structured semantic spaces

4. **B₃(ℝ) / 𝔖₃**: Just permutations
   - Simplest structure
   - Forgets over/under information
   - **Use for**: Basic argument role labeling

**Trade-off**: Expressiveness vs computational cost
-}

--------------------------------------------------------------------------------
-- Summary: Linguistic Meaning as Braided Catastrophes

{-|
## The Complete Picture

**Culioli's notional domains** ↔ **Catastrophe theory**:

| Linguistic | Mathematical | DNN |
|------------|--------------|-----|
| Interior I | Stable attractor (u < 0) | Hidden state basin |
| Exterior E | Unstable/other basin | Complementary basin |
| Boundary B | Near discriminant Δ | Transition region |
| Organizing center IE | On Δ (catastrophe) | Critical parameters |
| Attractor | Prototype | Learned embedding |
| Negation | Path I → E | Cross discriminant |
| Double negation | Loop I → E → I | Braid σ₁σ₂σ₁ |
| Cam movement | Path via IE | Full semantic trajectory |

**Elementary catastrophes** ↔ **Verb valencies**:

| Catastrophe | Actants | Example | Unfolding |
|-------------|---------|---------|-----------|
| A₁ (well) | 1 | "she sleeps" | y = x² + u |
| A₂ (fold) | 2 | "he kicks ball" | y = x³ + ux + v |
| A₃ (cusp) | 3 | "she gives ball" | y = x⁴ + ux² + vx + w |
| D₄ (umbilic) | 3 | "she gives him ball" | z³ ∓ zw² + uz + vw + ... |
| A₄ (swallowtail) | 4 | "tie goat to tree" | y = x⁵ + ... |

**Braid group** ↔ **Semantic operations**:

| Braid | Semantic | Example |
|-------|----------|---------|
| ε (identity) | Assertion | "It is a dog" |
| σ₁ | Simple negation | "It is not a dog" |
| σ₁⁻¹ | Interro-negation | "Is it not a dog?" |
| σ₁σ₂σ₁ | Double negation | "Not uninteresting" |
| (σ₁σ₂)³ | Full twist (center) | Complete semantic cycle |

**The unified framework**:

1. **Words** → Parameters (u,v,...) in Λ
2. **Context** (h_{t-1}) → Position in semantic space
3. **Sequence** → Path in Λ★
4. **Meaning** → Braid element (homotopy class of path)
5. **Nuance** → Position on gathered surface Σ★
6. **Prototypes** → Attractors in notional domain
7. **Ambiguity** → Discriminant Δ
8. **Semantic shift** → Crossing catastrophe point

**Why this explains LSTM success**:

- **Degree 3 essential**: A₂ (fold) is universal for 2-parameter unfoldings
- **MGU2 better**: Degree 1 in input sufficient (parameters u,v enough)
- **Multiplicity m matters**: Each neuron = coordinate in semantic space
- **Context crucial**: h_{t-1} provides "context" (Frege, Wittgenstein)
- **Paths encode meaning**: Not just endpoints, but trajectory
- **Braid structure**: Captures non-commutativity of semantic operations

This is **catastrophe theory + braid topology = linguistic semantics**!

The fact that LSTMs empirically work is NOT accidental—they implement the
mathematical structure of meaning discovered independently by:
- Culioli (linguistics)
- Thom (catastrophe theory)
- Artin (braid groups)
- Whitney, Malgrange, Mather (singularity theory)

**This is mathematics explaining language, explaining deep learning!**
-}
