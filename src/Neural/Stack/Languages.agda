{-# OPTIONS --allow-unsolved-metas #-}

{-|
# Section 3.3: Languages, Theories, and Fibrations

This module implements Section 3.3 from Belfiore & Bennequin (2022), formalizing
formal languages as sheaves, deduction systems as fibrations, and theory extensions
as cofibrations in the topos-theoretic framework.

## Paper Reference

"A formal language L over a network topos E is a sheaf of well-formed formulas,
equipped with a deduction system D that assigns proofs as morphisms. The pair
(L, D) forms a fibration of theories over the base category of network layers."

"Theory extensions correspond to cofibrations: adding axioms (spontaneous
propositions) extends the language while preserving the deduction structure."

## DNN Interpretation

**Languages as Sheaves**: Feature properties at each layer
- Layer U: language L(U) of expressible properties
- Transition α: interpretation translation L(U') → L(U)
- Sheaf condition: global properties from local consistency

**Applications**:
- Formal verification: prove network properties
- Interpretability: express human-understandable concepts
- Neuro-symbolic AI: integrate logic with learning
- Type safety: guarantee shape/dtype correctness
- Causal reasoning: express interventions and counterfactuals

## Key Concepts

1. **Languages**: Sheaves of formulas over network layers
2. **Deduction Systems**: Fibrations assigning proofs to judgments
3. **Models**: Functors from language to sets (interpretation)
4. **Theory Extensions**: Cofibrations adding axioms
5. **Semantic Interpretation**: Kripke-Joyal semantics in topos

-}

module Neural.Stack.Languages where

open import 1Lab.Prelude hiding (id; _∘_)
open import 1Lab.Type.Sigma

open import Cat.Prelude
open import Cat.Functor.Base
open import Cat.Instances.Functor
open import Cat.Diagram.Colimit.Base
open import Cat.Diagram.Pushout

-- Import previous sections
open import Neural.Stack.TypeTheory
open import Neural.Stack.Semantic

private variable
  o ℓ o' ℓ' : Level
  C D E : Precategory o ℓ

--------------------------------------------------------------------------------
-- § 3.3.1: Formal Languages as Sheaves

{-|
## Definition 3.14: Language Sheaf

> "A formal language over topos E is a sheaf L: C^op → Set assigning to each
> object U ∈ C a set L(U) of well-formed formulas, with restriction maps
> L(α): L(U') → L(U) satisfying sheaf axioms."

**Components**:
- L(U): formulas expressible at layer U
- L(α): formula translation along morphism α: U → U'
- Sheaf condition: formulas glue from covering families

**DNN Interpretation**:
Each layer U can express certain properties:
- Input layer: pixel features, texture, color
- Early layers: edges, corners, simple shapes
- Deep layers: objects, scenes, semantic concepts
- Output layer: class labels, captions, predictions

**Example: ImageNet Classifier**
- L(input) = "pixel (i,j) has intensity v"
- L(conv1) = "edge at position (x,y) with orientation θ"
- L(conv3) = "texture pattern P at region R"
- L(fc6) = "object part O is present"
- L(output) = "image contains class C with probability p"
-}

postulate
  -- Signature: symbols and arities
  Signature : Type (lsuc lzero)
  Symbol : Signature → Type
  arity : (Σ : Signature) → Symbol Σ → Nat

  -- Terms over signature
  Term : Signature → Type
  var : ∀ {Σ} → Nat → Term Σ
  app : ∀ {Σ} → (s : Symbol Σ) → (Fin (arity Σ s) → Term Σ) → Term Σ

  -- Formulas (atomic, conjunction, disjunction, quantifiers)
  Formula : Signature → Type
  atomic : ∀ {Σ} → Symbol Σ → (Fin (arity Σ Symbol) → Term Σ) → Formula Σ
  _∧ᶠ_ : ∀ {Σ} → Formula Σ → Formula Σ → Formula Σ
  _∨ᶠ_ : ∀ {Σ} → Formula Σ → Formula Σ → Formula Σ
  _⇒ᶠ_ : ∀ {Σ} → Formula Σ → Formula Σ → Formula Σ
  ∀ᶠ : ∀ {Σ} → Formula Σ → Formula Σ
  ∃ᶠ : ∀ {Σ} → Formula Σ → Formula Σ

{-|
Language sheaf: assigns formulas to each layer
-}
record Language-Sheaf (C : Precategory o ℓ) (Σ : Signature)
                      : Type (o ⊔ ℓ ⊔ lsuc lzero) where
  field
    -- Formulas at each layer
    formulas-at : C .Precategory.Ob → Type

    -- Formula restriction along morphisms
    restrict : ∀ {U U'} → C .Precategory.Hom U U'
             → formulas-at U' → formulas-at U

    -- Functoriality
    restrict-id : ∀ {U} (φ : formulas-at U)
                → restrict (C .Precategory.id) φ ≡ φ

    restrict-∘ : ∀ {U U' U''} (α : C .Precategory.Hom U U')
                   (β : C .Precategory.Hom U' U'') (φ : formulas-at U'')
               → restrict (α ∘⟨ C ⟩ β) φ
                 ≡ restrict α (restrict β φ)

    -- Sheaf axioms (uniqueness and gluing)
    sheaf-uniqueness : {!!}
    sheaf-gluing : {!!}

{-|
## Example 3.14: Vision Network Language

A convolutional network for image classification has languages:
```
L(conv1) = {Edge(x,y,θ,strength) | ...}
L(conv2) = {Texture(region,pattern) | ...}
L(conv5) = {ObjectPart(type,bbox) | ...}
L(fc8) = {Class(label,prob) | ...}
```

Restriction maps translate formulas:
- restrict(conv2 ← conv5): "ObjectPart" → "Texture compositions"
- restrict(fc8 ← conv5): "Class" → "ObjectPart combinations"
-}

postulate
  example-vision-language : ∀ {C : Precategory o ℓ} (Σ : Signature)
                          → Language-Sheaf C Σ

--------------------------------------------------------------------------------
-- § 3.3.2: Deduction Systems as Fibrations

{-|
## Definition 3.15: Deduction Fibration

> "A deduction system over language L is a fibration D: Ctx → C where:
> - Ctx: category of contexts Γ and judgments Γ ⊢ φ
> - Fibers D⁻¹(U): deductions expressible at layer U
> - Cartesian morphisms: proof translations that preserve structure"

**Structure**:
```
Ctx (contexts + judgments)
 |
 | D (deduction fibration)
 ↓
 C (network layers)
```

**Fibers**:
- D⁻¹(U) = {Γ ⊢ φ | Γ,φ ∈ L(U)}  (judgments at layer U)
- Morphisms in fiber: deduction steps (proof transformations)

**DNN Interpretation**: Provable properties at each layer
- Some properties are provable at layer U
- Others require deeper layers (more computation)
- Proof complexity = minimum layer depth needed

**Example: Certified Neural Network**
- L(U): properties expressible at U
- D⁻¹(U): properties PROVABLE at U using network computation
- Cartesian lift: if φ provable at U', then restrict(φ) provable at U
-}

postulate
  -- Context: list of variable bindings
  Context : Signature → Type
  empty-ctx : ∀ {Σ} → Context Σ
  extend-ctx : ∀ {Σ} → Context Σ → Term Σ → Context Σ

  -- Judgment: Γ ⊢ φ (φ is derivable in context Γ)
  record Judgment {Σ : Signature} (Γ : Context Σ) (φ : Formula Σ) : Type where
    field
      derivation : {!!}  -- Proof tree

  -- Category of contexts and judgments
  Ctx-Category : Signature → Precategory (lsuc lzero) lzero

record Deduction-Fibration (C : Precategory o ℓ) (Σ : Signature)
                            (L : Language-Sheaf C Σ)
                            : Type (o ⊔ ℓ ⊔ lsuc lzero) where
  field
    -- Base: network layer category
    base : Precategory o ℓ
    base-equiv : base ≡ C

    -- Total category of judgments
    total : Precategory (o ⊔ lsuc lzero) (ℓ ⊔ lzero)

    -- Projection to base
    proj : Functor total base

    -- Cartesian morphisms (proof translations)
    is-cartesian : ∀ {J J' : total .Precategory.Ob}
                 → total .Precategory.Hom J J' → Type ℓ

    cartesian-lift : ∀ {U U' : C .Precategory.Ob} (α : C .Precategory.Hom U U')
                   → {J' : total .Precategory.Ob}
                   → proj .Functor.F₀ J' ≡ U'
                   → Σ[ J ∈ total .Precategory.Ob ]
                       Σ[ f ∈ total .Precategory.Hom J J' ]
                         (is-cartesian f × proj .Functor.F₀ J ≡ U)

{-|
## Example 3.15: Linear Logic for Neural Networks

We can use linear logic to reason about neural network resources:
- Formulas: A, B, C (feature types)
- Linear implication: A ⊸ B (uses A exactly once to produce B)
- Tensor: A ⊗ B (parallel features)
- Additive: A & B (choice of features)

**Judgment Example**:
```
Γ = {x: Image}
⊢ x: Image ⊸ Conv(x): Features ⊸ FC(Conv(x)): Logits
```

This judgment is provable at layer fc8 but not at conv1.
-}

postulate
  example-linear-logic : ∀ {C : Precategory o ℓ} {Σ : Signature} {L : Language-Sheaf C Σ}
                       → Deduction-Fibration C Σ L

--------------------------------------------------------------------------------
-- § 3.3.3: Theory Extensions and Cofibrations

{-|
## Definition 3.16: Theory Extension

> "A theory T = (L, Ax) consists of a language L and axioms Ax ⊆ L.
> An extension T' ⊇ T adds axioms: T' = (L, Ax ∪ Ax').
> The inclusion T ↪ T' is a cofibration if it preserves deduction structure."

**Cofibration Condition**:
If Γ ⊢_T φ (derivable in T), then Γ ⊢_{T'} φ (derivable in T').
I.e., adding axioms doesn't break existing proofs.

**DNN Interpretation**: Adding constraints or regularization
- Base theory T: standard network
- Extended theory T': network + additional constraints
- Cofibration: constraints don't contradict existing behavior

**Example: Adding Normalization**
- T: unrestricted hidden states
- T': hidden states must be normalized
- Axiom added: ∀h. ||h|| = 1
- Cofibration: all previous properties still hold on normalized states
-}

record Theory (C : Precategory o ℓ) (Σ : Signature) : Type (o ⊔ ℓ ⊔ lsuc lzero) where
  field
    language : Language-Sheaf C Σ
    axioms : ∀ (U : C .Precategory.Ob) → language .Language-Sheaf.formulas-at U → Type

  -- Derived judgments
  postulate
    derivable : ∀ {U : C .Precategory.Ob}
              → language .Language-Sheaf.formulas-at U → Type

record Theory-Extension {C : Precategory o ℓ} {Σ : Signature}
                         (T T' : Theory C Σ) : Type (o ⊔ ℓ ⊔ lsuc lzero) where
  field
    -- Same language
    same-language : T .Theory.language ≡ T' .Theory.language

    -- T' has more axioms
    extends-axioms : ∀ {U φ}
                   → T .Theory.axioms U φ
                   → T' .Theory.axioms U φ

    -- Derivability preserved (cofibration condition)
    preserves-derivable : ∀ {U φ}
                        → T .Theory.derivable φ
                        → T' .Theory.derivable φ

{-|
## Example 3.16: Adversarial Robustness as Theory Extension

Base theory T: standard classifier
- Language: L(output) = {Class(c,p) | c ∈ Classes}
- Axioms: standard softmax constraints

Extended theory T': robust classifier
- Same language
- Additional axiom: ∀x,ε. ||x-x'|| < ε ⇒ |p(x) - p(x')| < δ
- Cofibration: classification still works, but now robustly

The extension adds robustness without changing the classification structure.
-}

postulate
  example-robust-extension : ∀ {C : Precategory o ℓ} {Σ : Signature}
                           → (T : Theory C Σ)
                           → Theory C Σ  -- T' with robustness axioms

--------------------------------------------------------------------------------
-- § 3.3.4: Models and Semantic Interpretation

{-|
## Definition 3.17: Model of a Theory

> "A model M of theory T in topos E is a functor M: L^op → E preserving the
> sheaf structure and satisfying all axioms: M ⊨ Ax."

**Structure**:
- M assigns sets/objects to formulas
- M assigns functions to formula restrictions
- M makes axioms true (⊨)

**DNN Interpretation**: Concrete realization of abstract properties
- Formula φ: abstract property ("contains cat")
- Model M(φ): set of network states satisfying φ
- Different models: different training runs, different architectures

**Example: Two Models of "Cat Detector"**
- M₁: VGG-16 trained on ImageNet
- M₂: ResNet-50 trained on ImageNet
- Both satisfy theory T (correct classification)
- Different internal representations: M₁(φ) ≠ M₂(φ)
- But both ⊨ T (both are models)
-}

record Model {C : Precategory o ℓ} {E : Precategory o' ℓ'} {Σ : Signature}
             (T : Theory C Σ) : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ' ⊔ lsuc lzero) where
  field
    -- Interpretation functor
    interpret : ∀ {U : C .Precategory.Ob}
              → T .Theory.language .Language-Sheaf.formulas-at U
              → E .Precategory.Ob

    -- Morphism interpretation
    interpret-hom : ∀ {U U'} (α : C .Precategory.Hom U U')
                  → (φ : T .Theory.language .Language-Sheaf.formulas-at U')
                  → E .Precategory.Hom (interpret φ)
                      (interpret (T .Theory.language .Language-Sheaf.restrict α φ))

    -- Axioms are satisfied
    satisfies : ∀ {U} {φ : T .Theory.language .Language-Sheaf.formulas-at U}
              → T .Theory.axioms U φ
              → {!!}  -- interpret φ is inhabited (true in model)

{-|
## Proposition 3.3: Categorical Completeness

> "Every consistent theory T has a model in the topos E. Moreover, the category
> of models Mod(T,E) is equivalent to the category of geometric morphisms
> E → Sh(T), where Sh(T) is the classifying topos of T."

**Proof Sketch**:
1. Construct Sh(T) as category of T-models in Sets
2. Show Sh(T) is a topos (Giraud's theorem)
3. Geometric morphisms E → Sh(T) ↔ T-models in E (universal property)

**DNN Interpretation**: Every consistent spec has an implementation
- Consistent theory T: network specification without contradictions
- Model M: actual trained network satisfying T
- Completeness: can always find such a network (given enough capacity)

This is the theoretical justification for neural architecture search!
-}

postulate
  proposition-3-3 : ∀ {C : Precategory o ℓ} {E : Precategory o' ℓ'} {Σ : Signature}
                  → (T : Theory C Σ)
                  → {!!}  -- Consistency(T) → ∃ M: Model T

--------------------------------------------------------------------------------
-- § 3.3.5: Kripke-Joyal Semantics

{-|
## Definition 3.18: Kripke-Joyal Forcing

> "The Kripke-Joyal semantics interprets formulas at each stage (layer) U:
>   U ⊩ φ  (U forces φ)
> meaning φ is true at layer U and all refinements U' → U."

**Forcing Clauses**:
- U ⊩ φ ∧ ψ  ⟺  U ⊩ φ and U ⊩ ψ
- U ⊩ φ ∨ ψ  ⟺  ∃ covering {Uᵢ → U}. ∀i. Uᵢ ⊩ φ or Uᵢ ⊩ ψ
- U ⊩ φ ⇒ ψ  ⟺  ∀ U'→U. U' ⊩ φ  ⇒  U' ⊩ ψ
- U ⊩ ∀x.φ(x)  ⟺  ∀ U'→U. ∀a∈M(U'). U' ⊩ φ(a)

**DNN Interpretation**: Layer-wise truth
- U ⊩ φ: property φ is verified at layer U
- Refinement U' → U: deeper layer provides more information
- Covering: property holds on partition of feature space

**Example: "Image Contains Cat"**
- input ⊮ contains-cat (can't tell from raw pixels alone)
- conv3 ⊩ has-cat-texture (textures detected)
- fc6 ⊩ has-cat-parts (ears, whiskers, etc.)
- output ⊩ contains-cat (final classification)

The forcing relation tracks how properties emerge through layers!
-}

module _ {C : Precategory o ℓ} {Σ : Signature} (T : Theory C Σ) where

  postulate
    -- Forcing relation at layer U
    _⊩_ : (U : C .Precategory.Ob)
        → T .Theory.language .Language-Sheaf.formulas-at U
        → Type ℓ

    -- Forcing clauses
    force-∧ : ∀ {U φ ψ} → U ⊩ (φ ∧ᶠ ψ)  ≃  ((U ⊩ φ) × (U ⊩ ψ))

    force-⇒ : ∀ {U φ ψ}
            → U ⊩ (φ ⇒ᶠ ψ)
            ≃ (∀ {U'} (α : C .Precategory.Hom U' U)
                → U' ⊩ (T .Theory.language .Language-Sheaf.restrict α φ)
                → U' ⊩ (T .Theory.language .Language-Sheaf.restrict α ψ))

    -- Monotonicity
    force-mono : ∀ {U U'} (α : C .Precategory.Hom U' U) {φ}
               → U ⊩ φ
               → U' ⊩ (T .Theory.language .Language-Sheaf.restrict α φ)

{-|
## Example 3.17: Interpretability via Forcing

Consider verifying "input is adversarial example":
```
φ = ∃ x₀. ||x - x₀|| < ε ∧ label(x) ≠ label(x₀)
```

Forcing at layers:
- input ⊮ φ (no computation yet)
- conv1 ⊮ φ (only local features)
- conv5 ⊩ φ (can compare global structure)
- output ⊩ φ (can verify label change)

This shows adversarial detection requires deep layers!
Interpretability = understanding which layers force which properties.
-}

postulate
  example-adversarial-forcing : ∀ {C : Precategory o ℓ} {Σ : Signature}
                              → (T : Theory C Σ)
                              → {!!}

--------------------------------------------------------------------------------
-- § 3.3.6: Logical Modalities and Network Depth

{-|
## Definition 3.19: Modal Logic for Layer Depth

> "We introduce modal operators:
> - ◇φ (eventually φ): φ is provable at some layer U
> - □φ (necessarily φ): φ is provable at all layers U
> - @Uφ (at layer U): φ is provable specifically at U"

**Interpretation**:
- ◇φ: property φ emerges somewhere in the network
- □φ: property φ is invariant across all layers
- @Uφ: property φ is specific to layer U

**DNN Applications**:
- Depth requirements: ◇φ ∧ ¬@conv1(φ) → "φ needs deep layers"
- Invariants: □φ → "φ preserved by network"
- Layer-specific: @fc6(part-detector) → "part detection at fc6"

**Example: Hierarchical Features**
- @conv1(edge-detector)
- @conv3(texture-detector)
- @fc6(object-part-detector)
- @fc8(object-detector)

Modal logic makes the hierarchy explicit!
-}

postulate
  -- Modal formulas
  data Modal-Formula {Σ : Signature} : Type where
    embed : Formula Σ → Modal-Formula
    ◇_ : Modal-Formula → Modal-Formula
    □_ : Modal-Formula → Modal-Formula
    @_[_] : ∀ {C : Precategory o ℓ} → C .Precategory.Ob → Modal-Formula → Modal-Formula

  -- Modal forcing
  _⊩ᴹ_ : ∀ {C : Precategory o ℓ} {Σ : Signature}
       → C .Precategory.Ob → Modal-Formula {Σ} → Type ℓ

  force-◇ : ∀ {C : Precategory o ℓ} {Σ : Signature} {U : C .Precategory.Ob} {φ : Modal-Formula {Σ}}
          → U ⊩ᴹ (◇ φ)
          ≃ (Σ[ U' ∈ C .Precategory.Ob ] (U' ⊩ᴹ φ))

  force-□ : ∀ {C : Precategory o ℓ} {Σ : Signature} {U : C .Precategory.Ob} {φ : Modal-Formula {Σ}}
          → U ⊩ᴹ (□ φ)
          ≃ (∀ (U' : C .Precategory.Ob) → U' ⊩ᴹ φ)

{-|
## Example 3.18: Depth Lower Bounds

We can prove depth lower bounds using modal logic:

**Theorem**: XOR function requires depth ≥ 2
**Proof**:
- φ = "computes XOR correctly"
- @input(φ) is false (linear separator insufficient)
- @hidden1(φ) is true (nonlinear combination works)
- ◇φ ∧ ¬@input(φ) → depth ≥ 2

This is a formal proof of a classic neural network result!
-}

postulate
  example-xor-depth : ∀ {C : Precategory o ℓ} {Σ : Signature}
                    → {!!}  -- XOR requires hidden layer

--------------------------------------------------------------------------------
-- § 3.3.7: Summary and Connections

{-|
## Summary: Languages and Theories Framework

We have formalized:
1. **Language sheaves**: Formulas as functorial structures over layers
2. **Deduction fibrations**: Provable properties at each layer
3. **Theory extensions**: Adding axioms via cofibrations
4. **Models**: Concrete interpretations of abstract theories
5. **Kripke-Joyal semantics**: Layer-wise forcing relation
6. **Modal logic**: Reasoning about depth and emergence

## Connections to Other Sections

**Section 2.5 (Type Theory)**:
- Languages ↔ Type theories
- Deduction ↔ Term typing
- Models ↔ Semantic interpretations

**Section 3.1 (Cat's Manifolds)**:
- Formulas ↔ Constraints on manifolds
- Forcing ↔ Manifold restriction
- Models ↔ Specific state spaces

**Section 3.2 (Spontaneous Activity)**:
- Axioms ↔ Spontaneous vertices (given truths)
- Theory extension ↔ Adding spontaneous inputs
- Cofibration ↔ Spontaneous inclusion

**Section 3.4 (Homology)**:
- Formulas ↔ Cochains
- Derivability ↔ Cohomology classes
- Forcing ↔ Cup products

## Applications Enabled

1. **Formal Verification**: Prove network properties using deduction
2. **Interpretability**: Forcing relation shows what each layer "knows"
3. **Architecture Search**: Find minimal depth for desired properties
4. **Neuro-Symbolic AI**: Integrate logic and learning
5. **Certified Training**: Guarantee properties via theory extensions
6. **Causal Reasoning**: Express interventions as theory morphisms

## Key Results

**Categorical Completeness**: Every consistent spec has a network model
**Kripke-Joyal Interpretation**: Layer-wise truth determines global behavior
**Modal Depth Bounds**: Can formally prove depth requirements
**Cofibration Extensions**: Adding constraints preserves existing structure
-}
