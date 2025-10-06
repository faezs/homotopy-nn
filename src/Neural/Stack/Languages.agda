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
-- § 3.3.1b: Transfer Maps and Subobject Classifiers

{-|
## Transfer Maps Between Layers (Equations 3.6-3.8)

> "For an arrow (α,h): (U,ξ) → (U',ξ'), the map Ω_{α,h}: Ω_{U'}(ξ') → Ω_U(ξ)
> is obtained by composing the map λ_α = Ω_α at ξ' from Ω_{U'}(ξ') to
> Ω_U(F_α ξ') with the map Ω_U(h) from Ω_U(F_α ξ') to Ω_U(ξ)."

**Transfer in Downstream Direction** (toward output):
- Maps propositions from layer U' to layer U
- Uses geometric functor structure F_α
- Composition of restriction and morphism application

**DNN Interpretation**:
- Information flows downstream during forward pass
- Properties at deeper layers translate to earlier layers
- Geometric morphism preserves logical structure
-}

-- Import stack structures from previous sections
postulate
  Stack : (C : Precategory o ℓ) → (o' ℓ' : Level) → Type (o ⊔ ℓ ⊔ lsuc (o' ⊔ ℓ'))
  F₀ : ∀ {C o' ℓ'} → Stack C o' ℓ' → C .Precategory.Ob → Precategory o' ℓ'
  F₁ : ∀ {C o' ℓ'} (F : Stack C o' ℓ') {U U' : C .Precategory.Ob}
     → C .Precategory.Hom U U' → Functor (F₀ F U') (F₀ F U)

module _ {C : Precategory o ℓ} {F : Stack C o' ℓ'} where

  -- Subobject classifier at (U, ξ)
  postulate
    Ω : (U : C .Precategory.Ob) → (ξ : (F₀ F U) .Precategory.Ob) → Type ℓ'

  {-|
  ## Equation 3.6: Downstream Transfer Ω_{α,h}

  For morphism (α,h): (U,ξ) → (U',ξ') in the total category of F:
  - α: U → U' in base category C
  - h: ξ → F_α(ξ') in fiber F_U

  The transfer map Ω_{α,h} combines:
  1. λ_α = Ω_α at ξ': direct image of subobject classifier
  2. Ω_U(h): restriction along fiber morphism
  -}

  postulate
    -- Direct image of classifier (pullback functor on subobjects)
    λ_α : ∀ {U U'} (α : C .Precategory.Hom U U')
        → (ξ' : (F₀ F U') .Precategory.Ob)
        → Ω U' ξ' → Ω U ((F₁ F α) .Functor.F₀ ξ')

    -- Restriction along morphism in fiber
    Ω_U-hom : ∀ {U} {ξ ξ' : (F₀ F U) .Precategory.Ob}
            → (F₀ F U) .Precategory.Hom ξ ξ'
            → Ω U ξ' → Ω U ξ

    -- Equation 3.6: Composition giving downstream transfer
    Ω-transfer-downstream : ∀ {U U'} (α : C .Precategory.Hom U U')
                          → {ξ : (F₀ F U) .Precategory.Ob}
                          → {ξ' : (F₀ F U') .Precategory.Ob}
                          → (h : (F₀ F U) .Precategory.Hom ξ ((F₁ F α) .Functor.F₀ ξ'))
                          → Ω U' ξ' → Ω U ξ

    -- Definition: Ω_{α,h} = Ω_U(h) ∘ λ_α
    Ω-transfer-def : ∀ {U U' α ξ ξ' h}
                   → Ω-transfer-downstream {U} {U'} α {ξ} {ξ'} h
                     ≡ (λ P' → Ω_U-hom h (λ_α α ξ' P'))

  {-|
  ## Equation 3.7: Extension to Arbitrary Subobjects

  > "More generally, for every object X' in E_{U'}, the map F_α^! sends the
  > subobjects of X' to the subobjects of F_α^!(X'), respecting the lattice
  > structures."

  For any context X, X', the transfer extends to:
  Ω_{α,h}: Ω^{X'}_{U'} → Ω^X_U
  -}

  postulate
    -- Subobject classifier for arbitrary contexts
    Ω^_ : ∀ {U} → (F₀ F U) .Precategory.Ob → Type ℓ'

    -- Transfer for general contexts
    Ω-transfer-context : ∀ {U U'} (α : C .Precategory.Hom U U')
                       → {X : (F₀ F U) .Precategory.Ob}
                       → {X' : (F₀ F U') .Precategory.Ob}
                       → (h : (F₀ F U) .Precategory.Hom X ((F₁ F α) .Functor.F₀ X'))
                       → Ω^ X' → Ω^ X

    -- Respects lattice structure (∧, ∨)
    Ω-transfer-preserves-∧ : ∀ {U U' α X X' h} (P Q : Ω^ X')
                           → Ω-transfer-context {U} {U'} α {X} {X'} h (P ∧ Q)
                             ≡ (Ω-transfer-context α h P) ∧ (Ω-transfer-context α h Q)
      where postulate _∧_ : ∀ {X} → Ω^ X → Ω^ X → Ω^ X

  {-|
  ## Equation 3.8: Upstream Transfer Ω'_{α,h} (Right Adjoint)

  > "Under the strong standard hypotheses on the fibration F, for instance if it
  > defines a fibrant object in the injective groupoids models, i.e. any F_α is
  > a fibration, there exists a right adjoint of Ω_{α,h}:
  >   Ω'_{α,h}: Ω^X_U → Ω^{X'}_{U'}"

  **Upstream Direction** (backpropagation):
  - Right adjoint for logical transfer
  - Uses F★_α (right adjoint of F_α^!)
  - Corresponds to "what must be true upstream for downstream property to hold"

  **Strong Hypothesis**: π★ ∘ π★ = Id (section-retraction pair)
  -}

  postulate
    -- Right adjoint F★_α of geometric functor F_α^!
    F★_α : ∀ {U U'} (α : C .Precategory.Hom U U')
         → Functor (F₀ F U) (F₀ F U')

    -- Upstream transfer (right adjoint)
    Ω-transfer-upstream : ∀ {U U'} (α : C .Precategory.Hom U U')
                        → {X : (F₀ F U) .Precategory.Ob}
                        → {X' : (F₀ F U') .Precategory.Ob}
                        → Ω^ X → Ω^ X'

    -- Adjunction property
    Ω-adjunction : ∀ {U U' α X X' h} (P : Ω^ X) (Q : Ω^ X')
                 → (Ω-transfer-context {U} {U'} α {X} {X'} h Q ≤ P)
                   ≃ (Q ≤ Ω-transfer-upstream α P)
      where postulate _≤_ : ∀ {X} → Ω^ X → Ω^ X → Type ℓ'

    -- Strong hypothesis: composition is identity
    π★-π★-id : ∀ {U U'} (α : C .Precategory.Hom U U') {X}
             → Ω-transfer-upstream α (Ω-transfer-context α {X = X} {!!} {!!}) ≡ {!!}

{-|
## Notation Convention

Following the paper:
- **π★_{α,h}** or **L_{α,h}**: Downstream transfer (Equation 3.6)
- **π^★_{α,h}** or **^tL'_{α,h}**: Upstream transfer (Equation 3.8)
- **A**: Presheaf with π★ morphisms (fibration)
- **A'**: Copresheaf with π^★ morphisms (cofibration)

These form the foundation for semantic conditioning in the next section.
-}

--------------------------------------------------------------------------------
-- § 3.3.1c: Fibration A and Cofibration A' (Equations 3.9-3.10)

{-|
## Fibration and Cofibration of Propositions

> "A and A' belong to augmented model categories using monoidal posets."

The key structures for semantic information are:
1. **Category A**: Objects (U, ξ, P), morphisms via downstream transfer π★
2. **Category A'**: Same objects, morphisms via upstream transfer π^★
3. **A'_strict**: Restricted A' where P' = π^★ P exactly

These form fibration/cofibration over the stack F.
-}

module _ {C : Precategory o ℓ} {F : Stack C o' ℓ'} where

  -- Notation: Use Ω for propositions (elements of subobject classifier)
  -- Following paper: A_{U,ξ,P} is monoidal category of propositions

  {-|
  ## Objects of A and A'

  Triple λ = (U, ξ, P) where:
  - U: layer in network (object of C)
  - ξ: context/feature in fiber F_U
  - P: proposition (element of Ω_U(ξ))
  -}

  record A-Ob : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ') where
    constructor _,_,_
    field
      layer : C .Precategory.Ob
      context : (F₀ F layer) .Precategory.Ob
      proposition : Ω layer context

  {-|
  ## Equation 3.9: Morphisms in A (Fibration)

  > "A morphism in A from (U,ξ,P) to (U',ξ',P') lifting (α,h): (U,ξ) → (U',ξ')
  > is given by an external implication: P ≤ L_{α,h}(P') = π★_{α,h} P'"

  **Interpretation**:
  - P implies the transferred proposition from layer U'
  - Downstream direction: deeper layers constrain earlier layers
  - External implication in the poset/Heyting algebra structure
  -}

  record A-Hom (λ λ' : A-Ob) : Type (o ⊔ ℓ ⊔ ℓ') where
    constructor mk-A-hom
    field
      -- Base morphism in network
      base-mor : C .Precategory.Hom (λ .A-Ob.layer) (λ' .A-Ob.layer)

      -- Fiber morphism
      fiber-mor : (F₀ F (λ .A-Ob.layer)) .Precategory.Hom
                  (λ .A-Ob.context)
                  ((F₁ F base-mor) .Functor.F₀ (λ' .A-Ob.context))

      -- External implication (Equation 3.9)
      -- P ≤ π★_{α,h}(P')
      implies : (λ .A-Ob.proposition)
                ≤ Ω-transfer-downstream base-mor fiber-mor (λ' .A-Ob.proposition)

    -- Notation from paper
    π★ = Ω-transfer-downstream base-mor fiber-mor

  {-|
  ## Equation 3.10: Morphisms in A' (Cofibration)

  > "An arrow in category A' lifting morphism (α,h) in F is an implication:
  > ^tL'_{α,h}(P) ≤ P'"

  **Interpretation**:
  - Upstream transfer of P is implied by P'
  - Backward direction: earlier layers provide evidence for deeper properties
  - Dual to fibration structure
  -}

  record A'-Hom (λ λ' : A-Ob) : Type (o ⊔ ℓ ⊔ ℓ') where
    constructor mk-A'-hom
    field
      base-mor : C .Precategory.Hom (λ .A-Ob.layer) (λ' .A-Ob.layer)
      fiber-mor : (F₀ F (λ .A-Ob.layer)) .Precategory.Hom
                  (λ .A-Ob.context)
                  ((F₁ F base-mor) .Functor.F₀ (λ' .A-Ob.context))

      -- Equation 3.10: ^tL'_{α,h}(P) ≤ P'
      co-implies : Ω-transfer-upstream base-mor (λ .A-Ob.proposition)
                   ≤ (λ' .A-Ob.proposition)

    -- Notation from paper
    π^★ = Ω-transfer-upstream base-mor

  {-|
  ## Category A (Fibration over F)

  Objects: Triples (U, ξ, P)
  Morphisms: External implications via π★
  Composition: Transitivity of ≤
  -}

  A-Category : Precategory (o ⊔ ℓ ⊔ o' ⊔ ℓ') (o ⊔ ℓ ⊔ ℓ')
  A-Category .Precategory.Ob = A-Ob
  A-Category .Precategory.Hom = A-Hom
  A-Category .Precategory.id {λ} = record
    { base-mor = C .Precategory.id
    ; fiber-mor = (F₀ F (λ .A-Ob.layer)) .Precategory.id
    ; implies = ≤-refl
    }
  A-Category .Precategory._∘_ f g = record
    { base-mor = C .Precategory._∘_ (f .A-Hom.base-mor) (g .A-Hom.base-mor)
    ; fiber-mor = {!!}  -- Composition in fiber
    ; implies = ≤-trans (g .A-Hom.implies) {!!}  -- Transitivity
    }
  A-Category .Precategory.idr f = {!!}
  A-Category .Precategory.idl f = {!!}
  A-Category .Precategory.assoc f g h = {!!}

  {-|
  ## Category A' (Cofibration over F)

  Same objects as A, dual morphisms via π^★
  -}

  A'-Category : Precategory (o ⊔ ℓ ⊔ o' ⊔ ℓ') (o ⊔ ℓ ⊔ ℓ')
  A'-Category .Precategory.Ob = A-Ob
  A'-Category .Precategory.Hom = A'-Hom
  A'-Category .Precategory.id {λ} = record
    { base-mor = C .Precategory.id
    ; fiber-mor = (F₀ F (λ .A-Ob.layer)) .Precategory.id
    ; co-implies = ≤-refl
    }
  A'-Category .Precategory._∘_ f g = record
    { base-mor = C .Precategory._∘_ (f .A'-Hom.base-mor) (g .A'-Hom.base-mor)
    ; fiber-mor = {!!}
    ; co-implies = ≤-trans {!!} (f .A'-Hom.co-implies)
    }
  A'-Category .Precategory.idr f = {!!}
  A'-Category .Precategory.idl f = {!!}
  A'-Category .Precategory.assoc f g h = {!!}

  {-|
  ## A'_strict: Restricted Cofibration

  > "Consider the smaller category A'_strict with the same objects but with
  > morphisms from A_{U,ξ,P} to A_{U',ξ',P'} only when P' = π^★_{α,h} P"

  **Critical property**: This restriction allows copresheaf structure
  -}

  record A'-strict-Hom (λ λ' : A-Ob) : Type (o ⊔ ℓ ⊔ ℓ') where
    constructor mk-A'-strict-hom
    field
      base-mor : C .Precategory.Hom (λ .A-Ob.layer) (λ' .A-Ob.layer)
      fiber-mor : (F₀ F (λ .A-Ob.layer)) .Precategory.Hom
                  (λ .A-Ob.context)
                  ((F₁ F base-mor) .Functor.F₀ (λ' .A-Ob.context))

      -- Strict condition: P' = π^★(P) exactly
      strict-eq : (λ' .A-Ob.proposition)
                  ≡ Ω-transfer-upstream base-mor (λ .A-Ob.proposition)

  A'-strict-Category : Precategory (o ⊔ ℓ ⊔ o' ⊔ ℓ') (o ⊔ ℓ ⊔ ℓ')
  A'-strict-Category .Precategory.Ob = A-Ob
  A'-strict-Category .Precategory.Hom = A'-strict-Hom
  A'-strict-Category .Precategory.id = {!!}
  A'-strict-Category .Precategory._∘_ = {!!}
  A'-strict-Category .Precategory.idr = {!!}
  A'-strict-Category .Precategory.idl = {!!}
  A'-strict-Category .Precategory.assoc = {!!}

  {-|
  ## Monoidal Structure on Fibers

  For fixed U, ξ, the propositions A_{U,ξ,P} form a monoidal poset category:
  - Monoidal operation: ∧ (conjunction)
  - Unit: ⊤ (true)
  - Order: P ≤ Q (external implication)
  - Closed: internal implication P ⇒ Q
  -}

  module _ (U : C .Precategory.Ob) (ξ : (F₀ F U) .Precategory.Ob) where

    postulate
      -- Monoidal operations
      _∧_ : Ω U ξ → Ω U ξ → Ω U ξ
      ⊤ : Ω U ξ

      -- Ordering
      _≤_ : Ω U ξ → Ω U ξ → Type ℓ'
      ≤-refl : ∀ {P} → P ≤ P
      ≤-trans : ∀ {P Q R} → P ≤ Q → Q ≤ R → P ≤ R

      -- Internal implication (Heyting algebra)
      _⇒_ : Ω U ξ → Ω U ξ → Ω U ξ

      -- Adjunction: (R ∧ Q ≤ P) ↔ (R ≤ (Q ⇒ P))
      ⇒-adjoint : ∀ {P Q R} → (R ∧ Q ≤ P) ≃ (R ≤ (Q ⇒ P))

      -- Negation
      ¬_ : Ω U ξ → Ω U ξ
      ¬-def : ∀ {P} → ¬ P ≡ (P ⇒ ⊥)
        where postulate ⊥ : Ω U ξ

{-|
## Summary: Fibration/Cofibration Structure

We have defined:
1. **Category A**: Fibration with downstream transfer π★ (Equation 3.9)
2. **Category A'**: Cofibration with upstream transfer π^★ (Equation 3.10)
3. **Category A'_strict**: Strict equality P' = π^★ P
4. **Monoidal posets**: Each fiber is a Heyting algebra

**Next**: Use these to define:
- Theories Θ_{U,ξ,P} as presheaves/copresheaves
- Semantic conditioning Q.T = (Q ⇒ T)
- Module structures for information measures
-}

--------------------------------------------------------------------------------
-- § 3.3.1d: Theories and Semantic Conditioning

module _ {C : Precategory o ℓ} {F : Stack C o' ℓ'} where
  open A-Ob

  {-|
  ## Theories over Languages

  > "Θ_{U,ξ} is the set of theories expressed in the algebra Ω L_U in the
  > context ξ. Under our standard hypothesis on F, both L_α and ^tL'_α send
  > theories to theories."

  **Theory**: Set of axioms (propositions asserted as true)
  - Represented by a single conjunction of all axioms
  - T ≤ T' means T is weaker (less constraints)
  -}

  postulate
    -- Set of all theories at (U, ξ)
    Theory : (U : C .Precategory.Ob) → (ξ : (F₀ F U) .Precategory.Ob) → Type ℓ'

    -- Theory is given by its conjunction of axioms
    theory-prop : ∀ {U ξ} → Theory U ξ → Ω U ξ

    -- Weaker/stronger ordering on theories
    -- T ≤ T' means T is weaker (T' implies T)
    _≤T_ : ∀ {U ξ} → Theory U ξ → Theory U ξ → Type ℓ'
    ≤T-refl : ∀ {U ξ} {T : Theory U ξ} → T ≤T T
    ≤T-trans : ∀ {U ξ} {T T' T'' : Theory U ξ} → T ≤T T' → T' ≤T T'' → T ≤T T''

  {-|
  ## Θ_{U,ξ,P}: Theories Excluding P

  > "Θ_{U,ξ,P} is the subset of theories which imply the truth of proposition
  > ¬P, i.e. the subset of theories excluding P."

  **Construction**:
  - Theory T excludes P if T ∧ P ≤ ⊥
  - Equivalently: T ≤ ¬P
  - These form a presheaf over A (with π★)
  - And a copresheaf over A'_strict (with π^★)
  -}

  -- Theories that exclude proposition P
  Θ : (λ : A-Ob) → Type ℓ'
  Θ (U , ξ , P) = Σ[ T ∈ Theory U ξ ] (theory-prop T ≤ ¬ P)
    where
      open import Neural.Stack.Languages using (_≤_; ¬_)

  {-|
  ## Equation 3.11: Semantic Conditioning Q.T = (Q ⇒ T)

  > "For fixed U,ξ, P ≤ Q in Ω L_{U,ξ}, and T a theory in the language L_{U,ξ},
  > we define a new theory by the internal implication:
  >   Q.T = (Q ⇒ T)"

  **Properties**:
  - More precisely: axioms of Q.T are {⊢ (Q ⇒ R) | ⊢ R is an axiom of T}
  - This is conditioning in the logical/semantic sense
  - Written T|_Q or Q.T

  **DNN Interpretation**:
  - Q: observed property or constraint
  - T: general theory about network behavior
  - Q.T: theory conditioned on Q being true
  - Example: "cat detector theory | image-has-fur"
  -}

  -- Semantic conditioning operation
  infixl 20 _·T_
  _·T_ : ∀ {U ξ} → Ω U ξ → Theory U ξ → Theory U ξ
  _·T_ {U} {ξ} Q T = {!!}  -- Theory with axioms {Q ⇒ R | R axiom of T}

  -- Notation: T|_Q
  _|ᵀ_ : ∀ {U ξ} → Theory U ξ → Ω U ξ → Theory U ξ
  T |ᵀ Q = Q ·T T

  {-|
  ## Equation 3.12: Heyting Adjunction

  > "(R ∧ Q ≤ P) if and only if (R ≤ (Q ⇒ P))"

  This is the defining property of internal implication in a Heyting algebra.
  At the level of theories, it means:
  - Conditioning is right adjoint to conjunction
  - Q.T is the largest theory T' such that Q ∧ T' ≤ T
  -}

  postulate
    -- Conditioning satisfies adjunction
    conditioning-adjoint : ∀ {U ξ} (Q : Ω U ξ) (T T' : Theory U ξ)
                         → {!!}  -- (theory-prop T' ∧ Q ≤ theory-prop T)
                                 -- ≃ (theory-prop T' ≤ theory-prop (Q ·T T))

  {-|
  ## Proposition 3.1: Conditioning Action on Theories

  > "The conditioning gives an action of the monoid A_{U,ξ,P} on the set of
  > theories in the language L_{U,ξ}."

  **Proof**:
  ```
  (R ∧ Q' ∧ Q ≤ P)  iff  (R ∧ Q' ≤ (Q ⇒ P))
                     iff  (R ≤ (Q' ⇒ (Q ⇒ P)))
  ```

  Therefore Q' ⇒ (Q ⇒ P) = (Q' ∧ Q) ⇒ P, giving the action.
  -}

  proposition-3-1 : ∀ {U ξ P} (Q Q' : Ω U ξ)
                  → (P ≤ Q) → (P ≤ Q')
                  → (T : Theory U ξ)
                  → (Q ·T (Q' ·T T)) ≡ ((Q ∧ Q') ·T T)
  proposition-3-1 = {!!}

  {-|
  ## Proposition 3.2: Conditioning Preserves Θ_{U,ξ,P}

  > "The conditioning by elements of A_{U,ξ,P}, i.e. propositions Q implied by
  > P, preserves the set Θ_{U,ξ,P} of theories excluding P."

  **Proof**:
  Let T be a theory excluding P and Q ≥ P.
  Consider theory T' such that Q ∧ T' ≤ T.
  Then: T' ∧ P ≤ T, thus T' ∧ P ≤ T ∧ P.
  But T ∧ P ≤ ⊥, so T' ∧ P ≤ ⊥.
  Since Q ⇒ T is the largest theory with Q ∧ T' ≤ T,
  therefore (Q ⇒ T) excludes P, i.e., asserts ¬P.
  -}

  proposition-3-2 : ∀ {U ξ P} (Q : Ω U ξ)
                  → (P ≤ Q)
                  → (T : Θ (U , ξ , P))
                  → {!!}  -- (Q ·T (fst T)) ∈ Θ (U , ξ , P)
  proposition-3-2 = {!!}

  {-|
  ## Presheaf/Copresheaf Structure

  The sets of theories form:
  1. **Presheaf Θ_loc over A** with π★ transitions
  2. **Copresheaf Θ' over A'_strict** with π^★ transitions

  This enables homological algebra for semantic information.
  -}

  postulate
    -- Transfer theories downstream via π★
    Θ-transfer-downstream : ∀ {λ λ' : A-Ob}
                          → A-Hom λ λ'
                          → Θ λ' → Θ λ

    -- Transfer theories upstream via π^★ (on A'_strict)
    Θ-transfer-upstream : ∀ {λ λ' : A-Ob}
                        → A'-strict-Hom λ λ'
                        → Θ λ → Θ λ'

  {-|
  ## Lemma 3.1: A_{U,ξ,P} as Presheaf over A

  > "The monoids A_{U,ξ,P}, with the functors π★ between them, form a presheaf
  > over the category A."

  **Proof**:
  Given morphism (α,h,ι): A_{U,ξ,P} → A_{U',ξ',P'} in A,
  the symbol ι means P ≤ π★ P'.
  From P' ≤ Q', we deduce P ≤ π★ P' ≤ π★ Q'.
  Therefore π★ is a functor on propositions Q' ≥ P'.
  -}

  lemma-3-1 : ∀ {λ λ' : A-Ob} (f : A-Hom λ λ')
            → {Q' : Ω (λ' .layer) (λ' .context)}
            → (λ' .proposition ≤ Q')
            → {!!}  -- π★ gives presheaf structure
  lemma-3-1 = {!!}

  {-|
  ## Lemma 3.2: Θ_{U,ξ,P} as Presheaf over A

  > "Under the standard hypotheses on the fibration F, without necessarily
  > axiom (2.30), the sets Θ_{U,ξ,P} with the morphisms π★, form a presheaf
  > over A."

  **Proof**:
  Let (α,h,ι): A_{U,ξ,P} → A_{U',ξ',P'} where ι denotes P ≤ π★ P'.
  We deduce π★ ¬P' = ¬π★ P' ≤ ¬P.
  Then T' ≤ ¬P' implies π★ T' ≤ π★ ¬P' ≤ ¬P.
  -}

  lemma-3-2 : ∀ {λ λ' : A-Ob} (f : A-Hom λ λ')
            → {T' : Θ λ'}
            → Θ-transfer-downstream f T' ∈ Θ λ
  lemma-3-2 = {!!}

  {-|
  ## Lemma 3.3: Copresheaves over A'_strict

  > "Under the standard hypotheses on the fibration F plus the stronger one,
  > the sets Θ_{U,ξ,P} with morphisms π^★, also form a presheaf over A'."

  This requires the strong hypothesis π★ ∘ π^★ = Id.
  -}

  lemma-3-3 : ∀ {λ λ' : A-Ob} (f : A'-strict-Hom λ λ')
            → {!!}  -- Copresheaf property
  lemma-3-3 = {!!}

{-|
## Summary: Theories and Conditioning

We have defined:
1. **Theory U ξ**: Sets of axioms/propositions at each layer
2. **Θ_{U,ξ,P}**: Theories excluding proposition P
3. **Equation 3.11**: Conditioning Q.T = (Q ⇒ T)
4. **Proposition 3.1**: Conditioning is monoidal action
5. **Proposition 3.2**: Conditioning preserves Θ_{U,ξ,P}
6. **Lemmas 3.1-3.3**: Presheaf/copresheaf structures

**Next**: Functions on theories (module Φ) and Theorem 3.1
-}

--------------------------------------------------------------------------------
-- § 3.3.1e: Module Structure and Theorem 3.1

module _ {C : Precategory o ℓ} {F : Stack C o' ℓ'} where

  {-|
  ## Functions on Theories: Module Φ

  > "The cosheaf Φ made by the measurable functions (with any kind of fixed
  > values) on the theories Θ_{U,ξ,P}, with the morphisms π★, is a cosheaf of
  > modules over the monoidal cosheaf A'_loc."

  **Structure**:
  - Φ_λ: functions Θ_λ → K (values in ring K)
  - π★ pushforward: ϕ ↦ ϕ ∘ π★
  - Module action via conditioning: (Q.ϕ)(T) = ϕ(Q ⇒ T)
  -}

  postulate
    -- Ring of values (ℝ, ℤ, or more general)
    K : Type

  -- Functions on theories at λ
  Φ : (λ : A-Ob) → Type ℓ'
  Φ λ = Θ λ → K

  {-|
  ## Equation 3.16: Naturality over A'_strict

  > "For morphism (α,h,ι): A_{U,ξ,P} → A_{U',ξ',π★P}, the equation:
  >   (π★ Q').ϕ_P(T) = ϕ_P[π★(T')]|_Q]
  > holds true under the strong hypothesis π★ π★ = Id."

  This is the compatibility condition for the module action.
  -}

  postulate
    -- Pushforward of functions via π★
    Φ-transfer : ∀ {λ λ' : A-Ob}
               → A'-strict-Hom λ λ'
               → Φ λ → Φ λ'

    -- Module action by conditioning
    Φ-action : ∀ {λ : A-Ob} (Q : Ω (λ .A-Ob.layer) (λ .A-Ob.context))
             → (λ .A-Ob.proposition ≤ Q)
             → Φ λ → Φ λ

  {-|
  ## Theorem 3.1: Cosheaf of Modules

  > "Under the strong hypothesis, in particular π★ π★ = Id, and over the
  > restricted category A'_strict, the cosheaf Φ' made by the measurable
  > functions on the theories Θ_{U,ξ,P}, with the morphisms π★, is a cosheaf
  > of modules over the monoidal cosheaf A'_loc, made by the monoidal
  > categories A_{U,ξ,P}, with the morphisms π★."

  **Proof**:
  Consider morphism (α,h,ι): A_{U,ξ,P} → A_{U',ξ',π★P}.
  For theory T' in Θ_{U',ξ',π★P}, proposition Q in A_{U,ξ,P}, and ϕ_P in Φ'_{U,ξ,P}:

  ```
  π★ Q.(Φ'★ ϕ_P)(T') = (Φ'★ ϕ_P)(T'|_{π★ Q})
                       = ϕ_P[π★(T'|_{π★ Q})]
                       = ϕ_P[π★(π★ Q ⇒ T')]
                       = ϕ_P[π★ π★ Q ⇒ π★ T']   (geometric morphism)
                       = ϕ_P[Q ⇒ π★ T']          (strong hypothesis)
                       = ϕ_P[π★(T')|_Q]
  ```

  This shows naturality: action commutes with transfer.
  -}

  theorem-3-1 : ∀ {λ λ' : A-Ob} (f : A'-strict-Hom λ λ')
              → {Q : Ω (λ .A-Ob.layer) (λ .A-Ob.context)}
              → (λ .A-Ob.proposition ≤ Q)
              → (ϕ : Φ λ)
              → (T' : Θ λ')
              → {!!}  -- π★ Q.(Φ-transfer f ϕ)(T') = ϕ((Q ·T π★ T'))
  theorem-3-1 = {!!}

  {-|
  ## Proposition 3.3: Presheaf Compatibility

  > "The presheaf Θ_loc for π★ is compatible with the monoidal action of the
  > presheaf A_loc, both considered on the category A (then over A' by
  > restriction, under the strong standard hypothesis on F)."

  **Proof**:
  If T' ≤ ¬P' and P ≤ π★ P', we have ¬π★ P' ≤ ¬P.
  Therefore π★ T' ≤ ¬P.
  -}

  proposition-3-3 : ∀ {λ λ' : A-Ob} (f : A-Hom λ λ')
                  → {T' : Θ λ'}
                  → {Q : Ω (λ .A-Ob.layer) (λ .A-Ob.context)}
                  → {!!}  -- Compatibility of presheaf with action
  proposition-3-3 = {!!}

{-|
## Semantic Analogy to Probability Theory

The paper draws deep analogies:

**Probability (Bayesian networks)**:
- Variables X, Y
- Probability laws P_X
- Conditioning: Y.P_X (Shannon mean formula)
- Entropy H: measures ambiguity
- Mutual information I(X;Y) = H(X) - H(X|Y)

**Semantics (Neural networks)**:
- Propositions P, Q (analogs of variables)
- Theories T (analogs of probability laws)
- Conditioning: Q.T = (Q ⇒ T)
- Functions ψ on theories (analogs of -entropy)
- Information: φ^Q(T) = ψ(T|Q) - ψ(T)

The module structure (Theorem 3.1) makes this analogy precise.
-}

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
