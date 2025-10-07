{-# OPTIONS --no-import-sorts #-}
{-|
# Categorical Extraction: Topos Theory → Compositional IR

**Purpose**: Extract categorical structures (functors, sheaves, fibrations) from
our topos-theoretic neural network formalizations and serialize them to a format
that preserves compositional structure.

## Key Insight

A sheaf F: C^op → Set IS already a computational structure:

- F₀: ForkVertex → Type           -- Tensors at vertices
- F₁: (v → w) → (F₀ w → F₀ v)    -- Operations (contravariant)
- F-∘: F₁(f ∘ g) ≡ F₁ g ∘ F₁ f   -- Composition is built-in!

The functoriality axiom F-∘ IS function composition. We don't need to serialize
a flat graph - we serialize the FUNCTOR directly and interpret it categorically.

## Compilation Pipeline

```
Topos Theory (Agda)
    ↓ extract-functor
Categorical IR (preserves F₀, F₁, F-∘)
    ↓ serialize-categorical
Protobuf / JSON
    ↓ Python categorical interpreter
JAX (composition automatic from F-∘)
```

## Contrast with Flat IR

**OLD (Flat IR)**: Graph with vertices/edges, loses compositional structure
**NEW (Categorical IR)**: Functor F: C → Set with F₀, F₁, functoriality preserved

-}

module Neural.Compile.Extract where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Path

open import Cat.Base
open import Cat.Functor.Base
open import Cat.Instances.Functor
open import Cat.Instances.Sets
open import Cat.Instances.Sheaves using (Sh[_,_])
open import Cat.Bi.Base using (Cat)

-- Temporarily commented out due to type error in Architecture.agda line 314
-- We'll re-enable once that's fixed
-- open import Neural.Topos.Architecture
-- open import Neural.Stack.Fibration

open import Data.Nat.Base using (Nat; zero; suc)
open import Data.List.Base using (List; []; _∷_; _++_)
open import Data.String.Base using (String)
open import Data.Bool.Base using (Bool; true; false)

private variable
  o ℓ o' ℓ' κ : Level

{-|
## Categorical IR Types

These types represent the categorical structures we extract from topos theory.
Unlike the flat IR in Neural.Compile.IR, these preserve compositional structure.
-}

{-|
### Object Mapping: F₀(c) = TensorSpec

For each object c in the base category, F₀(c) gives the tensor type at that vertex.
This is the "data in space" view from topos theory.
-}

-- Tensor specifications (from fibration fibers)
data TensorSpec : Type where
  tensor-scalar : TensorSpec
  tensor-vec    : Nat → TensorSpec
  tensor-mat    : Nat → Nat → TensorSpec
  tensor-prod   : TensorSpec → TensorSpec → TensorSpec  -- Product types
  tensor-sum    : TensorSpec → TensorSpec → TensorSpec  -- Sum types

-- Object mapping: c ↦ F₀(c)
record ObjectMapping : Type where
  constructor obj-map
  field
    obj-id : String              -- Object identifier
    tensor-type : TensorSpec     -- F₀(obj-id)

{-|
### Morphism Mapping: F₁(f) = ComputeOp

For each morphism f: c → c' in the base category, F₁(f) gives the operation.
The functoriality F₁(f ∘ g) = F₁(g) ∘ F₁(f) means composition works automatically!
-}

-- Operations (from sheaf structure)
data ComputeOp : Type where
  op-linear      : (in-dim out-dim : Nat) → ComputeOp
  op-conv        : (in-ch out-ch kernel : Nat) → ComputeOp
  op-activation  : String → ComputeOp  -- "relu", "sigmoid", "tanh", etc.
  op-fork        : (arity : Nat) → ComputeOp  -- From fork-star vertices!
  op-residual    : ComputeOp                   -- From conservation laws
  op-compose     : ComputeOp → ComputeOp → ComputeOp  -- Functoriality!
  op-identity    : ComputeOp                   -- Identity morphisms

-- Morphism mapping: f ↦ F₁(f)
record MorphismMapping : Type where
  constructor morph-map
  field
    morph-id : String           -- Morphism identifier
    source   : String           -- Source object ID
    target   : String           -- Target object ID
    operation : ComputeOp       -- F₁(morph-id)

{-|
### Functoriality Witnesses

These are the PROOFS that F preserves identity and composition.
In Python, these become structural guarantees - we don't need to check them
at runtime because they're proven in Agda!
-}

record FunctorialityWitness : Type where
  constructor make-functoriality-witness
  field
    -- F₁(id_c) = id_{F₀(c)}
    preserves-identity : List (String × String)  -- (obj-id, witness)

    -- F₁(f ∘ g) = F₁(g) ∘ F₁(f)
    preserves-composition : List (String × String × String)  -- (f, g, witness)

{-|
### Categorical Functor IR

This is the main IR type that preserves categorical structure.
Unlike flat IR, this directly represents F: C → Set as a functor.
-}

record CategoricalFunctor : Type where
  constructor cat-functor
  field
    name : String

    -- The functor structure
    objects : List ObjectMapping        -- F₀
    morphisms : List MorphismMapping    -- F₁
    functoriality : FunctorialityWitness -- F-id, F-∘

    -- Metadata
    inputs : List String   -- Input object IDs
    outputs : List String  -- Output object IDs

{-|
### Sheaf Conditions (Merge Constraints)

For fork vertices A★, the sheaf condition says:
  F(A★) ≅ ∏_{a' → A★} F(a')

This becomes a merge constraint: the tensor at A★ must be the product
of tensors from incoming vertices.
-}

record MergeConstraint : Type where
  constructor merge-at
  field
    fork-vertex : String              -- The A★ vertex
    incoming : List String            -- The a' vertices with a' → A★
    merge-type : TensorSpec           -- F(A★) should equal this product

{-|
### Sheaf IR

A sheaf is a functor + sheaf conditions (merge constraints).
-}

record SheafIR : Type where
  constructor sheaf-ir
  field
    functor : CategoricalFunctor
    merge-constraints : List MergeConstraint

{-|
### Fibration IR (Dependent Shapes)

A fibration π: E → B gives dependent types:
- For each b ∈ B, the fiber π⁻¹(b) is a category
- Objects in the fiber are "shapes over b"
- This is perfect for shape inference!
-}

record FiberShape : Type where
  constructor fiber-shape
  field
    base-obj : String       -- b ∈ B
    fiber-obj : String      -- ξ ∈ π⁻¹(b)
    shape : TensorSpec      -- The actual shape

record FibrationIR : Type where
  constructor fibration-ir
  field
    name : String
    base-functor : CategoricalFunctor       -- B
    total-functor : CategoricalFunctor      -- E
    projection : List (String × String)  -- π: E → B (total-obj ↦ base-obj)
    fibers : List FiberShape        -- Shapes in each fiber

{-|
## Extraction Functions

These extract categorical structures from our topos-theoretic modules.
-}

{-|
### Extract from Fork-Category

Given an OrientedGraph with fork construction, extract the categorical functor.

The key is that Fork-Category IS a category, and a sheaf on it IS a functor
F: Fork-Category^op → Set.

NOTE: Currently postulated due to type error in Neural.Topos.Architecture.
Once that's fixed, we'll implement the full extraction.
-}

-- Postulated types from Architecture (will be actual imports once fixed)
postulate
  OrientedGraph : (o ℓ : Level) → Type (lsuc o ⊔ lsuc ℓ)
  ForkVertex : {o ℓ : Level} → OrientedGraph o ℓ → Type (o ⊔ ℓ)
  Fork-Category : {o ℓ : Level} → OrientedGraph o ℓ → Precategory (o ⊔ ℓ) (o ⊔ ℓ)

-- Generic extraction from a functor (works for any category)
module ExtractFromFunctor {C : Precategory o ℓ} where
  -- Extract sheaf IR from a functor on any category
  postulate
    extract-sheaf : Functor (C ^op) (Sets o) → SheafIR

  -- Extract object mappings by traversing F₀
  postulate
    extract-objects : Functor (C ^op) (Sets o) → List ObjectMapping

  -- Extract morphism mappings by traversing F₁
  postulate
    extract-morphisms : Functor (C ^op) (Sets o) → List MorphismMapping

  -- TODO: Implement extraction by:
  -- 1. Enumerate objects in C
  -- 2. For each object c, extract F₀(c) as TensorSpec
  -- 3. Enumerate morphisms in C
  -- 4. For each morphism f, extract F₁(f) as ComputeOp
  -- 5. Use functoriality proofs F-id and F-∘ to build FunctorialityWitness

{-|
### Extract from Fibration

Given a fibration π: E → B, extract dependent shape information.

The fiber π⁻¹(b) contains all possible shapes at base object b.
This is exactly what we need for shape inference!
-}

module ExtractFromFibration {C : Precategory o ℓ} where
  -- For now, we postulate fibration extraction
  -- The actual implementation would use the Grothendieck construction
  -- from Neural.Stack.Fibration once it's importable
  postulate
    extract-fibration : FibrationIR

  -- TODO: Implement by traversing the Grothendieck construction
  -- 1. Base category B = C
  -- 2. Total category E = ∫F (Grothendieck construction)
  -- 3. Projection π : E → B
  -- 4. For each b ∈ B, extract shapes from fiber F(b)
  --
  -- The type would be:
  --   extract-fibration : (F : Functor (C ^op) (Cat o' ℓ')) → FibrationIR
  -- but we can't reference Cat properly yet

{-|
### Extract from Natural Transformations (Backpropagation)

From Section 1.4, backpropagation is a flow of natural transformations W → W.

A natural transformation η: F ⇒ G consists of components η_c: F(c) → G(c)
satisfying naturality: G(f) ∘ η_c = η_c' ∘ F(f).

This IS the chain rule for gradients!
-}

record GradientFlow : Type where
  constructor grad-flow
  field
    forward-functor : CategoricalFunctor   -- W
    backward-functor : CategoricalFunctor  -- W (same, but contravariant)
    components : List (String × ComputeOp) -- η_c for each object c

module ExtractBackprop {C : Precategory o ℓ} where
  -- Extract gradient flow from a natural transformation
  postulate
    extract-gradient : {F G : Functor C (Sets ℓ)} →
                       (η : F => G) →
                       GradientFlow

{-|
## Serialization to Protobuf-like Format

These functions serialize the categorical IR to a format suitable for
Python interpretation. We use a simple S-expression-like format here,
but in practice this would generate Protobuf.
-}

-- Serialize tensor spec
serialize-tensor : TensorSpec → String
serialize-tensor tensor-scalar = "scalar"
serialize-tensor (tensor-vec n) = "vec(" -- TODO: append nat
serialize-tensor (tensor-mat m n) = "mat(" -- TODO
serialize-tensor (tensor-prod s t) =
  "prod(" -- TODO: serialize s, t
serialize-tensor (tensor-sum s t) =
  "sum(" -- TODO

-- Serialize compute op
serialize-op : ComputeOp → String
serialize-op (op-linear m n) = "linear(" -- TODO
serialize-op (op-conv ic oc k) = "conv(" -- TODO
serialize-op (op-activation name) = "activation(" -- TODO
serialize-op (op-fork arity) = "fork(" -- TODO
serialize-op op-residual = "residual"
serialize-op (op-compose f g) =
  "compose(" -- TODO: serialize f, g
serialize-op op-identity = "id"

-- Serialize object mapping
serialize-obj-map : ObjectMapping → String
serialize-obj-map (obj-map id spec) =
  "(object " -- TODO: append id and serialize spec

-- Serialize morphism mapping
serialize-morph-map : MorphismMapping → String
serialize-morph-map (morph-map id src tgt op) =
  "(morphism " -- TODO

-- Serialize categorical functor
serialize-functor : CategoricalFunctor → String
serialize-functor (cat-functor name objs morphs funct ins outs) =
  "(functor " -- TODO: serialize all fields

-- Serialize sheaf IR
serialize-sheaf : SheafIR → String
serialize-sheaf (sheaf-ir funct merges) =
  "(sheaf " -- TODO

-- Serialize fibration IR
serialize-fibration : FibrationIR → String
serialize-fibration (fibration-ir name base-f total-f proj fibs) =
  "(fibration " -- TODO

{-|
## Example: Extract ConvNet from Topos.Architecture

Once Neural.Topos.Architecture is fixed, we'll import actual ConvNet examples
and extract them to categorical IR.
-}

-- TODO: Enable once Architecture.agda is fixed
-- module ConvNetExample where
--   open import Neural.Topos.Examples
--   example-convnet-ir : SheafIR
--   example-convnet-ir = extract-sheaf example-convnet
--   example-convnet-serialized : String
--   example-convnet-serialized = serialize-sheaf example-convnet-ir

{-|
## Notes on Implementation

### What We Gain

1. **Composition is free**: F₁(f ∘ g) = F₁(g) ∘ F₁(f) is proven in Agda,
   so Python doesn't need to track composition - it's automatic!

2. **Sheaf conditions as merge constraints**: F(A★) ≅ ∏ F(incoming)
   becomes a structural guarantee in the compiled code.

3. **Dependent shapes from fibrations**: π⁻¹(b) gives us all valid shapes
   at a given network point, enabling type-safe shape inference.

4. **Backprop from naturality**: η: F ⇒ G with naturality is exactly
   the chain rule, proven categorically.

### What's Missing (TODOs)

1. Actual layer IDs: Need to extract string identifiers from Layer type
2. Conversion of Nat to String: For serialization
3. Extraction traversal: Walk through functor structure to collect F₀, F₁
4. Protobuf schema: Real protocol buffer definitions
5. Examples: Extract from actual networks in Topos.Examples

### Python Side (Next Phase)

The Python categorical interpreter will:
```python
class FunctorCompiler:
    def __init__(self, categorical_ir: CategoricalFunctor):
        self.F0 = {obj.obj_id: compile_tensor_spec(obj.tensor_type)
                   for obj in categorical_ir.objects}
        self.F1 = {morph.morph_id: compile_operation(morph.operation)
                   for morph in categorical_ir.morphisms}

    def compile(self):
        # Composition is automatic from F-∘!
        def forward(x, path):
            return self.F1[path](x)  # Composition built-in
        return jit(forward)
```

No manual graph traversal needed - functoriality handles composition!
-}
