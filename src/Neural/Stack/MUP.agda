{-# OPTIONS --cubical --allow-unsolved-metas #-}

{-|
# MUP (Maximal Update Parameterization) as a Stack

Implements MUP scaling rules as a fibration over model widths.

## Key Insight
MUP forms a stack where:
- **Base category**: Model widths (Nat)
- **Fibers**: Networks at each width
- **MUP sections**: Learning rates and initializations that scale correctly

## MUP Scaling Rules (Yang et al. 2022)
1. **Hidden layer weights**: σ_init = base_std / √width
2. **Output layer weights**: σ_init = base_std / width
3. **Hidden layer LR**: lr = base_lr
4. **Output layer LR**: lr = base_lr / width

This ensures:
- Feature learning stays O(1) across widths
- Hyperparameters transfer from small to large models
- Infinite-width limit is well-defined

## References
- Yang et al. "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
- https://arxiv.org/abs/2203.03466
-}

module Neural.Stack.MUP where

open import 1Lab.Prelude
open import 1Lab.Path
open import 1Lab.Type

open import Data.Nat using (Nat; zero; suc; _+_; _*_)
open import Data.Fin using (Fin)

-- Import smooth reals for learning rates
open import Neural.Smooth.Base using (ℝ; _+ℝ_; _·ℝ_; _/ℝ_; sqrtℝ)

-- Import stack infrastructure
open import Neural.Stack.Base using (Discrete; Discrete-is-set)

--------------------------------------------------------------------------------
-- § 1: MUP Configuration Space
--------------------------------------------------------------------------------

-- | Base MUP hyperparameters (width-independent)
record MUPBase : Type where
  field
    base-lr      : ℝ  -- Base learning rate (e.g., 0.1)
    base-std     : ℝ  -- Base initialization std (e.g., 0.02)
    hidden-depth : Nat -- Number of hidden layers

-- | Width-dependent MUP hyperparameters
record MUPConfig (width : Nat) : Type where
  field
    base : MUPBase

    -- Scaled learning rates (MUP rules)
    hidden-lr : ℝ     -- = base-lr (no scaling for hidden)
    output-lr : ℝ     -- = base-lr / width (scaled down)

    -- Scaled initialization stds (MUP rules)
    hidden-std : ℝ    -- = base-std / √width
    output-std : ℝ    -- = base-std / width

    -- Width consistency
    model-width : Nat
    width-eq : model-width ≡ width

-- | MUP scaling functions
postulate
  mup-hidden-lr : MUPBase → Nat → ℝ
  mup-output-lr : MUPBase → Nat → ℝ
  mup-hidden-std : MUPBase → Nat → ℝ
  mup-output-std : MUPBase → Nat → ℝ

-- Axioms: MUP scaling rules
postulate
  mup-hidden-lr-def : ∀ (base : MUPBase) (w : Nat) →
    mup-hidden-lr base w ≡ MUPBase.base-lr base

  mup-output-lr-def : ∀ (base : MUPBase) (w : Nat) →
    mup-output-lr base w ≡ (MUPBase.base-lr base) /ℝ (ℝ-from-nat w)

  mup-hidden-std-def : ∀ (base : MUPBase) (w : Nat) →
    mup-hidden-std base w ≡ (MUPBase.base-std base) /ℝ (sqrtℝ (ℝ-from-nat w))

  mup-output-std-def : ∀ (base : MUPBase) (w : Nat) →
    mup-output-std base w ≡ (MUPBase.base-std base) /ℝ (ℝ-from-nat w)

-- Convert Nat to ℝ (postulated for now)
postulate
  ℝ-from-nat : Nat → ℝ

-- | Construct MUP config at specific width
mk-mup-config : (base : MUPBase) → (w : Nat) → MUPConfig w
mk-mup-config base w = record
  { base = base
  ; hidden-lr = mup-hidden-lr base w
  ; output-lr = mup-output-lr base w
  ; hidden-std = mup-hidden-std base w
  ; output-std = mup-output-std base w
  ; model-width = w
  ; width-eq = refl
  }

--------------------------------------------------------------------------------
-- § 2: MUP as a Stack (Fibration over Widths)
--------------------------------------------------------------------------------

-- | Base category: Discrete category of widths
-- Objects: Nat (model widths)
-- Morphisms: Only identities (widths don't change during training)

-- | Total space: Pairs (width, config)
MUPStack : Type
MUPStack = Σ Nat MUPConfig

-- | Projection to base (width)
π-width : MUPStack → Nat
π-width (w , _) = w

-- | Fiber over width w: All MUP configs at that width
MUPFiber : Nat → Type
MUPFiber w = MUPConfig w

-- | Section: Canonical MUP config from base hyperparameters
mup-section : MUPBase → (w : Nat) → MUPFiber w
mup-section = mk-mup-config

--------------------------------------------------------------------------------
-- § 3: Network Architecture with MUP
--------------------------------------------------------------------------------

-- | Layer specification
data LayerType : Type where
  InputLayer  : Nat → LayerType           -- Input dimension
  HiddenLayer : Nat → LayerType           -- Hidden width
  OutputLayer : Nat → LayerType           -- Output dimension

-- | MLP architecture
record MLPArch : Type where
  field
    input-dim  : Nat
    width      : Nat         -- Hidden width (the varying dimension)
    output-dim : Nat
    depth      : Nat         -- Number of hidden layers

-- | Layer with MUP hyperparameters
record MUPLayer : Type where
  field
    layer-type : LayerType
    lr         : ℝ           -- Learning rate for this layer
    init-std   : ℝ           -- Initialization std for this layer

-- | Build MUP-parameterized network
build-mup-network : MLPArch → MUPBase → MUPConfig (MLPArch.width arch)
build-mup-network arch base = mk-mup-config base (MLPArch.width arch)

--------------------------------------------------------------------------------
-- § 4: Compilation to JAX Configuration
--------------------------------------------------------------------------------

-- | JAX-compatible config (will serialize to JSON)
record JAXMUPConfig : Type where
  field
    -- Architecture
    input-dim  : Nat
    width      : Nat
    output-dim : Nat
    depth      : Nat

    -- MUP hyperparameters
    hidden-lr   : ℝ
    output-lr   : ℝ
    hidden-std  : ℝ
    output-std  : ℝ

    -- Training config
    batch-size  : Nat
    num-epochs  : Nat
    optimizer   : String  -- "sgd" or "adam"

open import Agda.Builtin.String using (String)

-- | Compile MUP stack to JAX config
compile-mup-to-jax : MLPArch → MUPConfig (MLPArch.width arch) → JAXMUPConfig
compile-mup-to-jax arch config = record
  { input-dim  = MLPArch.input-dim arch
  ; width      = MLPArch.width arch
  ; output-dim = MLPArch.output-dim arch
  ; depth      = MLPArch.depth arch
  ; hidden-lr  = MUPConfig.hidden-lr config
  ; output-lr  = MUPConfig.output-lr config
  ; hidden-std = MUPConfig.hidden-std config
  ; output-std = MUPConfig.output-std config
  ; batch-size = 32
  ; num-epochs = 100
  ; optimizer  = "adam"
  }

--------------------------------------------------------------------------------
-- § 5: MUP Transfer Theorem
--------------------------------------------------------------------------------

-- | Theorem: MUP configs at different widths are "compatible"
-- (Informally: hyperparameters transfer from small to large models)

postulate
  -- Feature learning stays O(1) across widths
  feature-learning-bounded : ∀ (base : MUPBase) (w1 w2 : Nat) →
    let c1 = mk-mup-config base w1
        c2 = mk-mup-config base w2
    in {!!}  -- ∃ bound. ∀ features. ||feature|| < bound

  -- Infinite width limit exists
  infinite-width-limit : ∀ (base : MUPBase) →
    {!!}  -- limit_{w→∞} mup-config base w exists

--------------------------------------------------------------------------------
-- § 6: Example: Concrete MUP Configuration
--------------------------------------------------------------------------------

-- | Standard MUP base config (from Yang et al.)
open import Neural.Smooth.Base using (_/ₙ_)

example-mup-base : MUPBase
example-mup-base = record
  { base-lr      = 1 /ₙ 10  -- 0.1 (as ℝ)
  ; base-std     = 1 /ₙ 50  -- 0.02 (as ℝ)
  ; hidden-depth = 3
  }

-- | Example: Small model (width=64)
example-small-arch : MLPArch
example-small-arch = record
  { input-dim  = 28 * 28    -- MNIST input
  ; width      = 64
  ; output-dim = 10         -- 10 classes
  ; depth      = 3
  }

-- | Example: Large model (width=1024)
example-large-arch : MLPArch
example-large-arch = record
  { input-dim  = 28 * 28
  ; width      = 1024
  ; output-dim = 10
  ; depth      = 3
  }

-- | MUP configs (hyperparameters transfer!)
example-small-config : MUPConfig 64
example-small-config = build-mup-network example-small-arch example-mup-base

example-large-config : MUPConfig 1024
example-large-config = build-mup-network example-large-arch example-mup-base

-- | Compile both to JAX
example-small-jax : JAXMUPConfig
example-small-jax = compile-mup-to-jax example-small-arch example-small-config

example-large-jax : JAXMUPConfig
example-large-jax = compile-mup-to-jax example-large-arch example-large-config

--------------------------------------------------------------------------------
-- § 7: Export for FFI
--------------------------------------------------------------------------------

-- Functions to extract from MUP config (for Haskell FFI)
postulate
  mup-get-hidden-lr  : ∀ {w} → MUPConfig w → ℝ
  mup-get-output-lr  : ∀ {w} → MUPConfig w → ℝ
  mup-get-hidden-std : ∀ {w} → MUPConfig w → ℝ
  mup-get-output-std : ∀ {w} → MUPConfig w → ℝ

-- JSON serialization (postulated, implemented in JAX module)
postulate
  mup-config-to-json : JAXMUPConfig → String
  jax-config-to-python : JAXMUPConfig → String  -- Generate Python code
