{-# OPTIONS --no-import-sorts #-}
{-|
# Examples of DNN Topoi

Concrete examples of neural network architectures and their associated topoi.

Following Belfiore & Bennequin (2022), Figure 1.4.
-}

module Neural.Topos.Examples where

open import 1Lab.Prelude
open import 1Lab.HLevel
open import 1Lab.Type.Sigma

open import Cat.Instances.Graphs using (Graph)
open import Data.Nat.Base using (Nat; zero; suc; _+_)
open import Data.Fin.Base using (Fin; fzero; fsuc)
open import Data.Bool.Base using (Bool; true; false)
open import Data.Sum.Base using (_⊎_; inl; inr)
open import Data.Sum.Properties using (⊎-is-hlevel)

open import Neural.Topos.Architecture

private variable
  o ℓ : Level

{-|
## Example 1: Simple Multi-Layer Perceptron (Chain)

A chain network: x₀ → h₁ → h₂ → y

No convergence, so no forks needed!
Poset X is just: y ≤ h₂ ≤ h₁ ≤ x₀
-}
module SimpleMLP where
  -- Graph with 4 vertices (input, 2 hidden, output) and 3 edges
  open OrientedGraph

  -- The underlying graph structure
  mlp-graph : Graph lzero lzero
  mlp-graph .Graph.Vertex = Fin 4
    -- 0 = input x₀
    -- 1 = hidden h₁
    -- 2 = hidden h₂
    -- 3 = output y
  mlp-graph .Graph.Edge i j =
    (i ≡ fzero) × (j ≡ fsuc fzero) ⊎  -- x₀ → h₁
    (i ≡ fsuc fzero) × (j ≡ fsuc (fsuc fzero)) ⊎  -- h₁ → h₂
    (i ≡ fsuc (fsuc fzero)) × (j ≡ fsuc (fsuc (fsuc fzero)))  -- h₂ → y
  mlp-graph .Graph.Vertex-is-set = hlevel 2
  mlp-graph .Graph.Edge-is-set {i} {j} =
    ⊎-is-hlevel 0 (×-is-hlevel 2 (hlevel 2) (hlevel 2))
                   (⊎-is-hlevel 0 (×-is-hlevel 2 (hlevel 2) (hlevel 2))
                                  (×-is-hlevel 2 (hlevel 2) (hlevel 2)))

  -- This is an oriented graph (no cycles, classical, no loops)
  postulate mlp-oriented : OrientedGraph lzero lzero
  -- In a real proof, we'd show:
  --   classical: at most one edge between vertices (obvious from definition)
  --   no-loops: no edge i → i (obvious)
  --   directed: no cycles (can be shown by case analysis on Fin 4)

  -- The poset X is trivial - just the chain!
  -- X-Vertex: 0, 1, 2, 3
  -- Ordering: 3 ≤ 2 ≤ 1 ≤ 0  (output ≤ ... ≤ input)
  --
  -- This is a totally ordered poset (a chain), which is the simplest case.

{-|
## Example 1b: SimpleMLP with MUP (Maximal Update Parameterization)

Extends SimpleMLP with MUP scaling rules (Yang et al. 2022).

**MUP ensures**: Feature learning stays O(1) across widths, enabling
hyperparameter transfer from small to large models.

**Architecture** (same chain):
```
input(784) → h₁(w) → h₂(w) → output(10)
```

**MUP Scaling** (w = hidden width):
- Hidden layers: init_std = base_std / √w, lr = base_lr
- Output layer:  init_std = base_std / w,  lr = base_lr / w

This allows testing at w=64, then transferring hyperparameters to w=1024!
-}
module SimpleMLP-MUP where
  open SimpleMLP public

  -- Import MUP infrastructure
  open import Neural.Stack.MUP
  open import Data.Nat using (_*_)
  open import Neural.Smooth.Base using (ℝ; _/ₙ_)

  -- Width function: hidden layers have variable width
  mlp-width : Fin 4 → Nat
  mlp-width fzero = 28 * 28  -- input (MNIST: 784)
  mlp-width (fsuc fzero) = 64       -- h₁ (variable for MUP test!)
  mlp-width (fsuc (fsuc fzero)) = 64 -- h₂ (variable for MUP test!)
  mlp-width (fsuc (fsuc (fsuc fzero))) = 10 -- output (10 classes)

  -- MUP base hyperparameters (width-independent)
  mlp-mup-base : MUPBase
  mlp-mup-base = record
    { base-lr      = 1 /ₙ 10  -- 0.1 (ℝ)
    ; base-std     = 1 /ₙ 50  -- 0.02 (ℝ)
    ; hidden-depth = 2     -- Two hidden layers
    }

  -- MUP config at specific width w
  mlp-mup-config : (w : Nat) → MUPConfig w
  mlp-mup-config w = mk-mup-config mlp-mup-base w

  -- Example: Small model (width=64)
  mlp-config-64 : MUPConfig 64
  mlp-config-64 = mlp-mup-config 64

  -- Example: Large model (width=1024)
  mlp-config-1024 : MUPConfig 1024
  mlp-config-1024 = mlp-mup-config 1024

  {-|
  ### Weight Spaces with MUP Scaling

  For each layer, the weight initialization follows MUP rules:
  - Input  → h₁:  σ = base_std / √64  = 0.025  (at width 64)
  - h₁     → h₂:  σ = base_std / √64  = 0.025
  - h₂     → out: σ = base_std / 64   = 0.00031 (stronger scaling!)

  At width 1024:
  - Input  → h₁:  σ = base_std / √1024 = 0.000625
  - h₁     → h₂:  σ = base_std / √1024 = 0.000625
  - h₂     → out: σ = base_std / 1024  = 0.0000195
  -}

  -- Matrix type for weights: m×n matrix with initialization std σ
  record Matrix (m n : Nat) (σ : ℝ) : Type where
    field
      entries : Fin m → Fin n → ℝ

  {-# COMPILE GHC Matrix = data Matrix #-}

  -- Weight space for each edge in the chain
  WeightSpace-MUP : (w : Nat) → Fin 4 → Type
  WeightSpace-MUP w fzero =
    -- Input → h₁: use hidden std
    let config = mlp-mup-config w
    in Matrix (mlp-width fzero) w (MUPConfig.hidden-std config)

  WeightSpace-MUP w (fsuc fzero) =
    -- h₁ → h₂: use hidden std
    let config = mlp-mup-config w
    in Matrix w w (MUPConfig.hidden-std config)

  WeightSpace-MUP w (fsuc (fsuc fzero)) =
    -- h₂ → output: use output std (stronger scaling!)
    let config = mlp-mup-config w
    in Matrix w (mlp-width (fsuc (fsuc (fsuc fzero))))
               (MUPConfig.output-std config)

  WeightSpace-MUP w (fsuc (fsuc (fsuc fzero))) =
    -- Output layer (no outgoing edges)
    Lift _ ⊤

  {-|
  ### Learning Rates with MUP Scaling

  - Hidden layers: lr = base_lr (no scaling)
  - Output layer:  lr = base_lr / width (scaled down)
  -}
  LearningRate-MUP : (w : Nat) → Fin 4 → ℝ
  LearningRate-MUP w fzero = MUPBase.base-lr mlp-mup-base
  LearningRate-MUP w (fsuc fzero) = MUPBase.base-lr mlp-mup-base
  LearningRate-MUP w (fsuc (fsuc fzero)) =
    MUPConfig.output-lr (mlp-mup-config w)  -- base_lr / w
  LearningRate-MUP w (fsuc (fsuc (fsuc fzero))) =
    MUPConfig.output-lr (mlp-mup-config w)

{-|
## Example 2: Convergent Network (ResNet-like)

Two input branches converging to a single layer:

```
  x₀,₁    x₀,₂     (inputs)
    ↓      ↓
    └──→ h ←──┘    (convergence - needs FORK!)
         ↓
         y         (output)
```

**Fork construction** (Section 1.3):
- Original vertices: x₀,₁, x₀,₂, h, y
- Convergence at h (two inputs)
- Add: A★ (star), A (tang)
- Structure in Γ_fork:
  ```
  x₀,₁ → A★ ← x₀,₂    (tips to star)
         ↓
         A             (star to tang)
         ↓
         h → y         (tang to handle, handle to output)
  ```

**Poset X** (remove A★):
```
Maximal (inputs/tangs):  x₀,₁  x₀,₂  A
                           ↓      ↓   ↓
                                  h
                                  ↓
Minimal (outputs/tips):           y
```

Ordering in X:
- y ≤ h  (output below hidden)
- h ≤ A  (handle below tang)
- A ≤ x₀,₁  (tang below tip)
- A ≤ x₀,₂  (tang below tip)

This forms a **diamond** (join-semilattice with top element y).
-}
module ConvergentNetwork where
  open OrientedGraph

  -- Original graph before fork
  data OriginalVertex : Type where
    input₁ : OriginalVertex
    input₂ : OriginalVertex
    hidden : OriginalVertex
    output : OriginalVertex

  data OriginalEdge : OriginalVertex → OriginalVertex → Type where
    edge-i1h : OriginalEdge input₁ hidden
    edge-i2h : OriginalEdge input₂ hidden
    edge-hy  : OriginalEdge hidden output

  -- hidden has 2 inputs → is-convergent!
  postulate
    input₁≠input₂ : ¬ (input₁ ≡ input₂)

  hidden-is-convergent : Σ[ x ∈ OriginalVertex ] Σ[ y ∈ OriginalVertex ]
                         (¬ (x ≡ y)) × OriginalEdge x hidden × OriginalEdge y hidden
  hidden-is-convergent = input₁ , input₂ , input₁≠input₂ , edge-i1h , edge-i2h

  -- After fork construction (Section 1.3):
  data ForkedVertex : Type where
    orig : OriginalVertex → ForkedVertex
    fork-star-h : ForkedVertex  -- A★ for hidden convergence
    fork-tang-h : ForkedVertex  -- A for hidden convergence

  -- Poset X (remove A★):
  data X-Vertex : Type where
    x-input₁ : X-Vertex
    x-input₂ : X-Vertex
    x-tang   : X-Vertex  -- A (the join point)
    x-hidden : X-Vertex
    x-output : X-Vertex

  -- Ordering: arrows go OPPOSITE to information flow
  data _≤_ : X-Vertex → X-Vertex → Type where
    -- Reflexivity
    ≤-refl : ∀ {x} → x ≤ x

    -- Output to hidden
    ≤-out-hid : x-output ≤ x-hidden

    -- Hidden to tang (A → h in fork)
    ≤-hid-tang : x-hidden ≤ x-tang

    -- Tang to inputs (A → x₀,₁ and A → x₀,₂ via A★)
    ≤-tang-i1 : x-tang ≤ x-input₁
    ≤-tang-i2 : x-tang ≤ x-input₂

    -- Transitivity
    ≤-trans : ∀ {x y z} → x ≤ y → y ≤ z → x ≤ z

  -- Illustration of the poset:
  {-
          x-input₁    x-input₂     (maximal - inputs)
               ↘        ↙
               x-tang              (maximal - tang A)
                  ↓
              x-hidden
                  ↓
              x-output             (minimal - output)

  This is a **diamond** poset!
  Meet (∧): x-input₁ ∧ x-input₂ = x-tang
  Join (∨): only x-output is below both paths
  -}

{-|
## Example 3: Complex Multi-Path Network

From Figure 1.4 in the paper:

```
       Input layers (maximal)
    x₀,₁           x₀,₂
      ↓              ↓
      c₁   ←    →   c₂         (convergent layers)
       ↓     ↘ ↙     ↓
        ↓      c₃     ↓
         ↘    ↓  ↘   ↙
           xₙ,₁ ... xₙ,₅       (output layers - minimal)
```

After fork construction, get multiple tangs (A, B, C for different convergence points).

The poset X is a **tree forest** joining at the outputs:
- Multiple maximal elements: inputs + tangs
- Multiple minimal elements: outputs + tips
- Tree structure in between
-}
module ComplexNetwork where
  -- This would have:
  -- - Original vertices for layers
  -- - Multiple convergence points → multiple forks
  -- - Resulting poset with complex partial order

  -- The poset structure matches the diagram shown:
  {-
    Maximal:
      Inputs: x₀,₁, x₀,₂
      Tangs: A, B, C (from forks at c₁, c₂, c₃)

    Internal:
      Various convergent layers c₁, c₂, c₃

    Minimal:
      Outputs: xₙ,₁, xₙ,₂, xₙ,₃, xₙ,₄, xₙ,₅
      Tips: (vertices that feed into convergence points)

  This gives the tree structure described in Theorem 1.2:
  "The poset X of a DNN is made by a finite number of trees,
   rooted in the maximal points and which are joined in the minimal points."
  -}

  postulate complex-network : OrientedGraph lzero lzero
  -- In practice, this would be constructed from a real ResNet/DenseNet architecture

{-|
## Visualization: From Graph Γ to Poset X

**Step 1**: Network graph Γ (information flows down)
```
x₀,₁ → h ← x₀,₂     (TWO arrows into h!)
       ↓
       y
```

**Step 2**: Fork construction (add A★, A)
```
x₀,₁ → A★ ← x₀,₂    (tips converge at star)
       ↓
       A            (star → tang)
       ↓
       h            (tang → handle)
       ↓
       y
```

**Step 3**: Remove A★, reverse arrows → Poset X
```
x₀,₁ ← A ← x₀,₂     (in category: arrows go opposite to info flow)
       ↓
       h
       ↓
       y

Ordering:
  y ≤ h ≤ A ≤ x₀,₁
  y ≤ h ≤ A ≤ x₀,₂
```

**Step 4**: Alexandrov coverage
- Each x has one covering: maximal sieve (all y ≤ x)
- For A: covered by {A, x₀,₁, x₀,₂}
- For h: covered by {h, A, x₀,₁, x₀,₂}
- For y: covered by {y, h, A, x₀,₁, x₀,₂}

**Step 5**: DNN Topos = Sh[X, Alexandrov]
- Sheaves F : X^op → Sets
- F(layer) = states at that layer
- Morphisms preserve layer structure
-}
