# Smooth Infinitesimal Analysis for Neural Networks

This directory implements **Synthetic Differential Geometry** (Smooth Infinitesimal Analysis) in Agda, providing rigorous foundations for calculus in neural network theory.

## Overview

**Smooth Infinitesimal Analysis** is an alternative approach to calculus that uses:
- **Nilsquare infinitesimals**: Quantities ε where ε² = 0 but ε ≠ 0
- **Microaffineness**: Every function on infinitesimals is affine
- **Exact derivatives**: f(x+ε) = f(x) + ε·f'(x) (not approximate!)
- **Automatic smoothness**: Every function has derivatives of all orders

This provides a constructive, synthetic approach to differential geometry compatible with Homotopy Type Theory.

## Motivation for Neural Networks

Current neural network theory uses:
- **Postulated ℝ** (real numbers without structure)
- **Informal derivatives** (∂L/∂w "computed" without foundations)
- **Hand-waved backpropagation** (chain rule applied intuitively)
- **Tangent spaces** (mentioned but not formalized)

Smooth Infinitesimal Analysis provides:
- **Rigorous ℝ** with field operations and order
- **Exact derivatives** via f(x+ε) = f(x) + ε·f'(x)
- **Provable chain rule** (composite-rule in Calculus.agda)
- **Tangent bundles** as maps from microneighbourhood Δ

## Module Structure

### Phase 1: Foundations (Completed ✓)

**`Base.agda`** - Smooth line and infinitesimals
- Smooth line ℝ with field operations (+, -, ·, /)
- Order relation < with transitivity, compatibility
- Microneighbourhood Δ = {ε : ε² = 0}
- Principle of Microaffineness: Every g : Δ → ℝ has form g(ε) = g(0) + b·ε
- Microcancellation: Can cancel ε from universal equations
- Theorem 1.1: Δ properties (in [0,0], non-degenerate, LEM fails)

**`Calculus.agda`** - Differential calculus
- Derivative f'(x) defined by f(x+ε) = f(x) + ε·f'(x)
- Higher derivatives f'', f''', f⁽ⁿ⁾
- Calculus rules: sum, scalar, product, quotient, chain, inverse
- Fermat's rule: Stationary points ⟺ f'(a) = 0
- Constancy Principle: f' = 0 ⟹ f constant
- Indecomposability: ℝ cannot be split into disjoint parts

**`Functions.agda`** - Special functions
- Square root √ : ℝ₊ → ℝ₊ with derivative 1/(2√x)
- Trigonometric: sin, cos with sin' = cos, cos' = -sin
- On infinitesimals: sin ε = ε, cos ε = 1
- Exponential: exp with exp' = exp, exp(x+y) = exp(x)·exp(y)
- Logarithm: log with log' = 1/x
- Hyperbolic: sinh, cosh

### Phase 2: Geometric Applications (Planned)

**`Geometry.agda`** - Areas, volumes, integration
- Fundamental Theorem of Calculus: A'(x) = f(x)
- Method of microvariations for areas/volumes
- Circle area = πr² via Kepler's method
- Volume of revolution
- Surface areas

**`Curves.agda`** - Parametric curves and curvature
- Parametric curves r(t) = (x(t), y(t))
- Tangent vectors as microincrements dr/dt
- Arc length s with ds/dt = √((dx/dt)² + (dy/dt)²)
- Curvature κ = |dT/ds|
- Osculating circle and radius of curvature

### Phase 3: Neural Network Integration (Planned)

**`Manifolds.agda`** - Smooth manifolds
- Replace postulated manifolds in `Stack/CatsManifold.agda`
- Tangent bundle TM with tangent vectors as Δ → M
- Cotangent bundle T*M (differential forms)
- Vector fields as sections
- Lie bracket

**`InformationGeometry.agda`** - Statistical manifolds
- Replace postulates in `Information/Geometry.agda`
- Fisher-Rao metric using smooth probability distributions
- KL divergence as smooth function D_KL(p||q)
- Geodesics as smooth curves minimizing divergence
- Natural gradient: ∇̃L = F⁻¹∇L

**`Dynamics.agda`** - Neural ODEs
- Neural ODEs: dx/dt = f(x, θ, t)
- Flow equations for continuous-time networks
- Backpropagation through ODE solvers (adjoint method)
- Lyapunov stability analysis
- Connection to `Dynamics/IntegratedInformation.agda`

**`Optimization.agda`** - Gradient-based optimization
- Gradient descent as flow on smooth manifold
- Critical points and Hessian analysis
- Natural gradient descent with Fisher-Rao geometry
- Constrained optimization (Lagrange multipliers)
- Connection to `Resources/Optimization.agda`

**`Backpropagation.agda`** - Rigorous backprop
- Formalize backprop using chain rule
- Gradient ∂L/∂w_ij via microvariations
- Connection to DirectedPath in `Topos/Architecture.agda`
- Natural transformations W → W interpretation
- Computational graph as smooth functor

**`Topos.agda`** - Topos-theoretic connections
- Smooth topos structure
- Internal language of smooth worlds
- Well-adapted models (where Microaffineness holds)
- Tangent bundle functor T : Smooth → Smooth
- Connection to `Topos/` modules

### Phase 4: Examples and Applications (Planned)

**`Examples.agda`** - Concrete calculations
- Area under parabola
- Circle area (πr²) and circumference
- Catenary curve
- Geodesics on sphere
- Simple network training dynamics

**`Applications.agda`** - Neural network use cases
- Activation functions (ReLU, sigmoid, tanh)
- Loss surfaces (MSE, cross-entropy)
- Batch normalization as projection
- Attention mechanism via geodesic interpolation
- Residual connections as parallel transport

## Key Principles

### 1. Infinitesimals are Nilsquare

```agda
Δ : Type
Δ = Σ[ ε ∈ ℝ ] (ε ·ℝ ε ≡ 0ℝ)
```

**Physical meaning**: ε is "smaller than any finite quantity" but ε ≠ 0.
Second-order effects (ε²) vanish exactly.

### 2. Microaffineness

```agda
Microaffine : Type
Microaffine = ∀ (g : Δ → ℝ) → Σ![ b ∈ ℝ ] (∀ (δ : Δ) → g δ ≡ g (0ℝ , refl) +ℝ (b ·ℝ ι δ))
```

**Physical meaning**: Functions on infinitesimals are affine (linear + constant).
Curves are locally straight at infinitesimal scale.

### 3. Fundamental Equation

```agda
fundamental-equation : (f : ℝ → ℝ) (x : ℝ) (δ : Δ) →
  f (x +ℝ ι δ) ≡ f x +ℝ (ι δ ·ℝ f ′[ x ])
```

**Physical meaning**: Derivative is the exact increment, not approximate.
This is the **defining equation** of calculus in smooth worlds.

### 4. Microcancellation

```agda
microcancellation : ∀ (a b : ℝ) →
  (∀ (δ : Δ) → (ι δ ·ℝ a) ≡ (ι δ ·ℝ b)) → a ≡ b
```

**Physical meaning**: Can cancel ε from equations that hold for all ε ∈ Δ.
Enables algebraic derivation of calculus rules.

## Comparison to Classical Analysis

| Classical Analysis | Smooth Infinitesimal Analysis |
|-------------------|-------------------------------|
| Limits: lim(Δx→0) | Infinitesimals: ε² = 0 |
| Approximate: f(x+h) ≈ f(x) + h·f'(x) | Exact: f(x+ε) = f(x) + ε·f'(x) |
| ε-δ proofs (complex) | Algebraic proofs (simple) |
| Not all functions smooth | All functions smooth |
| Trichotomy: a < b ∨ a = b ∨ b < a | No trichotomy (indistinguishable points) |
| ℝ = (-∞,0) ∪ {0} ∪ (0,∞) | ℝ indecomposable |

## Connection to Existing Codebase

This framework provides rigorous foundations for:

1. **`Neural.Information`** - Postulated ℝ replaced with smooth line
2. **`Neural.Information.Geometry`** - Tangent vectors, Fisher-Rao metric formalized
3. **`Neural.Topos.Architecture`** - Backpropagation as smooth natural transformations
4. **`Neural.Stack.CatsManifold`** - Smooth manifolds properly defined
5. **`Neural.Dynamics.*`** - Neural ODEs with actual derivatives
6. **`Neural.Resources.Optimization`** - Gradient descent on smooth manifolds

## Philosophical Background

Smooth Infinitesimal Analysis is based on **Synthetic Differential Geometry** (SDG), developed by:
- F. W. Lawvere (1967) - Categorical dynamics
- Anders Kock (1981) - Synthetic Differential Geometry book
- John L. Bell (1998, 2008) - A Primer of Infinitesimal Analysis
- Ieke Moerdijk & Gonzalo E. Reyes (1991) - Models for Smooth Infinitesimal Analysis

Key ideas:
- **Synthetic**: Axioms given directly (not constructed from sets)
- **Intuitionistic logic**: Law of excluded middle fails (ε = 0 ∨ ε ≠ 0 unprovable)
- **Topos theory**: Smooth worlds are toposes (categorical universes)
- **Compatible with HoTT**: Works in homotopy type theory / cubical Agda

## Advantages for Neural Networks

1. **Exact gradients**: Backpropagation is exact, not approximate
2. **Automatic differentiation**: Chain rule is provable, not assumed
3. **Smooth optimization**: Gradient descent has geometric meaning
4. **Information geometry**: Fisher-Rao metric properly formalized
5. **Tangent spaces**: Tangent bundle as Δ → M makes sense
6. **Topos connection**: Links to sheaf-theoretic DNN architectures

## Usage Example

```agda
-- Define a simple activation function
sigmoid : ℝ → ℝ
sigmoid x = (1ℝ /ℝ (1ℝ +ℝ exp (-ℝ x))) {proof-positive}

-- Its derivative is σ'(x) = σ(x)·(1 - σ(x))
sigmoid-deriv : ∀ (x : ℝ) →
  sigmoid ′[ x ] ≡ sigmoid x ·ℝ (1ℝ -ℝ sigmoid x)
sigmoid-deriv x = {proof using chain rule and exp' = exp}

-- Loss function
mse : (y y_true : ℝ) → ℝ
mse y y_true = (y -ℝ y_true) ·ℝ (y -ℝ y_true)

-- Gradient of loss w.r.t. prediction
∂mse/∂y : (y y_true : ℝ) → ℝ
∂mse/∂y y y_true = mse (y , y_true) ′[ y ]
                 ≡ (1ℝ +ℝ 1ℝ) ·ℝ (y -ℝ y_true)  -- by power rule

-- Backpropagation through sigmoid
backprop : (x y_true : ℝ) → ℝ
backprop x y_true =
  let y = sigmoid x
      ∂L/∂y = ∂mse/∂y y y_true
      ∂y/∂x = sigmoid-deriv x
  in ∂L/∂y ·ℝ ∂y/∂x  -- Chain rule!
```

## References

1. **John L. Bell** (2008), *A Primer of Infinitesimal Analysis* (2nd ed.), Cambridge University Press
   - Chapters 1-3: Foundations, calculus, geometry (implemented here)

2. **Anders Kock** (1981), *Synthetic Differential Geometry*, Cambridge University Press
   - Categorical foundations of smooth worlds

3. **Ieke Moerdijk & Gonzalo E. Reyes** (1991), *Models for Smooth Infinitesimal Analysis*, Springer
   - Topos-theoretic models

4. **F. W. Lawvere & Stephen H. Schanuel** (2009), *Conceptual Mathematics* (2nd ed.)
   - Category theory background

5. **Manin & Marcolli** (2024), *Homotopy-theoretic and categorical models of neural information networks*
   - Neural codes and directed graphs (our main reference)

6. **Belfiore & Bennequin** (2022), *Topos and Stacks of Deep Neural Networks*
   - Topos theory for DNNs (implemented in `Neural.Topos.*`)

## Future Directions

1. **Synthetic topology**: Open sets as abstract objects
2. **Synthetic Lie theory**: Lie groups and algebras
3. **Gauge theory**: Connections and curvature for neural fields
4. **Quantum networks**: Non-commutative smooth analysis
5. **Category of smooth worlds**: Morphisms between different smooth structures

## Contributing

When adding new modules:
1. Follow the naming convention: `Neural.Smooth.*`
2. Include comprehensive module documentation with paper references
3. Mark TODOs clearly for incomplete proofs
4. Use `postulate` for results to be proven later
5. Add physical interpretations for neural network applications
6. Connect to existing modules in `Neural.*`

## License

Same as parent project (homotopy-nn).
