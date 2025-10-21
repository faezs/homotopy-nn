# Topos-Theoretic Loss Computation

## Question: What is "Topos Loss"?

Current implementation uses standard MSE loss on sheaf sections (tensors):
```python
loss = F.mse_loss(predicted_sheaf.sections, target_sheaf.sections)
```

But this ignores the **categorical structure** of the topos! A proper topos loss should respect:

1. **Internal logic** (propositions in the topos)
2. **Sheaf conditions** (gluing axioms)
3. **Natural transformations** (morphisms between sheaves)
4. **Adjunction fidelity** (f^* ⊣ f_*)

---

## Formal Definition: Topos Loss

A topos loss L should be a **morphism in the internal language**:

```
L: 1 → Ω × ℝ₊
```

Where:
- 1 is the terminal object
- Ω is the subobject classifier (truth values)
- ℝ₊ is the non-negative reals

**Components**:

### 1. Internal Logic Loss (Already Implemented!)

```python
class InternalLogicLoss(nn.Module):
    """Loss as proposition: L = ∀U. (f_*(F_in)|_U ≡ F_target|_U)"""

    def compute(self, geometric_morphism, input_sheaf):
        predicted_sheaf = geometric_morphism.pushforward(input_sheaf)

        # Universal quantifier over all objects
        truth_values = []
        for obj_idx in range(predicted_sheaf.site.num_objects):
            pred = predicted_sheaf.at_object(obj_idx)
            target = self.target.at_object(obj_idx)

            # Equality as truth value in [0,1]
            equality = torch.exp(-torch.sum((pred - target) ** 2))
            truth_values.append(equality)

        # ∀ = min (conjunction)
        global_truth = torch.stack(truth_values).min()

        # Loss = negation of truth
        return 1.0 - global_truth
```

**Interpretation**: "For all opens U in the site, the predicted sheaf equals the target"

### 2. Sheaf Condition Violation

```python
def sheaf_condition_loss(sheaf: Sheaf) -> torch.Tensor:
    """
    Sheaf condition: F(U) ≅ lim_{i} F(U_i) over covering {U_i → U}

    Loss = Σ_U ||F(U) - glue({F(U_i)})||²
    """
    total_violation = 0.0

    for obj_idx in range(sheaf.site.num_objects):
        section_U = sheaf.at_object(obj_idx)
        covering = sheaf.site.coverage_families[obj_idx]

        # Glue sections from covering
        glued_sections = []
        for cover_obj in covering:
            if cover_obj != obj_idx:
                section_V = sheaf.at_object(cover_obj)
                restricted = sheaf.restrict(section_V, cover_obj, obj_idx)
                glued_sections.append(restricted)

        if len(glued_sections) > 0:
            # Gluing = limit (approximated as mean)
            glued = torch.stack(glued_sections).mean(dim=0)
            violation = torch.sum((section_U - glued) ** 2)
            total_violation += violation

    return total_violation / sheaf.site.num_objects
```

**Interpretation**: How badly does the sheaf violate the gluing axiom?

### 3. Adjunction Fidelity Loss

```python
def adjunction_loss(geo_morph: GeometricMorphism,
                    sheaf_in: Sheaf,
                    sheaf_out: Sheaf) -> torch.Tensor:
    """
    Adjunction: Hom(f^*(G), F) ≅ Hom(G, f_*(F))

    Check: ⟨f^*(G), F⟩ ≈ ⟨G, f_*(F)⟩
    """
    # Pullback then inner product with input
    pulled = geo_morph.pullback(sheaf_out)
    inner1 = torch.sum(pulled.sections * sheaf_in.sections)

    # Pushforward then inner product with output
    pushed = geo_morph.pushforward(sheaf_in)
    inner2 = torch.sum(sheaf_out.sections * pushed.sections)

    # Should be equal (adjunction)
    violation = torch.abs(inner1 - inner2)

    return violation
```

**Interpretation**: How well does f^* ⊣ f_* hold?

### 4. Natural Transformation Distance (Future Work)

For morphisms η: F ⇒ G between sheaves:

```python
def natural_transformation_loss(eta: NaturalTransformation,
                                 source: Sheaf,
                                 target: Sheaf) -> torch.Tensor:
    """
    Naturality: For f: U → V, the diagram commutes:

    F(V) --η_V--> G(V)
     |            |
    F(f)        G(f)
     |            |
     v            v
    F(U) --η_U--> G(U)

    Loss = Σ_{f: U→V} ||G(f) ∘ η_V - η_U ∘ F(f)||²
    """
    total_violation = 0.0

    for obj_u in range(source.site.num_objects):
        for obj_v in range(source.site.num_objects):
            if source.site.adjacency[obj_u, obj_v] > 0:
                # Top path: η_V then G(f)
                path1 = target.restrict(
                    eta.component(obj_v)(source.at_object(obj_v)),
                    obj_v, obj_u
                )

                # Bottom path: F(f) then η_U
                path2 = eta.component(obj_u)(
                    source.restrict(source.at_object(obj_v), obj_v, obj_u)
                )

                # Should commute
                violation = torch.sum((path1 - path2) ** 2)
                total_violation += violation

    return total_violation
```

**Interpretation**: How natural is the transformation?

---

## Combined Topos Loss

```python
def topos_loss(geometric_morphism: GeometricMorphism,
               input_sheaf: Sheaf,
               target_sheaf: Sheaf,
               weights: Dict[str, float] = None) -> torch.Tensor:
    """
    Combined topos-theoretic loss function.

    L_topos = α₁·L_internal + α₂·L_sheaf + α₃·L_adjunction

    Args:
        weights: {'internal': α₁, 'sheaf': α₂, 'adjunction': α₃}
    """
    if weights is None:
        weights = {'internal': 1.0, 'sheaf': 0.1, 'adjunction': 0.1}

    # 1. Internal logic: prediction equals target
    internal_loss_fn = InternalLogicLoss(target_sheaf)
    L_internal = internal_loss_fn.compute(geometric_morphism, input_sheaf)

    # 2. Sheaf condition: gluing axiom
    predicted = geometric_morphism.pushforward(input_sheaf)
    L_sheaf = sheaf_condition_loss(predicted)

    # 3. Adjunction: f^* ⊣ f_*
    L_adjunction = adjunction_loss(geometric_morphism, input_sheaf, target_sheaf)

    # Weighted sum
    total_loss = (weights['internal'] * L_internal +
                  weights['sheaf'] * L_sheaf +
                  weights['adjunction'] * L_adjunction)

    return total_loss
```

---

## Current Implementation vs Ideal

| Component | Current | Topos-Theoretic | Status |
|-----------|---------|-----------------|--------|
| **Main loss** | MSE on sections | Internal logic (∀U. pred ≡ target) | ✓ Implemented |
| **Sheaf condition** | Soft constraint | Gluing axiom violation | ✓ Implemented |
| **Adjunction** | Soft constraint | f^* ⊣ f_* fidelity | ✓ Implemented |
| **Naturality** | Not used | Naturality squares commute | ❌ Not implemented |
| **Limit preservation** | Not checked | f^* preserves finite limits | ❌ Not implemented |

---

## Why Use Topos Loss?

### 1. **Categorical Correctness**
Standard MSE treats sheaves as tensors, ignoring structure. Topos loss respects:
- Gluing axioms
- Functoriality
- Adjunctions

### 2. **Compositionality**
If we learn f: E₁ → E₂ and g: E₂ → E₃ with topos loss, then g ∘ f is also a geometric morphism with guarantees.

### 3. **Interpretability**
Loss components have categorical meaning:
- L_internal = "prediction is wrong"
- L_sheaf = "not a valid sheaf"
- L_adjunction = "not a valid geometric morphism"

### 4. **Generalization**
Training with categorical constraints should generalize better than raw tensor regression.

---

## Accuracy in Topos Space

**Current accuracy**: Discrete grid comparison
```python
accuracy = |{(i,j) : pred[i,j] == target[i,j]}| / (h × w)
```
This is in **ℤ₀₋₉^(h×w)** (decoded discrete space).

**Topos accuracy**: Sheaf space comparison
```python
sheaf_accuracy = 1 - ||f_*(F_in).sections - F_target.sections||₂ / ||F_target.sections||₂
```
This is in **ℝ^(num_objects × feature_dim)** (continuous sheaf space).

**Better metric**: Internal logic truth value
```python
topos_accuracy = ∀U. exp(-||pred(U) - target(U)||²)
```
Returns truth value ∈ [0,1] in the internal logic.

---

## Implementation Recommendation

Replace current loss computation:

```python
# Current (train_arc_geometric_production.py line 238):
loss = F.mse_loss(predicted_sheaf.sections, target_sheaf.sections)

# Proposed:
topos_loss_fn = ToposLoss(target_sheaf, weights={
    'internal': 1.0,
    'sheaf': 0.1,
    'adjunction': 0.1
})
loss = topos_loss_fn.compute(solver.geometric_morphism, input_sheaf)
```

This makes the loss **categorically meaningful** rather than just minimizing tensor distance.

---

## TensorBoard Logging

Now logging:
```
Loss/train                  - Internal logic loss
Loss/adjunction_violation   - Adjunction fidelity
Loss/sheaf_violation        - Gluing axiom violation
Metrics/accuracy            - Grid-space discrete accuracy
```

Run: `tensorboard --logdir=/Users/faezs/homotopy-nn/neural_compiler/topos/runs/`

---

## Next Steps

1. ✅ **TensorBoard setup** - Done!
2. ✅ **Explain topos loss** - Done!
3. ⏭️ **Implement ToposLoss class** - Create new class combining all losses
4. ⏭️ **Add limit preservation check** - Verify f^* preserves products/equalizers
5. ⏭️ **Natural transformation metrics** - Track naturality violations
6. ⏭️ **SMT verification** - Use SBV/Z3 to verify topos axioms hold

---

**Summary**: Current loss is tensor-space MSE. True topos loss would be a weighted combination of:
1. Internal logic proposition (prediction = target)
2. Sheaf gluing violation
3. Adjunction fidelity
4. Naturality of transformations

All components are already partially implemented - just need to combine them properly!
