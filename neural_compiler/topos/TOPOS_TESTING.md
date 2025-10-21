# Topos Property Testing

Comprehensive test suite for verifying categorical structure of geometric morphisms.

## Test Files

### 1. `test_topos_properties.py` - Property-Based Tests

Uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing. Generates random inputs and verifies categorical laws hold universally.

**Key Properties Tested:**

| Category | Properties | Description |
|----------|-----------|-------------|
| **Adjunction** | `f^* ⊣ f_*` | Unit/counit laws, triangle identities |
| **Functor Laws** | `F(id) = id`, `F(g∘f) = F(g)∘F(f)` | Identity preservation, composition |
| **Sheaf Condition** | `F(U) ≅ lim F(Uᵢ)` | Gluing axiom, restriction compatibility |
| **Geometric Morphism** | Pullback/pushforward | Structure preservation, continuity |
| **Grid Encoding** | Determinism, preservation | ARC-specific properties |
| **Integration** | Full pipeline | End-to-end verification |

**Run property tests:**
```bash
cd /Users/faezs/homotopy-nn/neural_compiler/topos
pytest test_topos_properties.py -v --hypothesis-show-statistics
```

**Configuration:**
- `max_examples=50`: Runs 50 random examples per property
- `deadline=5000`: 5 second timeout per test
- Generates random sheaf sections, ARC grids, morphisms

---

### 2. `test_topos_laws.py` - Direct Numerical Tests

Explicit tests on concrete examples. Complements property tests with known cases.

**Test Classes:**

| Class | Tests |
|-------|-------|
| `TestAdjunctionLaws` | Unit/counit triangle identities |
| `TestFunctorLaws` | Identity preservation, composition |
| `TestSheafCondition` | Gluing uniqueness, locality, compatibility |
| `TestPullbackPushforward` | Roundtrip properties, sheaf preservation |
| `TestGridEncoding` | Determinism, information preservation |
| `TestToposStructure` | Site coverage, finite sections |
| `TestRegressions` | Known examples (identity, zero grids) |

**Run direct tests:**
```bash
pytest test_topos_laws.py -v -s
```

---

## Categorical Laws Verified

### Adjunction Laws: f^* ⊣ f_*

1. **Unit Triangle Identity**
   ```
   (f_* ∘ η) ∘ f_* = f_*
   ```
   Where `η: id → f^* ∘ f_*` is the unit of adjunction.

2. **Counit Triangle Identity**
   ```
   f^* ∘ (ε ∘ f^*) = f^*
   ```
   Where `ε: f_* ∘ f^* → id` is the counit.

3. **Adjunction Symmetry**
   ```
   Hom(f_*(F), G) ≅ Hom(F, f^*(G))
   ```

**Verification:** `check_adjunction()` should return ≈ 0

---

### Functor Laws

1. **Identity Preservation**
   ```
   F(id_X) = id_{F(X)}
   ```
   Test: `f^* ∘ f_*` should be close to identity

2. **Composition**
   ```
   F(g ∘ f) = F(g) ∘ F(f)
   ```
   Test: Composing morphisms should be associative

**Verification:** Roundtrip error should be low

---

### Sheaf Condition

1. **Gluing Axiom**
   ```
   F(U) ≅ lim_{i} F(Uᵢ)
   ```
   For covering {Uᵢ} of U, sections glue uniquely.

2. **Locality**
   ```
   If s|_{Uᵢ} = 0 for all i, then s = 0
   ```

3. **Compatibility**
   ```
   If Uᵢ ⊆ Uⱼ, then F(Uⱼ)|_{Uᵢ} = F(Uᵢ)
   ```

**Verification:** `total_sheaf_violation()` should be ≈ 0

---

### Pullback Preservation

Geometric morphisms preserve limits:
```
f^* preserves all small limits
```

**Test:** Pullback of product is product of pullbacks

---

## Expected Results

### Untrained Network (Random Initialization)

| Law | Expected Range | Interpretation |
|-----|----------------|----------------|
| Adjunction violation | 1.0 - 10.0 | Not learned yet |
| Sheaf violation | 0.5 - 5.0 | Partial gluing |
| Roundtrip error | 5.0 - 50.0 | Far from identity |
| Section norm ratio | 0.5 - 2.0 | Some preservation |

### After Training

| Law | Target Range | Interpretation |
|-----|--------------|----------------|
| Adjunction violation | < 0.1 | Nearly perfect adjunction |
| Sheaf violation | < 0.5 | Strong gluing property |
| Roundtrip error | < 1.0 | Close to identity |
| Section norm ratio | 0.9 - 1.1 | Good structure preservation |

**Goal:** Training should **decrease all violations** toward 0.

---

## Running Tests

### Quick Test (Direct Tests Only)
```bash
pytest test_topos_laws.py -v -s
```

### Full Test Suite (Property + Direct)
```bash
pytest test_topos_properties.py test_topos_laws.py -v --hypothesis-show-statistics
```

### Run Specific Test Class
```bash
pytest test_topos_laws.py::TestAdjunctionLaws -v -s
```

### Run with Coverage
```bash
pytest test_topos_properties.py test_topos_laws.py --cov=geometric_morphism_torch --cov-report=html
```

---

## Interpreting Results

### All Tests Pass ✓
- Categorical structure is well-formed
- Topos axioms are satisfied (within tolerance)
- Safe to use for training

### Some Tests Fail ✗
- Check which laws are violated
- Inspect error magnitudes
- May indicate:
  - Network not trained enough
  - Hyperparameters need tuning
  - Architectural issue

### Example Output
```
test_adjunction_unit_triangle_identity PASSED
  Unit triangle error: 0.123456

test_sheaf_gluing_axiom PASSED
  Sheaf violation: 0.234567

test_functor_identity_preservation PASSED
  Identity preservation error: 1.234567
```

**Lower errors = better categorical structure**

---

## Integration with Training

The training script (`train_arc_geometric_production.py`) logs these violations in TensorBoard:

```python
# TensorBoard scalars
writer.add_scalar('ToposLaws/adjunction_violation', adj_violation, epoch)
writer.add_scalar('ToposLaws/sheaf_violation', sheaf_violation, epoch)
writer.add_scalar('ToposLaws/roundtrip_error', roundtrip_error, epoch)
writer.add_scalar('ToposLaws/section_norm_ratio', norm_ratio, epoch)
```

**Monitor these during training** to verify the network is learning categorical structure, not just memorizing grid transformations.

---

## Hypothesis Configuration

Property tests use these settings:

```python
@settings(max_examples=50, deadline=5000)
```

- `max_examples=50`: Runs 50 random test cases per property
- `deadline=5000`: 5 second timeout per test
- Can increase for more thorough testing:
  ```python
  @settings(max_examples=200, deadline=10000)
  ```

---

## Dependencies

```bash
pip install pytest hypothesis hypothesis[numpy]
```

Already available in the nix environment.

---

## Future Enhancements

1. **Lex morphism tests**: Verify left-exact functors preserve finite limits
2. **Subobject classifier**: Test Ω object properties
3. **Exponential objects**: Test internal hom [A, B]
4. **Natural isomorphisms**: Verify Yoneda lemma
5. **Grothendieck topology**: Test stability, transitivity axioms

---

## References

- **Adjunction**: Mac Lane, "Categories for the Working Mathematician", Ch. IV
- **Sheaves**: Mac Lane & Moerdijk, "Sheaves in Geometry and Logic", Ch. II
- **Geometric Morphisms**: Johnstone, "Topos Theory", Ch. 3
- **Property Testing**: [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
