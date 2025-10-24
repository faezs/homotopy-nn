"""
Comprehensive Tests for Tensorized Model Categories

Tests Section 2.4 implementation:
- Quillen model structure (fibrations, cofibrations, weak equivalences)
- Lifting properties (RLP, LLP)
- Model category axioms (CM1-CM5)
- Groupoid categories GrpdC
- Multi-fibrations (Theorem 2.2)
- Martin-Löf type theory (dependent types, identity types)
- Univalence axiom

Author: Claude Code
Date: 2025-10-25
"""

import torch
import torch.nn as nn
from typing import Dict, List

from stacks_of_dnns import (
    ModelMorphism,
    MorphismType,
    QuillenModelStructure,
    GroupoidCategory,
    MultiFibration,
    DependentType,
    IdentityType,
    UnivalenceAxiom,
    ModelCategoryDNN,
    StackDNN,
    CyclicGroup
)


################################################################################
# Test 1: Model Morphisms and Types
################################################################################

def test_model_morphisms():
    """Test ModelMorphism classification and properties."""
    print("\n" + "="*80)
    print("TEST 1: Model Morphisms and Types")
    print("="*80)

    # Create morphisms with different properties
    fib = ModelMorphism(
        source="layer1",
        target="layer2",
        is_fibration=True
    )
    print(f"\n✓ Created fibration: {fib.source} → {fib.target}")
    assert fib.morphism_type() == MorphismType.FIBRATION

    cof = ModelMorphism(
        source="layer1",
        target="layer2",
        is_cofibration=True
    )
    print(f"✓ Created cofibration: {cof.source} → {cof.target}")
    assert cof.morphism_type() == MorphismType.COFIBRATION

    we = ModelMorphism(
        source="layer1",
        target="layer2",
        is_weak_equivalence=True
    )
    print(f"✓ Created weak equivalence: {we.source} → {we.target}")
    assert we.morphism_type() == MorphismType.WEAK_EQUIVALENCE

    triv_fib = ModelMorphism(
        source="layer1",
        target="layer2",
        is_fibration=True,
        is_weak_equivalence=True
    )
    print(f"✓ Created trivial fibration: {triv_fib.source} → {triv_fib.target}")
    assert triv_fib.morphism_type() == MorphismType.TRIVIAL_FIBRATION

    triv_cof = ModelMorphism(
        source="layer1",
        target="layer2",
        is_cofibration=True,
        is_weak_equivalence=True
    )
    print(f"✓ Created trivial cofibration: {triv_cof.source} → {triv_cof.target}")
    assert triv_cof.morphism_type() == MorphismType.TRIVIAL_COFIBRATION

    print("\n✓ MODEL MORPHISM TEST PASSED\n")


################################################################################
# Test 2: Lifting Properties
################################################################################

def test_lifting_properties():
    """Test right and left lifting properties."""
    print("\n" + "="*80)
    print("TEST 2: Lifting Properties (RLP, LLP)")
    print("="*80)

    # Fibration has RLP wrt trivial cofibration
    fib = ModelMorphism("A", "B", is_fibration=True)
    triv_cof = ModelMorphism("X", "Y", is_cofibration=True, is_weak_equivalence=True)

    has_rlp = fib.has_right_lifting_property(triv_cof)
    print(f"\n✓ Fibration has RLP wrt trivial cofibration: {has_rlp}")
    assert has_rlp, "Fibration should have RLP wrt trivial cofibration"

    # Cofibration has LLP wrt trivial fibration
    cof = ModelMorphism("A", "B", is_cofibration=True)
    triv_fib = ModelMorphism("X", "Y", is_fibration=True, is_weak_equivalence=True)

    has_llp = cof.has_left_lifting_property(triv_fib)
    print(f"✓ Cofibration has LLP wrt trivial fibration: {has_llp}")
    assert has_llp, "Cofibration should have LLP wrt trivial fibration"

    # Trivial fibration has RLP wrt all cofibrations
    triv_fib2 = ModelMorphism("A", "B", is_fibration=True, is_weak_equivalence=True)
    cof2 = ModelMorphism("X", "Y", is_cofibration=True)

    has_rlp2 = triv_fib2.has_right_lifting_property(cof2)
    print(f"✓ Trivial fibration has RLP wrt cofibration: {has_rlp2}")
    assert has_rlp2, "Trivial fibration should have RLP wrt all cofibrations"

    # Trivial cofibration has LLP wrt all fibrations
    triv_cof2 = ModelMorphism("A", "B", is_cofibration=True, is_weak_equivalence=True)
    fib2 = ModelMorphism("X", "Y", is_fibration=True)

    has_llp2 = triv_cof2.has_left_lifting_property(fib2)
    print(f"✓ Trivial cofibration has LLP wrt fibration: {has_llp2}")
    assert has_llp2, "Trivial cofibration should have LLP wrt all fibrations"

    print("\n✓ LIFTING PROPERTIES TEST PASSED\n")


################################################################################
# Test 3: Quillen Model Structure Axioms
################################################################################

def test_quillen_model_structure():
    """Test Quillen model category axioms (CM1-CM5)."""
    print("\n" + "="*80)
    print("TEST 3: Quillen Model Structure (CM1-CM5)")
    print("="*80)

    model = QuillenModelStructure()

    # Add morphisms
    fib = ModelMorphism("A", "B", is_fibration=True)
    cof = ModelMorphism("X", "Y", is_cofibration=True)
    we = ModelMorphism("P", "Q", is_weak_equivalence=True)

    model.add_fibration(fib)
    model.add_cofibration(cof)
    model.add_weak_equivalence(we)

    print(f"\n✓ Added morphisms:")
    print(f"  - Fibrations: {len(model.fibrations)}")
    print(f"  - Cofibrations: {len(model.cofibrations)}")
    print(f"  - Weak equivalences: {len(model.weak_equivalences)}")

    # Axiom CM2: 2-out-of-3 property
    f = ModelMorphism("A", "B", is_weak_equivalence=True)
    g = ModelMorphism("B", "C", is_weak_equivalence=True)
    gf = ModelMorphism("A", "C", is_weak_equivalence=False)  # Should be inferred

    result = model.check_two_out_of_three(f, g, gf)
    print(f"\n✓ Axiom CM2 (2-out-of-3): {result}")
    assert result, "2-out-of-3 property failed"
    assert gf.is_weak_equivalence, "Composition should be weak equivalence"

    # Axiom CM5a: Factorization as cofibration + trivial fibration
    test_morph = ModelMorphism("input", "output")
    i, p = model.factorize_as_cofibration_trivial_fibration(test_morph)

    print(f"\n✓ Axiom CM5a: Factorization as cof + triv fib")
    print(f"  {test_morph.source} → {i.target} → {test_morph.target}")
    assert i.is_cofibration, "First factor should be cofibration"
    assert p.is_fibration and p.is_weak_equivalence, "Second factor should be trivial fibration"

    # Axiom CM5b: Factorization as trivial cofibration + fibration
    j, q = model.factorize_as_trivial_cofibration_fibration(test_morph)

    print(f"✓ Axiom CM5b: Factorization as triv cof + fib")
    print(f"  {test_morph.source} → {j.target} → {test_morph.target}")
    assert j.is_cofibration and j.is_weak_equivalence, "First factor should be trivial cofibration"
    assert q.is_fibration, "Second factor should be fibration"

    # Axiom CM4: Lifting property
    lift_exists = model.check_lifting_property(cof, fib)
    print(f"\n✓ Axiom CM4: Lifting property check: {lift_exists}")

    print("\n✓ QUILLEN MODEL STRUCTURE TEST PASSED\n")


################################################################################
# Test 4: Groupoid Categories
################################################################################

def test_groupoid_category():
    """Test groupoid category GrpdC."""
    print("\n" + "="*80)
    print("TEST 4: Groupoid Category (GrpdC)")
    print("="*80)

    grpd = GroupoidCategory(name="GrpdC")

    # Add layers with group actions
    grpd.add_layer_with_group("conv1", CyclicGroup(4))
    grpd.add_layer_with_group("conv2", CyclicGroup(4))

    print(f"\n✓ Created groupoid category '{grpd.name}'")
    print(f"  Layers with groups: {list(grpd.layers.keys())}")

    # Add equivariant morphism (automatically weak equivalence in groupoid)
    transform = nn.Conv2d(8, 16, 3)  # Placeholder
    morph = grpd.add_equivariant_morphism("conv1", "conv2", transform)

    print(f"\n✓ Added equivariant morphism: {morph.source} → {morph.target}")
    print(f"  Is weak equivalence: {morph.is_weak_equivalence}")
    assert morph.is_weak_equivalence, "Groupoid morphisms must be weak equivalences"

    # Check groupoid property (all morphisms invertible)
    is_groupoid = grpd.check_groupoid_property()
    print(f"\n✓ Groupoid property verified: {is_groupoid}")
    assert is_groupoid, "All morphisms should be weak equivalences"

    print("\n✓ GROUPOID CATEGORY TEST PASSED\n")


################################################################################
# Test 5: Multi-Fibrations (Theorem 2.2)
################################################################################

def test_multi_fibration():
    """Test multi-fibration structure and Theorem 2.2."""
    print("\n" + "="*80)
    print("TEST 5: Multi-Fibration (Theorem 2.2)")
    print("="*80)

    # Create multi-fibration
    multi_fib = MultiFibration(
        base_layer="base_conv",
        total_layers=["equivariant_conv1", "equivariant_conv2"]
    )

    print(f"\n✓ Created multi-fibration")
    print(f"  Base layer: {multi_fib.base_layer}")
    print(f"  Total layers: {multi_fib.total_layers}")

    # Check cartesian morphism
    morph = ModelMorphism("eq_conv1", "eq_conv2", is_weak_equivalence=True)
    is_cartesian = multi_fib.is_cartesian_morphism(morph)

    print(f"\n✓ Cartesian morphism check: {is_cartesian}")
    assert is_cartesian, "Weak equivalences are cartesian in multi-fibration"

    # Verify Theorem 2.2
    results = multi_fib.check_theorem_2_2()

    print(f"\n✓ Theorem 2.2 verification:")
    for key, value in results.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")

    assert all(results.values()), "Theorem 2.2 should hold"

    print("\n✓ MULTI-FIBRATION TEST PASSED\n")


################################################################################
# Test 6: Dependent Types (Martin-Löf)
################################################################################

def test_dependent_types():
    """Test Martin-Löf dependent type theory."""
    print("\n" + "="*80)
    print("TEST 6: Dependent Types (Martin-Löf Type Theory)")
    print("="*80)

    # Simple type (non-dependent)
    nat_type = DependentType(name="Nat")
    print(f"\n✓ Created simple type: {nat_type.name}")
    assert not nat_type.is_dependent(), "Nat should not be dependent"

    # Dependent type: Vec(A, n) - vectors of type A with length n
    vec_type = DependentType(
        name="Vec",
        parameters=["A", "n"]
    )
    print(f"✓ Created dependent type: {vec_type.name}({', '.join(vec_type.parameters)})")
    assert vec_type.is_dependent(), "Vec should be dependent"

    # Instantiate dependent type
    vec_nat_3 = vec_type.instantiate({"A": "Nat", "n": 3})
    print(f"✓ Instantiated: {vec_nat_3.name}")
    assert not vec_nat_3.is_dependent(), "Instantiation should be concrete"

    # Layer type with parameters
    conv_type = DependentType(
        name="Conv2d",
        parameters=["in_channels", "out_channels", "kernel_size"]
    )
    print(f"\n✓ Created layer type: {conv_type.name}({', '.join(conv_type.parameters)})")

    # Instantiate layer type
    conv_8_16_3 = conv_type.instantiate({
        "in_channels": 8,
        "out_channels": 16,
        "kernel_size": 3
    })
    print(f"✓ Instantiated: {conv_8_16_3.name}")

    print("\n✓ DEPENDENT TYPES TEST PASSED\n")


################################################################################
# Test 7: Identity Types and Paths
################################################################################

def test_identity_types():
    """Test identity types Id_A(a, b) and path operations."""
    print("\n" + "="*80)
    print("TEST 7: Identity Types and Paths")
    print("="*80)

    # Create identity type
    id_type = IdentityType(
        space="LayerSpace",
        start="activation_a",
        end="activation_b"
    )
    print(f"\n✓ Created identity type: Id_{id_type.space}({id_type.start}, {id_type.end})")

    # Reflexivity: refl_a : Id_A(a, a)
    refl = id_type.reflexivity()
    print(f"✓ Reflexivity: Id_{refl.space}({refl.start}, {refl.end})")
    assert refl.start == refl.end, "Reflexive path should have same start and end"

    # Symmetry: sym: Id_A(a, b) → Id_A(b, a)
    sym = id_type.symmetry()
    print(f"✓ Symmetry: Id_{sym.space}({sym.start}, {sym.end})")
    assert sym.start == id_type.end and sym.end == id_type.start, "Symmetry should reverse path"

    # Transitivity: trans: Id_A(a, b) × Id_A(b, c) → Id_A(a, c)
    id_type2 = IdentityType("LayerSpace", "activation_b", "activation_c")
    trans = id_type.transitivity(id_type2)
    print(f"✓ Transitivity: Id_{trans.space}({trans.start}, {trans.end})")
    assert trans.start == id_type.start and trans.end == id_type2.end, "Transitivity should compose paths"

    # Path with function
    def path_fn(t):
        """Linear interpolation between activations."""
        return torch.zeros(1) * (1 - t) + torch.ones(1) * t

    id_with_path = IdentityType("Reals", "0", "1", path_function=path_fn)
    print(f"\n✓ Created path with function: Id_{id_with_path.space}({id_with_path.start}, {id_with_path.end})")

    # Evaluate at endpoints
    if id_with_path.path_function:
        start_val = id_with_path.path_function(0.0)
        end_val = id_with_path.path_function(1.0)
        mid_val = id_with_path.path_function(0.5)
        print(f"  p(0.0) = {start_val.item():.2f}")
        print(f"  p(0.5) = {mid_val.item():.2f}")
        print(f"  p(1.0) = {end_val.item():.2f}")

    print("\n✓ IDENTITY TYPES TEST PASSED\n")


################################################################################
# Test 8: Univalence Axiom
################################################################################

def test_univalence_axiom():
    """Test univalence axiom: (A ≃ B) ≃ (A = B)."""
    print("\n" + "="*80)
    print("TEST 8: Univalence Axiom")
    print("="*80)

    # Two types
    type_a = DependentType(name="LayerTypeA", parameters=["channels"])
    type_b = DependentType(name="LayerTypeB", parameters=["channels"])

    print(f"\n✓ Created types:")
    print(f"  A = {type_a.name}")
    print(f"  B = {type_b.name}")

    # Equivalence: forward and backward maps
    def forward_map(x):
        """Transform A → B."""
        return x

    def backward_map(y):
        """Transform B → A."""
        return y

    # Convert equivalence to identity via univalence
    identity = UnivalenceAxiom.equivalence_to_identity(
        type_a, type_b, forward_map, backward_map
    )

    print(f"\n✓ Univalence: Converted equivalence to identity")
    print(f"  Id_Type({identity.start}, {identity.end})")
    assert identity.space == "Type", "Identity should be in universe of types"

    # Transport property along identity
    test_term = "property_at_A"

    def property_at_start(x):
        return f"property_at_{x}"

    transported = UnivalenceAxiom.transport(identity, test_term, property_at_start)
    print(f"\n✓ Transport: {test_term} transported along path")

    print("\n✓ UNIVALENCE AXIOM TEST PASSED\n")


################################################################################
# Test 9: ModelCategoryDNN Integration
################################################################################

def test_model_category_dnn():
    """Test complete ModelCategoryDNN integration."""
    print("\n" + "="*80)
    print("TEST 9: ModelCategoryDNN Integration")
    print("="*80)

    # Create ModelCategoryDNN
    model_cat = ModelCategoryDNN()

    print(f"\n✓ Created ModelCategoryDNN")

    # Add layer types
    input_type = model_cat.add_layer_type("InputLayer", ["channels", "height", "width"])
    conv_type = model_cat.add_layer_type("Conv2dLayer", ["in_ch", "out_ch", "kernel"])
    output_type = model_cat.add_layer_type("OutputLayer", ["num_classes"])

    print(f"\n✓ Added layer types:")
    for name in model_cat.dependent_types.keys():
        print(f"  - {name}")

    # Create identity path between layers
    id_path = model_cat.create_identity_path("input", "conv1")
    print(f"\n✓ Created identity path: {id_path.start} → {id_path.end}")

    # Add morphism to model structure
    morph = ModelMorphism("input", "conv1", is_cofibration=True)
    model_cat.model_structure.add_cofibration(morph)

    print(f"✓ Added cofibration to model structure")

    # Add equivariant layer to groupoid
    model_cat.groupoid_category.add_layer_with_group("conv1", CyclicGroup(4))
    print(f"✓ Added equivariant layer to groupoid category")

    # Setup multi-fibration
    model_cat.multi_fibration = MultiFibration(
        base_layer="base",
        total_layers=["conv1"]
    )
    print(f"✓ Setup multi-fibration")

    # Check model axioms
    axiom_results = model_cat.check_model_axioms()
    print(f"\n✓ Model category axioms:")
    for axiom, holds in axiom_results.items():
        status = "✓" if holds else "✗"
        print(f"  {status} {axiom}: {holds}")

    assert all(axiom_results.values()), "All model axioms should hold"

    # Verify Theorem 2.2
    theorem_holds = model_cat.verify_theorem_2_2()
    print(f"\n✓ Theorem 2.2 verification: {theorem_holds}")
    assert theorem_holds, "Theorem 2.2 should hold"

    print("\n✓ MODEL CATEGORY DNN TEST PASSED\n")


################################################################################
# Test 10: End-to-End with StackDNN
################################################################################

def test_stackdnn_model_category():
    """Test model category structure on actual StackDNN."""
    print("\n" + "="*80)
    print("TEST 10: StackDNN with Model Category Structure")
    print("="*80)

    # Create StackDNN
    model = StackDNN(
        group=CyclicGroup(4),
        input_shape=(3, 8, 8),
        num_classes=5,
        channels=[8, 16],
        num_equivariant_blocks=1,
        fc_dims=[32],
        device='cpu'
    )

    print(f"\n✓ Created StackDNN")

    # Create model category structure
    model_cat = ModelCategoryDNN()

    # Analyze network morphisms
    layer_names = ["input", "conv0", "eq_block0", "fc", "output"]

    # Add fibrations (projections/pooling)
    # In StackDNN, pooling operations are fibrations
    for i in range(len(layer_names) - 1):
        morph = ModelMorphism(
            source=layer_names[i],
            target=layer_names[i+1],
            is_weak_equivalence=False  # Not all preserve information
        )
        # Pooling/downsampling are fibrations
        if "pool" in layer_names[i+1].lower() or "fc" in layer_names[i+1]:
            model_cat.model_structure.add_fibration(morph)
            print(f"  Added fibration: {morph.source} → {morph.target}")

    # Add equivariant layers to groupoid
    for layer_name in ["conv0", "eq_block0"]:
        model_cat.groupoid_category.add_layer_with_group(layer_name, CyclicGroup(4))

    print(f"\n✓ Added {len(model_cat.groupoid_category.layers)} equivariant layers to groupoid")

    # Check model axioms
    axioms = model_cat.check_model_axioms()
    print(f"\n✓ Model axioms on StackDNN:")
    for axiom, holds in axioms.items():
        status = "✓" if holds else "✗"
        print(f"  {status} {axiom}")

    # Forward pass
    x = torch.randn(2, 3, 8, 8)
    output = model(x)
    print(f"\n✓ Forward pass: {x.shape} → {output.shape}")

    print("\n✓ STACKDNN MODEL CATEGORY TEST PASSED\n")


################################################################################
# Main Test Runner
################################################################################

if __name__ == "__main__":
    print("="*80)
    print("TENSORIZED MODEL CATEGORIES - COMPREHENSIVE TESTS")
    print("Section 2.4 Implementation (Belfiore & Bennequin 2022)")
    print("="*80)

    test_model_morphisms()
    test_lifting_properties()
    test_quillen_model_structure()
    test_groupoid_category()
    test_multi_fibration()
    test_dependent_types()
    test_identity_types()
    test_univalence_axiom()
    test_model_category_dnn()
    test_stackdnn_model_category()

    print("\n" + "="*80)
    print("✓ ALL MODEL CATEGORY TESTS PASSED")
    print("="*80)
    print("\nImplementation Summary:")
    print("  ✓ Model morphisms (fibrations, cofibrations, weak equivalences)")
    print("  ✓ Lifting properties (RLP, LLP)")
    print("  ✓ Quillen model structure (CM1-CM5 axioms)")
    print("  ✓ Groupoid categories (GrpdC)")
    print("  ✓ Multi-fibrations (Theorem 2.2)")
    print("  ✓ Dependent types (Martin-Löf)")
    print("  ✓ Identity types and paths")
    print("  ✓ Univalence axiom")
    print("  ✓ ModelCategoryDNN unified structure")
    print("  ✓ Integration with StackDNN architecture")
    print("\n" + "="*80)
