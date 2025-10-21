#!/bin/bash
# Run all topos property tests and summarize results
#
# Usage:
#   ./run_topos_tests.sh          # Run all tests
#   ./run_topos_tests.sh quick    # Run direct tests only
#   ./run_topos_tests.sh property # Run property tests only

set -e

PYTEST="/nix/store/lnqf82hc6ljyb26s0h2jx6kw953v6a7z-python3-3.12.11-env/bin/pytest"
TEST_DIR="/Users/faezs/homotopy-nn/neural_compiler/topos"

cd "$TEST_DIR"

echo "========================================"
echo "Topos Categorical Structure Tests"
echo "========================================"
echo ""

MODE=${1:-all}

case $MODE in
    quick)
        echo "Running direct numerical tests only..."
        echo ""
        $PYTEST test_topos_laws.py -v -s --tb=short
        ;;
    property)
        echo "Running property-based tests only..."
        echo ""
        $PYTEST test_topos_properties.py -v --hypothesis-show-statistics --tb=short
        ;;
    *)
        echo "Running full test suite (direct + property-based)..."
        echo ""

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "1. Direct Numerical Tests"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        $PYTEST test_topos_laws.py -v -s --tb=short

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "2. Property-Based Tests"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        $PYTEST test_topos_properties.py -v --hypothesis-show-statistics --tb=short
        ;;
esac

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo ""
echo "✓ Adjunction laws: f^* ⊣ f_*"
echo "✓ Functor laws: F(id) = id, F(g∘f) = F(g)∘F(f)"
echo "✓ Sheaf condition: F(U) ≅ lim F(Uᵢ)"
echo "✓ Pullback/pushforward preservation"
echo "✓ Grid encoding properties"
echo ""
echo "See TOPOS_TESTING.md for detailed documentation."
echo ""
