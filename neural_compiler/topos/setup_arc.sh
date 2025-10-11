#!/bin/bash

# Setup script for ARC-AGI 2 Solver
# This script downloads the dataset and verifies the installation

set -e  # Exit on error

echo "========================================================================"
echo "ARC-AGI 2 Solver Setup"
echo "========================================================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "arc_loader.py" ]; then
    echo -e "${RED}Error: Must run from neural_compiler/topos/ directory${NC}"
    exit 1
fi

echo -e "${BLUE}Step 1: Checking Python dependencies...${NC}"
python3 -c "import jax, flax, optax, matplotlib, numpy, tqdm" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All dependencies installed${NC}"
else
    echo -e "${RED}✗ Missing dependencies${NC}"
    echo "Installing dependencies..."
    pip install jax jaxlib flax optax matplotlib numpy tqdm
fi
echo ""

echo -e "${BLUE}Step 2: Downloading ARC dataset...${NC}"
cd ../..  # Go to homotopy-nn root

if [ -d "ARC-AGI" ]; then
    echo -e "${GREEN}✓ ARC dataset already downloaded${NC}"
else
    echo "Cloning ARC-AGI repository..."
    git clone https://github.com/fchollet/ARC-AGI.git
    echo -e "${GREEN}✓ Downloaded ARC dataset${NC}"
fi
echo ""

# Count tasks
TRAINING_TASKS=$(ls ARC-AGI/data/training/*.json 2>/dev/null | wc -l)
EVALUATION_TASKS=$(ls ARC-AGI/data/evaluation/*.json 2>/dev/null | wc -l)

echo "Dataset statistics:"
echo "  Training tasks: $TRAINING_TASKS"
echo "  Evaluation tasks: $EVALUATION_TASKS"
echo ""

cd neural_compiler/topos

echo -e "${BLUE}Step 3: Testing loader...${NC}"
python3 arc_loader.py --dataset_dir ../../ARC-AGI/data --limit 3 --stats > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Loader working correctly${NC}"
else
    echo -e "${RED}✗ Loader test failed${NC}"
    exit 1
fi
echo ""

echo -e "${BLUE}Step 4: Running quick test...${NC}"
echo "Training on 2 tasks with minimal evolution (this will take ~2-3 minutes)..."
python3 train_arc.py --limit 2 --generations 5 --population 5 --no_visualizations

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Test training completed successfully${NC}"
else
    echo ""
    echo -e "${RED}✗ Test training failed${NC}"
    exit 1
fi
echo ""

echo "========================================================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Quick experiment (5-10 minutes):"
echo "   python train_arc.py --limit 5 --generations 20"
echo ""
echo "2. Medium experiment (1-2 hours):"
echo "   python train_arc.py --limit 50 --generations 50 --population 30"
echo ""
echo "3. Full training (8-12 hours):"
echo "   python train_arc.py --split training --generations 50 --population 30"
echo ""
echo "4. Evaluation set (12-16 hours):"
echo "   python train_arc.py --split evaluation --generations 100 --population 50"
echo ""
echo "See README.md for more options and examples."
echo ""
echo "Target: Beat GPT-4's 23% and approach human 85% on ARC-AGI 2!"
echo "========================================================================"
