#!/bin/bash
# LangGraph + MedGemma Medical AI System

echo "========================================"
echo "LangGraph + MedGemma"
echo "Medical AI System"
echo "========================================"
echo ""

# Check virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Activate virtual environment:"
    echo "   source venv_py311/bin/activate"
    exit 1
fi

# Environment setup
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Let PyTorch use all available MPS memory

echo "✓ Environment ready"
echo ""
echo "Architecture:"
echo "  1. LangGraph Workflow → 2 nodes (diagnose → prescribe)"
echo "  2. MedGemma → AI Diagnosis Generation (MPS accelerated)"
echo ""
echo "Running 2 tasks:"
echo "  1. Lipid Profile (Text analysis)"
echo "  2. CT Coronary (Image + Text analysis)"
echo ""
echo "Estimated time: ~1 minute (MPS float16)"
echo ""
sleep 2

python3 main.py

echo ""
echo "========================================"
echo "Workflow completed!"
echo "========================================"
