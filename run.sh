#!/bin/bash
# RAG + LangGraph + MedGemma Medical AI System

echo "========================================"
echo "RAG + LangGraph + MedGemma"
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

echo "✓ Environment ready"
echo ""
echo "Architecture:"
echo "  1. RAG Retrieval → Medical Knowledge Base"
echo "  2. LangGraph Workflow → 3 nodes (retrieve → diagnose → prescribe)"
echo "  3. MedGemma → AI Diagnosis Generation"
echo ""
echo "Running 2 tasks:"
echo "  1. Lipid Profile (Text analysis)"
echo "  2. CT Coronary (Image + Text analysis)"
echo ""
echo "Estimated time: 6-10 minutes"
echo ""
sleep 3

python3 main.py

echo ""
echo "========================================"
echo "Workflow completed!"
echo "========================================"
