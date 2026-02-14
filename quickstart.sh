#!/bin/bash
# Quick Start Script for Medical AI User-Driven System
# This script helps you get started with the new architecture

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                🏥 Medical AI System - Quick Start                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv_py311" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv_py311
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv_py311/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p uploads logs
echo "✓ Directories created"

# Check for HuggingFace token
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo ""
    echo "⚠️  Warning: HUGGING_FACE_HUB_TOKEN not set"
    echo "   MedGemma model may require authentication."
    echo "   Set it with: export HUGGING_FACE_HUB_TOKEN=your_token"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                     ✅ Setup Complete!                                ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║                                                                       ║"
echo "║  Choose how to run the system:                                        ║"
echo "║                                                                       ║"
echo "║  1. Streamlit Web App (RECOMMENDED - easiest)                         ║"
echo "║     python main.py --mode streamlit                                   ║"
echo "║     Opens at: http://localhost:8501                                   ║"
echo "║                                                                       ║"
echo "║  2. Flask API Server:                                                 ║"
echo "║     python main.py --mode api                                         ║"
echo "║     Runs on: http://localhost:8080 (avoids AirPlay conflict)          ║"
echo "║                                                                       ║"
echo "║  3. Interactive CLI:                                                  ║"
echo "║     python main.py --mode cli                                         ║"
echo "║                                                                       ║"
echo "║  4. Run Tests:                                                        ║"
echo "║     python main.py --mode test                                        ║"
echo "║                                                                       ║"
echo "║  5. View Help:                                                        ║"
echo "║     python main.py --help                                             ║"
echo "║                                                                       ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Offer to start Streamlit
read -p "🚀 Start Streamlit web app now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🌐 Starting Streamlit web app..."
    echo "   Opening browser at http://localhost:8501"
    echo "   Press Ctrl+C to stop"
    echo ""
    python main.py --mode streamlit
else
    echo ""
    echo "You can start the app later with: python main.py --mode streamlit"
    echo ""
    echo "Quick start:"
    echo "  python main.py --mode streamlit  # Web interface"
    echo "  python main.py --mode api        # API server"
    echo "  python main.py --mode cli        # Command line"
    echo ""
fi