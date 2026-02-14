#!/bin/bash
# Simple script to run Streamlit web app
# Usage: ./run_streamlit.sh

echo "🚀 Starting Medical AI Streamlit App..."
echo ""
echo "🌐 The app will open at: http://localhost:8501"
echo "📱 You can also view it on your phone using your computer's IP address"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Check if in virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Activating virtual environment..."
    source venv_py311/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null
fi

# Run Streamlit
python -m streamlit run app.py --server.port=8501