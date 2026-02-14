#!/usr/bin/env python3
"""
Medical AI System - Main Entry Point
User-Driven Application Mode

This is the main entry point for the user-driven medical diagnosis system.
It replaces the old demo mode with a real application that processes
user inputs one at a time.

Usage:
    # Run API server (recommended for production)
    python main.py --mode api
    
    # Run CLI tool (interactive)
    python main.py --mode cli
    
    # Run tests
    python main.py --mode test
    
    # Run old demo (all 4 test cases)
    python main.py --mode demo
"""

import os
import sys
import logging

# Setup environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_streamlit():
    """Run the Streamlit web app."""
    print("\n" + "="*70)
    print("🏥 MEDICAL AI STREAMLIT APP")
    print("="*70)
    print("Starting Streamlit web interface...")
    print("="*70 + "\n")
    
    import subprocess
    import sys
    
    # Use port 8501 (Streamlit default) instead of 5000
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8501"]
    subprocess.run(cmd)

def run_api_server(host='0.0.0.0', port=8080, debug=False):
    """Run the Flask API server on port 8080 to avoid AirPlay conflict."""
    print("\n" + "="*70)
    print("🏥 MEDICAL AI API SERVER")
    print("="*70)
    print(f"Mode: User-Driven Single Request")
    print(f"Port: {port} (avoiding AirPlay conflict)")
    print(f"Memory: Lazy loading with cleanup after each request")
    print("="*70 + "\n")
    
    # Import and run server
    import api_server
    api_server.app.run(host=host, port=port, debug=debug)


def run_cli():
    """Run interactive CLI tool."""
    import cli_tool
    cli_tool.main()


def run_tests():
    """Run user-driven workflow tests."""
    import test_user_driven
    test_user_driven.main()

def run_multimodal_test():
    """Run multimodal image classification tests."""
    print("\n" + "="*70)
    print("🧪 MULTIMODAL CLASSIFICATION TEST")
    print("="*70)
    import test_multimodal_classification
    test_multimodal_classification.main()


def run_demo():
    """Run old demo mode (all 4 test cases)."""
    print("\n" + "="*70)
    print("⚠️  DEMO MODE - Running all 4 test cases")
    print("="*70 + "\n")
    
    import demo_supervisor
    demo_supervisor.demo_supervisor_classification()
    demo_supervisor.demo_manual_task_specification()


def print_help():
    """Print help information."""
    help_text = """
╔══════════════════════════════════════════════════════════════════════╗
║                    🏥 MEDICAL AI SYSTEM                               ║
║                    User-Driven Application                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  MODES:                                                               ║
║                                                                       ║
║  streamlit - Web interface (RECOMMENDED - easiest to use)             ║
║              Usage: python main.py --mode streamlit                   ║
║              Opens at: http://localhost:8501                          ║
║                                                                       ║
║  api       - Flask API server (HTTP requests)                         ║
║              Usage: python main.py --mode api [--port 8080]           ║
║                                                                       ║
║  cli       - Interactive command-line tool                            ║
║              Usage: python main.py --mode cli                         ║
║                                                                       ║
║  test      - Run user-driven workflow tests                           ║
║              Usage: python main.py --mode test                        ║
║                                                                       ║
║  demo      - Run old demo (all 4 test cases)                          ║
║              Usage: python main.py --mode demo                        ║
║                                                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  QUICK START:                                                         ║
║                                                                       ║
║  1. Start Streamlit (easiest):                                        ║
║     python main.py --mode streamlit                                   ║
║                                                                       ║
║  2. Open browser to:                                                  ║
║     http://localhost:8501                                             ║
║                                                                       ║
║  3. Enter medical text or upload images                               ║
║                                                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  ARCHITECTURE:                                                        ║
║                                                                       ║
║  User Input → Preprocessor → Supervisor → [One Task Only] → Response ║
║                                                                       ║
║  ✓ Automatic classification into 4 categories                         ║
║  ✓ Only the selected node executes                                    ║
║  ✓ Lazy loading: Model loads per request                              ║
║  ✓ Cleanup: Memory freed after each request                           ║
║  ✓ Single response: One diagnosis per request                         ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(help_text)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Medical AI System - User-Driven Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Streamlit web app (RECOMMENDED)
  python main.py --mode streamlit
  
  # Run Flask API server
  python main.py --mode api --port 8080
  
  # Run interactive CLI
  python main.py --mode cli
  
  # Run tests
  python main.py --mode test
  
  # Test multimodal image classification
  python main.py --mode multimodal
  
  # Run old demo
  python main.py --mode demo
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['streamlit', 'api', 'cli', 'test', 'multimodal', 'demo', 'help'],
        default='help',
        help='Operation mode (default: help)'
    )
    
    # API server options
    parser.add_argument('--host', default='0.0.0.0', help='API server host')
    parser.add_argument('--port', type=int, default=8080, help='API server port (default: 8080 to avoid AirPlay)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.mode == 'help':
        print_help()
    elif args.mode == 'streamlit':
        run_streamlit()
    elif args.mode == 'api':
        run_api_server(args.host, args.port, args.debug)
    elif args.mode == 'cli':
        run_cli()
    elif args.mode == 'test':
        run_tests()
    elif args.mode == 'multimodal':
        run_multimodal_test()
    elif args.mode == 'demo':
        run_demo()


if __name__ == '__main__':
    main()