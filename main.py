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

# Setup environment for Mac optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback for unsupported ops
os.environ["MALLOC_ARENA_MAX"] = "2"  # Limit memory arenas

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
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port=8501",
        "--logger.level=info",
    ]
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nStreamlit server stopped by user.")

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

def run_eval(dataset_path: str, output_path: str = "", generate_answers: bool = False,
             dry_run: bool = False, max_cases: int = 0, generated_output: str = "",
             generation_timeout_sec: int = 600, generate_only: bool = False,
             score_only: bool = False, eval_max_new_tokens: int = 128,
             force_regenerate: bool = False):
    """Run RAGAS evaluation workflow."""
    import subprocess
    import sys

    effective_generate = generate_answers or generate_only
    print("\n" + "="*70)
    print("RAGAS EVALUATION")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Generate answers: {effective_generate}")
    if output_path:
        print(f"Output: {output_path}")
    print("="*70 + "\n")

    cmd = [sys.executable, "evaluation/run_ragas.py", "--dataset", dataset_path]
    if output_path:
        cmd.extend(["--output", output_path])
    if generate_answers:
        cmd.append("--generate-answers")
    if generate_only:
        cmd.append("--generate-only")
    if score_only:
        cmd.append("--score-only")
    if force_regenerate:
        cmd.append("--force-regenerate")
    if dry_run:
        cmd.append("--dry-run")
    if max_cases > 0:
        cmd.extend(["--max-cases", str(max_cases)])
    if generated_output:
        cmd.extend(["--generated-output", generated_output])
    if generation_timeout_sec >= 0:
        cmd.extend(["--generation-timeout-sec", str(generation_timeout_sec)])
    if eval_max_new_tokens > 0:
        cmd.extend(["--eval-max-new-tokens", str(eval_max_new_tokens)])

    subprocess.run(cmd, check=False)


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
║  eval      - Run RAGAS evaluation metrics                              ║
║              Usage: python main.py --mode eval --generate-answers      ║
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
  
  # Run RAGAS eval
  python main.py --mode eval --generate-answers
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['streamlit', 'api', 'cli', 'test', 'multimodal', 'demo', 'eval', 'help'],
        default='help',
        help='Operation mode (default: help)'
    )
    
    # API server options
    parser.add_argument('--host', default='0.0.0.0', help='API server host')
    parser.add_argument('--port', type=int, default=8080, help='API server port (default: 8080 to avoid AirPlay)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--eval-dataset', default='evaluation/eval_cases.jsonl', help='Path to eval JSONL dataset')
    parser.add_argument('--eval-output', default='', help='Path to save eval report JSON')
    parser.add_argument('--generate-answers', action='store_true', help='Generate answers with MedicalGraph before scoring')
    parser.add_argument('--dry-run', action='store_true', help='Validate eval dataset without running RAGAS')
    parser.add_argument('--max-cases', type=int, default=0, help='Limit eval to first N cases (0 = all)')
    parser.add_argument('--generated-output', default='', help='Path to save generated answers JSONL before scoring')
    parser.add_argument('--generation-timeout-sec', type=int, default=600, help='Per-case timeout for eval generation (seconds)')
    parser.add_argument('--generate-only', action='store_true', help='Generate eval answers only (no RAGAS scoring)')
    parser.add_argument('--score-only', action='store_true', help='Run RAGAS scoring only using existing answers in dataset')
    parser.add_argument('--force-regenerate', action='store_true', help='Force regeneration even if answers already exist')
    parser.add_argument('--eval-max-new-tokens', type=int, default=128, help='Max new tokens for eval generation')
    
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
    elif args.mode == 'eval':
        run_eval(
            dataset_path=args.eval_dataset,
            output_path=args.eval_output,
            generate_answers=args.generate_answers,
            dry_run=args.dry_run,
            max_cases=args.max_cases,
            generated_output=args.generated_output,
            generation_timeout_sec=args.generation_timeout_sec,
            generate_only=args.generate_only,
            score_only=args.score_only,
            eval_max_new_tokens=args.eval_max_new_tokens,
            force_regenerate=args.force_regenerate,
        )


if __name__ == '__main__':
    main()
