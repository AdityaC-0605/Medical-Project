"""
CLI Tool for Medical AI - Interactive User-Driven Diagnosis
Command-line interface for submitting medical cases
"""

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.graph import MedicalGraph
from app.input_preprocessor import InputPreprocessor, preprocess_user_input
from app.state import WorkflowStatus

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*70)
    print("🏥 MEDICAL AI DIAGNOSTIC SYSTEM")
    print("="*70)
    print("User-Driven Workflow - Single Case Analysis")
    print("="*70 + "\n")


def print_result(state):
    """Print diagnosis result."""
    print("\n" + "="*70)
    print("📋 DIAGNOSIS RESULT")
    print("="*70)
    print(f"\nClassification: {state.task_type.upper()}")
    print(f"Status: {state.status}")
    
    if state.status == WorkflowStatus.COMPLETED:
        print("\n" + "-"*70)
        print("🩺 DIAGNOSIS:")
        print("-"*70)
        print(state.diagnosis)
        
        if state.prescription:
            print("\n" + "-"*70)
            print("💊 RECOMMENDATIONS:")
            print("-"*70)
            
            if 'medications' in state.prescription:
                print("\nMedications:")
                for med in state.prescription['medications']:
                    print(f"  • {med}")
            
            if 'lifestyle_changes' in state.prescription:
                print("\nLifestyle Changes:")
                for change in state.prescription['lifestyle_changes']:
                    print(f"  • {change}")
            
            if 'follow_up' in state.prescription:
                print(f"\nFollow-up: {state.prescription['follow_up']}")
    else:
        print(f"\n❌ Error: {state.error}")
    
    print("\n" + "="*70)


def run_text_case(text: str, metadata: dict = None):
    """Run a text-based case."""
    print(f"\n📝 Processing text input...")
    print(f"   Length: {len(text)} characters")
    
    # Preprocess
    input_data = preprocess_user_input(text=text, metadata=metadata)
    
    print(f"   Preprocessing complete")
    
    # Run workflow
    graph = MedicalGraph(preload_model=False)
    state = graph.run(input_data=input_data, cleanup_after=True)
    
    print_result(state)
    return state


def run_image_case(image_path: str, text: str = None):
    """Run an image-based case."""
    print(f"\n🖼️  Processing image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return None
    
    # Read image
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    print(f"   Size: {len(image_data) / 1024:.1f} KB")
    
    # Preprocess
    input_data = preprocess_user_input(
        image_data=image_data,
        image_filename=os.path.basename(image_path),
        metadata={'accompanying_text': text} if text else None
    )
    
    print(f"   Preprocessing complete")
    
    # Run workflow
    graph = MedicalGraph(preload_model=False)
    state = graph.run(input_data=input_data, cleanup_after=True)
    
    print_result(state)
    return state


def interactive_mode():
    """Run interactive CLI mode."""
    print_banner()
    
    print("Select input type:")
    print("  1. Text (medical report, symptoms, lab results)")
    print("  2. Image (CT scan, mammogram, ultrasound)")
    print("  3. Text + Image")
    print("  4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '4':
        print("\nGoodbye! 👋\n")
        return
    
    if choice == '1':
        print("\n" + "-"*70)
        print("Enter medical text (press Enter twice when done):")
        print("-"*70)
        
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == '' and lines and lines[-1].strip() == '':
                    break
                lines.append(line)
            except EOFError:
                break
        
        text = '\n'.join(lines).strip()
        
        if text:
            run_text_case(text)
        else:
            print("❌ No text provided")
    
    elif choice == '2':
        image_path = input("\nEnter image file path: ").strip()
        if image_path:
            run_image_case(image_path)
        else:
            print("❌ No image path provided")
    
    elif choice == '3':
        image_path = input("\nEnter image file path: ").strip()
        print("\nEnter accompanying text (press Enter twice when done):")
        print("-"*70)
        
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == '' and lines and lines[-1].strip() == '':
                    break
                lines.append(line)
            except EOFError:
                break
        
        text = '\n'.join(lines).strip()
        
        if image_path:
            run_image_case(image_path, text if text else None)
        else:
            print("❌ No image path provided")
    
    else:
        print("❌ Invalid choice")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Medical AI - User-Driven Diagnosis CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python cli_tool.py
  
  # Text input
  python cli_tool.py --text "Patient has LDL of 150 and HDL of 35..."
  
  # File input
  python cli_tool.py --file report.txt
  
  # Image input
  python cli_tool.py --image data/images/sample_ct.png
  
  # Image with text
  python cli_tool.py --image mammogram.png --text "50-year-old female with family history"
        """
    )
    
    parser.add_argument('--text', '-t', type=str, help='Text input for diagnosis')
    parser.add_argument('--file', '-f', type=str, help='File containing text input')
    parser.add_argument('--image', '-i', type=str, help='Image file path')
    parser.add_argument('--metadata', '-m', type=str, help='JSON metadata string')
    parser.add_argument('--interactive', '-I', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Check for interactive mode
    if args.interactive or (not args.text and not args.file and not args.image):
        interactive_mode()
        return
    
    print_banner()
    
    # Parse metadata if provided
    metadata = {}
    if args.metadata:
        import json
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("❌ Invalid metadata JSON")
            return
    
    # Handle file input
    text = args.text
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return
        with open(args.file, 'r') as f:
            text = f.read()
    
    # Run appropriate mode
    if args.image:
        run_image_case(args.image, text)
    elif text:
        run_text_case(text, metadata)
    else:
        print("❌ No input provided. Use --text, --file, or --image")
        parser.print_help()


if __name__ == '__main__':
    main()