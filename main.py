#!/usr/bin/env python3
"""
Medical AI System using LangGraph + MedGemma

Tests MedGemma on:
1. TEXT: Lipid Profile Analysis (Cardiology)
2. IMAGE + TEXT: CT Coronary Angiography (Radiology)

Architecture:
- LangGraph: 2-node workflow (diagnose → prescribe)
- MedGemma: google/medgemma-1.5-4b-it for diagnosis generation
"""

import os
import sys
import logging

# Setup environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Let MPS use all memory

# Configure logging to go to stderr only (not stdout)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.graph import MedicalGraph
from app.state import WorkflowStatus


def print_results(state):
    """Print formatted results."""
    # Handle both dict and MedicalState objects
    if isinstance(state, dict):
        task_type = state.get('task_type', 'unknown')
        query = state.get('query', '')
        diagnosis = state.get('diagnosis', '')
        prescription = state.get('prescription')
        start_time = state.get('start_time')
        end_time = state.get('end_time')
    else:
        task_type = state.task_type
        query = state.query
        diagnosis = state.diagnosis
        prescription = state.prescription
        start_time = state.start_time
        end_time = state.end_time
    
    output = []
    output.append("\n" + "="*80)
    output.append(f"RESULTS: {task_type.upper().replace('_', ' ')}")
    output.append("="*80)
    
    output.append(f"\n📋 Query:")
    output.append(f"   {query}")

    output.append("\n" + "-"*80)
    output.append("🔬 AI DIAGNOSIS (MedGemma):")
    output.append("-"*80)
    output.append(diagnosis if diagnosis else "No diagnosis generated")
    
    if prescription:
        output.append("\n" + "-"*80)
        output.append("💊 PRESCRIPTION & RECOMMENDATIONS:")
        output.append("-"*80)
        
        rx = prescription
        output.append(f"\n   Diagnosis: {rx.get('diagnosis_summary', 'N/A')}")
        
        if rx.get('medications'):
            output.append(f"\n   Medications:")
            for med in rx['medications']:
                output.append(f"      • {med['name']} {med['dosage']} - {med['frequency']}")
                output.append(f"        {med['instructions']}")
        
        if rx.get('lifestyle_modifications'):
            output.append(f"\n   Lifestyle Changes:")
            for item in rx['lifestyle_modifications']:
                output.append(f"      • {item}")
        
        output.append(f"\n   Follow-up: {rx.get('follow_up', 'N/A')}")
        
        if rx.get('referral'):
            output.append(f"   Referral: {rx.get('referral')}")
    
    if end_time and start_time:
        duration = end_time - start_time
        output.append(f"\n⏱️  Total time: {duration:.1f} seconds")
    
    output.append("\n" + "="*80)
    output.append("⚠️  DISCLAIMER: Educational use only. Consult healthcare professionals.")
    output.append("="*80 + "\n")
    
    # Print all at once to avoid interleaving
    print("\n".join(output), flush=True)


def main():
    """Run medical AI workflow."""
    print("\n" + "="*80)
    print("LANGGRAPH + MEDGEMMA")
    print("Medical AI System")
    print("="*80)
    print("\nArchitecture:")
    print("  1. LangGraph Workflow → 2 nodes")
    print("     • diagnose: MedGemma generates diagnosis")
    print("     • prescribe: Create treatment plan")
    print("  2. MedGemma → google/medgemma-1.5-4b-it")
    print("\nRunning 2 tasks:")
    print("  1. Lipid Profile (Text-only analysis)")
    print("  2. CT Coronary Angiography (Image + Text)")
    print(f"\n{'='*80}\n")
    
    # Initialize graph
    graph = MedicalGraph()
    
    # Task 1: Lipid Profile (Text)
    print(">>> TASK 1/2: Lipid Profile Analysis")
    state1 = graph.run("lipid_profile", {
        "ldl": 145,
        "hdl": 38,
        "triglycerides": 220,
        "total_cholesterol": 235,
        "age": 58,
        "sex": "Male"
    })
    print_results(state1)
    
    # Task 2: CT Coronary (Image + Text)
    print(">>> TASK 2/2: CT Coronary Angiography", flush=True)
    image_path = "data/images/sample_ct.png"
    
    try:
        state2 = graph.run("ct_coronary", {
            "stenosis_percent": 70,
            "vessel": "LAD",
            "finding": "Calcified plaque with 70% stenosis in proximal LAD",
            "image_path": image_path if os.path.exists(image_path) else None
        })
        print_results(state2)
    except Exception as e:
        print(f"\n❌ TASK 2 FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80, flush=True)
    print("✅ ALL TASKS COMPLETED", flush=True)
    print("="*80 + "\n", flush=True)


if __name__ == "__main__":
    main()
