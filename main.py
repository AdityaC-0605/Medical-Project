#!/usr/bin/env python3
"""
Medical AI System using RAG + LangGraph + MedGemma

Tests MedGemma on:
1. TEXT: Lipid Profile Analysis (Cardiology)
2. IMAGE + TEXT: CT Coronary Angiography (Radiology)

Architecture:
- RAG: Retrieves medical knowledge
- LangGraph: 3-node workflow (retrieve → diagnose → prescribe)
- MedGemma: google/medgemma-1.5-4b-it for diagnosis generation
"""

import os
import sys

# Setup environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.graph import MedicalGraph
from app.state import WorkflowStatus


def print_results(state):
    """Print formatted results."""
    print("\n" + "="*80)
    print(f"RESULTS: {state.task_type.upper().replace('_', ' ')}")
    print("="*80)
    
    print(f"\n📋 Query:")
    print(f"   {state.query}")
    
    print(f"\n📚 RAG Context:")
    print(f"   Retrieved {len(state.retrieved_context)} characters of medical knowledge")
    
    print("\n" + "-"*80)
    print("🔬 AI DIAGNOSIS (MedGemma):")
    print("-"*80)
    print(state.diagnosis)
    
    if state.prescription:
        print("\n" + "-"*80)
        print("💊 PRESCRIPTION & RECOMMENDATIONS:")
        print("-"*80)
        
        rx = state.prescription
        print(f"\n   Diagnosis: {rx.get('diagnosis_summary', 'N/A')}")
        
        if rx.get('medications'):
            print(f"\n   Medications:")
            for med in rx['medications']:
                print(f"      • {med['name']} {med['dosage']} - {med['frequency']}")
                print(f"        {med['instructions']}")
        
        if rx.get('lifestyle_modifications'):
            print(f"\n   Lifestyle Changes:")
            for item in rx['lifestyle_modifications']:
                print(f"      • {item}")
        
        print(f"\n   Follow-up: {rx.get('follow_up', 'N/A')}")
        
        if rx.get('referral'):
            print(f"   Referral: {rx.get('referral')}")
    
    if state.end_time and state.start_time:
        duration = state.end_time - state.start_time
        print(f"\n⏱️  Total time: {duration:.1f} seconds")
    
    print("\n" + "="*80)
    print("⚠️  DISCLAIMER: Educational use only. Consult healthcare professionals.")
    print("="*80 + "\n")


def main():
    """Run medical AI workflow."""
    print("\n" + "="*80)
    print("RAG + LANGGRAPH + MEDGEMMA")
    print("Medical AI System")
    print("="*80)
    print("\nArchitecture:")
    print("  1. RAG Retrieval → Medical Knowledge Base")
    print("  2. LangGraph Workflow → 3 nodes")
    print("     • retrieve: Get relevant medical guidelines")
    print("     • diagnose: MedGemma generates diagnosis")
    print("     • prescribe: Create treatment plan")
    print("  3. MedGemma → google/medgemma-1.5-4b-it")
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
    print(">>> TASK 2/2: CT Coronary Angiography")
    image_path = "data/images/sample_ct.png"
    state2 = graph.run("ct_coronary", {
        "stenosis_percent": 70,
        "vessel": "LAD",
        "finding": "Calcified plaque with 70% stenosis in proximal LAD",
        "image_path": image_path if os.path.exists(image_path) else None
    })
    print_results(state2)
    
    print("\n" + "="*80)
    print("✅ ALL TASKS COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
