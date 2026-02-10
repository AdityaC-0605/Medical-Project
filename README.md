# Medical AI System - LangGraph + MedGemma

A medical AI system using **LangGraph workflow** and **MedGemma** for diagnosis and prescription generation.

## Architecture

```
┌─────────────┐    ┌─────────────┐
│   DIAGNOSE  │───→│  PRESCRIBE  │
│  (MedGemma) │    │(Prescription│
│             │    │  Generator) │
└─────────────┘    └─────────────┘
```

**Components:**
- **LangGraph**: 2-node workflow orchestration (diagnose → prescribe)
- **MedGemma**: `google/medgemma-1.5-4b-it` for AI diagnosis
- **Prescription Generator**: Creates treatment plans based on diagnosis

## Project Structure

```
Medical-Project/
├── app/
│   ├── __init__.py
│   ├── state.py                    # State management
│   ├── graph.py                    # LangGraph workflow
│   └── core/
│       ├── __init__.py
│       ├── medgemma_client.py      # MedGemma integration
│       └── prescription_generator.py
├── data/
│   ├── images/                     # Medical images
│   └── knowledge_base/             # Medical guidelines
├── main.py                         # Entry point
├── run.sh                          # Run script
├── requirements.txt                # Dependencies
└── README.md
```

## Quick Start

```bash
# Activate virtual environment
source venv_py311/bin/activate

# Run the system
./run.sh
```

## Workflow

The system runs 2 medical tasks:

### 1. Lipid Profile Analysis (Text)
- **Input**: LDL, HDL, Triglycerides, Total Cholesterol, Age, Sex
- **MedGemma**: Generates detailed diagnosis
- **Output**: Medication recommendations + lifestyle changes

### 2. CT Coronary Angiography (Image + Text)
- **Input**: Medical image + stenosis data
- **MedGemma**: Multimodal diagnosis
- **Output**: Cardiac treatment plan

## Requirements

- Python 3.11
- 16GB+ RAM recommended
- Cached MedGemma model (~8.6GB)

## Runtime

- **Total time**: 6-10 minutes for both tasks
- **Per task**: 3-5 minutes (model loading once, then cached)

## Medical Disclaimer

This system is for educational purposes only. All AI-generated diagnoses and prescriptions must be reviewed by qualified healthcare professionals.
