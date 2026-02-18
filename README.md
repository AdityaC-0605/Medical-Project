# 🏥 Medical AI System — User-Driven Diagnosis with MedGemma

A **user-driven medical AI system** that accepts real patient inputs (images and text), automatically classifies medical cases using **MedGemma-powered image classification**, and generates structured AI-powered diagnoses and treatment recommendations via a **LangGraph** workflow.

## ✨ Key Features

- **🖼️ Multimodal Analysis**: Accepts medical images + clinical text for combined analysis
- **🧠 AI-Powered Diagnosis**: Uses Google's **MedGemma-1.5-4b-it** model for both classification and diagnosis
- **📊 MedGemma-Only Classification**: Image classification powered entirely by MedGemma — no keyword fallbacks
- **� Structured Assessments**: Produces organized clinical summaries, diagnoses, treatment plans, and follow-up recommendations
- **🖥️ Multiple Interfaces**: Web UI (Streamlit), REST API (Flask), and Interactive CLI
- **🍎 Apple Silicon Optimized**: MPS acceleration with stability fixes (greedy decoding, KV-cache, memory management)
- **⚡ Performance Optimized**: Reduced token generation, greedy decoding, and KV-cache for faster inference
- **🔒 Privacy-Focused**: 100% local processing — no data sent to external servers

## 🏗️ Architecture

```
User Input (Image / Text / Both)
    ↓
┌─────────────────────────────────┐
│  INPUT PREPROCESSOR             │
│  - Validation & sanitization    │
│  - Structured data extraction   │
│  - Medical field detection      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  SUPERVISOR NODE                │
│  - MedGemma image classifier    │
│  - Auto-routes to specialist    │
│  - Confidence-based decisions   │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  SPECIALIZED NODE               │
│  - CT Coronary                  │
│  - Lipid Profile                │
│  - Breast Imaging               │
│  - Biopsy Report                │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  DIAGNOSE NODE (MedGemma)       │
│  - Structured clinical analysis │
│  - 4-section assessment output  │
│  - Intelligent text parsing     │
└─────────────────────────────────┘
    ↓
Structured Response to User
  (Clinical Summary, Diagnosis,
   Treatment Plan, Follow-Up)
```

## 📋 Supported Medical Tasks

### 1. 🫀 CT Coronary Angiography
- **Inputs**: Cardiac CT images + clinical data
- **Analysis**: Coronary stenosis, plaque characterization, vessel assessment
- **Output**: Cardiac risk assessment + treatment recommendations

### 2. 🩸 Lipid Profile Analysis
- **Inputs**: Cholesterol panel (LDL, HDL, Triglycerides) + patient history
- **Analysis**: Cardiovascular risk stratification, metabolic assessment
- **Output**: Medication recommendations + lifestyle modifications

### 3. 🎀 Breast Imaging
- **Inputs**: Mammograms, ultrasounds + clinical context
- **Analysis**: Mass characterization, BI-RADS assessment, tissue evaluation
- **Output**: Imaging interpretation + follow-up recommendations

### 4. 🔬 Biopsy Report Analysis
- **Inputs**: Pathology reports + histology data
- **Analysis**: Tumor grading, staging, immunohistochemistry
- **Output**: Treatment planning + multidisciplinary recommendations

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.11+ required
python3 --version

# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token (required for MedGemma access)
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

### Option 1: Streamlit Web Interface (Recommended)
```bash
# Start the web app
python main.py --mode streamlit

# Opens automatically at http://localhost:8501
```

**Features:**
- 🖱️ Drag-and-drop image upload
- 📝 Text input for clinical context
- 📊 Real-time structured results display
- 🎨 Medical-themed UI with progress indicators

### Option 2: Flask API Server
```bash
# Start API server
python main.py --mode api

# Server runs on http://localhost:8080
```

**Test with curl:**
```bash
curl -X POST http://localhost:8080/api/diagnose/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "58-year-old male with LDL 145, HDL 38, TG 220. History of diabetes.",
    "metadata": {"age": 58, "sex": "Male"}
  }'
```

### Option 3: Interactive CLI
```bash
# Run interactive command-line tool
python main.py --mode cli
```

### Additional Modes
```bash
# Run demo mode (all 4 test cases)
python main.py --mode demo

# Run multimodal image classification tests
python main.py --mode multimodal-test

# Show help
python main.py --help
```

## 📖 Usage Examples

### Example 1: CT Coronary Analysis
```
# Upload cardiac CT image via Streamlit
# Add text: "65-year-old male with chest pain on exertion, diabetes x 10 years"

Expected Output:
🩺 AI Diagnosis & Analysis
✅ Medical assessment generated successfully

CLINICAL SUMMARY:
CT coronary angiography reveals significant calcified plaque in the
proximal LAD with approximately 70% luminal narrowing. The left
circumflex shows mild disease; right coronary artery is patent.

PRIMARY DIAGNOSIS:
Significant LAD stenosis, hemodynamically significant, likely
explaining exertional chest pain.

TREATMENT PLAN:
Cardiology consultation for functional assessment, optimize medical
therapy (antiplatelet + statin), consider stress testing.

FOLLOW-UP:
Follow-up cardiac imaging in 6 months with risk factor modification.
```

### Example 2: Image-Only Analysis
```
# Upload breast mammogram via Streamlit
# Leave text blank (optional — MedGemma analyzes image directly)

Expected Output:
CLINICAL SUMMARY:
Mammogram demonstrates scattered fibroglandular densities with an
irregular mass in the left upper outer quadrant, measuring ~1.5 cm.

PRIMARY DIAGNOSIS:
Suspicious mass — BI-RADS Category 4.

TREATMENT PLAN:
Ultrasound-guided core needle biopsy, bilateral mammographic
correlation, surgical oncology referral.

FOLLOW-UP:
Results review within 1 week; further management based on pathology.
```

## 🏗️ Project Structure

```
Medical-Project/
├── main.py                         # Entry point (streamlit/api/cli/demo modes)
├── app.py                          # Streamlit web interface
├── api_server.py                   # Flask REST API server
├── cli_tool.py                     # Interactive CLI tool
├── config.yaml                     # Configuration (model, workflow, safety, etc.)
├── requirements.txt                # Python dependencies
├── app/
│   ├── __init__.py
│   ├── graph.py                    # LangGraph workflow + Supervisor routing
│   ├── state.py                    # MedicalState + WorkflowStatus management
│   ├── input_preprocessor.py       # Input validation, text extraction, sanitization
│   └── core/
│       ├── __init__.py
│       ├── medgemma_client.py      # MedGemma model loading, inference, structured prompts
│       └── image_classifier.py     # MedGemma-powered medical image classification
├── data/
│   ├── knowledge_base/             # Clinical guidelines (cardiology, pathology, etc.)
│   ├── images/                     # Sample medical images for testing
│   └── processed/                  # Processed data outputs
├── logs/                           # Application logs
└── uploads/                        # Temporary image storage (auto-cleaned)
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Model settings
model:
  model_name: "google/medgemma-1.5-4b-it"
  device: "auto"                    # auto | cuda | mps | cpu
  max_new_tokens: 400               # Optimized for speed + quality
  generation:
    do_sample: false                 # Greedy decoding (stable on MPS)
    use_cache: true                  # KV-cache for faster generation

# Workflow settings
workflow:
  timeout: 420                       # 7-minute timeout
  preload_model: true                # Preload at startup

# Supervisor settings
supervisor:
  use_image_analysis: true           # MedGemma image classification
  use_llm_classification: true       # MedGemma text classification
  image_classification:
    min_confidence: 0.5              # 50% confidence threshold
    do_sample: false                 # Deterministic classification

# API settings
api:
  port: 8080                         # Avoids AirPlay conflict on macOS
  memory:
    cleanup_after_request: true      # Clear MPS cache after each request
    max_concurrent_requests: 1       # Sequential for memory efficiency
```

## 🖥️ System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **OS** | macOS 12+ / Linux / Windows | macOS 14+ (Apple Silicon) |
| **Python** | 3.11+ | 3.11+ |
| **RAM** | 8 GB | 16 GB+ |
| **Storage** | ~10 GB (model cache) | ~10 GB |
| **GPU** | CPU fallback available | Apple Silicon (MPS) or CUDA |

## 🔒 Privacy & Security

- ✅ **Local Processing**: All AI inference happens on your machine
- ✅ **No Data Upload**: Images and text never leave your device
- ✅ **Temporary Storage**: Uploaded files are auto-cleaned after processing
- ✅ **No External APIs**: Only connects to HuggingFace once for initial model download

## 🐛 Troubleshooting

### Issue: "No module named 'streamlit'"
```bash
pip install -r requirements.txt
```

### Issue: "Port 8080 already in use"
```bash
python main.py --mode api --port 8081
```

### Issue: "MedGemma model failed to load"
```bash
# Set HuggingFace token
export HUGGING_FACE_HUB_TOKEN=your_token_here

# Or create .env file
echo "HUGGING_FACE_HUB_TOKEN=your_token" > .env
```

### Issue: "Out of memory on Mac"
```bash
# The system uses ~8GB RAM when the model is loaded.
# Close other applications to free memory.
# Memory is automatically freed after each request (cleanup_after_request: true).
```

### Issue: "Segmentation fault on model load"
```bash
# Fixed in recent updates. Ensure these environment variables are set:
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

## 📝 Changelog

### February 17, 2026 — Performance Optimization
- Reduced `max_new_tokens` from 512 → 400 (20% faster generation)
- Optimized diagnose node to use 128 tokens for structured sections
- Improved prompt engineering for better quality with fewer tokens
- Overall target: **5–8 minutes** per assessment (down from 10+ minutes)

### February 13, 2026 — Diagnosis Consistency Fix
- Fixed issue where the system generated identical diagnoses for different medical images
- Improved input-specific prompt construction for distinct, accurate outputs

### February 10–12, 2026 — MPS Stability & Cleanup
- Fixed MedGemma loading on Apple Silicon (MPS) — resolved segfaults
- Implemented CPU-first model loading, then move to MPS
- Switched to `float32` for MPS stability, `float16` for CUDA
- Greedy decoding (`do_sample: false`) for deterministic, stable MPS output
- Removed unused files and legacy `prescription_generator.py`

### February 9, 2026 — Architecture Overhaul
- Added **MedGemma-powered image classifier** (`image_classifier.py`)
- Replaced keyword-based classification with AI-driven classification
- Integrated structured assessment output (Clinical Summary, Diagnosis, Treatment, Follow-Up)
- Added intelligent text splitting as fallback when section parsing fails
- Implemented model preloading for better first-request performance

### February 7, 2026 — MedGemma Integration
- Integrated MedGemma-1.5-4b-it for both text and multimodal tasks
- Fixed text-only and multimodal generation pipelines
- Added image token handling for Gemma3 processor compatibility

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system is for **educational and research purposes only**.

- 🚫 **Not for clinical diagnosis** without physician oversight
- 🚫 **Not a substitute for professional medical advice**
- ✅ **Always consult qualified healthcare professionals**
- ✅ **Verify all AI-generated recommendations**

## 🤝 Contributing

This is an educational project demonstrating AI in healthcare. Contributions welcome for:
- Additional medical task types
- Improved prompts and clinical queries
- UI/UX enhancements
- Documentation improvements

## 📚 Documentation

- `WORKFLOW_ARCHITECTURE.md` - Detailed explanation of the system's workflow and routing logic.

## 🙏 Acknowledgments

- **Google** for the MedGemma model
- **LangChain** for the LangGraph workflow framework
- **HuggingFace** for model hosting and the Transformers library

---

**Built with ❤️ for advancing medical AI education**

*Last Updated: February 17, 2026*