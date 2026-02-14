# 🏥 Medical AI System - User-Driven Diagnosis with MedGemma

A **user-driven medical AI system** that accepts real patient inputs (images and text), automatically classifies medical cases, and generates AI-powered diagnoses and treatment recommendations using **MedGemma** and **LangGraph**.

## ✨ Key Features

- **🖼️ Multimodal Analysis**: Accepts medical images + clinical text
- **🧠 AI-Powered Diagnosis**: Uses Google's MedGemma-1.5-4b-it model
- **📊 Automatic Classification**: Intelligently routes to appropriate specialists
- **💊 Treatment Recommendations**: Generates personalized prescriptions
- **🖥️ Multiple Interfaces**: Web UI (Streamlit), API, and CLI
- **🍎 macOS Optimized**: Memory-efficient with Apple Silicon support
- **🔒 Privacy-Focused**: Local processing, no data sent to external servers

## 🏗️ Architecture

```
User Input (Image/Text)
    ↓
┌─────────────────────────────┐
│  INPUT PREPROCESSOR         │
│  - Validation & Extraction  │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  SUPERVISOR NODE            │
│  - Auto-classification      │
│  - Routes to specialist     │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  SPECIALIZED NODE           │
│  - CT Coronary              │
│  - Lipid Profile            │
│  - Breast Imaging           │
│  - Biopsy Report            │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  DIAGNOSE (MedGemma)        │
│  - AI analysis              │
│  - Clinical assessment      │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  PRESCRIBE                  │
│  - Treatment plan           │
│  - Recommendations          │
└─────────────────────────────┘
    ↓
Response to User
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

# macOS: Install dependencies
pip install -r requirements.txt

# Set HuggingFace token (for MedGemma access)
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
- 📊 Real-time results display
- 🎨 Medical-themed UI

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

## 📖 Usage Examples

### Example 1: CT Coronary Analysis
```python
# Upload cardiac CT image
# Add text: "65-year-old male with chest pain on exertion, diabetes x 10 years"

Expected Output:
🩺 AI Diagnosis & Analysis
✅ Medical assessment generated successfully

IMAGING FINDINGS:
The CT coronary angiography reveals significant calcified plaque 
in the proximal left anterior descending artery with approximately 
70% luminal narrowing. The left circumflex shows mild disease. 
Right coronary artery is patent.

CLINICAL INTERPRETATION:
Given the degree of LAD stenosis and the patient's symptoms of 
exertional chest pain, these findings are hemodynamically significant 
and likely explain the clinical presentation.

RECOMMENDATIONS:
1. Cardiology consultation for functional assessment
2. Consider stress testing
3. Optimize medical therapy (antiplatelet + statin)
4. Risk factor modification
```

### Example 2: Image-Only Analysis
```python
# Upload breast mammogram
# Leave text blank (optional)

Expected Output:
The mammogram demonstrates bilateral breast tissue with scattered 
fibroglandular densities. In the left breast upper outer quadrant, 
there is an irregular mass with spiculated margins measuring 
approximately 1.5 cm. Associated microcalcifications are noted.

IMPRESSION:
Suspicious mass in left breast requiring further evaluation.
BI-RADS Category: 4

RECOMMENDATIONS:
1. Ultrasound-guided core needle biopsy
2. Bilateral mammographic correlation
3. Surgical oncology referral
```

## 🏗️ Project Structure

```
Medical-Project/
├── app.py                          # Streamlit web interface
├── api_server.py                   # Flask API server
├── cli_tool.py                     # Interactive CLI
├── main.py                         # Entry point with modes
├── app/
│   ├── graph.py                    # MedicalGraph with Supervisor
│   ├── state.py                    # State management
│   ├── input_preprocessor.py       # Input validation
│   └── core/
│       ├── medgemma_client.py      # MedGemma integration
│       ├── image_classifier.py     # Multimodal classification
│       └── prescription_generator.py
├── config.yaml                     # Configuration settings
├── requirements.txt                # Python dependencies
├── quickstart.sh                   # Quick setup script
└── uploads/                        # Temporary image storage
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Model settings
model:
  model_name: "google/medgemma-1.5-4b-it"
  max_new_tokens: 1024
  
# API settings
api:
  port: 8080  # Avoids AirPlay conflict on macOS
  
# Supervisor settings
supervisor:
  use_image_analysis: true
  min_confidence: 0.5
```

## 🖥️ System Requirements

- **OS**: macOS 12+ (Apple Silicon optimized) / Linux / Windows
- **Python**: 3.11+
- **RAM**: 16GB+ recommended (8GB minimum)
- **Storage**: ~10GB for MedGemma model cache
- **GPU**: Apple Silicon (MPS) or CUDA (optional, CPU fallback available)

## 🔒 Privacy & Security

- ✅ **Local Processing**: All AI inference happens locally
- ✅ **No Data Upload**: Images and text never leave your machine
- ✅ **Temporary Storage**: Uploaded files cleaned up after processing
- ✅ **No External APIs**: Only connects to HuggingFace for model download

## 🐛 Troubleshooting

### Issue: "No module named 'streamlit'"
```bash
pip install -r requirements.txt
```

### Issue: "Port 8080 already in use"
```bash
# Use different port
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
# Close other applications
# The system uses ~8GB RAM when model is loaded
# Memory is freed after each request
```

## 📚 Documentation

- `USER_DRIVEN_GUIDE.md` - Complete user guide
- `ARCHITECTURE_CHANGES.md` - Technical architecture details
- `MULTIMODAL_UPGRADE.md` - Image analysis features
- `PERFORMANCE_OPTIMIZATION.md` - Memory optimization guide
- `NEW_WORKFLOW_ARCHITECTURE.md` - New Image Analysis Node design

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system is for **educational and research purposes only**.

- 🚫 **Not for clinical diagnosis** without physician oversight
- 🚫 **Not a substitute for professional medical advice**
- ✅ **Always consult qualified healthcare professionals**
- ✅ **Verify all AI-generated recommendations**

## 🤝 Contributing

This is an educational project demonstrating AI in healthcare. Contributions welcome for:
- Additional medical task types
- Improved prompts and queries
- UI/UX enhancements
- Documentation improvements

## 📄 License

Educational Use License - See LICENSE file for details.

## 🙏 Acknowledgments

- **Google** for MedGemma model
- **LangChain** for LangGraph workflow framework
- **HuggingFace** for model hosting and transformers library

---

**Built with ❤️ for advancing medical AI education**

*Last Updated: February 2026*