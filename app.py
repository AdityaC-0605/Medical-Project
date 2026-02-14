"""
Medical AI Streamlit App - User-Driven Workflow
Interactive web interface for medical diagnosis
"""

import streamlit as st
import os
import sys
import logging
from datetime import datetime

# Setup environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.graph import MedicalGraph
from app.input_preprocessor import preprocess_user_input
from app.state import WorkflowStatus

# Page configuration
st.set_page_config(
    page_title="Medical AI Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .diagnosis-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    .prescription-box {
        background-color: #f0fff0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2ecc71;
        margin: 20px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_medical_graph():
    """Initialize MedicalGraph with lazy loading."""
    return MedicalGraph(preload_model=False)


def process_diagnosis(text_input, image_file, metadata):
    """Process diagnosis request."""
    try:
        if image_file is not None:
            image_data = image_file.getvalue()
            image_filename = image_file.name
            input_data = preprocess_user_input(
                text=text_input if text_input else None,
                image_data=image_data,
                image_filename=image_filename,
                metadata=metadata
            )
        else:
            input_data = preprocess_user_input(
                text=text_input,
                metadata=metadata
            )
        
        graph = get_medical_graph()
        
        with st.spinner('🧠 Analyzing with Medical AI...'):
            state = graph.run(input_data=input_data, cleanup_after=True)
        
        return state
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def display_results(state):
    """Display structured assessment results."""
    if not state:
        return
    
    st.markdown("---")
    st.markdown("### 📋 Results")
    
    # Metrics
    cols = st.columns(3)
    with cols[0]:
        st.metric("Classification", state.task_type.upper().replace("_", " "))
    with cols[1]:
        status = "✅" if state.status == WorkflowStatus.COMPLETED else "❌"
        st.metric("Status", f"{status} {state.status.value}")
    with cols[2]:
        if state.end_time and hasattr(state, 'start_time'):
            duration = round(state.end_time - state.start_time, 2)
            st.metric("Time", f"{duration}s")
    
    # Structured Assessment
    if state.structured_assessment:
        st.markdown("---")
        st.markdown("### 🩺 AI Clinical Assessment")
        st.success("✅ Medical assessment generated successfully")
        
        assessment = state.structured_assessment
        
        # Clinical Summary
        if assessment.get('clinical_summary'):
            with st.expander("📋 Clinical Summary", expanded=True):
                st.markdown(assessment['clinical_summary'])
        
        # Primary Diagnosis
        if assessment.get('primary_diagnosis'):
            with st.expander("🔍 Primary Diagnosis", expanded=True):
                st.markdown(assessment['primary_diagnosis'])
        
        # Differential Diagnoses
        if assessment.get('differentials'):
            with st.expander("📊 Differential Diagnoses"):
                st.markdown(assessment['differentials'])
        
        # Treatment Plan
        if assessment.get('treatment_plan'):
            with st.expander("💊 Treatment Plan", expanded=True):
                st.markdown(assessment['treatment_plan'])
        
        # Lifestyle Recommendations
        if assessment.get('lifestyle_recommendations'):
            with st.expander("🌟 Lifestyle Recommendations"):
                st.markdown(assessment['lifestyle_recommendations'])
        
        # Follow-up Plan
        if assessment.get('follow_up'):
            with st.expander("📅 Follow-up Plan", expanded=True):
                st.markdown(assessment['follow_up'])
    
    # Disclaimer
    st.warning("⚠️ **Disclaimer:** This is for educational purposes only. Consult healthcare professionals for actual medical advice.")


def main():
    """Main Streamlit app."""
    # Header
    st.markdown("<h1 class='main-header'>🏥 Medical AI Diagnosis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>User-Driven Medical Analysis with Automatic Classification</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📖 About")
        st.info("""
        This AI system analyzes medical inputs and classifies them into:
        - CT Coronary Angiography
        - Lipid Profile Analysis
        - Breast Imaging/Mammogram
        - Biopsy Report Analysis
        
        **Memory Efficient:** Model loads per request and cleans up automatically.
        """)
        
        st.markdown("### 📊 System Status")
        graph = get_medical_graph()
        if graph._model_loaded:
            st.success("✅ Model Loaded")
        else:
            st.info("💤 Model Ready (Lazy Loading)")
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Be specific with medical terms
        - Include lab values if available
        - Upload clear medical images
        - Provide patient age/sex for better analysis
        """)
    
    # Main input area
    st.markdown("### 📝 Enter Medical Information")
    
    # Input tabs
    tab1, tab2 = st.tabs(["📝 Text Input", "🖼️ Image + Text"])
    
    with tab1:
        text_input = st.text_area(
            "Medical Report / Symptoms / Lab Results",
            height=200,
            placeholder="Example: 58-year-old male with LDL 145 mg/dL, HDL 38 mg/dL. History of diabetes and hypertension...",
            key="text_input"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (optional)", min_value=0, max_value=120, value=0, key="age_text")
        with col2:
            sex = st.selectbox("Sex (optional)", ["", "Male", "Female"], key="sex_text")
        
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            if text_input.strip():
                metadata = {}
                if age > 0:
                    metadata['age'] = age
                if sex:
                    metadata['sex'] = sex
                
                state = process_diagnosis(text_input, None, metadata)
                if state:
                    display_results(state)
            else:
                st.warning("⚠️ Please enter medical text to analyze.")
    
    with tab2:
        image_file = st.file_uploader(
            "Upload Medical Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload CT scans, mammograms, ultrasounds, or other medical images"
        )
        
        if image_file:
            st.image(image_file, caption="Uploaded Image", width='stretch')
        
        st.markdown("##### 📝 Clinical Information (Optional)")
        st.info("💡 **Tip**: Adding patient details (age, symptoms, history) helps the AI provide more accurate analysis, but the system can analyze images alone.")
        
        image_text = st.text_area(
            "Patient Information (Optional)",
            height=120,
            placeholder="Example (optional):\n65-year-old male with chest pain\nHistory: Diabetes, hypertension\n\nOr leave blank for image-only analysis",
            key="image_text"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            age_img = st.number_input("Age (optional)", min_value=0, max_value=120, value=0, key="age_img")
        with col2:
            sex_img = st.selectbox("Sex (optional)", ["", "Male", "Female"], key="sex_img")
        
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            if image_file is not None:
                metadata = {}
                if age_img > 0:
                    metadata['age'] = age_img
                if sex_img:
                    metadata['sex'] = sex_img
                
                state = process_diagnosis(image_text if image_text else None, image_file, metadata)
                if state:
                    display_results(state)
            else:
                st.warning("⚠️ Please upload an image to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>Medical AI System v1.0 | User-Driven Workflow</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()