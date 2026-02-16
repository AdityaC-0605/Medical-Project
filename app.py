"""
Medical AI Streamlit App - User-Driven Workflow
"""

import streamlit as st
import os
import sys
import logging
from datetime import datetime

# Setup environment for Mac optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# FIXED: Correct imports for your package structure
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
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_medical_graph():
    """Initialize MedicalGraph with model preloading."""
    try:
        graph = MedicalGraph(preload_model=True)
        return graph
    except Exception as e:
        logger.error(f"Failed to initialize MedicalGraph: {e}")
        st.error(f"Failed to load model: {e}")
        return None

def process_diagnosis(text_input, image_file, metadata):
    """Process diagnosis request."""
    try:
        # Prepare input data using preprocessor
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
        
        # Get graph instance
        graph = get_medical_graph()
        if graph is None:
            st.error("Model not loaded. Please check your configuration.")
            return None
        
        # Run workflow with progress updates
        progress_text = st.empty()
        progress_text.info("🔄 Step 1/3: Classifying image... (~2 minutes)")
        
        state = graph.run(input_data=input_data, cleanup_after=False)
        
        if state and state.status.value == "completed":
            progress_text.success("✅ Analysis complete!")
        elif state and state.error:
            progress_text.error(f"❌ Error: {state.error}")
        else:
            progress_text.warning("⚠️ Analysis incomplete")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in process_diagnosis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        st.error(f"❌ Error during analysis: {str(e)}")
        return None

def display_results(state):
    """Display structured assessment results."""
    if not state:
        st.error("No results generated")
        return
    
    st.markdown("---")
    st.markdown("### 📋 Analysis Results")
    
    # Metrics row
    cols = st.columns(3)
    with cols[0]:
        st.metric("Classification", state.task_type.upper().replace("_", " "))
    with cols[1]:
        status_icon = "✅" if state.status == WorkflowStatus.COMPLETED else "❌"
        st.metric("Status", f"{status_icon} {state.status.value}")
    with cols[2]:
        if state.end_time and state.start_time:
            duration = round(state.end_time - state.start_time, 2)
            st.metric("Processing Time", f"{duration}s")
    
    # Error display if failed
    if state.status != WorkflowStatus.COMPLETED and state.error:
        st.error(f"Analysis failed: {state.error}")
        return
    
    # Structured Assessment
    if state.structured_assessment:
        st.markdown("---")
        st.success("✅ Medical assessment generated successfully")
        
        assessment = state.structured_assessment
        
        # Display sections in order
        sections = [
            ("📋 Clinical Summary", "clinical_summary", True),
            ("🔍 Primary Diagnosis", "primary_diagnosis", True),
            ("📊 Differential Diagnoses", "differentials", False),
            ("💊 Treatment Plan", "treatment_plan", True),
            ("🌟 Lifestyle Recommendations", "lifestyle_recommendations", False),
            ("📅 Follow-up Plan", "follow_up", True)
        ]
        
        for title, key, expanded in sections:
            content = assessment.get(key, "")
            if content and len(content.strip()) > 5:
                with st.expander(title, expanded=expanded):
                    st.markdown(content)
            else:
                with st.expander(f"{title} (Not generated)", expanded=False):
                    st.info("This section was not generated by the model.")
    else:
        st.warning("⚠️ No structured assessment was generated. The model may have failed to produce output.")
    
    # Disclaimer
    st.markdown("---")
    st.warning("⚠️ **Medical Disclaimer:** This AI-generated assessment is for educational purposes only. It does not constitute medical advice. Always consult qualified healthcare professionals for diagnosis and treatment decisions.")

def main():
    """Main Streamlit app."""
    # Header
    st.markdown("<h1 class='main-header'>🏥 Medical AI Diagnosis System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-Powered Medical Analysis with Automatic Classification</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📖 About")
        st.info("""
        This AI system analyzes medical inputs and classifies them into:
        - **CT Coronary Angiography**
        - **Lipid Profile Analysis**  
        - **Breast Imaging/Mammogram**
        - **Biopsy Report Analysis**
        
        Powered by MedGemma 1.5 4B
        """)
        
        st.markdown("### 📊 System Status")
        graph = get_medical_graph()
        if graph and graph._model_loaded:
            st.success("✅ Model Loaded & Ready")
        elif graph:
            st.info("⏳ Model Ready (Lazy Loading)")
        else:
            st.error("❌ Model Failed to Load")
        
        st.markdown("### 💡 Usage Tips")
        st.markdown("""
        - Be specific with medical terminology
        - Include lab values when available
        - Upload clear medical images
        - Provide patient demographics (age/sex)
        - First load may take 1-2 minutes
        """)
    
    # Main input area
    st.markdown("### 📝 Enter Medical Information")
    
    # Input tabs
    tab1, tab2 = st.tabs(["📝 Text Only", "🖼️ Image + Text"])
    
    with tab1:
        text_input = st.text_area(
            "Medical Report / Symptoms / Lab Results",
            height=200,
            placeholder="Example: 58-year-old male with LDL 145 mg/dL, HDL 38 mg/dL. History of diabetes and hypertension. Patient reports chest pain on exertion...",
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
                
                with st.spinner("Processing..."):
                    state = process_diagnosis(text_input, None, metadata)
                    if state:
                        display_results(state)
            else:
                st.warning("⚠️ Please enter medical text to analyze.")
    
    with tab2:
        image_file = st.file_uploader(
            "Upload Medical Image",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload CT scans, mammograms, ultrasounds, or other medical images"
        )
        
        if image_file:
            st.image(image_file, caption="Uploaded Image Preview", use_column_width=True)
        
        st.markdown("##### 📝 Clinical Information (Optional but Recommended)")
        st.info("💡 Adding clinical context improves analysis accuracy significantly")
        
        image_text = st.text_area(
            "Patient Information",
            height=120,
            placeholder="Example: 65-year-old female, screening mammogram. Family history of breast cancer...",
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
                
                with st.spinner("Processing image... This may take up to 60 seconds on first run"):
                    state = process_diagnosis(image_text if image_text else None, image_file, metadata)
                    if state:
                        display_results(state)
            else:
                st.warning("⚠️ Please upload an image to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>Medical AI System v1.0 | Powered by MedGemma</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()