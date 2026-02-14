"""
LangGraph Workflow for Medical AI with Supervisor Node
Architecture: Supervisor → Task Classification → Specialized Node → Diagnose → END
MedGemma generates structured diagnosis and treatment plan.
Optimized for macOS with memory management.
"""

import logging
from typing import Optional, Literal, Dict, Any
import gc
import os
import time

from app.state import MedicalState, WorkflowStatus
from app.core.medgemma_client import MedGemmaClient
from app.core.image_classifier import MedicalImageClassifier

logger = logging.getLogger(__name__)

# Task types supported by the system
TaskType = Literal["ct_coronary", "lipid_profile", "breast_imaging", "biopsy_report", "unknown"]


class MedicalGraph:
    """
    LangGraph workflow with Supervisor node for medical diagnosis.
    
    Workflow:
        Input → Supervisor (Classification) → [CT/Lipid/Breast/Biopsy] → Diagnose → END
    """

    def __init__(self, preload_model: bool = True):
        """
        Initialize MedicalGraph.
        
        Args:
            preload_model: Whether to preload MedGemma model immediately
        """
        self.medgemma: Optional[MedGemmaClient] = None
        self._model_loaded = False
        
        logger.info("=" * 80)
        logger.info("🏥 MEDICAL GRAPH INITIALIZATION")
        logger.info("=" * 80)
        
        # Only preload if requested (for memory optimization)
        if preload_model:
            logger.info("📦 Preload model requested: True")
            self._load_model()
        else:
            logger.info("📦 Preload model requested: False (will use lazy loading)")
        
        self.graph = self._build_graph()
        logger.info("✓ MedicalGraph initialized successfully")
        logger.info("=" * 80)
    
    def _load_model(self):
        """Load MedGemma model (lazy loading for memory optimization)."""
        if not self._model_loaded:
            logger.info("📥 LOADING MODEL")
            logger.info("  └─ Loading MedGemma model...")
            self.medgemma = MedGemmaClient()
            self._model_loaded = True
            logger.info("  ✓ MedGemma model loaded successfully")
        else:
            logger.info("📥 MODEL ALREADY LOADED")
    
    def _unload_model(self):
        """Unload MedGemma model to free memory."""
        if self._model_loaded and self.medgemma:
            logger.info("📤 UNLOADING MODEL")
            logger.info("  └─ Unloading MedGemma model to free memory...")
            self.medgemma.cleanup()
            self.medgemma = None
            self._model_loaded = False
            gc.collect()
            logger.info("  ✓ MedGemma model unloaded")
    
    def _build_graph(self):
        """Build LangGraph workflow with Supervisor and conditional routing."""
        logger.info("🔨 BUILDING WORKFLOW GRAPH")
        
        try:
            from langgraph.graph import StateGraph, END

            workflow = StateGraph(MedicalState)

            # Add nodes
            logger.info("  ├─ Adding nodes:")
            logger.info("  │  ├─ supervisor")
            logger.info("  │  ├─ ct_coronary")
            logger.info("  │  ├─ lipid_profile")
            logger.info("  │  ├─ breast_imaging")
            logger.info("  │  ├─ biopsy_report")
            logger.info("  │  └─ diagnose")
            
            workflow.add_node("supervisor", self._supervisor_node)
            workflow.add_node("ct_coronary", self._ct_coronary_node)
            workflow.add_node("lipid_profile", self._lipid_profile_node)
            workflow.add_node("breast_imaging", self._breast_imaging_node)
            workflow.add_node("biopsy_report", self._biopsy_report_node)
            workflow.add_node("diagnose", self._diagnose_node)

            # Set entry point
            workflow.set_entry_point("supervisor")
            logger.info("  ├─ Entry point: supervisor")

            # Conditional routing from supervisor
            logger.info("  ├─ Adding conditional edges from supervisor")
            workflow.add_conditional_edges(
                "supervisor",
                self._route_from_supervisor,
                {
                    "ct_coronary": "ct_coronary",
                    "lipid_profile": "lipid_profile", 
                    "breast_imaging": "breast_imaging",
                    "biopsy_report": "biopsy_report",
                    "unknown": END
                }
            )

            # All specialized nodes route to diagnose
            logger.info("  ├─ Adding edges to diagnose node")
            workflow.add_edge("ct_coronary", "diagnose")
            workflow.add_edge("lipid_profile", "diagnose")
            workflow.add_edge("breast_imaging", "diagnose")
            workflow.add_edge("biopsy_report", "diagnose")

            # Diagnose → END
            logger.info("  ├─ Adding edge: diagnose → END")
            workflow.add_edge("diagnose", END)

            logger.info("  ✓ Workflow graph built successfully")
            return workflow.compile()

        except ImportError as e:
            logger.warning(f"  ⚠ LangGraph not available ({e})")
            logger.info("  └─ Using sequential fallback workflow")
            return SequentialWorkflow(self)
    
    def _supervisor_node(self, state: MedicalState) -> MedicalState:
        """
        Supervisor node: Analyzes input and classifies medical task type.
        """
        node_start_time = time.time()
        logger.info("\n" + "=" * 80)
        logger.info("🎯 NODE: SUPERVISOR")
        logger.info("=" * 80)
        logger.info("📋 NODE FUNCTION: Analyzing input and classifying task type")
        logger.info("-" * 80)
        
        try:
            # Extract input data
            logger.info("📥 INPUT EXTRACTION")
            if isinstance(state, dict):
                input_data = state.get('input_data', {})
                existing_task_type = state.get('task_type', 'unknown')
                logger.info(f"  └─ State type: dict")
            else:
                input_data = state.input_data or {}
                existing_task_type = state.task_type
                logger.info(f"  └─ State type: MedicalState object")
            
            logger.info(f"  └─ Existing task type: {existing_task_type}")
            logger.info(f"  └─ Input data keys: {list(input_data.keys())}")
            
            # If task type is already specified, skip classification
            if existing_task_type and existing_task_type != 'unknown':
                logger.info("✓ Task type pre-specified, skipping classification")
                logger.info(f"  └─ Using task type: {existing_task_type}")
                return state
            
            # Classification logic based on input features
            logger.info("🔍 STARTING CLASSIFICATION")
            classification = self._classify_medical_task(input_data)
            
            # Update state with classification
            logger.info("📝 UPDATING STATE")
            if isinstance(state, dict):
                state['task_type'] = classification
                state['classification_confidence'] = 'high'
                logger.info(f"  └─ Updated dict state with task_type: {classification}")
            else:
                state.task_type = classification
                logger.info(f"  └─ Updated MedicalState with task_type: {classification}")
            
            node_duration = time.time() - node_start_time
            logger.info("-" * 80)
            logger.info(f"✓ SUPERVISOR NODE COMPLETED")
            logger.info(f"  └─ Classification: {classification.upper()}")
            logger.info(f"  └─ Duration: {node_duration:.2f}s")
            logger.info("=" * 80)
            return state
            
        except Exception as e:
            logger.error(f"❌ SUPERVISOR NODE ERROR: {e}")
            logger.exception("Full traceback:")
            if isinstance(state, dict):
                state['task_type'] = 'unknown'
            else:
                state.task_type = 'unknown'
            return state
    
    def _classify_medical_task(self, input_data: dict) -> TaskType:
        """
        Classify medical task based on input features.
        Uses MedGemma multimodal capabilities for image-based classification.
        """
        logger.info("  🔎 CLASSIFICATION PROCESS")
        
        # Get text content and metadata
        text_content = input_data.get('text_content', '')
        data_str = str(input_data).lower()
        full_text = (text_content + ' ' + data_str).lower()
        
        logger.info(f"    ├─ Text content length: {len(text_content)} chars")
        logger.info(f"    └─ Analyzing input...")
        
        # Priority 0: Use MedGemma for image content analysis
        image_path = input_data.get('image_path', '')
        if image_path and os.path.exists(image_path):
            logger.info("    📸 PRIORITY 0: Image-based classification using MedGemma")
            logger.info(f"      └─ Image path: {image_path}")
            
            try:
                logger.info("      └─ Initializing image classifier...")
                
                if not self._model_loaded:
                    logger.info("      └─ Loading MedGemma model...")
                    self._load_model()
                
                # Create classifier with shared MedGemma client
                classifier = MedicalImageClassifier(medgemma_client=self.medgemma)
                logger.info("      └─ Classifying image with context...")
                
                # Get classification from image content
                img_classification, confidence, reasoning = classifier.classify_with_text_context(
                    image_path=image_path,
                    text_context=text_content if text_content else None
                )
                
                logger.info(f"      ├─ Classification result: {img_classification}")
                logger.info(f"      ├─ Confidence: {confidence:.2f}")
                logger.info(f"      └─ Reasoning: {reasoning[:100]}...")
                
                # Map classifier output to TaskType
                classification_map = {
                    "ct_coronary": "ct_coronary",
                    "breast_imaging": "breast_imaging",
                    "chest_xray": "ct_coronary",
                    "general_radiology": "ct_coronary",
                    "unknown": None
                }
                
                if confidence >= 0.5 and img_classification in classification_map:
                    mapped_type = classification_map[img_classification]
                    if mapped_type:
                        logger.info(f"      ✓ Using MedGemma classification: {mapped_type}")
                        return mapped_type
                else:
                    logger.info(f"      ⚠ Low confidence ({confidence:.2f}), falling back...")
                        
            except Exception as e:
                logger.warning(f"      ⚠ Image classification failed: {e}")
        
        # Priority 1: Check filename markers
        if image_path:
            logger.info("    📁 PRIORITY 1: Filename-based classification")
            image_lower = image_path.lower()
            if any(term in image_lower for term in ['ct', 'coronary', 'cardiac', 'ccta', 'angiogram']):
                logger.info("      ✓ Found CT/Coronary markers in filename")
                return "ct_coronary"
            elif any(term in image_lower for term in ['mammogram', 'mammography', 'breast']):
                logger.info("      ✓ Found Breast markers in filename")
                return "breast_imaging"
        
        # Priority 2: Check for lipid profile markers
        logger.info("    🩸 PRIORITY 2: Text-based lipid profile detection")
        lipid_markers = ['ldl', 'hdl', 'triglycerides', 'cholesterol', 'lipid panel', 'lipid profile']
        if any(marker in full_text for marker in lipid_markers):
            logger.info(f"      ✓ Found lipid markers: {[m for m in lipid_markers if m in full_text]}")
            return "lipid_profile"
        
        # Priority 3: Check for breast imaging markers
        logger.info("    🔍 PRIORITY 3: Text-based breast imaging detection")
        breast_markers = ['breast', 'mammogram', 'mammography', 'birads', 'bi-rads']
        if any(marker in full_text for marker in breast_markers):
            logger.info(f"      ✓ Found breast markers: {[m for m in breast_markers if m in full_text]}")
            return "breast_imaging"
        
        # Priority 4: Check for CT coronary markers
        logger.info("    🫀 PRIORITY 4: Text-based CT coronary detection")
        ct_markers = ['stenosis', 'lad', 'lcx', 'rca', 'coronary artery', 'cardiac ct']
        if any(marker in full_text for marker in ct_markers):
            logger.info(f"      ✓ Found CT coronary markers: {[m for m in ct_markers if m in full_text]}")
            return "ct_coronary"
        
        # Priority 5: Check for biopsy markers
        logger.info("    🔬 PRIORITY 5: Text-based biopsy detection")
        biopsy_markers = ['biopsy', 'pathology', 'histology', 'specimen', 'carcinoma']
        if any(marker in full_text for marker in biopsy_markers):
            logger.info(f"      ✓ Found biopsy markers: {[m for m in biopsy_markers if m in full_text]}")
            return "biopsy_report"
        
        # Final fallback
        logger.info("    ⚠ No classification matched, using fallback: unknown")
        return "unknown"
    
    def _route_from_supervisor(self, state: MedicalState) -> str:
        """Router function for conditional edges from supervisor."""
        logger.info("\n" + "-" * 80)
        logger.info("🔄 ROUTING DECISION")
        logger.info("-" * 80)
        
        if isinstance(state, dict):
            task_type = state.get('task_type', 'unknown')
            logger.info(f"  └─ From dict state, task_type: {task_type}")
        else:
            task_type = state.task_type
            logger.info(f"  └─ From MedicalState, task_type: {task_type}")
        
        valid_tasks = ["ct_coronary", "lipid_profile", "breast_imaging", "biopsy_report"]
        
        if task_type in valid_tasks:
            logger.info(f"  ✓ Routing to: {task_type.upper()} node")
            logger.info("-" * 80)
            return task_type
        else:
            logger.warning(f"  ⚠ Unknown task type: {task_type}")
            logger.info("  └─ Routing to: END (workflow termination)")
            logger.info("-" * 80)
            return "unknown"
    
    def _ct_coronary_node(self, state: MedicalState) -> MedicalState:
        """CT Coronary Angiography specialized node."""
        node_start_time = time.time()
        logger.info("\n" + "=" * 80)
        logger.info("📊 NODE: CT_CORONARY")
        logger.info("=" * 80)
        logger.info("📋 NODE FUNCTION: Processing cardiac CT findings")
        logger.info("-" * 80)
        
        logger.info("📥 INPUT DATA")
        if isinstance(state, dict):
            input_data = state.get('input_data', {})
            logger.info(f"  └─ State type: dict")
        else:
            input_data = state.input_data or {}
            logger.info(f"  └─ State type: MedicalState object")
        
        logger.info(f"  └─ Current input_data keys: {list(input_data.keys())}")
        
        # Store task type in input_data for the diagnose node
        logger.info("📝 PROCESSING")
        logger.info("  └─ Setting task_type to 'ct_coronary'")
        input_data['task_type'] = 'ct_coronary'
        
        if isinstance(state, dict):
            state['input_data'] = input_data
            logger.info("  └─ Updated dict state")
        else:
            state.input_data = input_data
            logger.info("  └─ Updated MedicalState")
        
        node_duration = time.time() - node_start_time
        logger.info("-" * 80)
        logger.info(f"✓ CT_CORONARY NODE COMPLETED")
        logger.info(f"  └─ Task type set: ct_coronary")
        logger.info(f"  └─ Duration: {node_duration:.3f}s")
        logger.info("=" * 80)
        return state
    
    def _lipid_profile_node(self, state: MedicalState) -> MedicalState:
        """Lipid Profile specialized node."""
        node_start_time = time.time()
        logger.info("\n" + "=" * 80)
        logger.info("🩸 NODE: LIPID_PROFILE")
        logger.info("=" * 80)
        logger.info("📋 NODE FUNCTION: Processing cholesterol panel")
        logger.info("-" * 80)
        
        logger.info("📥 INPUT DATA")
        if isinstance(state, dict):
            input_data = state.get('input_data', {})
            logger.info(f"  └─ State type: dict")
        else:
            input_data = state.input_data or {}
            logger.info(f"  └─ State type: MedicalState object")
        
        # Check for lab values
        ldl = input_data.get('ldl')
        hdl = input_data.get('hdl')
        tg = input_data.get('triglycerides')
        
        logger.info("🔬 LAB VALUES DETECTED")
        if ldl:
            logger.info(f"  ├─ LDL: {ldl} mg/dL")
        if hdl:
            logger.info(f"  ├─ HDL: {hdl} mg/dL")
        if tg:
            logger.info(f"  └─ Triglycerides: {tg} mg/dL")
        if not any([ldl, hdl, tg]):
            logger.info("  └─ No specific lab values found")
        
        logger.info("📝 PROCESSING")
        logger.info("  └─ Setting task_type to 'lipid_profile'")
        input_data['task_type'] = 'lipid_profile'
        
        if isinstance(state, dict):
            state['input_data'] = input_data
            logger.info("  └─ Updated dict state")
        else:
            state.input_data = input_data
            logger.info("  └─ Updated MedicalState")
        
        node_duration = time.time() - node_start_time
        logger.info("-" * 80)
        logger.info(f"✓ LIPID_PROFILE NODE COMPLETED")
        logger.info(f"  └─ Task type set: lipid_profile")
        logger.info(f"  └─ Duration: {node_duration:.3f}s")
        logger.info("=" * 80)
        return state
    
    def _breast_imaging_node(self, state: MedicalState) -> MedicalState:
        """Breast Imaging specialized node."""
        node_start_time = time.time()
        logger.info("\n" + "=" * 80)
        logger.info("🔍 NODE: BREAST_IMAGING")
        logger.info("=" * 80)
        logger.info("📋 NODE FUNCTION: Processing mammogram/ultrasound")
        logger.info("-" * 80)
        
        logger.info("📥 INPUT DATA")
        if isinstance(state, dict):
            input_data = state.get('input_data', {})
            logger.info(f"  └─ State type: dict")
        else:
            input_data = state.input_data or {}
            logger.info(f"  └─ State type: MedicalState object")
        
        # Check for imaging details
        birads = input_data.get('birads_category')
        modality = input_data.get('imaging_modality', 'Unknown')
        
        logger.info("🖼️ IMAGING DATA")
        logger.info(f"  ├─ Modality: {modality}")
        if birads:
            logger.info(f"  └─ BI-RADS: {birads}")
        else:
            logger.info(f"  └─ BI-RADS: Not specified")
        
        logger.info("📝 PROCESSING")
        logger.info("  └─ Setting task_type to 'breast_imaging'")
        input_data['task_type'] = 'breast_imaging'
        
        if isinstance(state, dict):
            state['input_data'] = input_data
            logger.info("  └─ Updated dict state")
        else:
            state.input_data = input_data
            logger.info("  └─ Updated MedicalState")
        
        node_duration = time.time() - node_start_time
        logger.info("-" * 80)
        logger.info(f"✓ BREAST_IMAGING NODE COMPLETED")
        logger.info(f"  └─ Task type set: breast_imaging")
        logger.info(f"  └─ Duration: {node_duration:.3f}s")
        logger.info("=" * 80)
        return state
    
    def _biopsy_report_node(self, state: MedicalState) -> MedicalState:
        """Biopsy Report specialized node."""
        node_start_time = time.time()
        logger.info("\n" + "=" * 80)
        logger.info("🔬 NODE: BIOPSY_REPORT")
        logger.info("=" * 80)
        logger.info("📋 NODE FUNCTION: Processing pathology findings")
        logger.info("-" * 80)
        
        logger.info("📥 INPUT DATA")
        if isinstance(state, dict):
            input_data = state.get('input_data', {})
            logger.info(f"  └─ State type: dict")
        else:
            input_data = state.input_data or {}
            logger.info(f"  └─ State type: MedicalState object")
        
        # Check for pathology details
        report_text = input_data.get('report_text', '')
        specimen = input_data.get('specimen_type', 'Unknown')
        
        logger.info("📋 PATHOLOGY DATA")
        logger.info(f"  ├─ Specimen: {specimen}")
        logger.info(f"  └─ Report length: {len(report_text)} chars")
        
        logger.info("📝 PROCESSING")
        logger.info("  └─ Setting task_type to 'biopsy_report'")
        input_data['task_type'] = 'biopsy_report'
        
        if isinstance(state, dict):
            state['input_data'] = input_data
            logger.info("  └─ Updated dict state")
        else:
            state.input_data = input_data
            logger.info("  └─ Updated MedicalState")
        
        node_duration = time.time() - node_start_time
        logger.info("-" * 80)
        logger.info(f"✓ BIOPSY_REPORT NODE COMPLETED")
        logger.info(f"  └─ Task type set: biopsy_report")
        logger.info(f"  └─ Duration: {node_duration:.3f}s")
        logger.info("=" * 80)
        return state
    
    def _diagnose_node(self, state: MedicalState) -> MedicalState:
        """
        Diagnose node: Generates structured clinical assessment using MedGemma.
        Includes diagnosis and treatment plan.
        """
        node_start_time = time.time()
        logger.info("\n" + "=" * 80)
        logger.info("🤖 NODE: DIAGNOSE")
        logger.info("=" * 80)
        logger.info("📋 NODE FUNCTION: Generating structured clinical assessment with MedGemma")
        logger.info("-" * 80)
        
        # Load model if needed
        if not self._model_loaded:
            logger.info("📥 MODEL LOADING")
            logger.info("  └─ MedGemma model not loaded, loading now...")
            self._load_model()
        else:
            logger.info("📥 MODEL STATUS")
            logger.info("  ✓ MedGemma model already loaded")
        
        try:
            # Extract data
            logger.info("📥 DATA EXTRACTION")
            if isinstance(state, dict):
                task_type = state.get('task_type', 'unknown')
                input_data = state.get('input_data', {})
                logger.info(f"  └─ State type: dict")
            else:
                task_type = state.task_type
                input_data = state.input_data or {}
                logger.info(f"  └─ State type: MedicalState object")
            
            logger.info(f"  ├─ Task type: {task_type}")
            logger.info(f"  ├─ Input data keys: {list(input_data.keys())}")
            
            # Check for image
            image_path = input_data.get('image_path')
            has_image = image_path and os.path.exists(image_path)
            logger.info(f"  └─ Has image: {has_image}")
            if has_image:
                logger.info(f"      └─ Image path: {image_path}")
            
            # Generate structured assessment
            logger.info("\n🧠 GENERATING STRUCTURED ASSESSMENT")
            logger.info(f"  └─ Calling MedGemma for {task_type}...")
            logger.info(f"  └─ Max tokens: 1024")
            
            gen_start_time = time.time()
            structured_assessment = self.medgemma.generate_structured_assessment(
                task_type=task_type,
                input_data=input_data,
                max_new_tokens=1024
            )
            gen_duration = time.time() - gen_start_time
            
            # Log assessment details
            logger.info("\n📊 ASSESSMENT RESULTS")
            logger.info(f"  ├─ Generation time: {gen_duration:.2f}s")
            
            clinical_summary = structured_assessment.get('clinical_summary', '')
            primary_diagnosis = structured_assessment.get('primary_diagnosis', '')
            differentials = structured_assessment.get('differentials', '')
            treatment_plan = structured_assessment.get('treatment_plan', '')
            lifestyle = structured_assessment.get('lifestyle_recommendations', '')
            follow_up = structured_assessment.get('follow_up', '')
            
            logger.info(f"  ├─ Clinical Summary: {len(clinical_summary)} chars")
            logger.info(f"  ├─ Primary Diagnosis: {len(primary_diagnosis)} chars")
            logger.info(f"  ├─ Differentials: {len(differentials)} chars")
            logger.info(f"  ├─ Treatment Plan: {len(treatment_plan)} chars")
            logger.info(f"  ├─ Lifestyle Recommendations: {len(lifestyle)} chars")
            logger.info(f"  └─ Follow-up: {len(follow_up)} chars")
            
            # Update state with structured assessment
            logger.info("\n📝 UPDATING STATE")
            if isinstance(state, dict):
                state['structured_assessment'] = structured_assessment
                state['status'] = 'completed'
                state['end_time'] = time.time()
                logger.info("  └─ Updated dict state with assessment")
            else:
                state.structured_assessment = structured_assessment
                state.complete()
                logger.info("  └─ Updated MedicalState with assessment")
            
            node_duration = time.time() - node_start_time
            logger.info("-" * 80)
            logger.info(f"✓ DIAGNOSE NODE COMPLETED")
            logger.info(f"  └─ Assessment generated successfully")
            logger.info(f"  └─ Total node duration: {node_duration:.2f}s")
            logger.info("=" * 80)
            return state
            
        except Exception as e:
            logger.error(f"❌ DIAGNOSE NODE ERROR: {e}")
            logger.exception("Full traceback:")
            
            if isinstance(state, dict):
                state['error'] = f"Assessment generation failed: {e}"
                state['status'] = 'failed'
                state['end_time'] = time.time()
            else:
                state.fail(f"Assessment generation failed: {e}")
            return state
    
    def run(self, task_type: str = None, input_data: dict = None, 
            cleanup_after: bool = True) -> MedicalState:
        """
        Run the complete workflow with automatic classification.
        
        Args:
            task_type: Optional pre-specified task type
            input_data: Input data dictionary
            cleanup_after: Whether to unload model after execution
        
        Returns:
            Final state with structured assessment
        """
        workflow_start_time = time.time()
        
        logger.info("\n\n" + "=" * 80)
        logger.info("🚀 WORKFLOW EXECUTION STARTED")
        logger.info("=" * 80)
        
        try:
            # Auto-classify if task_type not provided
            logger.info("📋 WORKFLOW SETUP")
            if not task_type and input_data:
                logger.info("  └─ No task_type provided, will auto-classify")
                task_type = self._classify_medical_task(input_data)
                logger.info(f"  ✓ Auto-classified task: {task_type}")
            elif not task_type:
                logger.info("  ⚠ No task_type provided, using 'unknown'")
                task_type = "unknown"
            else:
                logger.info(f"  ✓ Using provided task_type: {task_type}")
            
            # Ensure task_type is in input_data
            if input_data is None:
                input_data = {}
                logger.info("  └─ Created empty input_data dict")
            input_data['task_type'] = task_type
            
            # Log input data details
            logger.info("📥 INPUT DATA SUMMARY")
            logger.info(f"  ├─ Task type: {task_type}")
            logger.info(f"  ├─ Input keys: {list(input_data.keys())}")
            if input_data.get('image_path'):
                logger.info(f"  ├─ Image: {input_data['image_path']}")
            if input_data.get('text_content'):
                text_len = len(input_data['text_content'])
                logger.info(f"  └─ Text content: {text_len} chars")
            
            # Create initial state
            logger.info("\n🏗️  CREATING INITIAL STATE")
            state = MedicalState(
                task_type=task_type,
                input_data=input_data
            )
            logger.info("  ✓ MedicalState created")
            
            logger.info("\n" + "=" * 80)
            logger.info("🏥 STARTING MEDICAL AI WORKFLOW")
            logger.info("=" * 80)
            logger.info(f"📌 Input type: {task_type}")
            logger.info("=" * 80)
            
            # Execute graph
            logger.info("\n▶️  EXECUTING WORKFLOW GRAPH")
            result = self.graph.invoke(state)
            
            # Convert result back to MedicalState if needed
            if isinstance(result, dict):
                final_state = MedicalState(
                    task_type=result.get('task_type', task_type),
                    input_data=result.get('input_data', input_data),
                    structured_assessment=result.get('structured_assessment'),
                    status=result.get('status', WorkflowStatus.FAILED),
                    error=result.get('error', ''),
                    start_time=result.get('start_time', state.start_time),
                    end_time=result.get('end_time')
                )
                logger.info("  └─ Converted dict result to MedicalState")
            else:
                final_state = result
                logger.info("  └─ Result is already MedicalState")
            
            # Log completion
            workflow_duration = time.time() - workflow_start_time
            logger.info("\n" + "=" * 80)
            logger.info("✅ WORKFLOW EXECUTION COMPLETED")
            logger.info("=" * 80)
            logger.info(f"📊 SUMMARY")
            logger.info(f"  ├─ Total duration: {workflow_duration:.2f}s")
            logger.info(f"  ├─ Status: {final_state.status}")
            logger.info(f"  ├─ Task: {final_state.task_type}")
            if final_state.structured_assessment:
                logger.info(f"  └─ Has assessment: Yes")
            else:
                logger.warning(f"  └─ Has assessment: No")
            logger.info("=" * 80)
            
            return final_state
            
        finally:
            # Cleanup model
            if cleanup_after:
                logger.info("\n🧹 CLEANUP")
                self._unload_model()
                logger.info("  ✓ Model cleaned up (memory freed)")
                logger.info("=" * 80 + "\n")


class SequentialWorkflow:
    """
    Fallback workflow that runs nodes sequentially.
    Used when LangGraph is not available.
    """
    
    def __init__(self, graph: MedicalGraph):
        self.graph = graph
        logger.info("  └─ SequentialWorkflow initialized (fallback)")
    
    def invoke(self, state: MedicalState) -> MedicalState:
        """Execute workflow sequentially."""
        logger.info("\n📍 SEQUENTIAL WORKFLOW EXECUTION")
        logger.info("-" * 80)
        
        # Supervisor classification
        logger.info("Step 1/4: Supervisor")
        state = self.graph._supervisor_node(state)
        
        # Route to appropriate specialized node
        task_type = state.task_type if not isinstance(state, dict) else state.get('task_type')
        logger.info(f"\nStep 2/4: Specialized Node ({task_type})")
        
        if task_type == "ct_coronary":
            state = self.graph._ct_coronary_node(state)
        elif task_type == "lipid_profile":
            state = self.graph._lipid_profile_node(state)
        elif task_type == "breast_imaging":
            state = self.graph._breast_imaging_node(state)
        elif task_type == "biopsy_report":
            state = self.graph._biopsy_report_node(state)
        else:
            logger.error(f"Unknown task type: {task_type}")
            if isinstance(state, dict):
                state['status'] = 'failed'
            else:
                state.fail(f"Unknown task type: {task_type}")
            return state
        
        # Diagnose (generates structured assessment)
        logger.info("\nStep 3/4: Diagnose")
        state = self.graph._diagnose_node(state)
        
        logger.info("\nStep 4/4: Complete")
        logger.info("-" * 80)
        
        return state
