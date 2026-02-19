"""
LangGraph Workflow for Medical AI with Supervisor Node
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

TaskType = Literal["ct_coronary", "lipid_profile", "breast_imaging", "biopsy_report", "unknown"]


class MedicalGraph:
    def __init__(self, preload_model: bool = True, diagnosis_max_new_tokens: int = 256):
        self.medgemma: Optional[MedGemmaClient] = None
        self._model_loaded = False
        self.diagnosis_max_new_tokens = diagnosis_max_new_tokens
        
        logger.info("=" * 80)
        logger.info("🏥 MEDICAL GRAPH INITIALIZATION")
        logger.info("=" * 80)
        
        if preload_model:
            logger.info("📦 Preload model requested: True")
            self._load_model()
        else:
            logger.info("📦 Preload model requested: False (will use lazy loading)")
        
        self.graph = self._build_graph()
        logger.info("✓ MedicalGraph initialized successfully")
        logger.info("=" * 80)
    
    def _load_model(self):
        """Load MedGemma model."""
        if not self._model_loaded:
            logger.info("📥 LOADING MODEL")
            logger.info("  └─ Loading MedGemma model...")
            self.medgemma = MedGemmaClient()
            success = self.medgemma.load()
            if not success:
                logger.error("❌ Failed to load MedGemma model!")
                raise RuntimeError("Failed to load MedGemma model")
            self._model_loaded = True
            logger.info("  ✓ MedGemma model loaded successfully")
    
    def _unload_model(self):
        """Unload MedGemma model."""
        if self._model_loaded and self.medgemma:
            logger.info("📤 UNLOADING MODEL")
            self.medgemma.cleanup()
            self.medgemma = None
            self._model_loaded = False
            gc.collect()
            logger.info("  ✓ MedGemma model unloaded")
    
    def _build_graph(self):
        """Build LangGraph workflow."""
        logger.info("🔨 BUILDING WORKFLOW GRAPH")
        
        try:
            from langgraph.graph import StateGraph, END

            workflow = StateGraph(MedicalState)

            workflow.add_node("supervisor", self._supervisor_node)
            workflow.add_node("ct_coronary", self._ct_coronary_node)
            workflow.add_node("lipid_profile", self._lipid_profile_node)
            workflow.add_node("breast_imaging", self._breast_imaging_node)
            workflow.add_node("biopsy_report", self._biopsy_report_node)
            workflow.add_node("diagnose", self._diagnose_node)

            workflow.set_entry_point("supervisor")
            
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

            workflow.add_edge("ct_coronary", "diagnose")
            workflow.add_edge("lipid_profile", "diagnose")
            workflow.add_edge("breast_imaging", "diagnose")
            workflow.add_edge("biopsy_report", "diagnose")
            workflow.add_edge("diagnose", END)

            logger.info("  ✓ Workflow graph built successfully")
            return workflow.compile()

        except ImportError as e:
            logger.warning(f"  ⚠ LangGraph not available ({e})")
            return SequentialWorkflow(self)
    
    def _supervisor_node(self, state: MedicalState) -> MedicalState:
        """Supervisor node: Analyzes input and classifies task."""
        logger.info("\n" + "=" * 80)
        logger.info("🎯 NODE: SUPERVISOR")
        
        try:
            input_data = state.input_data or {}
            existing_task_type = state.task_type
            
            logger.info(f"  └─ Existing task type: {existing_task_type}")
            
            if existing_task_type and existing_task_type != 'unknown':
                logger.info("✓ Task type pre-specified")
                return state
            
            # ONLY use MedGemma for classification, no fallbacks
            classification = self._classify_with_medgemma_only(input_data)
            state.task_type = classification
            
            logger.info(f"✓ Classification: {classification.upper()}")
            return state
            
        except Exception as e:
            logger.error(f"❌ SUPERVISOR ERROR: {e}")
            import traceback
            logger.error(traceback.format_exc())
            state.task_type = 'unknown'
            return state
    
    def _classify_with_medgemma_only(self, input_data: dict) -> TaskType:
        """Classify using ONLY MedGemma, no keyword fallback."""
        logger.info("  🔎 CLASSIFICATION WITH MEDGEMMA")
        
        image_path = input_data.get('image_path', '')
        text_content = input_data.get('text_content', '')
        
        if not image_path or not os.path.exists(image_path):
            logger.info("    ⚠ No image available for classification")
            return "unknown"
        
        try:
            if not self._model_loaded:
                self._load_model()
            
            classifier = MedicalImageClassifier(medgemma_client=self.medgemma)
            img_classification, confidence, reasoning = classifier.classify_with_text_context(
                image_path=image_path,
                text_context=text_content if text_content else None
            )
            
            logger.info(f"    ✓ MedGemma classified: {img_classification} (confidence: {confidence:.2f})")
            
            # Map to valid task types
            classification_map = {
                "ct_coronary": "ct_coronary",
                "breast_imaging": "breast_imaging",
                "chest_xray": "ct_coronary",  # Map to CT for now
                "brain_mri": "ct_coronary",    # Map to CT for now
                "abdominal_ct": "ct_coronary", # Map to CT for now
                "unknown": "unknown"
            }
            
            if confidence >= 0.5 and img_classification in classification_map:
                result = classification_map[img_classification]
                logger.info(f"    ✓ Using classification: {result}")
                return result
            else:
                logger.warning(f"    ⚠ Low confidence ({confidence:.2f}) or unknown, using: unknown")
                return "unknown"
                
        except Exception as e:
            logger.error(f"    ❌ MedGemma classification failed: {e}")
            return "unknown"
    
    def _route_from_supervisor(self, state: MedicalState) -> str:
        """Router function."""
        task_type = state.task_type
        logger.info(f"🔄 Routing: {task_type}")
        
        if task_type in ["ct_coronary", "lipid_profile", "breast_imaging", "biopsy_report"]:
            return task_type
        return "unknown"
    
    def _ct_coronary_node(self, state: MedicalState) -> MedicalState:
        """CT Coronary node."""
        logger.info("📊 NODE: CT_CORONARY")
        input_data = state.input_data or {}
        input_data['task_type'] = 'ct_coronary'
        state.input_data = input_data
        return state
    
    def _lipid_profile_node(self, state: MedicalState) -> MedicalState:
        """Lipid Profile node."""
        logger.info("🩸 NODE: LIPID_PROFILE")
        input_data = state.input_data or {}
        input_data['task_type'] = 'lipid_profile'
        state.input_data = input_data
        return state
    
    def _breast_imaging_node(self, state: MedicalState) -> MedicalState:
        """Breast Imaging node."""
        logger.info("🔍 NODE: BREAST_IMAGING")
        input_data = state.input_data or {}
        input_data['task_type'] = 'breast_imaging'
        state.input_data = input_data
        return state
    
    def _biopsy_report_node(self, state: MedicalState) -> MedicalState:
        """Biopsy Report node."""
        logger.info("🔬 NODE: BIOPSY_REPORT")
        input_data = state.input_data or {}
        input_data['task_type'] = 'biopsy_report'
        state.input_data = input_data
        return state
    
    def _diagnose_node(self, state: MedicalState) -> MedicalState:
        """Diagnose node."""
        logger.info("🤖 NODE: DIAGNOSE")
        
        if not self._model_loaded:
            self._load_model()
        
        try:
            task_type = state.task_type
            input_data = state.input_data or {}
            
            logger.info(f"  ├─ Task: {task_type}")
            logger.info(f"  ├─ Input keys: {list(input_data.keys())}")
            
            structured_assessment = self.medgemma.generate_structured_assessment(
                task_type=task_type,
                input_data=input_data,
                max_new_tokens=self.diagnosis_max_new_tokens
            )
            
            state.structured_assessment = structured_assessment
            state.complete()
            
            logger.info("✓ Assessment generated successfully")
            return state
            
        except Exception as e:
            logger.error(f"❌ DIAGNOSE ERROR: {e}")
            import traceback
            logger.error(traceback.format_exc())
            state.fail(f"Assessment failed: {e}")
            return state
    
    def run(self, task_type: str = None, input_data: dict = None, 
            cleanup_after: bool = True) -> MedicalState:
        """Run workflow."""
        logger.info("\n\n" + "=" * 80)
        logger.info("🚀 WORKFLOW STARTED")
        
        try:
            if not task_type and input_data:
                task_type = self._classify_with_medgemma_only(input_data)
                logger.info(f"  ✓ Auto-classified: {task_type}")
            elif not task_type:
                task_type = "unknown"
            
            if input_data is None:
                input_data = {}
            input_data['task_type'] = task_type
            
            state = MedicalState(
                task_type=task_type,
                input_data=input_data
            )
            
            # Run complete workflow
            result = self.graph.invoke(state)
            
            if isinstance(result, dict):
                final_state = MedicalState(
                    task_type=result.get('task_type', task_type),
                    input_data=result.get('input_data', input_data),
                    structured_assessment=result.get('structured_assessment'),
                    status=result.get('status', WorkflowStatus.COMPLETED),
                    error=result.get('error', ''),
                    start_time=state.start_time,
                    end_time=result.get('end_time')
                )
            else:
                final_state = result
            
            return final_state
            
        finally:
            if cleanup_after:
                self._unload_model()


class SequentialWorkflow:
    """Fallback when LangGraph not available."""
    
    def __init__(self, graph: MedicalGraph):
        self.graph = graph
    
    def invoke(self, state: MedicalState) -> MedicalState:
        """Execute sequentially."""
        state = self.graph._supervisor_node(state)
        
        task_type = state.task_type
        if task_type == "ct_coronary":
            state = self.graph._ct_coronary_node(state)
        elif task_type == "lipid_profile":
            state = self.graph._lipid_profile_node(state)
        elif task_type == "breast_imaging":
            state = self.graph._breast_imaging_node(state)
        elif task_type == "biopsy_report":
            state = self.graph._biopsy_report_node(state)
        else:
            state.fail(f"Unknown task: {task_type}")
            return state
        
        state = self.graph._diagnose_node(state)
        return state
