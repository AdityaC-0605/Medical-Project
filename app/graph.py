"""
LangGraph Workflow for Medical AI
"""

import logging
from typing import Optional

from app.state import MedicalState, WorkflowStatus
from app.core.medgemma_client import MedGemmaClient
from app.core.prescription_generator import PrescriptionGenerator

logger = logging.getLogger(__name__)


class MedicalGraph:
    """
    LangGraph workflow for medical diagnosis and prescription.

    Workflow:
        diagnose → prescribe
    """

    def __init__(self):
        self.medgemma = MedGemmaClient()
        self.prescription_gen = PrescriptionGenerator()
        self.graph = self._build_graph()
        logger.info("MedicalGraph initialized")
    
    def _build_graph(self):
        """Build LangGraph workflow."""
        try:
            from langgraph.graph import StateGraph, END

            workflow = StateGraph(MedicalState)

            # Add nodes
            workflow.add_node("diagnose", self._diagnose_node)
            workflow.add_node("prescribe", self._prescribe_node)

            # Set entry point
            workflow.set_entry_point("diagnose")

            # Define edges
            workflow.add_edge("diagnose", "prescribe")
            workflow.add_edge("prescribe", END)

            return workflow.compile()

        except ImportError:
            logger.warning("LangGraph not available, using sequential fallback")
            return SequentialWorkflow(self)
    
    def _diagnose_node(self, state: MedicalState) -> MedicalState:
        """MedGemma diagnosis node."""
        logger.info("Node: DIAGNOSE (MedGemma)")
        
        try:
            # Get image path if multimodal task
            image_path = None
            if state.task_type in ["ct_coronary", "breast_imaging"]:
                image_path = state.input_data.get("image_path")
                if image_path and not __import__('os').path.exists(image_path):
                    image_path = None
            
            # Generate diagnosis
            diagnosis = self.medgemma.generate_diagnosis(
                prompt=state.query,
                image_path=image_path
            )
            
            state.diagnosis = diagnosis
            logger.info(f"✓ Diagnosis generated ({len(diagnosis)} chars)")
            return state
            
        except Exception as e:
            logger.error(f"Diagnosis error: {e}")
            state.fail(f"Diagnosis generation failed: {e}")
            return state
    
    def _prescribe_node(self, state: MedicalState) -> MedicalState:
        """Prescription generation node."""
        logger.info("Node: PRESCRIBE")
        
        try:
            # Generate prescription
            prescription = self.prescription_gen.generate(
                task_type=state.task_type,
                input_data=state.input_data,
                diagnosis=state.diagnosis
            )
            
            state.prescription = prescription
            state.complete()
            
            logger.info("✓ Prescription generated")
            return state
            
        except Exception as e:
            logger.error(f"Prescription error: {e}")
            state.fail(f"Prescription generation failed: {e}")
            return state
    
    def run(self, task_type: str, input_data: dict) -> MedicalState:
        """
        Run the complete workflow.
        
        Args:
            task_type: Type of medical task
            input_data: Input data dictionary
        
        Returns:
            Final state with diagnosis and prescription
        """
        # Build query from input data
        query = self._build_query(task_type, input_data)
        
        # Create initial state
        state = MedicalState(
            task_type=task_type,
            input_data=input_data,
            query=query
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting workflow: {task_type}")
        logger.info(f"{'='*60}")
        
        # Execute graph
        result = self.graph.invoke(state)
        
        # Handle both dict and MedicalState return types
        if isinstance(result, dict):
            # Convert dict back to MedicalState if needed
            final_state = MedicalState(
                task_type=result.get('task_type', task_type),
                input_data=result.get('input_data', input_data),
                query=result.get('query', query),
                diagnosis=result.get('diagnosis', ''),
                prescription=result.get('prescription'),
                status=result.get('status', WorkflowStatus.FAILED),
                error=result.get('error', ''),
                start_time=result.get('start_time', state.start_time),
                end_time=result.get('end_time')
            )
        else:
            final_state = result
        
        return final_state
    
    def _build_query(self, task_type: str, data: dict) -> str:
        """Build query from input data."""
        if task_type == "lipid_profile":
            return (
                f"Patient: {data.get('age')}-year-old {data.get('sex')}. "
                f"Lipid Profile - LDL: {data.get('ldl')} mg/dL, "
                f"HDL: {data.get('hdl')} mg/dL, "
                f"Triglycerides: {data.get('triglycerides')} mg/dL, "
                f"Total Cholesterol: {data.get('total_cholesterol')} mg/dL"
            )
        elif task_type == "ct_coronary":
            return (
                f"CT Coronary Angiography: {data.get('stenosis_percent')}% stenosis "
                f"in {data.get('vessel')}. Finding: {data.get('finding')}"
            )
        elif task_type == "breast_imaging":
            return (
                f"Breast Imaging ({data.get('imaging_modality')}): "
                f"BI-RADS {data.get('birads_category')}. "
                f"Finding: {data.get('finding')}"
            )
        elif task_type == "biopsy_report":
            return f"Pathology Report: {data.get('report_text')}"
        else:
            return str(data)


class SequentialWorkflow:
    """
    Fallback workflow that runs nodes sequentially.
    Used when LangGraph is not available.
    """
    
    def __init__(self, graph: MedicalGraph):
        self.graph = graph
    
    def invoke(self, state: MedicalState) -> MedicalState:
        """Execute workflow sequentially."""
        # Diagnose
        state = self.graph._diagnose_node(state)
        if state.status == WorkflowStatus.FAILED:
            return state

        # Prescribe
        state = self.graph._prescribe_node(state)
        return state
