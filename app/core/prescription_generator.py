"""
Prescription Generator Module
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Medication:
    """Medication details."""
    name: str
    dosage: str
    frequency: str
    instructions: str


@dataclass
class Prescription:
    """Complete prescription."""
    diagnosis_summary: str
    medications: List[Medication]
    lifestyle_changes: List[str]
    follow_up: str
    referral: str
    generated_at: str


class PrescriptionGenerator:
    """Generate prescriptions based on diagnosis."""
    
    def __init__(self):
        logger.info("PrescriptionGenerator initialized")
    
    def generate(self, task_type: str, input_data: Dict, diagnosis: str) -> Dict[str, Any]:
        """
        Generate prescription based on task type and diagnosis.
        
        Args:
            task_type: Type of medical task
            input_data: Patient input data
            diagnosis: AI-generated diagnosis
        
        Returns:
            Prescription dictionary
        """
        if task_type == "lipid_profile":
            return self._generate_lipid_prescription(input_data, diagnosis)
        elif task_type == "ct_coronary":
            return self._generate_cardiac_prescription(input_data, diagnosis)
        elif task_type == "breast_imaging":
            return self._generate_breast_prescription(input_data, diagnosis)
        elif task_type == "biopsy_report":
            return self._generate_pathology_prescription(input_data, diagnosis)
        else:
            return self._generate_generic_prescription(input_data, diagnosis)
    
    def _generate_lipid_prescription(self, data: Dict, diagnosis: str) -> Dict[str, Any]:
        """Generate lipid management prescription."""
        ldl = data.get("ldl", 0)
        
        # Determine medications based on LDL
        medications = []
        if ldl > 190:
            medications.append({
                "name": "Atorvastatin",
                "dosage": "40-80 mg",
                "frequency": "Once daily",
                "instructions": "Take in the evening. High-intensity statin therapy."
            })
        elif ldl > 160:
            medications.append({
                "name": "Atorvastatin",
                "dosage": "20-40 mg",
                "frequency": "Once daily",
                "instructions": "Take in the evening. Moderate to high-intensity therapy."
            })
        elif ldl > 130:
            medications.append({
                "name": "Atorvastatin",
                "dosage": "10-20 mg",
                "frequency": "Once daily",
                "instructions": "Take in the evening. Reassess in 3 months."
            })
        
        # Add additional meds if needed
        tg = data.get("triglycerides", 0)
        if tg > 500:
            medications.append({
                "name": "Omega-3 Fatty Acids (EPA/DHA)",
                "dosage": "2-4 grams",
                "frequency": "Once or twice daily",
                "instructions": "Take with meals to reduce pancreatitis risk."
            })
        
        return {
            "diagnosis_summary": f"Dyslipidemia - Elevated LDL ({ldl} mg/dL)",
            "medications": medications,
            "lifestyle_modifications": [
                "Adopt Mediterranean or DASH diet",
                "Reduce saturated fat to <7% of daily calories",
                "Increase fiber intake to 25-30g daily",
                "Exercise 150 minutes/week moderate intensity",
                "Achieve 5-10% weight loss if overweight",
                "Smoking cessation if applicable"
            ],
            "monitoring": {
                "lipid_panel": "Repeat in 4-12 weeks after initiation, then every 3-12 months",
                "liver_function": "Check AST/ALT at baseline and if symptomatic"
            },
            "follow_up": "Primary care in 3 months",
            "referral": "Lipid specialist if LDL >190 or familial hypercholesterolemia suspected",
            "goals": {
                "ldl_target": "<100 mg/dL (or ≥50% reduction)",
                "non_hdl_target": "<130 mg/dL"
            }
        }
    
    def _generate_cardiac_prescription(self, data: Dict, diagnosis: str) -> Dict[str, Any]:
        """Generate cardiac/CAD prescription."""
        stenosis = data.get("stenosis_percent", 0)
        
        medications = []
        
        # Core therapy for significant CAD
        if stenosis >= 50:
            medications.extend([
                {
                    "name": "Aspirin",
                    "dosage": "81 mg",
                    "frequency": "Once daily",
                    "instructions": "Take with food. Continue indefinitely unless contraindicated."
                },
                {
                    "name": "Atorvastatin",
                    "dosage": "40-80 mg",
                    "frequency": "Once daily",
                    "instructions": "Take in evening. Goal LDL <70 mg/dL for established CAD."
                }
            ])
        
        # Additional therapy for severe stenosis
        if stenosis >= 70:
            medications.append({
                "name": "Metoprolol",
                "dosage": "25-50 mg",
                "frequency": "Twice daily",
                "instructions": "Monitor heart rate and blood pressure. Hold if HR <50 or SBP <90."
            })
        
        return {
            "diagnosis_summary": f"Coronary Artery Disease - {stenosis}% stenosis in {data.get('vessel', 'vessel')}",
            "medications": medications,
            "lifestyle_modifications": [
                "Heart-healthy diet (Mediterranean/DASH)",
                "Daily physical activity as tolerated",
                "Blood pressure control <130/80 mmHg",
                "Diabetes management if applicable (HbA1c <7%)",
                "Smoking cessation - mandatory",
                "Stress management techniques"
            ],
            "monitoring": {
                "lipid_panel": "Every 3-6 months",
                "symptoms": "Monitor for chest pain, dyspnea, palpitations"
            },
            "follow_up": f"{'Cardiology within 1-2 weeks' if stenosis >= 70 else 'Cardiology within 4-6 weeks'}",
            "referral": "Cardiology for stress testing or revascularization evaluation" if stenosis >= 50 else "Continue with primary care",
            "additional_testing": "Stress test or FFR if intermediate stenosis (50-70%) with symptoms"
        }
    
    def _generate_breast_prescription(self, data: Dict, diagnosis: str) -> Dict[str, Any]:
        """Generate breast imaging recommendations."""
        birads = str(data.get("birads_category", ""))
        
        if birads in ["4", "4A", "4B", "4C", "5"]:
            follow_up = "Urgent breast surgeon or interventional radiology referral within 1-2 weeks"
            referral = "Breast surgery/oncology for biopsy and treatment planning"
        elif birads == "3":
            follow_up = "Short-interval mammogram in 6 months"
            referral = "Continue with primary care, consider breast specialist if anxious"
        else:
            follow_up = "Routine screening per guidelines"
            referral = "None at this time"
        
        return {
            "diagnosis_summary": f"Breast Imaging - BI-RADS Category {birads}",
            "medications": [],
            "lifestyle_modifications": [
                "Maintain healthy weight (BMI 18.5-24.9)",
                "Limit alcohol to ≤1 drink/day",
                "Regular physical activity 150 min/week",
                "Monthly breast self-examination",
                "Adherence to screening schedule"
            ],
            "monitoring": {
                "imaging": follow_up,
                "self_exam": "Monthly, report any changes immediately"
            },
            "follow_up": follow_up,
            "referral": referral,
            "additional_testing": "Tissue biopsy if BI-RADS 4 or 5"
        }
    
    def _generate_pathology_prescription(self, data: Dict, diagnosis: str) -> Dict[str, Any]:
        """Generate pathology-based recommendations."""
        return {
            "diagnosis_summary": "Biopsy-proven malignancy" if "carcinoma" in diagnosis.lower() else "Pathology findings",
            "medications": [],
            "lifestyle_modifications": [
                "Maintain nutritious diet",
                "Gentle physical activity as tolerated",
                "Adequate rest and sleep",
                "Emotional support and counseling",
                "Smoking cessation if applicable"
            ],
            "monitoring": {
                "oncology": "Regular follow-up per treatment protocol"
            },
            "follow_up": "Urgent oncology consultation within 1 week",
            "referral": "Medical oncology for treatment planning. Consider multidisciplinary tumor board.",
            "additional_testing": "Staging workup including imaging and labs as indicated"
        }
    
    def _generate_generic_prescription(self, data: Dict, diagnosis: str) -> Dict[str, Any]:
        """Generic prescription template."""
        return {
            "diagnosis_summary": "See detailed AI diagnosis",
            "medications": [],
            "lifestyle_modifications": [
                "Maintain healthy lifestyle",
                "Regular exercise",
                "Balanced nutrition",
                "Adequate sleep",
                "Stress management"
            ],
            "monitoring": {
                "general": "As clinically indicated"
            },
            "follow_up": "Schedule follow-up with primary care physician",
            "referral": "As indicated based on diagnosis",
            "additional_testing": "As clinically indicated"
        }
