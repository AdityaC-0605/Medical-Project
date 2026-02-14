"""
Multimodal Medical Image Classifier using MedGemma
Uses vision capabilities to analyze image content and classify medical images
"""

import logging
from typing import Optional, Tuple
from app.core.medgemma_client import MedGemmaClient

logger = logging.getLogger(__name__)


class MedicalImageClassifier:
    """
    Classifies medical images using MedGemma's multimodal capabilities.
    """
    
    CATEGORIES = {
        "ct_coronary": "CT Coronary Angiography",
        "breast_imaging": "Breast Imaging - mammogram or ultrasound",
        "chest_xray": "Chest X-ray",
        "brain_mri": "Brain MRI",
        "abdominal_ct": "Abdominal CT",
        "unknown": "Unknown medical imaging"
    }
    
    def __init__(self, medgemma_client: Optional[MedGemmaClient] = None):
        """Initialize classifier with optional shared MedGemma client."""
        self.medgemma: Optional[MedGemmaClient] = medgemma_client
        self._model_loaded = medgemma_client is not None
        self._external_client = medgemma_client is not None
        logger.info("MedicalImageClassifier initialized" + 
                   (" (using shared client)" if self._external_client else ""))
    
    def _load_model(self):
        """Load MedGemma model if not using external client."""
        if not self._model_loaded and not self._external_client:
            logger.info("Loading MedGemma for classification...")
            self.medgemma = MedGemmaClient()
            self._model_loaded = True
    
    def _unload_model(self):
        """Unload model if not using external client."""
        if self._model_loaded and self.medgemma and not self._external_client:
            logger.info("Unloading MedGemma classifier...")
            self.medgemma.cleanup()
            self.medgemma = None
            self._model_loaded = False
    
    def classify_image(self, image_path: str, max_new_tokens: int = 128) -> Tuple[str, float, str]:
        """
        Classify a medical image with detailed analysis.
        
        Returns:
            Tuple of (classification, confidence_score, reasoning)
        """
        if not image_path:
            return "unknown", 0.0, "No image provided"
        
        try:
            self._load_model()
            
            # Build comprehensive classification prompt
            prompt = """You are a radiologist examining a medical image. 

Analyze this image carefully and identify:
1. What anatomical region is shown? (heart/chest, breast, brain, abdomen, etc.)
2. What imaging modality is used? (CT, MRI, X-ray, ultrasound, mammogram)
3. What specific anatomical structures are visible?

Based on your analysis, classify this image into ONE category:

- BREAST_IMAGING: Mammogram, breast ultrasound, or breast MRI showing breast tissue, masses, calcifications
- CT_CORONARY: CT scan showing heart, coronary arteries, cardiac structures, chest CT with cardiac focus
- CHEST_XRAY: Chest X-ray showing lungs, ribs, general chest structures
- BRAIN_MRI: Brain MRI or CT showing cerebral structures
- ABDOMINAL_CT: CT scan showing liver, kidneys, spleen, abdominal organs

Provide your classification and a brief explanation of what you see in the image.

CLASSIFICATION:"""
            
            logger.info(f"Classifying image: {image_path}")
            
            # Generate classification with more tokens for detailed response
            response = self.medgemma.generate_text(
                prompt=prompt,
                image_path=image_path,
                max_new_tokens=min(max_new_tokens, 128)  # Limit tokens for classification
            )
            
            # Parse response
            classification, confidence, reasoning = self._parse_response(response)
            
            logger.info(f"Classified as: {classification} (confidence: {confidence:.2f})")
            
            return classification, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "unknown", 0.0, f"Error: {str(e)}"
        
        finally:
            self._unload_model()
    
    def classify_with_text_context(self, image_path: str, text_context: Optional[str] = None, 
                                    max_new_tokens: int = 128) -> Tuple[str, float, str]:
        """
        Classify image with text context.
        
        Returns:
            Tuple of (classification, confidence_score, reasoning)
        """
        if not image_path:
            return "unknown", 0.0, "No image provided"
        
        try:
            self._load_model()
            
            # Build comprehensive prompt with context
            if text_context:
                prompt = f"""You are a radiologist examining a medical image.

CLINICAL CONTEXT: {text_context}

Analyze this image carefully and identify:
1. What anatomical region is shown? (heart/chest, breast, brain, abdomen, etc.)
2. What imaging modality is used? (CT, MRI, X-ray, ultrasound, mammogram)
3. What specific anatomical structures are visible?

Based on your analysis AND the clinical context, classify this image into ONE category:

- BREAST_IMAGING: Mammogram, breast ultrasound, or breast MRI showing breast tissue, masses, calcifications
- CT_CORONARY: CT scan showing heart, coronary arteries, cardiac structures, chest CT with cardiac focus
- CHEST_XRAY: Chest X-ray showing lungs, ribs, general chest structures
- BRAIN_MRI: Brain MRI or CT showing cerebral structures
- ABDOMINAL_CT: CT scan showing liver, kidneys, spleen, abdominal organs

Provide your classification and explain how the image features and clinical context support your decision.

CLASSIFICATION:"""
            else:
                prompt = """You are a radiologist examining a medical image. 

Analyze this image carefully and identify:
1. What anatomical region is shown? (heart/chest, breast, brain, abdomen, etc.)
2. What imaging modality is used? (CT, MRI, X-ray, ultrasound, mammogram)
3. What specific anatomical structures are visible?

Based on your analysis, classify this image into ONE category:

- BREAST_IMAGING: Mammogram, breast ultrasound, or breast MRI showing breast tissue, masses, calcifications
- CT_CORONARY: CT scan showing heart, coronary arteries, cardiac structures, chest CT with cardiac focus
- CHEST_XRAY: Chest X-ray showing lungs, ribs, general chest structures
- BRAIN_MRI: Brain MRI or CT showing cerebral structures
- ABDOMINAL_CT: CT scan showing liver, kidneys, spleen, abdominal organs

Provide your classification and a brief explanation of what you see in the image.

CLASSIFICATION:"""
            
            logger.info("Classifying image with context")
            
            response = self.medgemma.generate_text(
                prompt=prompt,
                image_path=image_path,
                max_new_tokens=min(max_new_tokens, 128)  # Limit tokens for classification
            )
            
            classification, confidence, reasoning = self._parse_response(response)
            
            # Boost confidence based on text context
            if text_context:
                text_lower = text_context.lower()
                if classification == "breast_imaging" and any(term in text_lower for term in ['breast', 'mammogram', 'birads']):
                    confidence = min(confidence + 0.1, 0.95)
                elif classification == "ct_coronary" and any(term in text_lower for term in ['coronary', 'heart', 'chest pain']):
                    confidence = min(confidence + 0.1, 0.95)
            
            logger.info(f"Classified as: {classification} (confidence: {confidence:.2f})")
            
            return classification, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Classification with context failed: {e}")
            return "unknown", 0.0, f"Error: {str(e)}"
        
        finally:
            self._unload_model()
    
    def _parse_response(self, response: str) -> Tuple[str, float, str]:
        """Parse classification response."""
        response_lower = response.lower()
        
        # Check for fallback indicator
        if "educational purposes" in response_lower or "model failed" in response_lower:
            logger.warning("Detected fallback response")
            return "unknown", 0.3, "Model failed to analyze image"
        
        # Classification keywords (priority order)
        keywords = [
            ("breast_imaging", ["breast", "mammogram", "mammography", "birads"]),
            ("ct_coronary", ["coronary", "cardiac", "heart ct", "lad", "lcx", "rca"]),
            ("chest_xray", ["chest x-ray", "chest xray", "lung", "thoracic"]),
            ("brain_mri", ["brain", "cerebral", "neurological"]),
            ("abdominal_ct", ["abdominal", "abdomen", "liver", "kidney"]),
        ]
        
        for category, patterns in keywords:
            if any(pattern in response_lower for pattern in patterns):
                confidence = 0.75
                if len(response) > 50:
                    confidence = min(confidence + 0.1, 0.9)
                return category, confidence, response[:150]
        
        return "unknown", 0.5, response[:150]


# Convenience function
def classify_medical_image(image_path: str, text_context: Optional[str] = None) -> Tuple[str, float, str]:
    """Quick classification function."""
    classifier = MedicalImageClassifier()
    
    if text_context:
        return classifier.classify_with_text_context(image_path, text_context)
    else:
        return classifier.classify_image(image_path)
