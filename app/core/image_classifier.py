"""Medical Image Classifier using MedGemma"""

import logging
import os
from typing import Optional, Tuple

from app.core.medgemma_client import MedGemmaClient

logger = logging.getLogger(__name__)


class MedicalImageClassifier:
    """Classifies medical images using MedGemma."""
    
    def __init__(self, medgemma_client: Optional[MedGemmaClient] = None):
        self.medgemma = medgemma_client
        self._external_client = medgemma_client is not None
        
        logger.info(f"ImageClassifier ready (shared={self._external_client})")
    
    def classify_with_text_context(
        self, 
        image_path: str, 
        text_context: Optional[str] = None,
        max_new_tokens: int = 128
    ) -> Tuple[str, float, str]:
        """Classify medical image using MedGemma."""
        if not image_path or not os.path.exists(image_path):
            return "unknown", 0.0, "No valid image"
        
        try:
            if not self.medgemma:
                self.medgemma = MedGemmaClient()
                if not self.medgemma.load():
                    return "unknown", 0.0, "Model load failed"
            
            # Build prompt - image token will be added by medgemma_client
            context = f"\nContext: {text_context[:100]}" if text_context else ""
            
            # Optimized classifier prompt for better accuracy
            prompt = f"""Analyze this medical image and classify it into exactly ONE category:

CT_CORONARY - Heart/coronary arteries scan
BREAST_IMAGING - Breast/mammogram imaging  
CHEST_XRAY - Chest/lungs X-ray
BRAIN_MRI - Brain scan
ABDOMINAL_CT - Abdomen scan
UNKNOWN - Cannot determine

Reply with ONLY the category name (one word, no explanation).{context}

Category:"""
            
            logger.info(f"🔍 Classifying: {os.path.basename(image_path)}")
            
            # Use fast greedy decoding for classification (deterministic & faster)
            response = self.medgemma.generate_text(
                prompt=prompt,
                image_path=image_path,
                max_new_tokens=20,  # Classification only needs short output
                do_sample=False     # Greedy decoding is faster
            )
            
            return self._parse_classification(response, text_context)
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "unknown", 0.0, str(e)
    
    def _parse_classification(self, response: str, text_context: Optional[str] = None) -> Tuple[str, float, str]:
        """Parse classification."""
        if not response:
            return "unknown", 0.0, "Empty"
        
        response_lower = response.lower()
        
        # Normalize response: replace underscores and hyphens with spaces
        response_normalized = response_lower.replace('_', ' ').replace('-', ' ')
        
        categories = [
            ("breast_imaging", ["breast", "mammogram", "mammography"]),
            ("ct_coronary", ["coronary", "cardiac", "heart", "ccta", "ct coronary"]),
            ("chest_xray", ["chest xray", "chest x ray", "thoracic", "chest"]),
            ("brain_mri", ["brain", "cerebral"]),
            ("abdominal_ct", ["abdominal", "abdomen"]),
        ]
        
        for category, keywords in categories:
            if any(kw in response_normalized for kw in keywords):
                confidence = 0.8
                if text_context and any(kw in text_context.lower() for kw in keywords):
                    confidence = 0.95
                return category, confidence, response[:150]
        
        return "unknown", 0.4, response[:150]