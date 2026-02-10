"""
MedGemma Client for Medical Diagnosis
"""

import logging
import time
from typing import Optional
import os

logger = logging.getLogger(__name__)


class MedGemmaClient:
    """
    Client for google/medgemma-1.5-4b-it model.
    Supports text-only and multimodal (image + text) inference.
    """
    
    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._loaded = False
        
        logger.info(f"MedGemmaClient initialized (model: {model_id})")
    
    def load(self) -> bool:
        """
        Load MedGemma model and processor.
        
        Returns:
            True if loaded successfully
        """
        if self._loaded:
            return True
        
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
            
            logger.info("Loading MedGemma model...")
            logger.info("This may take a few minutes on first run...")
            
            # Disable tokenizer parallelism
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Load model with macOS optimizations
            logger.info("Loading model weights...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            self._loaded = True
            logger.info("✅ MedGemma model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MedGemma: {e}")
            return False
    
    def generate_diagnosis(
        self,
        prompt: str,
        context: str = "",
        image_path: Optional[str] = None,
        max_new_tokens: int = 500
    ) -> str:
        """
        Generate medical diagnosis.
        
        Args:
            prompt: The medical query/prompt
            context: Retrieved medical knowledge from RAG
            image_path: Path to medical image (for multimodal)
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated diagnosis text
        """
        if not self._loaded and not self.load():
            raise RuntimeError("Failed to load MedGemma model")
        
        try:
            import torch
            from PIL import Image
            
            # Build full prompt with context
            if context:
                full_prompt = f"""Use the following medical guidelines to analyze this case:

MEDICAL KNOWLEDGE:
{context}

CASE TO ANALYZE:
{prompt}

Provide a detailed medical diagnosis including:
1. Interpretation of findings
2. Clinical significance and risk assessment
3. Evidence-based recommendations
4. Follow-up plan"""
            else:
                full_prompt = prompt
            
            # Prepare messages
            if image_path and os.path.exists(image_path):
                # Multimodal
                logger.info(f"Processing image: {image_path}")
                image = Image.open(image_path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": full_prompt}
                        ]
                    }
                ]
                chat_prompt = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                inputs = self.processor(
                    text=chat_prompt,
                    images=image,
                    return_tensors="pt"
                )
            else:
                # Text-only
                messages = [{"role": "user", "content": full_prompt}]
                chat_prompt = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                inputs = self.processor(text=chat_prompt, return_tensors="pt")
            
            # Generate
            logger.info(f"Generating diagnosis (max {max_new_tokens} tokens)...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy for consistency
                    repetition_penalty=1.1,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode
            response = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Extract just the model's response
            response = self._extract_response(response)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Generated {len(response)} chars in {elapsed:.1f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    def _extract_response(self, text: str) -> str:
        """Extract model's response from full output."""
        text = text.strip()
        
        # Try to find the model's turn
        if "model" in text.lower():
            parts = text.split("model")
            if len(parts) > 1:
                return parts[-1].strip()
        
        # Remove user prompt if present
        if "user" in text.lower():
            parts = text.split("user")
            if len(parts) > 1:
                text = parts[-1].strip()
        
        return text
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        self._loaded = False
        
        try:
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        logger.info("MedGemma resources cleaned up")
