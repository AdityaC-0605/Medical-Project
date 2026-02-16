"""MedGemma Client - Fixed image handling for Gemma3"""

import logging
import os
import re
from typing import Optional, Dict, Any, List
from PIL import Image

logger = logging.getLogger(__name__)


class MedGemmaClient:
    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._loaded = False
        self._device = None
        self.hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.image_token = "<<start_of_image>>"  # Gemma3 processor expects this token
        logger.info(f"MedGemmaClient initialized ({model_id})")
    
    def load(self) -> bool:
        """Load model and processor."""
        if self._loaded:
            return True
        
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
            
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
                logger.info("🍎 Using Apple Silicon MPS")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
            
            logger.info(f"Loading MedGemma on {self._device}...")
            
            logger.info("  └─ Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=self.hf_token,
            )
            logger.info("  ✓ Processor loaded")
            
            # Verify image token
            test_id = self.processor.tokenizer.convert_tokens_to_ids(self.image_token)
            logger.info(f"  ✓ Image token {self.image_token} = ID {test_id}")
            
            logger.info("  └─ Loading model weights...")
            torch_dtype = torch.float32 if str(self._device) == "mps" else torch.float16
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=self.hf_token,
            )
            
            self.model.eval()
            self._loaded = True
            
            logger.info("✅ MedGemma loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def generate_text(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 0.7
    ) -> str:
        """Generate text with optional image."""
        if not self._loaded and not self.load():
            return ""
        
        import torch
        
        try:
            has_image = image_path and os.path.exists(image_path)
            
            if has_image:
                # Load image
                images = self._load_images(image_path)
                
                # CRITICAL: Add image token at the very beginning
                # The token must be present for the processor to match with images
                if not prompt.startswith(self.image_token):
                    prompt = f"{self.image_token}\n{prompt}"
                
                logger.info(f"📝 Processing with image ({len(prompt)} chars)")
                logger.info(f"   Prompt starts with: {prompt[:50]}...")
                
                # Process with image
                inputs = self.processor(
                    text=prompt,
                    images=images,
                    return_tensors="pt",
                )
            else:
                logger.info(f"📝 Text-only ({len(prompt)} chars)")
                inputs = self.processor(
                    text=prompt, 
                    return_tensors="pt",
                )
            
            inputs = self._move_to_device(inputs)
            
            logger.info(f"🚀 Generating (max_tokens={max_new_tokens}, do_sample={do_sample})...")
            
            # Build generation kwargs
            gen_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.processor.tokenizer.pad_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "use_cache": True,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9
            
            with torch.no_grad():
                output_ids = self.model.generate(**gen_kwargs)
            
            generated_text = self._decode_output(inputs, output_ids)
            self._clear_cache()
            
            logger.info(f"✅ Generated {len(generated_text)} chars")
            if generated_text:
                logger.info(f"📖 Preview: {generated_text[:100]}...")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def _load_images(self, image_path: str, max_size: int = 512) -> List[Image.Image]:
        """Load and preprocess image."""
        image = Image.open(image_path).convert("RGB")
        
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return [image]
    
    def _move_to_device(self, inputs: Dict) -> Dict:
        """Move inputs to device."""
        import torch
        return {k: v.to(self._device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}
    
    def _clear_cache(self):
        """Clear MPS cache."""
        import torch
        if str(self._device) == "mps":
            try:
                torch.mps.empty_cache()
            except:
                pass
    
    def generate_structured_assessment(
        self,
        task_type: str,
        input_data: Dict[str, Any],
        max_new_tokens: int = 512
    ) -> Dict[str, str]:
        """Generate structured clinical assessment."""
        if not self._loaded and not self.load():
            return self._generate_fallback_response(task_type, input_data)
        
        import torch
        
        try:
            prompt = self._build_structured_prompt(task_type, input_data)
            image_path = input_data.get("image_path")
            has_image = image_path and os.path.exists(image_path)
            
            logger.info(f"🧠 MedGemma analyzing {task_type}")
            logger.info(f"📸 Image: {has_image}")
            
            if has_image:
                images = self._load_images(image_path)
                
                # Add image token
                if not prompt.startswith(self.image_token):
                    prompt = f"{self.image_token}\n{prompt}"
                
                logger.info(f"   Prompt: {prompt[:60]}...")
                
                inputs = self.processor(
                    text=prompt,
                    images=images,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=prompt,
                    return_tensors="pt",
                )
            
            inputs = self._move_to_device(inputs)
            
            logger.info(f"🚀 Generating assessment (max_tokens={max_new_tokens})...")
            
            # Build generation kwargs - optimized for faster generation
            gen_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,  # Greedy decoding for faster, consistent results
                "pad_token_id": self.processor.tokenizer.pad_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "use_cache": True,
            }
            
            with torch.no_grad():
                output_ids = self.model.generate(**gen_kwargs)
            
            generated_text = self._decode_output(inputs, output_ids)
            self._clear_cache()
            
            logger.info(f"✅ Generated {len(generated_text)} chars")
            
            if generated_text and len(generated_text) > 30:
                parsed = self._parse_structured_output(generated_text)
                # Log what we got for debugging
                logger.info(f"✓ Generated {len(generated_text)} chars")
                logger.info(f"✓ Parsed diagnosis: '{parsed.get('primary_diagnosis', 'EMPTY')[:50]}...'")
                
                # Accept if we have any diagnosis content
                if parsed.get("primary_diagnosis") and len(parsed["primary_diagnosis"].strip()) > 5:
                    logger.info(f"✓ Using parsed assessment")
                    return parsed
                else:
                    # If parsing failed but we have content, use the raw text as summary
                    logger.info(f"✓ Using raw text as fallback")
                    return {
                        "clinical_summary": generated_text[:500],
                        "primary_diagnosis": "See clinical summary above",
                        "differentials": "",
                        "treatment_plan": "",
                        "lifestyle_recommendations": "",
                        "follow_up": "Consult specialist for detailed evaluation"
                    }
            
            logger.warning(f"⚠️ Output too short ({len(generated_text)} chars), using fallback")
            return self._generate_fallback_response(task_type, input_data)
                
        except Exception as e:
            logger.error(f"❌ Assessment error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._generate_fallback_response(task_type, input_data)
    
    def _decode_output(self, inputs, output_ids) -> str:
        """Decode output."""
        try:
            if isinstance(inputs, dict) and 'input_ids' in inputs:
                input_len = inputs['input_ids'].shape[1]
            else:
                input_len = inputs.input_ids.shape[1]
            
            generated_ids = output_ids[0][input_len:]
            
            if generated_ids.numel() == 0:
                return ""
            
            text = self.processor.decode(generated_ids, skip_special_tokens=True)
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ Decode error: {e}")
            return ""
    
    def _build_structured_prompt(self, task_type: str, input_data: Dict[str, Any]) -> str:
        """Build medical prompt."""
        text_content = input_data.get("text_content", "")
        
        experts = {
            "ct_coronary": ("cardiologist", "cardiac CT", "coronary arteries"),
            "breast_imaging": ("radiologist", "breast imaging", "masses, calcifications"),
            "lipid_profile": ("cardiologist", "lipid panel", "cholesterol levels"),
            "biopsy_report": ("pathologist", "biopsy", "histology")
        }
        
        role, task, focus = experts.get(task_type, ("expert", "case", "findings"))
        
        prompt = f"""You are an expert {role}. Analyze this {task}.

Provide brief assessment:
CLINICAL SUMMARY: [brief summary]
PRIMARY DIAGNOSIS: [main finding]
TREATMENT: [recommendations]
FOLLOW-UP: [next steps]"""
        
        return prompt
    
    def _parse_structured_output(self, text: str) -> Dict[str, str]:
        """Parse structured output."""
        result = {
            "clinical_summary": "",
            "primary_diagnosis": "",
            "differentials": "",
            "treatment_plan": "",
            "lifestyle_recommendations": "",
            "follow_up": ""
        }
        
        if not text:
            return result
        
        sections = {
            "clinical_summary": ["CLINICAL SUMMARY:", "Clinical Summary:", "SUMMARY:"],
            "primary_diagnosis": ["PRIMARY DIAGNOSIS:", "Primary Diagnosis:", "DIAGNOSIS:"],
            "differentials": ["DIFFERENTIAL DIAGNOSES:", "Differential Diagnoses:", "DIFFERENTIALS:"],
            "treatment_plan": ["TREATMENT PLAN:", "Treatment Plan:", "TREATMENT:"],
            "lifestyle_recommendations": ["LIFESTYLE RECOMMENDATIONS:", "Lifestyle Recommendations:", "LIFESTYLE:"],
            "follow_up": ["FOLLOW-UP PLAN:", "Follow-up Plan:", "FOLLOW UP:", "FOLLOWUP:"]
        }
        
        text_upper = text.upper()
        positions = {}
        
        for key, headers in sections.items():
            for header in headers:
                idx = text_upper.find(header.upper())
                if idx != -1:
                    positions[key] = (idx, idx + len(header))
                    break
        
        if not positions:
            result["clinical_summary"] = text
            return result
        
        sorted_sections = sorted(positions.items(), key=lambda x: x[1][0])
        
        for i, (key, (start, end)) in enumerate(sorted_sections):
            if i + 1 < len(sorted_sections):
                next_start = sorted_sections[i + 1][1][0]
                content = text[end:next_start]
            else:
                content = text[end:]
            
            content = re.sub(r'^[:\s]+', '', content).strip()
            result[key] = content
        
        return result
    
    def _generate_fallback_response(self, task_type: str, input_data: Dict[str, Any]) -> Dict[str, str]:
        """Fallback response."""
        return {
            "clinical_summary": f"Assessment for {task_type}.",
            "primary_diagnosis": "Error in analysis. Consult specialist.",
            "differentials": "Requires evaluation",
            "treatment_plan": "Specialist referral recommended",
            "lifestyle_recommendations": "Healthy lifestyle",
            "follow_up": "Schedule with provider"
        }
    
    def cleanup(self):
        """Cleanup."""
        try:
            import torch
            import gc
            
            if self.model:
                del self.model
                self.model = None
            if self.processor:
                del self.processor
                self.processor = None
            
            gc.collect()
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            self._loaded = False
            
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")