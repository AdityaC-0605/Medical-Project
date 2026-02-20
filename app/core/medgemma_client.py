"""MedGemma Client - Fixed image handling for Gemma3"""

import json
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
            # float16 on CPU is typically much slower than float32.
            if str(self._device) == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                dtype=torch_dtype,
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
    
    def _load_images(self, image_path: str, max_size: int = 384) -> List[Image.Image]:
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
            
            logger.info(f"🧠 Analyzing {task_type}...")
            
            if has_image:
                images = self._load_images(image_path)
                
                # Add image token
                if not prompt.startswith(self.image_token):
                    prompt = f"{self.image_token}\n{prompt}"
                
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
            
            # Fast generation with minimal overhead
            gen_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.processor.tokenizer.pad_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "use_cache": True,
                "num_beams": 1,
            }
            
            with torch.no_grad():
                output_ids = self.model.generate(**gen_kwargs)
            
            generated_text = self._decode_output(inputs, output_ids)
            
            logger.info(f"✓ Generated {len(generated_text)} chars")
            
            if generated_text and len(generated_text) > 30:
                # Log the actual content for debugging
                logger.info(f"✓ Raw output ({len(generated_text)} chars): {generated_text[:200]}...")
                
                parsed = self._parse_structured_output(generated_text)
                logger.info(f"✓ Parsed sections: {[k for k, v in parsed.items() if v][:3]}")
                
                # Accept if we have any content in clinical_summary or primary_diagnosis
                has_content = (
                    (parsed.get("clinical_summary") and len(parsed["clinical_summary"].strip()) > 10) or
                    (parsed.get("primary_diagnosis") and len(parsed["primary_diagnosis"].strip()) > 5)
                )
                
                if has_content:
                    logger.info(f"✓ Using parsed assessment")
                    return parsed
                else:
                    # If parsing failed but we have content, split text intelligently
                    logger.info(f"✓ Using intelligent text split")
                    return self._intelligent_text_split(generated_text)
            
            logger.warning(f"⚠️ Output too short ({len(generated_text)} chars), using fallback")
            return self._generate_fallback_response(task_type, input_data)
                
        except Exception as e:
            if e.__class__.__name__ == "GenerationTimeout":
                # Let evaluation timeout handling decide the fallback behavior.
                raise
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
        """Build task-aware, grounded prompt with strict JSON output schema."""
        text_content = input_data.get("text_content", "")
        strict_grounding = bool(input_data.get("strict_grounding", False))
        has_text = bool(text_content and text_content.strip())

        task_cfg = {
            "ct_coronary": {
                "role": "cardiologist",
                "task": "coronary CT interpretation",
                "rules": (
                    "- Prioritize left main/proximal LAD risk and symptom burden.\n"
                    "- State urgency for cardiology/invasive evaluation when high-risk anatomy is present.\n"
                    "- Do not claim ACS/MI unless explicitly stated."
                ),
            },
            "breast_imaging": {
                "role": "breast radiologist",
                "task": "breast imaging assessment",
                "rules": (
                    "- Use BI-RADS language when available.\n"
                    "- Distinguish BI-RADS 3 surveillance from BI-RADS 4/5 biopsy pathway.\n"
                    "- Do not assign definitive cancer diagnosis without pathology."
                ),
            },
            "lipid_profile": {
                "role": "preventive cardiology specialist",
                "task": "lipid and cardiometabolic risk assessment",
                "rules": (
                    "- Use available LDL/HDL/triglycerides and comorbidity risk factors.\n"
                    "- Focus on risk reduction strategy and follow-up interval.\n"
                    "- Avoid unsupported medication dosing details."
                ),
            },
            "biopsy_report": {
                "role": "pathologist-oncology consultant",
                "task": "pathology-driven oncologic assessment",
                "rules": (
                    "- Use histology, grade, receptors, margins, and adverse features if present.\n"
                    "- Differentiate in-situ from invasive disease.\n"
                    "- Keep recommendations at planning level; avoid unsupported staging."
                ),
            },
        }
        cfg = task_cfg.get(
            task_type,
            {
                "role": "clinical specialist",
                "task": "medical case assessment",
                "rules": "- Use only available evidence and avoid unsupported assumptions.",
            },
        )

        strict_clause = (
            "Grounding rule: only use facts explicitly present in the context. "
            "If missing, write 'insufficient evidence'."
        )
        if strict_grounding:
            strict_clause += " Strict mode: do not infer severity, treatment, or timing beyond stated evidence."

        context_block = text_content[:900] if has_text else "No additional context provided."
        json_schema = (
            '{\n'
            '  "clinical_summary": "string",\n'
            '  "primary_diagnosis": "string",\n'
            '  "treatment_plan": "string",\n'
            '  "follow_up": "string",\n'
            '  "urgency": "routine|expedited|urgent",\n'
            '  "confidence": "low|medium|high",\n'
            '  "evidence_snippets": "short quoted facts from context, separated by ;",\n'
            '  "red_flags": "string",\n'
            '  "next_best_test": "string"\n'
            '}'
        )

        return f"""You are an expert {cfg["role"]}. Perform a {cfg["task"]}.

Task-specific guidance:
{cfg["rules"]}

Clinical context:
{context_block}

Output requirement:
- Return EXACTLY one JSON object, no markdown and no extra text.
- Use concise clinical language.
- If evidence is missing for a field, set the value to "insufficient evidence".
- {strict_clause}

Required JSON schema:
{json_schema}
"""
    
    def _parse_structured_output(self, text: str) -> Dict[str, str]:
        """Parse model output, preferring strict JSON then legacy header parsing."""
        result = self._empty_assessment()
        
        if not text:
            return result

        json_parsed = self._parse_json_assessment(text)
        if json_parsed:
            return json_parsed
        
        sections = {
            "clinical_summary": ["SUMMARY:", "CLINICAL SUMMARY:", "ASSESSMENT:"],
            "primary_diagnosis": ["DIAGNOSIS:", "PRIMARY DIAGNOSIS:", "FINDINGS:"],
            "differentials": ["DIFFERENTIALS:", "DIFFERENTIAL DIAGNOSES:", "DIFFERENTIAL DIAGNOSIS:"],
            "treatment_plan": ["TREATMENT:", "TREATMENT PLAN:", "RECOMMENDATIONS:"],
            "lifestyle_recommendations": ["LIFESTYLE:", "LIFESTYLE RECOMMENDATIONS:", "LIFESTYLE CHANGES:"],
            "follow_up": ["FOLLOWUP:", "FOLLOW-UP:", "FOLLOW UP:", "NEXT STEPS:", "FOLLOW-UP PLAN:"]
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
            # No headers found, use intelligent split
            logger.info("  No section headers found, using intelligent split")
            return self._intelligent_text_split(text)
        
        sorted_sections = sorted(positions.items(), key=lambda x: x[1][0])
        
        for i, (key, (start, end)) in enumerate(sorted_sections):
            if i + 1 < len(sorted_sections):
                next_start = sorted_sections[i + 1][1][0]
                content = text[end:next_start]
            else:
                content = text[end:]
            
            # Clean up the content
            content = re.sub(r'^[:\s]+', '', content).strip()
            
            # Remove markdown artifacts and formatting
            content = re.sub(r'\*+', '', content)  # Remove asterisks
            content = re.sub(r'\b(SUMMARY|DIAGNOSIS|TREATMENT|FOLLOWUP|IMPRESSION|FINDINGS):\s*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'^[-•]\s*', '', content, flags=re.MULTILINE)  # Remove bullet points
            content = re.sub(r'\s+', ' ', content).strip()
            
            result[key] = content
        
        return self._normalize_assessment(result)

    def _empty_assessment(self) -> Dict[str, str]:
        return {
            "clinical_summary": "",
            "primary_diagnosis": "",
            "differentials": "",
            "treatment_plan": "",
            "lifestyle_recommendations": "",
            "follow_up": "",
            "urgency": "",
            "confidence": "",
            "evidence_snippets": "",
            "red_flags": "",
            "next_best_test": "",
        }

    def _normalize_assessment(self, data: Dict[str, Any]) -> Dict[str, str]:
        result = self._empty_assessment()
        alias_map = {
            "clinical_summary": "clinical_summary",
            "summary": "clinical_summary",
            "primary_diagnosis": "primary_diagnosis",
            "diagnosis": "primary_diagnosis",
            "treatment_plan": "treatment_plan",
            "treatment": "treatment_plan",
            "follow_up": "follow_up",
            "followup": "follow_up",
            "urgency": "urgency",
            "confidence": "confidence",
            "evidence_snippets": "evidence_snippets",
            "evidence": "evidence_snippets",
            "red_flags": "red_flags",
            "next_best_test": "next_best_test",
            "differentials": "differentials",
            "lifestyle_recommendations": "lifestyle_recommendations",
        }

        for src_key, value in data.items():
            key = alias_map.get(str(src_key).strip().lower())
            if not key:
                continue
            text_value = str(value).strip()
            text_value = re.sub(r"\s+", " ", text_value)
            result[key] = text_value

        if not result["clinical_summary"]:
            result["clinical_summary"] = "insufficient evidence"
        if not result["primary_diagnosis"]:
            result["primary_diagnosis"] = "insufficient evidence"
        if not result["treatment_plan"]:
            result["treatment_plan"] = "insufficient evidence"
        if not result["follow_up"]:
            result["follow_up"] = "insufficient evidence"
        if result["urgency"] not in {"routine", "expedited", "urgent"}:
            result["urgency"] = "expedited"
        if result["confidence"] not in {"low", "medium", "high"}:
            result["confidence"] = "medium"
        return result

    def _extract_json_object(self, text: str) -> str:
        clean = text.strip()
        code_block = re.search(r"```(?:json)?\s*({[\s\S]*?})\s*```", clean, flags=re.IGNORECASE)
        if code_block:
            return code_block.group(1)
        first = clean.find("{")
        last = clean.rfind("}")
        if first != -1 and last != -1 and last > first:
            return clean[first:last + 1]
        return ""

    def _parse_json_assessment(self, text: str) -> Optional[Dict[str, str]]:
        blob = self._extract_json_object(text)
        if not blob:
            return None
        try:
            parsed = json.loads(blob)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        return self._normalize_assessment(parsed)
    
    def _generate_fallback_response(self, task_type: str, input_data: Dict[str, Any]) -> Dict[str, str]:
        """Fallback response."""
        return self._normalize_assessment({
            "clinical_summary": f"Assessment for {task_type}.",
            "primary_diagnosis": "Error in analysis. Consult specialist.",
            "differentials": "Requires evaluation",
            "treatment_plan": "Specialist referral recommended",
            "lifestyle_recommendations": "Healthy lifestyle",
            "follow_up": "Schedule with provider"
        })
    
    def _intelligent_text_split(self, text: str) -> Dict[str, str]:
        """Intelligently split text into sections when parser fails."""
        # Clean up the text
        text = text.strip()
        
        result = self._empty_assessment()
        
        if not text:
            result["clinical_summary"] = "No assessment generated"
            return result
        
        # Split by newlines first to preserve structure
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        if len(lines) <= 2:
            # Short text - split into sentences
            sentences = [s.strip() for s in text.split('. ') if s.strip()]
            
            if len(sentences) >= 4:
                # Distribute across sections
                result["clinical_summary"] = sentences[0] + "."
                result["primary_diagnosis"] = sentences[1] + "."
                result["treatment_plan"] = ". ".join(sentences[2:-1]) + "."
                result["follow_up"] = sentences[-1] + "."
            elif len(sentences) >= 2:
                result["clinical_summary"] = sentences[0] + "."
                result["primary_diagnosis"] = ". ".join(sentences[1:])
            else:
                result["clinical_summary"] = text
        else:
            # Multi-line text - use first part as summary, distribute rest
            result["clinical_summary"] = lines[0]
            
            # Look for specific patterns in remaining lines
            remaining_text = " ".join(lines[1:])
            
            # Try to identify diagnosis (often contains anatomical terms or findings)
            if any(term in remaining_text.lower() for term in ['artery', 'vessel', 'mass', 'lesion', 'calcification', 'stenosis', 'occlusion']):
                # Split remaining into roughly equal parts
                parts = remaining_text.split('. ')
                mid = len(parts) // 3
                
                if len(parts) >= 3:
                    result["primary_diagnosis"] = ". ".join(parts[:mid]) + "."
                    result["treatment_plan"] = ". ".join(parts[mid:2*mid]) + "."
                    result["follow_up"] = ". ".join(parts[2*mid:]) + "."
                else:
                    result["primary_diagnosis"] = remaining_text
            else:
                result["primary_diagnosis"] = remaining_text[:300]
        
        # Ensure no section is empty
        if not result["treatment_plan"]:
            result["treatment_plan"] = "Consult specialist for treatment recommendations"
        if not result["follow_up"]:
            result["follow_up"] = "Schedule follow-up with healthcare provider"
        
        return self._normalize_assessment(result)
    
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
