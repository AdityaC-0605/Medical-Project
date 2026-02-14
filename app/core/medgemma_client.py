"""
MedGemma Client for Medical Diagnosis
Supports text-only and multimodal (image + text) inference on MPS (Apple Silicon)
Generates structured clinical output including diagnosis and treatment plan.
"""

import logging
import os
import json
import re
from typing import Optional, Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)


class MedGemmaClient:
    """
    Client for google/medgemma-1.5-4b-it
    Optimized for Apple Silicon (MPS)
    Generates structured clinical assessments with diagnosis and treatment plans.
    """

    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._loaded = False
        self._device = None
        self.hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

        logger.info(f"MedGemmaClient initialized (model={self.model_id})")

    def load(self) -> bool:
        """Load model and processor."""
        if self._loaded:
            return True

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText

            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            # Determine device
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

            logger.info(f"Loading MedGemma on {self._device}...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=self.hf_token,
            )

            # Load model - use float32 for MPS stability, float16 for CUDA
            if str(self._device) == "mps":
                torch_dtype = torch.float32  # MPS works better with float32
            else:
                torch_dtype = torch.float16
                
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map="auto",  # Let transformers handle device placement
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=self.hf_token,
            )

            self.model.eval()
            self._loaded = True
            logger.info("MedGemma loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load MedGemma: {e}")
            return False

    def generate_structured_assessment(
        self,
        task_type: str,
        input_data: Dict[str, Any],
        max_new_tokens: int = 512
    ) -> Dict[str, str]:
        """
        Generate structured clinical assessment with diagnosis and treatment plan.
        
        Args:
            task_type: Type of medical task (ct_coronary, breast_imaging, etc.)
            input_data: Dictionary containing text_content, image_path, etc.
            max_new_tokens: Maximum tokens to generate (default 512 for Mac stability)
            
        Returns:
            Dictionary with clinical_summary, primary_diagnosis, differentials, 
            treatment_plan, and follow_up
        """
        if not self._loaded and not self.load():
            return self._generate_fallback_response(task_type, input_data)

        import torch

        try:
            # Build structured prompt
            prompt = self._build_structured_prompt(task_type, input_data)
            
            # Check for image
            image_path = input_data.get("image_path")
            has_image = image_path and os.path.exists(image_path)
            
            logger.info(f"Generating structured assessment for {task_type}")
            logger.info(f"Image present: {has_image}")
            logger.info(f"Max tokens: {max_new_tokens}")

            if has_image:
                # Process with image - let processor handle image token automatically
                image = Image.open(image_path).convert("RGB")
                
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                )
            else:
                # Text only
                inputs = self.processor(
                    text=prompt,
                    return_tensors="pt",
                )

            # Move to device
            inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}

            # Memory-efficient generation for Mac
            is_mps = str(self._device) == "mps"
            
            with torch.no_grad():
                if is_mps:
                    # Conservative settings for MPS to prevent crashes
                    logger.info("Using MPS-optimized generation")
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=min(max_new_tokens, 512),  # Limit tokens on MPS
                        do_sample=False,  # Greedy for stability
                        num_beams=1,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                    )
                else:
                    # Better quality on CUDA/CPU with sampling
                    logger.info("Using sampling for generation")
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                    )

            # Decode output
            generated_text = self._decode_output(inputs, output_ids)
            
            # Debug logging
            logger.info(f"Raw generated text length: {len(generated_text)}")
            if generated_text:
                logger.info(f"First 200 chars: {generated_text[:200]}")
                logger.info(f"Last 200 chars: {generated_text[-200:]}")
            
            if generated_text and len(generated_text) > 50:
                logger.info(f"Generated {len(generated_text)} characters")
                # Parse structured output
                return self._parse_structured_output(generated_text)
            else:
                logger.warning("Empty or too short response from model, using fallback")
                return self._generate_fallback_response(task_type, input_data)

        except Exception as e:
            logger.error(f"Generation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._generate_fallback_response(task_type, input_data)

    def _build_structured_prompt(self, task_type: str, input_data: Dict[str, Any]) -> str:
        """Build optimized clinical prompt for MedGemma."""
        
        # Get clinical context
        text_content = input_data.get("text_content", "")
        
        # Build case-specific context
        case_context = self._build_case_context(task_type, input_data)
        
        # Simplified but effective prompt
        prompt = f"""As a medical specialist, analyze this case and provide a detailed clinical assessment.

CASE INFORMATION:
{case_context}

CLINICAL NOTES:
{text_content if text_content else "No additional notes provided."}

Provide a comprehensive assessment with these sections:

CLINICAL SUMMARY: Summarize the key findings, patient presentation, and clinical significance.

PRIMARY DIAGNOSIS: State the most likely diagnosis with severity level and explain your reasoning.

DIFFERENTIAL DIAGNOSES: List 2-3 alternative diagnoses with supporting and opposing evidence for each.

TREATMENT PLAN: Recommend specific treatments including medication names, exact doses, frequency, and duration.

LIFESTYLE RECOMMENDATIONS: Detail dietary changes, exercise protocols, and risk factor modifications.

FOLLOW-UP PLAN: Specify timeline for follow-up, monitoring parameters, and when to seek urgent care.

Be specific, thorough, and use evidence-based medical guidelines."""
        
        return prompt

    def _build_case_context(self, task_type: str, input_data: Dict[str, Any]) -> str:
        """Build concise case-specific context."""
        
        if task_type == "ct_coronary":
            vessel = input_data.get("vessel", "unspecified")
            stenosis = input_data.get("stenosis_percent", "unknown")
            finding = input_data.get("finding", "")
            age = input_data.get("age", "unknown")
            
            context = f"Cardiac CT showing {stenosis}% stenosis in the {vessel}."
            if finding:
                context += f" Additional findings: {finding}."
            if age != "unknown":
                context += f" Patient age: {age}."
            return context
        
        elif task_type == "breast_imaging":
            birads = input_data.get("birads_category", "")
            finding = input_data.get("finding", "")
            modality = input_data.get("imaging_modality", "mammogram")
            age = input_data.get("age", input_data.get("patient_age", ""))
            
            context = f"Breast {modality} examination."
            if birads:
                context += f" BI-RADS assessment: {birads}."
            if finding:
                context += f" Findings: {finding}."
            if age:
                context += f" Patient age: {age}."
            return context
        
        elif task_type == "lipid_profile":
            ldl = input_data.get("ldl", "unknown")
            hdl = input_data.get("hdl", "unknown")
            tg = input_data.get("triglycerides", "unknown")
            total = input_data.get("total_cholesterol", "unknown")
            age = input_data.get("age", "unknown")
            
            values = []
            if ldl != "unknown":
                values.append(f"LDL {ldl}")
            if hdl != "unknown":
                values.append(f"HDL {hdl}")
            if tg != "unknown":
                values.append(f"TG {tg}")
            if total != "unknown":
                values.append(f"Total {total}")
            
            if values:
                return f"Lipid panel: {', '.join(values)} mg/dL. Age: {age}."
            else:
                return f"Lipid panel evaluation. Age: {age}."
        
        elif task_type == "biopsy_report":
            report_text = input_data.get("report_text", "")[:300]
            specimen = input_data.get("specimen_type", "")
            
            if specimen and report_text:
                return f"{specimen} pathology. {report_text}"
            elif report_text:
                return f"Pathology report: {report_text}"
            else:
                return "Pathology specimen review."
        
        else:
            finding = input_data.get("finding", input_data.get("text_content", ""))
            if finding:
                return finding[:400]
            else:
                return "General medical assessment."

    def _decode_output(self, inputs, output_ids) -> str:
        """Decode model output with robust error handling."""
        try:
            import torch
            
            # Get input length
            if hasattr(inputs, 'input_ids'):
                input_len = inputs.input_ids.shape[1]
            elif isinstance(inputs, dict) and 'input_ids' in inputs:
                input_len = inputs['input_ids'].shape[1]
            else:
                logger.warning("Could not determine input length, using full output")
                input_len = 0
            
            # Extract generated tokens (skip the input prompt)
            generated_ids = output_ids[0][input_len:]
            
            # Ensure we have tokens to decode
            if generated_ids.numel() == 0:
                logger.warning("No generated tokens to decode")
                return ""
            
            logger.info(f"Generated token IDs shape: {generated_ids.shape}")
            
            # Convert to CPU if needed for decoding
            if generated_ids.is_mps:
                generated_ids = generated_ids.cpu()
            
            # Decode with skip_special_tokens=True (cleaner output)
            text = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            # Clean up any remaining artifacts
            text = text.strip()
            
            if not text:
                logger.warning("Empty text after standard decoding")
                # Try with skip_special_tokens=False as fallback
                text_raw = self.processor.decode(generated_ids, skip_special_tokens=False)
                logger.info(f"Raw decoded (with special tokens): {text_raw[:100]}")
                text = text_raw.strip()
            
            # Remove common artifacts
            text = text.replace("<end_of_turn>", "").replace("<start_of_turn>", "").strip()
            
            return text
        except Exception as e:
            logger.error(f"Decode error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def _parse_structured_output(self, text: str) -> Dict[str, str]:
        """Parse structured output from model."""
        result = {
            "clinical_summary": "",
            "primary_diagnosis": "",
            "differentials": "",
            "treatment_plan": "",
            "lifestyle_recommendations": "",
            "follow_up": ""
        }
        
        # Parse sections
        sections = {
            "clinical_summary": ["CLINICAL SUMMARY:", "1. CLINICAL SUMMARY", "Clinical Summary:"],
            "primary_diagnosis": ["PRIMARY DIAGNOSIS:", "2. PRIMARY DIAGNOSIS", "Primary Diagnosis:"],
            "differentials": ["DIFFERENTIAL DIAGNOSES:", "3. DIFFERENTIAL", "Differential Diagnoses:"],
            "treatment_plan": ["TREATMENT PLAN:", "4. TREATMENT", "Treatment Plan:"],
            "lifestyle_recommendations": ["LIFESTYLE RECOMMENDATIONS:", "5. LIFESTYLE", "Lifestyle Recommendations:"],
            "follow_up": ["FOLLOW-UP PLAN:", "6. FOLLOW-UP", "Follow-up Plan:", "FOLLOW UP:"]
        }
        
        text_upper = text.upper()
        
        for key, headers in sections.items():
            for header in headers:
                if header.upper() in text_upper:
                    # Find the section
                    start_idx = text_upper.find(header.upper())
                    if start_idx != -1:
                        # Find end (next section or end of text)
                        end_idx = len(text)
                        for other_key, other_headers in sections.items():
                            if other_key != key:
                                for other_header in other_headers:
                                    other_idx = text_upper.find(other_header.upper(), start_idx + len(header))
                                    if other_idx != -1 and other_idx < end_idx:
                                        end_idx = other_idx
                        
                        # Extract content
                        content = text[start_idx + len(header):end_idx].strip()
                        # Clean up
                        content = re.sub(r'^[:\s]+', '', content)
                        result[key] = content
                        break
        
        # If parsing failed, use the whole text as summary
        if not any(result.values()):
            result["clinical_summary"] = text
        
        return result

    def _generate_fallback_response(self, task_type: str, input_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate fallback structured response."""
        
        if task_type == "ct_coronary":
            return {
                "clinical_summary": "Cardiac CT showing coronary artery disease findings requiring further evaluation.",
                "primary_diagnosis": "Coronary Artery Disease - Severity to be determined by cardiologist",
                "differentials": "1. Stable angina\n2. Acute coronary syndrome\n3. Microvascular angina",
                "treatment_plan": "1. Aspirin 81mg daily\n2. Atorvastatin 40mg daily\n3. Beta-blocker as indicated\n4. Cardiology referral for further evaluation",
                "lifestyle_recommendations": "1. Heart-healthy diet (Mediterranean/DASH)\n2. Regular exercise 150 min/week\n3. Smoking cessation if applicable\n4. Blood pressure control <130/80",
                "follow_up": "Cardiology appointment within 2-4 weeks. Repeat lipid panel in 3 months."
            }
        
        elif task_type == "breast_imaging":
            return {
                "clinical_summary": "Breast imaging study showing findings requiring specialist correlation.",
                "primary_diagnosis": "Breast imaging abnormality - BI-RADS assessment required",
                "differentials": "1. Benign lesion\n2. Fibrocystic changes\n3. Suspicious mass requiring biopsy",
                "treatment_plan": "1. Referral to breast specialist\n2. Possible biopsy based on BI-RADS\n3. Imaging correlation with prior studies",
                "lifestyle_recommendations": "1. Monthly breast self-examination\n2. Maintain healthy weight\n3. Limit alcohol consumption\n4. Regular exercise",
                "follow_up": "Breast imaging specialist within 1-2 weeks. BI-RADS follow-up per guidelines."
            }
        
        elif task_type == "lipid_profile":
            return {
                "clinical_summary": "Lipid panel showing dyslipidemia requiring lifestyle and possible pharmacologic intervention.",
                "primary_diagnosis": "Dyslipidemia - mixed hyperlipidemia pattern",
                "differentials": "1. Primary hyperlipidemia\n2. Secondary causes (hypothyroidism, diabetes)\n3. Familial hypercholesterolemia",
                "treatment_plan": "1. Atorvastatin 20-40mg daily\n2. Dietary modifications\n3. Consider ezetimibe if needed\n4. Omega-3 if TG elevated",
                "lifestyle_recommendations": "1. Mediterranean diet\n2. Reduce saturated fat <7% calories\n3. Exercise 150 min/week\n4. Weight loss if BMI >25",
                "follow_up": "Lipid panel in 4-12 weeks, then every 3-6 months. LFTs if on statin."
            }
        
        elif task_type == "biopsy_report":
            return {
                "clinical_summary": "Pathology report showing findings requiring oncology evaluation.",
                "primary_diagnosis": "Biopsy-proven pathology - specific diagnosis pending full pathology review",
                "differentials": "1. Benign process\n2. Premalignant lesion\n3. Malignant neoplasm",
                "treatment_plan": "1. Oncology referral\n2. Staging workup as indicated\n3. Multidisciplinary tumor board review\n4. Treatment planning",
                "lifestyle_recommendations": "1. Maintain nutrition\n2. Regular physical activity\n3. Emotional support/counseling\n4. Smoking cessation if applicable",
                "follow_up": "Urgent oncology consultation within 1-2 weeks. Staging studies as indicated."
            }
        
        else:
            return {
                "clinical_summary": "Clinical assessment based on available information.",
                "primary_diagnosis": "Diagnosis pending - clinical correlation required",
                "differentials": "1. Primary diagnosis\n2. Secondary causes\n3. Differential diagnoses to be considered",
                "treatment_plan": "1. Specialist referral\n2. Additional diagnostic testing\n3. Symptomatic management",
                "lifestyle_recommendations": "1. Healthy diet\n2. Regular exercise\n3. Adequate sleep\n4. Stress management",
                "follow_up": "Follow-up with primary care physician within 1-2 weeks."
            }

    def generate_text(self, prompt: str, image_path: Optional[str] = None, max_new_tokens: int = 128) -> str:
        """
        Simple text generation for classification tasks.
        Returns raw text response (not structured).
        """
        if not self._loaded and not self.load():
            return ""

        import torch

        try:
            has_image = image_path and os.path.exists(image_path)

            if has_image:
                # Process with image
                image = Image.open(image_path).convert("RGB")
                
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                )
            else:
                # Text only
                inputs = self.processor(
                    text=prompt,
                    return_tensors="pt",
                )

            # Move to device
            inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}

            logger.info(f"Generating text (max_tokens={max_new_tokens})")
            
            # Memory-efficient generation for Mac
            is_mps = str(self._device) == "mps"
            
            with torch.no_grad():
                if is_mps:
                    # Conservative settings for MPS stability
                    logger.info("Using MPS-optimized generation")
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=min(max_new_tokens, 256),  # Limit on MPS
                        do_sample=False,  # Greedy for stability
                        num_beams=1,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                    )
                else:
                    # Better quality on CUDA/CPU
                    logger.info("Using sampling for generation")
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                    )

            # Decode output
            generated_text = self._decode_output(inputs, output_ids)
            
            # Debug logging
            logger.info(f"Raw text length: {len(generated_text)}")
            if generated_text:
                logger.info(f"Response preview: {generated_text[:150]}...")
            else:
                logger.warning("Empty response from model")
                
            return generated_text

        except Exception as e:
            logger.error(f"Text generation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def cleanup(self):
        """Clean up resources aggressively to free memory."""
        try:
            import torch
            import gc

            # Move model to CPU first (helps with MPS)
            if self.model is not None:
                try:
                    self.model = self.model.cpu()
                except:
                    pass
                del self.model
                self.model = None
                
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            # Force garbage collection
            gc.collect()
            
            # Clear MPS cache if available
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                    torch.mps.synchronize()  # Wait for MPS operations to complete
                except Exception as e:
                    logger.debug(f"MPS cache clear: {e}")
                    
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception as e:
                    logger.debug(f"CUDA cache clear: {e}")
                
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

        self._loaded = False
        logger.info("MedGemma resources cleaned up")
