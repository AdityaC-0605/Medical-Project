"""
MedGemma Client for Medical Diagnosis
Supports text-only and multimodal (image + text) inference on MPS (Apple Silicon)
"""

import logging
import time
import os
import yaml
from typing import Optional

logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config.yaml"
    )
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load config.yaml: {e}. Using defaults.")
        return None


class MedGemmaClient:
    """
    Client for google/medgemma-1.5-4b-it
    Optimized for Apple Silicon MPS acceleration.
    """

    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._loaded = False
        self._device = None

        # Load config
        config = load_config()
        model_cfg = (config or {}).get("model", {})
        self.max_new_tokens = model_cfg.get("max_new_tokens", 256)
        self.hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

        logger.info(f"MedGemmaClient initialized (model={self.model_id})")

    # ------------------------------------------------------------------
    def load(self) -> bool:
        """Load model and processor."""
        if self._loaded:
            return True

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText

            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            # Resolve device: MPS > CUDA > CPU
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

            logger.info("Loading MedGemma processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=self.hf_token,
            )

            # Load in float16 as requested
            dtype = torch.float16
            
            logger.info(f"Loading model in {dtype} on {self._device} (using device_map)...")
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map=str(self._device),  # Direct load
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    token=self.hf_token,
                )
            except Exception as load_err:
                logger.error(f"Failed to load in {dtype}: {load_err}")
                raise load_err

            self.model.eval()

            # Override the model's generation_config so our settings win.
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = 1.0
            self.model.generation_config.top_p = 1.0

            self._loaded = True
            logger.info(f"✅ MedGemma loaded on {self._device} ({dtype})")
            return True

        except Exception as e:
            logger.exception("❌ Failed to load MedGemma")
            return False

    # ------------------------------------------------------------------
    def generate_diagnosis(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate a medical diagnosis from text (and optionally an image)."""
        if not self._loaded and not self.load():
            raise RuntimeError("MedGemma model failed to load")

        import torch
        from PIL import Image

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # --- build chat messages -----------------------------------------
        full_prompt = f"""
            You are a medical AI assistant.

            Analyze the clinical case below and provide a concise, professional medical assessment.

            INSTRUCTIONS:
            - Do NOT repeat the input text
            - Do NOT mention being an AI
            - Use clear medical terminology
            - Be factual and cautious
            - If information is insufficient, state reasonable assumptions

            OUTPUT FORMAT (strict):
            1. Clinical Summary (2–3 sentences)
            2. Key Findings
            3. Likely Diagnosis
            4. Risk Assessment
            5. Suggested Next Steps

            CLINICAL CASE:
            {prompt}
            """.strip()
        has_image = image_path and os.path.exists(image_path)

        if has_image:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_prompt},
                ],
            }]
        else:
            messages = [{"role": "user", "content": full_prompt}]

        # --- tokenise with chat template ---------------------------------
        chat_text = (
            "<bos><start_of_turn>user\n"
            f"{full_prompt}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n\n"
        )

        # Ensure Gemma-style turn markers are present
        if "<start_of_turn>" not in chat_text:
            chat_text = (
                "<bos><start_of_turn>user\n"
                f"{full_prompt}\n"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
            )

        # --- prepare model inputs ----------------------------------------
        if has_image:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(
                text=chat_text,
                images=image,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self._device)
        else:
            inputs = self.processor(
                text=chat_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self._device)

        # --- generate ----------------------------------------------------
        logger.info(f"Generating diagnosis (max_tokens={max_new_tokens})...")
        start = time.time()
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # NOTE: no repetition_penalty — torch.gather is broken/slow on MPS
            )
            
        decoded = self.processor.decode(output_ids[0], skip_special_tokens=True)
        logger.debug(f"Raw output:\n{decoded}")

        response = self._extract_response(decoded)
        response = self._post_process(response)

        elapsed = time.time() - start
        logger.info(f"✅ Generated {len(response)} chars in {elapsed:.1f}s")
        return response

    # ------------------------------------------------------------------
    def _extract_response(self, text: str) -> str:
        """Pull out only the model's reply from the decoded string."""
        text = text.strip()

        # Gemma format: everything after the last "model\n"
        if "model\n" in text:
            return text.split("model\n")[-1].strip()

        # Fallback: after last "model" keyword
        idx = text.rfind("model")
        if idx != -1:
            candidate = text[idx + 5:].lstrip(": \n")
            if candidate:
                return candidate

        return text

    # ------------------------------------------------------------------
    def _post_process(self, response: str) -> str:
        """Add disclaimer if missing."""
        response = response.strip()
        disclaimer = (
            "*Disclaimer: AI-generated assessment for educational purposes. "
            "Consult healthcare professionals.*"
        )
        if "disclaimer" not in response.lower():
            response += f"\n\n---\n{disclaimer}"
        return response

    # ------------------------------------------------------------------
    def cleanup(self):
        """Free model memory."""
        try:
            import torch
            import gc

            del self.model
            del self.processor
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self.model = None
        self.processor = None
        self._loaded = False
        logger.info("MedGemma resources cleaned up")