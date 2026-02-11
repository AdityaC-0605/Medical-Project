"""
MedGemma Client for Medical Diagnosis
Supports text-only and multimodal (image + text) inference on MPS (Apple Silicon)
"""

import logging
import os
import time
import yaml
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Config loader
# ---------------------------------------------------------
def load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config.yaml"
    )
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load config.yaml: {e}. Using defaults.")
        return {}


# ---------------------------------------------------------
# MedGemma Client
# ---------------------------------------------------------
class MedGemmaClient:
    """
    Client for google/medgemma-1.5-4b-it
    Optimized for Apple Silicon (MPS)
    """

    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._loaded = False
        self._device = None

        config = load_config()
        model_cfg = config.get("model", {})
        self.max_new_tokens = model_cfg.get("max_new_tokens", 256)
        self.hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

        logger.info(f"MedGemmaClient initialized (model={self.model_id})")

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------
    def load(self) -> bool:
        if self._loaded:
            return True

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText

            os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

            logger.info("Loading MedGemma model on %s (float16)...", self._device)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map=str(self._device),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=self.hf_token,
            )

            self.model.eval()

            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = 1.0
            self.model.generation_config.top_p = 1.0

            self._loaded = True
            logger.info("✅ MedGemma loaded successfully")
            return True

        except Exception:
            logger.exception("❌ Failed to load MedGemma")
            return False

    # -----------------------------------------------------
    # Generate diagnosis
    # -----------------------------------------------------
    
    def generate_diagnosis(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        if not self._loaded and not self.load():
            raise RuntimeError("MedGemma model failed to load")

        import torch
        from PIL import Image

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        has_image = image_path and os.path.exists(image_path)

        # -----------------------------
        # MEDGEMMA-CORRECT PROMPT STYLE
        # -----------------------------
        # IMPORTANT:
        # - No instructions
        # - No "Analyze..."
        # - Start the medical note directly
        clinical_text = (
            f"Clinical case:\n{prompt}\n\n"
            "Clinical assessment:\n"
        )

        # -----------------------------
        # TEXT ONLY
        # -----------------------------
        if not has_image:
            inputs = self.processor(
                text=clinical_text,
                return_tensors="pt",
            ).to(self._device)

        # -----------------------------
        # IMAGE + TEXT (CORRECT WAY)
        # -----------------------------
        else:
            image = Image.open(image_path).convert("RGB")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": clinical_text},
                ],
            }]

            # Apply chat template to convert messages to string
            prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=prompt_text,
                images=image,
                return_tensors="pt",
            ).to(self._device)

        # -----------------------------
        # GENERATION
        # -----------------------------
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # -----------------------------
        # DECODE ONLY NEW TOKENS
        # -----------------------------
        input_len = inputs.input_ids.shape[1]
        generated_tokens = output_ids[0][input_len:]

        decoded = self.processor.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        if not decoded:
            decoded = "Insufficient information to provide a definitive clinical assessment."

        decoded += (
            "\n\n---\n"
            "*Disclaimer: AI-generated assessment for educational purposes. "
            "Consult healthcare professionals.*"
        )

        return decoded

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------
    def cleanup(self):
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