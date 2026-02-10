"""
MedGemma Client for Medical Diagnosis
Supports text-only and multimodal (image + text) inference
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
    """

    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._loaded = False

        # Load config
        config = load_config()
        model_cfg = (config or {}).get("model", {})

        self.device = model_cfg.get("device", "auto")
        self.torch_dtype_str = model_cfg.get("torch_dtype", "float16")
        self.max_new_tokens = model_cfg.get("max_new_tokens", 800)

        gen_cfg = model_cfg.get("generation", {})
        self.do_sample = gen_cfg.get("do_sample", False)
        self.repetition_penalty = gen_cfg.get("repetition_penalty", 1.1)
        
        # Get HuggingFace token from environment
        self.hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

        logger.info(
            f"MedGemmaClient initialized "
            f"(model={self.model_id}, device={self.device})"
        )

    # ------------------------------------------------------------------
    def load(self) -> bool:
        """Load model and processor (lazy loading)."""
        if self._loaded:
            return True

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText

            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            logger.info("Loading MedGemma processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=self.hf_token
            )

            # Device selection
            if self.device == "auto":
                if torch.cuda.is_available():
                    target_device = "cuda"
                elif torch.backends.mps.is_available():
                    target_device = "mps"
                else:
                    target_device = "cpu"
            else:
                target_device = self.device

            # Dtype selection
            # MPS: use float16 with CPU-first loading + greedy decoding
            # (float32 is too slow on MPS, float16+sampling=gibberish)
            if self.torch_dtype_str == "float16":
                torch_dtype = torch.float16
            elif self.torch_dtype_str == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
            dtype_name = str(torch_dtype).split('.')[-1]

            # MPS strategy: load on CPU first, then move to MPS
            # This bypasses caching_allocator_warmup which crashes with 16GB buffer
            if target_device == "mps":
                logger.info(f"Loading MedGemma on CPU ({dtype_name}), then moving to MPS...")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    device_map="cpu",
                    dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    token=self.hf_token
                )
                logger.info("Moving model to MPS...")
                self.model = self.model.to("mps")
            else:
                logger.info(f"Loading MedGemma on {target_device} ({dtype_name})...")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    device_map=target_device,
                    dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    token=self.hf_token
                )

            self._loaded = True
            logger.info("✅ MedGemma model loaded successfully")
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
        """Generate diagnosis using MedGemma."""
        if not self._loaded and not self.load():
            raise RuntimeError("MedGemma model failed to load")

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        import torch
        from PIL import Image

        # Build prompt
        full_prompt = self._build_medical_prompt(prompt)

        messages = self._prepare_messages(full_prompt, image_path)

        # Apply chat template
        try:
            chat_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        except Exception:
            chat_prompt = ""

        # Fallback Gemma format
        if "<start_of_turn>" not in chat_prompt:
            if isinstance(messages[0]["content"], list):
                user_text = next(
                    x["text"] for x in messages[0]["content"]
                    if x["type"] == "text"
                )
            else:
                user_text = messages[0]["content"]

            # Manually construct prompt with BOS
            chat_prompt = (
                "<bos><start_of_turn>user\n"
                f"{user_text}"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
        elif not chat_prompt.startswith("<bos>"):
            # Ensure BOS is present if template didn't add it
            chat_prompt = "<bos>" + chat_prompt

        # Ensure image token structure for PaliGemma/MedGemma
        if image_path and os.path.exists(image_path) and "<image>" not in chat_prompt:
            # Prepend image token after BOS if BOS exists, else at start
            if chat_prompt.startswith("<bos>"):
                chat_prompt = chat_prompt.replace("<bos>", "<bos><image>\n", 1)
            else:
                chat_prompt = "<image>\n" + chat_prompt

        logger.debug(f"Final prompt:\n{chat_prompt}")

        # Prepare inputs
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(
                text=chat_prompt,
                images=image,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.model.device)
        else:
            inputs = self.processor(
                text=chat_prompt,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.model.device)

        logger.info(f"Generating diagnosis (max_tokens={max_new_tokens})...")
        start = time.time()

        # Sanitize logits to fix float16 inf/nan on MPS before sampling
        from transformers import LogitsProcessor, LogitsProcessorList

        class SanitizeLogitsProcessor(LogitsProcessor):
            """Cast logits to float32 and replace inf/nan to prevent sampling crashes on MPS."""
            def __call__(self, input_ids, scores):
                scores = scores.float()
                scores = torch.where(
                    torch.isfinite(scores), scores, torch.full_like(scores, -1e4)
                )
                return scores

        logits_processors = LogitsProcessorList([SanitizeLogitsProcessor()])

        # On MPS with float16, use greedy decoding to avoid numerical noise
        # that sampling (softmax) amplifies into gibberish
        is_mps = str(self.model.device).startswith("mps")

        with torch.no_grad():
            if is_mps:
                logger.info("Using greedy decoding (MPS float16 stability)")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    logits_processor=logits_processors,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    logits_processor=logits_processors,
                )

        # Decode output
        decoded = self.processor.decode(
            outputs[0],
            skip_special_tokens=True
        )

        logger.debug(f"Raw output:\n{decoded}")

        response = self._extract_response(decoded)
        response = self._post_process_response(response)

        elapsed = time.time() - start
        logger.info(f"✅ Generated {len(response)} chars in {elapsed:.1f}s")

        return response

    # ------------------------------------------------------------------
    def _build_medical_prompt(self, case_text: str) -> str:
        return (
            "Analyze this medical case and provide a clinical assessment.\n\n"
            f"{case_text}"
        )

    # ------------------------------------------------------------------
    def _prepare_messages(self, prompt: str, image_path: Optional[str]):
        if image_path and os.path.exists(image_path):
            return [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }]
        return [{"role": "user", "content": prompt}]

    # ------------------------------------------------------------------
    def _extract_response(self, text: str) -> str:
        text = text.strip()

        # If explicit Gemma format "model" turn is present, take what comes after
        if "model\n" in text:
            parts = text.split("model\n")
            if len(parts) > 1:
                return parts[-1].strip()
        
        # Handle case without newline
        if "model" in text:
            # Find the last occurrence of model
            idx = text.rfind("model")
            if idx != -1:
                candidate = text[idx + 5:].strip()
                # Only accept if it looks like the start of generation
                if candidate.startswith(":"):
                    candidate = candidate[1:].strip()
                return candidate

        return text

    # ------------------------------------------------------------------
    def _post_process_response(self, response: str) -> str:
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
        """Free memory."""
        try:
            import torch, gc
            del self.model
            del self.processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        self.model = None
        self.processor = None
        self._loaded = False
        logger.info("MedGemma resources cleaned up")