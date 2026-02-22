from pathlib import Path
from typing import Dict, Any, List

import torch
from langchain_core.runnables import Runnable
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from api.config import settings
from core.logger import get_logger
from core.exceptions import CustomException


class LocalLLMRunnable(Runnable):
    """
    LangGraph adapter for LOCAL LLM answer generation (GTX 1650 Ti - 4GB VRAM).
    Drop-in replacement for LLMRunnable using HuggingFace InferenceClient.
    """

    def __init__(self):
        self.logger = get_logger("orchestration.local_llm")
        
        # Load the same prompts as LLMRunnable
        base = Path("prompts/v1")
        self.system_prompt = (base / "system.txt").read_text()
        self.answer_prompt = (base / "answer.txt").read_text()

        # Model configuration for 4GB VRAM
        MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        # Aggressive 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.logger.info("event=LOCAL_LLM_LOADING | model=%s", MODEL_NAME)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with memory constraints
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "3.5GB"},  # Reserve for 4GB GPU
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        vram_gb = torch.cuda.memory_allocated() / 1e9
        
        self.logger.info(
            "event=LOCAL_LLM_INIT | model=%s | vram_gb=%.2f | device=%s",
            MODEL_NAME,
            vram_gb,
            self.model.device
        )

    def invoke(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        config=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate answer using local model with same interface as LLMRunnable.
        """
        
        if not chunks:
            return {
                "answer": "I don't have enough information to answer this question."
            }

        # Format sources exactly like LLMRunnable
        sources_text = "\n\n".join(
            f"[Page {c['metadata']['page_number']}] {c['content']}"
            for c in chunks
        )

        # Use the same answer prompt template
        user_prompt = self.answer_prompt.format(
            query=query,
            sources=sources_text,
        )

        try:
            # Create chat messages in Llama-3 format
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            # Tokenize with truncation for 4GB VRAM
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=2048  # Limit total input length
            ).to(self.model.device)
            
            # Calculate max_new_tokens (use settings.MAX_PROMPT_TOKENS if available)
            max_new_tokens = getattr(settings, 'MAX_PROMPT_TOKENS', 512)
            # Reduce if needed for 4GB VRAM
            max_new_tokens = min(max_new_tokens, 384)
            
            # Generate with same temperature as LLMRunnable (0.2)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.2,  # Match LLMRunnable temperature
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode only the generated tokens (skip input)
            answer = self.tokenizer.decode(
                outputs[0][input_ids.shape[-1]:],
                skip_special_tokens=True
            ).strip()
            
            # Clear CUDA cache to prevent memory buildup
            torch.cuda.empty_cache()
            
            self.logger.info(
                "event=LOCAL_LLM_SUCCESS | sources=%d | answer_len=%d | vram_gb=%.2f",
                len(chunks),
                len(answer),
                torch.cuda.memory_allocated() / 1e9
            )

            return {
                "answer": answer,
            }

        except Exception as e:
            self.logger.exception("event=LOCAL_LLM_FAILURE")
            raise CustomException(
                "Local LLM generation failed",
                error=e,
                context={
                    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                    "source_count": len(chunks),
                    "vram_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                },
            ) from e