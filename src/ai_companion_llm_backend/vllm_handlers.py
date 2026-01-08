"""
vLLM Offline Handlers - In-process inference using vLLM's LLM class.

This module provides handlers for direct vLLM inference without requiring
a separate server. Models are loaded directly into memory.

Usage:
    from ai_companion_llm_backend.vllm_handlers import VllmCausalModelHandler

    handler = VllmCausalModelHandler(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )

    response = handler.generate_answer([
        {"role": "user", "content": "Hello!"}
    ])
"""

import os
import random
import warnings
from typing import Any, Generator

import numpy as np

try:
    from vllm import LLM, SamplingParams
    from vllm.distributed import destroy_model_parallel
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    warnings.warn(
        "vllm is not installed. Install it with `pip install vllm` to use offline inference.",
        UserWarning
    )

try:
    from langchain_integrator import LangchainIntegrator
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn("langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ", UserWarning)
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False

from .logging import logger
from .base_handlers import BaseCausalModelHandler


class VllmCausalModelHandler(BaseCausalModelHandler):
    """
    Handler for vLLM offline inference (in-process, no server required).

    This handler loads models directly using vLLM's LLM class, providing
    high-performance inference without HTTP overhead.

    Args:
        model_id: HuggingFace model ID or local path
        lora_model_id: Optional LoRA adapter ID
        use_langchain: Whether to use LangchainIntegrator (default: False)
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
        gpu_memory_utilization: Fraction of GPU memory to use (default: 0.9)
        max_model_len: Maximum sequence length (default: auto-detected)
        dtype: Data type for model weights (default: "auto")
        quantization: Quantization method (e.g., "awq", "gptq", None)
        trust_remote_code: Whether to trust remote code (default: True)
        **kwargs: Additional parameters (temperature, top_p, top_k, etc.)
    """

    def __init__(
        self,
        model_id: str,
        lora_model_id: str | None = None,
        use_langchain: bool = False,
        **kwargs
    ):
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vllm is not installed. Install it with `pip install vllm`"
            )

        # vLLM-specific configuration
        self.tensor_parallel_size = int(kwargs.pop("tensor_parallel_size", 1))
        self.gpu_memory_utilization = float(kwargs.pop("gpu_memory_utilization", 0.9))
        self.max_model_len = kwargs.pop("max_model_len", None)
        self.dtype = kwargs.pop("dtype", "auto")
        self.quantization = kwargs.pop("quantization", None)
        self.trust_remote_code = kwargs.pop("trust_remote_code", True)
        self.enforce_eager = kwargs.pop("enforce_eager", False)

        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

        self.llm: LLM | None = None
        self.sampling_params: SamplingParams | None = None
        self.tokenizer = None

        # Handle max_length -> max_tokens
        if self.max_length > 0:
            self.max_tokens = self.max_length

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True

        self.set_seed(self.seed)
        self.load_model()

    def load_model(self):
        """Load the vLLM model or initialize LangchainIntegrator."""
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider=("self-provided", "vllm"),
                model_name=self.local_model_path,
                lora_model_name=self.local_lora_model_path,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                verbose=True,
            )
            logger.info(f"vLLM LangchainIntegrator initialized for: {self.model_id}")
        else:
            # Determine model path
            model_path = self.local_model_path
            if not os.path.exists(model_path):
                # Use HuggingFace model ID directly
                model_path = self.model_id

            logger.info(f"Loading vLLM model: {model_path}")

            # Build LLM kwargs
            llm_kwargs = {
                "model": model_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "dtype": self.dtype,
                "trust_remote_code": self.trust_remote_code,
                "enforce_eager": self.enforce_eager,
                "seed": self.seed,
            }

            if self.max_model_len is not None:
                llm_kwargs["max_model_len"] = self.max_model_len

            if self.quantization is not None:
                llm_kwargs["quantization"] = self.quantization

            # Load LoRA adapter if specified
            if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
                llm_kwargs["enable_lora"] = True
                logger.info(f"LoRA adapter enabled: {self.local_lora_model_path}")

            self.llm = LLM(**llm_kwargs)
            self.tokenizer = self.llm.get_tokenizer()

            logger.info(f"vLLM model loaded successfully: {model_path}")

    def generate_answer(self, history: list[dict], **kwargs) -> str:
        """
        Generate a response based on conversation history.

        Args:
            history: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional generation parameters

        Returns:
            Generated response text
        """
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)

        # Apply chat template
        prompt = self.load_template(history)

        # Get sampling parameters
        self.get_settings()

        # Handle LoRA adapter if specified
        lora_request = None
        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest(
                lora_name="adapter",
                lora_int_id=1,
                lora_path=self.local_lora_model_path,
            )

        # Generate
        if self.enable_streaming:
            return self._generate_streaming(prompt, lora_request)
        else:
            return self._generate_non_streaming(prompt, lora_request)

    def _generate_non_streaming(self, prompt: str, lora_request=None) -> str:
        """Generate response without streaming."""
        outputs = self.llm.generate(
            prompts=[prompt],
            sampling_params=self.sampling_params,
            lora_request=lora_request,
        )

        response = outputs[0].outputs[0].text

        # Handle thinking tokens if present
        if "</think>" in response:
            _, response = response.split("</think>", 1)

        return response.strip()

    def _generate_streaming(self, prompt: str, lora_request=None) -> str:
        """Generate response with streaming output."""
        outputs = self.llm.generate(
            prompts=[prompt],
            sampling_params=self.sampling_params,
            lora_request=lora_request,
        )

        response = outputs[0].outputs[0].text

        # Print for streaming effect
        for char in response:
            print(char, end="", flush=True)
        print()

        # Handle thinking tokens if present
        if "</think>" in response:
            _, response = response.split("</think>", 1)

        return response.strip()

    def get_settings(self):
        """Configure sampling parameters."""
        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k if self.top_k > 0 else -1,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed if self.seed != -1 else None,
        )

    def load_template(self, messages: list[dict]) -> str:
        """Apply chat template to messages."""
        # Check for special model-specific templates
        model_lower = self.model_id.lower()

        if "qwen3" in model_lower and "instruct" not in model_lower:
            # Qwen3 base models with thinking support
            return self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        else:
            return self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def unload_model(self):
        """Unload the model and free GPU memory."""
        if self.llm is not None:
            try:
                destroy_model_parallel()
            except Exception as e:
                logger.warning(f"Error during model parallel cleanup: {e}")

            del self.llm
            self.llm = None
            self.tokenizer = None

            # Force garbage collection
            import gc
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass

            logger.info("vLLM model unloaded and GPU memory freed")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload_model()
        except Exception:
            pass

    @staticmethod
    def set_seed(seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
