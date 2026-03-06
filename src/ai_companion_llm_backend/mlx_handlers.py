import random
import traceback
import os
import deprecated
import platform
import warnings
from typing import Any, Dict, List, Optional, Union, Iterator, Generator

import numpy as np
from PIL import Image, ImageFile

from transformers import AutoConfig

try:
    import mlx.core as mx
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("mlx is not installed. Please install it to use this library.", UserWarning)
    else:
        pass

try:
    from mlx_lm import load as mlx_lm_load
    from mlx_lm import generate as mlx_lm_generate
    from mlx_lm import stream_generate as mlx_lm_stream_generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    from mlx_lm.generate import GenerationResponse
    from mlx_lm.utils import load_config as mlx_lm_load_config
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("mlx_lm is not installed. Please install it to use MLX Chat.", UserWarning)
    else:
        pass

try:
    from mlx_vlm import load as mlx_vlm_load
    from mlx_vlm import generate as mlx_vlm_generate
    from mlx_vlm import stream_generate as mlx_vlm_stream_generate
    from mlx_vlm.utils import load_config as mlx_vlm_load_config
    from mlx_vlm.prompt_utils import apply_chat_template
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("mlx_vlm is not installed. Please install it to use MLX Multimodal Chat.", UserWarning)
    else:
        pass

try:
    from langchain_integrator import LangchainIntegrator
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn("langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ", UserWarning)
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False

from .logging import logger
from .base_handlers import BaseCausalModelHandler, BaseVisionModelHandler, BaseModelHandler, BaseOmniModelHandler

class MlxUnifiedModelHandler(BaseModelHandler):
    def __init__(self, model_id: str, lora_model_id: str | None = None, model_type="mlx", use_langchain: bool = True, image_input: str | List[str] | Image.Image | ImageFile.ImageFile | Any | None = None, audio_input: str | List[str] | Any | None = None, video_input: str | List[str] | Any | None = None, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, image_input, audio_input, video_input, **kwargs)

        self.arch = None

        self.sampler = None
        self.logits_processors = None
        self.tokenizer_config = kwargs.get("tokenizer_config", {})

        if self.max_length > 0:
            self.max_tokens = self.max_length
        else:
            if "qwen3" in self.model_id.lower():
                if "instruct" in self.model_id.lower():
                    self.max_tokens = 16384
                else:
                    self.max_tokens = 32768

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True

        self.set_seed(self.seed)
        self.load_model()
    
    def load_model(self):
        self.arch = AutoConfig.from_pretrained(self.local_model_path).architectures[0]

        if self.enable_langchain:
            if self.arch in self.check_is_multimodal_lm:
                print("langchain_mlx is currently not supported vision features.")
                self.enable_langchain = False
                self.model, self.processor = mlx_vlm_load(self.local_model_path, adapter_path=self.local_lora_model_path, lazy=True)
                self.config = mlx_vlm_load_config(self.local_model_path)
            else:
                self.langchain_integrator = LangchainIntegrator(
                    provider=("self-provided", "mlx"),
                    model_name=self.local_model_path,
                    lora_model_name=self.local_lora_model_path,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    verbose=True,
                    tokenizer_config=self.tokenizer_config
                )
        else:
            if self.arch in self.check_is_causal_lm:
                self.model, self.tokenizer = mlx_lm_load(self.local_model_path, adapter_path=self.local_lora_model_path, tokenizer_config=self.tokenizer_config)
            elif self.arch in self.check_is_multimodal_lm:
                self.model, self.processor = mlx_vlm_load(self.local_model_path, adapter_path=self.local_lora_model_path, lazy=True)
            else:
                logger.error(f"ERROR: Unsupported Task!")
                return

    def generate_answer(self, history, **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)

        text = self.load_template(history)
        self.get_settings()

        if self.enable_streaming:
            return self._generate_streaming(text)

        response = self._generate_answer(text)

        if "</think>" in response:
            _, response = response.split("</think>", 1)

        return response.strip()

    def get_settings(self):
        self.sampler = make_sampler(
            temp=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k
        )
        self.logits_processors = make_logits_processors(repetition_penalty=self.repetition_penalty)

    def load_template(self, messages):
        if self.arch in self.check_is_causal_lm:
            kwargs = dict(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
            if self.use_tools:
                kwargs["tools"] = self.tools
            template = self.tokenizer.apply_chat_template(**kwargs)
        elif self.arch in self.check_is_multimodal_lm:
            num_images = len(self.image_input) if self.image_input else 0
            num_audios = len(self.audio_input) if self.audio_input else 0
            kwargs = dict(
                processor=self.processor,
                config=self.config,
                prompt=messages,
                num_images=num_images,
                num_audios=num_audios,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            if self.use_tools:
                kwargs["tools"] = self.tools
            template = apply_chat_template(**kwargs)
        return template

    def _generate_answer(self, prompt):
        if self.arch in self.check_is_causal_lm:
            response = mlx_lm_generate(self.model, self.tokenizer, prompt=prompt, verbose=True, sampler=self.sampler, logits_processors=self.logits_processors, max_tokens=self.max_tokens, max_kv_size=2048)
        elif self.arch in self.check_is_multimodal_lm: 
            response = mlx_vlm_generate(self.model, self.processor, prompt=prompt, image=self.image_input, audio=self.audio_input, video=self.video_input, verbose=True, sampler=self.sampler, repetition_penalty=self.repetition_penalty, max_tokens=self.max_tokens, max_kv_size=2048)
        return response

    def _generate_streaming(self, prompt) -> Generator[str, None, None]:
        full_text = ""

        if self.arch in self.check_is_causal_lm:
            for response in mlx_lm_stream_generate(self.model, self.tokenizer, prompt=prompt, verbose=True, sampler=self.sampler, logits_processors=self.logits_processors, max_tokens=self.max_tokens):
                full_text += response.text

                if "<think>" in full_text and "</think>" not in full_text:
                    continue

                if "</think>" in full_text:
                    _, response_text = full_text.split("</think>", 1)
                    yield response_text.strip()
                else:
                    yield full_text.strip()

        elif self.arch in self.check_is_multimodal_lm:
            for response in mlx_vlm_stream_generate(self.model, self.processor, prompt=prompt, image=self.image_input, audio=self.audio_input, video=self.video_input, verbose=True, sampler=self.sampler, repetition_penalty=self.repetition_penalty, max_tokens=self.max_tokens):
                full_text += response

                if "<think>" in full_text and "</think>" not in full_text:
                    continue

                if "</think>" in full_text:
                    _, response_text = full_text.split("</think>", 1)
                    yield response_text.strip()
                else:
                    yield full_text.strip()

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        mx.random.seed(seed)


@deprecated.deprecated(reason="MlxCausalModelHandler is now merged with MlxVisionModelHandler. Use MlxUnifiedModelHandler instead.", version="1.0.0")
class MlxCausalModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="mlx", use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)
        
        self.sampler = None
        self.logits_processors = None
        self.tokenizer_config = self.get_eos_token()

        if self.max_length > 0:
            self.max_tokens = self.max_length
        else:
            if "qwen3" in self.model_id.lower():
                if "instruct" in self.model_id.lower():
                    self.max_tokens = 16384
                else:
                    self.max_tokens = 32768

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True

        self.set_seed(self.seed)
        self.load_model()
        
    def load_model(self):
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider=("self-provided", "mlx"),
                model_name=self.local_model_path,
                lora_model_name=self.local_lora_model_path,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                verbose=True,
                tokenizer_config=self.tokenizer_config
            )
        else:
            self.model, self.tokenizer = mlx_lm_load(self.local_model_path, adapter_path=self.local_lora_model_path, tokenizer_config=self.tokenizer_config)

    def generate_answer(self, history, **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            text = self.load_template(history)
            self.get_settings()
            response = mlx_lm_generate(self.model, self.tokenizer, prompt=text, verbose=True, sampler=self.sampler, logits_processors=self.logits_processors, max_tokens=self.max_tokens, max_kv_size=2048)

            if "</think>" in response:
                _, response = response.split("</think>", 1)
            
            return response.strip()

    def get_settings(self):
        self.sampler = make_sampler(
            temp=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k
        )
        self.logits_processors = make_logits_processors(repetition_penalty=self.repetition_penalty)
    
    def load_template(self, messages):
        if self.use_tools:
            return self.tokenizer.apply_chat_template(
                conversation=messages,
                tools=self.tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
        else:
            return self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )

    def get_eos_token(self):
        if "llama-3" in self.local_model_path.lower():
            return {"eos_token": "<|eot_id|>", "trust_remote_code": True}
        elif any(k in self.local_model_path.lower() for k in ["qwen2", "qwen3"]):
            return {"eos_token": "<|im_end|>", "trust_remote_code": True}
        elif any(k in self.local_model_path.lower() for k in ["mistral", "ministral", "mixtral", "magistral", "devstral"]):
            return {"eos_token": "</s>", "trust_remote_code": True}
        else:
            return {}

    def _generate_streaming(self, prompt_text: str) -> str | Generator[GenerationResponse, None, None]:
        """
        Generate text in chunks to avoid very long single-pass generations.
        Calls mlx_lm_generate repeatedly, appending the continuation each time.
        Stops if EOS or no progress is made.
        """

        generated_text = ""
        temp = ""
        for response in mlx_lm_stream_generate(self.model, self.tokenizer, prompt=prompt_text, sampler=self.sampler, logits_processors=self.logits_processors, max_tokens=self.max_tokens):
            print(response.text, end='', flush=True)
            if "<think>" in response.text:
                while "</think>" not in response.text:
                    temp += ''.join(response.text)
                else:
                    temp += ''.join(response.text)
                    _, generated_text = temp.split("</think>", 1)
                    yield generated_text.strip()
            else:
                generated_text += ''.join(response.text)
                yield generated_text.strip()


        return generated_text.strip()
    
    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        mx.random.seed(seed)
        
@deprecated.deprecated(reason="MlxVisionModelHandler is now merged with MlxCausalModelHandler. Use MlxUnifiedModelHandler instead.", version="1.0.0")
class MlxVisionModelHandler(BaseVisionModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="mlx", use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, image_input, **kwargs)

        self.sampler = None
        self.logits_processors = None
        self.tokenizer_config = kwargs.get("tokenizer_config", {})

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True

        self.set_seed(self.seed)
        self.load_model()

    def load_model(self):
        if self.enable_langchain and not self.image_input:
            self.langchain_integrator = LangchainIntegrator(
                provider="mlx",
                model_name=self.local_model_path,
                lora_model_name=self.local_lora_model_path,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                verbose=True
            )
        else:
            self.model, self.processor = mlx_vlm_load(self.local_model_path, adapter_path=self.local_lora_model_path, lazy=True)
            self.config = mlx_vlm_load_config(self.local_model_path)
        

    def generate_answer(self, history, **kwargs):
        if self.enable_langchain and not self.image_input:
            return self.langchain_integrator.generate_answer(history)
        else:
            formatted_prompt = self.load_template(history)
            self.get_settings()
            response = mlx_vlm_generate(self.model, self.processor, formatted_prompt, image=self.image_input, verbose=True, sampler=self.sampler, repetition_penalty=self.repetition_penalty, max_tokens=self.max_tokens, max_kv_size=2048)

            if "</think>" in response.text:
                _, response.text = response.text.split("</think>", 1)

            return response.text.strip()

    def get_settings(self):
        self.sampler = make_sampler(
            temp=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k
        )
        self.logits_processors = make_logits_processors(repetition_penalty=self.repetition_penalty)
        # return temperature, top_k, top_p, repetition_penalty

    def load_template(self, messages):
        if self.use_tools:
            return apply_chat_template(
                processor=self.processor,
                config=self.config,
                prompt=messages,
                num_images=len(self.image_input), # <-- history 자체를 전달
                tools=self.tools,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        else:
            return apply_chat_template(
                processor=self.processor,
                config=self.config,
                prompt=messages,
                num_images=len(self.image_input), # <-- history 자체를 전달
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )


    def _generate_streaming_vision(self, prompt_text: str) -> str | Generator[str, None, None]:
        generated_text = ""
        temp = ""

        for response in mlx_vlm_stream_generate(self.model, self.processor, prompt_text, image=self.image_input, verbose=True, sampler=self.sampler, repetition_penalty=self.repetition_penalty, max_tokens=self.max_tokens):
            print(response, end='', flush=True)
            if "<think>" in response:
                while "</think>" not in response:
                    temp += ''.join(response)
                else:
                    temp += ''.join(response)
                    _, generated_text = temp.split("</think>", 1)
                    yield generated_text.strip()
            else:
                generated_text += ''.join(response)
                yield generated_text.strip()

    def _generate_streaming(self, prompt_text: str) -> str | Generator[GenerationResponse, None, None]:
        """
        Generate text in chunks to avoid very long single-pass generations.
        Calls mlx_lm_generate repeatedly, appending the continuation each time.
        Stops if EOS or no progress is made.
        """

        generated_text = ""
        temp = ""
        for response in mlx_lm_stream_generate(self.model, self.tokenizer, prompt=prompt_text, verbose=True, sampler=self.sampler, logits_processors=self.logits_processors, max_tokens=self.max_tokens):
            print(response.text, end='', flush=True)
            if "<think>" in response.text:
                while "</think>" not in response.text:
                    temp += ''.join(response.text)
                else:
                    temp += ''.join(response.text)
                    _, generated_text = temp.split("</think>", 1)
                    yield generated_text.strip()
            else:
                generated_text += ''.join(response.text)
                yield generated_text.strip()

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        mx.random.seed(seed)