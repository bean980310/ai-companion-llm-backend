import os
import warnings
from typing import Any, Generator

from PIL import Image, ImageFile

import llama_cpp
from llama_cpp import Llama # gguf 모델을 로드하기 위한 라이브러리
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
from llama_cpp.llama_chat_format import get_chat_completion_handler

from .logging import logger
try:
    from langchain_integrator import LangchainIntegrator
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn("langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ", UserWarning)
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False

from .base_handlers import BaseCausalModelHandler, BaseVisionModelHandler


class GGUFCausalModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="gguf", device='cpu', use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

        self.n_gpu_layers = -1 if device != 'cpu' else 0
        self.sampler = None
        self.logits_processors = None

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
        
        self.load_model()
        
    def load_model(self):
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider=("self-provided", "gguf"),
                model_name=self.local_model_path,
                lora_model_name=self.local_lora_model_path,
                max_tokens=self.max_tokens,
                seed=self.seed,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                n_gpu_layers=self.n_gpu_layers,
                verbose=True,
            )
        else:
            self.model = Llama(
                model_path=self.local_model_path,
                lora_path=self.local_lora_model_path,
                n_gpu_layers=self.n_gpu_layers,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
                n_ctx=2048
            )
        
    def generate_answer(self, history, **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            prompt = [{"role": msg['role'], "content": msg['content']} for msg in history]
            response = self.model.create_chat_completion(
                messages=prompt,
                seed=self.seed,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repeat_penalty=self.repetition_penalty,
                max_tokens=self.max_tokens,
                stream=True
            )
            # answer = response["choices"][0]["message"]["content"]
            # Streaming response handling (if needed)
            answer = ""
            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                if 'role' in delta:
                    print(delta['role'], end=": ")
                elif 'content' in delta:
                    print(delta['content'], end="", flush=True)
                    answer += "".join(delta['content'])
                    
            return answer

    def get_settings(self):
        pass
    
    def load_template(self, messages):
        pass
        
    
class GGUFVisionModelHandler(BaseVisionModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="gguf", device='cpu', use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

        self.n_gpu_layers = -1 if device != 'cpu' else 0
        self.sampler = None
        self.logits_processors = None

        if self.max_length > 0:
            self.max_tokens = self.max_length
        else:
            self.max_tokens = 4096

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True
        
        self.load_model()
        
    def load_model(self):
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider=("self-provided", "gguf"),
                model_name=self.local_model_path,
                lora_model_name=self.local_lora_model_path,
                max_tokens=self.max_tokens,
                seed=self.seed,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                n_gpu_layers=self.n_gpu_layers,
                verbose=True,
            )
        else:
            self.model = Llama(
                model_path=self.local_model_path,
                lora_path=self.local_lora_model_path,
                n_gpu_layers=self.n_gpu_layers,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
                n_ctx=2048
            )
        
    def generate_answer(self, history, **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            prompt = [{"role": msg['role'], "content": msg['content']} for msg in history[:-1]]
            user_input = history[-1]['content']
            if isinstance(user_input, (list, dict)):
                text_content = user_input.get('text', "")
                image_content =user_input.get('image_url', [])

                content_list = [{"type": "text", "text": text_content}]
                if image_content:
                    for image in image_content:
                        content_list.append({"type": "image", "image_url": image})
                        
                prompt.append({"role": "user", "content": content_list})
            else:
                prompt.append({"role": "user", "content": user_input})
                        
                        
            response = self.model.create_chat_completion(
                messages=prompt,
                seed=self.seed,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repeat_penalty=self.repetition_penalty,
                max_tokens=self.max_tokens,
                stream=True
            )
            # answer = response["choices"][0]["message"]["content"]
            # Streaming response handling (if needed)
            answer = ""
            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                if 'role' in delta:
                    print(delta['role'], end=": ")
                elif 'content' in delta:
                    print(delta['content'], end="", flush=True)
                    answer += "".join(delta['content'])
                    
            return answer

    def get_settings(self):
        pass
    
    def load_template(self, messages):
        pass