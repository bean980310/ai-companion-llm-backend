import os
import traceback
import warnings
from typing import Any

from huggingface_hub import InferenceClient

from ..logging import logger
from ..base_handlers import BaseAPIClientWrapper

try:
    from langchain_integrator import LangchainIntegrator
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn("langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ", UserWarning)
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False

class HuggingfaceInferenceClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, **kwargs)

        self.hf_provider = self.selected_model.split(":")[-1] if ":" in self.selected_model else "auto"

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True
        self.load_model()
        

    def load_model(self):
        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider="hf-inference",
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                verbose=True,
                hf_provider=self.hf_provider
            )
        else: self.client = InferenceClient(token=self.api_key, provider=self.hf_provider,)

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]] | Any]], **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            # client = InferenceClient(
            #     token=self.api_key,
            #     provider=self.hf_provider,
            # )

            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            logger.info(f"[*] Huggingface Inference API 요청: {messages}")
            
            if self.enable_streaming is True:
                chat_stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    extra_body={
                        "top_k": self.top_k,
                        "repetition_penalty": self.repetition_penalty
                    },
                    stream=True
                )
                
                answer = ""

                for chunk in chat_stream:
                    print(chunk.choices[0].delta.content)
                    answer.join(chunk.choices[0].delta.content)

            else:
                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    extra_body={
                        "top_k": self.top_k,
                        "repetition_penalty": self.repetition_penalty
                    },
                    stream=False
                )

                answer = chat_completion.choices[0].message.content
                
            return answer.strip()