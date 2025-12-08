import os
from typing import Any, BinaryIO

from typing_extensions import Buffer
from PIL import Image, ImageFile
import ollama

SERVER_API_HOST="127.0.0.1:11434"

from ..base_handlers import BaseAPIClientWrapper
from langchain_integrator import LangchainIntegrator

class OllamaIntegrator(BaseAPIClientWrapper):
    def __init__(self, selected_model: str = None, api_key: str = "not-needed", use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | BinaryIO | Buffer | os.PathLike[str] | Any | None = None, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, image_input, **kwargs)

        self.system_prompt = None
        self.user_message = None
        self.chat_history = None

        if self.max_length > 0:
            self.max_tokens = self.max_length
        else:
            self.max_tokens = 4096

        self.server_url = str(kwargs.get("server_url", "http://localhost:11434"))
        # self.client = ollama.Client(host=self.server_url)

        if self.use_langchain: self.load_model()
        else: self.client = ollama.Client(host=self.server_url)

    def load_model(self):
        self.langchain_integrator = LangchainIntegrator(
            provider="lmstudio",
            model_name=self.model,
            api_key="not-needed",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            verbose=True,
        )

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]] | Any]], **kwargs):
        if self.use_langchain:
            return self.langchain_integrator.generate_answer(history)

        else:
            ollama.chat()