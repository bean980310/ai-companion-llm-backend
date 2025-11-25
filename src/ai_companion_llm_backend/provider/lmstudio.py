# WIP: src/pipelines/llm/api/lmstudio.py
import os
from typing import Any, BinaryIO

from typing_extensions import Buffer
from PIL import Image, ImageFile
import lmstudio as lms

SERVER_API_HOST = "localhost:1234"

lms.get_default_client(SERVER_API_HOST)

from ..base_handlers import BaseAPIClientWrapper
from ..langchain_integrator import LangchainIntegrator

class LMStudioIntegrator(BaseAPIClientWrapper):
    def __init__(self, selected_model: str = None, api_key: str = "not-needed", use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | BinaryIO | Buffer | os.PathLike[str] | Any | None = None, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, image_input, **kwargs)

        # self.lora_model_id: str | None = lora_model_id

        # if selected_model is not None:
        #     self.local_model_path = selected_model
        # else:
        #     self.local_model_path = self.model_id

        # self.local_lora_model_path = self.lora_model_id

        self.system_prompt = None
        self.user_message = None
        self.chat_history = None

        if self.max_length > 0:
            self.max_tokens = self.max_length
        else:
            self.max_tokens = 4096

        self.llm: lms.LLM | None = None
        self.chat: lms.Chat | None = None
        self.client: lms.Client | Any | None = None
        self.image_handle: lms.FileHandle | Any | None = None

        self.load_model()

    def load_model(self):
        if self.use_langchain:
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
        else:
            self.llm = lms.llm(self.model, config={"seed": self.seed})

    def generate_answer(self, history, **kwargs):
        if self.use_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            for msg in history[:-1]:
                if msg["role"] == "system":
                    self.system_prompt = msg["content"]

            self.chat = lms.Chat(self.system_prompt)

            for msg in history[:-1]:
                if msg["role"] == "user":
                    self.chat.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    self.chat.add_assistant_response(msg["content"])

            self.user_message = history[-1]["content"]

            if self.image_input is not None:
                self.image_handle = lms.prepare_image(self.image_input)
                self.chat.add_user_message(self.user_message, images=[self.image_handle])
            else:
                self.chat.add_user_message(self.user_message)

            if self.enable_streaming is True:
                streamer = self.llm.respond_stream(self.chat, config={
                    "temperature": self.temperature,
                    "topPSampling": self.top_p,
                    "topKSampling": self.top_k,
                    "repeatPenalty": self.repetition_penalty,
                    "maxTokens": self.max_tokens,
                })
                answer = ""
                for fragment in streamer:
                    print(fragment.content, end="", flush=True)
                    answer.join(fragment.content)

            else:
                response = self.llm.respond(self.chat, config={
                    "temperature": self.temperature,
                    "topPSampling": self.top_p,
                    "topKSampling": self.top_k,
                    "repeatPenalty": self.repetition_penalty,
                    "maxTokens": self.max_tokens,
                })
                answer = response.content

            return answer.strip()