import os
import base64
import traceback
from typing import Any

from PIL import Image, ImageFile

import openai
from openai import OpenAI
from openai.types.responses.response import Response
from openai.types.responses.response_stream_event import ResponseStreamEvent

from ..logging import logger
from ..base_handlers import BaseAPIClientWrapper
from langchain_integrator import LangchainIntegrator

class OpenAIClientWrapper(BaseAPIClientWrapper):
    def __init__(self, selected_model: str, api_key: str | None = None, use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, **kwargs):
        super().__init__(selected_model, api_key, use_langchain, image_input, **kwargs)

        # self.client = OpenAI(api_key=self.api_key)
        self.system_prompt: str = None

        if self.use_langchain: self.load_model()
        else: self.client = OpenAI(api_key=self.api_key)

    def load_model(self):
        self.langchain_integrator = LangchainIntegrator(
            provider="openai",
            model_name=self.model,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            verbose=True,
            enable_thinking=self.enable_thinking,
            image_input=self.image_input
        )

    def generate_answer(self, history: list[dict[str, str | list[dict[str, str]] | Any]], **kwargs):
        if self.use_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            # client = OpenAI(api_key=self.api_key)
            # openai.api_key = self.api_key

            self.system_prompt = next((msg['content'] for msg in history[:1] if msg['role'] == 'system'), None)

            # if self.system_prompt is not None:
            #     messages = [{"role": msg['role'], "content": msg['content']} for msg in history[1:-1]]
            # else:
            #     messages = [{"role": msg['role'], "content": msg['content']} for msg in history[:-1]]

            # messages = [{"role": msg['role'], "content": msg['content']} for msg in history[(1 if self.system_prompt else 0):-1]]
            messages = [{"role": msg['role'], "content": msg['content']} for msg in history[bool(self.system_prompt):-1]]

            if self.image_input is not None:
                with open(self.image_input, "rb") as f:
                    image = self.client.files.create(file=f, purpose='vision')

                new_message = {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": history[-1]["content"]["text"]},
                        {"type": "input_image", "file_id": image.id}
                    ],
                }
                messages.append(new_message)
            else:
                messages.append({"role": "user", "content": history[-1]["content"]})

            logger.info(f"[*] OpenAI API 요청: {messages}")

            # response = client.chat.completions.create(
            #     model=self.model,
            #     messages=messages,
            #     temperature=self.temperature,
            #     max_tokens=self.max_tokens,
            #     top_logprobs=self.top_k,
            #     top_p=self.top_p,
            #     frequency_penalty=self.repetition_penalty,
            #     presence_penalty=self.repetition_penalty,
            # )

            # answer = response.choices[0].message["content"]

            non_reasoning = "none" if "gpt-5.1" in self.model else "minimum"

            if self.enable_streaming is True:
                stream: openai.Stream[ResponseStreamEvent] = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    instructions=self.system_prompt,
                    store=False,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_logprobs=self.top_k,
                    top_p=self.top_p if self.enable_thinking is False else None,
                    stream=True,
                    reasoning={"effort": "medium"} if self.enable_thinking else {"effort": non_reasoning},
                    extra_body={
                        "frequency_penalty": self.repetition_penalty if self.enable_thinking is False else None,
                        "presence_penalty": self.repetition_penalty if self.enable_thinking is False else None,
                    }
                )
                answer = ""

                for event in stream:
                    print(event.text, end="", flush=True)
                    answer += event.text
                    
            else:
                response: Response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    instructions=self.system_prompt,
                    store=False,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_logprobs=self.top_k,
                    top_p=self.top_p if self.enable_thinking is False else None,
                    stream=False,
                    reasoning={"effort": "medium"} if self.enable_thinking else {"effort": non_reasoning},
                    extra_body={
                        "frequency_penalty": self.repetition_penalty if self.enable_thinking is False else None,
                        "presence_penalty": self.repetition_penalty if self.enable_thinking is False else None,
                    }
                )
                answer = response.output_text
                
            return answer.strip()

# class OpenAILangChainIntegration:
#     def __init__(self, api_key=None, model="gpt-4o-mini", temperature=0.6, top_p=0.9, top_k=40, repetition_penalty=1.0):
#         self.api_key = api_key
#         if not api_key:
#             logger.error("OpenAI API Key가 missing.")
#             raise "OpenAI API Key가 필요합니다."
        
#         self.model = model
#         self.llm = ChatOpenAI(
#             model=model,
#             temperature=temperature,
#             top_p=top_p,
#             top_logprobs=top_k,
#             frequency_penalty=repetition_penalty,
#             api_key=api_key,
#             max_tokens=2048
#         )