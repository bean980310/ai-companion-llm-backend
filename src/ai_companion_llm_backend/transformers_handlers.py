import os
import traceback
import warnings
import deprecated
import threading
import random
import base64
from typing import Any, Generator, List

from io import BytesIO
from PIL import Image, ImageFile
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoModelForImageTextToText, AutoModelForCausalLM, GenerationConfig, Llama4ForConditionalGeneration, TextStreamer, TextIteratorStreamer, Qwen3ForCausalLM, Qwen3MoeForCausalLM, Mistral3ForConditionalGeneration, MistralForCausalLM, Llama4Processor, LlamaTokenizer, set_seed, BatchEncoding, AutoModelForMultimodalLM, AutoConfig, Qwen3_5ForConditionalGeneration

from .logging import logger
try:
    from langchain_integrator import LangchainIntegrator
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = True
except ImportError:
    warnings.warn("langchain_integrator is required when use_langchain=True. Install it or set use_langchain=False. ", UserWarning)
    LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE = False
from .base_handlers import BaseCausalModelHandler, BaseVisionModelHandler, BaseOmniModelHandler, BaseModelHandler

class TransformersUnifiedModelHandler(BaseModelHandler):
    def __init__(self, model_id: str, lora_model_id: str | None = None, model_type='transformers', device='cpu', use_langchain: bool = True, image_input: str | [List[str]] | Image.Image | ImageFile.ImageFile | Any | None = None, audio_input: str | List[str] | Any | None = None, video_input: str | List[str] | Any | None = None, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, image_input, audio_input, video_input, **kwargs)

        if self.max_length > 0:
            self.max_tokens = self.max_length
        else:
            if "qwen3" in self.model_id.lower():
                if "instruct" in self.model_id.lower():
                    self.max_tokens = 16384
                else:
                    self.max_tokens = 32768

        self.max_new_tokens = self.max_tokens
        self.device = device

        set_seed(self.seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)

        if self.use_langchain and LANGCHAIN_INTEGRATOR_IS_INSTALLED_AND_AVAILABLE:
            self.enable_langchain = True
        self.load_model()

    def load_model(self):
        self.arch = AutoConfig.from_pretrained(self.local_model_path).architectures[0]
        if self.arch in self.check_is_causal_lm:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, trust_remote_code=True, device_map='auto')
        elif self.arch in self.check_is_image_text_to_text:
            if self.image_input:
                self.processor = AutoProcessor.from_pretrained(self.local_model_path, trust_remote_code=True)
                self.model = AutoModelForImageTextToText.from_pretrained(self.local_model_path, trust_remote_code=True, device_map='auto')
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
                self.model = AutoModelForImageTextToText.from_pretrained(self.local_model_path, trust_remote_code=True, device_map='auto')
        elif self.arch in self.check_is_any_to_any:
            if self.image_input or self.audio_input:
                self.processor = AutoProcessor.from_pretrained(self.local_model_path, trust_remote_code=True)
                self.model = AutoModelForMultimodalLM.from_pretrained(self.local_model_path, trust_remote_code=True, device_map='auto')
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
                self.model = AutoModelForMultimodalLM.from_pretrained(self.local_model_path, trust_remote_code=True, device_map='auto')
        else:
            logger.error(f"ERROR: Unsupported Task!")
            return

        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)

        if self.enable_langchain:
            if self.processor is not None:
                self.langchain_integrator = LangchainIntegrator(
                    provider=("self-provided", "transformers"),
                    model=self.model,
                    processor=self.processor,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    verbose=True
                )
            else:
                self.langchain_integrator = LangchainIntegrator(
                    provider=("self-provided", "transformers"),
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    verbose=True
                )

    def generate_answer(self, history, **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)

        self.images = []
        messages = self.process_messages(history)
        self.generation_config = self.get_settings()
        prompt = self.load_template(messages)

        if self.processor is not None:
            inputs = self.processor(text=prompt, images=self.images if self.images else None, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        if self.enable_streaming:
            return self._generate_streaming(inputs)

        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
        )

        decoder = self.processor if self.processor is not None else self.tokenizer
        generated_text = decoder.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        )

        if "</think>" in generated_text:
            _, generated_text = generated_text.split("</think>", 1)

        return generated_text.strip()

    def _generate_streaming(self, inputs):
        decoder = self.processor if self.processor is not None else self.tokenizer
        streamer = TextIteratorStreamer(decoder, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(inputs)
        generation_kwargs.update(generation_config=self.generation_config, streamer=streamer)

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        full_text = ""
        for chunk in streamer:
            full_text += chunk

            if "<think>" in full_text and "</think>" not in full_text:
                continue

            if "</think>" in full_text:
                _, response_text = full_text.split("</think>", 1)
                yield response_text.strip()
            else:
                yield full_text.strip()

        thread.join()


    def get_settings(self):
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )

    def load_template(self, messages):
        applier = self.processor if self.processor is not None else self.tokenizer
        kwargs = dict(
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.enable_thinking
        )
        if self.use_tools:
            kwargs["tools"] = self.tools
        return applier.apply_chat_template(messages, **kwargs)

@deprecated.deprecated(reason="TransformersCausalModelHandler is now merged with TransformersVisionModelHandler. Use TransformersUnifiedModelHandler instead.", version="1.0.0")
class TransformersCausalModelHandler(BaseCausalModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, **kwargs)

        if self.max_length > 0:
            self.max_tokens = self.max_length
        else:
            if "qwen3" in self.model_id.lower():
                if "instruct" in self.model_id.lower():
                    self.max_tokens = 16384
                else:
                    self.max_tokens = 32768

        self.max_new_tokens = self.max_tokens
        self.device = device

        set_seed(self.seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)

        if self.enable_langchain:
            self.enable_langchain = True
        self.load_model()
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, trust_remote_code=True, device_map='auto')
        
        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)

        if self.enable_langchain:
            self.langchain_integrator = LangchainIntegrator(
                provider=("self-provided", "transformers"),
                model=self.model,
                tokenizer=self.tokenizer,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                verbose=True
            )
        
    def generate_answer(self, history, **kwargs):
        if self.enable_langchain:
            return self.langchain_integrator.generate_answer(history)
        else:
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            # If kwargs are provided, update the settings
            self.config = self.get_settings()

            input_ids = self.load_template(prompt_messages)

            if self.enable_streaming:
                streamer = TextStreamer(self.tokenizer, skip_prompt=True)
                
                # outputs = self.model.generate(
                #     input_ids,
                #     generation_config=self.config,
                #     do_sample=True,
                # )
                
                _ = self.model.generate(
                    input_ids,
                    generation_config=self.config,
                    streamer=streamer,
                )
                
                # generated_text = self.tokenizer.decode(
                #     outputs[0][input_ids.shape[-1]:],
                #     skip_special_tokens=True
                # )
                
                generated_text = ""
                
                for text in streamer:
                    generated_text += text

                try:
                    index=len(generated_text)-generated_text[::-1].index(151668)
                except:
                    index=0
                
                generated_thinking = generated_text[:index]
                generated_text = generated_text[index:]
            else:
                generated_ids = self.model.generate(
                    input_ids,
                    generation_config=self.config,
                )
                
            return generated_text.strip()
        
    def get_settings(self):
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )

    def load_template(self, messages):
        if self.use_tools:
            return self.tokenizer.apply_chat_template(
                messages,
                tools=self.tools,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
                enable_thinking=self.enable_thinking
            )
        else:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
                enable_thinking=self.enable_thinking
            )
        
    def _generate_streaming(self, input_ids: Any | str | list[int] | list[str] | list[list[int]] | BatchEncoding) -> str | Generator[str, Any, None]:
        """
        Generate text in chunks to avoid very long single-pass generations.
        Calls mlx_lm_generate repeatedly, appending the continuation each time.
        Stops if EOS or no progress is made.
        """

        streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        _ = self.model.generate(
            input_ids,
            generation_config=self.config,
            streamer=streamer,
        )

        generated_thinking = ""
        generated_text = ""
        temp = ""
        for response in streamer:
            print(response, end='', flush=True)
            if "<think>" in response:
                while "</think>" not in response:
                    temp.join(response)
                else:
                    temp.join(response)
                    _, generated_text = temp.split("</think>", 1)
                    yield generated_text.strip()
                
            else:
                generated_text.join(response)
                yield generated_text.strip()

@deprecated.deprecated(reason="TransformersVisionModelHandler is now merged with TransformersCausalModelHandler. Use TransformersUnifiedModelHandler instead.", version="1.0.0")
class TransformersVisionModelHandler(BaseVisionModelHandler):
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, image_input, **kwargs)

        self.max_new_tokens = self.max_tokens

        self.device = device

        set_seed(self.seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)
        self.load_model()

    def load_model(self):
        if self.image_input:
            self.processor = AutoProcessor.from_pretrained(self.local_model_path, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(self.local_model_path, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(self.local_model_path, trust_remote_code=True)

        if self.local_lora_model_path and os.path.exists(self.local_lora_model_path):
            self.model = PeftModel.from_pretrained(self.model, self.local_lora_model_path)

    def generate_answer(self, history, **kwargs):
        try:
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]

            self.config = self.get_settings()

            inputs = self.load_template(prompt_messages)
            
            streamer = TextStreamer(self.processor, skip_prompt=True)

            # outputs = self.model.generate(
            #     **inputs,
            #     generation_config=self.config,
            #     do_sample=True,
            # )
            
            _ = self.model.generate(
                **inputs,
                generation_config=self.config,
                streamer=streamer
            )

            # generated_text = self.processor.decode(
            #     outputs[0],
            #     skip_special_tokens=True
            # )
            
            generated_text = ""
            
            for text in streamer:
                generated_text += text

            return generated_text.strip()
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}")
            return f"Error generating answer: {str(e)}\n\n{traceback.format_exc()}"

    def get_settings(self):
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )

    def load_template(self, messages):
        if self.image_input:
            if self.use_tools:
                return self.processor.apply_chat_template(
                    messages,
                    tools=self.tools,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False,
                    enable_thinking=self.enable_thinking
                )
            else:
                return self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False,
                    enable_thinking=self.enable_thinking
                )
        else:
            if self.use_tools:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tools=self.tools,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False,
                    enable_thinking=self.enable_thinking
                )
            else:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False,
                    enable_thinking=self.enable_thinking
                )
        
    def _generate_streaming(self, inputs:  Any | str | list[int] | list[str] | list[list[int]] | BatchEncoding):
        streamer = TextStreamer(self.processor, skip_prompt=True)

        _ = self.model.generate(
            **inputs,
            generation_config=self.config,
            streamer=streamer
        )

        generated_thinking = ""
        generated_text = ""
        temp = ""

        for response in streamer:
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

