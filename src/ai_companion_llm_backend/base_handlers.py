from abc import abstractmethod
from typing import Any, BinaryIO, List, Optional, Union
from pathlib import Path
import os
import platform
import warnings

from typing_extensions import Buffer
import torch.nn
from PIL import Image, ImageFile
from pydantic import SecretStr

from ai_companion_gateway.base_handlers import (  # noqa: F401  (re-export for backwards compatibility)
    BaseAPIClientWrapper,
    BaseModel,
    UnsupportedTaskError,
)
from ai_companion_gateway.interfaces.tools import ToolExecutor, ToolSpec  # noqa: F401

try:
    import mlx.nn
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("langchain_mlx is not installed. Please install it to use MLX features.", UserWarning)
    else:
        pass

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, GenerationMixin, PreTrainedModel, AutoModel, AutoProcessor, ProcessorMixin, AutoConfig, PretrainedConfig, GenerationConfig, PythonBackend, TokenizersBackend
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES
from peft import PeftModel
from llama_cpp import Llama

try:
    from mlx_lm.tokenizer_utils import TokenizerWrapper, SPMStreamingDetokenizer, BPEStreamingDetokenizer, NaiveStreamingDetokenizer
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("langchain_mlx is not installed. Please install it to use MLX features.", UserWarning)
    else:
        pass


FileInput = Union[str, "os.PathLike[str]", list[str], Path, "os.PathLike[Path]", list[Path]]


class BaseModelHandler(BaseModel):
    def __init__(self, model_id: str, lora_model_id: str | None = None, use_langchain: bool = True, image_input: str | List[str] | Image.Image | ImageFile.ImageFile | Any | None = None, audio_input: str | List[str] | Any | None = None, video_input: str | List[str] | Any | None = None, **kwargs):
        super().__init__(use_langchain, image_input, audio_input, video_input, **kwargs)
        self.model_id: str = model_id
        self.lora_model_id: str | None = lora_model_id
        self.config: AutoConfig | PretrainedConfig | Any | None = None
        self.generation_config: GenerationConfig | Any | None = None
        self.config_json = dict(kwargs.get("config_json", {}))
        self.local_model_path: str = os.path.join("./models/llm", model_id)
        self.local_lora_model_path: str | None = os.path.join("./models/llm/loras", lora_model_id) if lora_model_id else None

        self.processor: AutoProcessor | ProcessorMixin | Any | None = None
        self.tokenizer: AutoTokenizer | PythonBackend | TokenizersBackend | PreTrainedTokenizerBase | TokenizerWrapper | type[SPMStreamingDetokenizer | BPEStreamingDetokenizer | NaiveStreamingDetokenizer] | Any | None = None
        self.model: torch.nn.Module | mlx.nn.Module | PreTrainedModel | GenerationMixin | AutoModelForCausalLM | AutoModelForImageTextToText | AutoModel | PeftModel | Llama | Any | None = None

        # Transformer architecture groups used to pick a message-processing
        # strategy; only local inference handlers need these.
        self.check_is_causal_lm = list(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values())
        self.check_is_multimodal_lm = list(MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES.values())
        self.check_is_image_text_to_text = list(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values())
        self.check_is_any_to_any = list(set(self.check_is_multimodal_lm) - set(self.check_is_image_text_to_text))

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate_answer(self, history: list[dict[str, str | list[dict[str, str | Image.Image | Any]] | Any]], **kwargs):
        pass

    @abstractmethod
    def get_settings(self):
        pass

    @abstractmethod
    def load_template(self, messages: list[dict[str, str | list[dict[str, str]]]]):
        pass

    def process_messages(self, history: list[dict[str, str | list[dict[str, str]]]]):
        messages = []
        if self.images is None:
            self.images = []

        if self.arch in self.check_is_causal_lm:
            for msg in history[:-1]:
                if msg["role"] == "system":
                    messages.append({"role": "system", "content": msg["content"]})
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                if msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})

        else:
            for msg in history[:-1]:
                if msg["role"] == "system":
                    system_message = []
                    for content in msg["content"]:
                        if content["type"] == "text":
                            system_message.append({"type": "text", "text": content["text"]})
                    messages.append({"role": "system", "content": system_message})
                if msg["role"] == "user":
                    user_history = []
                    for content in msg["content"]:
                        if content["type"] == "text":
                            user_history.append({"type": "text", "text": content["text"]})
                        elif content["type"] == "image":
                            user_history.append({"type": "image"})
                            self.images.append(self.decode_base64(content["url"]))
                        elif content["type"] == "video":
                            user_history.append({"type": "video"})
                        elif content["type"] == "audio":
                            user_history.append({"type": "audio"})
                    messages.append({"role": "user", "content": user_history})
                if msg["role"] == "assistant":
                    ai_history = []
                    for content in msg["content"]:
                        if content["type"] == "text":
                            ai_history.append({"type": "text", "text": content["text"]})
                    messages.append({"role": "assistant", "content": ai_history})

        user_message = history[-1]

        if self.arch in self.check_is_causal_lm:
            user_inputs = user_message["content"]
        else:
            user_inputs = []
            for content in user_message["content"]:
                if content["type"] == "text":
                    user_inputs.append({"type": "text", "text": content["text"]})
                elif content["type"] == "image":
                    user_inputs.append({"type": "image"})
                    self.images.append(self.image_input)

        messages.append({"role": "user", "content": user_inputs})
        return messages

    def generate_chat_title(self, first_message: str, image_input=None) -> str:
        """
        Generate a chat title based on the first message.
        Uses the API to summarize the message into a short title.
        """
        prompt = f"Create a very short chat title (max 5-7 words) that summarizes the following message. Reply with ONLY the title, no quotes or extra text:\n\n{first_message}"

        history = [{"role": "system", "content": "You are a helpful assistant that creates concise chat titles."}, {"role": "user", "content": prompt}]

        # Temporarily reduce max_tokens for title generation
        original_max_tokens = self.max_tokens
        self.max_tokens = 30

        try:
            title = self.generate_answer(history)
            # Clean up the title
            title = title.strip().strip("\"'").strip()
            # Truncate if too long
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        except Exception as e:
            from ai_companion_core.logging import logger

            logger.warning(f"Failed to generate chat title via API: {e}")
            # Fallback to truncated first message
            if isinstance(first_message, str):
                return first_message[:50] + "..." if len(first_message) > 50 else first_message
            return "New Chat"
        finally:
            self.max_tokens = original_max_tokens

    def decode_base64(self, b64_str: str):
        import base64
        from io import BytesIO

        if "," in b64_str:
            b64_str = b64_str.split(",")[1]

        image_bytes = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_bytes)).convert("RGB")


class BaseCausalModelHandler(BaseModelHandler):
    def __init__(self, model_id: str, lora_model_id: str | None = None, use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, None, **kwargs)

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate_answer(self, history: list[dict[str, str | list[dict[str, str | Image.Image | Any]] | Any]], **kwargs):
        pass

    @abstractmethod
    def get_settings(self):
        pass

    @abstractmethod
    def load_template(self, messages: list[dict[str, str | list[dict[str, str]]]]):
        pass


class BaseVisionModelHandler(BaseModelHandler):
    def __init__(self, model_id: str, lora_model_id: str | None = None, use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, image_input, **kwargs)

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate_answer(self, history: list[dict[str, str | list[dict[str, str | Image.Image | Any]] | Any]], **kwargs):
        pass

    @abstractmethod
    def get_settings(self):
        pass

    @abstractmethod
    def load_template(self, messages: list[dict[str, str | list[dict[str, str]]]]):
        pass


class BaseOmniModelHandler(BaseModelHandler):
    def __init__(self, model_id: str, lora_model_id: str | None = None, use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, audio_input: str | List[str] | Any | None = None, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, image_input, audio_input, **kwargs)

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate_answer(self, history: list[dict[str, str | list[dict[str, str | Image.Image | Any]] | Any]], **kwargs):
        pass

    @abstractmethod
    def get_settings(self):
        pass

    @abstractmethod
    def load_template(self, messages: list[dict[str, str | list[dict[str, str]]]]):
        pass


__all__ = [
    "BaseModel",
    "BaseAPIClientWrapper",
    "UnsupportedTaskError",
    "FileInput",
    "BaseModelHandler",
    "BaseCausalModelHandler",
    "BaseVisionModelHandler",
    "BaseOmniModelHandler",
]
