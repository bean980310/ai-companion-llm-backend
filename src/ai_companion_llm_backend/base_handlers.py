from abc import ABC, abstractmethod
from functools import partial
from typing import Any, BinaryIO, Union, List
import os
import platform
import warnings
import base64
import random

from typing_extensions import Buffer
import torch.nn
from PIL import Image, ImageFile

try:
    import mlx.nn
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("langchain_mlx is not installed. Please install it to use MLX features.", UserWarning)
    else:
        pass
    
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase, GenerationMixin, PreTrainedModel, AutoModelForImageTextToText, AutoModel, AutoProcessor, ProcessorMixin, AutoConfig, PretrainedConfig, GenerationConfig
from peft import PeftModel
from llama_cpp import Llama

try:
    from mlx_lm.tokenizer_utils import TokenizerWrapper, SPMStreamingDetokenizer, BPEStreamingDetokenizer, NaiveStreamingDetokenizer
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("langchain_mlx is not installed. Please install it to use MLX features.", UserWarning)
    else:
        pass

class BaseModel(ABC):
    def __init__(self, use_langchain: bool = True, image_input: str | List[str] | Image.Image | List[Image.Image] | ImageFile.ImageFile | List[ImageFile.ImageFile] | Any | None = None, audio_input: str | List[str] | Any | None = None, **kwargs):
        self.use_langchain = use_langchain
        self.image_input = image_input
        self.audio_input = audio_input
        self.enable_streaming = bool(kwargs.get("enable_streaming", False))

        self.max_tokens = int(kwargs.get("max_tokens", 4096))
        self.max_length = int(kwargs.get("max_length", -1))
        self.seed = int(kwargs.get("seed", 42))
        self.temperature = float(kwargs.get("temperature", 1.0))
        self.top_k = int(kwargs.get("top_k", 50))
        self.top_p = float(kwargs.get("top_p", 1.0))
        self.repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))
        self.enable_thinking = bool(kwargs.get("enable_thinking", False))
        self.enable_langchain = False

        self.langchain_integrator = None

        self.use_chunking = bool(kwargs.get("use_chunking", False))
        self.chunk_size = int(kwargs.get("chunk_size", 1024))

        if self.seed == -1:
            self.seed = random.randint(0, 4294967295)

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate_answer(self, history, **kwargs):
        pass

class BaseModelHandler(BaseModel):
    def __init__(self, model_id: str, lora_model_id: str | None = None, use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, audio_input: str | List[str] | Any | None = None, **kwargs):
        super().__init__(use_langchain, image_input, audio_input, **kwargs)
        self.model_id: str = model_id
        self.lora_model_id: str | None = lora_model_id
        self.config: AutoConfig | PretrainedConfig | GenerationConfig | Any | None = None
        self.local_model_path: str = os.path.join("./models/llm", model_id)
        self.local_lora_model_path: str | None = os.path.join("./models/llm/loras", lora_model_id) if lora_model_id else None

        self.processor: AutoProcessor | ProcessorMixin | Any | None = None
        self.tokenizer: AutoTokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast | PreTrainedTokenizerBase | TokenizerWrapper | type[SPMStreamingDetokenizer] | partial[SPMStreamingDetokenizer] | type[BPEStreamingDetokenizer] | type[NaiveStreamingDetokenizer] | Any | None = None
        self.model: torch.nn.Module | mlx.nn.Module | PreTrainedModel | GenerationMixin | AutoModelForCausalLM | AutoModelForImageTextToText | AutoModel | PeftModel | Llama | Any | None = None

    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def generate_answer(self, history, **kwargs):
        pass
    
    @abstractmethod
    def get_settings(self):
        pass
    
    @abstractmethod
    def load_template(self, messages):
        pass

    def generate_chat_title(self, first_message: str, image_input=None) -> str:
        """
        Generate a chat title based on the first message.
        Uses the API to summarize the message into a short title.
        """
        prompt = (
            "Create a very short chat title (max 5-7 words) that summarizes the following message. "
            "Reply with ONLY the title, no quotes or extra text:\n\n"
            f"{first_message}"
        )

        history = [
            {"role": "system", "content": "You are a helpful assistant that creates concise chat titles."},
            {"role": "user", "content": prompt}
        ]

        # Temporarily reduce max_tokens for title generation
        original_max_tokens = self.max_tokens
        self.max_tokens = 30

        try:
            title = self.generate_answer(history)
            # Clean up the title
            title = title.strip().strip('"\'').strip()
            # Truncate if too long
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        except Exception as e:
            from .logging import logger
            logger.warning(f"Failed to generate chat title via API: {e}")
            # Fallback to truncated first message
            if isinstance(first_message, str):
                return first_message[:50] + "..." if len(first_message) > 50 else first_message
            return "New Chat"
        finally:
            self.max_tokens = original_max_tokens

class BaseCausalModelHandler(BaseModelHandler):
    def __init__(self, model_id: str, lora_model_id: str | None = None, use_langchain: bool = True, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, None, **kwargs)
        
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def generate_answer(self, history, **kwargs):
        pass
    
    @abstractmethod
    def get_settings(self):
        pass
    
    @abstractmethod
    def load_template(self, messages):
        pass

class BaseVisionModelHandler(BaseModelHandler):
    def __init__(self, model_id: str, lora_model_id: str | None = None, use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, image_input, **kwargs)
        
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def generate_answer(self, history, **kwargs):
        pass
    
    @abstractmethod
    def get_settings(self):
        pass
    
    @abstractmethod
    def load_template(self, messages):
        pass

class BaseMultimodalModelHandler(BaseModelHandler):
    def __init__(self, model_id: str, lora_model_id: str | None = None, use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | Any | None = None, audio_input: str | List[str] | Any | None = None, **kwargs):
        super().__init__(model_id, lora_model_id, use_langchain, image_input, audio_input, **kwargs)
        
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def generate_answer(self, history, **kwargs):
        pass
    
    @abstractmethod
    def get_settings(self):
        pass
    
    @abstractmethod
    def load_template(self, messages):
        pass

class BaseAPIClientWrapper(BaseModel):
    def __init__(self, selected_model: str, api_key: str | None = None , use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | BinaryIO | Buffer | os.PathLike[str] | Any | None = None, audio_input: str | List[str] | Any | None = None, **kwargs):
        super().__init__(use_langchain, image_input, audio_input, **kwargs)
        self.model = selected_model
        self.api_key = api_key

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate_answer(self, history, **kwargs):
        pass

    def generate_chat_title(self, first_message: str, image_input=None) -> str:
        """
        Generate a chat title based on the first message.
        Uses the API to summarize the message into a short title.
        """
        prompt = (
            "Create a very short chat title (max 5-7 words) that summarizes the following message. "
            "Reply with ONLY the title, no quotes or extra text:\n\n"
            f"{first_message}"
        )

        history = [
            {"role": "system", "content": "You are a helpful assistant that creates concise chat titles."},
            {"role": "user", "content": prompt}
        ]

        # Temporarily reduce max_tokens for title generation
        original_max_tokens = self.max_tokens
        self.max_tokens = 30

        try:
            title = self.generate_answer(history)
            # Clean up the title
            title = title.strip().strip('"\'').strip()
            # Truncate if too long
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        except Exception as e:
            from .logging import logger
            logger.warning(f"Failed to generate chat title via API: {e}")
            # Fallback to truncated first message
            if isinstance(first_message, str):
                return first_message[:50] + "..." if len(first_message) > 50 else first_message
            return "New Chat"
        finally:
            self.max_tokens = original_max_tokens

    @staticmethod
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            if image_path.rsplit('.')[-1] == "jpg" or "jpeg":
                data_mime="image/jpeg"
            elif image_path.rsplit('.')[-1] == "png":
                data_mime="image/png"
            elif image_path.rsplit('.')[-1] == "webp":
                data_mime="image/webp"
            elif image_path.rsplit('.')[-1] == "gif":
                data_mime="image/gif"

            image = base64.b64encode(image_file.read()).decode("utf-8")

        return image, data_mime