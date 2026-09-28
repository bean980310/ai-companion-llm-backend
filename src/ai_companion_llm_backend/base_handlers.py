from abc import ABC, abstractmethod
from functools import partial
from typing import Any, BinaryIO, Union, List, Optional, override
from pathlib import Path
from io import BytesIO
import os
import platform
import warnings
import base64
import random

from typing_extensions import Buffer
import torch.nn
from PIL import Image, ImageFile
from mem0 import Memory
from pydantic import SecretStr

from .interfaces.tools import ToolExecutor, ToolSpec
from .tool_calling import DEFAULT_MAX_TOOL_ITERATIONS, run_tool_loop

try:
    import mlx.nn
except ImportError:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        warnings.warn("langchain_mlx is not installed. Please install it to use MLX features.", UserWarning)
    else:
        pass

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, GenerationMixin, PreTrainedModel, AutoModelForImageTextToText, AutoModel, AutoProcessor, ProcessorMixin, AutoConfig, PretrainedConfig, GenerationConfig, PythonBackend, TokenizersBackend
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


class UnsupportedTaskError(Exception):
    pass


FileInput = Union[str, "os.PathLike[str]", list[str], Path, "os.PathLike[Path]", list[Path]]


class BaseModel(ABC):
    def __init__(self, use_langchain: bool = True, image_input: Optional[str | List[str] | Image.Image | List[Image.Image] | ImageFile.ImageFile | List[ImageFile.ImageFile] | Any] = None, audio_input: Optional[str | List[str] | Any] = None, video_input: Optional[str | List[str] | Any] = None, **kwargs):
        self.use_langchain = use_langchain
        self.images: Optional[str | list[str] | Image.Image | list[Image.Image]] = None
        self.audios: Optional[str | list[str]] = None
        self.videos: Optional[str | list[str]] = None
        self.image_input = image_input
        self.audio_input = audio_input
        self.video_input = video_input
        self.enable_streaming = bool(kwargs.get("enable_streaming", False))
        self.use_tools = bool(kwargs.get("use_tools", False))
        # Raw tool schemas (legacy apply_chat_template path). May be a list of
        # OpenAI-style dicts or backend ToolSpec objects.
        self.tools: list = list(kwargs.get("tools", []))

        # Provider-agnostic tool-calling loop settings.
        # ``tool_specs`` are normalized ToolSpec objects; ``tool_executor`` is a
        # callable ``(name, arguments) -> ToolResult | str`` invoked for each
        # requested tool call. ``max_tool_iterations`` caps the loop.
        self.tool_specs: list[ToolSpec] = list(kwargs.get("tool_specs", []))
        self.tool_executor: ToolExecutor | None = kwargs.get("tool_executor", None)
        self.max_tool_iterations = int(kwargs.get("max_tool_iterations", DEFAULT_MAX_TOOL_ITERATIONS))

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
        self.extra_kwargs = dict[str, Any](kwargs.get("extra_kwargs", {}))

        self.arch = None

        self.check_is_causal_lm = list(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values())
        self.check_is_multimodal_lm = list(MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES.values())
        self.check_is_image_text_to_text = list(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values())
        self.check_is_any_to_any = list(set(self.check_is_multimodal_lm) - set(self.check_is_image_text_to_text))
        self.memory = Memory()

        if self.seed == -1:
            self.seed = random.randint(0, 4294967295)

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate_answer(self, history: list[dict[str, str | list[dict[str, str | Image.Image | Any]] | Any]], **kwargs):
        pass


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
        self.tokenizer: AutoTokenizer | PythonBackend | TokenizersBackend | PreTrainedTokenizerBase | TokenizerWrapper | type[SPMStreamingDetokenizer | BPEStreamingDetokenizer | NaiveStreamingDetokenizer] | partial[SPMStreamingDetokenizer] | Any | None = None
        self.model: torch.nn.Module | mlx.nn.Module | PreTrainedModel | GenerationMixin | AutoModelForCausalLM | AutoModelForImageTextToText | AutoModel | PeftModel | Llama | Any | None = None
        self.memory = Memory()

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


class BaseAPIClientWrapper(BaseModel):
    def __init__(self, selected_model: str, api_key: str | SecretStr | None = None, use_langchain: bool = True, image_input: str | Image.Image | ImageFile.ImageFile | BinaryIO | Buffer | os.PathLike[str] | Any | None = None, audio_input: str | List[str] | Any | None = None, **kwargs):
        super().__init__(use_langchain, image_input, audio_input, **kwargs)
        self.model = selected_model
        self.api_key = api_key

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate_answer(self, history: list[dict[str, str | list[dict[str, str | Image.Image | Any]] | Any]], **kwargs):
        pass

    # ------------------------------------------------------------------
    # Provider-agnostic tool calling
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_tool_specs(tools: list) -> list[ToolSpec]:
        """Coerce a mixed list of ToolSpec/dict/object schemas into ToolSpec."""
        specs: list[ToolSpec] = []
        for tool in tools:
            if isinstance(tool, ToolSpec):
                specs.append(tool)
                continue
            if isinstance(tool, dict):
                # Either an OpenAI function schema or a flat tool dict.
                if tool.get("type") == "function" and "function" in tool:
                    fn = tool["function"]
                    specs.append(
                        ToolSpec(
                            name=fn.get("name", ""),
                            description=fn.get("description", ""),
                            parameters=fn.get("parameters", {"type": "object", "properties": {}}),
                        )
                    )
                else:
                    specs.append(
                        ToolSpec(
                            name=tool.get("name", ""),
                            description=tool.get("description", ""),
                            parameters=tool.get("parameters") or tool.get("input_schema") or {"type": "object", "properties": {}},
                            server=tool.get("server_name") or tool.get("server"),
                        )
                    )
                continue
            # Duck-typed objects (e.g. MCPTool)
            schema = getattr(tool, "input_schema", None) or getattr(tool, "parameters", None) or {"type": "object", "properties": {}}
            if isinstance(schema, list):
                schema = {"type": "object", "properties": {}}
            specs.append(
                ToolSpec(
                    name=getattr(tool, "name", str(tool)),
                    description=getattr(tool, "description", "") or "",
                    parameters=schema,
                    server=getattr(tool, "server_name", None),
                )
            )
        return specs

    def get_tool_specs(self) -> list[ToolSpec]:
        """Resolved, normalized tool specs for this wrapper."""
        if self.tool_specs:
            return self.tool_specs
        return self._normalize_tool_specs(self.tools)

    def can_run_tools(self) -> bool:
        """Whether a tool-calling loop can be executed."""
        return bool(self.get_tool_specs()) and callable(self.tool_executor)

    @staticmethod
    def content_to_openai(content: Any) -> Any:
        """Convert internal multimodal content into OpenAI chat content format."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)
        parts: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                parts.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                if item.get("type") == "text" or "text" in item:
                    parts.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "image":
                    url = item.get("url") or item.get("image_url")
                    if url:
                        parts.append({"type": "image_url", "image_url": {"url": url}})
        if not parts:
            return ""
        if len(parts) == 1 and parts[0]["type"] == "text":
            return parts[0]["text"]
        return parts

    def history_to_openai_messages(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build OpenAI chat messages from the internal history format."""
        messages: list[dict[str, Any]] = []
        for msg in history:
            role = msg.get("role")
            if role not in ("system", "user", "assistant"):
                continue
            messages.append({"role": role, "content": self.content_to_openai(msg.get("content", ""))})
        return messages

    def run_tool_calling(
        self,
        history: list[dict[str, Any]],
        *,
        extra_body: dict[str, Any] | None = None,
        system_tool_hint: str | None = None,
        max_tokens: int | None = None,
        **completion_kwargs: Any,
    ) -> str:
        """
        Execute the provider-agnostic tool-calling loop against
        ``self.client.chat.completions`` (OpenAI-compatible endpoint).

        Subclasses that use a non-OpenAI client should override this method.
        """
        if not self.can_run_tools():
            raise RuntimeError("Tool calling is not configured (missing tool_specs or tool_executor)")

        return run_tool_loop(
            client=self.client,
            model=self.model,
            messages=self.history_to_openai_messages(history),
            tool_specs=self.get_tool_specs(),
            executor=self.tool_executor,
            temperature=self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            max_iterations=self.max_tool_iterations,
            extra_body=extra_body,
            system_tool_hint=system_tool_hint,
            **completion_kwargs,
        ).strip()

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

    @staticmethod
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            if image_path.rsplit(".")[-1] == "jpg" or "jpeg":
                data_mime = "image/jpeg"
            elif image_path.rsplit(".")[-1] == "png":
                data_mime = "image/png"
            elif image_path.rsplit(".")[-1] == "webp":
                data_mime = "image/webp"
            elif image_path.rsplit(".")[-1] == "gif":
                data_mime = "image/gif"

            image = base64.b64encode(image_file.read()).decode("utf-8")

        return image, data_mime
