from .transformers_handlers import (
    TransformersCausalModelHandler,
    TransformersVisionModelHandler,
    TransformersUnifiedModelHandler,
)
from .gguf_handlers import GGUFCausalModelHandler
from .mlx_handlers import (
    MlxCausalModelHandler,
    MlxVisionModelHandler,
    MlxUnifiedModelHandler,
)
from .vllm_handlers import VllmCausalModelHandler

# Provider clients and the tool-calling loop now live in ai-companion-gateway.
# Re-exported here for backwards compatibility.
from ai_companion_gateway.provider.vllm import vLLMClientWrapper
from ai_companion_gateway.provider.lmstudio import LMStudioIntegrator
from ai_companion_gateway.provider.ollama import OllamaIntegrator
from .base_handlers import BaseModelHandler  # noqa: F401
from ai_companion_gateway.base_handlers import BaseAPIClientWrapper  # noqa: F401
from ai_companion_gateway.interfaces.tools import ToolCall, ToolExecutor, ToolResult, ToolSpec
from ai_companion_gateway.tool_calling import (
    build_tool_specs_from_mcp,
    provider_supports_tools,
    run_tool_loop,
    run_tool_loop_responses,
)

__all__ = [
    "TransformersCausalModelHandler",
    "TransformersVisionModelHandler",
    "TransformersUnifiedModelHandler",
    "GGUFCausalModelHandler",
    "MlxCausalModelHandler",
    "MlxVisionModelHandler",
    "MlxUnifiedModelHandler",
    "VllmCausalModelHandler",
    "vLLMClientWrapper",
    "LMStudioIntegrator",
    "OllamaIntegrator",
    "BaseAPIClientWrapper",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "ToolExecutor",
    "run_tool_loop",
    "run_tool_loop_responses",
    "build_tool_specs_from_mcp",
    "provider_supports_tools",
]

__version__ = "0.8.0"
