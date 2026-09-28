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
from .provider.vllm import vLLMClientWrapper
from .provider.lmstudio import LMStudioIntegrator
from .provider.ollama import OllamaIntegrator
# from .langchain_integrator.langchain import LangchainIntegrator

from .interfaces.tools import ToolCall, ToolExecutor, ToolResult, ToolSpec
from .tool_calling import (
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
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "ToolExecutor",
    "run_tool_loop",
    "run_tool_loop_responses",
    "build_tool_specs_from_mcp",
    "provider_supports_tools",
]

__version__ = "0.7.0"
