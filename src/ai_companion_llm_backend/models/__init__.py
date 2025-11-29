from ..utils import get_all_local_models

local_models_data = get_all_local_models()
transformers_local = local_models_data["transformers"]
gguf_local = local_models_data["gguf"]
mlx_local = local_models_data["mlx"]

PROVIDER_LIST = ["openai", "anthropic", "google-genai", "perplexity", "xai", 'mistralai', "openrouter", "hf-inference", "ollama", "lmstudio", "self-provided"]

REASONING_BAN = ["non-reasoning"]
REASONING_CONTROLABLE = ["qwen3", "gpt-oss", "gpt-5", "claude-sonnet-4", "claude-opus-4", "claude-haiku-4", "gemini-2.5-flash", "gemini-3"]
REASONING_KWD = ["reasoning", "qwq", "r1", "deepseek-r1", "think", "thinking", "deephermes", "hermes-4", "magistral", "o1", "o3", "o4", "gemini-2.5-pro"] + REASONING_CONTROLABLE
