from __future__ import annotations

from typing import Dict, List, Optional, Callable
from ..interfaces.models import ModelSpec, ModelCapabilities
from ..interfaces.providers import PROVIDER_ID as ProviderId

ModelLoader = Callable[[ProviderId], List[ModelSpec]]

class ModelRegistry:
    _models_by_provider: Dict[ProviderId, Dict[str, ModelSpec]] = {}
    _loader: Optional[ModelLoader] = None
    _loaded_providers: set[ProviderId] = set()

    @classmethod
    def set_loader(cls, loader: ModelLoader):
        cls._loader = loader


    @classmethod
    def ensure_loaded(cls, provider_id: ProviderId):
        if provider_id in cls._loaded_providers:
            return

        if cls._loader is None:
            cls._loaded_providers.add(provider_id)
            return
        
        try:
            specs = cls._loader(provider_id)
        except Exception as e:
            cls._loaded_providers.add(provider_id)
            return

    @classmethod  
    def register(cls, spec: ModelSpec):
        bucket = cls._models_by_provider.setdefault(spec.provider_id, {})
        if spec.id in bucket:
            raise ValueError(f"Model already registered: {spec.provider_id}/{spec.id}")
        bucket[spec.id] = spec

    @classmethod
    def get(cls, provider_id: ProviderId, model_id: str) -> ModelSpec:
        cls.ensure_loaded(provider_id)
        try:
            return cls._models_by_provider[provider_id][model_id]
        except KeyError:
            raise KeyError(f"Unknown model: {provider_id}/{model_id}")

    @classmethod
    def list_models(
        cls,
        provider_id: ProviderId,
        *,
        require_chat: Optional[bool] = None,
        require_embeddings: Optional[bool] = None,
        require_vision: Optional[bool] = None,
        require_tools: Optional[bool] = None,
        require_json_mode: Optional[bool] = None,
        require_thinking_param: Optional[bool] = None,
    ) -> List[ModelSpec]:
        cls.ensure_loaded(provider_id)
        models = list(cls._models_by_provider.get(provider_id, {}).values())

        def ok(m: ModelSpec) -> bool:
            c = m.capabilities
            if require_chat is not None and c.chat != require_chat: return False
            if require_embeddings is not None and c.embeddings != require_embeddings: return False
            if require_vision is not None and c.vision != require_vision: return False
            if require_tools is not None and c.tools != require_tools: return False
            if require_json_mode is not None and c.json_mode != require_json_mode: return False
            if require_thinking_param is not None and c.thinking_param != require_thinking_param: return False
            return True

        return [m for m in models if ok(m)]


# ---- 실제 등록(예시) ----
ModelRegistry.register(
    ModelSpec(
        id="gpt-4.1-mini",
        provider_id="openai",
        display_name="GPT-4.1 mini",
        context_length=128_000,
        max_output_tokens=16_384,
        cost_input_per_1m=0.3,
        cost_output_per_1m=1.2,
        capabilities=ModelCapabilities(
            chat=True, vision=True, tools=True, json_mode=True, thinking_param=False
        )
    )
)

ModelRegistry.register(
    ModelSpec(
        id="claude-3.7-sonnet",
        provider_id="anthropic",
        display_name="Claude 3.7 Sonnet",
        context_length=200_000,
        max_output_tokens=8_192,
        capabilities=ModelCapabilities(
            chat=True, vision=True, tools=True, json_mode=False, thinking_param=True
        )
    )
)

ModelRegistry.register(
    ModelSpec(
        id="qwen2.5-14b-instruct",
        provider_id="ollama",
        display_name="Qwen2.5 14B Instruct (Ollama)",
        context_length=32_768,
        capabilities=ModelCapabilities(
            chat=True, embeddings=False, vision=False, tools=False, json_mode=False
        ),
        extra={"quant": "Q4_K_M"}
    )
)