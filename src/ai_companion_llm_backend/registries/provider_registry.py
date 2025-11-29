from __future__ import annotations
from typing import Dict, Iterable, Optional, List
from ..interfaces.providers import ProviderSpec, ProviderCapabilities, PROVIDER_ID as ProviderId


class ProviderRegistry:
    _providers: Dict[ProviderId, ProviderSpec] = {}

    @classmethod
    def register(cls, spec: ProviderSpec):
        if spec.id in cls._providers:
            raise ValueError(f"Provider already registered: {spec.id}")
        cls._providers[spec.id] = spec

    @classmethod
    def get(cls, provider_id: ProviderId) -> ProviderSpec:
        try:
            return cls._providers[provider_id]
        except KeyError:
            raise KeyError(f"Unknown provider: {provider_id}")

    @classmethod
    def list(cls) -> List[ProviderSpec]:
        return list(cls._providers.values())

    @classmethod
    def list_ids(cls) -> List[ProviderId]:
        return list(cls._providers.keys())


# ---- 실제 등록(예시) ----
ProviderRegistry.register(
    ProviderSpec(
        id="openai",
        display_name="OpenAI",
        requires_api_key=True,
        base_url_hint="https://api.openai.com/v1",
        default_kwargs={"temperature": 1.0, "top_p": 1.0},
        capabilities=ProviderCapabilities(
            chat=True, embeddings=True, vision=True, tools=True, json_mode=True, streaming=True
        ),
    )
)

ProviderRegistry.register(
    ProviderSpec(
        id="anthropic",
        display_name="Anthropic",
        requires_api_key=True,
        base_url_hint="https://api.anthropic.com/v1",
        default_kwargs={"temperature": 1.0},
        capabilities=ProviderCapabilities(
            chat=True, vision=True, tools=True, json_mode=False, embeddings=False, streaming=True
        ),
    )
)

ProviderRegistry.register(
    ProviderSpec(
        id="google-genai",
        display_name="Google AI",
        requires_api_key=True,
        base_url_hint="https://generativelanguage.googleapis.com/v1beta",
        default_kwargs={"temperature": 1.0, "top_p": 1, "top_k": 20},
        capabilities=ProviderCapabilities(
            chat=True, vision=True, tools=True, json_mode=False, embeddings=False, streaming=True
        ),
    )
)

ProviderRegistry.register(
    ProviderSpec(
        id="ollama",
        display_name="Ollama (Local)",
        requires_api_key=False,
        base_url_hint="http://localhost:11434",
        default_kwargs={"temperature": 0.7, "top_p": 1, "top_k": 20, "repeat_penalty": 1.05},
        capabilities=ProviderCapabilities(
            chat=True, vision=False, tools=False, embeddings=True, streaming=True
        ),
    )
)