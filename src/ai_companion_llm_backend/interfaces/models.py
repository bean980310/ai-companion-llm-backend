from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

@dataclass(frozen=True)
class ModelCapabilities:
    chat: bool = True
    embeddings: bool = False
    vision: bool = False
    tools: bool = False
    json_mode: bool = False
    thinking_param: bool = False   # enable_thinking 같은 특화 옵션 여부

@dataclass(frozen=True)
class ModelSpec:
    id: str                      # "gpt-4.1-mini" 같은 내부 키
    provider_id: str             # ProviderId와 맞춰
    display_name: str            # UI 표시명
    context_length: int
    max_output_tokens: Optional[int] = None
    cost_input_per_1m: Optional[float] = None
    cost_output_per_1m: Optional[float] = None
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    extra: Dict[str, Any] = field(default_factory=dict)  # 모델 특화 메타