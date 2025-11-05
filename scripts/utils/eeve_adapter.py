"""
EEVE-style Adapter Modules
얇은 추가 레이어를 Transformer 블록에 삽입하여 한국어 적응
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdapterLayer(nn.Module):
    """
    EEVE-style Adapter Layer

    Bottleneck 구조:
    hidden_size -> down_size -> activation -> up_size -> hidden_size
    """

    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 256,
        dropout: float = 0.1,
        init_scale: float = 0.01
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.adapter_size = adapter_size

        # Down projection
        self.down_proj = nn.Linear(hidden_size, adapter_size, bias=True)

        # Up projection
        self.up_proj = nn.Linear(adapter_size, hidden_size, bias=True)

        # Activation
        self.activation = nn.GELU()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Gate (learnable scaling)
        self.gate = nn.Parameter(torch.zeros(1))

        # 작은 값으로 초기화 (안정성)
        nn.init.normal_(self.down_proj.weight, std=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=init_scale)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            adapted_hidden_states: [batch_size, seq_len, hidden_size]
        """
        residual = hidden_states

        # Adapter forward
        x = self.down_proj(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)

        # Gated residual connection
        output = residual + self.gate * x

        return output


class GatedAdapter(nn.Module):
    """
    Gated Adapter with separate gates for different pathways
    """

    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 256,
        dropout: float = 0.1,
        init_scale: float = 0.01
    ):
        super().__init__()

        # Adapter
        self.down_proj = nn.Linear(hidden_size, adapter_size, bias=True)
        self.up_proj = nn.Linear(adapter_size, hidden_size, bias=True)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Gating network
        self.gate_linear = nn.Linear(hidden_size, 1, bias=True)

        # 초기화
        nn.init.normal_(self.down_proj.weight, std=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=init_scale)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.zeros_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, -3.0)  # 초기에는 거의 닫힘

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        # Adapter forward
        x = self.down_proj(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)

        # Dynamic gating
        gate = torch.sigmoid(self.gate_linear(hidden_states))

        # Gated combination
        output = residual + gate * x

        return output


class ParallelAdapter(nn.Module):
    """
    Parallel Adapter (병렬 구조)
    원본 블록과 어댑터를 병렬로 실행 후 결합
    """

    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 256,
        dropout: float = 0.1,
        init_scale: float = 0.01
    ):
        super().__init__()

        # Adapter branch
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, adapter_size, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_size, hidden_size, bias=True)
        )

        # Mixing weight
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # 초기화
        for module in self.adapter:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=init_scale)
                nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Parallel paths
        adapter_output = self.adapter(hidden_states)

        # Weighted combination
        output = hidden_states + self.alpha * adapter_output

        return output


class HierarchicalAdapter(nn.Module):
    """
    Hierarchical Adapter (Phase B용 단순 버전)

    2-layer 구조:
    - Layer 1: Language-specific (Korean focused)
    - Layer 2: Task-specific (embedding)

    Full version (cross-lingual, dynamic routing 등)은 나중에
    """

    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 256,
        dropout: float = 0.1,
        init_scale: float = 0.01
    ):
        super().__init__()

        # Layer 1: Language-specific adapter (Korean)
        self.language_adapter = nn.Sequential(
            nn.Linear(hidden_size, adapter_size, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_size, hidden_size, bias=True)  # 수정: hidden_size 출력
        )

        # Layer 2: Task-specific adapter (Embedding)
        self.task_adapter = nn.Sequential(
            nn.Linear(hidden_size, adapter_size // 2, bias=True),  # 수정: hidden_size 입력
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_size // 2, hidden_size, bias=True)
        )

        # Learnable gates
        self.gate1 = nn.Parameter(torch.zeros(1))
        self.gate2 = nn.Parameter(torch.zeros(1))

        # 초기화
        for module in [self.language_adapter, self.task_adapter]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=init_scale)
                    nn.init.zeros_(layer.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        2-layer hierarchical forward

        Args:
            hidden_states: [B, L, hidden_size]

        Returns:
            output: [B, L, hidden_size]
        """
        residual = hidden_states

        # Layer 1: Language-specific (이제 차원이 맞음)
        lang_output = self.language_adapter(hidden_states)
        lang_output = hidden_states + self.gate1 * lang_output

        # Layer 2: Task-specific
        task_output = self.task_adapter(lang_output)
        output = residual + self.gate2 * task_output

        return output


def inject_adapters(
    model: nn.Module,
    adapter_type: str = "bottleneck",
    adapter_size: int = 256,
    dropout: float = 0.1,
    layer_indices: Optional[list] = None
) -> nn.Module:
    """
    Transformer 모델에 어댑터 주입

    Args:
        model: Transformer 모델
        adapter_type: "bottleneck", "gated", "parallel"
        adapter_size: 어댑터 hidden size
        dropout: Dropout rate
        layer_indices: 어댑터를 추가할 레이어 인덱스 (None이면 전체)

    Returns:
        어댑터가 추가된 모델
    """
    # Qwen 모델 구조 확인
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise ValueError("지원되지 않는 모델 구조입니다.")

    # 어댑터 타입 선택
    adapter_class = {
        "bottleneck": AdapterLayer,
        "gated": GatedAdapter,
        "parallel": ParallelAdapter,
        "hierarchical": HierarchicalAdapter
    }.get(adapter_type, AdapterLayer)

    # 레이어 인덱스 결정
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    # 각 레이어에 어댑터 추가
    for idx in layer_indices:
        if idx >= len(layers):
            continue

        layer = layers[idx]
        # Qwen3/Qwen2는 config.hidden_size 사용
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            hidden_size = model.config.hidden_size
        elif hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'embed_dim'):
            hidden_size = layer.self_attn.embed_dim
        else:
            hidden_size = 768  # default fallback

        # 어댑터 생성
        adapter = adapter_class(
            hidden_size=hidden_size,
            adapter_size=adapter_size,
            dropout=dropout
        )

        # 레이어의 dtype과 device를 감지하여 어댑터를 같은 dtype/device로 이동
        # 첫 번째 파라미터의 dtype과 device를 사용
        first_param = next(layer.parameters())
        adapter = adapter.to(dtype=first_param.dtype, device=first_param.device)

        # 레이어에 어댑터 추가
        layer.adapter = adapter

        # Forward hook 수정 (어댑터 적용)
        original_forward = layer.forward

        def forward_with_adapter(self, *args, **kwargs):
            output = original_forward(*args, **kwargs)

            # output이 tuple인 경우 (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
                hidden_states = self.adapter(hidden_states)
                return (hidden_states,) + output[1:]
            else:
                return self.adapter(output)

        layer.forward = forward_with_adapter.__get__(layer, type(layer))

    return model


def freeze_except_adapters(model: nn.Module, also_train_embeddings: bool = False):
    """
    어댑터를 제외한 모든 파라미터 프리징

    Args:
        model: 모델
        also_train_embeddings: 임베딩도 학습할지 여부
    """
    # 전체 freeze
    for param in model.parameters():
        param.requires_grad = False

    # 어댑터만 unfreeze
    for name, param in model.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True

    # 임베딩도 학습하는 경우
    if also_train_embeddings:
        for name, param in model.named_parameters():
            if 'embed_tokens' in name or 'lm_head' in name:
                param.requires_grad = True

    # 학습 가능한 파라미터 카운트
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"✓ Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    return model


def get_adapter_params(model: nn.Module) -> list:
    """어댑터 파라미터만 추출"""
    return [p for n, p in model.named_parameters() if 'adapter' in n and p.requires_grad]
