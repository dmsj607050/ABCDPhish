"""Serializable architecture and preprocessing configuration."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AIDRConfig:
    encoder_backend: str = "lightweight"
    stage: str = "full"  # baseline -> disentangled -> relations -> full
    hidden_dim: int = 128
    num_heads: int = 4
    encoder_layers: int = 2
    relation_layers: int = 2
    num_queries: int = 2
    dropout: float = 0.1
    url_max_length: int = 256
    html_max_tokens: int = 512
    max_dom_nodes: int = 128
    max_targets: int = 8
    image_size: int = 224
    patch_size: int = 16
    html_model_name: str = "answerdotai/ModernBERT-base"
    visual_model_name: str = "google/siglip2-base-patch16-224"
    freeze_backbones: bool = True
    local_files_only: bool = False
    num_brands: int = 0
    intent_names: tuple = ("login", "credentials", "payment", "download")
    role_bias_strength: float = 1.0
    lambda_app: float = 0.2
    lambda_intent: float = 0.2
    lambda_align: float = 0.1
    lambda_relation: float = 0.0

    def __post_init__(self):
        if self.encoder_backend not in {"lightweight", "pretrained"}:
            raise ValueError("encoder_backend must be lightweight or pretrained")
        if self.stage not in {"baseline", "disentangled", "relations", "full"}:
            raise ValueError("Unknown ablation stage")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        for name in ("hidden_dim", "num_heads", "encoder_layers", "relation_layers", "num_queries",
                     "url_max_length", "html_max_tokens", "max_dom_nodes", "max_targets", "image_size", "patch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if not self.intent_names:
            raise ValueError("intent_names cannot be empty")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
