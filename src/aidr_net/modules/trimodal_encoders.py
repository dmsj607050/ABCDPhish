"""M2: URL Char-CNN/Transformer, structure-aware HTML, and visual patch encoding."""
import torch
from torch import nn

from .common import TokenTransformer


class URLCharacterEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.embedding = nn.Embedding(257, d, padding_idx=0)
        self.position = nn.Embedding(config.url_max_length, d)
        self.convolution = nn.Conv1d(d, d, kernel_size=3, padding=1)
        self.transformer = TokenTransformer(d, config.num_heads, config.encoder_layers, config.dropout)
        self.identity_projection = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, ids):
        mask = ids != 0
        values = self.embedding(ids)
        values = values + torch.nn.functional.gelu(self.convolution(values.transpose(1, 2)).transpose(1, 2))
        values += self.position(torch.arange(ids.shape[1], device=ids.device))
        return self.transformer(values, mask), mask


class StructureAwareHTMLEncoder(nn.Module):
    def __init__(self, config, backbone_config=None):
        super().__init__()
        self.pretrained = config.encoder_backend == "pretrained"
        d = config.hidden_dim
        if self.pretrained:
            from transformers import AutoConfig, AutoModel
            if backbone_config is None:
                self.backbone = AutoModel.from_pretrained(config.html_model_name, local_files_only=config.local_files_only)
            else:
                fields = dict(backbone_config)
                self.backbone = AutoModel.from_config(AutoConfig.for_model(fields.pop("model_type"), **fields))
            self.projection = nn.Linear(self.backbone.config.hidden_size, d)
            if config.freeze_backbones:
                self.backbone.requires_grad_(False)
        else:
            self.embedding = nn.Embedding(257, d, padding_idx=0)
            self.position = nn.Embedding(config.html_max_tokens, d)
        self.tag_embedding = nn.Embedding(32, d, padding_idx=0)
        self.depth_embedding = nn.Embedding(32, d, padding_idx=0)
        self.transformer = TokenTransformer(d, config.num_heads, config.encoder_layers, config.dropout)

    def train(self, mode=True):
        super().train(mode)
        if self.pretrained and not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, ids, mask, tags, depths):
        if self.pretrained:
            safe_mask = mask.clone()
            safe_mask[:, 0] = True
            values = self.projection(self.backbone(input_ids=ids, attention_mask=safe_mask.long()).last_hidden_state)
        else:
            values = self.embedding(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        values = values + self.tag_embedding(tags) + self.depth_embedding(depths)
        return self.transformer(values, mask)


class ScreenshotPatchEncoder(nn.Module):
    def __init__(self, config, backbone_config=None):
        super().__init__()
        self.pretrained = config.encoder_backend == "pretrained"
        d = config.hidden_dim
        if self.pretrained:
            from transformers import AutoConfig, AutoModel
            if backbone_config is None:
                model = AutoModel.from_pretrained(config.visual_model_name, local_files_only=config.local_files_only)
            else:
                fields = dict(backbone_config)
                model = AutoModel.from_config(AutoConfig.for_model(fields.pop("model_type"), **fields))
            self.backbone = getattr(model, "vision_model", model)
            if getattr(self.backbone.config, "model_type", "") != "siglip_vision_model":
                raise ValueError("Use fixed-resolution SigLIP 2 (siglip_vision_model), not NaFlex")
            if self.backbone.config.patch_size != config.patch_size or self.backbone.config.image_size != config.image_size:
                raise ValueError("Configured patch_size/image_size must match visual checkpoint")
            self.projection = nn.Linear(self.backbone.config.hidden_size, d)
            if config.freeze_backbones:
                self.backbone.requires_grad_(False)
        else:
            self.patch_embedding = nn.Conv2d(3, d, config.patch_size, stride=config.patch_size)
            self.position = nn.Parameter(torch.randn(1, (config.image_size // config.patch_size) ** 2, d) * 0.02)
            self.transformer = TokenTransformer(d, config.num_heads, config.encoder_layers, config.dropout)

    def train(self, mode=True):
        super().train(mode)
        if self.pretrained and not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, pixels, available):
        if self.pretrained:
            values = self.projection(self.backbone(pixel_values=pixels).last_hidden_state)
        else:
            values = self.patch_embedding(pixels).flatten(2).transpose(1, 2) + self.position
        mask = available[:, None].expand(-1, values.shape[1])
        if not self.pretrained:
            values = self.transformer(values, mask)
        return values * mask.unsqueeze(-1), mask
