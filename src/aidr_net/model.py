"""AIDR-Net orchestration: M2 -> M3 -> M4 -> M5 -> M6, with residual evidence."""
import torch
from torch import nn

from .modules.common import AttentionPool, masked_mean
from .modules.evidence_fusion import EvidenceAwareFusion
from .modules.prediction_heads import PredictionHeads
from .modules.relational_reasoning import CrossModalRelationalReasoner
from .modules.semantic_disentanglement import AppearanceIntentDisentangler
from .modules.trimodal_encoders import ScreenshotPatchEncoder, StructureAwareHTMLEncoder, URLCharacterEncoder


class AIDRNet(nn.Module):
    def __init__(self, config, backbone_configs=None):
        super().__init__()
        self.config = config
        self.url_encoder = URLCharacterEncoder(config)
        backbone_configs = backbone_configs or {}
        self.html_encoder = StructureAwareHTMLEncoder(config, backbone_configs.get("html"))
        self.visual_encoder = ScreenshotPatchEncoder(config, backbone_configs.get("visual"))
        self.original_pool = AttentionPool(config.hidden_dim)
        self.role_pool = AttentionPool(config.hidden_dim)
        self.disentangler = AppearanceIntentDisentangler(config) if config.stage != "baseline" else None
        self.reasoner = CrossModalRelationalReasoner(config) if config.stage in {"relations", "full"} else None
        self.fusion = EvidenceAwareFusion(config)
        self.heads = PredictionHeads(config)

    def backbone_configs(self):
        if self.config.encoder_backend != "pretrained":
            return {}
        return {"html": self.html_encoder.backbone.config.to_dict(),
                "visual": self.visual_encoder.backbone.config.to_dict()}

    def forward(self, batch):
        available = batch["available"]
        if not available.any(1).all():
            raise ValueError("Every sample needs at least one available modality")
        url_tokens, url_mask = self.url_encoder(batch["url_ids"])
        url = masked_mean(url_tokens, url_mask) * available[:, :1]
        html_tokens = self.html_encoder(batch["html_ids"], batch["html_mask"], batch["html_tags"], batch["html_depths"])
        visual_tokens, visual_mask = self.visual_encoder(batch["pixel_values"], available[:, 2])
        original, modality_weights = self.original_pool(torch.stack([url, masked_mean(html_tokens, batch["html_mask"]),
                                                                      masked_mean(visual_tokens, visual_mask)], 1), available)
        roles = relation = None
        role_summary = relation_summary = torch.zeros_like(original)
        role_available = available[:, 1:].any(1) if self.disentangler is not None else torch.zeros_like(available[:, 0])
        relation_available = torch.zeros_like(role_available)
        role_quality = batch["role_quality"].flatten(1).mean(1)
        relation_quality = torch.zeros_like(role_quality)
        if self.disentangler is not None:
            roles = self.disentangler(html_tokens, batch["html_mask"], visual_tokens, visual_mask,
                                      batch["html_roles"], batch["visual_roles"])
            role_mask = available[:, 1:].repeat_interleave(2, dim=1)
            role_summary, _ = self.role_pool(torch.cat([roles["html"], roles["visual"]], 1), role_mask)
        if self.reasoner is not None:
            b, count, length = batch["target_ids"].shape
            target_tokens, target_token_mask = self.url_encoder(batch["target_ids"].reshape(b*count, length))
            targets = masked_mean(target_tokens, target_token_mask).reshape(b, count, -1)
            identity_tokens, identity_mask = self.url_encoder(batch["url_identity_ids"])
            identity = self.url_encoder.identity_projection(masked_mean(identity_tokens, identity_mask))
            targets = self.url_encoder.identity_projection(targets)
            relation = self.reasoner(roles, identity, targets, batch["target_mask"], batch["target_facts"], available, batch["role_quality"])
            relation_summary = relation["summary"]
            relation_available = relation["mask"].any(1)
            relation_quality = relation["quality"].sum(1) / relation["mask"].sum(1).clamp_min(1)
        branch_mask = torch.stack([available.any(1), role_available, relation_available], 1)
        quality = torch.stack([available.float().mean(1), role_quality, relation_quality], 1)
        if self.config.stage == "full":
            fused, weights = self.fusion(original, role_summary, relation_summary, branch_mask, quality)
        else:
            # Controlled incremental baselines: equal branch weighting, no learned evidence gate.
            weights = branch_mask.float() / branch_mask.sum(1, keepdim=True)
            fused = (torch.stack([original, role_summary, relation_summary], 1) * weights[:, :, None]).sum(1)
        output = self.heads(fused, roles)
        output.update({"roles": roles, "relations": relation, "fusion_weights": weights,
                       "modality_weights": modality_weights, "branch_available": branch_mask})
        return output
