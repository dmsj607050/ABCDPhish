"""M4: four typed relations plus joint Relation Transformer.

Different domains are observable facts, not proof of unauthorized submission.
"""
import torch
from torch import nn

from .common import AttentionPool, TokenTransformer

RELATION_NAMES = ("appearance_consistency", "intent_consistency", "brand_domain_identity", "visit_submission_target")


class PairRelationEncoder(nn.Module):
    def __init__(self, dim, extra=0):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dim * 4 + extra, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim), nn.LayerNorm(dim))

    def forward(self, left, right, facts=None):
        pieces = [left, right, (left-right).abs(), left*right]
        if facts is not None:
            pieces.append(facts)
        return self.network(torch.cat(pieces, dim=-1))


class CrossModalRelationalReasoner(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.appearance = PairRelationEncoder(d)
        self.intent = PairRelationEncoder(d)
        self.identity = PairRelationEncoder(d)
        self.target = PairRelationEncoder(d, extra=3)
        self.target_pool = AttentionPool(d)
        self.type_embedding = nn.Parameter(torch.randn(1, 4, d) * 0.02)
        self.summary_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.transformer = TokenTransformer(d, config.num_heads, config.relation_layers, config.dropout)
        self.evidence_head = nn.Linear(d, 1)

    def forward(self, roles, identity, targets, target_mask, target_facts, available, role_quality):
        h, s = roles["html"], roles["visual"]
        page_mask = available[:, 1:]
        page_app = (torch.stack([h[:, 0], s[:, 0]], 1) * page_mask[:, :, None]).sum(1) / page_mask.sum(1, keepdim=True).clamp_min(1)
        target_pairs = self.target(identity[:, None].expand_as(targets), targets, target_facts)
        target_relation, target_weights = self.target_pool(target_pairs, target_mask)
        relations = torch.stack([self.appearance(h[:, 0], s[:, 0]), self.intent(h[:, 1], s[:, 1]),
                                 self.identity(identity, page_app), target_relation], 1)
        both = available[:, 1] & available[:, 2]
        masks = torch.stack([both, both, available[:, 0] & page_mask.any(1),
                             available[:, 0] & available[:, 1] & target_mask.any(1)], 1)
        quality = torch.stack([role_quality[:, :, 0].amin(1), role_quality[:, :, 1].amin(1),
                               role_quality[:, :, 0].amax(1), target_mask.any(1).float()], 1) * masks
        tokens = torch.cat([self.summary_token.expand(identity.shape[0], -1, -1), relations + self.type_embedding], 1)
        token_mask = torch.cat([torch.ones_like(masks[:, :1]), masks], 1)
        output = self.transformer(tokens, token_mask)
        return {"summary": output[:, 0] * masks.any(1)[:, None], "tokens": output[:, 1:],
                "mask": masks, "quality": quality, "logits": self.evidence_head(output[:, 1:]).squeeze(-1),
                "target_weights": target_weights}
