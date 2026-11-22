"""M3: modality-specific role-guided multi-query cross-attention.

Role priors are soft biases, never hard partitions or auxiliary labels.
No orthogonality assumption: appearance and intent may share useful evidence.
"""
import math

import torch
from torch import nn

from .common import masked_softmax


class RoleGuidedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_heads
        self.queries = config.num_queries
        self.dim = config.hidden_dim
        self.bias_strength = config.role_bias_strength
        self.role_queries = nn.Parameter(torch.randn(2, self.queries, self.dim) * 0.02)
        self.query = nn.Linear(self.dim, self.dim)
        self.key = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        self.output = nn.Linear(self.dim, self.dim)
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, tokens, mask, priors):
        b, length, d = tokens.shape
        h, q = self.heads, 2 * self.queries
        queries = self.query(self.role_queries.reshape(q, d)).view(q, h, d // h).permute(1, 0, 2)
        keys = self.key(tokens).view(b, length, h, d // h).transpose(1, 2)
        values = self.value(tokens).view(b, length, h, d // h).transpose(1, 2)
        logits = torch.einsum("hqd,bhld->bhql", queries, keys) / math.sqrt(d // h)
        role_bias = priors.transpose(1, 2).repeat_interleave(self.queries, dim=1)
        logits = logits + self.bias_strength * role_bias[:, None]
        weights = masked_softmax(logits, mask[:, None, None, :])
        attended = torch.einsum("bhql,bhld->bhqd", weights, values).transpose(1, 2).reshape(b, 2, self.queries, d)
        representations = self.norm(self.output(attended) + self.role_queries).mean(2)
        representations = representations * mask.any(1)[:, None, None]
        # [B, role(A/I), query, token]; model attention is not causal attribution.
        return representations, weights.mean(1).reshape(b, 2, self.queries, length)


class AppearanceIntentDisentangler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.html_queries = RoleGuidedQueryAttention(config)
        self.visual_queries = RoleGuidedQueryAttention(config)

    def forward(self, html, html_mask, visual, visual_mask, html_roles, visual_roles):
        h, ha = self.html_queries(html, html_mask, html_roles)
        s, sa = self.visual_queries(visual, visual_mask, visual_roles)
        return {"html": h, "visual": s, "html_attention": ha, "visual_attention": sa}
