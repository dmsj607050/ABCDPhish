"""Mask-safe pooling and attention shared by network modules."""
import torch
from torch import nn


def masked_mean(values, mask):
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1)


def masked_softmax(logits, mask, dim=-1):
    mask = mask.expand_as(logits)
    weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(dim)
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(dim, keepdim=True).clamp_min(1e-8)


class TokenTransformer(nn.Module):
    def __init__(self, dim, heads, layers, dropout):
        super().__init__()
        block = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)

    def forward(self, values, mask):
        # PyTorch attention requires at least one visible key per sample.
        safe_mask = mask.clone()
        safe_mask[:, 0] = True
        result = self.encoder(values, src_key_padding_mask=~safe_mask)
        return result * mask.unsqueeze(-1)


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, values, mask):
        weights = masked_softmax(self.score(values).squeeze(-1), mask)
        return (values * weights.unsqueeze(-1)).sum(1), weights
