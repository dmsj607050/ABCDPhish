"""M5: evidence-aware gating over original, disentangled, and relation branches."""
import torch
from torch import nn

from .common import masked_softmax


class EvidenceAwareFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.projections = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d)) for _ in range(3)])
        # All branch features + 3 availability flags + 3 quality scores.
        self.gate = nn.Sequential(nn.Linear(d * 3 + 6, d), nn.GELU(), nn.Linear(d, 3))

    def forward(self, original, disentangled, relation, availability, quality):
        features = torch.stack([layer(x) for layer, x in zip(self.projections, (original, disentangled, relation))], 1)
        features = features * availability[:, :, None]
        inputs = torch.cat([features.flatten(1), availability.float(), quality], dim=-1)
        weights = masked_softmax(self.gate(inputs), availability)
        return (features * weights[:, :, None]).sum(1), weights
