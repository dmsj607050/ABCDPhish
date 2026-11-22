"""M6: phishing classification and optional role/alignment auxiliary heads."""
import torch
from torch import nn


class PredictionHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.classifier = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(d, 1))
        self.appearance = nn.Linear(d, config.num_brands) if config.num_brands > 0 else None
        self.intent = nn.Linear(d, len(config.intent_names))
        self.alignment = nn.ModuleList([nn.Sequential(nn.Linear(d * 4, d), nn.GELU(), nn.Linear(d, 1)) for _ in range(2)])

    def forward(self, fused, roles=None):
        logits = self.classifier(fused).squeeze(-1)
        result = {"logits": logits, "probabilities": logits.sigmoid()}
        if roles is not None:
            h, s = roles["html"], roles["visual"]
            appearances = torch.stack([h[:, 0], s[:, 0]], 1)
            intents = torch.stack([h[:, 1], s[:, 1]], 1)
            if self.appearance is not None:
                result["appearance_logits"] = self.appearance(appearances)
            result["intent_logits"] = self.intent(intents)
            result["alignment_logits"] = torch.cat([head(torch.cat([h[:, i], s[:, i], (h[:, i]-s[:, i]).abs(), h[:, i]*s[:, i]], -1))
                                                    for i, head in enumerate(self.alignment)], -1)
        return result
