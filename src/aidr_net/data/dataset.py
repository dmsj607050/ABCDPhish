"""JSONL records use paths relative to their manifest, not the working directory."""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .preprocessing import EvidencePreprocessor


class WebpageDataset(Dataset):
    def __init__(self, manifest, config, require_labels=False):
        self.path = Path(manifest).resolve()
        self.records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f"Empty manifest: {manifest}")
        if require_labels and any(r.get("label") not in (0, 1) for r in self.records):
            raise ValueError("Training/evaluation requires a binary label on every record")
        self.preprocess = EvidencePreprocessor(config)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        try:
            return self.preprocess(self.records[index], self.path.parent)
        except Exception as exc:
            raise ValueError(f"{self.path}, record {index + 1}: {exc}") from exc


def collate_webpages(samples):
    return {key: [s[key] for s in samples] if key == "metadata" else torch.stack([s[key] for s in samples]) for key in samples[0]}


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
