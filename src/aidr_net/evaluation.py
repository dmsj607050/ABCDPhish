"""Evaluate a saved checkpoint on a separate labeled manifest."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.dataset import WebpageDataset, collate_webpages
from .inference import load_model
from .training.train import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, config = load_model(args.checkpoint, args.device)
    dataset = WebpageDataset(args.manifest, config, require_labels=True)
    metrics = evaluate(model, DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_webpages), args.device, config)
    text = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
