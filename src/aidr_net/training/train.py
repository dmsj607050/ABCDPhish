"""CLI: train with explicit train/validation manifests and save best checkpoint."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import AIDRConfig
from ..data.dataset import WebpageDataset, collate_webpages, to_device
from ..data.static_evidence import normalize_url
from ..model import AIDRNet
from .losses import compute_losses
from .metrics import binary_metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    labels, probabilities, total, count = [], [], 0., 0
    for batch in loader:
        batch = to_device(batch, device)
        output = model(batch)
        loss = compute_losses(output, batch, config)["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        size = len(batch["label"])
        total += loss.item() * size
        count += size
        labels.extend(batch["label"].cpu().tolist())
        probabilities.extend(output["probabilities"].cpu().tolist())
    return {"loss": total / count, **binary_metrics(labels, probabilities)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output", default="runs/aidr")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, and lr must be positive")
    seed_everything(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))
    config = AIDRConfig.load(args.config)
    train = WebpageDataset(args.train, config, require_labels=True)
    val = WebpageDataset(args.val, config, require_labels=True)
    train_urls = {normalize_url(r["url"]) for r in train.records if r.get("url")}
    val_urls = {normalize_url(r["url"]) for r in val.records if r.get("url")}
    if train_urls & val_urls:
        raise ValueError("Train/validation contain identical URLs. Provide disjoint splits.")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_webpages)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_webpages)
    device = torch.device(args.device)
    model = AIDRNet(config).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    config.save(destination / "config.json")
    if config.encoder_backend == "pretrained":
        train.preprocess.tokenizer.save_pretrained(destination / "assets" / "html")
        train.preprocess.image_processor.save_pretrained(destination / "assets" / "visual")
    (destination / "run_arguments.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    best = float("inf")
    relation_label_counts = [0] * 4
    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()),
                      "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad), "device": str(device)}), flush=True)
    with (destination / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total, count = 0., 0
            components = {}
            for batch in train_loader:
                batch = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch)
                losses = compute_losses(output, batch, config)
                if output["relations"] is not None and config.lambda_relation > 0:
                    counts = ((batch["relation_labels"] != -100) & output["relations"]["mask"]).sum(0).cpu().tolist()
                    relation_label_counts = [a+b for a, b in zip(relation_label_counts, counts)]
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError("Non-finite training loss")
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                size = len(batch["label"])
                total += losses["total"].item() * size
                count += size
                for name, value in losses.items():
                    components[name] = components.get(name, 0.) + value.item() * size
            metrics = {"epoch": epoch, "train_loss": total/count,
                       "train_components": {k: v/count for k, v in components.items()},
                       "validation": evaluate(model, val_loader, device, config)}
            line = json.dumps(metrics, ensure_ascii=False, allow_nan=False)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            checkpoint = {"model": model.state_dict(), "config": config.to_dict(), "epoch": epoch,
                          "metrics": metrics, "seed": args.seed, "relation_label_counts": relation_label_counts,
                          "backbone_configs": model.backbone_configs()}
            torch.save(checkpoint, destination / "last.pt")
            if metrics["validation"]["loss"] < best:
                best = metrics["validation"]["loss"]
                torch.save(checkpoint, destination / "best.pt")


if __name__ == "__main__":
    main()
