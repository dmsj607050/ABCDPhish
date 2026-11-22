"""Predict static records and export evidence, availability, and attention maps."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import AIDRConfig
from .data.dataset import WebpageDataset, collate_webpages, to_device
from .model import AIDRNet
from .modules.relational_reasoning import RELATION_NAMES


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = AIDRConfig(**checkpoint["config"])
    assets = Path(checkpoint_path).resolve().parent / "assets"
    if config.encoder_backend == "pretrained" and assets.is_dir():
        config.html_model_name = str(assets / "html")
        config.visual_model_name = str(assets / "visual")
        config.local_files_only = True
    model = AIDRNet(config, backbone_configs=checkpoint.get("backbone_configs"))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.relation_label_counts = checkpoint.get("relation_label_counts", [0] * 4)
    return model.to(device).eval(), config


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="runs/predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--attention", action="store_true", help="Include large role-attention arrays")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must lie in [0,1]")
    torch.set_num_threads(max(1, args.cpu_threads))
    model, config = load_model(args.checkpoint, args.device)
    dataset = WebpageDataset(args.manifest, config)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_webpages)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for batch in loader:
            output = model(to_device(batch, args.device))
            for i, metadata in enumerate(batch["metadata"]):
                probability = output["probabilities"][i].item()
                row = {**metadata, "phishing_probability": probability, "prediction": int(probability >= args.threshold),
                       "threshold": args.threshold, "available_modalities": batch["available"][i].tolist(),
                       "branch_available": output["branch_available"][i].tolist(),
                       "fusion_weights": dict(zip(("original", "disentangled", "relation"), output["fusion_weights"][i].cpu().tolist())),
                       "modality_weights": output["modality_weights"][i].cpu().tolist(),
                       "relation_label_counts": model.relation_label_counts,
                       "evidence_note": "Attention/gates are model diagnostics; domain differences do not establish lack of authorization."}
                relation = output["relations"]
                if relation is not None:
                    row["relations"] = {name: {"available": bool(relation["mask"][i, j]),
                                               "evidence_quality": relation["quality"][i, j].item(),
                                               "auxiliary_score": relation["logits"][i, j].sigmoid().item() if config.lambda_relation > 0 and model.relation_label_counts[j] > 0 and relation["mask"][i, j] else None}
                                        for j, name in enumerate(RELATION_NAMES)}
                    row["target_weights"] = relation["target_weights"][i].cpu().tolist()
                if args.attention and output["roles"] is not None:
                    row["role_attention"] = {name: output["roles"][name][i].cpu().tolist() for name in ("html_attention", "visual_attention")}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Wrote {len(dataset)} predictions to {destination}")


if __name__ == "__main__":
    main()
