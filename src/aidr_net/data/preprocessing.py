"""M1 -> M2 tensor interface, with node/patch role priors and evidence provenance."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .static_evidence import INTENT_PATTERN, extract_evidence


def byte_ids(text, length):
    values = list(text.encode("utf-8"))[:length]
    # 0=padding, 1..256=UTF-8 bytes. No vocabulary learned from validation/test.
    return torch.tensor([x + 1 for x in values] + [0] * (length - len(values)), dtype=torch.long)


class EvidencePreprocessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = self.image_processor = None
        if config.encoder_backend == "pretrained":
            from transformers import AutoImageProcessor, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.html_model_name, use_fast=True, local_files_only=config.local_files_only)
            if not self.tokenizer.is_fast:
                raise ValueError("A fast tokenizer is required for DOM/token alignment")
            self.image_processor = AutoImageProcessor.from_pretrained(config.visual_model_name, local_files_only=config.local_files_only)

    def __call__(self, record, root=Path(".")):
        c = self.config
        root = Path(root)
        html = record.get("html", "")
        if record.get("html_path"):
            html = (root / record["html_path"]).read_text(encoding="utf-8", errors="replace")
        evidence = extract_evidence(record.get("url", ""), html, c.max_dom_nodes, c.max_targets)
        image = None
        if record.get("screenshot_path"):
            with Image.open(root / record["screenshot_path"]) as source:
                image = source.convert("RGB")
        if not evidence["host"] and not evidence["html_available"] and image is None:
            raise ValueError("At least one usable modality is required")
        nodes = evidence["nodes"]
        if self.tokenizer:
            text = "\n".join(n["text"] for n in nodes)
            encoded = self.tokenizer(text, max_length=c.html_max_tokens, padding="max_length", truncation=True, return_offsets_mapping=True)
            spans, pos = [], 0
            for node in nodes:
                spans.append((pos, pos + len(node["text"]), node))
                pos += len(node["text"]) + 1
            html_ids = torch.tensor(encoded["input_ids"])
            html_mask = torch.tensor(encoded["attention_mask"], dtype=torch.bool)
            tags, depths, roles = [], [], []
            cursor = 0
            for start, end in encoded["offset_mapping"]:
                while cursor < len(spans) and spans[cursor][1] <= start:
                    cursor += 1
                node = spans[cursor][2] if end > start and cursor < len(spans) and spans[cursor][0] < end else None
                tags.append(node["tag"] if node else 0)
                depths.append(node["depth"] if node else 0)
                roles.append(node["role"] if node else [0., 0.])
        else:
            ids, tags, depths, roles = [], [], [], []
            for node in nodes:
                values = list(node["text"].encode("utf-8")) + [10]
                ids.extend(x + 1 for x in values)
                tags.extend([node["tag"]] * len(values))
                depths.extend([node["depth"]] * len(values))
                roles.extend([node["role"]] * len(values))
            length = c.html_max_tokens
            html_ids = torch.tensor((ids[:length] + [0] * length)[:length], dtype=torch.long)
            html_mask = html_ids != 0
            tags = (tags[:length] + [0] * length)[:length]
            depths = (depths[:length] + [0] * length)[:length]
            roles = (roles[:length] + [[0., 0.]] * length)[:length]
        # Empty HTML must not become an available modality via tokenizer special tokens.
        html_mask &= bool(nodes)
        if image is None:
            image_input = Image.new("RGB", (c.image_size, c.image_size))
        else:
            image_input = image
        if self.image_processor:
            pixels = self.image_processor(images=image_input, return_tensors="pt")["pixel_values"][0]
            if getattr(self.image_processor, "do_center_crop", False):
                raise ValueError("Visual role coordinates require resize without center cropping")
            if pixels.ndim != 3 or tuple(pixels.shape[-2:]) != (c.image_size, c.image_size):
                raise ValueError("Use fixed-resolution SigLIP 2 and matching image_size; NaFlex is not supported")
        else:
            resized = image_input.resize((c.image_size, c.image_size), Image.Resampling.BICUBIC)
            pixels = torch.from_numpy(np.array(resized, dtype=np.float32).copy()).permute(2, 0, 1) / 127.5 - 1
        grid = c.image_size // c.patch_size
        visual_roles = torch.zeros(grid * grid, 2)
        # Optional OCR/region annotations use normalized [x0,y0,x1,y1] in ORIGINAL image.
        # Fixed-resolution processing resizes without cropping (checked in pretrained adapter).
        for region in record.get("visual_regions", []):
            box = region["box"]
            if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                raise ValueError("visual_regions.box must be normalized xyxy")
            confidence = float(region.get("confidence", 1.0))
            if not 0 <= confidence <= 1:
                raise ValueError("region confidence must lie in [0,1]")
            role = region.get("role")
            if role not in {None, "appearance", "intent", "both"}:
                raise ValueError("Unknown visual region role")
            prior = [float(role in {"appearance", "both"}), float(role in {"intent", "both"})]
            if role is None and INTENT_PATTERN.search(region.get("text", "")):
                prior[1] = 0.7  # weak lexical prior, never an auxiliary ground-truth label
            for y in range(grid):
                for x in range(grid):
                    overlap = max(0., min((x+1)/grid, box[2]) - max(x/grid, box[0])) * max(0., min((y+1)/grid, box[3]) - max(y/grid, box[1]))
                    weight = min(1., overlap * grid * grid) * confidence
                    visual_roles[y*grid+x] = torch.maximum(visual_roles[y*grid+x], torch.tensor(prior) * weight)
        targets = evidence["targets"]
        target_ids = torch.zeros(c.max_targets, c.url_max_length, dtype=torch.long)
        target_mask = torch.zeros(c.max_targets, dtype=torch.bool)
        target_facts = torch.zeros(c.max_targets, 3)
        for i, target in enumerate(targets):
            from .static_evidence import host_and_domain
            target_host, target_domain = host_and_domain(target["url"])
            target_ids[i] = byte_ids(f"host={target_host} domain={target_domain}", c.url_max_length)
            target_mask[i] = target["valid"]
            target_facts[i] = torch.tensor(target["facts"])
        appearance = record.get("appearance_labels", [-100, -100])
        intent = record.get("intent_labels", [[-100.] * len(c.intent_names) for _ in range(2)])
        align = record.get("alignment_labels", [-100., -100.])
        relation = record.get("relation_labels", [-100.] * 4)
        if len(appearance) != 2 or any(v != -100 and (not isinstance(v, int) or not 0 <= v < c.num_brands) for v in appearance):
            raise ValueError("appearance_labels must contain HTML/screenshot brand IDs, or -100")
        if len(intent) != 2:
            raise ValueError("intent_labels must contain separate HTML and screenshot label lists")
        for name, values, length in (("intent_labels[HTML]", intent[0], len(c.intent_names)), ("intent_labels[Screenshot]", intent[1], len(c.intent_names)), ("alignment_labels", align, 2), ("relation_labels", relation, 4)):
            if len(values) != length or any(v not in (-100, 0, 1) for v in values):
                raise ValueError(f"{name} must have {length} entries from -100, 0, 1")
        label = record.get("label", -100)
        if label not in (-100, 0, 1):
            raise ValueError("label must be 0=benign, 1=phishing, or -100=unknown")
        available = torch.tensor([bool(evidence["host"]), bool(html_mask.any()), image is not None])
        html_roles = torch.tensor(roles, dtype=torch.float32)
        quality = torch.stack([html_roles.amax(0), visual_roles.amax(0)])
        quality *= available[1:, None]
        return {"url_ids": byte_ids(evidence["url"], c.url_max_length),
                "url_identity_ids": byte_ids(f"host={evidence['host']} domain={evidence['domain']}", c.url_max_length), "html_ids": html_ids,
                "html_mask": html_mask, "html_tags": torch.tensor(tags), "html_depths": torch.tensor(depths),
                "html_roles": html_roles, "pixel_values": pixels, "visual_roles": visual_roles,
                "target_ids": target_ids, "target_mask": target_mask, "target_facts": target_facts,
                "available": available, "role_quality": quality, "label": torch.tensor(label, dtype=torch.float32),
                "appearance_labels": torch.tensor(appearance, dtype=torch.long), "intent_labels": torch.tensor(intent, dtype=torch.float32),
                "alignment_labels": torch.tensor(align, dtype=torch.float32), "relation_labels": torch.tensor(relation, dtype=torch.float32),
                "metadata": {"id": record.get("id"), "evidence": evidence, "visual_regions": record.get("visual_regions", [])}}
