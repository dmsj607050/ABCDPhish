"""M6: missing labels are masked; no pseudo authorization/behavior labels."""
import torch
import torch.nn.functional as F


def masked_bce(logits, labels, mask):
    valid = (labels != -100) & mask
    if not valid.any():
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(logits[valid], labels[valid])


def compute_losses(output, batch, config):
    losses = {"classification": masked_bce(output["logits"], batch["label"], torch.ones_like(batch["label"], dtype=torch.bool))}
    zero = output["logits"].sum() * 0
    losses.update({"appearance": zero, "intent": zero, "alignment": zero, "relation": zero})
    modalities = batch["available"][:, 1:]
    if "appearance_logits" in output:
        labels = batch["appearance_labels"]
        valid = (labels != -100) & modalities
        if valid.any():
            losses["appearance"] = F.cross_entropy(output["appearance_logits"][valid], labels[valid])
    if "intent_logits" in output:
        labels = batch["intent_labels"]
        losses["intent"] = masked_bce(output["intent_logits"], labels, modalities[:, :, None].expand_as(labels))
    if "alignment_logits" in output:
        valid = modalities.all(1)[:, None].expand(-1, 2)
        losses["alignment"] = masked_bce(output["alignment_logits"], batch["alignment_labels"], valid)
    if output["relations"] is not None:
        losses["relation"] = masked_bce(output["relations"]["logits"], batch["relation_labels"], output["relations"]["mask"])
    losses["total"] = (losses["classification"] + config.lambda_app * losses["appearance"]
                       + config.lambda_intent * losses["intent"] + config.lambda_align * losses["alignment"]
                       + config.lambda_relation * losses["relation"])
    return losses
