"""Behavioral checks: missing evidence, supervision masks, gradient flow, reload."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image

from aidr_net import AIDRConfig, AIDRNet
from aidr_net.data.dataset import collate_webpages
from aidr_net.data.preprocessing import EvidencePreprocessor
from aidr_net.data.static_evidence import extract_evidence, host_and_domain
from aidr_net.inference import load_model
from aidr_net.training.losses import compute_losses
from aidr_net.training.metrics import binary_metrics


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.config = AIDRConfig(hidden_dim=32, num_heads=4, encoder_layers=1, relation_layers=1,
                                 url_max_length=64, html_max_tokens=128, image_size=32, patch_size=8,
                                 max_targets=2, dropout=0., num_brands=3)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.new("RGB", (64, 32), "white").save(self.root / "page.png")
        self.record = {"url": "https://bank.example.com/login", "html": '<h1>Bank</h1><form action="https://submit.example.net/post"><input type="password"><button>Login</button></form>',
                       "screenshot_path": "page.png", "label": 1,
                       "visual_regions": [{"box": [0, 0, .5, .5], "role": "appearance"}, {"box": [.5, .5, 1, 1], "role": "intent"}]}

    def tearDown(self):
        self.temp.cleanup()

    def batch(self, records, config=None):
        preprocessor = EvidencePreprocessor(config or self.config)
        return collate_webpages([preprocessor(record, self.root) for record in records])

    def test_end_to_end_gradients_and_checkpoint(self):
        record = {**self.record, "appearance_labels": [1, 2], "intent_labels": [[1, 1, 0, 0], [1, -100, 0, 0]], "alignment_labels": [0, 1]}
        batch = self.batch([record, {**record, "label": 0}])
        model = AIDRNet(self.config)
        result = model(batch)
        losses = compute_losses(result, batch, self.config)
        losses["total"].backward()
        for name in ("url_encoder.embedding.weight", "html_encoder.embedding.weight", "visual_encoder.patch_embedding.weight",
                     "disentangler.html_queries.role_queries", "disentangler.visual_queries.role_queries",
                     "reasoner.appearance.network.0.weight", "reasoner.intent.network.0.weight", "reasoner.identity.network.0.weight",
                     "reasoner.target.network.0.weight", "fusion.gate.0.weight", "heads.classifier.0.weight"):
            parameter = dict(model.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertEqual(tuple(result["roles"]["html"].shape), (2, 2, 32))
        self.assertEqual(tuple(result["roles"]["visual_attention"].shape), (2, 2, 2, 16))
        self.assertTrue((result["relations"]["mask"]).all())
        model.eval()
        with torch.no_grad():
            before = model(batch)["probabilities"]
        checkpoint = self.root / "test.pt"
        torch.save({"model": model.state_dict(), "config": self.config.to_dict()}, checkpoint)
        loaded, _ = load_model(checkpoint)
        with torch.no_grad():
            after = loaded(batch)["probabilities"]
        torch.testing.assert_close(before, after)

    def test_all_seven_nonempty_modality_combinations(self):
        records = []
        for bits in range(1, 8):
            r = {"label": bits % 2}
            if bits & 1:
                r["url"] = self.record["url"]
            if bits & 2:
                r["html"] = self.record["html"]
            if bits & 4:
                r["screenshot_path"] = "page.png"
            records.append(r)
        batch = self.batch(records)
        model = AIDRNet(self.config)
        output = model(batch)
        loss = compute_losses(output, batch, self.config)
        self.assertTrue(torch.isfinite(loss["total"]))
        self.assertEqual(loss["appearance"].item(), 0.)
        self.assertEqual(loss["intent"].item(), 0.)
        self.assertEqual(loss["alignment"].item(), 0.)
        self.assertTrue(torch.isfinite(output["probabilities"]).all())
        self.assertTrue(torch.equal(output["fusion_weights"][~output["branch_available"]], torch.zeros_like(output["fusion_weights"][~output["branch_available"]])))
        torch.testing.assert_close(output["fusion_weights"].sum(1), torch.ones(7))
        loss["total"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_missing_modality_values_do_not_change_prediction(self):
        batch = self.batch([{"url": self.record["url"], "label": 1}])
        model = AIDRNet(self.config).eval()
        with torch.no_grad():
            before = model(batch)["probabilities"]
            batch["pixel_values"].normal_(0, 100)
            batch["html_ids"].fill_(10)
            after = model(batch)["probabilities"]
        torch.testing.assert_close(before, after)

    def test_auxiliary_labels_on_missing_modality_are_ignored(self):
        record = {"url": self.record["url"], "appearance_labels": [1, 1], "intent_labels": [[1]*4]*2, "alignment_labels": [1, 1]}
        batch = self.batch([record])
        losses = compute_losses(AIDRNet(self.config)(batch), batch, self.config)
        self.assertEqual(losses["total"].item(), 0.)

    def test_ablation_stages(self):
        for stage in ("baseline", "disentangled", "relations", "full"):
            config = replace(self.config, stage=stage)
            batch = self.batch([self.record], config)
            output = AIDRNet(config)(batch)
            compute_losses(output, batch, config)["total"].backward()
            self.assertEqual(output["roles"] is None, stage == "baseline")
            self.assertEqual(output["relations"] is None, stage in {"baseline", "disentangled"})

    def test_optional_relation_supervision_is_masked(self):
        config = replace(self.config, lambda_relation=0.3)
        record = {**self.record, "relation_labels": [1, 0, 1, 0]}
        model = AIDRNet(config)
        batch = self.batch([record], config)
        output = model(batch)
        losses = compute_losses(output, batch, config)
        self.assertGreater(losses["relation"].item(), 0)
        losses["total"].backward()
        self.assertGreater(model.reasoner.evidence_head.weight.grad.abs().sum().item(), 0)
        # URL alone cannot establish any of the four relations, even if labels exist.
        batch = self.batch([{"url": self.record["url"], "relation_labels": [1, 0, 1, 0]}], config)
        losses = compute_losses(model(batch), batch, config)
        self.assertEqual(losses["relation"].item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_backward(self):
        from aidr_net.data.dataset import to_device
        batch = to_device(self.batch([self.record]), "cuda")
        model = AIDRNet(self.config).cuda()
        loss = compute_losses(model(batch), batch, self.config)["total"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_target_resolution_and_domain_boundaries(self):
        evidence = extract_evidence("https://a.example.co.uk/path", '<base href="https://cdn.test/root/"><form action="post"><button formaction="/other">Pay</button></form><form></form>')
        self.assertEqual([x["url"] for x in evidence["targets"]], ["https://cdn.test/root/post", "https://a.example.co.uk/path", "https://cdn.test/other"])
        self.assertEqual(host_and_domain("https://x.example.co.uk")[1], "example.co.uk")
        self.assertNotEqual(host_and_domain("https://a.github.io")[1], host_and_domain("https://b.github.io")[1])
        self.assertEqual(host_and_domain("javascript:alert(1)"), ("", ""))

    def test_metric_ties(self):
        metrics = binary_metrics([0, 1, 0, 1], [.5] * 4)
        self.assertEqual(metrics["auroc"], .5)
        self.assertEqual(metrics["average_precision"], .5)
        self.assertIsNone(binary_metrics([1], [.9])["auroc"])

    def test_reject_no_evidence_and_invalid_annotations(self):
        preprocess = EvidencePreprocessor(self.config)
        with self.assertRaises(ValueError):
            preprocess({})
        with self.assertRaises(ValueError):
            preprocess({**self.record, "visual_regions": [{"box": [0, 0, 224, 224]}]}, self.root)


if __name__ == "__main__":
    unittest.main()
