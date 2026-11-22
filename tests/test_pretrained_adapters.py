"""Exercise real HF adapters using tiny random LOCAL checkpoints, no downloads."""
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from aidr_net import AIDRConfig, AIDRNet
from aidr_net.data.dataset import collate_webpages
from aidr_net.data.preprocessing import EvidencePreprocessor
from aidr_net.training.losses import compute_losses


class PretrainedAdapterTests(unittest.TestCase):
    def test_local_modernbert_and_siglip_token_and_patch_adapters(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast,
                                  SiglipConfig, SiglipImageProcessor, SiglipModel)
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            html_path, visual_path = root / "html", root / "visual"
            tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "Bank": 2, "Login": 3}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]").save_pretrained(html_path)
            ModernBertModel(ModernBertConfig(vocab_size=4, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                            num_attention_heads=4, max_position_embeddings=64, local_attention=16,
                                            pad_token_id=0, cls_token_id=1, sep_token_id=1, bos_token_id=1, eos_token_id=1,
                                            _attn_implementation="eager")).save_pretrained(html_path)
            config = SiglipConfig(text_config={"vocab_size": 8, "hidden_size": 32, "intermediate_size": 64,
                                               "num_hidden_layers": 1, "num_attention_heads": 4, "max_position_embeddings": 16,
                                               "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0},
                                  vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                                                 "num_attention_heads": 4, "image_size": 32, "patch_size": 8})
            SiglipModel(config).save_pretrained(visual_path)
            SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(visual_path)
            Image.new("RGB", (64, 32), "white").save(root / "page.png")
            aidr_config = AIDRConfig(encoder_backend="pretrained", html_model_name=str(html_path), visual_model_name=str(visual_path),
                                     local_files_only=True, hidden_dim=32, num_heads=4, encoder_layers=1, relation_layers=1,
                                     image_size=32, patch_size=8, html_max_tokens=32, url_max_length=32, max_targets=1)
            preprocessor = EvidencePreprocessor(aidr_config)
            records = [{"url": "https://example.test", "html": '<h1>Bank</h1><form><input type="password"><button>Login</button></form>',
                        "screenshot_path": "page.png", "label": 1}, {"url": "https://missing.test", "label": 0}]
            batch = collate_webpages([preprocessor(r, root) for r in records])
            model = AIDRNet(aidr_config).train()
            output = model(batch)
            self.assertEqual(output["roles"]["visual_attention"].shape[-1], 16)
            self.assertEqual(output["roles"]["html_attention"].shape[-1], 32)
            compute_losses(output, batch, aidr_config)["total"].backward()
            self.assertFalse(model.html_encoder.backbone.training)
            self.assertFalse(model.visual_encoder.backbone.training)
            self.assertTrue(all(p.grad is None for p in model.html_encoder.backbone.parameters()))
            self.assertGreater(model.html_encoder.projection.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.visual_encoder.projection.weight.grad.abs().sum().item(), 0)
            from aidr_net.inference import load_model
            preprocessor.tokenizer.save_pretrained(root / "assets" / "html")
            preprocessor.image_processor.save_pretrained(root / "assets" / "visual")
            checkpoint = root / "pretrained.pt"
            torch.save({"model": model.state_dict(), "config": aidr_config.to_dict(), "backbone_configs": model.backbone_configs()}, checkpoint)
            model.eval()
            with torch.no_grad():
                before = model(batch)["probabilities"]
            loaded, loaded_config = load_model(checkpoint)
            self.assertTrue(loaded_config.local_files_only)
            # Checkpoint construction reads no model weights from original HF directories.
            self.assertEqual(loaded_config.html_model_name, str(root / "assets" / "html"))
            restored_preprocessor = EvidencePreprocessor(loaded_config)
            restored_batch = collate_webpages([restored_preprocessor(r, root) for r in records])
            with torch.no_grad():
                after = loaded(restored_batch)["probabilities"]
            torch.testing.assert_close(before, after)
