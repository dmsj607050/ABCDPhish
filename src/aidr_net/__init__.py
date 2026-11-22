"""AIDR-Net: six explicit stages for static multimodal phishing detection."""
from .config import AIDRConfig
from .model import AIDRNet

__all__ = ["AIDRConfig", "AIDRNet"]
