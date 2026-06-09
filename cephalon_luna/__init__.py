"""
Cephalon Luna - Personal AI Assistant
A lightweight Transformer-based language model for CPU training and inference.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Personal AI Assistant based on Transformer"

from .config import ConfigSmall, ConfigLarge, Config
from .config.transformer import TransformerModel
from .config.data import Trainer, create_dataloader, DocumentProcessor

__all__ = [
    'Config',
    'ConfigSmall',
    'ConfigLarge',
    'TransformerModel',
    'Trainer',
    'create_dataloader',
    'DocumentProcessor'
]