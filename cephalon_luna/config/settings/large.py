"""
Optimized configuration for powerful machines
(GPU CUDA 12GB+)
"""

from typing import Dict, Any, Optional
from .base import Config
from .constants import *


class ConfigLarge(Config):
    """Configuration optimized for powerful machines with GPU"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        if config_dict is None:
            from .loader import load_config_yaml
            config_dict = load_config_yaml()
        
        # Update only the values we want to override
        if 'model' not in config_dict:
            config_dict['model'] = {}
        
        config_dict['model'].update({
            'embedding_dim': 768,
            'num_layers': 12,
            'num_heads': 12,
            'max_sequence_length': 512,
            'intermediate_dim': 3072,
        })
        
        if 'training' not in config_dict:
            config_dict['training'] = {}
        
        config_dict['training'].update({
            'device': 'cuda',
            'learning_rate': 1e-4,
            'batch_size': 32,
            'num_epochs': 20,
            'warmup_steps': 500,
            'num_workers': 4,
            'weight_decay': 0.01,
        })
        
        # Pass create_dirs=False to avoid creating directories during config loading
        super().__init__(config_dict, create_dirs=False)