"""
Optimized configuration for small machines
(CPU 2-cores with 8GB RAM)
"""

from typing import Dict, Any, Optional
from .base import Config
from .constants import *


class ConfigSmall(Config):
    """Configuration optimized for resource-constrained machines"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        if config_dict is None:
            from .loader import load_config_yaml
            config_dict = load_config_yaml()
        
        # Update only the values we want to override
        if 'model' not in config_dict:
            config_dict['model'] = {}
        
        config_dict['model'].update({
            'embedding_dim': 256,
            'num_layers': 3,
            'num_heads': 4,
            'intermediate_dim': 1024,
        })
        
        if 'training' not in config_dict:
            config_dict['training'] = {}
        
        config_dict['training'].update({
            'device': 'cpu',
            'batch_size': 4,
            'num_epochs': 5,
            'warmup_steps': 50,
            'num_workers': 0,
        })
        
        # Pass create_dirs=False to avoid creating directories during config loading
        super().__init__(config_dict, create_dirs=False)