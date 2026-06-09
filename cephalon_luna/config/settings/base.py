"""
Base configuration class
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional

from .constants import *
from .loader import load_config_yaml


class Config:
    """Base configuration class for Cephalon Luna"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None, create_dirs: bool = True):
        """
        Initialize configuration from YAML or dictionary
        
        Args:
            config_dict: Configuration dictionary (loads from YAML if None)
            create_dirs: Whether to create directories (default: True)
        """
        if config_dict is None:
            config_dict = load_config_yaml()
        
        # =====================================================================
        # MODEL ARCHITECTURE
        # =====================================================================
        model_config = config_dict.get('model', {})
        
        self.tokenizer_name = model_config.get('tokenizer_name', DEFAULT_TOKENIZER)
        self.vocab_size = model_config.get('vocab_size', DEFAULT_VOCAB_SIZE)
        self.embedding_dim = model_config.get('embedding_dim', DEFAULT_EMBEDDING_DIM)
        self.num_layers = model_config.get('num_layers', DEFAULT_NUM_LAYERS)
        self.num_heads = model_config.get('num_heads', DEFAULT_NUM_HEADS)
        self.max_sequence_length = model_config.get('max_sequence_length', DEFAULT_MAX_SEQUENCE_LENGTH)
        self.intermediate_dim = model_config.get('intermediate_dim', DEFAULT_INTERMEDIATE_DIM)
        self.dropout_rate = model_config.get('dropout_rate', DEFAULT_DROPOUT_RATE)
        
        # =====================================================================
        # TRAINING HYPERPARAMETERS
        # =====================================================================
        training_config = config_dict.get('training', {})

        self.device = self._setup_device(training_config.get('device', DEFAULT_DEVICE))
        self.seed = training_config.get('seed', DEFAULT_SEED)
        self.learning_rate = float(training_config.get('learning_rate', DEFAULT_LEARNING_RATE))  # ← ADICIONE float()
        self.batch_size = int(training_config.get('batch_size', DEFAULT_BATCH_SIZE))  # ← ADICIONE int()
        self.num_epochs = int(training_config.get('num_epochs', DEFAULT_NUM_EPOCHS))  # ← ADICIONE int()
        self.gradient_clip = float(training_config.get('gradient_clip', DEFAULT_GRADIENT_CLIP))  # ← ADICIONE float()
        self.warmup_steps = int(training_config.get('warmup_steps', DEFAULT_WARMUP_STEPS))  # ← ADICIONE int()
        self.num_workers = int(training_config.get('num_workers', DEFAULT_NUM_WORKERS))  # ← ADICIONE int()

        # Optimizer
        self.optimizer_name = training_config.get('optimizer_name', DEFAULT_OPTIMIZER)
        self.betas = training_config.get('betas', DEFAULT_BETAS)
        self.weight_decay = float(training_config.get('weight_decay', DEFAULT_WEIGHT_DECAY))  # ← ADICIONE float()
        
        # =====================================================================
        # TEXT GENERATION
        # =====================================================================
        generation_config = config_dict.get('generation', {})
        
        self.temperature = generation_config.get('temperature', DEFAULT_TEMPERATURE)
        self.max_tokens = generation_config.get('max_tokens', DEFAULT_MAX_TOKENS)
        self.top_k = generation_config.get('top_k', DEFAULT_TOP_K)
        self.top_p = generation_config.get('top_p', DEFAULT_TOP_P)
        
        # =====================================================================
        # PATHS
        # =====================================================================
        paths_config = config_dict.get('paths', {})
        
        self.data_raw_path = paths_config.get('data_raw', DEFAULT_DATA_RAW)
        self.data_processed_path = paths_config.get('data_processed', DEFAULT_DATA_PROCESSED)
        self.data_cache_path = paths_config.get('data_cache', DEFAULT_DATA_CACHE)
        self.checkpoints_path = paths_config.get('models_checkpoints', DEFAULT_CHECKPOINTS)
        self.trained_models_path = paths_config.get('models_trained', DEFAULT_TRAINED_MODELS)
        
        # Create directories if they don't exist (only if create_dirs=True)
        if create_dirs:
            self._create_directories()
        
        # =====================================================================
        # DATA CONFIGURATION
        # =====================================================================
        data_config = config_dict.get('data', {})
        
        self.training_file = data_config.get('file_training', 'data/processed/train_data.json')
        self.validation_file = data_config.get('file_validation', 'data/processed/val_data.json')
        self.validation_split = data_config.get('validation_split', DEFAULT_VALIDATION_SPLIT)
        self.shuffle = data_config.get('shuffle', DEFAULT_SHUFFLE)
        self.supported_formats = data_config.get('supported_formats', DEFAULT_SUPPORTED_FORMATS)
        
        # =====================================================================
        # DOCUMENT PROCESSING
        # =====================================================================
        doc_config = config_dict.get('documents', {})
        
        self.chunk_size = doc_config.get('chunk_size', DEFAULT_CHUNK_SIZE)
        self.chunk_overlap = doc_config.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP)
        self.encoding = doc_config.get('encoding', DEFAULT_ENCODING)
    
    def _setup_device(self, device_config: str) -> str:
        """
        Setup device (CPU, CUDA, MPS)
        
        Args:
            device_config: Device name ('auto', 'cpu', 'cuda', 'mps')
        
        Returns:
            Confirmed device name
        """
        if device_config == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():  # Apple Silicon
                return 'mps'
            else:
                return 'cpu'
        
        return device_config
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.data_raw_path,
            self.data_processed_path,
            self.data_cache_path,
            self.checkpoints_path,
            self.trained_models_path,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def __str__(self) -> str:
        """String representation of configuration"""
        config_str = "\n" + "="*70 + "\n"
        config_str += "CEPHALON LUNA - MODEL CONFIGURATION\n"
        config_str += "="*70 + "\n"
        
        config_str += f"Device: {self.device}\n"
        config_str += f"Embedding Dimension: {self.embedding_dim}\n"
        config_str += f"Number of Layers: {self.num_layers}\n"
        config_str += f"Attention Heads: {self.num_heads}\n"
        config_str += f"Learning Rate: {self.learning_rate}\n"
        config_str += f"Batch Size: {self.batch_size}\n"
        config_str += f"Epochs: {self.num_epochs}\n"
        config_str += "="*70 + "\n"
        
        return config_str