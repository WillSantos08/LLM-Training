# ============================================================================
# MODEL DEFAULTS
# ============================================================================
DEFAULT_TOKENIZER = "gpt2"
DEFAULT_VOCAB_SIZE = 50257
DEFAULT_EMBEDDING_DIM = 256
DEFAULT_NUM_LAYERS = 3
DEFAULT_NUM_HEADS = 4
DEFAULT_MAX_SEQUENCE_LENGTH = 256
DEFAULT_INTERMEDIATE_DIM = 1024
DEFAULT_DROPOUT_RATE = 0.1

# ============================================================================
# TRAINING DEFAULTS
# ============================================================================
DEFAULT_DEVICE = "cpu"
DEFAULT_SEED = 42
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_EPOCHS = 5
DEFAULT_GRADIENT_CLIP = 1.0
DEFAULT_WARMUP_STEPS = 50
DEFAULT_NUM_WORKERS = 0

DEFAULT_OPTIMIZER = "adamw"
DEFAULT_BETAS = [0.9, 0.95]
DEFAULT_WEIGHT_DECAY = 0.1

# ============================================================================
# GENERATION DEFAULTS
# ============================================================================
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 100
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9

# ============================================================================
# PATHS
# ============================================================================
DEFAULT_DATA_RAW = "data/raw"
DEFAULT_DATA_PROCESSED = "data/processed"
DEFAULT_DATA_CACHE = "data/cache"
DEFAULT_CHECKPOINTS = "models/checkpoints"
DEFAULT_TRAINED_MODELS = "models/trained"

# ============================================================================
# DATA
# ============================================================================
DEFAULT_TRAINING_FILE = "cephalon_luna/data/processed/train_data.json"
DEFAULT_VALIDATION_FILE = "cephalon_luna/data/processed/val_data.json"
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_SHUFFLE = True
DEFAULT_SUPPORTED_FORMATS = ['pdf', 'docx', 'txt', 'json', 'csv', 'md']

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_ENCODING = "utf-8"

# ============================================================================
# LOGGING
# ============================================================================
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SAVE_LOG = True