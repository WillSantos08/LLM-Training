from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size:  int   = 2000
    context_len: int   = 256
    d_model:     int   = 256
    num_heads:   int   = 8
    num_layers:  int   = 6
    d_ff:        int   = 1024
    dropout:     float = 0.1
    pad_id:      int   = 0