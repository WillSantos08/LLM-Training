"""
Feed-forward network for Transformer
"""

import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, embedding_dim: int, intermediate_dim: int, dropout_rate: float = 0.1):
        """
        Args:
            embedding_dim: Input/output dimension
            intermediate_dim: Hidden layer dimension
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.linear1 = nn.Linear(embedding_dim, intermediate_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(intermediate_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, seq_len, embedding_dim)
        
        Returns:
            Transformed tensor
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x