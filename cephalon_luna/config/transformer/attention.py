"""
Attention mechanisms for Transformer
"""

import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    """Single attention head"""
    
    def __init__(self, embedding_dim: int, num_heads: int):
        """
        Args:
            embedding_dim: Total embedding dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.head_dim = embedding_dim // num_heads
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(embedding_dim, self.head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, self.head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, self.head_dim, bias=False)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch, seq_len, embedding_dim)
            mask: Causal mask for autoregressive generation
        
        Returns:
            Attention output
        """
        # Compute Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        
        return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, embedding_dim: int, num_heads: int, dropout_rate: float = 0.1):
        """
        Args:
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Create multiple attention heads
        self.heads = nn.ModuleList([
            AttentionHead(embedding_dim, num_heads)
            for _ in range(num_heads)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch, seq_len, embedding_dim)
            mask: Causal mask
        
        Returns:
            Multi-head attention output
        """
        # Run all heads in parallel
        head_outputs = [head(x, mask) for head in self.heads]
        
        # Concatenate heads
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # Final projection
        output = self.output_projection(concatenated)
        output = self.dropout(output)
        
        return output