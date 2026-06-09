"""
Transformer-based language model
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import MultiHeadAttention
from .feed_forward import FeedForwardNetwork


class TransformerBlock(nn.Module):
    """Single Transformer block"""
    
    def __init__(self, embedding_dim: int, num_heads: int, intermediate_dim: int, dropout_rate: float = 0.1):
        """
        Args:
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads
            intermediate_dim: Feed-forward intermediate dimension
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(embedding_dim, intermediate_dim, dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch, seq_len, embedding_dim)
            mask: Causal mask
        
        Returns:
            Transformed tensor
        """
        # Attention with residual connection
        x = x + self.attention(self.norm1(x), mask)
        
        # Feed-forward with residual connection
        x = x + self.feed_forward(self.norm2(x))
        
        return x


class TransformerModel(nn.Module):
    """Transformer-based language model (decoder-only, like GPT)"""
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        
        self.config = config
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_embedding = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.embedding_dim,
                config.num_heads,
                config.intermediate_dim,
                config.dropout_rate
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        # Share weights between embedding and output layer
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = self._count_parameters()
        print(f"✅ Model created with {total_params:,} parameters")
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask to prevent attention to future tokens
        
        Args:
            seq_len: Sequence length
            device: Device (cpu or cuda)
        
        Returns:
            Causal mask (lower triangular matrix)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            input_ids: Input token indices (batch, seq_len)
            targets: Target token indices for computing loss (optional)
        
        Returns:
            logits: Predicted logits (batch, seq_len, vocab_size)
            loss: Language modeling loss (if targets provided)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        
        # Token + positional embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.positional_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        mask = self._create_causal_mask(seq_len, input_ids.device)
        
        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm and output projection
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively
        
        Args:
            prompt_ids: Initial prompt token indices (batch, prompt_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering (None to disable)
            top_p: Top-p (nucleus) filtering (None to disable)
        
        Returns:
            Generated token indices (batch, prompt_len + generated_tokens)
        """
        self.eval()
        
        generated = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            # Limit context to max sequence length
            context = generated[:, -self.config.max_sequence_length:]
            
            # Get logits
            logits, _ = self.forward(context)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated