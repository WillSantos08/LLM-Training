# cephalon_luna/core/model.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config.model_config import ModelConfig


class CausalAttention(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0

        self.num_heads = cfg.num_heads
        self.d_head    = cfg.d_model // cfg.num_heads
        self.d_model   = cfg.d_model
        self.dropout_p = cfg.dropout

        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(self.d_model, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.dropout_p if self.training else 0.0,
            is_causal = True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(out))


class FeedForward(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ff   = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class LunaModel(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.num_layers)
        ])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.tok_emb.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        scale = 0.02 / math.sqrt(2 * self.cfg.num_layers)
        for name, p in self.named_parameters():
            if "proj.weight" in name:
                nn.init.normal_(p, std=scale)

    def forward(
        self,
        ids:     torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T   = ids.shape
        device = ids.device

        assert T <= self.cfg.context_len, (
            f"Sequência {T} > context_len {self.cfg.context_len}"
        )

        pos    = torch.arange(T, device=device).unsqueeze(0)
        x      = self.drop(self.tok_emb(ids) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x      = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=self.cfg.pad_id,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids:     torch.Tensor,
        max_new_tokens: int   = 100,
        temperature:    float = 0.8,
        top_k:          int   = 40,
        eos_id:         int   = 3,
    ) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            ctx       = prompt_ids[:, -self.cfg.context_len:]
            logits, _ = self(ctx)
            logits    = logits[:, -1, :] / temperature

            if top_k > 0:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, [-1]]] = float("-inf")

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            if next_token.item() == eos_id:
                break

            prompt_ids = torch.cat([prompt_ids, next_token], dim=1)

        return prompt_ids

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        """Salva modelo e config juntos."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Salvar config separado como dict puro
        cfg_dict = {
            "vocab_size":  self.cfg.vocab_size,
            "context_len": self.cfg.context_len,
            "d_model":     self.cfg.d_model,
            "num_heads":   self.cfg.num_heads,
            "num_layers":  self.cfg.num_layers,
            "d_ff":        self.cfg.d_ff,
            "dropout":     self.cfg.dropout,
            "pad_id":      self.cfg.pad_id,
        }

        torch.save(
            {
                "cfg":   cfg_dict,
                "state": self.state_dict(),
            },
            path,
        )
        print(f"  💾 Modelo salvo : {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LunaModel":
        """Carrega modelo compatível com PyTorch 2.6+."""

        # Adicionar ModelConfig como global seguro
        torch.serialization.add_safe_globals([ModelConfig])

        ckpt = torch.load(
            path,
            map_location = device,
            weights_only = True,   # ✅ seguro no PyTorch 2.6+
        )

        # Reconstruir config a partir do dict puro
        cfg_data = ckpt["cfg"]

        # Suporte a checkpoint antigo (cfg era objeto) e novo (cfg é dict)
        if isinstance(cfg_data, dict):
            cfg = ModelConfig(**cfg_data)
        else:
            cfg = cfg_data

        model = cls(cfg)
        model.load_state_dict(ckpt["state"])
        model = model.to(device)
        print(f"  ✅ Modelo carregado : {path}")
        return model