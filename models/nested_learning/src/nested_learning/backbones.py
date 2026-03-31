from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fast_state import AttentionKVCache


@dataclass
class AttentionConfig:
    dim: int
    heads: int
    dropout: float = 0.0
    use_flash: bool = True
    causal: bool = True
    qk_l2_norm: bool = False
    qk_norm_eps: float = 1e-6
    local_conv_window: int | None = None


class SelfAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        if config.dim % config.heads != 0:
            msg = f"dim must be divisible by heads (got dim={config.dim}, heads={config.heads})"
            raise ValueError(msg)
        self.config = config
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.dim)
        self.local_conv: nn.Conv1d | None = None
        if config.local_conv_window is not None:
            window = int(config.local_conv_window)
            if window <= 0:
                raise ValueError("local_conv_window must be positive")
            self.local_conv = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=window,
                groups=config.dim,
                padding=0,
                bias=False,
            )

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        *,
        kv_cache: AttentionKVCache | None = None,
        return_kv_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionKVCache]:
        residual = x
        attn_inp = x
        if kv_cache is not None and self.local_conv is not None:
            raise RuntimeError(
                "kv_cache with local_conv_window is not supported in this implementation."
            )
        if self.local_conv is not None:
            kernel = self.local_conv.kernel_size[0]
            attn_inp = attn_inp.transpose(1, 2)
            # Causal depthwise conv: only attends to past tokens.
            attn_inp = F.pad(attn_inp, (kernel - 1, 0))
            attn_inp = self.local_conv(attn_inp).transpose(1, 2)
        q, k, v = self._compute_qkv(attn_inp)
        past_len = 0
        k_all = k
        v_all = v
        if kv_cache is not None:
            if kv_cache.key.size(0) != k.size(0):
                raise ValueError("kv_cache batch dimension must match input batch dimension")
            if kv_cache.key.size(1) != k.size(1) or kv_cache.key.size(-1) != k.size(-1):
                raise ValueError("kv_cache shape is incompatible with attention heads/head_dim")
            past_len = int(kv_cache.key.size(2))
            k_all = torch.cat([kv_cache.key, k], dim=2)
            v_all = torch.cat([kv_cache.value, v], dim=2)
        attn_output = self._scaled_dot_product_attn(q, k_all, v_all, past_len=past_len)
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        out = self.norm(residual + attn_output)
        if return_kv_cache:
            return out, AttentionKVCache(key=k_all.detach(), value=v_all.detach())
        return out

    def _compute_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (x.size(0), x.size(1), self.heads, self.head_dim)
        q = q.view(*shape).transpose(1, 2)
        k = k.view(*shape).transpose(1, 2)
        v = v.view(*shape).transpose(1, 2)
        if self.config.qk_l2_norm:
            q = F.normalize(q, dim=-1, eps=self.config.qk_norm_eps)
            k = F.normalize(k, dim=-1, eps=self.config.qk_norm_eps)
        return q, k, v

    def _scaled_dot_product_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        past_len: int = 0,
    ) -> torch.Tensor:
        dropout_p = self.config.dropout if self.training else 0.0
        attn_mask = None
        if self.config.causal and past_len > 0:
            query_len = int(q.size(-2))
            key_len = int(k.size(-2))
            key_positions = torch.arange(key_len, device=q.device)
            query_positions = past_len + torch.arange(query_len, device=q.device)
            attn_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        is_causal = self.config.causal and attn_mask is None
        device_type = q.device.type
        if (
            device_type == "cuda"
            and torch.cuda.is_available()
            and hasattr(torch.backends, "cuda")
            and hasattr(torch.backends.cuda, "sdp_kernel")
        ):
            with torch.backends.cuda.sdp_kernel(  # type: ignore[attr-defined]
                enable_flash=self.config.use_flash,
                enable_mem_efficient=True,
                enable_math=not self.config.use_flash,
            ):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
