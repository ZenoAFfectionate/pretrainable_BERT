import math

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """ Implementation of Rotary Positional Embedding """

    def __init__(self, dim, max_seq_len=1024, base=10000):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # avoid data repetition by generating complete cache
        self.register_buffer("cos_cached", torch.empty(1, 1, 0, dim))
        self.register_buffer("sin_cached", torch.empty(1, 1, 0, dim))
        self._build_cache(max_seq_len)  # initialize cache
    
    def _build_cache(self, seq_len, device="cpu"):
        """ build or extend position cache """
        if seq_len <= self.cos_cached.shape[2]:
            return  # no need to extend
        
        # recalculate cache size (double)
        new_max_len = max(seq_len, self.max_seq_len * 2)
        self.max_seq_len = new_max_len
        
        # generate frequency matrix
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        position = torch.arange(new_max_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", position, inv_freq)
        
        # generate whole positional cache
        cache = torch.cat([freqs, freqs], dim=-1)
        
        # update cache to make sure they are on the correct device 
        self.register_buffer("cos_cached", cache.cos()[None, None, :, :].to(device))
        self.register_buffer("sin_cached", cache.sin()[None, None, :, :].to(device))

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        seq_len = q.size(2)  # [batch_size, head_num, seq_len, head_dim]
        device = q.device
        
        # dynamic cache extension
        if seq_len > self.max_seq_len or self.cos_cached.device != device:
            self._build_cache(seq_len, device)

        cos = self.cos_cached[:, :, :seq_len, :]  # use cache view
        sin = self.sin_cached[:, :, :seq_len, :]  # use cache view

        rotated_q = (q * cos) + (self.rotate_half(q) * sin)
        rotated_k = (k * cos) + (self.rotate_half(k) * sin)
        
        return rotated_q, rotated_k


class MultiHeadAttention(nn.Module):
    """ Implementation of the multi-head attention for visual task """

    def __init__(self, emb_size, head_num=8, dropout=0.1):
        super().__init__()
        assert emb_size % head_num == 0, "embedding size is not divisible by head number"
        self.emb_size = emb_size
        self.head_num = head_num
        self.head_dim = emb_size // head_num
        # fuse the queries, keys and values into one matrix
        self.q_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.k_proj = nn.Linear(emb_size, emb_size, bias=False)
        self.v_proj = nn.Linear(emb_size, emb_size, bias=False)

        self.scale = 1 / math.sqrt(self.head_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        # initialize the rotary positional embedding
        self.rotary_embedding = RotaryEmbedding(self.head_dim)

        self._init_weights()

    def _init_weights(self):
        """ initialize weights of attention module """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)


    def forward(self, x, mask=None):
        q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=self.head_num)
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.head_num)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.head_num)

        q, k = self.rotary_embedding(q, k)  # apply rotary embedding

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, torch.finfo(attn_scores.dtype).min)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        context  = torch.matmul(attn_weights, v)
        context  = rearrange(context , 'b h n d -> b n (h d)')

        return self.projection(context)
