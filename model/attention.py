import math

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention for BERT.

    Delegates the softmax-weighted matmul to
    ``torch.nn.functional.scaled_dot_product_attention``, which on recent
    PyTorch + CUDA picks Flash Attention 2 when the inputs qualify. This
    avoids materializing the [B, H, S, S] attention-score tensor and is
    substantially faster + more memory-frugal than the hand-rolled version.
    """

    def __init__(self, emb_size, head_num=8, dropout=0.1):
        super().__init__()
        assert emb_size % head_num == 0, "embedding size is not divisible by head number"
        self.emb_size = emb_size
        self.head_num = head_num
        self.head_dim = emb_size // head_num
        self.dropout_p = dropout

        # Fused QKV projection: one linear (3*emb_size) beats three separate
        # linears by avoiding two extra kernel launches and weight loads.
        self.qkv_proj = nn.Linear(emb_size, 3 * emb_size, bias=True)
        self.projection = nn.Linear(emb_size, emb_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        H, D = self.head_num, self.head_dim

        qkv = self.qkv_proj(x)  # [B, S, 3*H*D]
        qkv = qkv.view(B, S, 3, H, D).permute(2, 0, 3, 1, 4)  # [3, B, H, S, D]
        q, k, v = qkv.unbind(0)

        # SDPA accepts an additive float mask or a bool key-padding mask.
        # The caller ships a [B, 1, 1, S] bool mask where True = keep.
        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )  # [B, H, S, D]

        context = context.transpose(1, 2).contiguous().view(B, S, H * D)
        return self.projection(context)


class DisentangledSelfAttention(nn.Module):
    """
    DeBERTa Disentangled Self-Attention.

    Attention score between positions i and j is the sum of three terms:
        A_{i,j} = Q_c_i · K_c_j         (content-to-content)
                + Q_c_i · K_r_{δ(i,j)}  (content-to-position)
                + K_c_j · Q_r_{δ(j,i)}  (position-to-content)
    where δ(i, j) is the relative distance clipped to [-k, k] and shifted to [0, 2k].
    The combined score is scaled by 1 / sqrt(3 * d_head).
    """

    def __init__(self, emb_size, head_num=8, dropout=0.1, max_relative_positions=512):
        super().__init__()
        assert emb_size % head_num == 0, "embedding size is not divisible by head number"
        self.emb_size = emb_size
        self.head_num = head_num
        self.head_dim = emb_size // head_num
        self.max_relative_positions = max_relative_positions

        self.q_proj = nn.Linear(emb_size, emb_size, bias=True)
        self.k_proj = nn.Linear(emb_size, emb_size, bias=True)
        self.v_proj = nn.Linear(emb_size, emb_size, bias=True)

        self.pos_q_proj = nn.Linear(emb_size, emb_size, bias=True)
        self.pos_k_proj = nn.Linear(emb_size, emb_size, bias=True)

        self.relative_pos_emb = nn.Parameter(
            torch.zeros(2 * max_relative_positions + 1, emb_size)
        )

        self.scale = 1.0 / math.sqrt(3 * self.head_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

        # Cache the [S, S] clipped relative-position index: shape depends only
        # on seq_len + device, not on batch content, so it's safe to memoize.
        self._rel_idx_cache: dict = {}

        self._init_weights()

    def _init_weights(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj,
                     self.pos_q_proj, self.pos_k_proj, self.projection):
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        nn.init.normal_(self.relative_pos_emb, mean=0.0, std=0.02)

    def _get_relative_positions(self, seq_len, device):
        """Return [seq_len, seq_len] tensor where entry (i, j) = clipped(j - i) + k.

        Memoized per (seq_len, device): the tensor depends only on those two
        and recomputing it per forward per layer is pure overhead (12 layers
        × forward + backward = 24 redundant launches per step).
        """
        key = (seq_len, device)
        cached = self._rel_idx_cache.get(key)
        if cached is not None:
            return cached
        positions = torch.arange(seq_len, dtype=torch.long, device=device)
        rel = positions.unsqueeze(0) - positions.unsqueeze(1)
        rel = torch.clamp(rel, -self.max_relative_positions, self.max_relative_positions)
        rel = rel + self.max_relative_positions
        self._rel_idx_cache[key] = rel
        return rel

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        H, D = self.head_num, self.head_dim

        # Content projections: [B, H, S, D]
        q_c = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=H)
        k_c = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=H)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=H)

        # Position projections of the relative position embeddings: [H, 2k+1, D]
        q_r = rearrange(self.pos_q_proj(self.relative_pos_emb), 'n (h d) -> h n d', h=H)
        k_r = rearrange(self.pos_k_proj(self.relative_pos_emb), 'n (h d) -> h n d', h=H)

        # Content-to-content: [B, H, S, S]
        attn_cc = torch.matmul(q_c, k_c.transpose(-2, -1))

        # Relative position index matrix: idx[i, j] = clipped(j - i) + k
        rel_idx = self._get_relative_positions(S, x.device)  # [S, S]

        # Content-to-position:
        #   cp_score[b, h, i, δ] = q_c[b, h, i] · k_r[h, δ]
        #   attn_cp[b, h, i, j]  = cp_score[b, h, i, idx[i, j]]
        cp_score = torch.einsum('bhid,hjd->bhij', q_c, k_r)  # [B, H, S, 2k+1]
        cp_idx = rel_idx.unsqueeze(0).unsqueeze(0).expand(B, H, S, S)
        attn_cp = torch.gather(cp_score, dim=-1, index=cp_idx)

        # Position-to-content:
        #   pc_score[b, h, j, δ] = k_c[b, h, j] · q_r[h, δ]
        #   attn_pc[b, h, i, j]  = pc_score[b, h, j, idx[j, i]]
        # Using gather along dim -1 with index shape [B, H, S(=j), S(=i)] directly uses idx[j, i],
        # then transpose swaps (j, i) -> (i, j).
        pc_score = torch.einsum('bhjd,hkd->bhjk', k_c, q_r)  # [B, H, S, 2k+1]
        pc_idx = rel_idx.unsqueeze(0).unsqueeze(0).expand(B, H, S, S)
        attn_pc = torch.gather(pc_score, dim=-1, index=pc_idx).transpose(-2, -1)

        attn_scores = (attn_cc + attn_cp + attn_pc) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, torch.finfo(attn_scores.dtype).min)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = rearrange(context, 'b h n d -> b n (h d)')

        return self.projection(context)
