import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import SublayerConnection, PositionwiseFeedForward, BERTEmbedding
from .attention import MultiHeadAttention, DisentangledSelfAttention


class BERTBlock(nn.Module):
    """BERT Transformer Block = MultiHeadAttention + FeedForward with sublayer connections."""

    def __init__(self, hidden_dim, head_num, forward_dim, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, head_num, dropout)
        self.attn_sublayer = SublayerConnection(hidden_dim, dropout)

        self.feedforward = PositionwiseFeedForward(hidden_dim, forward_dim, dropout)
        self.feed_sublayer = SublayerConnection(hidden_dim, dropout)

    def forward(self, x, mask):
        x = self.attn_sublayer(x, lambda _x: self.attention(_x, mask=mask))
        x = self.feed_sublayer(x, self.feedforward)
        return x


class DeBERTaBlock(nn.Module):
    """DeBERTa Transformer Block = DisentangledSelfAttention + FeedForward with sublayer connections."""

    def __init__(self, hidden_dim, head_num, forward_dim, dropout, max_relative_positions=512):
        super().__init__()
        self.attention = DisentangledSelfAttention(
            hidden_dim, head_num, dropout=dropout,
            max_relative_positions=max_relative_positions,
        )
        self.attn_sublayer = SublayerConnection(hidden_dim, dropout)

        self.feedforward = PositionwiseFeedForward(hidden_dim, forward_dim, dropout)
        self.feed_sublayer = SublayerConnection(hidden_dim, dropout)

    def forward(self, x, mask):
        x = self.attn_sublayer(x, lambda _x: self.attention(_x, mask=mask))
        x = self.feed_sublayer(x, self.feedforward)
        return x


class _BaseEncoder(nn.Module):
    """Shared encoder shell: embedding + transformer blocks + final LayerNorm."""

    def __init__(self, vocab_size, hidden, n_layers, attn_heads, dropout, block_factory):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, dropout=dropout)
        self.transformer_blocks = nn.ModuleList(
            [block_factory() for _ in range(n_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, token_ids, segment_info):
        # Key padding mask: [B, 1, 1, S], broadcast to [B, H, S, S]
        attention_mask = (token_ids != 0).unsqueeze(1).unsqueeze(2)

        x = self.embedding(token_ids, segment_info)
        for layer in self.transformer_blocks:
            x = layer(x, attention_mask)

        return self.output_norm(x)


class BERT(_BaseEncoder):
    """ BERT: Bidirectional Encoder Representations from Transformers """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__(
            vocab_size=vocab_size,
            hidden=hidden,
            n_layers=n_layers,
            attn_heads=attn_heads,
            dropout=dropout,
            block_factory=lambda: BERTBlock(hidden, attn_heads, hidden * 4, dropout),
        )


class DeBERTa(_BaseEncoder):
    """ DeBERTa: Decoding-enhanced BERT with Disentangled Attention """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12,
                 dropout=0.1, max_relative_positions=512):
        super().__init__(
            vocab_size=vocab_size,
            hidden=hidden,
            n_layers=n_layers,
            attn_heads=attn_heads,
            dropout=dropout,
            block_factory=lambda: DeBERTaBlock(
                hidden, attn_heads, hidden * 4, dropout,
                max_relative_positions=max_relative_positions,
            ),
        )
        self.max_relative_positions = max_relative_positions


class NextSentencePrediction(nn.Module):
    """ 2-class classification head: is_next vs is_not_next. Outputs raw logits. """

    def __init__(self, hidden):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.classifier(x[:, 0])


class MaskedLanguageModel(nn.Module):
    """BERT MLM head: Linear(H,H) + GELU + LayerNorm + Linear(H,V).

    Two optimizations beyond the trivial ``Linear(H, V)``:

    * **Tied output weights.** ``tie_weights`` points ``self.decoder.weight``
      at the token-embedding matrix, matching the original BERT paper and
      removing ~V*H params (~23M for Base).
    * **Sparse forward.** When ``mlm_mask`` is supplied (a ``[B, S]`` bool
      tensor marking positions with MLM labels), only those positions are
      projected to vocab logits. On a standard 15% mask rate this cuts the
      vocab matmul by ~6×, which is one of the heaviest operations in a
      pretraining step.
    """

    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.transform = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.decoder = nn.Linear(hidden, vocab_size, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def tie_weights(self, token_embedding_weight: torch.Tensor) -> None:
        """Point the decoder weight at the input token embedding (BERT paper)."""
        assert self.decoder.weight.shape == token_embedding_weight.shape, (
            f"tied-weight shape mismatch: decoder {tuple(self.decoder.weight.shape)} "
            f"vs embedding {tuple(token_embedding_weight.shape)}"
        )
        self.decoder.weight = token_embedding_weight

    def forward(self, hidden, mlm_mask=None):
        if mlm_mask is not None:
            hidden = hidden[mlm_mask]  # [M, H]
        h = self.transform(hidden)
        h = F.gelu(h)
        h = self.norm(h)
        return self.decoder(h)


class EnhancedMaskDecoder(nn.Module):
    """
    DeBERTa Enhanced Mask Decoder (EMD).
    Combines contextualized representations with absolute position embeddings
    for better masked token prediction. Outputs raw logits.

    Same sparse-forward + weight-tying optimizations as ``MaskedLanguageModel``.
    """

    def __init__(self, hidden_size, vocab_size, max_position_embeddings=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.position_embeddings.weight, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

        if self.dense.bias is not None:
            nn.init.zeros_(self.dense.bias)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)

    def tie_weights(self, token_embedding_weight: torch.Tensor) -> None:
        """Point the decoder weight at the input token embedding."""
        assert self.decoder.weight.shape == token_embedding_weight.shape
        self.decoder.weight = token_embedding_weight

    def forward(self, contextualized_embeddings, mlm_mask=None, position_ids=None):
        batch_size, seq_len, _ = contextualized_embeddings.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long,
                                        device=contextualized_embeddings.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        position_embeddings = self.position_embeddings(position_ids)

        enhanced = contextualized_embeddings + position_embeddings
        enhanced = self.layer_norm(enhanced)

        # Gather masked positions before the two hidden-sized matmuls so we
        # project only the positions that contribute to the loss.
        if mlm_mask is not None:
            enhanced = enhanced[mlm_mask]  # [M, H]

        enhanced = F.gelu(self.dense(enhanced))
        return self.decoder(enhanced)


class BERTLM(nn.Module):
    """ BERT pre-training model: Next Sentence Prediction + Masked Language Model. """

    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.next_st = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)
        # Tie the MLM decoder weight to the input token embedding (BERT paper).
        self.mask_lm.tie_weights(self.bert.embedding.token.weight)

    def forward(self, x, segment_label, mlm_mask=None):
        x = self.bert(x, segment_label)
        return self.next_st(x), self.mask_lm(x, mlm_mask=mlm_mask)


class DeBERTaLM(nn.Module):
    """ DeBERTa pre-training model: Next Sentence Prediction + Enhanced Mask Decoder. """

    def __init__(self, deberta: DeBERTa, vocab_size):
        super().__init__()
        self.bert = deberta  # alias for API compatibility with BERTLM
        self.deberta = deberta
        self.next_st = NextSentencePrediction(self.deberta.hidden)
        self.enhanced_mask_decoder = EnhancedMaskDecoder(self.deberta.hidden, vocab_size)
        # Tie EMD's output projection to the input token embedding.
        self.enhanced_mask_decoder.tie_weights(self.deberta.embedding.token.weight)

    def forward(self, x, segment_label, mlm_mask=None):
        contextualized = self.deberta(x, segment_label)
        nsp_output = self.next_st(contextualized)
        emd_output = self.enhanced_mask_decoder(contextualized, mlm_mask=mlm_mask)
        return nsp_output, emd_output
