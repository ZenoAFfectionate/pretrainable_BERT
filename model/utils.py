import math

import torch
import torch.nn as nn


class SublayerConnection(nn.Module):
    """ residual connection and layer normalization """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # post-layer norm structure like BERT (original paper)
        return self.norm(x + self.dropout(sublayer(x)))


class PositionwiseFeedForward(nn.Module):
    """Feed-forward network with GELU activation.

    One dropout after the final linear is enough; the outer
    :class:`SublayerConnection` already dropouts the residual branch before
    LayerNorm. Adding a mid-FFN dropout (as an older draft did) amounts to
    double regularization and is not what the BERT paper specifies.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
        self.reset_parameters()  # Apply custom initialization
    
    def reset_parameters(self):
        """Initialize weights with normal distribution"""
        nn.init.normal_(self.weight, mean=0, std=0.02)
        # Zero-initialize padding token
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Same initialization as token embeddings"""
        nn.init.normal_(self.weight, mean=0, std=0.02)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)


class PositionalEmbedding(nn.Module):
    """BERT-style learnable absolute positional embeddings"""

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Use learnable positional embeddings like BERT
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self._init_weights()

    def _init_weights(self):
        """Initialize positional embeddings with normal distribution"""
        nn.init.normal_(self.position_embeddings.weight, mean=0, std=0.02)

    def forward(self, x):
        '''Generate positional embeddings for input sequence'''
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )

        # Create position indices
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)  # [batch_size, seq_len]

        return self.position_embeddings(position_ids)  # [batch_size, seq_len, d_model]
    

class BERTEmbedding(nn.Module):
    """
    BERT Embedding module combining:
        1. Token embeddings
        2. Positional embeddings
        3. Segment embeddings

    Sums all three, applies LayerNorm, then Dropout — matching the original
    BERT paper (see §3.2: "The input representation... is the sum of the
    corresponding token, segment, and position embeddings" fed through
    a LayerNorm + Dropout). Skipping the LayerNorm inflates gradient scale
    on the embedding tables and hurts early-training stability.
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.embed_size = embed_size

        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)

        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, segment_label):
        tok_emb = self.token(sequence)
        pos_emb = self.position(sequence)
        seg_emb = self.segment(segment_label)

        x = tok_emb + pos_emb + seg_emb
        return self.dropout(self.norm(x))
