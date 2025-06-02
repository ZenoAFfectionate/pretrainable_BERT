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
        # pre-layer norm structure improves stability
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """ Feed-forward network with GELU activation """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self._init_weights()
    
    def _init_weights(self):
        ''' Xavier initalization fit for Transformer '''
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.net(x)


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

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Precompute positional encodings during initialization
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)

        # stable calculation of frequency terms using log space
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)  # sin to even
        pe[:, 1::2] = torch.cos(position * div_term)  # cos to odds

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        '''Slice precomputed positional embeddings to sequence length'''
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )
        return self.pe[:, :seq_len]  # [1, seq_len, d_model]
    

class BERTEmbedding(nn.Module):
    """
    BERT Embedding module combining:
        1. Token embeddings
        2. Positional embeddings
        3. Segment embeddings
    
    Sums all embeddings and applies dropout
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.embed_size = embed_size
        self.dropout = nn.Dropout(p=dropout)

        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)

    def forward(self, sequence, segment_label):
        tok_emb = self.token(sequence)
        pos_emb = self.position(sequence)
        seg_emb = self.segment(segment_label)
        
        x = tok_emb + pos_emb + seg_emb
        return self.dropout(x)
