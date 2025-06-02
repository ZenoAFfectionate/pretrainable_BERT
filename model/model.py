import torch.nn as nn

from .utils import *
import torch.nn.functional as F
from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden_dim, head_num, forward_dim, dropout):
        """
        :param hidden_dim: hidden size of transformer
        :param head_dim: head sizes of multi-head attention
        :param forward_dim: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        self.attention = MultiHeadAttention(hidden_dim, head_num)
        self.attn_sublayer = SublayerConnection(hidden_dim, dropout)

        self.feedforward = PositionwiseFeedForward(hidden_dim, forward_dim, dropout)
        self.feed_sublayer = SublayerConnection(hidden_dim, dropout)


    def forward(self, x, mask):
        x = self.attn_sublayer(x, lambda _x: self.attention.forward(_x, mask=mask))
        x = self.feed_sublayer(x, self.feedforward)
        return x  # self.dropout(x)
    

class BERT(nn.Module):
    """ Bidirectional Encoder Representations from Transformers """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

        self.output_norm = nn.LayerNorm(hidden)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """ model weight initalization """
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, token_ids, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        attention_mask = (token_ids != 0).unsqueeze(1).unsqueeze(2)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(token_ids, segment_info)

        # running over multiple transformer blocks
        for layer in self.transformer_blocks:
            x = layer(x, attention_mask)

        return self.output_norm(x)


class NextSentencePrediction(nn.Module):
    """ 2-class classification model : is_next, is_not_next """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2)  # 二分类
        )

    def forward(self, x):
        # x = torch.mean(x, dim=1)
        # return F.log_softmax(self.linear(x), dim=-1)
        return F.log_softmax(self.classifier(x[:, 0]), dim=-1)


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.linear.bias = self.bias

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_st = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_st(x), self.mask_lm(x)
