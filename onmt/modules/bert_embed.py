import torch
import torch.nn as nn


class TokenEmb(nn.Embedding):
    """ Embeddings for tokens.
    """
    def __init__(self, vocab_size, hidden_size=768, padding_idx=0):
        super(TokenEmb, self).__init__(vocab_size, hidden_size,
                                       padding_idx=padding_idx)


class SegmentEmb(nn.Embedding):
    """ Embeddings for token's type: sentence A(0), sentence B(1). Padding with 0.
    """
    def __init__(self, type_vocab_size=2, hidden_size=768, padding_idx=0):
        super(SegmentEmb, self).__init__(type_vocab_size, hidden_size,
                                         padding_idx=padding_idx)


class PositionEmb(nn.Embedding):
    """ Embeddings for token's position.
    """
    def __init__(self, max_position=512, hidden_size=768):
        super(PositionEmb, self).__init__(max_position, hidden_size)


class BertEmbeddings(nn.Module):
    """ BERT input embeddings is sum of:
           1. Token embeddings: called word_embeddings
           2. Segmentation embeddings: called token_type_embeddings
           3. Position embeddings: called position_embeddings
    """
    def __init__(self, vocab_size, embed_size, pad_idx=0, dropout=0.1):
        """
        Args:
            vocab_size: int. Size of the embedding vocabulary.
            embed_size: int. Width of the word embeddings.
            dropout: dropout rate
        """
        super(BertEmbeddings, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.word_padding_idx = pad_idx
        self.word_embeddings = TokenEmb(vocab_size, hidden_size=embed_size,
                                        padding_idx=pad_idx)
        self.position_embeddings = PositionEmb(512, hidden_size=embed_size)
        self.token_type_embeddings = SegmentEmb(2, hidden_size=embed_size,
                                                padding_idx=pad_idx)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        """
        Args:
            input_ids: word ids in shape [batch, seq, hidden_size].
            token_type_ids: token type ids in shape [batch, seq].
        Output:
            embeddings: word embeds in shape [batch, seq, hidden_size].
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long,
                       device=input_ids.device)  # [0, 1,..., seq_length-1]
        # [[0,1,...,seq_length-1]] -> [[0,1,...,seq_length-1] *batch_size]
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        # in our version, LN is done in EncoderLayer before fed into Attention
        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def update_dropout(self, dropout):
        self.dropout.p = dropout
