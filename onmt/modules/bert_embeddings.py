import torch
import torch.nn as nn


class BertEmbeddings(nn.Module):
    """ BERT input embeddings is sum of:
           1. Token embeddings: called word_embeddings
           2. Segmentation embeddings: called token_type_embeddings
           3. Position embeddings: called position_embeddings
        Ref: https://arxiv.org/abs/1810.04805 section 3.2
    """
    def __init__(self, vocab_size, embed_size=768, pad_idx=0,
                 dropout=0.1, max_position=512, num_sentence=2):
        """
        Args:
            vocab_size: int. Size of the embedding vocabulary.
            embed_size: int. Width of the word embeddings.
            dropout: dropout rate
            pad_idx: padding index
            max_position: max sentence length in input
        """
        super(BertEmbeddings, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.word_padding_idx = pad_idx
        # Token embeddings: for input tokens
        self.word_embeddings = nn.Embedding(vocab_size, embed_size,
                                        padding_idx=pad_idx)
        # Position embeddings: for Position Encoding
        self.position_embeddings = nn.Embedding(max_position, embed_size)
        # Segmentation embeddings: for distinguish sentences A/B
        self.token_type_embeddings = nn.Embedding(num_sentence, embed_size,
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
        # NOTE: in our version, LayerNorm is done in EncoderLayer
        # before fed into Attention comparing to original implementation
        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def update_dropout(self, dropout):
        self.dropout.p = dropout
