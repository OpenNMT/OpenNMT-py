import torch

import torch.nn as nn
import onmt
from onmt.utils.fn_activation import GELU


class BertLM(nn.Module):
    """
    BERT Language Model for pretraining, trained with 2 task :
    Next Sentence Prediction Model + Masked Language Model
    """
    def __init__(self, bert):
        """
        Args:
            bert: BERT model which should be trained
        """
        super(BertLM, self).__init__()
        self.bert = bert
        self.vocab_size = bert.vocab_size
        self.cls = BertPreTrainingHeads(self.bert.d_model, self.vocab_size,
                                self.bert.embeddings.word_embeddings.weight)

    def forward(self, input_ids, token_type_ids, input_mask=None,
                output_all_encoded_layers=False):
        """
        Args:
            input_ids: shape [batch, seq] padding ids=0
            token_type_ids: shape [batch, seq], A(0), B(1), pad(0)
            input_mask: shape [batch, seq], 1 for masked position(that padding)
        Returns:
            seq_class_log_prob: next sentence predi, (batch, 2)
            prediction_log_prob: masked lm predi, (batch, seq, vocab)
        """
        x, pooled_out = self.bert(input_ids, token_type_ids, input_mask,
                                  output_all_encoded_layers)
        seq_class_log_prob, prediction_log_prob = self.cls(x, pooled_out)
        return seq_class_log_prob, prediction_log_prob


class BertPreTrainingHeads(nn.Module):
    """
    Bert Pretraining Heads: Masked Language Models, Next Sentence Prediction
    """
    def __init__(self, hidden_size, vocab_size, embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.next_sentence = NextSentencePrediction(hidden_size)
        self.mask_lm = MaskedLanguageModel(hidden_size, vocab_size,
                                           embedding_weights)

    def forward(self, x, pooled_out):
        """
        Args:
            x: list of out of all_encoder_layers, shape (batch, seq, d_model)
            pooled_output: transformed output of last layer's hidden_states
        Returns:
            seq_class_log_prob: next sentence prediction, (batch, 2)
            prediction_log_prob: masked lm prediction, (batch, seq, vocab)
        """
        seq_class_log_prob = self.next_sentence(pooled_out)
        prediction_log_prob = self.mask_lm(x[-1])
        return seq_class_log_prob, prediction_log_prob


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden_size, vocab_size,
                 bert_word_embedding_weights=None):
        """
        Args:
            hidden_size: output size of BERT model
            vocab_size: total vocab size
            bert_word_embedding_weights: reuse embedding weights if set
        """
        super(MaskedLanguageModel, self).__init__()
        self.transform = BertPredictionTransform(hidden_size)
        self.reuse_emb = (True
                          if bert_word_embedding_weights is not None
                          else False)
        if self.reuse_emb:  # NOTE: reinit ?
            assert hidden_size == bert_word_embedding_weights.size(1)
            assert vocab_size == bert_word_embedding_weights.size(0)
            self.decode = nn.Linear(bert_word_embedding_weights.size(1),
                                    bert_word_embedding_weights.size(0),
                                    bias=False)
            self.decode.weight = bert_word_embedding_weights
            self.bias = nn.Parameter(torch.zeros(vocab_size))
        else:
            self.decode = nn.Linear(hidden_size, vocab_size, bias=False)
            self.bias = nn.Parameter(torch.zeros(vocab_size))

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: last layer output of bert, shape (batch, seq, d_model)
        Returns:
            prediction_log_prob: shape (batch, seq, vocab)
        """
        x = self.transform(x)  # (batch, seq, d_model)
        prediction_scores = self.decode(x) + self.bias  # (batch, seq, vocab)
        prediction_log_prob = self.softmax(prediction_scores)
        return prediction_log_prob


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_random_next
    """

    def __init__(self, hidden_size):
        """
        Args:
            hidden_size: BERT model output size
        """
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: last layer's output of bert encoder, shape (batch, src, d_model)
        Returns:
            seq_class_prob: shape (batch_size, 2)
        """
        seq_relationship_score = self.linear(x)  # (batch, 2)
        seq_class_log_prob = self.softmax(seq_relationship_score)  # (batch, 2)
        return seq_class_log_prob


class BertPredictionTransform(nn.Module):
    def __init__(self, hidden_size):
        super(BertPredictionTransform, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = GELU()  # get_activation fn
        self.layer_norm = onmt.models.BertLayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: BERT model output size (batch, seq, d_model)
        """
        hidden_states = self.layer_norm(self.activation(
                                        self.dense(hidden_states)))
        return hidden_states
