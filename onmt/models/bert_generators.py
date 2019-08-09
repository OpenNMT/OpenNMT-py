import torch

import torch.nn as nn
import onmt
from onmt.utils import get_activation_fn


class BertPreTrainingHeads(nn.Module):
    """
    Bert Pretraining Heads: Masked Language Models, Next Sentence Prediction
    """
    def __init__(self, hidden_size, vocab_size):
        super(BertPreTrainingHeads, self).__init__()
        self.next_sentence = NextSentencePrediction(hidden_size)
        self.mask_lm = MaskedLanguageModel(hidden_size, vocab_size)

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

    def __init__(self, hidden_size, vocab_size):
        """
        Args:
            hidden_size: output size of BERT model
            vocab_size: total vocab size
            bert_word_embedding_weights: reuse embedding weights if set
        """
        super(MaskedLanguageModel, self).__init__()
        self.transform = BertPredictionTransform(hidden_size)

        self.decode = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: first output of bert encoder, (batch, seq, d_model)
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
            x: second output of bert encoder, (batch, d_model)
        Returns:
            seq_class_prob: shape (batch_size, 2)
        """
        seq_relationship_score = self.linear(x)  # (batch, 2)
        seq_class_log_prob = self.softmax(seq_relationship_score)
        return seq_class_log_prob


class BertPredictionTransform(nn.Module):
    def __init__(self, hidden_size):
        super(BertPredictionTransform, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = get_activation_fn('gelu')
        self.layer_norm = onmt.encoders.BertLayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: BERT model output size (batch, seq, d_model)
        """
        hidden_states = self.layer_norm(self.activation(
                                        self.dense(hidden_states)))
        return hidden_states


class ClassificationHead(nn.Module):
    """
    n-class Sentence classification head
    """

    def __init__(self, hidden_size, n_class, dropout=0.1):
        """
        Args:
            hidden_size: BERT model output size
        """
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, n_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, all_hidden, pooled):
        """
        Args:
            all_hidden: layer output of BERT, list [(batch, seq, d_model)]
            pooled: last layer hidden [CLS] of BERT, (batch, d_model)
        Returns:
            class_log_prob: shape (batch_size, 2)
            None: this is a placeholder for token level prediction task
        """
        pooled = self.dropout(pooled)
        score = self.linear(pooled)  # (batch, n_class)
        class_log_prob = self.softmax(score)  # (batch, n_class)
        return class_log_prob, None


class TokenTaggingHead(nn.Module):
    """
    n-class Token Tagging head
    """

    def __init__(self, hidden_size, n_class, dropout=0.1):
        """
        Args:
            hidden_size: BERT model output size
        """
        super(TokenTaggingHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, n_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, all_hidden, pooled):
        """
        Args:
            all_hidden: layer output of BERT, list [(batch, seq, d_model)]
            pooled: last layer hidden [CLS] of BERT, (batch, d_model)
        Returns:
            None: this is a placeholder for sentence level task
            tok_class_log_prob: shape (batch, seq, n_class)
        """
        last_hidden = all_hidden[-1]
        last_hidden = self.dropout(last_hidden)  # (batch, seq, d_model)
        score = self.linear(last_hidden)  # (batch, seq, n_class)
        tok_class_log_prob = self.softmax(score)  # (batch, seq, n_class)
        return None, tok_class_log_prob


class TokenGenerationHead(nn.Module):
    """
    Token generation head: generation token from input sequence
    """

    def __init__(self, hidden_size, vocab_size):
        """
        Args:
            hidden_size: output size of BERT model
            vocab_size: total vocab size
            bert_word_embedding_weights: reuse embedding weights if set
        """
        super(TokenGenerationHead, self).__init__()
        self.transform = BertPredictionTransform(hidden_size)

        self.decode = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, all_hidden, pooled):
        """
        Args:
            all_hidden: layer output of BERT, list [(batch, seq, d_model)]
        Returns:
            None: this is a placeholder for sentence level task
            prediction_log_prob: shape (batch, seq, vocab)
        """
        last_hidden = all_hidden[-1]
        y = self.transform(last_hidden)  # (batch, seq, d_model)
        prediction_scores = self.decode(y) + self.bias  # (batch, seq, vocab)
        prediction_log_prob = self.softmax(prediction_scores)
        return None, prediction_log_prob
