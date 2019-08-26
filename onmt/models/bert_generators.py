import torch

import torch.nn as nn
import onmt
from onmt.utils import get_activation_fn


class BertPreTrainingHeads(nn.Module):
    """
    Bert Pretraining Heads: Masked Language Models, Next Sentence Prediction

    Args:
            hidden_size (int): output size of BERT model
            vocab_size (int): total vocab size
    """
    def __init__(self, hidden_size, vocab_size):
        super(BertPreTrainingHeads, self).__init__()
        self.next_sentence = NextSentencePrediction(hidden_size)
        self.mask_lm = MaskedLanguageModel(hidden_size, vocab_size)

    def forward(self, x, pooled_out):
        """
        Args:
            x (list of Tensor): all_encoder_layers, shape ``(B, S, H)``
            pooled_output (Tensor): second output of bert encoder, ``(B, H)``
        Returns:
            seq_class_log_prob (Tensor): next sentence prediction, ``(B, 2)``
            prediction_log_prob (Tensor): mlm prediction, ``(B, S, vocab)``
        """
        seq_class_log_prob = self.next_sentence(pooled_out)
        prediction_log_prob = self.mask_lm(x[-1])
        return seq_class_log_prob, prediction_log_prob


class MaskedLanguageModel(nn.Module):
    """predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size

    Args:
            hidden_size (int): output size of BERT model
            vocab_size (int): total vocab size
    """

    def __init__(self, hidden_size, vocab_size):
        super(MaskedLanguageModel, self).__init__()
        self.transform = BertPredictionTransform(hidden_size)

        self.decode = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x (Tensor): first output of bert encoder, ``(B, S, H)``
        Returns:
            prediction_log_prob (Tensor): shape ``(B, S, vocab)``
        """
        x = self.transform(x)  # (batch, seq, d_model)
        prediction_scores = self.decode(x) + self.bias  # (batch, seq, vocab)
        prediction_log_prob = self.softmax(prediction_scores)
        return prediction_log_prob


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_random_next

    Args:
            hidden_size (int): BERT model output size
    """

    def __init__(self, hidden_size):
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x (Tensor): second output of bert encoder, ``(B, H)``
        Returns:
            seq_class_prob (Tensor): ``(B, 2)``
        """
        seq_relationship_score = self.linear(x)  # (batch, 2)
        seq_class_log_prob = self.softmax(seq_relationship_score)
        return seq_class_log_prob


class BertPredictionTransform(nn.Module):
    """{Linear(h,h), Activation, LN} block."""

    def __init__(self, hidden_size):
        """
        Args:
            hidden_size (int): BERT model hidden layer size.
        """

        super(BertPredictionTransform, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = get_activation_fn('gelu')
        self.layer_norm = onmt.encoders.BertLayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (Tensor): BERT encoder output ``(B, S, H)``
        """

        hidden_states = self.layer_norm(self.activation(
                                        self.dense(hidden_states)))
        return hidden_states


class ClassificationHead(nn.Module):
    """n-class Sentence classification head

    Args:
        hidden_size (int): BERT model output size
        n_class (int): number of classification label
    """

    def __init__(self, hidden_size, n_class, dropout=0.1):
        """
        """
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, n_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, all_hidden, pooled):
        """
        Args:
            all_hidden (list of Tensor): layers output, list [``(B, S, H)``]
            pooled (Tensor): last layer hidden [CLS], ``(B, H)``
        Returns:
            class_log_prob (Tensor): shape ``(B, 2)``
            None: this is a placeholder for token level prediction task
        """

        pooled = self.dropout(pooled)
        score = self.linear(pooled)  # (batch, n_class)
        class_log_prob = self.softmax(score)  # (batch, n_class)
        return class_log_prob, None


class TokenTaggingHead(nn.Module):
    """n-class Token Tagging head

    Args:
        hidden_size (int): BERT model output size
        n_class (int): number of tagging label
    """

    def __init__(self, hidden_size, n_class, dropout=0.1):
        super(TokenTaggingHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, n_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, all_hidden, pooled):
        """
        Args:
            all_hidden (list of Tensor): layers output, list [``(B, S, H)``]
            pooled (Tensor): last layer hidden [CLS], ``(B, H)``
        Returns:
            None: this is a placeholder for sentence level task
            tok_class_log_prob (Tensor): shape ``(B, S, n_class)``
        """
        last_hidden = all_hidden[-1]
        last_hidden = self.dropout(last_hidden)  # (batch, seq, d_model)
        score = self.linear(last_hidden)  # (batch, seq, n_class)
        tok_class_log_prob = self.softmax(score)  # (batch, seq, n_class)
        return None, tok_class_log_prob


class TokenGenerationHead(nn.Module):
    """
    Token generation head: generation token from input sequence

    Args:
            hidden_size (int): output size of BERT model
            vocab_size (int): total vocab size
    """

    def __init__(self, hidden_size, vocab_size):
        super(TokenGenerationHead, self).__init__()
        self.transform = BertPredictionTransform(hidden_size)

        self.decode = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, all_hidden, pooled):
        """
        Args:
            all_hidden (list of Tensor): layers output, list [``(B, S, H)``]
            pooled (Tensor): last layer hidden [CLS], ``(B, H)``
        Returns:
            None: this is a placeholder for sentence level task
            prediction_log_prob (Tensor): shape ``(B, S, vocab)``
        """
        last_hidden = all_hidden[-1]
        y = self.transform(last_hidden)  # (batch, seq, d_model)
        prediction_scores = self.decode(y) + self.bias  # (batch, seq, vocab)
        prediction_log_prob = self.softmax(prediction_scores)
        return None, prediction_log_prob
