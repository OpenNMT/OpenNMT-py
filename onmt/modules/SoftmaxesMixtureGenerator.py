import torch.nn as nn
import torch.nn.functional as F


class SoftmaxesMixtureGenerator(nn.Module):
    """
    Computes a weighted sum of several softmaxes.
    
    The number of softmaxes is set by `opt.softmaxes_mixture`.

    See the paper for details.
    
    Papers:
        * Zhilin Yang et al. 2017. Breaking the Softmax Bottleneck:
          A High-Rank RNN Language Model.
    """
    def __init__(self, opt, vocab_size):
        super(SoftmaxesMixtureGenerator, self).__init__()

        K = self.K = opt.softmaxes_mixture
        self.vocab_size = vocab_size
        assert K >= 1

        self.context_size = opt.tgt_word_vec_size
        self.mixture_weight = nn.Linear(opt.rnn_size, K)
        self.context = nn.Sequential(
            nn.Linear(opt.rnn_size, K * self.context_size),
            nn.Tanh())
        self.linear_final = nn.Linear(self.context_size, vocab_size)

    def forward(self, hidden):
        batch_size = hidden.size(0)

        # Compute mixture weights (\pi_{c,k} in the paper)
        weight_logit = self.mixture_weight(hidden)  # (batch, K)
        weight = F.softmax(weight_logit, dim=1)     # (batch, K)

        # Compute K new context vectors (h_{c_t,k} in the paper)
        context = self.context(hidden).view(batch_size, self.K, self.context_size)
        # TODO: dropout?

        # Compute K softmaxes and mix them according to weights
        logits = self.linear_final(context)           # (batch, K, vocab)
        probs = F.softmax(logits, dim=2)              # (batch, K, vocab)
        probs = (probs * weight.unsqueeze(2)).sum(1)  # (batch, vocab)
        probs = probs.log()

        return probs
