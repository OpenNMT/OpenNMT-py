import onmt
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable


class CopyGenerator(nn.Module):
    """
    Generator module that additionally considers copying
    words directly from the source.
    """

    def __init__(self, opt, src_dict, tgt_dict):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(opt.rnn_size, tgt_dict.size())
        self.linear_copy = nn.Linear(opt.rnn_size, 1)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    def forward(self, hidden, attn, verbose=False):
        """
        Computes p(w) = p(z=1) p_{copy}(w|z=0)  +  p(z=0) * p_{softmax}(w|z=0)


        Args:
            hidden (FloatTensor): (tgt_len*batch) x hidden
            attn (FloatTensor):   (tgt_len*batch) x src_len

        Returns:
            prob (FloatTensor):   (tgt_len*batch) x vocab
            attn (FloatTensor):   (tgt_len*batch) x src_len
        """
        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, onmt.Constants.UNK] = -float('inf')
        logits[:, onmt.Constants.PAD] = -float('inf')
        prob = F.softmax(logits)

        # Probability of copying p(z=1) batch
        copy = F.sigmoid(self.linear_copy(hidden))

        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob,  1 - copy.expand_as(prob))
        mul_attn = torch.mul(attn, copy.expand_as(attn))
        return out_prob, mul_attn

    def _debug_copy(self, src, copy, prob, out_prob, attn, mul_attn):
        v, mid = prob[0].data.max(0)
        print("Initial:", self.tgt_dict.getLabel(mid[0], "FAIL"), v[0])
        print("COPY %3f" % copy.data[0][0])
        _, ids = attn[0].cpu().data.sort(0, descending=True)
        for j in ids[:10].tolist():
            src_idx = src[0, j].data[0]
            print("\t%s\t\t%d\t%3f\t%3f" % (
                self.src_dict.getLabel(src_idx),
                j,
                attn[0, j].data[0],
                mul_attn[0, j].data[0]))


def CopyCriterion(probs, attn, targ, align, eps=1e-12):
    copies = attn.mul(Variable(align)).sum(-1).add(eps)
    # Can't use UNK, must copy.
    out = torch.log(probs.gather(1, targ.view(-1, 1)).view(-1) + copies + eps)
    out = out.mul(targ.ne(onmt.Constants.PAD).float())
    return -out.sum()
