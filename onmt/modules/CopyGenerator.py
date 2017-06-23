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
        
    def forward(self, hidden, src, attn, verbose=False):
        """
        Computes p(w) = p(z=1) p_{copy}(w|z=0)  +  p(z=0) * p_{softmax}(w|z=0)
        
        
        Args:
            hidden (FloatTensor): (tgt_len*batch) x hidden
            attn (FloatTensor):   (tgt_len*batch) x src_len

        Returns:
            prob (FloatTensor):   (tgt_len*batch) x vocab
            attn (FloatTensor):   (tgt_len*batch) x src_len
        """
        # Hack to get around log.
        eps = 1e-10
                
        
        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, onmt.Constants.UNK] = -1e20
        logits[:, onmt.Constants.PAD] = -1e20
        prob = F.softmax(logits)

        # Probability of copying p(z=1) batch
        copy = F.sigmoid(self.linear_copy(hidden))
        
        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob,  1 - copy.expand_as(prob))
        mul_attn = torch.mul(attn, copy.expand_as(attn))
        # self._debug_copy(src, copy, prob, out_prob, attn, mul_attn)
        return out_prob, mul_attn
        
        # # # Probability of copying each word: p(z) * p_{copy}(w|z)
        # # # ((tgt_len *batch) x src_len)
        # # copy = copy.view(copy.size() + (1,))
        # # mul_attn = torch.mul(attn, copy.expand_as(attn))


        # # Add in the extra scores.
        # out_prob = prob.clone()
        
        # # Add batch offsets to make indexing work.
        # for b in range(src_to_target.size(0)):
        #     src_to_target[b] += b * prob.size(1)
        # index_add = Variable(src_to_target.view(-1))

        # # Mess with views to make index_add fast.
        # b = src.size(0)
        # size = out_prob.size()
        # out_prob = out_prob.view(size[0] // b, b * out_prob.size(1))
        # prob = prob.view_as(out_prob)
        # mul_attn = mul_attn.view(size[0] // b, b * mul_attn.size(1))
        # out_prob = prob.index_add(1, index_add, mul_attn)
        # out_prob = out_prob.view(size)
        # if verbose:
        #     prob = out_prob.view(size)
        #     mul_attn = mul_attn.view(attn.size())
        #     self._debug_copy(src, copy, prob, out_prob, attn, mul_attn)
        
        # # Drop padding and renorm.
        # out_prob[:, onmt.Constants.PAD] = eps
        # norm = out_prob.sum(1).expand(out_prob.size())
        # return out_prob.div(norm).add(eps).log()

    def _debug_copy(self, src, copy, prob, out_prob, attn, mul_attn):
        v, mid = prob[0].data.max(0)
        print("Initial:", self.tgt_dict.getLabel(mid[0], "FAIL"), v[0])
        print("COPY %3f" % copy.data[0][0])
        _, ids = attn[0].cpu().data.sort(0, descending=True)
        total = {}
        for j in ids[:10].tolist():
            src_idx = src[0 , j].data[0]
            print("\t%s\t\t%d\t%3f\t%3f" % (
                self.src_dict.getLabel(src_idx),
                j,
                attn[0, j].data[0],
                mul_attn[0, j].data[0]))


def copy_criterion(probs, attn, targ, align, eps=1e-12):
    copies = attn.mul(Variable(align)).sum(-1).add(eps)
    # Can't use UNK, must copy.
    out = torch.log(probs.gather(1, targ.view(-1, 1)).view(-1) + copies + eps)
    out = out.mul(targ.ne(onmt.Constants.PAD).float())
    return -out.sum()
