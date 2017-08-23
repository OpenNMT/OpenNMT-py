import onmt
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from onmt.modules import aeq


class CopyGenerator(nn.Module):
    """
    Generator module that additionally considers copying
    words directly from the source.
    """

    def __init__(self, opt, src_dict, tgt_dict):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(opt.rnn_size, len(tgt_dict))
        self.linear_copy = nn.Linear(opt.rnn_size, 1)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.sm = nn.Softmax()

    def forward(self, hidden, attn, src_map, verbose=False):
        """
        Computes p(w) = p(z=1) p_{copy}(w|z=0)  +  p(z=0) * p_{softmax}(w|z=0)
        """
        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)

        if True:
            logits[:, self.tgt_dict.stoi[onmt.IO.PAD_WORD]] = -float('inf')

            prob = self.sm(logits)
            # Probability of copying p(z=1) batch
            copy = F.sigmoid(self.linear_copy(hidden))

            # Probibility of not copying: p_{word}(w) * (1 - p(z))
            out_prob = torch.mul(prob,  1 - copy.expand_as(prob))
            mul_attn = torch.mul(attn, copy.expand_as(attn))
            copy_prob = torch.bmm(mul_attn.view(-1, batch, slen)
                                  .transpose(0, 1),
                                  src_map.transpose(0, 1)).transpose(0, 1)
            copy_prob = copy_prob.contiguous().view(-1, cvocab)
            dynamic_probs = torch.cat([out_prob, copy_prob], 1)
        else:
            # copy = self.linear_copy(hidden)
            copy_logit = torch.bmm(attn.view(-1, batch, slen).transpose(0, 1),
                                   src_map.transpose(0, 1)).transpose(0, 1)
            copy_logit = copy_logit.contiguous().view(-1, cvocab)
            copy_logit.data.masked_fill_(copy_logit.data.eq(0), -1e20)
            # print(copy_logit)
            # exit()
            # copy_logit[:, 0] = -1e20
            # copy_logit[:, 1] = -1e20
            dynamic_logits = torch.cat([logits, copy_logit], 1)
            dynamic_probs = self.sm(dynamic_logits.contiguous())
        return dynamic_probs

    def k_debug_copy(self, src, copy, prob, out_prob, attn, mul_attn):
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
