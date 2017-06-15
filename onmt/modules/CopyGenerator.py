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
        self.alignment = torch.cuda.LongTensor(src_dict.align(tgt_dict))
        self.tgt_dict = tgt_dict
        self.src_dict = src_dict
        
    def forward(self, hidden, src, attn, verbose=False):
        """
        Computes p(w) = p(z=1) p_{copy}(w|z=0)  +  p(z=0) * p_{softmax}(w|z=0)
        
        
        Args:
            hidden (FloatTensor): (tgt_len*batch) x hidden
            src (LongTensor):     (batch) x src_len
            attn (FloatTensor):   (tgt_len*batch) x src_len

        Returns:
            p (FloatTensor):   (tgt_len*batch) x vocab
        """
        # Hack to get around log.
        eps = 1e-10
                
        # Original probabilities.
        logits = self.linear(hidden)
        mod_logits = logits.clone()
        mod_logits[:, onmt.Constants.COPY] = -1e20
        prob = F.softmax(mod_logits)

        # Probability of copying p(z=1) batch
        copy = F.sigmoid(logits[:, onmt.Constants.COPY])
        
        # Mapping of source words to target words for copying.
        # ((tgt_len *batch) x src_len)
        src_to_target = self.alignment[src.data.view(-1)].view(src.data.size())

        # Probability of copying each word: p(z) * p_{copy}(w|z)
        # ((tgt_len *batch) x src_len)
        copy = copy.view(copy.size() + (1,))
        mul_attn = torch.mul(attn, copy.expand_as(attn))

        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        prob = torch.mul(prob,  1 - copy.expand_as(prob))

        # Add in the extra scores.
        out_prob = prob.clone()
        
        # Add batch offsets to make indexing work.
        for b in range(src_to_target.size(0)):
            src_to_target[b] += b * prob.size(1)
        index_add = Variable(src_to_target.view(-1))

        # Mess with views to make index_add fast.
        b = src.size(0)
        size = out_prob.size()
        out_prob = out_prob.view(size[0] // b, b * out_prob.size(1))
        prob = prob.view_as(out_prob)
        mul_attn = mul_attn.view(size[0] // b, b * mul_attn.size(1))
        out_prob = prob.index_add(1, index_add, mul_attn)
        out_prob = out_prob.view(size)
        prob = out_prob.view(size)
        mul_attn = mul_attn.view(attn.size())
        if verbose:
            self._debug_copy(src, copy, prob, out_prob, attn, mul_attn)
        
        # Drop padding and renorm.
        out_prob[:, onmt.Constants.PAD] = eps
        norm = out_prob.sum(1).expand(out_prob.size())

        return out_prob.div(norm).add(eps).log()

    def _debug_copy(self, src, copy, prob, out_prob, attn, mul_attn):
        print("COPY %3f" % copy.data[0][0])
        v, mid = prob[0].data.max(0)
        print("initial:", self.tgt_dict.getLabel(mid[0], "FAIL"), v[0])
        
        _, ids = attn[0].cpu().data.sort(0, descending=True)
        total = {}
        for j in ids[:10].tolist():
            src_idx = src[0 , j].data[0]
            al = self.alignment[src_idx]
            print("\t%s\t%s\t\t%3f\t%d\t%3f\t%3f" % (
                self.src_dict.getLabel(src_idx),
                self.tgt_dict.getLabel(al),
                prob[0, al].data[0],
                j,
                attn[0, j].data[0],
                mul_attn[0, j].data[0]))
        v, mid = out_prob[0].data.max(0)
        print("Final:", self.tgt_dict.getLabel(mid[0], "FAIL"), v[0])

