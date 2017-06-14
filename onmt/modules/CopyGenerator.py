import onmt
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable


"""
Generator module that additionally considers copying 
words directly from the source. 
"""

class CopyGenerator(nn.Module):
    def __init__(self, opt, src_dict, tgt_dict):
        super(CopyGenerator, self).__init__()
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.linear = nn.Linear(opt.rnn_size, tgt_dict.size())
        self.alignment = torch.cuda.LongTensor(src_dict.align(tgt_dict))
        self.opt = opt
        
    def forward(self, hidden, src, attn):
        """
        src : src_len x batch x features
        attn : (len * batch) x src_len
        hidden : (len * batch) x hidden
        """
        
        # Add a variable for each of the source words.
        words = src[:, :, 0].data.contiguous()
        
        size = words.size()
        full_size = hidden.size(0)
        batch_size = size[1]
        len_size = full_size // batch_size

        # Original probabilities.
        logits = self.linear(hidden)
        copy = F.sigmoid(logits[:, onmt.Constants.COPY])
        logits2 = logits.clone()
        logits2[:, onmt.Constants.COPY] = 1e-10
        prob = F.softmax(logits2)
        debug = False # copy.data[0] > 0.1
        if debug:
            print("\tCOPY %3f"%copy.data[0])
            v, mid = prob[0].data.max(0)
            print(self.tgt_dict.getLabel(mid[0], "FAIL"), v[0])
        
        # Mapping of source words to targets
        src_to_target = Variable(self.alignment[words.view(-1)]\
                                 .contiguous().view(size) \
                                 .t().contiguous())

        # Probability of copying each word.
        mul_attn = torch.mul(attn, copy.view(copy.size() + (1,)).expand_as(attn))
        eps = 1e-10
        
        # Add in the copying of each word to the generated prob.
        prob_size = prob.size()
        prob = torch.mul(prob,  1 - copy.view(copy.size() + (1,)).expand_as(prob))
        prob = prob.view(len_size, batch_size, -1)
        mul_attn = mul_attn.view(len_size, batch_size, -1)
        out_prob = prob.clone()


        
        for b in range(batch_size):
            out_prob[:, b] = prob[:, b].index_add(1, src_to_target[b], mul_attn[:, b])
        if debug:
            _, ids = attn[0].cpu().data.sort(0, descending=True)


            total = {}
            for j in ids.tolist():
                src_idx = src[j, 0, 0].data[0]
                al = self.alignment[src_idx]
                lab = self.tgt_dict.getLabel(al),
                total.setdefault(lab, 0)
                total[lab] += mul_attn[0, 0, j].data[0]
            print(total)
            for j in ids[:10].tolist():
                src_idx = src[j, 0, 0].data[0]
                al = self.alignment[src_idx]
                print("\t%s\t%s\t%3f\t%3f\t%d\t%3f" % (self.src_dict.getLabel(src_idx),
                                             self.tgt_dict.getLabel(al),
                                             prob[0, 0, al].data[0],
                                             out_prob[0, 0, al].data[0],
                                             j,
                                             mul_attn[0, 0, j].data[0]))

        # Drop any uncopyable terms. 
        out_prob[:, :, onmt.Constants.PAD] = 1e-10
        if debug:
            v, mid = out_prob[0,0].data.max(0)
            print(self.tgt_dict.getLabel(mid[0], "FAIL"), v[0])

        # out_prob[:, :, onmt.Constants.COPY] = 1e-20
        norm = out_prob.sum(2).expand(out_prob.size())
        out_prob = out_prob.div(norm)
        return out_prob.add(eps).log().view(prob_size)
