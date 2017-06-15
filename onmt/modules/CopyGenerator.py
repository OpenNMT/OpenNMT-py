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

    def forward(self, hidden, src, attn):
        """
        hidden (FloatTensor): batch x hidden
        src (LongTensor):     batch x src_len #x features  #src_len x batch x features 
        attn (FloatTensor):   batch x src_len
        """
        # Hack to get around log.
        eps = 1e-10
                
        # Original probabilities.
        logits = self.linear(hidden)
        mod_logits = logits.clone()
        mod_logits[:, onmt.Constants.COPY] = 1e-10
        prob = F.softmax(mod_logits)

        # Probability of copying p(z) batch
        copy = F.sigmoid(logits[:, onmt.Constants.COPY])
        
        # Mapping of source words to targets
        # (batch x src_len)
        src_to_target = Variable(self.alignment[src.data.view(-1)].view(src.data.size()))

        # Probability of copying each word: p(z) * attn
        # (tgt_len x batch x src_len)
        copy = copy.view(copy.size() + (1,))
        mul_attn = torch.mul(attn, copy.expand_as(attn))

        # Probibility of not copying: p(w) * (1 - p(z))
        prob = torch.mul(prob,  1 - copy.expand_as(prob))

        # Add in the extra scores.
        out_prob = prob.clone()
        for b in range(prob.size(0)):
            out_prob[b] = prob[b].index_add(0, src_to_target[b], mul_attn[b])

        # drop padding and renorm.
        out_prob[:, onmt.Constants.PAD] = eps
        norm = out_prob.sum(1).expand(out_prob.size())
        return out_prob.div(norm).add(eps).log()

        # debug = False # copy.data[0] > 0.1
        # if debug:
        #     print("\tCOPY %3f"%copy.data[0])
        #     v, mid = prob[0].data.max(0)
        #     print(self.tgt_dict.getLabel(mid[0], "FAIL"), v[0])

        # Add in the copying of each word to the generated prob.
        # prob_size = prob.size()        
        # prob = prob.view(len_size, batch_size, -1)
        # mul_attn = mul_attn.view(len_size, batch_size, -1)
        # out_prob = prob.clone()
        
            
        # if debug:
        #     _, ids = attn[0].cpu().data.sort(0, descending=True)

        #     total = {}
        #     for j in ids.tolist():
        #         src_idx = src[j, 0, 0].data[0]
        #         al = self.alignment[src_idx]
        #         lab = self.tgt_dict.getLabel(al),
        #         total.setdefault(lab, 0)
        #         total[lab] += mul_attn[0, 0, j].data[0]
        #     print(total)
        #     for j in ids[:10].tolist():
        #         src_idx = src[j, 0, 0].data[0]
        #         al = self.alignment[src_idx]
        #         print("\t%s\t%s\t%3f\t%3f\t%d\t%3f" % (
        # self.src_dict.getLabel(src_idx),
        #                                      self.tgt_dict.getLabel(al),
        #                                      prob[0, 0, al].data[0],
        #                                      out_prob[0, 0, al].data[0],
        #                                      j,
        #                                      mul_attn[0, 0, j].data[0]))

        # Drop any uncopyable terms.

        # if debug:
        #     v, mid = out_prob[0,0].data.max(0)
        #     print(self.tgt_dict.getLabel(mid[0], "FAIL"), v[0])

        # out_prob[:, :, onmt.Constants.COPY] = 1e-20
