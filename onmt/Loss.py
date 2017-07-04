import onmt
import onmt.Constants
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable

def NMTCriterion(vocabSize, opt):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit

class MemoryEfficientLoss:
    def __init__(self, generator, crit, eval=False):
        self.generator = generator
        self.crit = crit
        self.eval = eval
        
    def loss(batch, outputs, attns=None, coverage=None, copy=None):
        """
        Args:
            outputs (FloatTensor): tgt_len x batch x rnn_size
            generator (Function): ( any x rnn_size ) -> ( any x tgt_vocab )
            crit (Criterion): ( any x tgt_vocab )
            batch (Batch): Data object
            eval (bool): train or eval
            attns (FloatTensor): src_len x batch

        Returns:
            loss (float): accumulated loss value
            grad_output: grad of loss wrt outputs
            grad_attns: grad of loss wrt attns
            num_correct (int): number of correct targets

        """
        targets = batch.tgt[1:]

        # compute generations one piece at a time
        num_correct, loss = 0, 0

        # These will require gradients.
        outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)
        batch_size = batch.batchSize
        d = {"out": outputs, "tgt": targets}

        if attns is not None:
            attns = Variable(attns.data, requires_grad=(not eval), volatile=eval)
            d["attn"] = attns
            d["align"] = batch.alignment[1:]

        if coverage is not None:
            coverage = Variable(coverage.data, requires_grad=(not eval),
                                volatile=eval)
            d["coverage"] = coverage

        for k in d:
            d[k] = torch.split(d[k], opt.max_generator_batches)

        for i, targ_t in enumerate(d["tgt"]):
            out_t = d["out"][i].view(-1, d["out"][i].size(2))

            # Depending on generator type.
            if attns is None:
                scores_t = generator(out_t)
                loss_t = crit(scores_t, targ_t.view(-1))
            else:
                attn_t = d["attn"][i]
                align_t = d["align"][i].view(-1, d["align"][i].size(2))
                words = batch.words().t().contiguous()
                attn_t = attn_t.view(-1, d["attn"][i].size(2))

                # probability of words, probability of attn
                scores_t, c_attn_t = generator(out_t, words, attn_t)
                loss_t = crit(scores_t, c_attn_t, targ_t.view(-1), align_t)

            if coverage is not None:
                loss_t += 0.1 * torch.min(d["coverage"][i], d["attn"][i]).sum()

            pred_t = scores_t.data.max(1)[1]
            num_correct_t = pred_t.eq(targ_t.data) \
                                  .masked_select(
                                      targ_t.ne(onmt.Constants.PAD).data) \
                                  .sum()
            num_correct += num_correct_t
            loss += loss_t.data[0]
            if not eval:
                loss_t.div(batch_size).backward()

        # Return the gradients
        inputs = []
        grads = []
        def add(v):
            if v and v.grad is not None:
                inputs.append(v)
                grads.append(v.grad.data)
        add(outputs, attns, coverage)
                
        return {"loss" : loss,
                "num_correct": num_correct,
                "num_words" : targets.data.ne(onmt.Constants.PAD).sum()}, inputs, grads
    
