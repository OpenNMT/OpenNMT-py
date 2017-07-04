import onmt
import onmt.Constants
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import sys
import math


def NMTCriterion(vocabSize, opt):
    """
    Construct the standard NMT Criterion
    """
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


class Statistics:
    """
    Training loss function statistics.
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = 0
        self.n_words = 0
        self.n_correct = 0
        self.n_src_words = 0
        self.start_time = time.time()
        
    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / float(self.n_words))
    
    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time
    
    def output(self, epoch, batch, n_batches):
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f;" +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5), 
               self.n_words / (t + 1e-5),
               t))
        sys.stdout.flush()

    def log(self, prefix, experiment, optim):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", optim.lr)
    

def chunkVars(max_len, ls):
    ls_var = Variable(ls.data, requires_grad=(not self.eval),
                     volatile=self.eval)
    return torch.split(ls_var, self.max_batches)
    


    def loss(self, batch, outputs, attns):

class MemoryEfficientLoss:
    """
    Class for best batchin the loss for NMT.
    """
    def __init__(self, opt, generator, crit, eval=False):
        """
        Args:
            generator (Function): ( any x rnn_size ) -> ( any x tgt_vocab )
            crit (Criterion): ( any x tgt_vocab )
            eval (bool): train or eval
        """
        self.generator = generator
        self.crit = crit
        self.eval = eval
        self.max_batches = opt.max_generator_batches

    def score(self, loss_t, scores_t, targ_t):
        pred_t = scores_t.data.max(1)[1]
        non_padding = targ_t.ne(onmt.Constants.PAD).data
        num_correct_t = pred_t.eq(targ_t.data) \
                              .masked_select(non_padding) \
                              .sum()
        return Statistics(loss_t.data[0], non_padding.sum(),
                          num_correct_t)


    def compute_loss(shard):
        if :
            scores_t = self.generator(out_t)
            loss_t = self.crit(scores_t, targ_t)
        else:
            
    
    def loss(self, batch, outputs, attns):
        """
        Args:
            batch (Batch): Data object
            outputs (FloatTensor): tgt_len x batch x rnn_size
            attns (FloatTensor): src_len x batch
        Returns:
            stats (dict): Statistics about loss
            inputs: list of variables with grads
            grads: list of grads corresponding to inputs
        """
        stats = Statistics()
        def bottle(v):
            return v.view(-1, v.size(2))
        d = {"out": outputs, "tgt": batch.tgt[1:], "attn": attns,
             "align" batch.alignment[1:]}
        v = {}
        n_shards = outputs.size(0) // self.max_batches
        shards = [{} for _ in range(n_shards)]
        for k in d:
            v[k] = Variable(v[k].data, requires_grad=(not self.eval),
                            volatile=self.eval) \
                            if v[k] is Variable else v[k]
            splits = torch.split(v[k], self.max_batches)
            for i, v in enumerate(splits):
                shards[i][k] = bottle(v)

        for s in shards:    
            loss_t, scores_t = self.compute_loss(s) 
            stats.update(self.score(loss_t, scores_t, target_t))
            if not eval:
                loss_t.div(batch.batchSize).backward()

        # Return the gradients
        inputs = []
        grads = []
        if not eval:
            for k in v:
                if v[k] and v[k].grad is not None:
                    inputs.append(v[k])
                    grads.append(v[k].grad.data)
        return stats, inputs, grads
    
