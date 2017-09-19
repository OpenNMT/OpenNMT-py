import torch
import onmt
import onmt.modules


class Trainer(object):
    def __init__(self, model, train_iter, valid_iter,
                 train_loss, valid_loss, fields, optim,
                 batch_size, gpuid,
                 truncated_decoder, max_generator_batches):
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.fields = fields
        self.optim = optim
        self.batch_size = batch_size
        self.gpuid = gpuid
        self.truncated_decoder = truncated_decoder
        self.max_generator_batches = max_generator_batches

        # Set model in training mode.
        self.model.train()

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats = onmt.Statistics()
        report_stats = onmt.Statistics()

        for i, batch in enumerate(self.train_iter):
            target_size = batch.tgt.size(0)

            dec_state = None
            _, src_lengths = batch.src

            src = onmt.IO.make_features(batch, 'src')
            tgt = onmt.IO.make_features(batch, 'tgt')
            report_stats.n_src_words += src_lengths.sum()

            # Truncated BPTT
            trunc_size = self.truncated_decoder if self.truncated_decoder \
                else target_size

            for j in range(0, target_size-1, trunc_size):
                # (1) Create truncated target.
                tgt = tgt[j: j + trunc_size]

                # (2) F-prop all but generator.

                # Main training loop
                self.model.zero_grad()
                outputs, attn, dec_state = \
                    self.model(src, tgt, src_lengths, dec_state)

                # (2) F-prop/B-prob generator in shards for memory
                # efficiency.
                batch_stats = onmt.Statistics()
                # make_loss_batch doesn't really need to be a method of
                # ComputeLoss
                gen_state = self.train_loss.make_loss_batch(
                        outputs, batch, attn, (j, j + trunc_size))
                shard_size = self.max_generator_batches
                for shard in onmt.Loss.shards(gen_state, shard_size):

                    # Compute loss and backprop shard.
                    loss, stats = self.train_loss.compute_loss(
                            batch=batch, **shard)
                    loss.div(batch.batch_size).backward()
                    batch_stats.update(stats)

                # (3) Update the parameters and statistics.
                self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

            if report_func is not None:
                report_func(epoch, i, len(self.train_iter),
                            total_stats.start_time, self.optim.lr,
                            report_stats)
                report_stats = onmt.Statistics()

        return total_stats

    def validate(self):
        """ Called for each epoch to validate. """
        # Set model in eval mode.
        self.model.eval()

        stats = onmt.Statistics()

        for batch in self.valid_iter:
            _, src_lengths = batch.src
            src = onmt.IO.make_features(batch, 'src')
            tgt = onmt.IO.make_features(batch, 'tgt')
            outputs, attn, _ = self.model(src, tgt, src_lengths)
            gen_state = self.valid_loss.make_loss_batch(
                outputs, batch, attn, (0, batch.tgt.size(0)))
            _, batch_stats = self.valid_loss.compute_loss(
                        batch=batch, **gen_state)
            stats.update(batch_stats)

        # Set model back to train mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        """ Called for each epoch to update learning rate. """
        return self.optim.updateLearningRate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, valid_stats):
        """ Called conditionally each epoch to save a snapshot. """
        model_state_dict = (self.model.module.state_dict()
                            if len(self.gpuid) > 1
                            else self.model.state_dict())
        # Exclude the 'generator' state.
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = (self.model.generator.module.state_dict()
                                if len(self.gpuid) > 1
                                else self.model.generator.state_dict())
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.IO.ONMTDataset.save_vocab(self.fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))
