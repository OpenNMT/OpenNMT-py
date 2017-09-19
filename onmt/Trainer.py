import onmt
import onmt.modules


class Trainer(object):
    def __init__(self, model, train_data, valid_data, fields, optim,
                 batch_size, gpuid, copy_attn, copy_attn_force,
                 truncated_decoder, max_generator_batches,
                 report_every, exp_host, experiment):
        # Basic attributes.
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.fields = fields
        self.optim = optim
        self.batch_size = batch_size
        self.copy_attn = copy_attn
        self.truncated_decoder = truncated_decoder
        self.max_generator_batches = max_generator_batches
        self.report_every = report_every
        self.exp_host = exp_host
        self.experiment = experiment

        # Define criterion.
        padding_idx = fields['tgt'].vocab.stoi[onmt.IO.PAD_WORD]
        if not copy_attn:
            self.criterion = onmt.Loss.nmt_criterion(
                len(fields['tgt'].vocab), gpuid, padding_idx)
        else:
            self.criterion = onmt.modules.CopyCriterion(
                len(fields['tgt'].vocab), copy_attn_force, padding_idx)

        # Create a train data iterator.
        self.train_iterator = onmt.IO.OrderedIterator(
            dataset=train_data, batch_size=batch_size,
            device=gpuid[0] if gpuid else -1,
            repeat=False)

        # Create a validate data iterator.
        self.valid_iterator = onmt.IO.OrderedIterator(
            dataset=valid_data, batch_size=batch_size,
            device=gpuid[0] if gpuid else -1,
            train=False, sort=True)

    def train(self, epoch):
        """ Called for each epoch to train. """
        closs = onmt.Loss.LossCompute(self.model.generator, self.criterion,
                                      self.fields["tgt"].vocab,
                                      self.train_data, epoch,
                                      self.copy_attn)

        total_stats = onmt.Loss.Statistics()
        report_stats = onmt.Loss.Statistics()

        for i, batch in enumerate(self.train_iterator):
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
                batch_stats = onmt.Loss.Statistics()
                # make_loss_batch doesn't really need to be a method of
                # ComputeLoss
                gen_state = closs.make_loss_batch(outputs, batch, attn,
                                                  (j, j + trunc_size))
                shard_size = self.max_generator_batches
                for shard in onmt.Loss.shards(gen_state, shard_size):

                    # Compute loss and backprop shard.
                    loss, stats = closs.compute_loss(batch=batch,
                                                     **shard)
                    loss.div(batch.batch_size).backward()
                    batch_stats.update(stats)

                # (3) Update the parameters and statistics.
                self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

            if i % self.report_every == -1 % self.report_every:
                report_stats.output(epoch, i+1, len(self.train_iterator),
                                    total_stats.start_time)
                if self.exp_host:
                    report_stats.log("progress", self.experiment, self.optim)
                report_stats = onmt.Loss.Statistics()
        return total_stats

    def validate(self):
        """ Called for each epoch to validate. """
        # Set model in eval mode.
        self.model.eval()

        loss = onmt.Loss.LossCompute(self.model.generator, self.criterion,
                                     self.fields["tgt"].vocab,
                                     self.valid_data, 0,
                                     self.copy_attn)
        stats = onmt.Loss.Statistics()

        for batch in self.valid_iterator:
            _, src_lengths = batch.src
            src = onmt.IO.make_features(batch, 'src')
            tgt = onmt.IO.make_features(batch, 'tgt')
            outputs, attn, _ = self.model(src, tgt, src_lengths)
            gen_state = loss.make_loss_batch(
                outputs, batch, attn, (0, batch.tgt.size(0)))
            _, batch_stats = loss.compute_loss(batch=batch, **gen_state)
            stats.update(batch_stats)

        # Set model back to train mode.
        self.model.train()

        return stats
