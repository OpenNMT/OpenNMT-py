#!/usr/bin/env python
""" Classifier Class and builder """
from __future__ import print_function
import codecs
import time

import torch
import torchtext.data
import onmt.model_builder
import onmt.inputters as inputters
from onmt.inputters.inputter import max_tok_len
from onmt.utils.misc import set_random_seed


def build_classifier(opt, logger=None, out_file=None):
    """Return a classifier with result redirect to `out_file`."""

    if out_file is None:
        out_file = codecs.open(opt.output, 'w', 'utf-8')

    load_model = onmt.model_builder.load_test_model
    fields, model, model_opt = load_model(opt, opt.model)

    classifier = Classifier.from_opt(
        model,
        fields,
        opt,
        model_opt,
        out_file=out_file,
        logger=logger
    )
    return classifier


def build_tagger(opt, logger=None, out_file=None):
    """Return a tagger with result redirect to `out_file`."""

    if out_file is None:
        out_file = codecs.open(opt.output, 'w', 'utf-8')

    load_model = onmt.model_builder.load_test_model
    fields, model, model_opt = load_model(opt, opt.model)

    tagger = Tagger.from_opt(
        model,
        fields,
        opt,
        model_opt,
        out_file=out_file,
        logger=logger
    )
    return tagger


class Predictor(object):
    """Predictor a batch of data with a saved model.

    Args:
        model (nn.Sequential): model to use
        fields (dict[str, torchtext.data.Field]): A dict of field.
        gpu (int): GPU device. Set to negative for no GPU.
        data_type (str): Source data type.
        verbose (bool): output every predition with confidences.
        report_time (bool): Print/log total time/frequency.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
            self,
            model,
            fields,
            gpu=-1,
            verbose=False,
            out_file=None,
            report_time=True,
            logger=None,
            seed=-1):
        self.model = model
        self.fields = fields

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = torch.device("cuda", self._gpu) \
            if self._use_cuda else torch.device("cpu")

        self.verbose = verbose
        self.report_time = report_time
        self.out_file = out_file
        self.logger = logger

        set_random_seed(seed, self._use_cuda)

    @classmethod
    def from_opt(
            cls,
            model,
            fields,
            opt,
            model_opt,
            out_file=None,
            logger=None):
        """Alternate constructor.

        Args:
            model (onmt.modules): See :func:`__init__()`.
            fields (dict[str, torchtext.data.Field]): See
                :func:`__init__()`.
            opt (argparse.Namespace): Command line options
            model_opt (argparse.Namespace): Command line options saved with
                the model checkpoint.
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """

        return cls(
            model,
            fields,
            gpu=opt.gpu,
            verbose=opt.verbose,
            out_file=out_file,
            logger=logger,
            seed=opt.seed)

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)


class Classifier(Predictor):
    """classify a batch of sentences with a saved model.

    Args:
        model (nn.Sequential): BERT model to use for classify
        fields (dict[str, torchtext.data.Field]): A dict of field.
        gpu (int): GPU device. Set to negative for no GPU.
        data_type (str): Source data type.
        verbose (bool): output every predition with confidences.
        report_time (bool): Print/log total time/frequency.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
            self,
            model,
            fields,
            gpu=-1,
            verbose=False,
            out_file=None,
            report_time=True,
            logger=None,
            seed=-1):
        super(Classifier, self).__init__(
            model,
            fields,
            gpu=gpu,
            verbose=verbose,
            out_file=out_file,
            report_time=report_time,
            logger=logger,
            seed=seed)
        label_field = self.fields["category"]
        self.label_vocab = label_field.vocab

    def classify(self, data, batch_size, tokenizer, delimiter=' ||| ',
                 max_seq_len=256, batch_type="sents"):
        """Classify content of ``data``.

        Args:
            data (list of str): ['Sentence1 ||| Sentence2',...].
            batch_size (int): size of examples per mini-batch
            batch_type (str): Batch grouping for batch_size. Chose from
                {'sents', 'tokens'}, default batch_size count by sentence.

        Returns:
            all_predictions (list of str):[c1, ..., cn].
        """

        dataset = inputters.ClassifierDataset(
            self.fields, data, tokenizer, max_seq_len, delimiter)

        data_iter = torchtext.data.Iterator(
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=max_tok_len if batch_type == "tokens" else None,
            device=self._dev,
            train=False,
            sort=False,
            sort_within_batch=False,
            shuffle=False
        )

        all_predictions = []

        start_time = time.time()

        for batch in data_iter:
            pred_sents_labels = self.classify_batch(batch)
            all_predictions.extend(pred_sents_labels)
            self.out_file.write('\n'.join(pred_sents_labels) + '\n')
            self.out_file.flush()

        end_time = time.time()

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total classification time: %f s" % total_time)
            self._log("Average classification time: %f s" % (
                total_time / len(all_predictions)))
            self._log("Sentences per second: %f" % (
                len(all_predictions) / total_time))
        return all_predictions

    def classify_batch(self, batch):
        """Classify a batch of sentences."""
        with torch.no_grad():
            input_ids, _ = batch.tokens
            token_type_ids = batch.segment_ids
            all_encoder_layers, pooled_out = self.model(
                input_ids, token_type_ids)
            seq_class_log_prob, _ = self.model.generator(
                all_encoder_layers, pooled_out)
            # Predicting
            pred_sents_ids = seq_class_log_prob.argmax(-1).tolist()
            pred_sents_labels = [self.label_vocab.itos[index]
                                 for index in pred_sents_ids]
            if self.verbose:
                seq_class_prob = seq_class_log_prob.exp()
                category_probs = seq_class_prob.tolist()
                preds = ['\t'.join(map(str, category_prob)) + '\t' + pred
                         for category_prob, pred in zip(
                         category_probs, pred_sents_labels)]
                return preds
            return pred_sents_labels


class Tagger(Predictor):
    """Tagging a batch of sentences with a saved model.

    Args:
        model (nn.Sequential): BERT model to use for Tagging
        fields (dict[str, torchtext.data.Field]): A dict of field.
        gpu (int): GPU device. Set to negative for no GPU.
        data_type (str): Source data type.
        verbose (bool): output every predition with confidences.
        report_time (bool): Print/log total time/frequency.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
            self,
            model,
            fields,
            gpu=-1,
            verbose=False,
            out_file=None,
            report_time=True,
            logger=None,
            seed=-1):
        super(Tagger, self).__init__(
            model,
            fields,
            gpu=gpu,
            verbose=verbose,
            out_file=out_file,
            report_time=report_time,
            logger=logger,
            seed=seed)
        label_field = self.fields["token_labels"]
        self.label_vocab = label_field.vocab
        self.pad_token = label_field.pad_token
        self.pad_index = self.label_vocab.stoi[self.pad_token]

    def tagging(self, data, batch_size, tokenizer, delimiter=' ',
                max_seq_len=256, batch_type="sents"):
        """Tagging content of ``data``.

        Args:
            data (list of str): ['T1 T2 ... Tn',...].
            batch_size (int): size of examples per mini-batch

        Returns:
            all_predictions (list of list of str): [['L1', ..., 'Ln'],...].
        """
        dataset = inputters.TaggerDataset(
            self.fields, data, tokenizer, max_seq_len, delimiter)

        data_iter = torchtext.data.Iterator(
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=max_tok_len if batch_type == "tokens" else None,
            device=self._dev,
            train=False,
            sort=False,
            sort_within_batch=False,
            shuffle=False
        )

        all_predictions = []

        start_time = time.time()

        for batch in data_iter:
            pred_tokens_tag = self.tagging_batch(batch)
            all_predictions.extend(pred_tokens_tag)
            for pred_sent in pred_tokens_tag:
                self.out_file.write('\n'.join(pred_sent) + '\n' + '\n')
            self.out_file.flush()

        end_time = time.time()

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total tagging time (s): %f" % total_time)
            self._log("Average tagging time (s): %f" % (
                total_time / len(all_predictions)))
            self._log("Sentence per second: %f" % (
                len(all_predictions) / total_time))
        return all_predictions

    def tagging_batch(self, batch):
        """Tagging a batch of sentences."""
        with torch.no_grad():
            # Batch
            input_ids, _ = batch.tokens
            token_type_ids = batch.segment_ids
            taggings = batch.token_labels
            # Forward
            all_encoder_layers, pooled_out = self.model(
                input_ids, token_type_ids)
            _, prediction_log_prob = self.model.generator(
                all_encoder_layers, pooled_out)
            # Predicting
            pred_tag_ids = prediction_log_prob.argmax(-1)
            non_padding = taggings.ne(self.pad_index)
            batch_tag_ids, batch_mask = list(pred_tag_ids), list(non_padding)
            batch_tag_select_ids = [pred.masked_select(mask).tolist()
                                    for pred, mask in
                                    zip(batch_tag_ids, batch_mask)]

            pred_tokens_tag = [[self.label_vocab.itos[index]
                                for index in tag_select_ids]
                               for tag_select_ids in batch_tag_select_ids]
            return pred_tokens_tag
