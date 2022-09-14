import torch
import numpy as np
import codecs
import onmt.opts as opts
from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser
from onmt.utils.loss import LMLossCompute
from onmt.inputters import DynamicDataset, str2sortkey, OrderedIterator
from onmt.inputters.text_dataset import InferenceDataIterator, \
                                        InferenceDataReader
from onmt.transforms import make_transforms, get_transforms_cls, TransformPipe
from onmt.model_builder import load_test_model

"""
This script scores all sentences of a file using dynamic data.
For this purpose we use the same pipeline as the validation of a file
Below is an example of settings of a config.yaml file

model: lm-de.news2021_step_100000.pt
src: newstest2014-ref.de
tgt: newstest2014-ref.de
transforms: [onmt_tokenize]
batch_size: 16
gpu: 0
src_subword_type: bpe
src_subword_model: subwords.en_de.bpe
src_onmttok_kwargs: '{"mode": "aggressive"}'
tgt_subword_type: bpe
tgt_subword_model: subwords.en_de.bpe
tgt_onmttok_kwargs: '{"mode": "aggressive"}'

Output is the data and tab separated score
use the -output setting for preds + scores
Corpus PPL is in the logger.info
"""


def _get_parser():
    parser = ArgumentParser(description='LM_scoring.py')
    opts.config_opts(parser)
    opts.translate_opts(parser, dynamic=True)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser._get_all_transform_translate(opt)
    ArgumentParser._validate_transforms_opts(opt)
    ArgumentParser.validate_translate_opts_dynamic(opt)

    init_logger(opt.log_file)
    set_random_seed(opt.seed, False)
    out_file = codecs.open(opt.output, "w+", "utf-8")

    # Load model for inference
    fields, model, model_opt = load_test_model(opt)

    # Build transforms
    transforms_cls = get_transforms_cls(opt._all_transform)
    transforms = make_transforms(opt, transforms_cls, fields)
    data_transform = [
        transforms[name] for name in opt.transforms if name in transforms
    ]
    transform = TransformPipe.build_from(data_transform)

    device = (
        torch.device("cuda", opt.gpu)
        if opt.gpu > -1
        else torch.device("cpu")
    )
    model.to(device)
    model.eval()

    # Build datareader based on src AND tgt (should be equal)
    data_reader = InferenceDataReader(opt.src, opt.tgt, opt.src_feats)

    tgt_field = dict(fields)["tgt"].base_field
    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
    # Cannot use build_loss_compute() we need reduction 'none' in the criterion
    # to get the loss of each sentence instead of the loss of the full batch
    criterion = torch.nn.NLLLoss(ignore_index=padding_idx, reduction='none')
    loss_gen = model.generator
    valid_loss = LMLossCompute(criterion, loss_gen,
                               lambda_coverage=model_opt.lambda_coverage,
                               lambda_align=model_opt.lambda_align)
    valid_loss.to(device)

    cumul_loss = 0.0
    cumul_length = 0
    # Now we can pipe the full file through the model using the Iterator
    with torch.no_grad():
        for i, (src_shard, tgt_shard, feats_shard) in enumerate(data_reader):
            logger.info("Translating shard %d." % i)
            data_iter = InferenceDataIterator(src_shard, tgt_shard,
                                              feats_shard, transform)
            data = DynamicDataset(
                fields,
                data=data_iter,
                sort_key=str2sortkey[opt.data_type],
                filter_pred=None,
            )
            data_iter2 = OrderedIterator(
                dataset=data,
                device=device,
                batch_size=opt.batch_size,
                batch_size_fn=None,
                train=False,
                sort=False,
                sort_within_batch=True,
                shuffle=False,
            )
            for i, batch in enumerate(data_iter2):
                # reminder a batch includes .src .tgt .indices and it is sorted
                batch_size = len(batch)
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                    else (batch.src, None)
                tgt = batch.tgt
                outputs, attns = model(src, tgt, src_lengths,
                                       with_align=False)
                # Compute and retrieve the loss for EACH sentence
                lossflat, _ = valid_loss(batch, outputs, attns)
                loss = lossflat.view(-1, batch_size)
                mask = (loss != 0)
                sent_loss = torch.sum(loss, dim=0) / mask.sum(dim=0)
                sent_ppl = torch.exp(sent_loss)
                cumul_loss += loss.sum().item()
                cumul_length += mask.sum().cpu()
                # Now we need to rearrange the batch of ppl
                # in the original order with indices
                sent_ppl_orig = sent_ppl.gather(0, batch.indices.argsort(0))
                for j in range(batch_size):
                    srctxt = src_shard[i * opt.batch_size + j]
                    out_file.write(srctxt.strip().decode("UTF-8") +
                                   "\t" + str(sent_ppl_orig[j].item()) + "\n")
        logger.info("Loss: %.2f Tokens: %d Corpus PPL: %.2f" %
                    (cumul_loss, cumul_length,
                     np.exp(cumul_loss / cumul_length)))
        out_file.close()


if __name__ == "__main__":
    main()
