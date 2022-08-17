import torch
import numpy as np
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

"""This script computes the ppl of a file using dynamic data."""
"""For this purpose we use the same pipeline as the validation of a file"""

# Set sharing strategy manually instead of default based on the OS.
torch.multiprocessing.set_sharing_strategy('file_system')


def _get_parser():
    parser = ArgumentParser(description='corpus_ppl.py')
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

    # We could use build_loss_compute without reduction 'none' in the criterion
    # to get the loss of each sentence instead of the loss of the full batch
    # However this is more in line with the other script score_sentences.py
    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
    criterion = torch.nn.NLLLoss(ignore_index=padding_idx, reduction='none')
    loss_gen = model.generator
    valid_loss = LMLossCompute(criterion, loss_gen,
                               lambda_coverage=model_opt.lambda_coverage,
                               lambda_align=model_opt.lambda_align)
    valid_loss.to(device)

    cumul_loss = 0.0
    cumul_length = 0

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
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                    else (batch.src, None)
                tgt = batch.tgt
                outputs, attns = model(src, tgt, src_lengths,
                                       with_align=False)
                # Compute loss.
                loss, stats = valid_loss(batch, outputs, attns)
                batch_loss = loss.sum().item()
                batch_length = src_lengths.sum().cpu().numpy()
                cumul_loss += batch_loss
                cumul_length += batch_length
        avg_loss = cumul_loss / cumul_length
        print("loss: ", cumul_loss, " length: ", cumul_length, " avg loss ",
              avg_loss, "Corpus PPL: ", np.exp(avg_loss))


if __name__ == "__main__":
    main()
