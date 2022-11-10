import os
import torch
import numpy as np
import codecs
import onmt.opts as opts
from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser
from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter
from onmt.inputters.inputter import IterOnDevice
from onmt.utils.loss import LMLossCompute
from onmt.constants import DefaultTokens, CorpusTask
from onmt.transforms import get_transforms_cls, TransformPipe
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
    ppl_file = codecs.open(opt.output + ".ppl", "w+", "utf-8")

    device = (
        torch.device("cuda", opt.gpu)
        if opt.gpu > -1
        else torch.device("cpu")
    )

    vocabs, model, model_opt = load_test_model(opt)
    padding_idx = vocabs['tgt'][DefaultTokens.PAD]
    criterion = torch.nn.NLLLoss(ignore_index=padding_idx, reduction='none')
    loss_gen = model.generator
    valid_loss = LMLossCompute(criterion, loss_gen,
                               lambda_coverage=model_opt.lambda_coverage,
                               lambda_align=model_opt.lambda_align)
    valid_loss.to(device)

    transforms_cls = get_transforms_cls(opt._all_transform)

    infer_iter = build_dynamic_dataset_iter(
        opt, transforms_cls, vocabs, task=CorpusTask.INFER,
        copy=False)

    if infer_iter is not None:
        infer_iter = IterOnDevice(infer_iter, opt.gpu)

    data_transform = [
        infer_iter.transforms[name] for name in
        opt.transforms if name in infer_iter.transforms
    ]
    _ = TransformPipe.build_from(data_transform)

    model.to(device)
    model.eval()

    cumul_loss = 0.0
    cumul_length = 0
    # Now we can pipe the full file through the model using the Iterator

    for i, batch in enumerate(infer_iter):
        # reminder a batch includes .src .tgt .indices and it is sorted
        batch_size = len(batch['srclen'])
        src = batch['src']
        src_len = batch['srclen']
        tgt = batch['tgt']

        outputs, attns = model(src, tgt, src_len,
                               with_align=False)
        # Compute and retrieve the loss for EACH sentence
        lossflat, _ = valid_loss(batch, outputs, attns)
        loss = lossflat.view(batch_size, -1)
        mask = (loss != 0)
        sent_loss = torch.sum(loss, dim=1) / mask.sum(dim=1)
        sent_ppl = torch.exp(sent_loss)
        cumul_loss += loss.sum().item()
        cumul_length += mask.sum().cpu()
        # Now we need to rearrange the batch of ppl
        # in the original order with indices
        sent_ppl_orig = sent_ppl.gather(0, batch['indices'].argsort(0))
        for j in range(batch_size):
            ppl_file.write(str(sent_ppl_orig[j].item()) + "\n")
    logger.info("Loss: %.2f Tokens: %d Corpus PPL: %.2f" %
                (cumul_loss, cumul_length,
                 np.exp(cumul_loss / cumul_length)))
    ppl_file.close()

    os.system("paste " + opt.src + " " + opt.output + ".ppl > " + opt.output)


if __name__ == "__main__":
    main()
