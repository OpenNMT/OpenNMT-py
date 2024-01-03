import os
import torch
import numpy as np
import codecs
import onmt.opts as opts
from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser
from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter
from onmt.utils.loss import LossCompute
from onmt.constants import DefaultTokens, CorpusTask
from onmt.transforms import get_transforms_cls
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
    parser = ArgumentParser(description="LM_scoring.py")
    opts.translate_opts(parser)
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

    device = torch.device("cuda", opt.gpu) if opt.gpu > -1 else torch.device("cpu")

    vocabs, model, model_opt = load_test_model(opt)
    padding_idx = vocabs["tgt"][DefaultTokens.PAD]
    criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_idx, reduction="none")
    valid_loss = LossCompute(
        criterion,
        model.generator,
        tgt_shift_index=0,
        lambda_coverage=model_opt.lambda_coverage,
        lambda_align=model_opt.lambda_align,
    )
    valid_loss.to(device)

    transforms_cls = get_transforms_cls(opt._all_transform)
    infer_iter = build_dynamic_dataset_iter(
        opt,
        transforms_cls,
        vocabs,
        task=CorpusTask.INFER,
        copy=False,
        device_id=opt.gpu,
    )

    model.to(device)
    model.eval()

    cumul_loss = 0.0
    cumul_length = 0
    # Now we can pipe the full file through the model using the Iterator

    for i, (batch, bucket_idx) in enumerate(infer_iter):
        # reminder a batch includes .src .tgt .indices and it is sorted
        batch_size = len(batch["srclen"])
        src = batch["src"]
        src_len = batch["srclen"]
        # print(batch)
        outputs, attns = model(src, None, src_len, with_align=False)
        # Compute and retrieve the loss for EACH sentence
        loss, _ = valid_loss(batch, outputs, attns)
        loss = loss.view(batch_size, -1)  # (B, T)
        losspertoken = loss.sum(1) / batch["tgt"][:, 1:, 0].ne(padding_idx).sum(1)
        ppl = torch.exp(losspertoken)
        cumul_loss += loss.sum().item()
        cumul_length += batch["tgt"][:, 1:, 0].ne(padding_idx).sum().cpu()
        # Now we need to rearrange the batch of ppl
        # in the original order with indices
        sent_ppl_orig = ppl.gather(
            0,
            torch.tensor(
                sorted(
                    range(len(batch["cid_line_number"])),
                    key=lambda k: batch["cid_line_number"][k],
                ),
                device=ppl.device,
            ),
        )
        for j in range(batch_size):
            ppl_file.write(str(sent_ppl_orig[j].item()) + "\n")
    logger.info(
        "Loss: %.2f Tokens: %d Corpus PPL: %.2f"
        % (cumul_loss, cumul_length, np.exp(cumul_loss / cumul_length))
    )
    ppl_file.close()

    os.system('paste "' + opt.src + '" "' + opt.output + '".ppl > "' + opt.output + '"')


if __name__ == "__main__":
    main()
