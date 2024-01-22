import copy
import numpy as np
import time
from onmt.inference_engine import InferenceEnginePY
import onmt.opts as opts
from onmt.utils.logging import init_logger
from onmt.utils.parse import ArgumentParser
from onmt.utils.misc import use_gpu, set_random_seed


def tokenize_dataset(opt, context_length):
    print("Tokenization...")
    # Clean and Concat the dataset
    x = open(opt.src, "r").readlines()
    xx = [_x for _x in x if _x != " \n"]
    from onmt.transforms.tokenize import SentencePieceTransform

    tokenizer = SentencePieceTransform(opt)
    tokenizer.warm_up()
    tokens = tokenizer._tokenize(xx)
    print("Done !")
    return tokens


def evaluate(opt):
    """Score the wikitext2 testset"""
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser._get_all_transform_translate(opt)
    ArgumentParser._validate_transforms_opts(opt)
    ArgumentParser.validate_translate_opts_dynamic(opt)
    logger = init_logger(opt.log_file)
    set_random_seed(opt.seed, use_gpu(opt))

    # Tokenize the dataset.
    opt.src = "wikitext-2-raw-v1/wikitext-2-raw/wiki.test.raw"
    tokens = tokenize_dataset(opt, context_length=512)

    # Build the translator (along with the model.
    engine_opt = copy.copy(opt)
    engine_opt._all_transform = []
    engine = InferenceEnginePY(engine_opt)

    # Score the dataset.
    stride = 512
    max_seq_length = 2048
    engine_opt.batch_type = "sents"
    engine_opt.batch_size = 1
    seq_len = len(tokens)
    score_results = []
    nlls = []
    src = []
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_seq_length, seq_len)
        src.append(" ".join(tokens[begin_loc:end_loc]))
    start_time = time.time()
    score_results = engine.score_list(src=src)
    nlls = [_score for (_score, _length) in score_results]
    lengths = [_length for (_score, _length) in score_results]
    ppl = np.exp(-np.sum(nlls) / np.sum(lengths))
    engine.terminate()
    end_time = time.time()
    logger.info("total run time %.2f" % (end_time - start_time))
    logger.info(
        "wikitext-2 perplexity with rolling likelihood and sliding window size 1000 and stride 512 %.2f"  # noqa: E501
        % (ppl)
    )


def _get_parser():
    parser = ArgumentParser(description="run_wikitext-2_benchmark.py")
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    evaluate(opt)


if __name__ == "__main__":
    main()
