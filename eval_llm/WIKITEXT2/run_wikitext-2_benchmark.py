import copy
import json
import numpy as np
import os
import sentencepiece as spm
import time
from onmt.constants import DefaultTokens
from onmt.inference_engine import InferenceEnginePY
import onmt.opts as opts
from onmt.utils.logging import init_logger
from onmt.utils.parse import ArgumentParser
from onmt.utils.misc import use_gpu, set_random_seed


def compute_file_ppl(output_filename):
    with open(output_filename, "r") as f:
        run_results = json.load(f)
    nlls = []
    lengths = []
    for i, _res in enumerate(run_results["scored_results"]):
        print(_res)
        nlls.append(_res[0])
        lengths.append(_res[1])
    file_ppl = np.exp(-np.sum(nlls) / np.sum(lengths))
    print("wikitext-2 ppl: %.4f" % file_ppl)


def tokenize_dataset(opt, context_length):
    print("Tokenization...")

    # Concat the dataset
    x = open(opt.src, "r").readlines()
    x = [_x.rstrip("\n") for _x in x]
    y = DefaultTokens.SEP.join(x)

    with open(opt.src + ".temp", "w") as writer:
        writer.write(y)

    # Tokenize
    SP = spm.SentencePieceProcessor(opt.src_subword_model)
    x = open(opt.src + ".temp", "r").readlines()
    tokens = SP.encode(x[0], out_type=str)


    def make_chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    with open(opt.src + f".tokenized.context_{context_length}", "w") as writer:
        for _chunk in make_chunks(tokens, context_length - 1):
            writer.write(" ".join(_chunk) + "\n")
            print(len(_chunk))

    print("Done !")

    z = open(opt.src + f".tokenized.context_{context_length}", "r").readlines()
    print(len(z[0].split(" ")))


def evaluate(opt):
    """Score the wikitext2 testset"""
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser._get_all_transform_translate(opt)
    ArgumentParser._validate_transforms_opts(opt)
    ArgumentParser.validate_translate_opts_dynamic(opt)
    logger = init_logger(opt.log_file)
    set_random_seed(opt.seed, use_gpu(opt))

    run_results = {}
    dir_name = os.path.dirname(opt.models[0])
    base_name = os.path.basename(opt.models[0])

    output_filename = os.path.join(
        dir_name, "wikitext-2_benchmark_%s.json" % base_name[:-3]
    )

    # Build the translator (along with the model.
    engine_opt = copy.copy(opt)
    engine_opt._all_transform = []
    engine = InferenceEnginePY(engine_opt)

    # Tokenize the dataset.
    opt.src = "wikitext-2-raw-v1/wikitext-2-raw/wiki.test.raw"
    tokenize_dataset(opt, context_length=512)

    # Score the tokeznized dataset
    engine.opt.src = opt.src + f".tokenized.context_{512}"
    start_time = time.time()
    scored_results = engine.score_file()
    engine.terminate()
    run_results["scored_results"] = scored_results

    with open(output_filename, "w") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    compute_file_ppl(output_filename)

    end_time = time.time()
    logger.info("total run time %.2f" % (end_time - start_time))


def _get_parser():
    parser = ArgumentParser(description="run_wikitext-2_benchmark.py")
    opts.config_opts(parser)
    opts.translate_opts(parser, dynamic=True)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    evaluate(opt)


if __name__ == "__main__":
    main()
