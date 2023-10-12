import json
import os
import time

import onmt.opts as opts
from onmt.inference_engine import InferenceEnginePY
from onmt.utils.parse import ArgumentParser
from onmt.utils.misc import use_gpu, set_random_seed

OUTDIR = "outputs/"
DATASET = "input_examples.txt"


def _get_parser():
    parser = ArgumentParser(description="simple_inference_engine_py.py")
    opts.config_opts(parser)
    opts.translate_opts(parser, dynamic=True)
    opts.model_opts(parser)
    return parser


def evaluate(opt, output_filename):
    run_results = {}
    # Build the translator (along with the model)
    engine = InferenceEnginePY(opt)
    engine.opt.src = DATASET
    print(os.path.exists(engine.opt.src))
    start = time.time()
    scores, preds = engine.infer_file()
    engine.terminate()
    scores = [_score.cpu().numpy().tolist() for _score in sum(scores, [])]
    dur = time.time() - start
    print(f"Time to generate {len(preds)} answers: {dur}s")
    run_results = {"pred_answers": preds, "score": scores, "duration": dur}
    with open(output_filename, "w") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser._get_all_transform_translate(opt)
    ArgumentParser._validate_transforms_opts(opt)
    ArgumentParser.validate_translate_opts_dynamic(opt)
    set_random_seed(opt.seed, use_gpu(opt))
    base_name = os.path.basename(opt.models[0])
    print("# model: ", base_name)
    output_filename = os.path.join(OUTDIR, f"outputs_{base_name}_py.json")
    evaluate(opt, output_filename)


if __name__ == "__main__":
    main()

# python3 simple_inference_py.py -config translate_opts_py.yaml
