import argparse
import json
import os
import time
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.utils.misc import use_gpu, set_random_seed


parser = argparse.ArgumentParser()
parser.add_argument(
    "-inference_config_file", help="Inference config file", required=True, type=str
)
parser.add_argument(
    "-inference_mode",
    help="Inference mode",
    required=True,
    type=str,
    choices=["py", "ct2"],
)
parser.add_argument(
    "-input_file",
    help="File with formatted input examples.",
    required=True,
    type=str,
)
parser.add_argument(
    "-output_directory",
    help="Output directory.",
    required=True,
    type=str,
)
args = parser.parse_args()
inference_config_file = args.inference_config_file
inference_mode = args.inference_mode
input_file = args.input_file
output_directory = args.output_directory


def _get_parser():
    parser = ArgumentParser(description="simple_inference_engine_py.py")
    opts.translate_opts(parser)
    opts.model_opts(parser)
    return parser


def evaluate(opt, output_filename):
    run_results = {}
    # Build the translator (along with the model)
    if inference_mode == "py":
        print("Inference with py ...")
        from onmt.inference_engine import InferenceEnginePY

        engine = InferenceEnginePY(opt)
    elif inference_mode == "ct2":
        print("Inference with ct2 ...")
        from onmt.inference_engine import InferenceEngineCT2

        engine = InferenceEngineCT2(opt)
    engine.opt.src = input_file
    start = time.time()
    scores, preds = engine.infer_file()
    engine.terminate()
    dur = time.time() - start
    print(f"Time to generate {len(preds)} answers: {dur}s")
    if inference_mode == "py":
        scores = [
            [_score.cpu().numpy().tolist() for _score in _scores] for _scores in scores
        ]
    run_results = {"pred_answers": preds, "score": scores, "duration": dur}
    with open(output_filename, "w") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)


def main():
    base_args = ["-config", inference_config_file]
    parser = _get_parser()
    opt = parser.parse_args(base_args)
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser._get_all_transform_translate(opt)
    ArgumentParser._validate_transforms_opts(opt)
    ArgumentParser.validate_translate_opts_dynamic(opt)
    set_random_seed(opt.seed, use_gpu(opt))
    base_name = os.path.basename(opt.models[0])
    print("# model: ", base_name)
    output_filename = os.path.join(output_directory, f"outputs_{base_name}_py.json")
    evaluate(opt, output_filename)


if __name__ == "__main__":
    main()
