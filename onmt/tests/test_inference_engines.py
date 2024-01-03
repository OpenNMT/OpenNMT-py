import json
import time
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.utils.misc import use_gpu, set_random_seed


def _get_parser():
    parser = ArgumentParser(description="simple_inference_engine_py.py")
    opts.translate_opts(parser)
    return parser


def evaluate(opt, inference_mode, input_file, out, method):
    print("# input file", input_file)
    run_results = {}
    # Build the translator (along with the model)
    if inference_mode == "py":
        print("Inference with py ...")
        from onmt.inference_engine import InferenceEnginePY

        engine = InferenceEnginePY(opt)
    elif inference_mode == "ct2":
        print("Inference with ct2 ...")
        from onmt.inference_engine import InferenceEngineCT2

        opt.src_subword_vocab = opt.models[0] + "/vocabulary.json"
        engine = InferenceEngineCT2(opt)
    start = time.time()
    if method == "file":
        engine.opt.src = input_file
        scores, preds = engine.infer_file()
    elif method == "list":
        src = open(input_file, ("r")).readlines()
        scores, preds = engine.infer_list(src)
    engine.terminate()
    dur = time.time() - start
    print(f"Time to generate {len(preds)} answers: {dur}s")
    if inference_mode == "py":
        scores = [
            [_score.cpu().numpy().tolist() for _score in _scores] for _scores in scores
        ]
    run_results = {"pred_answers": preds, "score": scores, "duration": dur}
    output_filename = out + f"_{method}.json"
    with open(output_filename, "w") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)


def main():
    # Required arguments
    parser = ArgumentParser()
    parser.add_argument("-model", help="Path to model.", required=True, type=str)
    parser.add_argument(
        "-model_task",
        help="Model task.",
        required=True,
        type=str,
        choices=["lm", "seq2seq"],
    )
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
        "-out",
        help="Output filename.",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    model = args.model
    inference_config_file = args.inference_config_file
    base_args = ["-config", inference_config_file]
    parser = _get_parser()
    opt = parser.parse_args(base_args)
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser._get_all_transform_translate(opt)
    ArgumentParser._validate_transforms_opts(opt)
    ArgumentParser.validate_translate_opts_dynamic(opt)
    set_random_seed(opt.seed, use_gpu(opt))
    opt.models = [model]
    opt.model_task = args.model_task

    evaluate(
        opt,
        inference_mode=args.inference_mode,
        input_file=args.input_file,
        out=args.out,
        method="file",
    )
    evaluate(
        opt,
        inference_mode=args.inference_mode,
        input_file=args.input_file,
        out=args.out,
        method="list",
    )


if __name__ == "__main__":
    main()
