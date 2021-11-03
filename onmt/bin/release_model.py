#!/usr/bin/env python
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Release an OpenNMT-py model for inference")
    parser.add_argument("--model", "-m",
                        help="The model path", required=True)
    parser.add_argument("--output", "-o",
                        help="The output path", required=True)
    parser.add_argument("--format",
                        choices=["pytorch", "ctranslate2"],
                        default="pytorch",
                        help="The format of the released model")
    parser.add_argument("--quantization", "-q",
                        choices=["int8", "int16", "float16", "int8_float16"],
                        default=None,
                        help="Quantization type for CT2 model.")
    opt = parser.parse_args()

    model = torch.load(opt.model, map_location=torch.device("cpu"))
    if opt.format == "pytorch":
        model["optim"] = None
        torch.save(model, opt.output)
    elif opt.format == "ctranslate2":
        import ctranslate2
        if not hasattr(ctranslate2, "__version__"):
            raise RuntimeError(
                "onmt_release_model script requires ctranslate2 >= 2.0.0"
            )
        converter = ctranslate2.converters.OpenNMTPyConverter(opt.model)
        converter.convert(opt.output, force=True,
                          quantization=opt.quantization)


if __name__ == "__main__":
    main()
