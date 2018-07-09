#!/usr/bin/env python
import argparse
import torch


def average_models(model_files):
    vocab = None
    opt = None
    avg_model = None
    avg_generator = None

    for i, model_file in enumerate(model_files):
        m = torch.load(model_file)
        model_weights = m['model']
        generator_weights = m['generator']

        if i == 0:
            vocab, opt = m['vocab'], m['opt']
            avg_model = model_weights
            avg_generator = generator_weights
        else:
            for (k, v) in avg_model.items():
                avg_model[k].mul_(i).add_(model_weights[k]).div_(i + 1)

            for (k, v) in avg_generator.items():
                avg_generator[k].mul_(i).add_(generator_weights[k]).div_(i + 1)

    final = {"vocab": vocab, "opt": opt, "optim": None,
             "generator": avg_generator, "model": avg_model}
    return final


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-models", "-m", nargs="+", required=True,
                        help="List of models")
    parser.add_argument("-output", "-o", required=True,
                        help="Output file")
    opt = parser.parse_args()

    final = average_models(opt.models)
    torch.save(final, opt.output)


if __name__ == "__main__":
    main()
