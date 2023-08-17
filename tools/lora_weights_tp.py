import argparse
import torch

from onmt.utils.logging import init_logger, logger
from onmt.models.model_saver import load_checkpoint
from onmt.inputters.inputter import dict_to_vocabs, vocabs_to_dict
from onmt.model_builder import build_base_model


"""
    This script merges or concat LoRa weights from partial checkpoints
    saved during a finetuning with the tensor parallel mode, into the main model
    * merge
        the weights of LoRa are merged and we save the merged model without
        the optimizer hence in the same format as the original base model.
    * concat
        the weights of LoRa are added to the base model in the way the model
        can be further trained as it was stopped. The Optimizer is also
        restored from the LoRa checkpoint
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format",
        type=str,
        choices=["pytorch"],
        default="pytorch",
        help="""Format to save the output model""",
    )
    parser.add_argument(
        "--base_model", type=str, required=True, help="""Path to the base model"""
    )
    parser.add_argument(
        "--lora_model",
        type=str,
        required=True,
        help="""Path to the partial lora checkpoints""",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="""step""",
    )
    parser.add_argument(
        "--nb_checkpoints",
        type=int,
        required=True,
        help="""Number of partial checkpoints""",
    )
    parser.add_argument(
        "--action",
        type=str,
        required=False,
        default="merge",
        choices=["merge", "concat"],
        help="""Path to the model directory""",
    )
    opt = parser.parse_args()

    init_logger()

    finetuned_models = [
        (f"{opt.lora_model}_device_{device_id}_step_{opt.step}.pt")
        for device_id in range(opt.nb_checkpoints)
    ]
    opt.output = f"{opt.lora_model}_step_{opt.step}_{opt.action}.pt"


def main():
    base_checkpoint = load_checkpoint(opt.base_model)
    for i in range(opt.nb_checkpoints):
        device_id = i
        partial_checkpoint = load_checkpoint(finetuned_models[i])
        partial_opt = partial_checkpoint["opt"]
        partial_opt.quant_layers = (
            []
        )  # we need to remove any quantization to merge weights
        partial_opt.world_size = 1
        partial_opt.gpu_ranks = [0]
        with open("partial_opt", "w") as w:
            w.write(str(partial_opt))
        if i == 0:
            print("## Loading base model")
            vocabs = dict_to_vocabs(partial_checkpoint["vocab"])
            model = build_base_model(partial_opt, vocabs)
            if "model" in base_checkpoint.keys():
                model.load_state_dict(
                    base_checkpoint,
                    precision=torch.float32,
                    device=torch.device("cpu"),
                    strict=False,
                )
            else:
                basepath = (
                    opt.base_model[:-3]
                    if opt.base_model[-3:] == ".pt"
                    else opt.base_model
                )
                model.load_safe_state_dict(
                    basepath,
                    precision=torch.float32,
                    device=torch.device("cpu"),
                    strict=False,
                )
            print("done !")

        if "model" in partial_checkpoint.keys():
            print("## Loading checkpoint ", device_id)
            model.load_state_dict_tp(
                partial_checkpoint,
                precision=torch.float32,
                device=torch.device("cpu"),
                strict=False,
                device_id=device_id,
            )
            print("done !")

    if opt.action == "merge":
        model.eval()  # this merges automatically LoRa weights in main
        model.half()  # We keep FP16 for all
        optim = None
        model_state_dict = model.state_dict()
        if opt.format == "pytorch":
            model_state_dict = {
                k: v
                for k, v in model_state_dict.items()
                if "generator" not in k and "lora" not in k
            }
        new_opt = base_checkpoint["opt"]
    elif opt.action == "concat":
        model.half()  # We keep FP16 for all
        optim = partial_checkpoint["optim"]
        model_state_dict = model.state_dict()
        if opt.format == "pytorch":
            model_state_dict = {
                k: v for k, v in model_state_dict.items() if "generator" not in k
            }
        new_opt = partial_opt
    else:
        raise ValueError("action not supported, please choose merge or concat")
    if opt.format == "pytorch":
        generator_state_dict = model.generator.state_dict()
        new_checkpoint = {
            "model": model_state_dict,
            "generator": generator_state_dict,
            "vocab": vocabs_to_dict(vocabs),
            "opt": new_opt,
            "optim": optim,
        }
        logger.info("Saving merged model")
        if opt.output[-3:] == ".pt":
            torch.save(new_checkpoint, opt.output)
        else:
            torch.save(new_checkpoint, opt.output + ".pt")


if __name__ == "__main__":
    main()
