import torch
import argparse
from onmt.utils.logging import init_logger, logger
from onmt.models.model_saver import load_checkpoint
from onmt.inputters.inputter import dict_to_vocabs, vocabs_to_dict
from onmt.model_builder import build_base_model
from safetensors import safe_open
from safetensors.torch import save_file
import glob

"""
    This script merges or concat LoRa weights into the main model
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
        "--action",
        type=str,
        required=False,
        default="merge",
        choices=["merge", "concat"],
        help="""Path to the model directory""",
    )
    parser.add_argument(
        "--base_model", type=str, required=True, help="""Path to the base model"""
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="""Path to the lora checkpoint""",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="""Path to the output model"""
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pytorch", "safetensors"],
        default="pytorch",
        help="""Format to save the output model""",
    )

    opt = parser.parse_args()

    init_logger()

    base_checkpoint = load_checkpoint(opt.base_model)

    lora_checkpoint = load_checkpoint(opt.lora_weights)

    vocabs = dict_to_vocabs(lora_checkpoint["vocab"])

    lora_opt = lora_checkpoint["opt"]

    lora_opt.quant_layers = []  # we need to remove any quantization to merge weights
    lora_opt.parallel_mode = "data_parallel"

    model = build_base_model(lora_opt, vocabs)

    if "model" in base_checkpoint.keys():
        model.load_state_dict(
            base_checkpoint,
            precision=torch.float32,
            device=torch.device("cpu"),
            strict=False,
        )
    else:
        basepath = (
            opt.base_model[:-3] if opt.base_model[-3:] == ".pt" else opt.base_model
        )
        model.load_safe_state_dict(
            basepath,
            precision=torch.float32,
            device=torch.device("cpu"),
            strict=False,
        )

    if "model" in lora_checkpoint.keys():
        model.load_state_dict(
            lora_checkpoint,
            precision=torch.float32,
            device=torch.device("cpu"),
            strict=False,
        )
    else:
        lorapath = (
            opt.lora_weights[:-3]
            if opt.lora_weights[-3:] == ".pt"
            else opt.lora_weights
        )
        model.load_safe_state_dict(
            lorapath,
            precision=torch.float32,
            device=torch.device("cpu"),
            strict=False,
        )

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
        optim = lora_checkpoint["optim"]
        model_state_dict = model.state_dict()
        if opt.format == "pytorch":
            model_state_dict = {
                k: v for k, v in model_state_dict.items() if "generator" not in k
            }
        new_opt = lora_opt
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
    elif opt.format == "safetensors":
        new_checkpoint = {
            "vocab": vocabs_to_dict(vocabs),
            "opt": new_opt,
            "optim": optim,
        }
        if opt.output[-3:] == ".pt":
            torch.save(new_checkpoint, opt.output)
        else:
            torch.save(new_checkpoint, opt.output + ".pt")
        fileout = opt.output[:-3] if opt.output[-3:] == ".pt" else opt.output
        basepath = (
            opt.base_model[:-3] if opt.base_model[-3:] == ".pt" else opt.base_model
        )
        shards = glob.glob(basepath + ".*.safetensors")
        f = []
        for i, shard in enumerate(shards):
            shard_dict = {}
            f.append(safe_open(shard, framework="pt", device="cpu"))
            for key in f[i].keys():
                shard_dict[key] = model_state_dict[key]
            logger.info("saving shard" + fileout + ".{:02d}.safetensors".format(i))
            save_file(shard_dict, fileout + ".{:02d}.safetensors".format(i))
