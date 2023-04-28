import torch
import argparse
from onmt.utils.logging import init_logger
from onmt.inputters.inputter import dict_to_vocabs, vocabs_to_dict
from onmt.model_builder import build_base_model
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
    parser.add_argument('--action', type=str, required=False,
                        default='merge', choices=['merge', 'concat'],
                        help="""Path to the model directory""")
    parser.add_argument('--base_model', type=str, required=True,
                        help="""Path to the base model""")
    parser.add_argument('--lora_weights', type=str, required=True,
                        help="""Path to the lora checkpoint""")
    parser.add_argument('--output', type=str, required=True,
                        help="""Path to the output model""")
    opt = parser.parse_args()

    init_logger()

    base_checkpoint = torch.load(opt.base_model,
                                 map_location=torch.device('cpu'))

    lora_checkpoint = torch.load(opt.lora_weights,
                                 map_location=torch.device('cpu'))

    vocabs = dict_to_vocabs(lora_checkpoint['vocab'])

    lora_opt = lora_checkpoint['opt']

    lora_opt.quant_layers = []

    model = build_base_model(lora_opt, vocabs, base_checkpoint)

    model.load_state_dict(lora_checkpoint['model'], strict=False)

    if opt.action == 'merge':
        model.eval()  # this merges automatically LoRa weights in main
        model.half()  # We keep FP16 for all
        optim = None
        model_state_dict = model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k and 'lora' not in k}
        new_opt = base_checkpoint['opt']
    elif opt.action == 'concat':
        model.half()  # We keep FP16 for all
        optim = lora_checkpoint['optim']
        model_state_dict = model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        new_opt = lora_opt
    else:
        raise ValueError(
            "action not supported, please choose merge or concat")

    generator_state_dict = model.generator.state_dict()

    new_checkpoint = {
        'model': model_state_dict,
        'generator': generator_state_dict,
        'vocab': vocabs_to_dict(vocabs),
        'opt': new_opt,
        'optim': optim,
        }
    torch.save(new_checkpoint, opt.output)
