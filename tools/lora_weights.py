import torch
import torch.nn as nn
import argparse
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import dict_to_vocabs, vocabs_to_dict
from onmt.model_builder import build_base_model
from onmt.modules import Linear

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

    base_checkpoint = torch.load(opt.base_model,
                                 map_location=torch.device('cpu'))
    model_opt = ArgumentParser.ckpt_model_opts(base_checkpoint['opt'])

    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocabs = dict_to_vocabs(base_checkpoint['vocab'])

    model_opt.update_vocab = False

    lora_checkpoint = torch.load(opt.lora_weights,
                                 map_location=torch.device('cpu'))

    layers = lora_checkpoint['model'].keys()

    model = build_base_model(model_opt, vocabs, base_checkpoint)

    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in layers:
            model._modules[name] = Linear(
                module.in_features,
                module.out_features,
                r=1,
                lora_alpha=1,
                lora_dropout=0,
                bias=False)

    model.half()  # We keep FP16 for all
    model.load_state_dict(lora_checkpoint['model'], strict=False)

    if opt.action == 'merge':
        model.eval()  # this merges automatically LoRa weights in main
        optim = None

    elif opt.action == 'concat':
        model.train()
        optim = lora_checkpoint['optim']

    model_state_dict = model.state_dict()
    model_state_dict = {k: v for k, v in model_state_dict.items()
                        if 'generator' not in k}
    generator_state_dict = model.generator.state_dict()
    new_checkpoint = {
        'model': model_state_dict,
        'generator': generator_state_dict,
        'vocab': vocabs_to_dict(vocabs),
        'opt': model_opt,
        'optim': optim,
        }
    torch.save(new_checkpoint, opt.output)
