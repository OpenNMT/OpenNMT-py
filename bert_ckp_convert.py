#!/usr/bin/env python
""" Convert weights of huggingface Bert to onmt Bert"""
from argparse import ArgumentParser
import torch
from onmt.encoders.bert import BertEncoder
from onmt.models.bert_generators import BertPreTrainingHeads
from onmt.modules.bert_embeddings import BertEmbeddings
from collections import OrderedDict
import re


def decrement(matched):
    value = int(matched.group(1))
    if value < 1:
        raise ValueError('Value Error when converting string')
    string = "bert.encoder.layer.{}.output.LayerNorm".format(value-1)
    return string


def mapping_key(key, max_layers):
    if 'bert.embeddings' in key:
        key = key

    elif 'bert.encoder' in key:
        # convert layer_norm weights
        key = re.sub(r'bert.encoder.0.layer_norm\.(.*)',
                     r'bert.embeddings.LayerNorm.\1', key)
        key = re.sub(r'bert.encoder\.(\d+)\.layer_norm',
                     decrement, key)
        # convert attention weights
        key = re.sub(r'bert.encoder\.(\d+)\.self_attn.linear_keys\.(.*)',
                     r'bert.encoder.layer.\1.attention.self.key.\2', key)
        key = re.sub(r'bert.encoder\.(\d+)\.self_attn.linear_values\.(.*)',
                     r'bert.encoder.layer.\1.attention.self.value.\2', key)
        key = re.sub(r'bert.encoder\.(\d+)\.self_attn.linear_query\.(.*)',
                     r'bert.encoder.layer.\1.attention.self.query.\2', key)
        key = re.sub(r'bert.encoder\.(\d+)\.self_attn.final_linear\.(.*)',
                     r'bert.encoder.layer.\1.attention.output.dense.\2', key)
        # convert feed forward weights
        key = re.sub(r'bert.encoder\.(\d+)\.feed_forward.layer_norm\.(.*)',
                     r'bert.encoder.layer.\1.attention.output.LayerNorm.\2',
                     key)
        key = re.sub(r'bert.encoder\.(\d+)\.feed_forward.w_1\.(.*)',
                     r'bert.encoder.layer.\1.intermediate.dense.\2', key)
        key = re.sub(r'bert.encoder\.(\d+)\.feed_forward.w_2\.(.*)',
                     r'bert.encoder.layer.\1.output.dense.\2', key)

    elif 'bert.layer_norm' in key:
        key = re.sub(r'bert.layer_norm',
                     r'bert.encoder.layer.' + str(max_layers - 1) +
                     '.output.LayerNorm', key)
    elif 'bert.pooler' in key:
        key = key
    elif 'generator.next_sentence' in key:
        key = re.sub(r'generator.next_sentence.linear\.(.*)',
                     r'cls.seq_relationship.\1', key)
    elif 'generator.mask_lm' in key:
        key = re.sub(r'generator.mask_lm.bias',
                     r'cls.predictions.bias', key)
        key = re.sub(r'generator.mask_lm.decode.weight',
                     r'cls.predictions.decoder.weight', key)
        key = re.sub(r'generator.mask_lm.transform.dense\.(.*)',
                     r'cls.predictions.transform.dense.\1', key)
        key = re.sub(r'generator.mask_lm.transform.layer_norm\.(.*)',
                     r'cls.predictions.transform.LayerNorm.\1', key)
    else:
        raise KeyError("Unexpected keys! Please provide HuggingFace weights")
    return key


def convert_bert_weights(bert_model, weights, n_layers=12):
    bert_model_keys = bert_model.state_dict().keys()
    bert_weights = OrderedDict()
    generator_weights = OrderedDict()
    model_weights = {"bert": bert_weights,
                     "generator": generator_weights}
    hugface_keys = weights.keys()
    try:
        for key in bert_model_keys:
            hugface_key = mapping_key(key, n_layers)
            if hugface_key not in hugface_keys:
                if 'LayerNorm' in hugface_key:
                    # Fix LayerNorm of old huggingface ckp
                    hugface_key = re.sub(r'LayerNorm.weight',
                                         r'LayerNorm.gamma', hugface_key)
                    hugface_key = re.sub(r'LayerNorm.bias',
                                         r'LayerNorm.beta', hugface_key)
                    if hugface_key in hugface_keys:
                        print("[OLD Weights file]gamma/beta is used in " +
                              "naming BertLayerNorm.")
                    else:
                        raise KeyError("Key %s not found in weight file"
                                       % hugface_key)
                else:
                    raise KeyError("Key %s not found in weight file"
                                   % hugface_key)
            if 'generator' not in key:
                onmt_key = re.sub(r'bert\.(.*)', r'\1', key)
                model_weights['bert'][onmt_key] = weights[hugface_key]
            else:
                onmt_key = re.sub(r'generator\.(.*)', r'\1', key)
                model_weights['generator'][onmt_key] = weights[hugface_key]
    except ValueError:
        print("Unsuccessful convert!")
        exit()
    return model_weights


def main():
    parser = ArgumentParser()
    parser.add_argument("--layers", type=int, default=None, required=True)

    parser.add_argument("--bert_model_weights_file", "-i", type=str,
                        default=None, required=True, help="Path to the "
                        "huggingface Bert weights file download from "
                        "https://github.com/huggingface/pytorch-transformers")

    parser.add_argument("--output_name", "-o", type=str,
                        default=None, required=True,
                        help="output onmt version Bert weight file Path")
    args = parser.parse_args()

    n_layers = args.layers
    print("Model contain {} layers.".format(n_layers))

    bert_model_weights = args.bert_model_weights_file
    print("Load weights from {}.".format(bert_model_weights))

    bert_weights = torch.load(bert_model_weights)
    embeddings = BertEmbeddings(105879)
    bert_encoder = BertEncoder(embeddings)
    generator = BertPreTrainingHeads(bert_encoder.d_model,
                                     embeddings.vocab_size)
    bertlm = torch.nn.Sequential(OrderedDict([
                            ('bert', bert_encoder),
                            ('generator', generator)]))
    model_weights = convert_bert_weights(bertlm, bert_weights, n_layers)

    ckp = {'model': model_weights['bert'],
           'generator': model_weights['generator']}

    outfile = args.output_name
    print("Converted weights file in {}".format(outfile))
    torch.save(ckp, outfile)


if __name__ == '__main__':
    main()
