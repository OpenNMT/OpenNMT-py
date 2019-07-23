#!/usr/bin/env python
""" Convert ckp of huggingface to onmt version"""
from argparse import ArgumentParser
from pathlib import Path
# import pytorch_pretrained_bert
# from pytorch_pretrained_bert.modeling import BertForPreTraining
import torch
import onmt
from collections import OrderedDict
import re

# -1
def decrement(matched):
    value = int(matched.group(1))
    if value < 1:
        raise ValueError('Value Error when converting string')
    string = "bert.encoder.layer.{}.output.LayerNorm".format(value-1)
    return string

def convert_key(key, max_layers):
    if 'bert.embeddings' in key:
        key = key

    elif 'bert.transformer_encoder' in key:
        # convert layer_norm weights
        key = re.sub(r'bert.transformer_encoder.0.layer_norm\.(.*)',
                        r'bert.embeddings.LayerNorm.\1', key)
        key = re.sub(r'bert.transformer_encoder\.(\d+)\.layer_norm',
                        decrement, key) # TODO
        # convert attention weights
        key = re.sub(r'bert.transformer_encoder\.(\d+)\.self_attn.linear_keys\.(.*)',
                        r'bert.encoder.layer.\1.attention.self.key.\2', key)
        key = re.sub(r'bert.transformer_encoder\.(\d+)\.self_attn.linear_values\.(.*)',
                        r'bert.encoder.layer.\1.attention.self.value.\2', key)
        key = re.sub(r'bert.transformer_encoder\.(\d+)\.self_attn.linear_query\.(.*)',
                        r'bert.encoder.layer.\1.attention.self.query.\2', key)
        key = re.sub(r'bert.transformer_encoder\.(\d+)\.self_attn.final_linear\.(.*)',
                        r'bert.encoder.layer.\1.attention.output.dense.\2', key)
        # convert feed forward weights
        key = re.sub(r'bert.transformer_encoder\.(\d+)\.feed_forward.layer_norm\.(.*)',
                        r'bert.encoder.layer.\1.attention.output.LayerNorm.\2', key)
        key = re.sub(r'bert.transformer_encoder\.(\d+)\.feed_forward.w_1\.(.*)',
                        r'bert.encoder.layer.\1.intermediate.dense.\2', key)
        key = re.sub(r'bert.transformer_encoder\.(\d+)\.feed_forward.w_2\.(.*)',
                        r'bert.encoder.layer.\1.output.dense.\2', key)

    elif 'bert.layer_norm' in key:
        key = re.sub(r'bert.layer_norm',
                        r'bert.encoder.layer.'+str(max_layers-1)+'.output.LayerNorm', key)
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
        raise ValueError("Unexpected keys!")    
    return key


def load_bert_weights(bert_model, weights_dict, n_layers=12):
    bert_model_keys = bert_model.state_dict().keys()
    weights_keys = weights_dict.keys()
    bert_weights = OrderedDict()
    generator_weights = OrderedDict()
    model_weights = {"bert": bert_weights,
                     "generator": generator_weights}
    try:
        for key in bert_model_keys:
            key_huggingface = convert_key(key, n_layers)
            # model_weights[key] = converted_key
            if 'generator' not in key:
                truncted_key = re.sub(r'bert\.(.*)', r'\1', key)
                model_weights['bert'][truncted_key] = weights_dict[key_huggingface]
            else:
                truncted_key = re.sub(r'generator\.(.*)', r'\1', key)
                model_weights['generator'][truncted_key] = weights_dict[key_huggingface]
    except ValueError:
        print("Unsuccessful convert!")
        exit()
    return model_weights


def main():
    parser = ArgumentParser()
    parser.add_argument("--layers", type=int, default=None, required=True)
    # parser.add_argument("--bert_model", type=str, default="bert-base-multilingual-uncased")#,  # required=True,
                        # choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                        #          "bert-base-multilingual-uncased", "bert-base-chinese"])
    # parser.add_argument("--bert_model_weights_path", type=str, default="PreTrainedBertckp/")
    # parser.add_argument("--output_dir", type=Path, default="PreTrainedBertckp/")
    # parser.add_argument("--output_name", type=str, default="onmt-bert-base-multilingual-uncased.weights")
    parser.add_argument("--bert_model_weights_file", "-i", type=str, default="PreTrainedBertckp/bert-base-multilingual-uncased.pt")
    parser.add_argument("--output_name", "-o", type=str, default="PreTrainedBertckp/onmt-bert-base-multilingual-uncased.weights")
    args = parser.parse_args()

    n_layers = args.layers
    print("Model contain {} layers.".format(n_layers))

    bert_model_weights = args.bert_model_weights_file
    print("Load weights from {}.".format(bert_model_weights))

    bert_weights = torch.load(bert_model_weights)
    embeddings = onmt.modules.bert_embeddings.BertEmbeddings(105879)
    bert_encoder = onmt.encoders.BertEncoder(embeddings)
    generator = onmt.models.BertPreTrainingHeads(bert_encoder.d_model, bert_encoder.vocab_size)
    bertlm = torch.nn.Sequential(OrderedDict([
                            ('bert', bert_encoder),
                            ('generator', generator)]))
    model_weights = load_bert_weights(bertlm, bert_weights, n_layers)

    ckp={'model': model_weights['bert'], 'generator': model_weights['generator']}

    outfile = args.output_name
    print("Converted weights file in {}".format(outfile))
    torch.save(ckp, outfile)


if __name__ == '__main__':
    main()
