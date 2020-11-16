import torch

def load_pretrained_weights(model, gpt2_path):
    """
    This function can be used for loading pre-trained GPT2 weights, except the embedding layers (wte & wpe).

    Not sure, where to put this file or to call. So adding it here, can be called anywhere according to the need.

    Args:
        model (NMTModel): the initial seq2seq model with GPT2Decoder as decoder
        gpt2_path (str): path to pre-trained gpt2 weights downloaded from huggingface's transformers. This can 
        be downloaded from https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin


    Return:
        model (NMTModel): model loaded with pre-trained weights for decoder
    """

    model_dict = model.state_dict()
    state_dict = torch.load(gpt2_path) #pretrained weights

    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        if "ln_1.weight" in key:
            new_key = key.replace("ln_1.weight", "layer_norm_1.weight")
            new_keys.append(new_key)
            old_keys.append(key)
        if "ln_1.bias" in key:
            new_key = key.replace("ln_1.bias", "layer_norm_1.bias")
            new_keys.append(new_key)
            old_keys.append(key)
        if "ln_2.weight" in key:
            new_key = key.replace("ln_2.weight", "layer_norm_2.weight")
            new_keys.append(new_key)
            old_keys.append(key)
        if "ln_2.bias" in key:
            new_key = key.replace("ln_2.bias", "layer_norm_2.bias")
            new_keys.append(new_key)
            old_keys.append(key)

        if "mlp.c_fc.weight" in key:
            new_key = key.replace("mlp.c_fc.weight", "feed_forward.c_fc.weight")
            new_keys.append(new_key)
            old_keys.append(key)
        if "mlp.c_fc.bias" in key:
            new_key = key.replace("mlp.c_fc.bias", "feed_forward.c_fc.bias")
            new_keys.append(new_key)
            old_keys.append(key)

        if "mlp.c_proj.weight" in key:
            new_key = key.replace("mlp.c_proj.weight", "feed_forward.c_proj.weight")
            new_keys.append(new_key)
            old_keys.append(key)
        if "mlp.c_proj.bias" in key:
            new_key = key.replace("mlp.c_proj.bias", "feed_forward.c_proj.bias")
            new_keys.append(new_key)
            old_keys.append(key)

        if "attn.c_attn.weight" in key:
            new_key = key.replace("attn.c_attn.weight", "self_attn.c_attn.weight")
            new_keys.append(new_key)
            old_keys.append(key)
        if "attn.c_attn.bias" in key:
            new_key = key.replace("attn.c_attn.bias", "self_attn.c_attn.bias")
            new_keys.append(new_key)
            old_keys.append(key)

        if "attn.c_proj.weight" in key:
            new_key = key.replace("attn.c_proj.weight", "self_attn.c_proj.weight")
            new_keys.append(new_key)
            old_keys.append(key)
        if "attn.c_proj.bias" in key:
            new_key = key.replace("attn.c_proj.bias", "self_attn.c_proj.bias")
            new_keys.append(new_key)
            old_keys.append(key)

        tmp = ".".join(key.split(".")[2:])
        if "attn.bias" == tmp:
            new_key = key.replace("attn.bias", "self_attn.bias")
            new_keys.append(new_key)
            old_keys.append(key)

    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key]=state_dict.pop(old_key)

    pretrained_dict = {}
    for ks, vs in state_dict.items():
        for km, vm in model_dict.items():
            if "decoder.transformer_layers" in km:
                ks1 = '.'.join(ks.split('.')[1:])
                km1 = '.'.join(km.split('.')[2:])
                if ks1 == km1:
                    pretrained_dict[km] = vs

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model
