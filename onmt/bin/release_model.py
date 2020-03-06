#!/usr/bin/env python
import argparse
import torch


def get_ctranslate2_model_spec(opt):
    """Creates a CTranslate2 model specification from the model options."""
    with_relative_position = getattr(opt, "max_relative_positions", 0) > 0
    is_ct2_compatible = (
        opt.encoder_type == "transformer"
        and opt.decoder_type == "transformer"
        and opt.enc_layers == opt.dec_layers
        and getattr(opt, "self_attn_type", "scaled-dot") == "scaled-dot"
        and ((opt.position_encoding and not with_relative_position)
             or (with_relative_position and not opt.position_encoding)))
    if not is_ct2_compatible:
        return None
    import ctranslate2
    num_heads = getattr(opt, "heads", 8)
    if opt.enc_layers == opt.dec_layers:
        opt.layers = opt.enc_layers
    return ctranslate2.specs.TransformerSpec(
        opt.layers,
        num_heads,
        with_relative_position=with_relative_position)


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
                        choices=["int8", "int16"],
                        default=None,
                        help="Quantization type for CT2 model.")
    opt = parser.parse_args()

    model = torch.load(opt.model)
    if opt.format == "pytorch":
        model["optim"] = None
        torch.save(model, opt.output)
    elif opt.format == "ctranslate2":
        model_spec = get_ctranslate2_model_spec(model["opt"])
        if model_spec is None:
            raise ValueError("This model is not supported by CTranslate2. Go "
                             "to https://github.com/OpenNMT/CTranslate2 for "
                             "more information on supported models.")
        import ctranslate2
        converter = ctranslate2.converters.OpenNMTPyConverter(opt.model)
        converter.convert(opt.output, model_spec, force=True,
                          quantization=opt.quantization)


if __name__ == "__main__":
    main()
