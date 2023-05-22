import torch

m = torch.load("finetuned_llama7B/llama7B-vicuna-onmt_step_4000.concat.pt")
m['opt'].encoder_type = "transformer_lm"
torch.save(
    m, "finetuned_llama7B/llama7B-vicuna-onmt_step_4000.concat_added_key.pt")
