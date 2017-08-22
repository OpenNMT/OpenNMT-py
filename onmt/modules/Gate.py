"""
Context gate is a decoder module that takes as input the previous word
embedding, the current decoder state and the attention state, and produces a
gate.
The gate can be used to select the input from the target side context
(decoder state), from the source context (attention state) or both.
"""
import torch
import torch.nn as nn


def ContextGateFactory(type, embeddings_size, decoder_size,
                       attention_size, output_size):
    """Returns the correct ContextGate class"""

    gate_types = {'source': SourceContextGate,
                  'target': TargetContextGate,
                  'both': BothContextGate}

    assert type in gate_types, "Not valid ContextGate type: {0}".format(type)
    return gate_types[type](embeddings_size, decoder_size, attention_size,
                            output_size)


class ContextGate(nn.Module):
    """Implement up to the computation of the gate"""

    def __init__(self, embeddings_size, decoder_size,
                 attention_size, output_size):
        super(ContextGate, self).__init__()
        input_size = embeddings_size + decoder_size + attention_size
        self.gate = nn.Linear(input_size, output_size, bias=True)
        self.sig = nn.Sigmoid()
        self.source_proj = nn.Linear(attention_size, output_size)
        self.target_proj = nn.Linear(embeddings_size + decoder_size,
                                     output_size)

    def forward(self, prev_emb, dec_state, attn_state):
        input_tensor = torch.cat((prev_emb, dec_state, attn_state), dim=1)
        z = self.sig(self.gate(input_tensor))
        proj_source = self.source_proj(attn_state)
        proj_target = self.target_proj(
            torch.cat((prev_emb, dec_state), dim=1))
        return z, proj_source, proj_target


class SourceContextGate(nn.Module):
    """Apply the context gate only to the source context"""

    def __init__(self, embeddings_size, decoder_size,
                 attention_size, output_size):
        super(SourceContextGate, self).__init__()
        self.context_gate = ContextGate(embeddings_size, decoder_size,
                                        attention_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state):
        z, source, target = self.context_gate(
            prev_emb, dec_state, attn_state)
        return self.tanh(target + z * source)


class TargetContextGate(nn.Module):
    """Apply the context gate only to the target context"""

    def __init__(self, embeddings_size, decoder_size,
                 attention_size, output_size):
        super(TargetContextGate, self).__init__()
        self.context_gate = ContextGate(embeddings_size, decoder_size,
                                        attention_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state):
        z, source, target = self.context_gate(prev_emb, dec_state, attn_state)
        return self.tanh(z * target + source)


class BothContextGate(nn.Module):
    """Apply the context gate to both contexts"""

    def __init__(self, embeddings_size, decoder_size,
                 attention_size, output_size):
        super(BothContextGate, self).__init__()
        self.context_gate = ContextGate(embeddings_size, decoder_size,
                                        attention_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state):
        z, source, target = self.context_gate(prev_emb, dec_state, attn_state)
        return self.tanh((1. - z) * target + z * source)
