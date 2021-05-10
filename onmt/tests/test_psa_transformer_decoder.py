"""
Here come the tests for attention types and their compatibility
"""
import unittest
import torch

from onmt.decoders.transformer import TransformerLMPseudoSelfAttentionDecoder
from onmt.modules import Embeddings
from onmt.modules.position_ffn import ActivationFunction


class TestPSADecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        emb = Embeddings(
            word_vec_size=100,
            position_encoding=True,
            feat_merge="concat",
            feat_vec_exponent=0.7,
            feat_vec_size=-1,
            dropout=0,
            word_padding_idx=1,
            feat_padding_idx=[],
            word_vocab_size=100,
            feat_vocab_sizes=[],
            sparse=False,
            freeze_word_vecs=False,
        )
        cls.psa_transformer_decoder = TransformerLMPseudoSelfAttentionDecoder(
            num_layers=2,
            d_model=100,
            heads=2,
            d_ff=100,
            copy_attn=False,
            self_attn_type="scaled-dot",
            dropout=0,
            attention_dropout=0,
            embeddings=emb,
            max_relative_positions=0,
            aan_useffn=False,
            full_context_alignment=None,
            alignment_layer=None,
            alignment_heads=None,
            pos_ffn_activation_fn=ActivationFunction.relu,
        )
        batch_size = 3
        src_len = 58
        tgt_len = 12
        cls.memory_bank = torch.rand([src_len, batch_size, 100])
        cls.tgt = torch.randint(3, 99, [tgt_len, batch_size, 1])
        cls.src = torch.randint(3, 99, [src_len, batch_size, 1])
        cls.memory_lengths = torch.tensor([src_len] * batch_size)
        cls.memory_lengths[0] -= 3
        cls.psa_transformer_decoder.init_state(
            cls.src, cls.memory_bank, cls.memory_bank
        )

    def test_psa_transformer_caching_equals_no_caching(
        self,
    ):
        dec_outs, _ = self.psa_transformer_decoder(
            self.tgt[1:3],
            self.memory_bank,
            memory_lengths=self.memory_lengths,
            step=None,
        )
        dec_outs_step_0, _ = self.psa_transformer_decoder(
            self.tgt[1:2],
            self.memory_bank,
            memory_lengths=self.memory_lengths,
            step=0,
        )
        dec_outs_step_1, _ = self.psa_transformer_decoder(
            self.tgt[2:3],
            self.memory_bank,
            memory_lengths=self.memory_lengths,
            step=1,
        )
        # randomness might cause failing (seed is set to avoid that)
        # small differences are expected due to masking with huge negative
        # float but not infinite
        self.assertTrue(dec_outs_step_1.allclose(dec_outs[1:]))


if __name__ == "__main__":
    unittest.main()
