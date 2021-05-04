"""
Here come the tests for attention types and their compatibility
"""
import unittest
import torch

from onmt.modules import (
    MultiHeadedAttention,
    MultiHeadedPseudoSelfAttention,
)
from onmt.utils.misc import sequence_mask
from onmt.decoders.transformer import TransformerDecoderLayerBase


class TestPseudoSelfAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        max_relative_positions = 0
        heads = 2
        cls.d_model = 16
        cls.pseudo_self_attention = MultiHeadedPseudoSelfAttention(
            heads,
            cls.d_model,
            dropout=0,
            max_relative_positions=max_relative_positions,
        )
        cls.self_attention = MultiHeadedAttention(
            heads,
            cls.d_model,
            dropout=0,
            max_relative_positions=max_relative_positions,
        )
        torch.nn.init.constant_(
            cls.pseudo_self_attention.linear_keys.weight, 1
        )
        torch.nn.init.constant_(
            cls.pseudo_self_attention.linear_values.weight, 1
        )
        torch.nn.init.constant_(
            cls.pseudo_self_attention.linear_query.weight, 1
        )
        torch.nn.init.constant_(cls.self_attention.linear_keys.weight, 1)
        torch.nn.init.constant_(cls.self_attention.linear_values.weight, 1)
        torch.nn.init.constant_(cls.self_attention.linear_query.weight, 1)

        torch.nn.init.constant_(cls.pseudo_self_attention.linear_keys.bias, 0)
        torch.nn.init.constant_(
            cls.pseudo_self_attention.linear_values.bias, 0
        )
        torch.nn.init.constant_(cls.pseudo_self_attention.linear_query.bias, 0)
        torch.nn.init.constant_(cls.self_attention.linear_keys.bias, 0)
        torch.nn.init.constant_(cls.self_attention.linear_values.bias, 0)
        torch.nn.init.constant_(cls.self_attention.linear_query.bias, 0)

        torch.nn.init.constant_(
            cls.pseudo_self_attention.final_linear.weight, 1
        )
        torch.nn.init.constant_(cls.pseudo_self_attention.final_linear.bias, 1)
        torch.nn.init.constant_(cls.self_attention.final_linear.weight, 1)
        torch.nn.init.constant_(cls.self_attention.final_linear.bias, 1)

    def test_pseudo_self_attention_is_self_attention_without_encoding(self):
        X = torch.zeros(
            (3, 5, self.d_model)
        )  # (batch_size, seq_len, dim_model)
        Y = torch.ones((3, 8, self.d_model))
        pseudo_key_value = torch.cat([X, Y], axis=1)

        output_self_attn, _ = self.self_attention(Y, Y, Y, attn_type="self")
        output_pseudo_self_attn, _ = self.pseudo_self_attention(
            pseudo_key_value, pseudo_key_value, Y, attn_type="self"
        )
        self.assertTrue(output_self_attn.equal(output_pseudo_self_attn))

    def test_masked_pseudo_self_attention_equals_premasked_encoder(self):
        X = 0.3 * torch.ones(
            (4, 5, self.d_model)
        )  # (batch_size, seq_len, dim_model)
        X[0, 4:, :] = 1000
        X[1, 3:, :] = 1000

        X_premasked = 0.3 * torch.ones(
            (4, 5, self.d_model)
        )  # (batch_size, seq_len, dim_model)
        X_premasked[0, 4:, :] = 0
        X_premasked[1, 3:, :] = 0

        Y = torch.ones((4, 8, self.d_model))

        pseudo_key_value = torch.cat([X, Y], axis=1)
        masked_pseudo_key_value = torch.cat([X_premasked, Y], axis=1)

        src_pad_mask = ~sequence_mask(torch.tensor([4, 3, 1, 5]), 5).unsqueeze(
            1
        )
        no_mask_src_pad_mask = ~sequence_mask(
            torch.tensor([5, 5, 5, 5]), 5
        ).unsqueeze(1)
        tgt_pad_mask = ~sequence_mask(torch.tensor([8, 3, 8, 1]), 8).unsqueeze(
            1
        )

        dec_mask = TransformerDecoderLayerBase._compute_dec_mask(
            tgt_pad_mask, future=False
        )

        pseudo_mask = torch.cat(
            [src_pad_mask.repeat(1, dec_mask.size(-1), 1), dec_mask], axis=-1
        )
        no_mask_pseudo_mask = torch.cat(
            [no_mask_src_pad_mask.repeat(1, dec_mask.size(-1), 1), dec_mask],
            axis=-1,
        )

        output, _ = self.pseudo_self_attention(
            pseudo_key_value,
            pseudo_key_value,
            Y,
            mask=pseudo_mask,
            attn_type="self",
        )

        output_masked, _ = self.pseudo_self_attention(
            masked_pseudo_key_value,
            masked_pseudo_key_value,
            Y,
            mask=no_mask_pseudo_mask,
            attn_type="self",
        )

        self.assertTrue(output.equal(output_masked))
