import torch
import unittest
from math import sqrt
from onmt.modules.multi_headed_attn import rotaryembeddings, apply_rotary_emb, shape


def PS(query, key):
    query_out, key_out = Phi(query, key)
    scores = torch.matmul(query_out, key_out.transpose(2, 3))
    print(scores.size())
    return scores


def generate_padded_seq_embedding(seq_embedding, pad_embedding, nb_pad_tokens):
    padded_seq_embedding = torch.cat(
        (pad_embedding.repeat(1, nb_pad_tokens, 1),
         seq_embedding),
        dim=1)  # left padding
    return padded_seq_embedding


def Phi(query, key, interleave=True):
    query = shape(query, dimperhead)
    key = shape(key, dimperhead)
    init_rope = rotaryembeddings(dimperhead)
    start_pos = key.size(2)
    seqlen = query.size(2)  # 1
    print(start_pos, seqlen)
    if seqlen > init_rope.size(0):
        init_rope = rotaryembeddings(dimperhead, maxseqlen=(seqlen + 2048))
    rope = init_rope[start_pos : start_pos + seqlen]
    query_out, key_out = apply_rotary_emb(
        query, key, rope, interleave
    )
    query_out /= sqrt(dimperhead)
    return query_out, key_out


length, modeldim = 10, 4096
dimperhead = 128
pad_embedding = torch.rand(1, 1, modeldim)
seq_embedding = torch.rand(1, length, modeldim)
query = torch.rand(1, 1, modeldim)
ref_key = generate_padded_seq_embedding(seq_embedding, pad_embedding, nb_pad_tokens=0)
ref_scores = PS(query=query, key=ref_key)


def test_rotary_embeddings(nb_pad_tokens):
    key = generate_padded_seq_embedding(seq_embedding, pad_embedding, nb_pad_tokens)
    scores = PS(query=query, key=key)
    scores = scores[:, :, :, -length:]
    print(scores[:, 0, :, :])  # print only on first head
    print(ref_scores[:, 0, :, :])
    return scores


class TestRotaryEmbedding(unittest.TestCase):
    def test_rotary_embeddings_100PAD(self):
        scores = test_rotary_embeddings(nb_pad_tokens=100)
        self.assertEqual(
            torch.allclose(ref_scores[:, :, :, :], scores[:, :, :, :]),
            True
        )

    def test_rotary_embeddings_1000PAD(self):
        scores = test_rotary_embeddings(nb_pad_tokens=1000)
        self.assertEqual(
            torch.allclose(ref_scores[:, :, :, :], scores[:, :, :, :]),
            True
        )


if __name__ == '__main__':
    unittest.main()
