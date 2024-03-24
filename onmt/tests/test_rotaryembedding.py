import torch
import unittest
from math import sqrt
from onmt.modules.multi_headed_attn import rotaryembeddings, apply_rotary_emb, shape


def PS(queries, keys, offsets, step):
    query_out, key_out = Phi(queries, keys, offsets, step)
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

# def test_rotary_embeddings(nb_pad_tokens):
#     key = generate_padded_seq_embedding(seq_embedding, pad_embedding, nb_pad_tokens)
#     scores = PS(query=query, key=key)
#     scores = scores[:, :, :, -length:]
#     print(scores[:, 0, :, :])  # print only on first head
#     print(ref_scores[:, 0, :, :])
#     return scores


# class TestRotaryEmbedding(unittest.TestCase):
#     def test_rotary_embeddings_100PAD(self):
#         scores = test_rotary_embeddings(nb_pad_tokens=100)
#         self.assertEqual(
#             torch.allclose(ref_scores[:, :, :, :], scores[:, :, :, :]),
#             True
#         )

#     def test_rotary_embeddings_1000PAD(self):
#         scores = test_rotary_embeddings(nb_pad_tokens=1000)
#         self.assertEqual(
#             torch.allclose(ref_scores[:, :, :, :], scores[:, :, :, :]),
#             True
#         )

batch_size, modeldim = 2, 4096
dimperhead = 128
pad_embedding = torch.rand(1, 1, modeldim)
length1 = 10
length2 = 5
seq1_embedding = torch.rand(1, length1, modeldim)
seq2_embedding = torch.rand(1, length2, modeldim)
queries = torch.rand(batch_size, 1, modeldim)
print('queries', queries.size())
key1 = seq1_embedding
key2 = generate_padded_seq_embedding(seq2_embedding, pad_embedding, nb_pad_tokens=length1 - length2)
ref_key2 = seq2_embedding
keys = torch.cat((key1, key2), dim=0)
print(key1.size(), key2.size(), keys.size())
offsets = [0, 5]
init_rope = rotaryembeddings(dimperhead)
print(init_rope.size()) # torch.Size([2048, 128]


def apply_rotary_emb(query, key, ropes, interleave):
    if interleave:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        query_ = query.float().reshape(*query.shape[:-1], -1, 2)
        query_ = torch.view_as_complex(query_)
        key_ = key.float().reshape(*key.shape[:-1], -1, 2)
        key_ = torch.view_as_complex(key_)
    
        query_out = []
        key_out  = []
        for i, _rope in enumerate(ropes):
            _rope = _rope[:, : _rope.size(1) // 2].view(1, query_.size(1), 1, query_.size(3))
            query_out.append(torch.view_as_real(query_[i, :, :]* _rope).flatten(3))
            key_out.append(torch.view_as_real(key_[i, :, :] * _rope).flatten(3))
        query_out = torch.cat(query_out, dim=0)
        key_out = torch.cat(key_out, dim=0)
        return query_out.transpose(1, 2).type_as(query), key_out.transpose(
            1, 2
        ).type_as(key)
    # else:
    #     cos, sin = rope.real, rope.imag
    #     q_embed = (query * cos) + (rotate_half(query) * sin)
    #     k_embed = (key * cos) + (rotate_half(key) * sin)
    #     return q_embed.type_as(query), k_embed.type_as(key)


def Phi(query, key, offsets, step, interleave=True):
    query = shape(query, dimperhead)
    key = shape(key, dimperhead)
    init_rope = rotaryembeddings(dimperhead)
    seqlen = query.size(2)  # 1
    start_pos = step
    print(start_pos, seqlen)
    if seqlen > init_rope.size(0):
        init_rope = rotaryembeddings(dimperhead, maxseqlen=(seqlen + 2048))
    ropes = [init_rope[step + _offset: step + _offset + seqlen] for _offset in offsets]
    query_out, key_out = apply_rotary_emb(query, key, ropes, interleave)
    return query_out, key_out


batch_size, modeldim = 2, 4096
dimperhead = 128
offset = 1000
pad_embedding = torch.rand(1, 1, modeldim)
length2 = 10
length1 = length1 + offset
seq1_embedding = torch.rand(1, length1, modeldim)
seq2_embedding = torch.rand(1, length2, modeldim)
queries = torch.rand(batch_size, 1, modeldim)
# print('queries', queries.size())
key1 = seq1_embedding
key2 = generate_padded_seq_embedding(seq2_embedding, pad_embedding, nb_pad_tokens=offset)
ref_key2 = seq2_embedding
keys = torch.cat((key1, key2), dim=0)
# print(key1.size(), key2.size(), keys.size())
offsets = [0, offset]
init_rope = rotaryembeddings(dimperhead)
# print(init_rope.size()) # torch.Size([2048, 128]


# scores = PS(queries, keys, offsets, step=1)
# print(scores[:, 0, :, :]) # print on the 
# scores = PS(queries, keys, offsets, step=5)
# print(scores[:, 0, :, :])

ref_scores = PS(queries[1, :, :].unsqueeze(0), ref_key2, offsets=[0], step=0)
scores = PS(queries, keys, offsets, step=0)
print(ref_scores[:, 0, :, :])
print(scores[1, 0, :, -length2:])

# scores = PS(queries, keys, offsets, step=5)
# print(scores[:, 0, :, :])
# if __name__ == '__main__':
#     # unittest.main()
