# -*- coding: utf-8 -*-

import torch
import random
import inspect
from itertools import islice, repeat, accumulate


def split_corpus(path, shard_size, default=None):
    """yield a `list` containing `shard_size` line of `path`,
    or repeatly generate `default` if `path` is None.
    """
    if path is not None:
        return _split_corpus(path, shard_size)
    else:
        return repeat(default)


def _split_corpus(path, shard_size):
    """Yield a `list` containing `shard_size` line of `path`.
    """
    with open(path, "rb") as f:
        if shard_size <= 0:
            yield f.readlines()
        else:
            while True:
                shard = list(islice(f, shard_size))
                if not shard:
                    break
                yield shard


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpu_ranks') and len(opt.gpu_ranks) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)


def set_random_seed(seed, is_cuda):
    """Sets the random seed."""
    if seed > 0:
        torch.manual_seed(seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if is_cuda and seed > 0:
        # These ensure same initialization in multi gpu mode
        torch.cuda.manual_seed(seed)


def generate_relative_positions_matrix(length, max_relative_positions,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length+1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    if transpose:
        z_t = z.transpose(1, 2)
        x_tz_matmul = torch.matmul(x_t_r, z_t)
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


def fn_args(fun):
    """Returns the list of function arguments name."""
    return inspect.getfullargspec(fun).args


def report_matrix(row_label, column_label, matrix):
    header_format = "{:>10.10} " + "{:>10.7} " * len(row_label)
    row_format = "{:>10.10} " + "{:>10.7f} " * len(row_label)
    output = header_format.format("", *row_label) + '\n'
    for word, row in zip(column_label, matrix):
        max_index = row.index(max(row))
        row_format = row_format.replace(
            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
        row_format = row_format.replace(
            "{:*>10.7f} ", "{:>10.7f} ", max_index)
        output += row_format.format(word, *row) + '\n'
        row_format = "{:>10.10} " + "{:>10.7f} " * len(row_label)
    return output


def make_batch_align_matrix(index_tensor, size=None, normalize=False):
    """
    Convert a sparse index_tensor into a batch of alignment matrix,
    with row normalize to the sum of 1 if set normalize.

    Args:
        index_tensor (LongTensor): ``(N, 3)`` of [batch_id, tgt_id, src_id]
        size (List[int]): Size of the sparse tensor.
        normalize (bool): if normalize the 2nd dim of resulting tensor.
    """
    n_fill, device = index_tensor.size(0), index_tensor.device
    value_tensor = torch.ones([n_fill], dtype=torch.float)
    dense_tensor = torch.sparse_coo_tensor(
        index_tensor.t(), value_tensor, size=size, device=device).to_dense()
    if normalize:
        row_sum = dense_tensor.sum(-1, keepdim=True)  # sum by row(tgt)
        # threshold on 1 to avoid div by 0
        torch.nn.functional.threshold(row_sum, 1, 1, inplace=True)
        dense_tensor.div_(row_sum)
    return dense_tensor


def extract_alignment(align_matrix, tgt_mask, src_lens, n_best):
    """
    Extract a batched align_matrix into its src indice alignment lists,
    with tgt_mask to filter out invalid tgt position as EOS/PAD.
    BOS already excluded from tgt_mask in order to match prediction.

    Args:
        align_matrix (Tensor): ``(B, tgt_len, src_len)``,
            attention head normalized by Softmax(dim=-1)
        tgt_mask (BoolTensor): ``(B, tgt_len)``, True for EOS, PAD.
        src_lens (LongTensor): ``(B,)``, containing valid src length
        n_best (int): a value indicating number of parallel translation.
        * B: denote flattened batch as B = batch_size * n_best.

    Returns:
        alignments (List[List[FloatTensor]]): ``(batch_size, n_best,)``,
         containing valid alignment matrix for each translation.
    """
    batch_size_n_best = align_matrix.size(0)
    assert batch_size_n_best % n_best == 0

    alignments = [[] for _ in range(batch_size_n_best // n_best)]

    # treat alignment matrix one by one as each have different lengths
    for i, (am_b, tgt_mask_b, src_len) in enumerate(
            zip(align_matrix, tgt_mask, src_lens)):
        valid_tgt = ~tgt_mask_b
        valid_tgt_len = valid_tgt.sum()
        # get valid alignment (sub-matrix from full paded aligment matrix)
        am_valid_tgt = am_b.masked_select(valid_tgt.unsqueeze(-1)) \
                           .view(valid_tgt_len, -1)
        valid_alignment = am_valid_tgt[:, :src_len]  # only keep valid src
        alignments[i // n_best].append(valid_alignment)

    return alignments


def build_align_pharaoh(valid_alignment):
    """Convert valid alignment matrix to i-j Pharaoh format.(0 indexed)"""
    align_pairs = []
    tgt_align_src_id = valid_alignment.argmax(dim=-1)

    for tgt_id, src_id in enumerate(tgt_align_src_id.tolist()):
        align_pairs.append(str(src_id) + "-" + str(tgt_id))
    align_pairs.sort(key=lambda x: x[-1])  # sort by tgt_id
    align_pairs.sort(key=lambda x: x[0])  # sort by src_id
    return align_pairs


def to_word_align(src, tgt, subword_align, mode):
    """Convert subword alignment to word alignment.

    Args:
        src (string): tokenized sentence in source language.
        tgt (string): tokenized sentence in target language.
        subword_align (string): align_pharaoh correspond to src-tgt.
        mode (string): tokenization mode used by src and tgt,
            choose from ["joiner", "spacer"].

    Returns:
        word_align (string): converted alignments correspand to
            detokenized src-tgt.
    """
    src, tgt = src.strip().split(), tgt.strip().split()
    subword_align = {(int(a), int(b)) for a, b in (x.split("-")
                     for x in subword_align.split())}
    if mode == 'joiner':
        src_map = subword_map_by_joiner(src, marker='￭')
        tgt_map = subword_map_by_joiner(tgt, marker='￭')
    elif mode == 'spacer':
        src_map = subword_map_by_spacer(src, marker='▁')
        tgt_map = subword_map_by_spacer(tgt, marker='▁')
    else:
        raise ValueError("Invalid value for argument mode!")
    word_align = " ".join({"{}-{}".format(src_map[a], tgt_map[b])
                          for a, b in subword_align})
    return word_align


def subword_map_by_joiner(subwords, marker='￭'):
    flags = [0] * len(subwords)
    for i, tok in enumerate(subwords):
        if tok.endswith(marker):
            flags[i] = 1
        if tok.startswith(marker):
            assert i >= 1 and flags[i-1] != 1, \
                "Sentence `{}` not correct!".format(" ".join(subwords))
            flags[i-1] = 1
    marker_acc = list(accumulate([0] + flags[:-1]))
    word_group = [(i - maker_sofar) for i, maker_sofar
                  in enumerate(marker_acc)]
    return word_group


def subword_map_by_spacer(subwords, marker='▁'):
    word_group = list(accumulate([int(marker in x) for x in subwords]))
    if word_group[0] == 1:  # when dummy prefix is set
        word_group = [item - 1 for item in word_group]
    return word_group
