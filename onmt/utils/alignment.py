# -*- coding: utf-8 -*-

import torch
from itertools import accumulate
from onmt.constants import SubwordMarker


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


def extract_alignment(align_matrix, tgt_mask, src_len, n_best):
    """
    Extract a batched align_matrix into its src indice alignment lists,
    with tgt_mask to filter out invalid tgt position as EOS/PAD.
    BOS already excluded from tgt_mask in order to match prediction.

    Args:
        align_matrix (Tensor): ``(B, tgt_len, src_len)``,
            attention head normalized by Softmax(dim=-1)
        tgt_mask (BoolTensor): ``(B, tgt_len)``, True for EOS, PAD.
        src_len (LongTensor): ``(B,)``, containing valid src lengths
        n_best (int): a value indicating number of parallel translation.
        * B: denote flattened batch as B = batch_size * n_best.

    Returns:
        alignments (List[List[FloatTensor|None]]): ``(batch_size, n_best,)``,
         containing valid alignment matrix (or None if blank prediction)
         for each translation.
    """
    batch_size_n_best = align_matrix.size(0)
    assert batch_size_n_best % n_best == 0

    alignments = [[] for _ in range(batch_size_n_best // n_best)]

    # treat alignment matrix one by one as each have different lengths
    for i, (am_b, tgt_mask_b, src_len) in enumerate(
            zip(align_matrix, tgt_mask, src_len)):
        valid_tgt = ~tgt_mask_b
        valid_tgt_len = valid_tgt.sum()
        if valid_tgt_len == 0:
            # No alignment if not exist valid tgt token
            valid_alignment = None
        else:
            # get valid alignment (sub-matrix from full paded aligment matrix)
            am_valid_tgt = am_b.masked_select(valid_tgt.unsqueeze(-1)) \
                               .view(valid_tgt_len, -1)
            valid_alignment = am_valid_tgt[:, :src_len]  # only keep valid src
        alignments[i // n_best].append(valid_alignment)

    return alignments


def build_align_pharaoh(valid_alignment):
    """Convert valid alignment matrix to i-j (from 0) Pharaoh format pairs,
    or empty list if it's None.
    """
    align_pairs = []
    if isinstance(valid_alignment, torch.Tensor):
        tgt_align_src_id = valid_alignment.argmax(dim=-1)

        for tgt_id, src_id in enumerate(tgt_align_src_id.tolist()):
            align_pairs.append(str(src_id) + "-" + str(tgt_id))
        align_pairs.sort(key=lambda x: int(x.split('-')[-1]))  # sort by tgt_id
        align_pairs.sort(key=lambda x: int(x.split('-')[0]))  # sort by src_id
    return align_pairs


def to_word_align(src, tgt, subword_align, m_src='joiner', m_tgt='joiner'):
    """Convert subword alignment to word alignment.

    Args:
        src (string): tokenized sentence in source language.
        tgt (string): tokenized sentence in target language.
        subword_align (string): align_pharaoh correspond to src-tgt.
        m_src (string): tokenization mode used in src,
            can be ["joiner", "spacer"].
        m_tgt (string): tokenization mode used in tgt,
            can be ["joiner", "spacer"].

    Returns:
        word_align (string): converted alignments correspand to
            detokenized src-tgt.
    """
    assert m_src in ["joiner", "spacer"], "Invalid value for argument m_src!"
    assert m_tgt in ["joiner", "spacer"], "Invalid value for argument m_tgt!"

    src, tgt = src.strip().split(), tgt.strip().split()
    subword_align = {(int(a), int(b)) for a, b in (x.split("-")
                     for x in subword_align.split())}

    src_map = (subword_map_by_spacer(src) if m_src == 'spacer'
               else subword_map_by_joiner(src))

    tgt_map = (subword_map_by_spacer(src) if m_tgt == 'spacer'
               else subword_map_by_joiner(src))

    word_align = list({"{}-{}".format(src_map[a], tgt_map[b])
                       for a, b in subword_align})
    word_align.sort(key=lambda x: int(x.split('-')[-1]))  # sort by tgt_id
    word_align.sort(key=lambda x: int(x.split('-')[0]))  # sort by src_id
    return " ".join(word_align)


# Helper functions
def begin_uppercase(token):
    return token == SubwordMarker.BEGIN_UPPERCASE


def end_uppercase(token):
    return token == SubwordMarker.END_UPPERCASE


def begin_case(token):
    return token == SubwordMarker.BEGIN_CASED


def case_markup(token):
    return begin_uppercase(token) \
        or end_uppercase(token) \
        or begin_case(token)


def subword_map_by_joiner(subwords,
                          original_subwords=None,
                          marker=SubwordMarker.JOINER):
    """Return word id for each subword token (annotate by joiner)."""

    flags = [1] * len(subwords)
    j = 0
    finished = True
    for i, tok in enumerate(subwords):

        previous_tok = subwords[i-1] if i else ""  # Previous N-1 token
        previous_tok_2 = subwords[i-2] if i > 1 else ""  # Previous N-2 token
        # Keeps track of the original words/subwords
        # ('prior_tokenization' option)
        current_original_subword = "" if not original_subwords \
            else original_subwords[j] if j < len(original_subwords) else ""

        if tok.startswith(marker) and tok != current_original_subword:
            flags[i] = 0
        elif (previous_tok.endswith(marker)
                or begin_case(previous_tok)
                or begin_uppercase(previous_tok)) \
                and not finished:
            flags[i] = 0
        elif previous_tok_2.endswith(marker) \
                and case_markup(previous_tok) \
                and not finished:
            flags[i] = 0
        elif end_uppercase(tok) and tok != current_original_subword:
            flags[i] = 0
        else:
            finished = False
            if tok == current_original_subword:
                finished = True
            j += 1

    flags[0] = 0
    word_group = list(accumulate(flags))

    if original_subwords:
        assert max(word_group) < len(original_subwords)
    return word_group


def subword_map_by_spacer(subwords, marker=SubwordMarker.SPACER):
    """Return word id for each subword token (annotate by spacer)."""
    flags = [0] * len(subwords)
    for i, tok in enumerate(subwords):
        if marker in tok:
            if case_markup(tok.replace(marker, "")):
                if i < len(subwords)-1:
                    flags[i] = 1
            else:
                if i > 0:
                    previous = subwords[i-1].replace(marker, "")
                    if not case_markup(previous):
                        flags[i] = 1

    # In case there is a final case_markup when new_spacer is on
    for i in range(1, len(subwords)-1):
        if case_markup(subwords[-i]):
            flags[-i] = 0
        elif subwords[-i] == marker:
            flags[-i] = 0
            break

    word_group = list(accumulate(flags))
    if word_group[0] == 1:  # when dummy prefix is set
        word_group = [item - 1 for item in word_group]
    return word_group
