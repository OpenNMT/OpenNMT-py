import torch
from onmt.constants import DefaultTokens, CorpusTask
from torch.nn.utils.rnn import pad_sequence
from onmt.utils.logging import logger
from onmt.inputters.example import Example


def parse_features(line, n_feats=0, defaults=None):
    """
    Parses text lines with features appended to each token.
    Ex.: This￨A￨B is￨A￨A a￨C￨A test￨A￨B
    """
    text, feats = [], [[] for _ in range(n_feats)]
    check, count = 0, 0
    for token in line.split(' '):
        tok, *fts = token.strip().split("￨")
        check += len(fts)
        count += 1
        if not fts and defaults is not None:
            if isinstance(defaults, str):
                defaults = defaults.split("￨")
            assert len(defaults) == n_feats, \
                "The number of provided defaults does not " \
                "match the number of feats"
            fts = defaults
        assert len(fts) == n_feats, \
            f"The number of fetures does not match the " \
            f"expected number of features. Found {len(fts)} " \
            f"features in the data but {n_feats} were expected"
        text.append(tok)
        for i in range(n_feats):
            feats[i].append(fts[i])
    # Check if all tokens have features or none at all
    assert check == 0 or check == count*n_feats, "Some features are missing"
    feats = [" ".join(x) for x in feats] if n_feats > 0 else None
    return " ".join(text), feats


def parse_align_idx(align_pharaoh):
    """
    Parse Pharaoh alignment into [[<src>, <tgt>], ...]
    """
    align_list = align_pharaoh.strip().split(' ')
    flatten_align_idx = []
    for align in align_list:
        try:
            src_idx, tgt_idx = align.split('-')
        except ValueError:
            logger.warning("{} in `{}`".format(align, align_pharaoh))
            logger.warning("Bad alignement line exists. Please check file!")
            raise
        flatten_align_idx.append([int(src_idx), int(tgt_idx)])
    return flatten_align_idx


def process(task, bucket, **kwargs):
    """Returns valid transformed bucket from bucket."""
    _, transform, cid = bucket[0]
    # We apply the same TransformPipe to all the bucket
    processed_bucket = transform.batch_apply(
       bucket, is_train=(task == CorpusTask.TRAIN), corpus_name=cid)
    if processed_bucket:
        for i in range(len(processed_bucket)):
            (example, transform, cid) = processed_bucket[i]
            example.clean()
            processed_bucket[i] = example
        return processed_bucket
    else:
        return None


def tensorify(vocabs, minibatch):
    """
    This function transforms a batch of Examples in tensors

    Returns  Dict of batch Tensors
        {'src': [batchsize, seq_len, n_feats+1],
         'tgt' : [batchsize, seq_len, n_feats+1],
         'indices' : [batchsize],
         'srclen': [batchsize],
         'tgtlen': [batchsize],
         'align': alignment sparse tensor
        }
    """
    tensor_batch = {}
    tbatchsrc = [torch.LongTensor(ex.src_ids) for ex in minibatch]
    padidx = vocabs['src'][DefaultTokens.PAD]
    tbatchsrc = pad_sequence(tbatchsrc, batch_first=True,
                             padding_value=padidx)

    tbatchfs = [tbatchsrc]
    if minibatch[0].src_feats is not None:
        for feat_id in range(len(minibatch[0].src_feats_ids)):
            tbatchfeat = [torch.LongTensor(ex.src_feats_ids[feat_id])
                          for ex in minibatch]
            padidx = vocabs['src_feats'][feat_id][DefaultTokens.PAD]
            tbatchfeat = pad_sequence(tbatchfeat, batch_first=True,
                                      padding_value=padidx)
            tbatchfs.append(tbatchfeat)
    tbatchsrc = torch.stack(tbatchfs, dim=2)
    tensor_batch['src'] = tbatchsrc

    tensor_batch['indices'] = \
        torch.LongTensor([ex.index for ex in minibatch])
    tensor_batch['srclen'] = \
        torch.LongTensor([len(ex.src_ids) for ex in minibatch])

    if minibatch[0].tgt is not None:
        tbatchtgt = [torch.LongTensor(ex.tgt_ids) for ex in minibatch]
        padidx = vocabs['tgt'][DefaultTokens.PAD]
        tbatchtgt = pad_sequence(tbatchtgt, batch_first=True,
                                 padding_value=padidx)
        tensor_batch['tgtlen'] = \
            torch.LongTensor([len(ex.tgt_ids) for ex in minibatch])

        tbatchfs = [tbatchtgt]
        if minibatch[0].tgt_feats is not None:
            for feat_id in range(len(minibatch[0].tgt_feats_ids)):
                tbatchfeat = [torch.LongTensor(ex.tgt_feats_ids[feat_id])
                              for ex in minibatch]
                padidx = vocabs['tgt_feats'][feat_id][DefaultTokens.PAD]
                tbatchfeat = pad_sequence(tbatchfeat, batch_first=True,
                                          padding_value=padidx)
                tbatchfs.append(tbatchfeat)
        tbatchtgt = torch.stack(tbatchfs, dim=2)
        tensor_batch['tgt'] = tbatchtgt

    if minibatch[0].align is not None:
        sparse_idx = []
        for i, ex in enumerate(minibatch):
            for src, tgt in parse_align_idx(ex.align):
                sparse_idx.append([i, tgt + 1, src])
        tbatchalign = torch.LongTensor(sparse_idx)
        tensor_batch['align'] = tbatchalign

    if minibatch[0].src_map is not None:
        src_vocab_size = max([max(ex.src_map) for ex in minibatch]) + 1
        src_map = torch.zeros(len(tensor_batch['srclen']),
                              tbatchsrc.size(1),
                              src_vocab_size)
        for i, ex in enumerate(minibatch):
            for j, t in enumerate(ex.src_map):
                src_map[i, j, t] = 1
        tensor_batch['src_map'] = src_map

    if minibatch[0].alignment is not None:
        alignment = torch.zeros(len(tensor_batch['srclen']),
                                tbatchtgt.size(1)).long()
        for i, ex in enumerate(minibatch):
            alignment[i, :len(ex.alignment)] = \
                torch.LongTensor(ex.alignment)
        tensor_batch['alignment'] = alignment

    if minibatch[0].src_ex_vocab:
        tensor_batch['src_ex_vocab'] = \
            [ex.src_ex_vocab for ex in minibatch]

    return tensor_batch


def text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if ex.tgt is not None:
        return max(len(ex.src_ids), len(ex.tgt_ids))
    return len(ex.src_ids)


def textbatch_to_tensor(vocabs, batch, is_train=False):
    """
    This is a hack to transform a simple batch of texts
    into a tensored batch to pass through _translate()
    """
    numeric = []
    infer_iter = []
    for i, ex in enumerate(batch):
        if isinstance(ex, bytes):
            ex = ex.decode("utf-8")
        if is_train:
            toks = ex
        else:
            toks = ex.strip("\n").split()
        example = Example(toks, toks)
        example.add_index(i)
        example.numericalize(vocabs)
        numeric.append(example)

    numeric.sort(key=text_sort_key, reverse=True)
    infer_iter = [tensorify(vocabs, numeric)]
    return infer_iter
