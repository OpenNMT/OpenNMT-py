import torch
import pyonmttok
from onmt.constants import DefaultTokens, CorpusTask, ModelTask
from torch.nn.utils.rnn import pad_sequence
from onmt.utils.logging import logger
from collections import Counter


def text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if ex['tgt']:
        return len(ex['src']['src_ids']), len(ex['tgt']['tgt_ids'])
    return len(ex['src']['src_ids'])


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch, max_tgt_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new['src']['src_ids']) + 2)
    src_elements = count * max_src_in_batch
    # Tgt: [w1 ... wM <eos>]
    if new['tgt'] is not None:
        max_tgt_in_batch = max(max_tgt_in_batch,
                               len(new['tgt']['tgt_ids']) + 1)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def process(task, item):
    """Return valid transformed example from `item`."""
    example, transform, cid = item
    # this is a hack: appears quicker to apply it here
    # than in the ParallelCorpusIterator
    maybe_example = transform.apply(example,
                                    is_train=(task
                                              == CorpusTask.TRAIN),
                                    corpus_name=cid)
    if maybe_example is None:
        return None

    maybe_example['src'] = {"src": ' '.join(maybe_example['src'])}

    # Make features part of src like
    # {'src': {'src': ..., 'feat1': ...., 'feat2': ....}}
    if 'src_feats' in maybe_example:
        for feat_name, feat_value in maybe_example['src_feats'].items():
            maybe_example['src'][feat_name] = ' '.join(feat_value)
        del maybe_example['src_feats']
    if maybe_example['tgt'] is not None:
        maybe_example['tgt'] = {'tgt': ' '.join(maybe_example['tgt'])}
    if 'align' in maybe_example:
        maybe_example['align'] = ' '.join(maybe_example['align'])

    # at this point an example looks like:
    # {'src': {'src': ..., 'feat1': ...., 'feat2': ....},
    #  'tgt': {'tgt': ...},
    #  'src_original': ['tok1', ...'tokn'],
    #  'tgt_original': ['tok1', ...'tokm'],
    #  'indices' : seq in bucket
    #  'align': ...,
    # }

    return maybe_example


def numericalize(vocabs, example):
    """
    """
    numeric = example
    numeric['src']['src_ids'] = []
    if vocabs['data_task'] == ModelTask.SEQ2SEQ:
        src_text = example['src']['src'].split()
        numeric['src']['src_ids'] = vocabs['src'](src_text)
        if example['tgt'] is not None:
            numeric['tgt']['tgt_ids'] = []
            tgt_text = example['tgt']['tgt'].split()
            numeric['tgt']['tgt_ids'] = \
                vocabs['tgt']([DefaultTokens.BOS] + tgt_text
                              + [DefaultTokens.EOS])

    elif vocabs['data_task'] == ModelTask.LANGUAGE_MODEL:
        src_text = example['src']['src'].split()
        numeric['src']['src_ids'] = \
            vocabs['src']([DefaultTokens.BOS] + src_text)
        if example['tgt'] is not None:
            numeric['tgt']['tgt_ids'] = []
            tgt_text = example['tgt']['tgt'].split()
            numeric['tgt']['tgt_ids'] = \
                vocabs['tgt'](tgt_text + [DefaultTokens.EOS])
    else:
        raise ValueError(
                f"Something went wrong with task {vocabs['data_task']}"
        )

    if 'src_feats' in vocabs.keys():
        for featname in vocabs['src_feats'].keys():
            src_feat = example['src'][featname].split()
            vf = vocabs['src_feats'][featname]
            # we'll need to change this if we introduce tgt feat
            numeric['src'][featname] = vf(src_feat)

    return numeric


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


def tensorify(vocabs, minibatch):
    """
    This function transforms a batch of example in tensors
    Each example looks like
    {'src': {'src': ..., 'feat1': ..., 'feat2': ..., 'src_ids': ...},
     'tgt': {'tgt': ..., 'tgt_ids': ...},
     'src_original': ['tok1', ...'tokn'],
     'tgt_original': ['tok1', ...'tokm'],
     'indices' : seq in bucket
     'align': ...,
    }
    Returns  Dict of batch Tensors
        {'src': [seqlen, batchsize, n_feats],
         'tgt' : [seqlen, batchsize, n_feats=1],
         'indices' : [batchsize],
         'srclen': [batchsize],
         'tgtlen': [batchsize],
         'align': alignment sparse tensor
        }
    """
    tensor_batch = {}
    tbatchsrc = [torch.LongTensor(ex['src']['src_ids']) for ex in minibatch]
    padidx = vocabs['src'][DefaultTokens.PAD]
    tbatchsrc = pad_sequence(tbatchsrc, batch_first=True,
                             padding_value=padidx)
    if len(minibatch[0]['src'].keys()) > 2:
        tbatchfs = [tbatchsrc]
        for feat in minibatch[0]['src'].keys():
            if feat not in ['src', 'src_ids']:
                tbatchfeat = [torch.LongTensor(ex['src'][feat])
                              for ex in minibatch]
                padidx = vocabs['src_feats'][feat][DefaultTokens.PAD]
                tbatchfeat = pad_sequence(tbatchfeat, batch_first=True,
                                          padding_value=padidx)
                tbatchfs.append(tbatchfeat)
        tbatchsrc = torch.stack(tbatchfs, dim=2)
    else:
        tbatchsrc = tbatchsrc[:, :, None]
    # Need to add features in last dimensions

    tensor_batch['src'] = tbatchsrc
    tensor_batch['indices'] = torch.LongTensor([ex['indices']
                                                for ex in minibatch])
    tensor_batch['srclen'] = torch.LongTensor([len(ex['src']['src_ids'])
                                               for ex in minibatch])

    if minibatch[0]['tgt'] is not None:
        tbatchtgt = [torch.LongTensor(ex['tgt']['tgt_ids'])
                     for ex in minibatch]
        padidx = vocabs['tgt'][DefaultTokens.PAD]
        tbatchtgt = pad_sequence(tbatchtgt, batch_first=True,
                                 padding_value=padidx)
        tbatchtgt = tbatchtgt[:, :, None]
        tbatchtgtlen = torch.LongTensor([len(ex['tgt']['tgt_ids'])
                                         for ex in minibatch])
        tensor_batch['tgt'] = tbatchtgt
        tensor_batch['tgtlen'] = tbatchtgtlen

    if 'align' in minibatch[0].keys() and minibatch[0]['align'] is not None:
        sparse_idx = []
        for i, ex in enumerate(minibatch):
            for src, tgt in parse_align_idx(ex['align']):
                sparse_idx.append([i, tgt + 1, src])
        tbatchalign = torch.LongTensor(sparse_idx)
        tensor_batch['align'] = tbatchalign

    if 'src_map' in minibatch[0].keys():
        src_vocab_size = max([max(ex['src_map']) for ex in minibatch]) + 1
        src_map = torch.zeros(len(tensor_batch['srclen']),
                              tbatchsrc.size(1),
                              src_vocab_size)
        for i, ex in enumerate(minibatch):
            for j, t in enumerate(ex['src_map']):
                src_map[i, j, t] = 1
        tensor_batch['src_map'] = src_map

    if 'alignment' in minibatch[0].keys():
        alignment = torch.zeros(len(tensor_batch['srclen']),
                                tbatchtgt.size(1)).long()
        for i, ex in enumerate(minibatch):
            alignment[i, :len(ex['alignment'])] = \
                torch.LongTensor(ex['alignment'])
        tensor_batch['alignment'] = alignment

    if 'src_ex_vocab' in minibatch[0].keys():
        tensor_batch['src_ex_vocab'] = [ex['src_ex_vocab']
                                        for ex in minibatch]

    return tensor_batch


def textbatch_to_tensor(vocabs, batch):
    """
    This is a hack to transform a simple batch of texts
    into a tensored batch to pass through _translate()
    """
    numeric = []
    infer_iter = []
    for i, ex in enumerate(batch):
        if isinstance(ex, bytes):
            ex = ex.decode("utf-8")
        toks = ex.strip("\n").split()
        idxs = vocabs['src'](toks)
        # Need to add features also in 'src'
        numeric.append({'src': {'src': ex.strip("\n").split(),
                                'src_ids': idxs},
                        'srclen': len(ex.strip("\n").split()),
                        'tgt': None,
                        'indices': i,
                        'align': None})
    numeric.sort(key=text_sort_key, reverse=True)
    infer_iter = [tensorify(vocabs, numeric)]
    return infer_iter


def _addcopykeys(vocabs, example):
    """Create copy-vocab and numericalize with it.
    In-place adds ``"src_map"`` to ``example``. That is the copy-vocab
    numericalization of the tokenized ``example["src"]``. If ``example``
    has a ``"tgt"`` key, adds ``"alignment"`` to example. That is the
    copy-vocab numericalization of the tokenized ``example["tgt"]``. The
    alignment has an initial and final UNK token to match the BOS and EOS
    tokens.
    Args:
        vocabs
        example (dict): An example dictionary with a ``"src"`` key and
            maybe a ``"tgt"`` key. (This argument changes in place!)
    Returns:
        ``example``, changed as described.
    """

    src = example['src']['src'].split()
    src_ex_vocab = pyonmttok.build_vocab_from_tokens(
        Counter(src),
        maximum_size=0,
        minimum_frequency=1,
        special_tokens=[DefaultTokens.UNK,
                        DefaultTokens.PAD,
                        DefaultTokens.BOS,
                        DefaultTokens.EOS])
    src_ex_vocab.default_id = src_ex_vocab[DefaultTokens.UNK]
    # make a small vocab containing just the tokens in the source sequence

    # Map source tokens to indices in the dynamic dict.
    example['src_map'] = src_ex_vocab(src)
    example['src_ex_vocab'] = src_ex_vocab

    if example['tgt'] is not None:
        if vocabs['data_task'] == ModelTask.SEQ2SEQ:
            tgt = [DefaultTokens.UNK] + example['tgt']['tgt'].split() \
                  + [DefaultTokens.UNK]
        elif vocabs['data_task'] == ModelTask.LANGUAGE_MODEL:
            tgt = example['tgt']['tgt'].split() \
                  + [DefaultTokens.UNK]
        example['alignment'] = src_ex_vocab(tgt)
    return example
