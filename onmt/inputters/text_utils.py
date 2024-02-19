import torch
import pyonmttok
from onmt.constants import DefaultTokens, CorpusTask, ModelTask
from torch.nn.utils.rnn import pad_sequence
from onmt.utils.logging import logger
from collections import Counter


def parse_features(line, n_feats=0, defaults=None):
    """
    Parses text lines with features appended to each token.
    Ex.: This￨A￨B is￨A￨A a￨C￨A test￨A￨B
    """
    text, feats = [], [[] for _ in range(n_feats)]
    check, count = 0, 0
    for token in line.split(" "):
        tok, *fts = token.strip("\n").split("￨")
        check += len(fts)
        count += 1
        if not fts and defaults is not None:
            if isinstance(defaults, str):
                defaults = defaults.split("￨")
            if n_feats > 0:
                assert len(defaults) == n_feats  # Security check
                fts = defaults
        assert len(fts) == n_feats, (
            f"The number of fetures does not match the "
            f"expected number of features. Found {len(fts)} "
            f"features in the data but {n_feats} were expected."
        )
        text.append(tok)
        for i in range(n_feats):
            feats[i].append(fts[i])
    # Check if all tokens have features or none at all
    assert (
        check == 0 or check == count * n_feats
    ), "Some tokens are missing features. Please check your data."
    feats = [" ".join(x) for x in feats] if n_feats > 0 else None
    return " ".join(text), feats


def append_features_to_text(text, features):
    """
    It appends features to subwords when dumping to file
    """
    text_tok = text.split(" ")
    feats_tok = [x.split(" ") for x in features]

    pretty_toks = []
    for tok, *feats in zip(text_tok, *feats_tok):
        feats = "￨".join(feats)
        if feats:
            pretty_toks.append(f"{tok}￨{feats}")
        else:
            pretty_toks.append(tok)
    return " ".join(pretty_toks)


def text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if ex["tgt"]:
        return len(ex["src"]["src_ids"]), len(ex["tgt"]["tgt_ids"])
    return len(ex["src"]["src_ids"])


def clean_example(maybe_example):
    maybe_example["src"] = {"src": " ".join(maybe_example["src"])}
    # Make features part of src like
    # {'src': {'src': ..., 'feats': [...., ....]}}
    if "src_feats" in maybe_example:
        maybe_example["src"]["feats"] = [
            " ".join(x) for x in maybe_example["src_feats"]
        ]
        del maybe_example["src_feats"]
    if maybe_example["tgt"] is not None:
        maybe_example["tgt"] = {"tgt": " ".join(maybe_example["tgt"])}
    if "align" in maybe_example:
        maybe_example["align"] = " ".join(maybe_example["align"])
    return maybe_example


def process(task, bucket, **kwargs):
    """Returns valid transformed bucket from bucket."""
    transform_cid_to_examples = {}
    for example in bucket:
        transform_cid = (example[1], example[2])
        if transform_cid not in transform_cid_to_examples:
            transform_cid_to_examples[transform_cid] = []
        transform_cid_to_examples[transform_cid].append(example)

    processed_bucket = []
    # careful below it will return a bucket sorted by corpora
    # but we sort by length later and shuffle batches
    for (transform, cid), sub_bucket in transform_cid_to_examples.items():
        transf_bucket = transform.batch_apply(
            sub_bucket, is_train=(task == CorpusTask.TRAIN), corpus_name=cid
        )
        for example, transform, cid in transf_bucket:
            example = clean_example(example)
            if len(example["src"]["src"]) > 0:
                processed_bucket.append(example)

        # at this point an example looks like:
        # {'src': {'src': ..., 'feats': [....]},
        #  'tgt': {'tgt': ...},
        #  'src_original': ['tok1', ...'tokn'],
        #  'tgt_original': ['tok1', ...'tokm'],
        #  'cid': corpus id
        #  'cid_line_number' : cid line number
        #  'align': ...,
        # }
    if len(processed_bucket) > 0:
        return processed_bucket
    else:
        return None


def numericalize(vocabs, example):
    """ """
    decoder_start_token = vocabs["decoder_start_token"]
    numeric = example
    numeric["src"]["src_ids"] = []
    if vocabs["data_task"] == ModelTask.SEQ2SEQ:
        src_text = example["src"]["src"].split(" ")
        numeric["src"]["src_ids"] = vocabs["src"](src_text)
        if example["tgt"] is not None:
            numeric["tgt"]["tgt_ids"] = []
            tgt_text = example["tgt"]["tgt"].split(" ")
            numeric["tgt"]["tgt_ids"] = vocabs["tgt"](
                [decoder_start_token] + tgt_text + [DefaultTokens.EOS]
            )

    elif vocabs["data_task"] == ModelTask.LANGUAGE_MODEL:
        src_text = example["src"]["src"].split(" ")
        if decoder_start_token != "":
            src_text = [decoder_start_token] + src_text
        numeric["src"]["src_ids"] = vocabs["src"](src_text)
        if example["tgt"] is not None:
            numeric["tgt"]["tgt_ids"] = []
            tgt_text = example["tgt"]["tgt"].split(" ")
            numeric["tgt"]["tgt_ids"] = vocabs["tgt"](tgt_text + [DefaultTokens.EOS])
            if decoder_start_token == "":
                numeric["tgt"]["tgt_ids"] = numeric["tgt"]["tgt_ids"][1:]
    else:
        raise ValueError(f"Something went wrong with task {vocabs['data_task']}")

    if "feats" in example["src"]:
        numeric_feats = []
        for fv, feat in zip(vocabs["src_feats"], example["src"]["feats"]):
            numeric_feats.append(fv(feat.split(" ")))
        numeric["src"]["feats"] = numeric_feats
    return numeric


def parse_align_idx(align_pharaoh):
    """
    Parse Pharaoh alignment into [[<src>, <tgt>], ...]
    """
    align_list = align_pharaoh.strip().split(" ")
    flatten_align_idx = []
    for align in align_list:
        try:
            src_idx, tgt_idx = align.split("-")
        except ValueError:
            logger.warning("{} in `{}`".format(align, align_pharaoh))
            logger.warning("Bad alignement line exists. Please check file!")
            raise
        flatten_align_idx.append([int(src_idx), int(tgt_idx)])
    return flatten_align_idx


def tensorify(vocabs, minibatch, device, left_pad=False):
    """
    This function transforms a batch of example in tensors
    Each example looks like
    {'src': {'src': ..., 'feats': [...], 'src_ids': ...},
     'tgt': {'tgt': ..., 'tgt_ids': ...},
     'src_original': ['tok1', ...'tokn'],
     'tgt_original': ['tok1', ...'tokm'],
     'cid': corpus id
     'cid_line_number' : corpus id line number
     'ind_in_bucket': index in bucket
     'align': ...,
    }
    Returns  Dict of batch Tensors
        {'src': [seqlen, batchsize, n_feats+1],
         'tgt' : [seqlen, batchsize, n_feats=1],
         'cid': [batchsize],
         'cid_line_number' : [batchsize],
         'ind_in_bucket': [batchsize],
         'srclen': [batchsize],
         'tgtlen': [batchsize],
         'align': alignment sparse tensor
        }
    """
    tensor_batch = {}
    if left_pad:
        tbatchsrc = [
            torch.tensor(ex["src"]["src_ids"], dtype=torch.long, device=device).flip(
                dims=[0]
            )
            for ex, indice in minibatch
        ]
    else:
        tbatchsrc = [
            torch.tensor(ex["src"]["src_ids"], dtype=torch.long, device=device)
            for ex, indice in minibatch
        ]
    padidx = vocabs["src"][DefaultTokens.PAD]
    tbatchsrc = pad_sequence(tbatchsrc, batch_first=True, padding_value=padidx)
    if "feats" in minibatch[0][0]["src"]:
        tbatchfs = [tbatchsrc]
        for feat_id in range(len(minibatch[0][0]["src"]["feats"])):
            if left_pad:
                tbatchfeat = [
                    torch.tensor(
                        ex["src"]["feats"][feat_id], dtype=torch.long, device=device
                    ).flip(dims=[0])
                    for ex, indice in minibatch
                ]
            else:
                tbatchfeat = [
                    torch.tensor(
                        ex["src"]["feats"][feat_id], dtype=torch.long, device=device
                    )
                    for ex, indice in minibatch
                ]
            padidx = vocabs["src_feats"][feat_id][DefaultTokens.PAD]
            tbatchfeat = pad_sequence(
                tbatchfeat, batch_first=True, padding_value=padidx
            )
            tbatchfs.append(tbatchfeat)
        tbatchsrc = torch.stack(tbatchfs, dim=2)
    else:
        # Need to add features in last dimensions
        tbatchsrc = tbatchsrc[:, :, None]

    if left_pad:
        tensor_batch["src"] = tbatchsrc.flip(dims=[1])
    else:
        tensor_batch["src"] = tbatchsrc

    tensor_batch["srclen"] = torch.tensor(
        [len(ex["src"]["src_ids"]) for ex, indice in minibatch],
        dtype=torch.long,
        device=device,
    )

    if minibatch[0][0]["tgt"] is not None:
        if left_pad:
            tbatchtgt = [
                torch.tensor(
                    ex["tgt"]["tgt_ids"], dtype=torch.long, device=device
                ).flip(dims=[0])
                for ex, indice in minibatch
            ]
        else:
            tbatchtgt = [
                torch.tensor(ex["tgt"]["tgt_ids"], dtype=torch.long, device=device)
                for ex, indice in minibatch
            ]

        padidx = vocabs["tgt"][DefaultTokens.PAD]
        tbatchtgt = pad_sequence(tbatchtgt, batch_first=True, padding_value=padidx)
        tbatchtgt = tbatchtgt[:, :, None]
        tbatchtgtlen = torch.tensor(
            [len(ex["tgt"]["tgt_ids"]) for ex, indice in minibatch],
            dtype=torch.long,
            device=device,
        )
        if left_pad:
            tensor_batch["tgt"] = tbatchtgt.flip(dims=[1])
        else:
            tensor_batch["tgt"] = tbatchtgt
        tensor_batch["tgtlen"] = tbatchtgtlen

    if "align" in minibatch[0][0].keys() and minibatch[0][0]["align"] is not None:
        sparse_idx = []
        for i, (ex, indice) in enumerate(minibatch):
            for src, tgt in parse_align_idx(ex["align"]):
                sparse_idx.append([i, tgt + 1, src])
        tbatchalign = torch.tensor(sparse_idx, dtype=torch.long, device=device)
        tensor_batch["align"] = tbatchalign

    if "src_map" in minibatch[0][0].keys():
        src_vocab_size = max([max(ex["src_map"]) for ex, indice in minibatch]) + 1
        src_map = torch.zeros(
            len(tensor_batch["srclen"]),
            tbatchsrc.size(1),
            src_vocab_size,
            device=device,
        )
        for i, (ex, indice) in enumerate(minibatch):
            for j, t in enumerate(ex["src_map"]):
                src_map[i, j, t] = 1
        tensor_batch["src_map"] = src_map

    if "alignment" in minibatch[0][0].keys():
        alignment = torch.zeros(
            len(tensor_batch["srclen"]),
            tbatchtgt.size(1),
            dtype=torch.long,
            device=device,
        )
        for i, (ex, indice) in enumerate(minibatch):
            alignment[i, : len(ex["alignment"])] = torch.tensor(
                ex["alignment"], dtype=torch.long, device=device
            )
        tensor_batch["alignment"] = alignment

    if "src_ex_vocab" in minibatch[0][0].keys():
        tensor_batch["src_ex_vocab"] = [ex["src_ex_vocab"] for ex, indice in minibatch]

    tensor_batch["ind_in_bucket"] = [indice for ex, indice in minibatch]

    tensor_batch["cid"] = [ex["cid"] for ex, indice in minibatch]
    tensor_batch["cid_line_number"] = [
        ex["cid_line_number"] for ex, indice in minibatch
    ]

    return tensor_batch


def textbatch_to_tensor(vocabs, batch, device, is_train=False):
    """
    This is a hack to transform a simple batch of texts
    into a tensored batch to pass through _translate()
    """
    numeric = []
    infer_iter = []
    for i, ex in enumerate(batch):
        # Keep it consistent with dynamic data
        ex["srclen"] = len(ex["src"]["src"].split(" "))
        ex["in_in_bucket"] = i
        ex["cid"] = "text"
        ex["cid_line_number"] = i
        ex["align"] = None
        numeric.append((numericalize(vocabs, ex), i))
    infer_iter = [(tensorify(vocabs, numeric, device), 0)]  # force bucket_idx to 0
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
    src = example["src"]["src"].split(" ")
    src_ex_vocab = pyonmttok.build_vocab_from_tokens(
        Counter(src),
        maximum_size=0,
        minimum_frequency=1,
        special_tokens=[
            DefaultTokens.UNK,
            DefaultTokens.PAD,
            DefaultTokens.BOS,
            DefaultTokens.EOS,
        ],
    )
    src_ex_vocab.default_id = src_ex_vocab[DefaultTokens.UNK]
    # make a small vocab containing just the tokens in the source sequence

    # Map source tokens to indices in the dynamic dict.
    example["src_map"] = src_ex_vocab(src)
    example["src_ex_vocab"] = src_ex_vocab

    if example["tgt"] is not None:
        if vocabs["data_task"] == ModelTask.SEQ2SEQ:
            tgt = (
                [DefaultTokens.UNK]
                + example["tgt"]["tgt"].split(" ")
                + [DefaultTokens.UNK]
            )
        elif vocabs["data_task"] == ModelTask.LANGUAGE_MODEL:
            tgt = example["tgt"]["tgt"].split(" ") + [DefaultTokens.UNK]
        example["alignment"] = src_ex_vocab(tgt)
    return example
