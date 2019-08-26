"""
This file is lifted from huggingface and adapted for onmt structure.
Ref in https://github.com/huggingface/pytorch-transformers/.
"""
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve

from random import random, randrange, randint, shuffle, choice
from onmt.utils.bert_tokenization import BertTokenizer, \
     PRETRAINED_VOCAB_ARCHIVE_MAP
from onmt.utils.file_utils import cached_path
from preprocess_bert_new import build_vocab_from_tokenizer
import numpy as np
import json
from onmt.inputters.inputter import get_bert_fields
from onmt.inputters.dataset_bert import BertDataset, truncate_seq_pair
import torch
import collections


class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(
                str(self.document_shelf_filepath), flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        # Uses the current_idx to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs
            # proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If sentence weighting is False, chose doc equally
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq,
                                 whole_word_mask, vocab_dict):
    """Creates the predictions for the masked LM. This is mostly copied from
    the Huggingface BERT repo, but pregenerate lm_labels_ids."""
    vocab_list = list(vocab_dict.keys())
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any
        # subsequence tokens are prefixed with ##. So whenever we see the ##,
        # we append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (whole_word_mask and len(cand_indices) >= 1
                and token.startswith("##")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_list)
            masked_lms.append(MaskedLmInstance(index=index,
                                               label=tokens[index]))
            # Replace true token with masked token
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]
    lm_labels_ids = [-1 for _ in tokens]
    for (i, token) in zip(mask_indices, masked_token_labels):
        lm_labels_ids[i] = vocab_dict[token]
    assert len(lm_labels_ids) == len(tokens)
    return tokens, mask_indices, masked_token_labels, lm_labels_ids


def create_instances_from_document(
        doc_database, doc_idx, vocab_dict, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, whole_word_mask):
    """This code is mostly a duplicate of the equivalent function from
    HuggingFace BERT's repo. But we use lm_labels_ids rather than
    mask_indices and masked_token_labels."""
    document = doc_database[doc_idx]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by user's
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into
                # `A` (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = randrange(1, len(current_chunk))

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []

                # Random next
                if len(current_chunk) == 1 or random() < 0.5:
                    is_next = False
                    target_b_length = target_seq_length - len(tokens_a)

                    # Sample a random document with longer docs being
                    # sampled more frequently
                    random_document = doc_database.sample_doc(
                        current_idx=doc_idx, sentence_weighted=True)

                    random_start = randrange(0, len(random_document))
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we
                    # "put them back" so they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_next = True
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + \
                    tokens_b + ["[SEP]"]
                segment_ids = [0 for _ in range(len(tokens_a) + 2)] + \
                              [1 for _ in range(len(tokens_b) + 1)]

                tokens, _, _, lm_labels_ids = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq,
                    whole_word_mask, vocab_dict)

                instance = {
                    "tokens": tokens,
                    "segment_ids": segment_ids,
                    "is_next": is_next,
                    # "masked_lm_positions": masked_lm_positions,
                    # "masked_lm_labels": masked_lm_labels,
                    "lm_labels_ids": lm_labels_ids}
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def build_document_database(input_file, tokenizer, reduce_memory):
    with DocumentDatabase(reduce_memory=reduce_memory) as docs:
        with input_file.open() as f:
            doc = []
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                line = line.strip()
                if line == "":
                    docs.add_document(doc)
                    doc = []
                else:
                    tokens = tokenizer.tokenize(line)
                    doc.append(tokens)
            if len(doc) != 0:  # If didn't end on a newline, still add
                docs.add_document(doc)
        if len(docs) <= 1:
            exit("""ERROR: No document breaks were found in the input file!
                 These are necessary to ensure that random NextSentences
                 are not sampled from the same document. Please add blank
                 lines to indicate breaks between documents in your file.
                 If your dataset does not contain multiple documents,
                 blank lines can be inserted at any natural boundary,
                 such as the ends of chapters, sections or paragraphs.""")
        return docs


def create_instances_from_docs(doc_database, vocab_dict, args):
    docs_instances = []
    for doc_idx in trange(len(doc_database), desc="Document"):
        doc_instances = create_instances_from_document(
            doc_database, doc_idx, vocab_dict=vocab_dict,
            max_seq_length=args.max_seq_len,
            short_seq_prob=args.short_seq_prob,
            masked_lm_prob=args.masked_lm_prob,
            max_predictions_per_seq=args.max_predictions_per_seq,
            whole_word_mask=args.do_whole_word_mask)
        docs_instances.extend(doc_instances)
    return docs_instances


def save_data_as_json(instances, json_name):
    instances_json = [json.dumps(instance) for instance in instances]
    num_instances = 0
    with open(json_name, 'w') as json_file:
        for instance in instances_json:
            json_file.write(instance + '\n')
            num_instances += 1
    return num_instances


def _get_parser():
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", type=str, default="dataset")
    parser.add_argument('--corpus_type', type=str, default="train",
                        choices=['train', 'valid'],
                        help="Choose from ['train', 'valid'], " +
                        "Vocab file will be generate if `train`")
    parser.add_argument("--vocab_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased",
                                 "bert-base-cased", "bert-large-cased",
                                 "bert-base-multilingual-uncased",
                                 "bert-base-multilingual-cased",
                                 "bert-base-chinese",
                                 "bert-base-german-cased",
                                 "bert-large-uncased-whole-word-masking",
                                 "bert-large-cased-whole-word-masking",
                                 "bert-base-cased-finetuned-mrpc"],
                        help="Pretrained vocab model use to tokenizer text.")

    parser.add_argument("--do_lower_case",  action="store_true")
    parser.add_argument("--do_whole_word_mask", action="store_true",
                        help="Whether to use whole word masking.")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="""Reduce memory usage for large datasets
                        by keeping data on disc rather than in memory""")

    parser.add_argument("--epochs_to_generate", type=int, default=2,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Prob. of a short sentence as training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Prob. of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Max number of tokens to mask in each sequence")
    parser.add_argument("--save_json", action="store_true",
                        help='save a copy of data in json form.')
    return parser


def main(args):
    tokenizer = BertTokenizer.from_pretrained(
        args.vocab_model, do_lower_case=args.do_lower_case)

    docs = build_document_database(
        args.input_file, tokenizer, args.reduce_memory)

    fields = get_bert_fields()
    vocab_dict = tokenizer.vocab
    args.output_dir.mkdir(exist_ok=True)

    # Build file corpus.pt
    for epoch in trange(args.epochs_to_generate, desc="Epoch"):
        docs_instances = create_instances_from_docs(docs, vocab_dict, args)

        # build BertDataset from instances collected from different document
        dataset = BertDataset(fields, docs_instances)
        epoch_filename = args.output_dir / "{}.{}.{}.pt".format(
            args.output_name, args.corpus_type, epoch)
        dataset.save(epoch_filename)
        print("output file {}, num_example {}, max_seq_len {}".format(
            epoch_filename, len(docs_instances), args.max_seq_len))

        if args.save_json:
            json_name = args.output_dir / "{}.{}.{}.json".format(
                args.output_name, args.corpus_type, epoch)
            num_instances = save_data_as_json(docs_instances, json_name)
            metrics_file = args.output_dir / "{}.{}.{}.metrics.json".format(
                args.output_name, args.corpus_type, epoch)
            with metrics_file.open('w') as metrics_file:
                metrics = {
                    "num_training_examples": num_instances,
                    "max_seq_len": args.max_seq_len
                }
                metrics_file.write(json.dumps(metrics))

    # Build file Vocab.pt
    if args.corpus_type == "train":
        vocab_file_url = PRETRAINED_VOCAB_ARCHIVE_MAP[args.vocab_model]
        vocab_dir = Path.joinpath(args.output_dir,
                                  "%s-vocab.txt" % (args.vocab_model))
        cached_vocab = cached_path(vocab_file_url, cache_dir=vocab_dir)
        print("Vocab file is Cached at %s." % cached_vocab)
        fields_vocab = build_vocab_from_tokenizer(
            fields, tokenizer, None)
        bert_vocab_file = Path.joinpath(args.output_dir,
                                        "%s.vocab.pt" % (args.output_name))
        print("Build Fields Vocab file.")
        torch.save(fields_vocab, bert_vocab_file)


if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()
    main(args)
