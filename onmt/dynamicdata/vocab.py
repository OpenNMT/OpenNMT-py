import collections
import gzip
import itertools
import os
import torch
import torchtext
import onmt.inputters


class SortedCounter(collections.Counter):
    """A Counter, with most_common replaced with a version
    that sorts results by count and key, rather than just count.
    The keys (counted elements) must be sortable."""
    def most_common(self, n=None, keep_ties=False):
        """List the n most common elements and their counts from the most
        common to the least.  If n is None, then list all element counts.
        Ties are sorted by element.
        If keep_ties is set to True, more than n results may be returned,
        if elements are tied for last place."""

        all_most_common = super().most_common()
        if n is None:
            return sorted(all_most_common,
                          key=lambda x: (-x[1], x[0]))
        i = n - 1
        try:
            i_count = all_most_common[i][1]
            while all_most_common[i][1] == i_count:
                i += 1
            all_most_common = all_most_common[:i]
        except IndexError:
            pass
        all_most_common = sorted(all_most_common,
                                 key=lambda x: (-x[1], x[0]))
        if keep_ties:
            return all_most_common
        return all_most_common[:n]


class Vocabulary():
    def __init__(self, data_config):
        self.data_config = data_config

    def add(self, task, tpl):
        raise NotImplementedError()

    def save_all(self, segmentation='words'):
        vocabdir = os.path.join(
            self.data_config['meta']['shard']['rootdir'],
            'transforms', segmentation)
        os.makedirs(vocabdir, exist_ok=True)
        for key in self.tokens:
            path = os.path.join(vocabdir, '{}.vocab'.format(key))
            self.save(key, path)

    def save(self, key, path):
        tokens = self.tokens[key]
        with open(path, 'w') as fobj:
            for w, c in tokens.most_common():
                fobj.write('{}\t{}\n'.format(c, w))

    def path(self, key, segmentation=None):
        if segmentation is None:
            segmentation = self.data_config['meta']['train']['name']
        path = os.path.join(
            self.data_config['meta']['shard']['rootdir'],
            'transforms', segmentation,
            '{}.vocab'.format(key))
        return path

    def load(self, path):
        counter = SortedCounter()
        with open(path, 'r') as fobj:
            seen_noncomment = False
            is_float = False
            for line in fobj:
                if line[0] == '#' and not seen_noncomment:
                    # skip comments in beginning of file
                    continue
                else:
                    seen_noncomment = True
                c, w = line.rstrip('\n').split(None, 1)
                if is_float:
                    c = int(float(c)) + 1
                else:
                    try:
                        c = int(c)
                    except ValueError:
                        is_float = True
                        c = int(float(c)) + 1
                counter[w] += c
        return counter


class SimpleSharedVocabulary(Vocabulary):
    """ Uses a single counter for all tasks and sides.
    Assumes space-based tokenization, no factors. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens = {'shared': SortedCounter()}

    def add(self, task, tpl):
        for line in tpl:
            self.tokens['shared'].update(line.split())


def no_tokenize(pretokenized):
    return pretokenized


def _build_field_vocab(field,
                       counter,
                       size_multiple=1,
                       specials=None,
                       **kwargs):
    # this is basically copy-pasted from torchtext via onmt.
    all_specials = [
        field.unk_token, field.pad_token, field.init_token, field.eos_token
    ]
    if specials is not None:
        all_specials.extend(list(specials))
    filtered_specials = [tok for tok in all_specials if tok is not None]
    field.vocab = field.vocab_cls(counter,
                                  specials=filtered_specials,
                                  **kwargs)
    if size_multiple > 1:
        onmt.inputters.inputter._pad_vocab_to_multiple(
            field.vocab, size_multiple)


def load_vocabulary(data_config):
    tokencounter = Vocabulary(data_config)
    counters = {}
    if data_config['meta']['shard']['share_vocab']:
        vocab_path = data_config['meta']['train']['vocab_path']
        counters['shared'] = tokencounter.load(vocab_path)
    else:
        raise NotImplementedError()
    return counters


def prepare_fields(data_config, counters, specials):
    # TODO: support for features
    src_nfeats = 0
    tgt_nfeats = 0
    fields = onmt.inputters.get_fields(
        'text',
        src_nfeats, tgt_nfeats)
    src_base_field = fields['src'].base_field
    tgt_base_field = fields['tgt'].base_field
    src_base_field.tokenize = no_tokenize
    tgt_base_field.tokenize = no_tokenize
    if data_config['meta']['shard']['share_vocab']:
        _build_field_vocab(tgt_base_field,
                           counters['shared'],
                           specials=specials)
        src_base_field.vocab = tgt_base_field.vocab
    else:
        raise NotImplementedError()
    return fields


def save_fields(data_config, fields):
    segmentation = data_config['meta']['train']['name']
    vocabdir = os.path.join(
        data_config['meta']['shard']['rootdir'],
        'transforms', segmentation)
    os.makedirs(vocabdir, exist_ok=True)
    path = os.path.join(
        vocabdir, 'fields.pt')
    if os.path.exists(path):
        raise Exception('Refusing to clobber existing {}'.format(path))
    with open(path, 'wb') as fobj:
        torch.save(fields, fobj)


def load_fields(data_config):
    segmentation = data_config['meta']['train']['name']
    path = os.path.join(
        data_config['meta']['shard']['rootdir'],
        'transforms', segmentation,
        'fields.pt')
    with open(path, 'rb') as fobj:
        fields = torch.load(fobj)
    return fields


def save_transforms(data_config, transform_models, transforms):
    segmentation = data_config['meta']['train']['name']
    path = os.path.join(
        data_config['meta']['shard']['rootdir'],
        'transforms', segmentation,
        'transforms.pt')
    if os.path.exists(path):
        raise Exception('Refusing to clobber existing {}'.format(path))
    with open(path, 'wb') as fobj:
        torch.save(transform_models, fobj)
        torch.save(transforms, fobj)


def load_transforms(data_config):
    segmentation = data_config['meta']['train']['name']
    path = os.path.join(
        data_config['meta']['shard']['rootdir'],
        'transforms', segmentation,
        'transforms.pt')
    with open(path, 'rb') as fobj:
        transform_models = torch.load(fobj)
        transforms = torch.load(fobj)
    return transform_models, transforms
