"""Transforms relate to noising from BART: based on code of fairseq."""
import math
import numpy as np
import torch

from typing import Sequence, Callable
from onmt.constants import DefaultTokens, SubwordMarker
from onmt.transforms import register_transform
from .transform import Transform


def _subword_start_by_joiner(tokens: Sequence[str]) -> Sequence[bool]:
    """Find word start in a subword list marked by joiner."""
    flag = [True] * len(tokens)
    for i, token in enumerate(tokens):
        if token.startswith(SubwordMarker.JOINER) and i != 0:
            flag[i] = False
        if token.endswith(SubwordMarker.JOINER):
            try:
                flag[i+1] = False
            except IndexError:
                print("Sentence `{}` not correct!".format(" ".join(token)))
                raise
    return flag


def _subword_start_by_spacer(tokens: Sequence[str]) -> Sequence[bool]:
    """Find word start in a subword list marked by spacer(as prefix)."""
    flag = [x.startswith(SubwordMarker.SPACER) for x in tokens]
    flag[0] = True
    return flag


def word_start_finder(ignore_subword=False, is_joiner=False) -> Callable:
    """Return callable to find all word start in the token list."""
    if not ignore_subword:
        if is_joiner:
            return _subword_start_by_joiner
        else:
            return _subword_start_by_spacer
    else:
        return lambda tokens: [True] * len(tokens)


class BARTNoising(object):
    """Noise from BART."""

    def __init__(self, vocab, mask_tok=DefaultTokens.MASK, mask_ratio=0.0,
                 insert_ratio=0.0, permute_sent_ratio=0.0, poisson_lambda=3.0,
                 replace_length=-1, rotate_ratio=0.0, mask_length='subword',
                 random_ratio=0.0, is_joiner=False,
                 full_stop_token=DefaultTokens.SENT_FULL_STOPS):
        self.vocab = vocab

        self.mask_tok = mask_tok

        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.insert_ratio = insert_ratio
        self.rotate_ratio = rotate_ratio
        self.permute_sent_ratio = permute_sent_ratio

        self.full_stop_token = full_stop_token

        # -1: keep everything (i.e. 1 mask per token)
        #  0: replace everything (i.e. no mask)
        #  1: 1 mask per span
        if replace_length not in [-1, 0, 1]:
            raise ValueError(f'invalid arg: replace_length={replace_length}')
        self.replace_length = replace_length

        if mask_length not in ['subword', 'word', 'span-poisson']:
            raise ValueError(f'invalid arg: mask-length={mask_length}')
        if mask_length == 'subword' and replace_length not in [0, 1]:
            raise ValueError('if using subwords, use replace-length=1 or 0')

        if mask_length == 'subword' or is_joiner is None:
            # view each subword as word start / input is word level token
            self._is_word_start = word_start_finder(ignore_subword=True)
        else:
            self._is_word_start = word_start_finder(is_joiner=is_joiner)

        self.mask_span_distribution = None
        if mask_length == 'span-poisson':
            self.mask_span_distribution = self._make_poisson(poisson_lambda)
        self.mask_length = mask_length
        self.poisson_lambda = poisson_lambda

    def _make_poisson(self, poisson_lambda):
        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-poisson_lambda)
        k_factorial = 1
        ps = []
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= poisson_lambda
            k_factorial *= (k + 1)
            if ps[-1] < 0.0000001:
                break
        ps = torch.FloatTensor(ps)
        return torch.distributions.Categorical(ps)

    def _is_full_stop(self, token):
        return True if token in self.full_stop_token else False

    def permute_sentences(self, tokens, p=1.0):
        if len(tokens) == 1:
            return tokens
        full_stops = np.array([self._is_full_stop(token) for token in tokens])
        # Pretend it ends with a full stop so last span is a sentence
        full_stops[-1] = True

        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero()[0] + 2

        n_sentences = sentence_ends.size
        if n_sentences == 1:
            return tokens

        n_to_permute = math.ceil((n_sentences * 2 * p) / 2.0)

        substitutions = np.random.permutation(n_sentences)[:n_to_permute]
        ordering = np.arange(0, n_sentences)
        ordering[substitutions] = substitutions[np.random.permutation(
            n_to_permute)]

        result = [tok for tok in tokens]
        index = 0
        for i in ordering:
            sentence = tokens[(sentence_ends[i - 1] if i > 0 else 0):
                              sentence_ends[i]]
            result[index:index + len(sentence)] = sentence
            index += len(sentence)
        assert len(result) == len(tokens), "Error when permute sentences."
        return result

    def whole_word_mask(self, tokens, p=1.0):  # text span mask/infilling
        is_word_start = torch.tensor(self._is_word_start(tokens)).int()
        n_mask = int(math.ceil(is_word_start.sum() * p))
        n_insert = 0
        if n_mask == 0:
            return tokens

        if self.mask_span_distribution is not None:  # Text (span) Infilling
            lengths = self.mask_span_distribution.sample(
                sample_shape=(n_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < n_mask:
                lengths = torch.cat([
                    lengths,
                    self.mask_span_distribution.sample(
                        sample_shape=(n_mask,))
                ], dim=0)
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < n_mask:
                i += 1
            lengths[i] = n_mask - (0 if i == 0 else cum_length[i - 1])
            n_mask = i + 1
            lengths = lengths[:n_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            n_insert = n_mask - lengths.size(0)
            n_mask -= n_insert
            if n_mask == 0:
                return self.insertion_noise(tokens, n_insert / len(tokens))

            assert (lengths > 0).all()
        else:  # Token Masking
            lengths = torch.ones((n_mask,)).long()
        # assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:n_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(n_mask).uniform_() < self.random_ratio

        tokens_length = len(tokens)
        # assert tokens_length - 1 not in indices
        to_keep = torch.ones(tokens_length, dtype=torch.bool)

        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            for i in indices.tolist():
                tokens[i] = self.mask_tok
            random_tok_ids = torch.randint(
                0, len(self.vocab), size=(mask_random.sum(),)).tolist()
            for i, rid in zip(indices[mask_random].tolist(), random_tok_ids):
                tokens[i] = self.vocab[rid]

        if tokens_length - 1 in indices:
            uncompleted = (indices != tokens_length - 1)
            indices = indices[uncompleted]
            mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]

        # acts as a long length, so spans don't go over the end of doc
        is_word_start[-1] = 255

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1  # 1 for the position already masked
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                # next position from each word_start
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token: 1 mask/remove per span
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]: 1 mask per token
                    for i in indices.tolist():
                        tokens[i] = self.mask_tok
                    random_tok_ids = torch.randint(
                        0, len(self.vocab), size=(mask_random.sum(),)).tolist()
                    for i, rid in zip(
                            indices[mask_random].tolist(), random_tok_ids):
                        tokens[i] = self.vocab[rid]
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                # to cover whole token
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    for i in indices.tolist():
                        tokens[i] = self.mask_tok
                    random_tok_ids = torch.randint(
                        0, len(self.vocab), size=(mask_random.sum(),)).tolist()
                    for i, rid in zip(
                            indices[mask_random].tolist(), random_tok_ids):
                        tokens[i] = self.vocab[rid]

                # assert tokens_length - 1 not in indices

        tokens = [tok for tok, keep in zip(tokens, to_keep.tolist())
                  if keep is True]

        if n_insert > 0:
            tokens = self.insertion_noise(tokens, n_insert / len(tokens))

        return tokens

    def insertion_noise(self, tokens, p=1.0):
        n_tokens = len(tokens)
        n_insert = math.ceil(n_tokens * p)
        if n_insert == 0:
            return tokens
        n_random = math.ceil(n_insert * self.random_ratio)

        noise_indices = np.random.permutation(n_tokens + n_insert)[:n_insert]
        noise_mask = np.zeros(shape=(n_tokens + n_insert,), dtype=bool)
        noise_mask[noise_indices] = 1

        result = np.empty(shape=(n_tokens + n_insert,), dtype=object)
        result[noise_indices[n_random:]] = self.mask_tok
        if n_random > 0:
            result[noise_indices[:n_random]] = np.random.choice(
                self.vocab, size=n_random)
        result[~noise_mask] = tokens

        assert all([item is not None for item in result]),\
            "Error when inserting noise."
        return result.tolist()

    def rolling_noise(self, tokens, p=1.0):
        if np.random.random() >= p:
            return tokens
        offset = np.random.randint(0, max(1, len(tokens) - 1) + 1)
        return tokens[offset:] + tokens[0:offset]

    def apply(self, tokens):
        if self.vocab is None:
            raise ValueError("Inject BART noise requires a valid vocabulary.")

        if self.permute_sent_ratio > 0.0:
            tokens = self.permute_sentences(tokens, self.permute_sent_ratio)

        if self.mask_ratio > 0.0:
            tokens = self.whole_word_mask(tokens, self.mask_ratio)

        if self.insert_ratio > 0.0:
            tokens = self.insertion_noise(tokens, self.insert_ratio)

        if self.rotate_ratio > 0.0:
            tokens = self.rolling_noise(tokens, self.rotate_ratio)
        return tokens

    def __repr__(self):
        cls_name = type(self).__name__
        kwargs = {}
        if self.permute_sent_ratio > 0.0:
            kwargs['permute_sent_ratio'] = self.permute_sent_ratio
            kwargs['full_stop_token'] = self.full_stop_token
        if self.insert_ratio > 0.0:
            kwargs['insert_ratio'] = self.insert_ratio
        if self.rotate_ratio > 0.0:
            kwargs['rotate_ratio'] = self.rotate_ratio
        if self.random_ratio > 0.0:
            kwargs['random_ratio'] = self.random_ratio
        if self.mask_ratio > 0.0:
            kwargs['mask_ratio'] = self.mask_ratio
            kwargs['mask_length'] = self.mask_length
            kwargs['poisson_lambda'] = self.poisson_lambda
            kwargs['replace_length'] = self.replace_length
        cls_args = ', '.join(
            [f'{kw}={arg}' for kw, arg in kwargs.items()])
        return '{}({})'.format(cls_name, cls_args)


@register_transform(name='bart')
class BARTNoiseTransform(Transform):
    def __init__(self, opts):
        super().__init__(opts)

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to BART."""
        group = parser.add_argument_group("Transform/BART")
        group.add("--permute_sent_ratio", "-permute_sent_ratio",
                  type=float, default=0.0,
                  help="Permute this proportion of sentences "
                       "(boundaries defined by {}) in all inputs.".format(
                        DefaultTokens.SENT_FULL_STOPS))
        group.add("--rotate_ratio", "-rotate_ratio", type=float, default=0.0,
                  help="Rotate this proportion of inputs.")
        group.add("--insert_ratio", "-insert_ratio", type=float, default=0.0,
                  help="Insert this percentage of additional random tokens.")
        group.add("--random_ratio", "-random_ratio", type=float, default=0.0,
                  help="Instead of using {}, use random token "
                       "this often.".format(DefaultTokens.MASK))

        group.add("--mask_ratio", "-mask_ratio", type=float, default=0.0,
                  help="Fraction of words/subwords that will be masked.")
        group.add("--mask_length", "-mask_length", type=str, default="subword",
                  choices=["subword", "word", "span-poisson"],
                  help="Length of masking window to apply.")
        group.add("--poisson_lambda", "-poisson_lambda",
                  type=float, default=3.0,
                  help="Lambda for Poisson distribution to sample span length "
                       "if `-mask_length` set to span-poisson.")
        group.add("--replace_length", "-replace_length",
                  type=int, default=-1, choices=[-1, 0, 1],
                  help="When masking N tokens, replace with 0, 1, "
                       "or N tokens. (use -1 for N)")

    def warm_up(self, vocabs):
        super().warm_up(None)
        if vocabs is None:
            self.bart_noise = None
            return
        self.vocabs = vocabs

        subword_type = self.opts.src_subword_type
        if self.opts.mask_length == 'subword':
            if subword_type == 'none':
                raise ValueError(
                    f'src_subword_type={subword_type} incompatible with '
                    f'mask_length={self.opts.mask_length}!')
        is_joiner = (subword_type == 'bpe') if subword_type != 'none' else None
        self.bart_noise = BARTNoising(
            self.vocabs['src'].itos,
            mask_tok=DefaultTokens.MASK,
            mask_ratio=self.opts.mask_ratio,
            insert_ratio=self.opts.insert_ratio,
            permute_sent_ratio=self.opts.permute_sent_ratio,
            poisson_lambda=self.opts.poisson_lambda,
            replace_length=self.opts.replace_length,
            rotate_ratio=self.opts.rotate_ratio,
            mask_length=self.opts.mask_length,
            random_ratio=self.opts.random_ratio,
            is_joiner=is_joiner
        )

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply BART noise to src side tokens."""
        if is_train and self.vocabs is not None:
            src = self.bart_noise.apply(example['src'])
            example['src'] = src
        return example

    def _repr_args(self):
        """Return str represent key arguments for BART."""
        return repr(self.bart_noise)
