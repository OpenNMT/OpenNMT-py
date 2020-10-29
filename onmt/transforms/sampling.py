"""Transforms relate to hamming distance sampling."""
import random
import numpy as np
from onmt.utils.logging import logger
from onmt.constants import DefaultTokens
from onmt.transforms import register_transform
from .transform import Transform


class HammingDistanceSampling(object):
    """Functions related to (negative) Hamming Distance Sampling."""

    def _softmax(self, x):
        softmax = np.exp(x)/sum(np.exp(x))
        return softmax

    def _sample_replace(self, vocab, reject):
        """Sample a token from `vocab` other than `reject`."""
        token = reject
        while token == reject:
            token = random.choice(vocab)
        return token

    def _sample_distance(self, tokens, temperature):
        """Sample number of tokens to corrupt from `tokens`."""
        n_tokens = len(tokens)
        indices = np.arange(n_tokens)
        logits = indices * -1 * temperature
        probs = self._softmax(logits)
        distance = np.random.choice(indices, p=probs)
        return distance

    def _sample_position(self, tokens, distance):
        n_tokens = len(tokens)
        chosen_indices = random.sample(range(n_tokens), k=distance)
        return chosen_indices


class HammingDistanceSamplingTransform(Transform, HammingDistanceSampling):
    """Abstract Transform class based on HammingDistanceSampling."""

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        np.random.seed(seed)
        random.seed(seed)


@register_transform(name='switchout')
class SwitchOutTransform(HammingDistanceSamplingTransform):
    """
    SwitchOut.
    :cite:`DBLP:journals/corr/abs-1808-07512`
    """

    def __init__(self, opts):
        super().__init__(opts)

    def warm_up(self, vocabs):
        super().warm_up(None)
        self.vocabs = vocabs
        if vocabs is None:
            logger.warning(
                "Switchout disable as no vocab, shouldn't happen in training!")

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group("Transform/SwitchOut")
        group.add("-switchout_temperature", "--switchout_temperature",
                  type=float, default=1.0,
                  help="Sampling temperature for SwitchOut. :math:`\\tau^{-1}`"
                       " in :cite:`DBLP:journals/corr/abs-1808-07512`. "
                       "Smaller value makes data more diverse.")

    def _parse_opts(self):
        self.temperature = self.opts.switchout_temperature

    def _switchout(self, tokens, vocab, stats=None):
        assert vocab is not None, "vocab can not be None for SwitchOut."
        # 1. sample number of tokens to corrupt
        n_chosen = self._sample_distance(tokens, self.temperature)
        # 2. sample positions to corrput
        chosen_indices = self._sample_position(tokens, distance=n_chosen)
        # 3. sample corrupted values
        out = []
        for (i, tok) in enumerate(tokens):
            if i in chosen_indices:
                tok = self._sample_replace(vocab, reject=tok)
                out.append(tok)
            else:
                out.append(tok)
        if stats is not None:
            stats.switchout(n_switchout=n_chosen, n_total=len(tokens))
        return out

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply switchout to both src and tgt side tokens."""
        if is_train and self.vocabs is not None:
            src = self._switchout(
                example['src'], self.vocabs['src'].itos, stats)
            tgt = self._switchout(
                example['tgt'], self.vocabs['tgt'].itos, stats)
            example['src'], example['tgt'] = src, tgt
        return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('switchout_temperature', self.temperature)


@register_transform(name='tokendrop')
class TokenDropTransform(HammingDistanceSamplingTransform):
    """Random drop tokens from sentence."""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Token_Drop")
        group.add("-tokendrop_temperature", "--tokendrop_temperature",
                  type=float, default=1.0,
                  help="Sampling temperature for token deletion.")

    def _parse_opts(self):
        self.temperature = self.opts.tokendrop_temperature

    def _token_drop(self, tokens, stats=None):
        # 1. sample number of tokens to corrupt
        n_chosen = self._sample_distance(tokens, self.temperature)
        # 2. sample positions to corrput
        chosen_indices = self._sample_position(tokens, distance=n_chosen)
        # 3. Drop token on chosen position
        out = [tok for (i, tok) in enumerate(tokens)
               if i not in chosen_indices]
        if stats is not None:
            stats.token_drop(n_dropped=n_chosen, n_total=len(tokens))
        return out

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply token drop to both src and tgt side tokens."""
        if is_train:
            src = self._token_drop(example['src'], stats)
            tgt = self._token_drop(example['tgt'], stats)
            example['src'], example['tgt'] = src, tgt
        return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('worddrop_temperature', self.temperature)


@register_transform(name='tokenmask')
class TokenMaskTransform(HammingDistanceSamplingTransform):
    """Random mask tokens from src sentence."""

    MASK_TOK = DefaultTokens.MASK

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Token_Mask")
        group.add('-tokenmask_temperature', '--tokenmask_temperature',
                  type=float, default=1.0,
                  help="Sampling temperature for token masking.")

    def _parse_opts(self):
        self.temperature = self.opts.tokenmask_temperature

    @classmethod
    def get_specials(cls, opts):
        """Get special vocabs added by prefix transform."""
        return ({cls.MASK_TOK}, set())

    def _token_mask(self, tokens, stats=None):
        # 1. sample number of tokens to corrupt
        n_chosen = self._sample_distance(tokens, self.temperature)
        # 2. sample positions to corrput
        chosen_indices = self._sample_position(tokens, distance=n_chosen)
        # 3. mask word on chosen position
        out = []
        for (i, tok) in enumerate(tokens):
            tok = self.MASK_TOK if i in chosen_indices else tok
            out.append(tok)
        if stats is not None:
            stats.token_mask(n_masked=n_chosen, n_total=len(tokens))
        return out

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply word drop to both src and tgt side tokens."""
        if is_train:
            src = self._token_mask(example['src'], stats)
            tgt = self._token_mask(example['tgt'], stats)
            example['src'], example['tgt'] = src, tgt
        return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('tokenmask_temperature', self.temperature)
