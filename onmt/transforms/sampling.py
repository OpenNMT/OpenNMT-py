"""Transforms relate to hamming distance sampling."""
import random
import numpy as np
from onmt.constants import DefaultTokens
from onmt.transforms import register_transform
from .transform import Transform, ObservableStats


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


class SwitchOutStats(ObservableStats):
    """Runing statistics for counting tokens being switched out."""

    __slots__ = ["changed", "total"]

    def __init__(self, changed: int, total: int):
        self.changed = changed
        self.total = total

    def update(self, other: "SwitchOutStats"):
        self.changed += other.changed
        self.total += other.total


@register_transform(name='switchout')
class SwitchOutTransform(HammingDistanceSamplingTransform):
    """
    SwitchOut.
    :cite:`DBLP:journals/corr/abs-1808-07512`
    """

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def require_vocab(cls):
        """Override this method to inform it need vocab to start."""
        return True

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
        # 1. sample number of tokens to corrupt
        n_chosen = self._sample_distance(tokens, self.temperature)
        # 2. sample positions to corrput
        chosen_indices = self._sample_position(tokens, distance=n_chosen)
        # 3. sample corrupted values
        for i in chosen_indices:
            tokens[i] = self._sample_replace(vocab, reject=tokens[i])
        if stats is not None:
            stats.update(SwitchOutStats(n_chosen, len(tokens)))
        return tokens

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply switchout to both src and tgt side tokens."""
        if is_train:
            example['src'] = self._switchout(
                example['src'], self.vocabs['src'].ids_to_tokens, stats)
            example['tgt'] = self._switchout(
                example['tgt'], self.vocabs['tgt'].ids_to_tokens, stats)
        return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('switchout_temperature', self.temperature)


class TokenDropStats(ObservableStats):
    """Runing statistics for counting tokens being switched out."""

    __slots__ = ["dropped", "total"]

    def __init__(self, dropped: int, total: int):
        self.dropped = dropped
        self.total = total

    def update(self, other: "TokenDropStats"):
        self.dropped += other.dropped
        self.total += other.total


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
        n_items = len(tokens)
        # 1. sample number of tokens to corrupt
        n_chosen = self._sample_distance(tokens, self.temperature)
        # 2. sample positions to corrput
        chosen_indices = self._sample_position(tokens, distance=n_chosen)
        # 3. Drop token on chosen position
        out = [tok for (i, tok) in enumerate(tokens)
               if i not in chosen_indices]
        if stats is not None:
            stats.update(TokenDropStats(n_chosen, n_items))
        return out

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply token drop to both src and tgt side tokens."""
        if is_train:
            example['src'] = self._token_drop(example['src'], stats)
            example['tgt'] = self._token_drop(example['tgt'], stats)
        return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('tokendrop_temperature', self.temperature)


class TokenMaskStats(ObservableStats):
    """Runing statistics for counting tokens being switched out."""

    __slots__ = ["masked", "total"]

    def __init__(self, masked: int, total: int):
        self.masked = masked
        self.total = total

    def update(self, other: "TokenMaskStats"):
        self.masked += other.masked
        self.total += other.total


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
        for i in chosen_indices:
            tokens[i] = self.MASK_TOK
        if stats is not None:
            stats.update(TokenDropStats(n_chosen, len(tokens)))
        return tokens

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply word drop to both src and tgt side tokens."""
        if is_train:
            example['src'] = self._token_mask(example['src'], stats)
        return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('tokenmask_temperature', self.temperature)
