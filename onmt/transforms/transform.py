"""Base Transform class and relate utils."""
import torch
from onmt.utils.logging import logger
from onmt.utils.misc import check_path
from onmt.inputters.fields import get_vocabs


class Transform(object):
    """A Base class that every transform method should derived from."""

    def __init__(self, opts):
        """Initialize Transform by parsing `opts` and add them as attribute."""
        self.opts = opts
        self._parse_opts()

    def _set_seed(self, seed):
        """Reproducibility: Set seed for non-deterministic transform."""
        pass

    def warm_up(self, vocabs=None):
        """Procedure needed after initialize and before apply.

        This should be override if there exist any procedure necessary
        before `apply`, like setups based on parsed options or load models,
        etc.
        """
        if self.opts.seed > 0:
            self._set_seed(self.opts.seed)

    @classmethod
    def add_options(cls, parser):
        """Available options relate to this Transform."""
        pass

    @classmethod
    def _validate_options(cls, opts):
        """Extra checks to validate options added from `add_options`."""
        pass

    @classmethod
    def get_specials(cls, opts):
        return (set(), set())

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply transform to `example`.

        Args:
            example (dict): a dict of field value, ex. src, tgt;
            is_train (bool): Indicate if src/tgt is training data;
            stats (TransformStatistics): a statistic object.
        """
        raise NotImplementedError

    def __getstate__(self):
        """Pickling following for rebuild."""
        state = {"opts": self.opts}
        if hasattr(self, 'vocabs'):
            state['vocabs'] = self.vocabs
        return state

    def _parse_opts(self):
        """Parse opts to set/reset instance's attributes.

        This should be override if there are attributes other than self.opts.
        To make sure we recover from picked state.
        (This should only contain attribute assignment, other routine is
        suggest to define in `warm_up`.)
        """
        pass

    def __setstate__(self, state):
        """Reload when unpickling from save file."""
        self.opts = state["opts"]
        self._parse_opts()
        vocabs = state.get('vocabs', None)
        self.warm_up(vocabs=vocabs)

    def stats(self):
        """Return statistic message."""
        return ''

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return ''

    def __repr__(self):
        cls_name = type(self).__name__
        cls_args = self._repr_args()
        return '{}({})'.format(cls_name, cls_args)


class TransformStatistics(object):
    """Return a statistic counter for Transform."""

    def __init__(self):
        """Initialize statistic counter."""
        self.reset()

    def reset(self):
        """Statistic counters for all transforms."""
        self.filtered = 0
        self.words, self.subwords = 0, 0
        self.n_switchouted, self.so_total = 0, 0
        self.n_dropped, self.td_total = 0, 0
        self.n_masked, self.tm_total = 0, 0

    def filter_too_long(self):
        """Update filtered sentence counter."""
        self.filtered += 1

    def subword(self, subwords, words):
        """Update subword counter."""
        self.words += words
        self.subwords += subwords

    def switchout(self, n_switchout, n_total):
        """Update switchout counter."""
        self.n_switchouted += n_switchout
        self.so_total += n_total

    def token_drop(self, n_dropped, n_total):
        """Update token drop counter."""
        self.n_dropped += n_dropped
        self.td_total += n_total

    def token_mask(self, n_masked, n_total):
        """Update token mask counter."""
        self.n_masked += n_masked
        self.tm_total += n_total

    def report(self):
        """Return transform statistics report and reset counter."""
        msg = ''
        if self.filtered > 0:
            msg += f'Filtred sentence: {self.filtered} sent\n'.format()
        if self.words > 0:
            msg += f'Subword(SP/Tokenizer): {self.words} -> {self.subwords} tok\n'  # noqa: E501
        if self.so_total > 0:
            msg += f'SwitchOut: {self.n_switchouted}/{self.so_total} tok\n'
        if self.td_total > 0:
            msg += f'Token dropped: {self.n_dropped}/{self.td_total} tok\n'
        if self.tm_total > 0:
            msg += f'Token masked: {self.n_masked}/{self.tm_total} tok\n'
        self.reset()
        return msg


class TransformPipe(Transform):
    """Pipeline built by a list of Transform instance."""

    def __init__(self, opts, transform_list):
        """Initialize pipeline by a list of transform instance."""
        self.opts = None  # opts is not required
        self.transforms = transform_list
        self.statistics = TransformStatistics()

    @classmethod
    def build_from(cls, transform_list):
        """Return a `TransformPipe` instance build from `transform_list`."""
        for transform in transform_list:
            assert isinstance(transform, Transform), \
                "transform should be a instance of Transform."
        transform_pipe = cls(None, transform_list)
        return transform_pipe

    def warm_up(self, vocabs):
        """Warm up Pipeline by iterate over all transfroms."""
        for transform in self.transforms:
            transform.warm_up(vocabs)

    @classmethod
    def get_specials(cls, opts, transforms):
        """Return all specials introduced by `transforms`."""
        src_specials, tgt_specials = set(), set()
        for transform in transforms:
            _src_special, _tgt_special = transform.get_specials(transform.opts)
            src_specials.update(_src_special)
            tgt_specials.update(tgt_specials)
        return (src_specials, tgt_specials)

    def apply(self, example, is_train=False, **kwargs):
        """Apply transform pipe to `example`.

        Args:
            example (dict): a dict of field value, ex. src, tgt.

        """
        for transform in self.transforms:
            example = transform.apply(
                example, is_train=is_train, stats=self.statistics, **kwargs)
            if example is None:
                break
        return example

    def __getstate__(self):
        """Pickling following for rebuild."""
        return (self.opts, self.transforms, self.statistics)

    def __setstate__(self, state):
        """Reload when unpickling from save file."""
        self.opts, self.transforms, self.statistics = state

    def stats(self):
        """Return statistic message."""
        return self.statistics.report()

    def _repr_args(self):
        """Return str represent key arguments for class."""
        info_args = []
        for transform in self.transforms:
            info_args.append(repr(transform))
        return ', '.join(info_args)


def make_transforms(opts, transforms_cls, fields):
    """Build transforms in `transforms_cls` with vocab of `fields`."""
    vocabs = get_vocabs(fields) if fields is not None else None
    transforms = {}
    for name, transform_cls in transforms_cls.items():
        transform_obj = transform_cls(opts)
        transform_obj.warm_up(vocabs)
        transforms[name] = transform_obj
    return transforms


def get_specials(opts, transforms_cls_dict):
    """Get specials of transforms that should be registed in Vocab."""
    all_specials = {'src': set(), 'tgt': set()}
    for name, transform_cls in transforms_cls_dict.items():
        src_specials, tgt_specials = transform_cls.get_specials(opts)
        all_specials['src'].update(src_specials)
        all_specials['tgt'].update(tgt_specials)
    logger.info(f"Get special vocabs from Transforms: {all_specials}.")
    return all_specials


def save_transforms(transforms, save_data, overwrite=True):
    """Dump `transforms` object."""
    transforms_path = "{}.transforms.pt".format(save_data)
    check_path(transforms_path, exist_ok=overwrite, log=logger.warning)
    logger.info(f"Saving Transforms to {transforms_path}.")
    torch.save(transforms, transforms_path)


def load_transforms(opts):
    """Load dumped `transforms` object."""
    transforms_path = "{}.transforms.pt".format(opts.save_data)
    transforms = torch.load(transforms_path)
    logger.info("Transforms loaded.")
    return transforms
