"""Base Transform class and relate utils."""
import torch
from onmt.utils.logging import logger
from onmt.utils.misc import check_path


class Transform(object):
    """A Base class that every transform method should derived from."""

    def __init__(self, opts):
        """Initialize Transform by parsing `opts` and add them as attribute."""
        self.opts = opts
        self._parse_opts()

    def _set_seed(self, seed):
        """Reproducibility: Set seed for non-deterministic transform."""
        pass

    @classmethod
    def require_vocab(cls):
        """Override this method to inform it need vocab to start."""
        return False

    def warm_up(self, vocabs=None):
        """Procedure needed after initialize and before apply.

        This should be override if there exist any procedure necessary
        before `apply`, like setups based on parsed options or load models,
        etc.
        """
        if self.opts.seed > 0:
            self._set_seed(self.opts.seed)
        if self.require_vocab():
            if vocabs is None:
                raise ValueError(f"{type(self).__name__} requires vocabs!")
            self.vocabs = vocabs

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

    def apply_reverse(self, translated):
        return translated

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


class ObservableStats:
    """A running observable statistics."""

    __slots__ = []

    def name(self) -> str:
        """Return class name as name for statistics."""
        return type(self).__name__

    def update(self, other: "ObservableStats"):
        """Update current statistics with others."""
        raise NotImplementedError

    def __str__(self) -> str:
        return "{}({})".format(
            self.name(),
            ", ".join(
                f"{name}={getattr(self, name)}" for name in self.__slots__
            )
        )


class TransformStatistics:
    """A observer containing runing statistics."""
    def __init__(self):
        self.observables = {}

    def update(self, observable: ObservableStats):
        """Adding observable to observe/updating existing observable."""
        name = observable.name()
        if name not in self.observables:
            self.observables[name] = observable
        else:
            self.observables[name].update(observable)

    def report(self):
        """Pop out all observing statistics and reporting them."""
        msgs = []
        report_ids = list(self.observables.keys())
        for name in report_ids:
            observable = self.observables.pop(name)
            msgs.append(f"\t\t\t* {str(observable)}")
        if len(msgs) != 0:
            return "\n".join(msgs)
        else:
            return ""


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

    def apply_reverse(self, translated):
        for transform in self.transforms:
            translated = transform.apply_reverse(translated)
        return translated

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


def make_transforms(opts, transforms_cls, vocabs):
    """Build transforms in `transforms_cls` with vocab of `fields`."""
    transforms = {}
    for name, transform_cls in transforms_cls.items():
        if transform_cls.require_vocab() and vocabs is None:
            logger.warning(
                f"{transform_cls.__name__} require vocab to apply, skip it."
            )
            continue
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
