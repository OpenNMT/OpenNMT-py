"""Module that contain iterator used for dynamic data."""
from itertools import cycle

from torchtext.data import batch as torchtext_batch
from onmt.inputters import str2sortkey, max_tok_len, OrderedIterator
from onmt.inputters.corpus import get_corpora, build_corpora_iters,\
    DatasetAdapter
from onmt.transforms import make_transforms


class MixingStrategy(object):
    """Mixing strategy that should be used in Data Iterator."""

    def __init__(self, iterables, weights):
        """Initilize neccessary attr."""
        self._valid_iterable(iterables, weights)
        self.iterables = iterables
        self.weights = weights

    def _valid_iterable(self, iterables, weights):
        iter_keys = iterables.keys()
        weight_keys = weights.keys()
        if iter_keys != weight_keys:
            raise ValueError(
                f"keys in {iterables} & {iterables} should be equal.")

    def __iter__(self):
        raise NotImplementedError


class SequentialMixer(MixingStrategy):
    """Generate data sequentially from `iterables` which is exhaustible."""

    def _iter_datasets(self):
        for ds_name, ds_weight in self.weights.items():
            for _ in range(ds_weight):
                yield ds_name

    def __iter__(self):
        for ds_name in self._iter_datasets():
            iterable = self.iterables[ds_name]
            yield from iterable


class WeightedMixer(MixingStrategy):
    """A mixing strategy that mix data weightedly and iterate infinitely."""

    def __init__(self, iterables, weights):
        super().__init__(iterables, weights)
        self._iterators = {
            ds_name: iter(generator)
            for ds_name, generator in self.iterables.items()
        }

    def _reset_iter(self, ds_name):
        self._iterators[ds_name] = iter(self.iterables[ds_name])

    def _iter_datasets(self):
        for ds_name, ds_weight in self.weights.items():
            for _ in range(ds_weight):
                yield ds_name

    def __iter__(self):
        for ds_name in cycle(self._iter_datasets()):
            iterator = self._iterators[ds_name]
            try:
                item = next(iterator)
            except StopIteration:
                self._reset_iter(ds_name)
                iterator = self._iterators[ds_name]
                item = next(iterator)
            finally:
                yield item


class DynamicDatasetIter(object):
    """Yield batch from (multiple) plain text corpus.

    Args:
        corpora (dict[str, ParallelCorpus]): collections of corpora to iterate;
        corpora_info (dict[str, dict]): corpora infos correspond to corpora;
        transforms (dict[str, Transform]): transforms may be used by corpora;
        fields (dict[str, Field]): fields dict for convert corpora into Tensor;
        is_train (bool): True when generate data for training;
        batch_type (str): batching type to count on, choices=[tokens, sents];
        batch_size (int): numbers of examples in a batch;
        batch_size_multiple (int): make batch size multiply of this;
        data_type (str): input data type, currently only text;
        bucket_size (int): accum this number of examples in a dynamic dataset;
        pool_factor (int): accum this number of batch before sorting;
        skip_empty_level (str): security level when encouter empty line;
        stride (int): iterate data files with this stride;
        offset (int): iterate data files with this offset.

    Attributes:
        batch_size_fn (function): functions to calculate batch_size;
        sort_key (function): functions define how to sort examples;
        dataset_adapter (DatasetAdapter): organize raw corpus to tensor adapt;
        mixer (MixingStrategy): the strategy to iterate corpora.
    """

    def __init__(self, corpora, corpora_info, transforms, fields, is_train,
                 batch_type, batch_size, batch_size_multiple, data_type="text",
                 bucket_size=2048, pool_factor=8192,
                 skip_empty_level='warning', stride=1, offset=0):
        self.corpora = corpora
        self.transforms = transforms
        self.fields = fields
        self.corpora_info = corpora_info
        self.is_train = is_train
        self.init_iterators = False
        self.batch_size = batch_size
        self.batch_size_fn = max_tok_len \
            if is_train and batch_type == "tokens" else None
        self.batch_size_multiple = batch_size_multiple
        self.device = 'cpu'
        self.sort_key = str2sortkey[data_type]
        self.bucket_size = bucket_size
        self.pool_factor = pool_factor
        if stride <= 0:
            raise ValueError(f"Invalid argument for stride={stride}.")
        self.stride = stride
        self.offset = offset
        if skip_empty_level not in ['silent', 'warning', 'error']:
            raise ValueError(
                f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level

    @classmethod
    def from_opts(cls, corpora, transforms, fields, opts, is_train,
                  stride=1, offset=0):
        """Initilize `DynamicDatasetIter` with options parsed from `opts`."""
        batch_size = opts.batch_size if is_train else opts.valid_batch_size
        if opts.batch_size_multiple is not None:
            batch_size_multiple = opts.batch_size_multiple
        else:
            batch_size_multiple = 8 if opts.model_dtype == "fp16" else 1
        return cls(
            corpora, opts.data, transforms, fields, is_train, opts.batch_type,
            batch_size, batch_size_multiple, data_type=opts.data_type,
            bucket_size=opts.bucket_size, pool_factor=opts.pool_factor,
            skip_empty_level=opts.skip_empty_level,
            stride=stride, offset=offset
        )

    def _init_datasets(self):
        datasets_iterables = build_corpora_iters(
            self.corpora, self.transforms,
            self.corpora_info, self.is_train,
            skip_empty_level=self.skip_empty_level,
            stride=self.stride, offset=self.offset)
        self.dataset_adapter = DatasetAdapter(self.fields, self.is_train)
        datasets_weights = {
            ds_name: int(self.corpora_info[ds_name]['weight'])
            for ds_name in datasets_iterables.keys()
        }
        if self.is_train:
            self.mixer = WeightedMixer(datasets_iterables, datasets_weights)
        else:
            self.mixer = SequentialMixer(datasets_iterables, datasets_weights)
        self.init_iterators = True

    def _bucketing(self):
        buckets = torchtext_batch(
            self.mixer,
            batch_size=self.bucket_size,
            batch_size_fn=None)
        yield from buckets

    def __iter__(self):
        if self.init_iterators is False:
            self._init_datasets()
        for bucket in self._bucketing():
            dataset = self.dataset_adapter(bucket)
            train_iter = OrderedIterator(
                dataset,
                self.batch_size,
                pool_factor=self.pool_factor,
                batch_size_fn=self.batch_size_fn,
                batch_size_multiple=self.batch_size_multiple,
                device=self.device,
                train=self.is_train,
                sort=False,
                sort_within_batch=True,
                sort_key=self.sort_key,
                repeat=False,
            )
            for batch in train_iter:
                yield batch


def build_dynamic_dataset_iter(fields, transforms_cls, opts, is_train=True,
                               stride=1, offset=0):
    """Build `DynamicDatasetIter` from fields & opts."""
    transforms = make_transforms(opts, transforms_cls, fields)
    corpora = get_corpora(opts, is_train)
    if corpora is None:
        assert not is_train, "only valid corpus is ignorable."
        return None
    return DynamicDatasetIter.from_opts(
        corpora, transforms, fields, opts, is_train,
        stride=stride, offset=offset)
