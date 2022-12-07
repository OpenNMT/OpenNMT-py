"""Module that contain iterator used for dynamic data."""
import torch
from itertools import cycle
from onmt.constants import CorpusTask, ModelTask
from onmt.inputters.text_corpus import get_corpora, build_corpora_iters
from onmt.inputters.text_utils import text_sort_key, max_tok_len, process,\
    numericalize, tensorify, _addcopykeys
from onmt.transforms import make_transforms
from onmt.utils.logging import logger
from onmt.utils.misc import RandomShuffler
from torch.utils.data import DataLoader


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
        self._iterators = {}
        self._counts = {}
        for ds_name in self.iterables.keys():
            self._reset_iter(ds_name)

    def _logging(self):
        """Report corpora loading statistics."""
        msgs = []
        for ds_name, ds_count in self._counts.items():
            msgs.append(f"\t\t\t* {ds_name}: {ds_count}")
        logger.info("Weighted corpora loaded so far:\n"+"\n".join(msgs))

    def _reset_iter(self, ds_name):
        self._iterators[ds_name] = iter(self.iterables[ds_name])
        self._counts[ds_name] = self._counts.get(ds_name, 0) + 1
        self._logging()

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


class DynamicDatasetIter(torch.utils.data.IterableDataset):
    """Yield batch from (multiple) plain text corpus.

    Args:
        corpora (dict[str, ParallelCorpus]): collections of corpora to iterate;
        corpora_info (dict[str, dict]): corpora infos correspond to corpora;
        transforms (dict[str, Transform]): transforms may be used by corpora;
        vocabs (dict[str, Vocab]): vocab dict for convert corpora into Tensor;
        task (str): CorpusTask.TRAIN/VALID/INFER;
        batch_type (str): batching type to count on, choices=[tokens, sents];
        batch_size (int): numbers of examples in a batch;
        batch_size_multiple (int): make batch size multiply of this;
        data_type (str): input data type, currently only text;
        bucket_size (int): accum this number of examples in a dynamic dataset;
        bucket_size_init (int): initialize the bucket with this
        amount of examples;
        bucket_size_increment (int): increment the bucket
        size with this amount of examples;
        copy (Bool): if True, will add specific items for copy_attn
        skip_empty_level (str): security level when encouter empty line;
        stride (int): iterate data files with this stride;
        offset (int): iterate data files with this offset.

    Attributes:
        batch_size_fn (function): functions to calculate batch_size;
        sort_key (function): functions define how to sort examples;
        mixer (MixingStrategy): the strategy to iterate corpora.
    """

    def __init__(self, corpora, corpora_info, transforms, vocabs, task,
                 batch_type, batch_size, batch_size_multiple, data_type="text",
                 bucket_size=2048, bucket_size_init=-1,
                 bucket_size_increment=0, copy=False,
                 skip_empty_level='warning', stride=1, offset=0):
        super(DynamicDatasetIter).__init__()
        self.corpora = corpora
        self.transforms = transforms
        self.vocabs = vocabs
        self.corpora_info = corpora_info
        self.task = task
        self.init_iterators = False
        self.batch_size = batch_size
        self.batch_size_fn = max_tok_len if batch_type == "tokens" else None
        self.batch_size_multiple = batch_size_multiple
        self.device = 'cpu'
        self.sort_key = text_sort_key
        self.bucket_size = bucket_size
        self.bucket_size_init = bucket_size_init
        self.bucket_size_increment = bucket_size_increment
        self.copy = copy
        if stride <= 0:
            raise ValueError(f"Invalid argument for stride={stride}.")
        self.stride = stride
        self.offset = offset
        if skip_empty_level not in ['silent', 'warning', 'error']:
            raise ValueError(
                f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level
        self.random_shuffler = RandomShuffler()

    @classmethod
    def from_opt(cls, corpora, transforms, vocabs, opt, task, copy,
                 stride=1, offset=0):
        """Initilize `DynamicDatasetIter` with options parsed from `opt`."""
        corpora_info = {}
        batch_size = opt.valid_batch_size if (task == CorpusTask.VALID) \
            else opt.batch_size
        if task != CorpusTask.INFER:
            if opt.batch_size_multiple is not None:
                batch_size_multiple = opt.batch_size_multiple
            else:
                batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1
            corpora_info = opt.data
            bucket_size = opt.bucket_size
            bucket_size_init = opt.bucket_size_init
            bucket_size_increment = opt.bucket_size_increment
            skip_empty_level = opt.skip_empty_level
        else:
            batch_size_multiple = 1
            corpora_info[CorpusTask.INFER] = {'transforms': opt.transforms}
            corpora_info[CorpusTask.INFER]['weight'] = 1
            # bucket_size = batch_size
            bucket_size = 16384
            bucket_size_init = -1
            bucket_size_increment = 0
            skip_empty_level = 'warning'
        if task == CorpusTask.INFER and \
           vocabs['data_task'] == ModelTask.LANGUAGE_MODEL:
            # We only support
            batch_size_multiple = 1
            batch_size = 1
        return cls(
            corpora, corpora_info, transforms, vocabs, task, opt.batch_type,
            batch_size, batch_size_multiple, data_type=opt.data_type,
            bucket_size=bucket_size, bucket_size_init=bucket_size_init,
            bucket_size_increment=bucket_size_increment,
            copy=copy,
            skip_empty_level=skip_empty_level,
            stride=stride, offset=offset
        )

    def _init_datasets(self, worker_id):
        if self.num_workers > 0:
            stride = self.stride * self.num_workers
            offset = self.offset * self.num_workers + worker_id
        else:
            stride = self.stride
            offset = self.offset
        datasets_iterables = build_corpora_iters(
            self.corpora, self.transforms, self.corpora_info,
            skip_empty_level=self.skip_empty_level,
            stride=stride, offset=offset)
        datasets_weights = {
            ds_name: int(self.corpora_info[ds_name]['weight'])
            for ds_name in datasets_iterables.keys()
        }
        if self.task == CorpusTask.TRAIN:
            self.mixer = WeightedMixer(datasets_iterables, datasets_weights)
        else:
            self.mixer = SequentialMixer(datasets_iterables, datasets_weights)
        self.init_iterators = True

    def _tuple_to_json_with_tokIDs(self, tuple_bucket):
        bucket = []
        tuple_bucket = process(self.task, tuple_bucket)
        for example in tuple_bucket:
            if example is not None:
                if self.copy:
                    example = _addcopykeys(self.vocabs, example)
                bucket.append(numericalize(self.vocabs, example))
        return bucket

    def _bucketing(self):
        """
        Add up to bucket_size examples from the mixed corpora according
        to the above strategy. example tuple is converted to json and
        tokens numericalized.
        """
        bucket = []
        if self.bucket_size_init > 0:
            _bucket_size = self.bucket_size_init
        else:
            _bucket_size = self.bucket_size
        for ex in self.mixer:
            bucket.append(ex)
            if len(bucket) == _bucket_size:
                yield self._tuple_to_json_with_tokIDs(bucket)
                bucket = []
                if _bucket_size < self.bucket_size:
                    _bucket_size += self.bucket_size_increment
                else:
                    _bucket_size = self.bucket_size
        if bucket:
            yield self._tuple_to_json_with_tokIDs(bucket)

    def batch_iter(self, data, batch_size, batch_size_fn=None,
                   batch_size_multiple=1):
        """Yield elements from data in chunks of batch_size,
        where each chunk size is a multiple of batch_size_multiple.
        """
        if batch_size_fn is None:
            def batch_size_fn(new, count, sofar):
                return count
        minibatch, size_so_far, seen = [], 0, []
        for ex in data:
            if (
                   (ex['src']['src'] not in seen) or
                   (self.task != CorpusTask.TRAIN)
            ):
                seen.append(ex['src']['src'])
                minibatch.append(ex)
                size_so_far = batch_size_fn(ex, len(minibatch),
                                            size_so_far)
                if size_so_far >= batch_size:
                    overflowed = 0
                    if size_so_far > batch_size:
                        overflowed += 1
                    if batch_size_multiple > 1:
                        overflowed += (
                            (len(minibatch) - overflowed)
                            % batch_size_multiple)
                    if overflowed == 0:
                        yield minibatch
                        minibatch, size_so_far, seen = [], 0, []
                    else:
                        if overflowed == len(minibatch):
                            logger.warning(
                                 "The batch will be filled until we reach"
                                 " %d, its size may exceed %d tokens"
                                 % (batch_size_multiple, batch_size)
                                 )
                        else:
                            yield minibatch[:-overflowed]
                            minibatch = minibatch[-overflowed:]
                            size_so_far, seen = 0, []
                            for i, ex in enumerate(minibatch):
                                size_so_far = batch_size_fn(ex, i + 1,
                                                            size_so_far)
        if minibatch:
            yield minibatch

    def __iter__(self):
        for bucket in self._bucketing():
            # For TRAIN we need to group examples by length
            # for faster performance, but otherwise, sequential.
            if self.task == CorpusTask.TRAIN:
                bucket = sorted(bucket, key=self.sort_key)
            p_batch = list(self.batch_iter(
                bucket,
                self.batch_size,
                batch_size_fn=self.batch_size_fn,
                batch_size_multiple=self.batch_size_multiple))
            # For TRAIN we shuffle batches within the bucket
            # otherwise sequential
            if self.task == CorpusTask.TRAIN:
                p_batch = self.random_shuffler(p_batch)
            for minibatch in p_batch:
                # for specific case of rnn_packed need to be sorted
                # within the batch
                minibatch.sort(key=self.sort_key, reverse=True)
                tensor_batch = tensorify(self.vocabs, minibatch)
                yield tensor_batch


def build_dynamic_dataset_iter(opt, transforms_cls, vocabs, copy=False,
                               task=CorpusTask.TRAIN, stride=1, offset=0):
    """
    Build `DynamicDatasetIter` from opt.
    Typically this function is called for CorpusTask.[TRAIN,VALID,INFER]
    from the main tain / translate scripts
    We disable automatic batching in the DataLoader.
    The custom optimized batching is performed by the
    custom class DynamicDatasetIter inherited from IterableDataset
    (and not by a custom collate function).
    We load opt.bucket_size examples, sort them and yield
    mini-batchs of size opt.batch_size.
    The bucket_size must be large enough to ensure homogeneous batches.
    Each worker will load opt.prefetch_factor mini-batches in
    advance to avoid the GPU waiting during the refilling of the bucket.
    """
    transforms = make_transforms(opt, transforms_cls, vocabs)
    corpora = get_corpora(opt, task)
    if corpora is None:
        assert task != CorpusTask.TRAIN, "only valid corpus is ignorable."
        return None
    data_iter = DynamicDatasetIter.from_opt(
        corpora, transforms, vocabs, opt, task, copy=copy,
        stride=stride, offset=offset)
    data_iter.num_workers = opt.num_workers if \
        hasattr(opt, 'num_workers') else 0
    if data_iter.num_workers == 0 or task == CorpusTask.INFER:
        data_iter._init_datasets(0)  # when workers=0 init_fn not called
        data_loader = data_iter
    else:
        data_loader = DataLoader(data_iter, batch_size=None,
                                 pin_memory=True,
                                 multiprocessing_context="spawn",
                                 num_workers=data_iter.num_workers,
                                 worker_init_fn=data_iter._init_datasets,
                                 prefetch_factor=opt.prefetch_factor)
    return data_loader
