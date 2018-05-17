import glob
import onmt
import torch
from onmt.Utils import use_gpu


def make_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_size_fn = None
    if is_train and opt.batch_type == "tokens":
        global max_src_in_batch, max_tgt_in_batch

        def batch_size_fn(new, count, sofar):
            global max_src_in_batch, max_tgt_in_batch
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            max_src_in_batch = max(max_src_in_batch,  len(new.src) + 2)
            max_tgt_in_batch = max(max_tgt_in_batch,  len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)

    device = opt.gpuid[0] if opt.gpuid else -1
    print("here===>", datasets, fields, batch_size, batch_size_fn,
                           device, is_train)
    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train)


def make_loss_compute(model, tgt_vocab, opt):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force,
            opt.copy_loss_by_seqlength)
    else:
        compute = onmt.Loss.NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing)

    if use_gpu(opt):
        compute.cuda()

    return compute


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        print('=========type=====>', type(self.cur_iter), ' len==> ', len(self.cur_iter))
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return onmt.io.OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device, train=self.is_train,
            sort=False, sort_within_batch=True,
            repeat=False)


def lazily_load_dataset(corpus_type, opt):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset)))
        return dataset


    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one onmt.io.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield lazy_dataset_loader(pt, corpus_type)


def load_fields(dataset, data_type, checkpoint, opt):
    if checkpoint is not None:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = onmt.io.load_fields_from_vocab(
            checkpoint['vocab'], data_type)
    else:
        fields = onmt.io.load_fields_from_vocab(
            torch.load(opt.data + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    if data_type == 'text':
        print(' * vocabulary size. source = %d; target = %d' %
              (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        print(' * vocabulary size. target = %d' %
              (len(fields['tgt'].vocab)))

    return fields