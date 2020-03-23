import torchtext
from onmt.inputters import str2sortkey
from onmt.inputters.inputter import max_tok_len, OrderedIterator


class DatasetAdaptor():
    """ creates torchtext Datasets from TaskMixer buckets """
    def __init__(self, fields, has_tgt=True):
        self.fields = fields
        self.has_tgt = has_tgt
        self._select_fields()

    def _select_fields(self):
        self.field_list = []
        for col in ('src', 'tgt', 'indices'):
            # try:
            #     field = self.fields[col].base_field
            # except AttributeError:
            field = self.fields[col]
            self.field_list.append((col, field))

    def _to_examples(self, bucket):
        examples = []
        for tpl in bucket:
            if not self.has_tgt:
                src, indices = tpl
                tpl = (src, (), indices)
            examples.append(torchtext.data.Example.fromlist(
                tpl, self.field_list))
        return examples

    def __call__(self, bucket):
        examples = self._to_examples(bucket)
        dataset = torchtext.data.Dataset(
            examples, self.field_list)
        return dataset


def build_dataset_adaptor_iter(mixer,
                               dataset_adaptor,
                               opt,
                               mb_callback,
                               data_loader_step,
                               is_train=True):
    batch_size = opt.batch_size
    batch_size_fn = max_tok_len \
        if opt.batch_type == "tokens" else None
    batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1
    # data loader cannot access the gpu when CUDA Compute Mode is set to
    # Exclusive_Process. Otherwise results in
    # "CUDA error: all CUDA-capable devices are busy or unavailable"
    #device = "cuda" if opt.gpu_ranks else "cpu"
    device = 'cpu'
    sort_key = str2sortkey[opt.data_type]

    i = data_loader_step
    for bucket in mixer():
        dataset = dataset_adaptor(bucket)
        train_iter = OrderedIterator(
            dataset,
            batch_size,
            batch_size_fn=batch_size_fn,
            batch_size_multiple=batch_size_multiple,
            device=device,
            train=is_train,
            sort=False,
            sort_within_batch=True,
            sort_key=sort_key,
            repeat=False,
        )
        # due to the separation of producer and consumer,
        # the gradient update count is not easily available here
        # and the mixer is not easily available in the trainer.
        for batch in train_iter:
            batch.data_loader_step = i
            mixer.maybe_adjust_mix(i)
            if mb_callback is not None:
                mb_callback(i)
            yield batch
            i += 1
