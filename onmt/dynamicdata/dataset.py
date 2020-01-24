import torchtext
from onmt.inputters import str2sortkey
from onmt.inputters.inputter import max_tok_len, OrderedIterator

class DatasetAdaptor():
    """ creates torchtext Datasets from GroupMixer buckets """
    def __init__(self, fields):
        self.fields = fields
        self._select_fields()

    def _select_fields(self):
        self.field_list = []
        for col in ('src', 'tgt', 'indices'):
            #try:
            #    field = self.fields[col].base_field
            #except AttributeError:
            field = self.fields[col]
            self.field_list.append((col, field))

    def _to_examples(self, bucket):
        examples = [torchtext.data.Example.fromlist(tpl, self.field_list)
                    for tpl in bucket]
        return examples

    def __call__(self, bucket):
        examples = self._to_examples(bucket)
        dataset = torchtext.data.Dataset(
            examples, self.field_list)
        return dataset

def build_dataset_adaptor_iter(mixer, dataset_adaptor, opt, mb_callback, is_train=True):
    batch_size = opt.batch_size
    batch_size_fn = max_tok_len \
        if opt.batch_type == "tokens" else None
    batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1
    device = "cuda" if opt.gpu_ranks else "cpu"
    # FIXME: --data_type is a prep option, not train
    sort_key = str2sortkey['text']  #[opt.data_type]

    i = 0
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
            repeat=False)
        # due to the separation of producer and consumer,
        # the gradient update count is not easily available here
        # and the mixer is not easily available in the trainer.
        for batch in train_iter:
            mixer.maybe_adjust_mix(i)
            if mb_callback is not None:
                mb_callback(i)
            yield batch
            i += 1
