import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example


def bert_text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    return len(ex.tokens)


class BertDataset(TorchtextDataset):
    """Defines a BERT dataset composed of Examples along with its Fields.
    Args:
        fields_dict (dict[str, Field]): a dict containing all Field with
            its name.
        instances (Iterable[dict[]]): a list of dictionary, each dict
            represent one Example with its field specified by fields_dict.
    """

    def __init__(self, fields_dict, instances,
                 sort_key=bert_text_sort_key, filter_pred=None):
        self.sort_key = sort_key
        examples = []
        ex_fields = {k: [(k, v)] for k, v in fields_dict.items()}
        for instance in instances:
            ex = Example.fromdict(instance, ex_fields)
            examples.append(ex)
        fields_list = list(fields_dict.items())

        super(BertDataset, self).__init__(examples, fields_list, filter_pred)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)
