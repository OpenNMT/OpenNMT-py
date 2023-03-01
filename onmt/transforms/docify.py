from onmt.transforms import register_transform
from .transform import Transform
import copy


@register_transform(name='docify')
class DocifyTransform(Transform):
    """
    Convert source and target examples to uppercase.

    This transform uses `unicodedata` to normalize the converted
    uppercase strings as this is needed for some languages (e.g. Greek).
    One issue is that the normalization removes all diacritics and
    accents from the uppercased strings, even though in few occasions some
    diacritics should be kept even in the uppercased form.
    """

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Add an option for the corpus ratio to apply this transform."""

        group = parser.add_argument_group("Transform/Docify")
        group.add("--paragraph_delimiter", "-paragraph_delimiter", type=str,
                  default='｟newline｠', help="Newline delimiter.")
        group.add("--doc_length", "-doc_length", type=int,
                  default=1, help="Number of tokens per doc.")

    def _parse_opts(self):
        self.paragraph_delimiter = self.opts.paragraph_delimiter
        self.doc_length = self.opts.doc_length

    @classmethod
    def get_specials(cls, opts):
        """Add newline tag to src and tgt vocabs."""

        src_specials, tgt_specials = ['｟newline｠'], ['｟newline｠']
        return (src_specials, tgt_specials)

    def warm_up(self, vocabs=None):
        """ Init the doc to empty. """
        super().warm_up(None)
        self.doc = {}
        self.doc['src'] = []
        self.doc['tgt'] = []
        self.doc['indices'] = 0

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Convert source and target examples to uppercase."""

        if example['tgt'] is not None:
            cur_len = max(len(self.doc['src'] + example['src']),
                          len(self.doc['tgt'] + example['tgt']))
        else:
            cur_len = len(self.doc['src'] + example['src'])

        if example['tgt'] is not None:
            if len(example['src']) == 0 and len(example['tgt']) == 0:
                doc2 = copy.deepcopy(self.doc)
                self.doc['src'] = []
                self.doc['tgt'] = []
                self.doc['indices'] = example['indices']
                return doc2
            elif cur_len > self.doc_length:
                if len(self.doc['src']) == 0:
                    return example
                doc2 = copy.deepcopy(self.doc)
                self.doc['src'] = example['src'] + [self.paragraph_delimiter]
                self.doc['tgt'] = example['tgt'] + [self.paragraph_delimiter]
                self.doc['indices'] = example['indices']
                return doc2
            else:
                self.doc['src'] += example['src'] + [self.paragraph_delimiter]
                self.doc['tgt'] += example['tgt'] + [self.paragraph_delimiter]
                return None
        else:
            self.doc['tgt'] = None
            if len(example['src']) == 0:
                doc2 = copy.deepcopy(self.doc)
                self.doc['src'] = []
                self.doc['indices'] = example['indices']
                return doc2
            elif cur_len > self.doc_length:
                if len(self.doc['src']) == 0:
                    return example
                doc2 = copy.deepcopy(self.doc)
                self.doc['src'] = example['src'] + [self.paragraph_delimiter]
                self.doc['indices'] = example['indices']
                return doc2
            else:
                self.doc['src'] += example['src'] + [self.paragraph_delimiter]
                return None

    def apply_reverse(self, translated):
        segments = translated.strip(self.paragraph_delimiter)
        segments = segments.split(self.paragraph_delimiter)
        segments = [segment.strip(' ') for segment in segments]
        return segments
