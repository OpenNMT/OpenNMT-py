from onmt.utils.logging import logger
from onmt.transforms import register_transform
from .transform import Transform

from rapidfuzz import fuzz, process
import numpy as np
import time

_FUZZY_TOKEN = '｟fuzzy｠'


class FuzzyMatcher(object):
    """Class for creating and setting up fuzzy matchers."""

    def __init__(self, tm_path, corpus_ratio, threshold=70, tm_delimiter='\t'):
        self.threshold = threshold
        self.corpus_ratio = corpus_ratio
        self.tm_delimiter = tm_delimiter
        self.dict_tm = self._create_tm(tm_path)

    def _create_tm(self, tm_path):
        src_segments, tgt_segments = list(), list()
        with open(tm_path, mode='r', encoding='utf-8') as file:
            pairs = file.readlines()
            for pair in pairs:
                source, target = map(str, pair.split(self.tm_delimiter))

                # Filter out very short or very long sentences
                # from the TM for better performance
                if len(source) < 4 or len(source) > 70:
                    continue
                src_segments.append(source.strip())
                tgt_segments.append(target.strip())
        logger.info(f'Translation Memory size for fuzzymatching transform: '
                    f'{len(src_segments)}')
        return [src_segments, tgt_segments]

    def _get_match(self, query):
        """Not used. We use the `_get_batch_match` method
        in conjuction with the transform's `batch_apply`
        for acceptable performance."""

        # We can speed things up a bit by addding the argument
        # `processor=None` to avoid preprocessing the string
        result = process.extractOne(
            query, self.dict_tm, scorer=fuzz.ratio, score_cutoff=self.threshold
        )
        if result is not None and result[1] < 100:  # We don't want exact match
            target_match = result[2]
            return self._concatenated_example(query, target_match)
        else:
            return query

    def _get_batch_match(self, query_list):
        logger.info(f'Starting fuzzy matching on {len(query_list)} examples')
        fuzzy_count = 0
        start = time.time()
        augmented = list()

        # We split the bucket (`query_list`) and perform fuzzy matching
        # in smaller batches in order to reduce memory usage.
        # Perfomance is not affected.
        portion = 25
        corpus_batches = np.array_split(query_list, portion)
        for corpus_batch in corpus_batches:
            plist = list(corpus_batch)
            if fuzzy_count >= len(query_list) * self.corpus_ratio:
                augmented.extend(plist)
                continue

            results = process.cdist(plist,
                                    self.dict_tm[0],
                                    scorer=fuzz.ratio,
                                    dtype=np.uint8,
                                    score_cutoff=self.threshold,
                                    workers=-1)

            matches = np.any(results, 1)
            argmax = np.argmax(results, axis=1)
            for idx, s in enumerate(plist):
                # Probably redundant
                if _FUZZY_TOKEN in s:
                    continue
                # We don't want exact matches
                if matches[idx] and results[idx][argmax[idx]] < 100:
                    if fuzzy_count >= len(query_list) * self.corpus_ratio:
                        break
                    plist[idx] = s + _FUZZY_TOKEN + \
                        self.dict_tm[1][argmax[idx]]
                    fuzzy_count += 1
            augmented.extend(plist)

        end = time.time()
        logger.info(f'FuzzyMatching Transform: Added {fuzzy_count} '
                    f'fuzzies in {end-start} secs')

        return augmented

    # Only used for single example fuzzy matching with `_get_match`
    def _concatenated_example(self, query, target_match):
        if target_match is None:
            return query
        return query + _FUZZY_TOKEN + target_match


@register_transform(name='fuzzymatching')
class FuzzyTransform(Transform):
    """Perform fuzzy matching against a translation memory and
    augment source examples with target matches for Neural Fuzzy Adaptation.

    TODO: cite <https://aclanthology.org/P19-1175>"
    """

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Options for fuzzy matching."""

        group = parser.add_argument_group("Transform/FuzzyMatching")
        group.add("--tm_path", "-tm_path",
                  type=str, help="Path to a flat text TM.")
        group.add("--fuzzy_corpus_ratio", "-fuzzy_corpus_ratio", type=float,
                  default=0.1,
                  help="Ratio of corpus to augment with fuzzy matches.")
        group.add("--fuzzy_threshold", "-fuzzy_threshold", type=int,
                  default=70, help="The fuzzy matching threshold.")
        group.add("--tm_delimiter", "-tm_delimiter",
                  type=str, default="\t",
                  help="The delimiter used in the flat text TM.")

    def _parse_opts(self):
        self.tm_path = self.opts.tm_path
        self.fuzzy_corpus_ratio = self.opts.fuzzy_corpus_ratio
        self.fuzzy_threshold = self.opts.fuzzy_threshold
        self.tm_delimiter = self.opts.tm_delimiter

    @classmethod
    def get_specials(cls, opts):
        """Add the fuzzy mark token to the src vocab."""
        return ([_FUZZY_TOKEN], list())

    def warm_up(self, vocabs=None):
        """Create the fuzzy matcher."""

        super().warm_up(None)
        self.matcher = FuzzyMatcher(self.tm_path,
                                    self.fuzzy_corpus_ratio,
                                    self.fuzzy_threshold,
                                    self.tm_delimiter)

    def apply(self, example, is_train=False, stats=None, **kwargs):
        return example

    def batch_apply(self, batch, is_train=False, stats=None, **kwargs):
        src_segments = list()
        for (ex, _, _) in batch:

            # Apply a basic filtering to leave out very short or very long
            # sentences and speed up things a bit during fuzzy matching
            if len(' '.join(ex['src'])) > 4 and len(' '.join(ex['src'])) < 70:
                src_segments.append(' '.join(ex['src']))
            else:
                src_segments.append('')
        fuzzied_src = self.matcher._get_batch_match(src_segments)
        assert (len(src_segments) == len(fuzzied_src))
        for idx, (example, _, _) in enumerate(batch):
            if fuzzied_src[idx] != '':
                example['src'] = fuzzied_src[idx].split()

        return batch
