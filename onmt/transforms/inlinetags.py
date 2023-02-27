from onmt.utils.logging import logger
from onmt.transforms import register_transform
from .transform import Transform

import random
import ahocorasick
import string


class InlineTagger(object):
    """Class for augmenting source and target sentences
    with inline tags (placeholders).

    It requires a prepared tab-delimited dictionary of source-target
    words and phrases. A dictionary can be created with
    tools such as fastalign and it should ideally contain enough long
    phrases (of 3-4 words or more) for more realistic applications
    of start and end tags. A dictionary with 20.000-30.000 entries
    should give sufficient number of matches."""

    def __init__(self, tags_dictionary_path, max_tags,
                 tag_corpus_ratio=0.1, src_delimiter='｟fuzzy｠'):
        self.max_tags = max_tags
        self.tagged_examples = 0
        self.total_processed_examples = 0
        self.tag_corpus_ratio = tag_corpus_ratio
        self.src_delimiter = src_delimiter
        self.internal_dictionary = self._create_internal_dictionary(
            tags_dictionary_path
        )
        self.automaton = self._create_automaton()

    def _create_internal_dictionary(self, tags_dictionary_path):
        logger.info('Creating tag dictionary for tagging transform...')
        dictionary = list()
        with open(tags_dictionary_path, mode='r', encoding='utf-8') as file:
            pairs = file.readlines()
            for pair in pairs:
                src_term, tgt_term = map(str, pair.split('\t'))
                dictionary.append((src_term.strip(), tgt_term.strip()))
        logger.info(f'Created tag dictionary with {len(dictionary)} entries.')
        return dictionary

    def _create_automaton(self):
        automaton = ahocorasick.Automaton()
        for entry in self.internal_dictionary:
            automaton.add_word(entry[0], (entry[0], entry[1]))

        automaton.make_automaton()
        return automaton

    def _tagged_src_tgt(self, src_example, tgt_example):
        """Uses the dictionary to find exact source matches with corresponding
        target matches and adds both paired tags and standalone tags."""

        maybe_augmented = src_example.split(self.src_delimiter)
        source_only = maybe_augmented[0].strip()

        augmented_part = maybe_augmented[1].strip() \
            if len(maybe_augmented) > 0 else None

        tokenized_source_string = source_only.split()
        tokenized_target_string = tgt_example.split()

        src_offset, tgt_offset = 0, 0
        src_with_tags, tgt_with_tags = list(), list()

        # We set the start number of tags to a random number from 1
        # to 12 + the number of subsequent tags that
        # will be added. We also apply weights to this choice so tags
        # are more probable to start from 1, then from 2, etc.
        # This way we cover most scenarios met in real usage and
        # the system will learn to handle a fairly large number of
        # numbered tags (but not an excessively large number)
        paired_tag_start_num = random.choices(
            range(1, 13), weights=range(12, 0, -1), k=1
        )[0]
        single_tag_start_num = random.choices(
            range(1, 13), weights=range(12, 0, -1), k=1
        )[0]

        is_match = False
        tag_counter = 0
        for src_match_end, pair in self.automaton.iter_long(source_only):
            if tag_counter == self.max_tags:
                break

            src_match_start = src_match_end - len(pair[0]) + 1
            tgt_match_start = tgt_example.find(pair[1], tgt_offset)
            tgt_match_end = tgt_match_start + len(pair[1])

            # Make sure we only search for exact matches (we don't want
            # to match part of words) and perform some bound checking
            if (
                (pair[1] not in ' '.join(tokenized_target_string))
                or (
                    len(source_only) != src_match_end + 1
                    and not (
                        source_only[src_match_end + 1].isspace()
                        or source_only[src_match_end + 1] in string.punctuation
                    )
                )
                or (
                    not source_only[src_match_start - 1].isspace()
                    and src_match_start != 0
                )
            ):
                continue
            else:
                source_index = 0
                for i, w in enumerate(tokenized_source_string):
                    if source_index == src_match_start:
                        source_index = i
                        break
                    else:
                        source_index += len(w) + 1

                target_index = 0
                for i, w in enumerate(tokenized_target_string):
                    if target_index == tgt_match_start:
                        target_index = i
                        break
                    else:
                        target_index += len(w) + 1

                src_term = ' '.join(
                    tokenized_source_string[
                        source_index: source_index + len(pair[0].split())
                    ]
                )
                tgt_term = ' '.join(
                    tokenized_target_string[
                        target_index: target_index + len(pair[1].split())
                    ]
                )

                src_single_tags = (
                    f'{source_only[src_offset: src_match_start]}'
                    f'｟ph_{single_tag_start_num}_std｠{src_term}'
                )
                src_paired_tags = (
                    f'{source_only[src_offset: src_match_start]}'
                    f'｟ph_{paired_tag_start_num}_beg｠{src_term}'
                    f'｟ph_{paired_tag_start_num}_end｠'
                )

                tgt_single_tags = (
                    f'{tgt_example[tgt_offset: tgt_match_start]}'
                    f'｟ph_{single_tag_start_num}_std｠{tgt_term} '
                )
                tgt_paired_tags = (
                    f'{tgt_example[tgt_offset: tgt_match_start]}'
                    f'｟ph_{paired_tag_start_num}_beg｠{tgt_term}'
                    f'｟ph_{paired_tag_start_num}_end｠ '
                )

                # Make a weighted choice between paired tags or single tags.
                # We usually encounter, and thus here we favor, paired tags
                # with a ratio 1/3.
                choice = random.choices(
                    [src_single_tags, src_paired_tags], weights=(1, 3), k=1
                )

                src_with_tags.append(choice[0])
                src_offset = src_match_end + 1

                if choice[0] is src_single_tags:
                    tgt_with_tags.append(tgt_single_tags)
                    single_tag_start_num += 1
                else:
                    tgt_with_tags.append(tgt_paired_tags)
                    paired_tag_start_num += 1

                tgt_offset = tgt_match_end + 1
                tag_counter += 1
                is_match = True
        self.total_processed_examples += 1
        if is_match:
            self.tagged_examples += 1
            if augmented_part is not None:
                src_with_tags.append(source_only[src_offset:] +
                                     self.src_delimiter +
                                     augmented_part)
            else:
                src_with_tags.append(source_only[src_offset:])

            tgt_with_tags.append(tgt_example[tgt_offset:])

            return (
                ' '.join(src_with_tags).split(),
                ' '.join(tgt_with_tags).split(),
            )
        else:
            return (src_example.split(), tgt_example.split())


@register_transform(name='inlinetags')
class InlineTagsTransform(Transform):
    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Available options for adding inline tags."""

        group = parser.add_argument_group("Transform/InlineTags")
        group.add("--tags_dictionary_path", "-tags_dictionary_path",
                  type=str, help="Path to a flat term dictionary.")
        group.add("--tags_corpus_ratio", "-tags_corpus_ratio", type=float,
                  default=0.1, help="Ratio of corpus to augment with tags.")
        group.add("--max_tags", "-max_tags", type=int,
                  default=12, help="Maximum number for numbering tags.")
        group.add("--src_delimiter", "-src_delimiter", type=str,
                  default='｟fuzzy｠',
                  help="Any special token used for augmented src sentences. "
                  "The default is the fuzzy token used in the "
                  "FuzzyMatch transform.")

    def _parse_opts(self):
        self.tags_dictionary_path = self.opts.tags_dictionary_path
        self.tags_corpus_ratio = self.opts.tags_corpus_ratio
        self.max_tags = self.opts.max_tags
        self.src_delimiter = self.opts.src_delimiter

    @classmethod
    def get_specials(cls, opts):
        """Add up to 20 placeholders to src and tgt vocabs."""

        src_specials, tgt_specials = list(), list()
        tags = list()
        for i in range(1, 21):
            tags.extend(['｟ph_' + str(i) + '_beg｠',
                         '｟ph_' + str(i) + '_end｠',
                         '｟ph_' + str(i) + '_std｠'])

        src_specials.extend(tags)
        tgt_specials.extend(tags)

        return (src_specials, tgt_specials)

    def warm_up(self, vocabs=None):
        """Create the tagger."""

        super().warm_up(None)
        self.tagger = InlineTagger(self.tags_dictionary_path,
                                   self.max_tags,
                                   self.tags_corpus_ratio,
                                   self.src_delimiter)

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Add tags (placeholders) to source and target segments."""

        if self.tagger.tagged_examples/self.tagger.total_processed_examples \
                > self.tags_corpus_ratio:
            return example

        src_tgt_pair = self.tagger._tagged_src_tgt(
            ' '.join(example['src']), ' '.join(example['tgt'])
        )
        example['src'] = src_tgt_pair[0]
        example['tgt'] = src_tgt_pair[1]

        return example

