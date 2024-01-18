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

    def __init__(
        self,
        tags_dictionary_path,
        max_tags,
        paired_start_tag,
        paired_end_tag,
        isolated_tag,
        src_delimiter,
        tag_corpus_ratio=0.1,
    ):
        self.max_tags = max_tags
        self.tag_corpus_ratio = tag_corpus_ratio
        self.src_delimiter = src_delimiter
        self.internal_dictionary = self._create_internal_dictionary(
            tags_dictionary_path
        )
        self.paired_stag_prefix, self.paired_stag_suffix = map(
            str, paired_start_tag.split("#")
        )
        self.paired_etag_prefix, self.paired_etag_suffix = map(
            str, paired_end_tag.split("#")
        )
        self.isolated_tag_prefix, self.isolated_tag_suffix = map(
            str, isolated_tag.split("#")
        )

        self.automaton = self._create_automaton()

    def _create_internal_dictionary(self, tags_dictionary_path):
        dictionary = list()
        with open(tags_dictionary_path, mode="r", encoding="utf-8") as file:
            pairs = file.readlines()
            for pair in pairs:
                src_term, tgt_term = map(str, pair.split("\t"))
                dictionary.append((src_term.strip(), tgt_term.strip()))
        return dictionary

    def _create_automaton(self):
        automaton = ahocorasick.Automaton()
        for entry in self.internal_dictionary:
            automaton.add_word(entry[0], (entry[0], entry[1]))

        automaton.make_automaton()
        return automaton

    def _tagged_src_tgt(self, src_example, tgt_example) -> tuple:
        """Uses the dictionary to find exact source matches with corresponding
        target matches and adds both paired tags and standalone tags."""

        maybe_augmented = src_example.split(self.src_delimiter)
        source_only = maybe_augmented[0].strip()

        augmented_part = (
            maybe_augmented[1].strip() if len(maybe_augmented) > 1 else None
        )

        tokenized_source_string = source_only.split(" ")
        tokenized_target_string = tgt_example.split(" ")

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
            range(1, self.max_tags + 1), weights=range(self.max_tags, 0, -1), k=1
        )[0]
        single_tag_start_num = random.choices(
            range(1, self.max_tags + 1), weights=range(self.max_tags, 0, -1), k=1
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
                (pair[1] not in " ".join(tokenized_target_string))
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

                src_term = " ".join(
                    tokenized_source_string[
                        source_index : source_index + len(pair[0].split(" "))
                    ]
                )
                tgt_term = " ".join(
                    tokenized_target_string[
                        target_index : target_index + len(pair[1].split(" "))
                    ]
                )

                # Create all possible tag forms. We inject a special
                # unicode char (∥) as a placeholder for whitespace in order
                # to keep the indices unaltered. This char is replaced with
                # spaces before we return the augmented examples.
                src_single_tags = (
                    f"{source_only[src_offset: src_match_start]}"
                    f"{self.isolated_tag_prefix}{single_tag_start_num}"
                    f"{self.isolated_tag_suffix}∥{src_term}∥"
                )
                src_paired_tags = (
                    f"{source_only[src_offset: src_match_start]}"
                    f"{self.paired_stag_prefix}{paired_tag_start_num}"
                    f"{self.paired_stag_suffix}∥{src_term}∥"
                    f"{self.paired_etag_prefix}{paired_tag_start_num}"
                    f"{self.paired_etag_suffix}"
                )

                tgt_single_tags = (
                    f"{tgt_example[tgt_offset: tgt_match_start]}"
                    f"{self.isolated_tag_prefix}{single_tag_start_num}"
                    f"{self.isolated_tag_suffix}∥{tgt_term}∥"
                )
                tgt_paired_tags = (
                    f"{tgt_example[tgt_offset: tgt_match_start]}"
                    f"{self.paired_stag_prefix}{paired_tag_start_num}"
                    f"{self.paired_stag_suffix}∥{tgt_term}∥"
                    f"{self.paired_etag_prefix}{paired_tag_start_num}"
                    f"{self.paired_etag_suffix}∥"
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
        if is_match:
            if augmented_part is not None:
                src_with_tags.append(
                    source_only[src_offset:] + self.src_delimiter + augmented_part
                )
            else:
                src_with_tags.append(source_only[src_offset:])

            tgt_with_tags.append(tgt_example[tgt_offset:])

            return (
                "".join(src_with_tags).replace("∥", " ").split(" "),
                "".join(tgt_with_tags).replace("∥", " ").split(" "),
            ), is_match
        else:
            return (src_example.split(" "), tgt_example.split(" ")), is_match


@register_transform(name="inlinetags")
class InlineTagsTransform(Transform):
    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Available options for adding inline tags."""

        group = parser.add_argument_group("Transform/InlineTags")
        group.add(
            "--tags_dictionary_path",
            "-tags_dictionary_path",
            type=str,
            help="Path to a flat term dictionary.",
        )
        group.add(
            "--tags_corpus_ratio",
            "-tags_corpus_ratio",
            type=float,
            default=0.1,
            help="Ratio of corpus to augment with tags.",
        )
        group.add(
            "--max_tags",
            "-max_tags",
            type=int,
            default=12,
            help="Maximum number of tags that can be added to " "a single sentence.",
        )
        group.add(
            "--paired_stag",
            "-paired_stag",
            type=str,
            default="｟ph_#_beg｠",
            help="The format of an opening paired inline tag. "
            "Must include the character #.",
        )
        group.add(
            "--paired_etag",
            "-paired_etag",
            type=str,
            default="｟ph_#_end｠",
            help="The format of a closing paired inline tag. "
            "Must include the character #.",
        )
        group.add(
            "--isolated_tag",
            "-isolated_tag",
            type=str,
            default="｟ph_#_std｠",
            help="The format of an isolated inline tag. "
            "Must include the character #.",
        )
        group.add(
            "--src_delimiter",
            "-src_delimiter",
            type=str,
            default="｟fuzzy｠",
            help="Any special token used for augmented src sentences. "
            "The default is the fuzzy token used in the "
            "FuzzyMatch transform.",
        )

    def _parse_opts(self):
        self.tags_dictionary_path = self.opts.tags_dictionary_path
        self.tags_corpus_ratio = self.opts.tags_corpus_ratio
        self.max_tags = self.opts.max_tags
        self.src_delimiter = self.opts.src_delimiter

    @classmethod
    def get_specials(cls, opts):
        """Add up to self.max_tags * 2 placeholders to src and tgt vocabs."""

        # Check if the tags include the
        # mandatory "#" number placeholder"
        if (
            "#" not in opts.paired_stag
            or "#" not in opts.paired_etag
            or "#" not in opts.isolated_tag
        ):
            logger.error("Inline tags must include the number " 'placeholder "#"')

        # We split the user-defined tags in the # placeholder
        # in order to number them
        paired_stag_prefix, paired_stag_suffix = map(str, opts.paired_stag.split("#"))
        paired_etag_prefix, paired_etag_suffix = map(str, opts.paired_etag.split("#"))
        isolated_tag_prefix, isolated_tag_suffix = map(
            str, opts.isolated_tag.split("#")
        )

        src_specials, tgt_specials = list(), list()
        tags = list()
        for i in range(1, opts.max_tags * 2):
            tags.extend(
                [
                    paired_stag_prefix + str(i) + paired_stag_suffix,
                    paired_etag_prefix + str(i) + paired_etag_suffix,
                    isolated_tag_prefix + str(i) + isolated_tag_suffix,
                ]
            )

        src_specials.extend(tags)
        tgt_specials.extend(tags)

        return (src_specials, tgt_specials)

    def warm_up(self, vocabs=None):
        """Create the tagger."""

        super().warm_up(None)
        self.tagger = InlineTagger(
            self.tags_dictionary_path,
            self.max_tags,
            self.opts.paired_stag,
            self.opts.paired_etag,
            self.opts.isolated_tag,
            self.src_delimiter,
            self.tags_corpus_ratio,
        )

    def batch_apply(self, batch, is_train=False, stats=None, **kwargs):
        bucket_size = len(batch)
        examples_with_tags = 0

        for i, (ex, _, _) in enumerate(batch):
            # Skip half examples to speed up the transform. This sets
            # a hard limit of 0.5 to the `tags_corpus_ratio`, which is
            # excessive and should be avoided anyway.
            if i % 2 == 0:
                original_src = ex["src"]
                original_tgt = ex["tgt"]
                augmented_example, is_match = self.apply(ex, is_train, stats, **kwargs)
                if is_match and (
                    examples_with_tags < bucket_size * self.tags_corpus_ratio
                ):
                    examples_with_tags += 1
                    ex["src"] = augmented_example["src"]
                    ex["tgt"] = augmented_example["tgt"]
                else:
                    ex["src"] = original_src
                    ex["tgt"] = original_tgt

        logger.debug(f"Added tags to {examples_with_tags}/{bucket_size} examples")
        return batch

    def apply(self, example, is_train=False, stats=None, **kwargs) -> tuple:
        """Add tags (placeholders) to source and target segments."""

        src_tgt_pair, is_match = self.tagger._tagged_src_tgt(
            " ".join(example["src"]), " ".join(example["tgt"])
        )
        example["src"] = src_tgt_pair[0]
        example["tgt"] = src_tgt_pair[1]

        return example, is_match
