#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Most code taken from: https://github.com/alvations/sacremoses
# Which in turn is based on the Moses punctuation normalizer.
# https://github.com/moses-smt/mosesdecoder/blob/master/scripts/
# tokenizer/normalize-punctuation.perl


from onmt.utils.logging import logger
import re
import regex

from itertools import chain

from onmt.transforms import register_transform
from .transform import Transform


class MosesPunctNormalizer:
    """
    This is a Python port of the Moses punctuation normalizer
    """

    EXTRA_WHITESPACE = [
        (r"\r", r""),
        (r"\(", r" ("),
        (r"\)", r") "),
        (r" +", r" "),
        (r"\) ([.!:?;,])", r")\g<1>"),
        (r"\( ", r"("),
        (r" \)", r")"),
        (r"(\d) %", r"\g<1>%"),
        (r" :", r":"),
        (r" ;", r";"),
    ]

    NORMALIZE_UNICODE_IF_NOT_PENN = [(r"`", r"'"), (r"''", r' " ')]

    NORMALIZE_UNICODE = [
        ("„", r'"'),
        ("“", r'"'),
        ("”", r'"'),
        ("–", r"-"),
        ("—", r" - "),
        (r" +", r" "),
        ("´", r"'"),
        ("([a-zA-Z])‘([a-zA-Z])", r"\g<1>'\g<2>"),
        ("([a-zA-Z])’([a-zA-Z])", r"\g<1>'\g<2>"),
        ("‘", r"'"),
        ("‚", r"'"),
        ("’", r"'"),
        (r"''", r'"'),
        ("´´", r'"'),
        ("…", r"..."),
    ]

    FRENCH_QUOTES = [
        ("\u00A0«\u00A0", r'"'),
        ("«\u00A0", r'"'),
        ("«", r'"'),
        ("\u00A0»\u00A0", r'"'),
        ("\u00A0»", r'"'),
        ("»", r'"'),
    ]

    HANDLE_PSEUDO_SPACES = [
        ("\u00A0%", r"%"),
        ("nº\u00A0", "nº "),
        ("\u00A0:", r":"),
        ("\u00A0ºC", " ºC"),
        ("\u00A0cm", r" cm"),
        ("\u00A0\\?", "?"),
        ("\u00A0\\!", "!"),
        ("\u00A0;", r";"),
        (",\u00A0", r", "),
        (r" +", r" "),
    ]

    EN_QUOTATION_FOLLOWED_BY_COMMA = [(r'"([,.]+)', r'\g<1>"')]

    DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA = [
        (r',"', r'",'),
        (r'(\.+)"(\s*[^<])', r'"\g<1>\g<2>'),
        # don't fix period at end of sentence
    ]

    DE_ES_CZ_CS_FR = [
        ("(\\d)\u00A0(\\d)", r"\g<1>,\g<2>"),
    ]

    OTHER = [
        ("(\\d)\u00A0(\\d)", r"\g<1>.\g<2>"),
    ]

    # Regex substitutions from replace-unicode-punctuation.perl
    # https://github.com/moses-smt/mosesdecoder/blob/master/
    # scripts/tokenizer/replace-unicode-punctuation.perl
    REPLACE_UNICODE_PUNCTUATION = [
        ("，", ","),
        (r"。\s*", ". "),
        ("、", ","),
        ("”", '"'),
        ("“", '"'),
        ("∶", ":"),
        ("：", ":"),
        ("？", "?"),
        ("《", '"'),
        ("》", '"'),
        ("）", ")"),
        ("！", "!"),
        ("（", "("),
        ("；", ";"),
        ("」", '"'),
        ("「", '"'),
        ("０", "0"),
        ("１", "1"),
        ("２", "2"),
        ("３", "3"),
        ("４", "4"),
        ("５", "5"),
        ("６", "6"),
        ("７", "7"),
        ("８", "8"),
        ("９", "9"),
        (r"．\s*", ". "),
        ("～", "~"),
        ("’", "'"),
        ("…", "..."),
        ("━", "-"),
        ("〈", "<"),
        ("〉", ">"),
        ("【", "["),
        ("】", "]"),
        ("％", "%"),
    ]

    def __init__(self):
        """ """

    def normalize(
        self,
        text,
        lang="en",
        penn=True,
        norm_quote_commas=True,
        norm_numbers=True,
        pre_replace_unicode_punct=False,
        post_remove_control_chars=False,
    ):
        """
        Returns a string with normalized punctuation.
        :param language: The two-letter language code.
        :type lang: str
        :param penn: Normalize Penn Treebank style quotations.
        :type penn: bool
        :param norm_quote_commas: Normalize quotations and commas
        :type norm_quote_commas: bool
        :param norm_numbers: Normalize numbers
        :type norm_numbers: bool
        """
        self.substitutions = [
            self.EXTRA_WHITESPACE,
            self.NORMALIZE_UNICODE,
            self.FRENCH_QUOTES,
            self.HANDLE_PSEUDO_SPACES,
        ]

        if penn:
            # Adds the penn substitutions after extra_whitespace regexes.
            self.substitutions.insert(1, self.NORMALIZE_UNICODE_IF_NOT_PENN)

        if norm_quote_commas:
            if lang == "en":
                self.substitutions.append(self.EN_QUOTATION_FOLLOWED_BY_COMMA)
            elif lang in ["de", "es", "fr"]:
                self.substitutions.append(self.DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA)

        if norm_numbers:
            if lang in ["de", "es", "cz", "cs", "fr"]:
                self.substitutions.append(self.DE_ES_CZ_CS_FR)
            else:
                self.substitutions.append(self.OTHER)

        self.substitutions = list(chain(*self.substitutions))

        # Optionally, replace unicode puncts BEFORE normalization.
        if pre_replace_unicode_punct:
            text = self.replace_unicode_punct(text)

        # Actual normalization.
        for regexp, substitution in self.substitutions:
            text = re.sub(regexp, substitution, str(text))

        # Optionally, replace unicode puncts BEFORE normalization.
        if post_remove_control_chars:
            text = self.remove_control_chars(text)

        return text.strip()

    def replace_unicode_punct(self, text):
        for regexp, substitution in self.REPLACE_UNICODE_PUNCTUATION:
            text = re.sub(regexp, substitution, str(text))
        return text

    def remove_control_chars(self, text):
        return regex.sub(r"\p{C}", "", text)


@register_transform(name="normalize")
class NormalizeTransform(Transform):
    """
    Normalize source and target based on Moses script.
    """

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Add an option for the corpus ratio to apply this transform."""

        group = parser.add_argument_group("Transform/Normalize")
        group.add(
            "--src_lang", "-src_lang", type=str, default="", help="Source language code"
        )
        group.add(
            "--tgt_lang", "-tgt_lang", type=str, default="", help="Target language code"
        )
        group.add("--penn", "-penn", type=bool, default=True, help="Penn substitution")
        group.add(
            "--norm_quote_commas",
            "-norm_quote_commas",
            type=bool,
            default=True,
            help="Normalize quotations and commas",
        )
        group.add(
            "--norm_numbers",
            "-norm_numbers",
            type=bool,
            default=True,
            help="Normalize numbers",
        )
        group.add(
            "--pre_replace_unicode_punct",
            "-pre_replace_unicode_punct",
            type=bool,
            default=False,
            help="Replace unicode punct",
        )
        group.add(
            "--post_remove_control_chars",
            "-post_remove_control_chars",
            type=bool,
            default=False,
            help="Remove control chars",
        )

    def _parse_opts(self):
        self.src_lang = self.opts.src_lang
        self.tgt_lang = self.opts.tgt_lang
        self.penn = self.opts.penn
        self.norm_quote_commas = self.opts.norm_quote_commas
        self.norm_numbers = self.opts.norm_numbers
        self.pre_replace_unicode_punct = self.opts.pre_replace_unicode_punct
        self.post_remove_control_chars = self.opts.post_remove_control_chars

    @staticmethod
    def _get_opt(corpus, opt, def_val):
        """Get opt string of a `corpus`."""
        if "normalize" in corpus["transforms"]:
            value = corpus.get(opt, def_val)
            normalize = value
        else:
            normalize = None
        return normalize

    @classmethod
    def get_opt_dict(cls, opts, opt, def_val):
        """Get normalize settings correspond to corpus in `opts`."""
        normalize_dict = {}
        # normalize dict src/tgt for each dataset
        if hasattr(opts, "data"):
            for c_name, corpus in opts.data.items():
                normalize = cls._get_opt(corpus, opt, def_val)
                if normalize is not None:
                    logger.debug(f"Get {opt} for {c_name}: {normalize}")
                    normalize_dict[c_name] = normalize

        return normalize_dict

    def warm_up(self, vocabs=None):
        """Set options for each dataset."""
        super().warm_up(None)
        self.src_lang_dict = self.get_opt_dict(self.opts, "src_lang", "")
        self.tgt_lang_dict = self.get_opt_dict(self.opts, "tgt_lang", "")
        self.penn_dict = self.get_opt_dict(self.opts, "penn", True)
        self.norm_quote_commas_dict = self.get_opt_dict(
            self.opts, "norm_quote_commas", True
        )
        self.norm_numbers_dict = self.get_opt_dict(self.opts, "norm_numbers", True)
        self.pre_dict = self.get_opt_dict(self.opts, "pre_replace_unicode_punct", False)
        self.post_dict = self.get_opt_dict(
            self.opts, "post_remove_control_chars", False
        )
        self.src_lang_dict["infer"] = self.src_lang
        self.tgt_lang_dict["infer"] = self.tgt_lang
        self.penn_dict["infer"] = self.penn
        self.norm_quote_commas_dict["infer"] = self.norm_quote_commas
        self.norm_numbers_dict["infer"] = self.norm_numbers
        self.pre_dict["infer"] = self.pre_replace_unicode_punct
        self.post_dict["infer"] = self.post_remove_control_chars

        self.mpn = MosesPunctNormalizer()

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Normalize source and target examples."""
        corpus_name = kwargs.get("corpus_name", None)
        if corpus_name is None:
            raise ValueError("corpus_name is required.")

        src_str = self.mpn.normalize(
            " ".join(example["src"]),
            self.src_lang_dict[corpus_name],
            self.penn_dict[corpus_name],
            self.norm_quote_commas_dict[corpus_name],
            self.norm_numbers_dict[corpus_name],
            self.pre_dict[corpus_name],
            self.post_dict[corpus_name],
        )
        example["src"] = src_str.split(" ")

        if example["tgt"] is not None:
            tgt_str = self.mpn.normalize(
                " ".join(example["tgt"]),
                self.tgt_lang_dict[corpus_name],
                self.penn_dict[corpus_name],
                self.norm_quote_commas_dict[corpus_name],
                self.norm_numbers_dict[corpus_name],
                self.pre_dict[corpus_name],
                self.post_dict[corpus_name],
            )
            example["tgt"] = tgt_str.split(" ")

        return example
