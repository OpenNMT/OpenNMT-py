import re
from onmt.utils.logging import logger
from onmt.transforms import register_transform
from .transform import Transform

import spacy
import ahocorasick


class TermMatcher(object):
    def __init__(
        self,
        termbase_path,
        src_spacy_language_model,
        tgt_spacy_language_model,
        term_example_ratio,
        src_term_stoken,
        tgt_term_stoken,
        tgt_term_etoken,
        src_delimiter,
        term_corpus_ratio=0.2,
    ):
        self.term_example_ratio = term_example_ratio
        self.src_nlp = spacy.load(src_spacy_language_model, disable=["parser", "ner"])
        self.tgt_nlp = spacy.load(tgt_spacy_language_model, disable=["parser", "ner"])

        # We exclude tokenization for contractions in
        # order to avoid inconsistencies with pyonmtok's tokenization.
        # (e.g. "I ca n't" with spacy, "I can ' t" with pyonmttok)
        self.src_nlp.tokenizer.rules = {
            key: value
            for key, value in self.src_nlp.tokenizer.rules.items()
            if "'" not in key and "’" not in key and "‘" not in key
        }
        self.tgt_nlp.tokenizer.rules = {
            key: value
            for key, value in self.tgt_nlp.tokenizer.rules.items()
            if "'" not in key and "’" not in key and "‘" not in key
        }
        self.internal_termbase = self._create_internal_termbase(termbase_path)
        self.automaton = self._create_automaton()
        self.sents_with_terms = 0
        self.total_processed_examples = 0
        self.term_corpus_ratio = term_corpus_ratio
        self.src_term_stoken = src_term_stoken
        self.tgt_term_stoken = tgt_term_stoken
        self.tgt_term_etoken = tgt_term_etoken
        self.src_delimiter = src_delimiter

    def _create_internal_termbase(self, termbase_path):
        logger.info("Creating termbase with lemmas for Terminology transform")
        src_stopwords = self.src_nlp.Defaults.stop_words
        tgt_stopwords = self.tgt_nlp.Defaults.stop_words
        termbase = list()
        with open(termbase_path, mode="r", encoding="utf-8") as file:
            pairs = file.readlines()
            for pair in pairs:
                src_term, tgt_term = map(str, pair.split("\t"))
                src_lemma = " ".join(
                    "∥".join(tok.lemma_.split()) for tok in self.src_nlp(src_term)
                ).strip()
                tgt_lemma = " ".join(
                    tok.lemma_ for tok in self.tgt_nlp(tgt_term)
                ).strip()
                if (
                    src_lemma.lower() not in src_stopwords
                    and tgt_lemma.lower() not in tgt_stopwords
                ):
                    termbase.append((src_lemma, tgt_lemma))
        logger.info(
            f"Created termbase with {len(termbase)} lemmas "
            f"for Terminology transform"
        )
        return termbase

    def _create_automaton(self):
        automaton = ahocorasick.Automaton()
        for term in self.internal_termbase:
            automaton.add_word(term[0], (term[0], term[1]))
        automaton.make_automaton()
        return automaton

    def _src_sentence_with_terms(self, source_string, target_string) -> tuple:

        maybe_augmented = source_string.split(self.src_delimiter)
        source_only = maybe_augmented[0].strip()
        augmented_part = (
            maybe_augmented[1].strip() if len(maybe_augmented) > 1 else None
        )

        doc_src = self.src_nlp(source_only)
        doc_tgt = self.tgt_nlp(target_string)

        # Perform tokenization with spacy for consistency.
        tokenized_source = [tok.text for tok in doc_src]
        lemmatized_source = ["∥".join(tok.lemma_.lower().split()) for tok in doc_src]
        lemmatized_target = [tok.lemma_.lower() for tok in doc_tgt]

        lemmatized_source_string = " ".join(lemmatized_source)

        offset = 0
        source_with_terms = list()
        term_counter = 0

        max_terms_allowed = int(len(tokenized_source) * self.term_example_ratio)
        is_match = False
        for match_end, pair in self.automaton.iter_long(lemmatized_source_string):

            if term_counter == max_terms_allowed:
                break

            match_start = match_end - len(pair[0]) + 1

            # We ensure that the target lemma is present in the lemmatized
            # target string, that the match is an exact match (there is
            # whitespace before or after the term)
            # and we perform some bound checking.
            if (
                (pair[1].lower() not in " ".join(lemmatized_target).lower())
                or (
                    len(lemmatized_source_string) != match_end + 1
                    and not (lemmatized_source_string[match_end + 1].isspace())
                )
                or (
                    not lemmatized_source_string[match_start - 1].isspace()
                    and match_start != 0
                )
            ):
                continue
            else:
                term_counter += 1

                # Map the lemmatized string match index to
                # the lemmatized list index
                lemma_list_index = 0
                for i, w in enumerate(lemmatized_source):
                    if lemma_list_index == match_start:
                        lemma_list_index = i
                        break
                    else:
                        lemma_list_index += len(w) + 1

                # We need to know if the term is multiword
                num_words_in_src_term = len(pair[0].split())
                src_term = " ".join(
                    tokenized_source[
                        lemma_list_index : lemma_list_index + num_words_in_src_term
                    ]
                ).strip()

                # Join multiword target lemmas with a unique separator so
                # we can treat them as single word and not change the indices.
                tgt_term = pair[1].replace(" ", "∥").rstrip().lower()
                source_with_terms.append(
                    f"{lemmatized_source_string[offset: match_start]}"
                    f"{self.src_term_stoken}∥{src_term}∥{self.tgt_term_stoken}∥"
                    f"{tgt_term}∥{self.tgt_term_etoken}"
                )

                offset = match_end + 1
                is_match = True

        self.total_processed_examples += 1
        if is_match:
            self.sents_with_terms += 1
            source_with_terms.append(lemmatized_source_string[offset:])
            tokenized_source_with_terms = "".join(source_with_terms).split()

            if not (
                len(tokenized_source)
                == len(lemmatized_source)
                == len(tokenized_source_with_terms)
            ):
                final_string = " ".join(tokenized_source)
                fixed_punct = re.sub(r" ([^\w\s｟\-\–])", r"\1", final_string)
                return fixed_punct.split(), not is_match

            # Construct the final source from the lemmatized list
            # that contains the terms. We compare the tokens in the
            # term-augmented lemma list with the tokens in the original
            # lemma list. If the lemma is the same, then we replace with
            # the token from the original tokenized source list. If they
            # are not the same, it means the lemma has been augemented
            # with a term, so we inject this in the final list.
            completed_tokenized_source = list()
            for idx in range(len(tokenized_source_with_terms)):
                # Restore the spaces in multi-word terms
                src_lemma = tokenized_source_with_terms[idx].replace("∥", " ")
                if lemmatized_source[idx].replace("∥", " ") == src_lemma:
                    completed_tokenized_source.append(tokenized_source[idx])
                else:
                    completed_tokenized_source.append(src_lemma)

            if augmented_part is not None:
                final_string = " ".join(
                    completed_tokenized_source
                    + [self.src_delimiter]
                    + augmented_part.split()
                )
            else:
                final_string = " ".join(completed_tokenized_source)

            fixed_punct = re.sub(r" ([^\w\s｟\-\–])", r"\1", final_string)
            return fixed_punct.split(), is_match
        else:
            final_string = " ".join(tokenized_source)
            fixed_punct = re.sub(r" ([^\w\s｟\-\–])", r"\1", final_string)
            return fixed_punct.split(), not is_match


@register_transform(name="terminology")
class TerminologyTransform(Transform):
    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Available options for terminology matching."""

        group = parser.add_argument_group("Transform/Terminology")
        group.add(
            "--termbase_path",
            "-termbase_path",
            type=str,
            help="Path to a flat term dictionary.",
        )
        group.add(
            "--src_spacy_language_model",
            "-src_spacy_language_model",
            type=str,
            help="Path to the src spacy language model.",
        )
        group.add(
            "--tgt_spacy_language_model",
            "-tgt_spacy_language_model",
            type=str,
            help="Path to the tgt spacy language model.",
        )
        group.add(
            "--term_corpus_ratio",
            "-term_corpus_ratio",
            type=float,
            default=0.3,
            help="Ratio of corpus to augment with terms.",
        )
        group.add(
            "--term_example_ratio",
            "-term_example_ratio",
            type=float,
            default=0.3,
            help="Max terms allowed in an example.",
        )
        group.add(
            "--src_term_stoken",
            "-src_term_stoken",
            type=str,
            help="The source term start tag.",
            default="｟src_term_start｠",
        )
        group.add(
            "--tgt_term_stoken",
            "-tgt_term_stoken",
            type=str,
            help="The target term start tag.",
            default="｟tgt_term_start｠",
        )
        group.add(
            "--tgt_term_etoken",
            "-tgt_term_etoken",
            type=str,
            help="The target term end tag.",
            default="｟tgt_term_end｠",
        )
        group.add(
            "--s_delimiter",
            "-s_delimiter",
            type=str,
            help="Any special token used for augmented src sentences. "
            "The default is the fuzzy token used in the "
            "FuzzyMatch transform.",
            default="｟fuzzy｠",
        )

    def _parse_opts(self):
        self.termbase_path = self.opts.termbase_path
        self.src_spacy_language_model = self.opts.src_spacy_language_model
        self.tgt_spacy_language_model = self.opts.tgt_spacy_language_model
        self.term_corpus_ratio = self.opts.term_corpus_ratio
        self.term_example_ratio = self.opts.term_example_ratio
        self.s_delimiter = self.opts.s_delimiter
        self.src_term_stoken = self.opts.src_term_stoken
        self.tgt_term_stoken = self.opts.tgt_term_stoken
        self.tgt_term_etoken = self.opts.tgt_term_etoken

    @classmethod
    def get_specials(cls, opts):
        """Add the term tokens to the src vocab."""
        src_specials = list()
        src_specials.extend(
            [opts.src_term_stoken, opts.tgt_term_stoken, opts.tgt_term_etoken]
        )
        return (src_specials, list())

    def warm_up(self, vocabs=None):
        """Create the terminology matcher."""

        super().warm_up(None)
        self.termmatcher = TermMatcher(
            self.termbase_path,
            self.src_spacy_language_model,
            self.tgt_spacy_language_model,
            self.term_example_ratio,
            self.src_term_stoken,
            self.tgt_term_stoken,
            self.tgt_term_etoken,
            self.s_delimiter,
            self.term_corpus_ratio,
        )

    def batch_apply(self, batch, is_train=False, stats=None, **kwargs):
        bucket_size = len(batch)
        examples_with_terms = 0

        for (ex, _, _) in batch:
            augmented_example, is_match = self.apply(ex, is_train, stats, **kwargs)
            if is_match and (
                examples_with_terms < bucket_size * self.term_corpus_ratio
            ):
                examples_with_terms += 1
                ex["src"] = augmented_example["src"]
        logger.info(f"Added terms to {examples_with_terms}/{bucket_size} examples")
        return batch

    def apply(self, example, is_train=False, stats=None, **kwargs) -> tuple:
        """Add terms to source examples."""

        example["src"], is_match = self.termmatcher._src_sentence_with_terms(
            " ".join(example["src"]), " ".join(example["tgt"])
        )
        return example, is_match
