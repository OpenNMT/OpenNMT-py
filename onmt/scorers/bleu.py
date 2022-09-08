from .scorer import Scorer
from onmt.scorers import register_scorer
from sacrebleu import corpus_bleu


@register_scorer(metric='BLEU')
class BleuScorer(Scorer):
    """BLEU scorer class."""

    def __init__(self, opts):
        """Initialize necessary options for sentencepiece."""
        super().__init__(opts)

    def compute_score(self, preds, texts_refs):
        try:
            score = corpus_bleu(preds, [texts_refs]).score
        except Exception:
            score = 0
        return score
