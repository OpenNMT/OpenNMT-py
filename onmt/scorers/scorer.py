"""Base Scorer class and relate utils."""


class Scorer(object):
    """A Base class that every scoring method should derived from."""

    def __init__(self, opts):
        self.opts = opts

    def compute_score(self, preds, texts_refs):
        raise NotImplementedError


def build_scorers(opts, scorers_cls):
    """Build scorers in `scorers_cls`."""
    scorers = {}
    for metric, scorer_cls in scorers_cls.items():
        scorer_obj = scorer_cls(opts)
        scorers[metric] = {"scorer": scorer_obj, "value": 0}
    return scorers
