
"""Module for dynamic scoring"""
import os
import importlib

from .scorer import Scorer, build_scorers

AVAILABLE_SCORERS = {}


def get_scorers_cls(metric_names):
    """Returns a dict with scorers related to the metrics
    indicated in `metric_names`."""
    scorers_cls = {}
    for name in metric_names:
        if name not in AVAILABLE_SCORERS.keys():
            raise ValueError("specified metric not supported!")
        scorers_cls[name] = AVAILABLE_SCORERS[name]
    return scorers_cls


__all__ = ["get_scorers_cls", "build_scorers"]


def register_scorer(metric):
    """Scorer register that can be used to add new scorer class."""

    def register_scorer_cls(cls):
        if metric in AVAILABLE_SCORERS.keys():
            raise ValueError(
                'Cannot register duplicate scorer for metric ({})'.format(
                    metric))
        if not issubclass(cls, Scorer):
            raise ValueError('scorer ({}: {}) must extend Scorer'.format(
                metric, cls.__name__))
        AVAILABLE_SCORERS[metric] = cls
        return cls

    return register_scorer_cls


# Auto import python files in this directory
scorer_dir = os.path.dirname(__file__)
for file in os.listdir(scorer_dir):
    path = os.path.join(scorer_dir, file)
    if not file.startswith('_') and not file.startswith('.') and (
            file.endswith('.py') or os.path.isdir(path)):
        file_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(
            'onmt.scorers.' + file_name)
