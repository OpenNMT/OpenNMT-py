"""Module defining various utilities."""
from onmt.utils.misc import aeq, use_gpu
from onmt.utils.report_manager import ReportMgr, build_report_manager
from onmt.utils.statistics import Statistics
from onmt.utils.optimizers import build_optim, MultipleOptimizer, \
    Optimizer

__all__ = ["aeq", "use_gpu", "ReportMgr",
           "build_report_manager", "Statistics",
           "build_optim", "MultipleOptimizer", "Optimizer"]
