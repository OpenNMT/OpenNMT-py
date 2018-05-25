"""Module defining various utilities."""
from onmt.utils.misc import aeq, use_gpu
from onmt.utils.report_manager import ReportMgr, build_report_manager
from onmt.utils.statistics import Statistics
from onmt.utils.multi_utils import is_master, multi_init, \
    all_reduce_and_rescale_tensors

__all__ = ["aeq", "use_gpu", "ReportMgr", "build_report_manager", "Statistics",
           "is_master", "multi_init", "all_reduce_and_rescale_tensors"]
