"""Module defining various utilities."""
from onmt.utils.misc import aeq, use_gpu
from onmt.utils.report_manager import ReportMgr, build_report_manager
from onmt.utils.statistics import Statistics
from onmt.utils.rnn_factory import rnn_factory
from onmt.utils.cnn_factory import StackedCNN
from onmt.utils.loss import build_loss_compute
