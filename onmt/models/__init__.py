"""Module defining models."""
from onmt.models.model import NMTModel
from onmt.models.SRU import check_sru_requirement
CAN_USE_SRU = check_sru_requirement()
if CAN_USE_SRU:
    from onmt.models.SRU import SRU
