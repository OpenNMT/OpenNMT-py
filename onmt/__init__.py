""" Main entry point of the ONMT library """
import onmt.inputters
import onmt.encoders
import onmt.decoders
import onmt.models
import onmt.utils
import onmt.modules
import sys
import onmt.utils.optimizers

onmt.utils.optimizers.Optim = onmt.utils.optimizers.Optimizer
sys.modules["onmt.Optim"] = onmt.utils.optimizers

# For Flake
__all__ = [
    onmt.inputters,
    onmt.encoders,
    onmt.decoders,
    onmt.models,
    onmt.utils,
    onmt.modules,
]

__version__ = "3.5.0"
