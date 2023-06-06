import os
import torch
import re
from collections import deque
from onmt.utils.logging import logger
from onmt.inputters.inputter import vocabs_to_dict
from onmt.modules.lora import lora_state_dict


def build_model_saver(model_opt, opt, model, vocabs, optim):
    # _check_save_model_path
    save_model_path = os.path.abspath(opt.save_model)
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    model_saver = ModelSaver(
        opt.save_model,
        model,
        model_opt,
        vocabs,
        optim,
        opt.keep_checkpoint,
    )
    return model_saver


def load_checkpoint(ckpt_path):
    """Load checkpoint from `ckpt_path` if any else return `None`."""
    checkpoint = None
    if ckpt_path:
        logger.info("Loading checkpoint from %s" % ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))

        if "model" in checkpoint.keys():
            # This preserves backward-compat for models using customed layernorm
            def fix_key(s):
                s = re.sub(
                    r"(.*)\.layer_norm((_\d+)?)\.b_2", r"\1.layer_norm\2.bias", s
                )
                s = re.sub(
                    r"(.*)\.layer_norm((_\d+)?)\.a_2", r"\1.layer_norm\2.weight", s
                )
                return s

            checkpoint["model"] = {
                fix_key(k): v for k, v in checkpoint["model"].items()
            }
            # Force add_ffnbias to True if bias found in model w_1 keys
            for key in checkpoint["model"].keys():
                if "w_1.bias" in key:
                    checkpoint["opt"].add_ffnbias = True

        # fix v2 compatibility
        if "generator" in checkpoint.keys():
            if "0.weight" in checkpoint["generator"]:
                checkpoint["generator"]["weight"] = checkpoint["generator"].pop(
                    "0.weight"
                )
            if "0.bias" in checkpoint["generator"]:
                checkpoint["generator"]["bias"] = checkpoint["generator"].pop("0.bias")
        # end of patch for backward compatibility

    return checkpoint


class ModelSaverBase(object):
    """Base class for model saving operations

    Inherited classes must implement private methods:
    * `_save`
    * `_rm_checkpoint
    """

    def __init__(self, base_path, model, model_opt, vocabs, optim, keep_checkpoint=-1):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.vocabs = vocabs
        self.optim = optim
        self.last_saved_step = None
        self.keep_checkpoint = keep_checkpoint
        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)

    def save(self, step, moving_average=None):
        """Main entry point for model saver

        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """

        if self.keep_checkpoint == 0 or step == self.last_saved_step:
            return

        save_model = self.model
        if moving_average:
            model_params_data = []
            for avg, param in zip(moving_average, save_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data

        chkpt, chkpt_name = self._save(step, save_model)
        self.last_saved_step = step

        if moving_average:
            for param_data, param in zip(model_params_data, save_model.parameters()):
                param.data = param_data

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_name)

    def _save(self, step, model):
        """Save a resumable checkpoint.

        Args:
            step (int): step number
            model (nn.Module): torch model to save

        Returns:
            (object, str):

            * checkpoint: the saved object
            * checkpoint_name: name (or path) of the saved checkpoint
        """

        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """

        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """Simple model saver to filesystem"""

    def _save(self, step, model):
        if (
            hasattr(self.model_opt, "lora_layers")
            and len(self.model_opt.lora_layers) > 0
        ) or (
            hasattr(self.model_opt, "lora_embedding") and self.model_opt.lora_embedding
        ):
            model_state_dict = lora_state_dict(model, bias="lora_only")
            generator_state_dict = None
        else:
            model_state_dict = model.state_dict()
            model_state_dict = {
                k: v for k, v in model_state_dict.items() if "generator" not in k
            }
            generator_state_dict = model.generator.state_dict()

        checkpoint = {
            "model": model_state_dict,
            "generator": generator_state_dict,
            "vocab": vocabs_to_dict(self.vocabs),
            "opt": self.model_opt,
            "optim": self.optim.state_dict(),
        }

        logger.info("Saving checkpoint %s_step_%d.pt" % (self.base_path, step))
        checkpoint_path = "%s_step_%d.pt" % (self.base_path, step)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        if os.path.exists(name):
            os.remove(name)
