import torch
import gc


def get_total_memory(verbose=True):
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data')
                                        and torch.is_tensor(obj.data)):
                try:
                    csize = obj.clone().numpy().nbytes
                except:
                    csize = obj.clone().data.numpy().nbytes
                total += csize
        except:
            # Happens when hasattr is not implemented
            pass
    if verbose:
        print("total params in GB: {:.2f}".format(total / 1024 / 1024 / 1014))
    return total


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def use_gpu(opt):
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)
