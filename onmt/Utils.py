def aeq(base, *rest):
    """ Assert the first arg equals to each of the rest."""
    for a in rest[:]:
        assert a == base, "base(" + str(base) \
            + ") doesn't equals to each of " + str(rest)

def use_gpu(opt):
    return len(opt.gpuid) > 0
