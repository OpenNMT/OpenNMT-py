from pyhanlp import HanLP
from snownlp import SnowNLP
import pkuseg


def wrap_str_func(func):
    """
    Wrapper to apply str function to the proper key of return_dict.
    """
    def wrapper(some_dict):
        some_dict["seg"] = [func(item) for item in some_dict["seg"]]
        return some_dict
    return wrapper


# Chinese segmentation
@wrap_str_func
def zh_segmentator(line):
    return " ".join(pkuseg.pkuseg().cut(line))


# Chinese simplify -> Chinese traditional standard
@wrap_str_func
def zh_traditional_standard(line):
    return HanLP.convertToTraditionalChinese(line)


# Chinese simplify -> Chinese traditional (HongKong)
@wrap_str_func
def zh_traditional_hk(line):
    return HanLP.s2hk(line)


# Chinese simplify -> Chinese traditional (Taiwan)
@wrap_str_func
def zh_traditional_tw(line):
    return HanLP.s2tw(line)


# Chinese traditional -> Chinese simplify (v1)
@wrap_str_func
def zh_simplify(line):
    return HanLP.convertToSimplifiedChinese(line)


# Chinese traditional -> Chinese simplify (v2)
@wrap_str_func
def zh_simplify_v2(line):
    return SnowNLP(line).han
