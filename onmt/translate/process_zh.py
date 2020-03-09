from pyhanlp import HanLP
from snownlp import SnowNLP
import pkuseg


def update_seg(func):
    def wrapper(line):
        line["seg"] = [func(item) for item in line["seg"]]
        return line
    return wrapper


# Chinese segmentation
@update_seg
def zh_segmentator(line):
    return " ".join(pkuseg.pkuseg().cut(line))


# Chinese simplify -> Chinese traditional standard
@update_seg
def zh_traditional_standard(line):
    return HanLP.convertToTraditionalChinese(line)


# Chinese simplify -> Chinese traditional (HongKong)
@update_seg
def zh_traditional_hk(line):
    return HanLP.s2hk(line)


# Chinese simplify -> Chinese traditional (Taiwan)
@update_seg
def zh_traditional_tw(line):
    return HanLP.s2tw(line)


# Chinese traditional -> Chinese simplify (v1)
@update_seg
def zh_simplify(line):
    return HanLP.convertToSimplifiedChinese(line)


# Chinese traditional -> Chinese simplify (v2)
@update_seg
def zh_simplify_v2(line):
    return SnowNLP(line).han
