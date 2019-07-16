from pyhanlp import HanLP
from snownlp import SnowNLP
import pkuseg


# Chinese segmentation
def ZH_segmentator(line):
    return " ".join(pkuseg.pkuseg().cut(line))


# Chinese simplify -> Chinese traditional standard
def ZH_traditional_standard(line):
    return HanLP.convertToTraditionalChinese(line)


# Chinese simplify -> Chinese traditional (HongKong)
def ZH_traditional_HK(line):
    return HanLP.s2hk(line)


# Chinese simplify -> Chinese traditional (Taiwan)
def ZH_traditional_TW(line):
    return HanLP.s2tw(line)


# Chinese traditional -> Chinese simplify (v1)
def ZH_simplify(line):
    return HanLP.convertToSimplifiedChinese(line)


# Chinese traditional -> Chinese simplify (v2)
def ZH_simplify_v2(line):
    return SnowNLP(line).han
