from pyhanlp import HanLP
from snownlp import SnowNLP
import pkuseg


# Chinese segmentation
def ZH_segmentator(line):
    segments = pkuseg.pkuseg().cut(line)
    sequence = " ".join(segments)
    return sequence


# Chinese simplify -> Chinese traditional standard
def ZH_traditional_standard(line):
    line_traditional = HanLP.convertToTraditionalChinese(line)
    return line_traditional


# Chinese simplify -> Chinese traditional (HongKong)
def ZH_traditional_HK(line):
    line_traditional = HanLP.s2hk(line)
    return line_traditional


# Chinese simplify -> Chinese traditional (Taiwan)
def ZH_traditional_TW(line):
    line_traditional = HanLP.s2tw(line)
    return line_traditional


# Chinese traditional -> Chinese simplify (v1)
def ZH_simplify(u_line):
    simple_line = HanLP.convertToSimplifiedChinese(u_line)
    return simple_line


# Chinese traditional -> Chinese simplify (v2)
def ZH_simplify_v2(u_line):
    simple_line = SnowNLP(u_line).han
    return simple_line
