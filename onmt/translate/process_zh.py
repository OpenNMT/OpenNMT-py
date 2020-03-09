from pyhanlp import HanLP
from snownlp import SnowNLP
import pkuseg


# Chinese segmentation
def zh_segmentator(line):
	line["seg"] = [" ".join(pkuseg.pkuseg().cut(item))
	               for item in line["seg"]]
    return line


# Chinese simplify -> Chinese traditional standard
def zh_traditional_standard(line):
	line["seg"] = [HanLP.convertToTraditionalChinese(item)
				   for item in line["seg"]]
    return line


# Chinese simplify -> Chinese traditional (HongKong)
def zh_traditional_hk(line):
	line["seg"] = [HanLP.s2hk(item) for item in line["seg"]]
    return line


# Chinese simplify -> Chinese traditional (Taiwan)
def zh_traditional_tw(line):
	line["seg"] = [HanLP.s2tw(item) for item in line["seg"]]
    return HanLP.s2tw(line)


# Chinese traditional -> Chinese simplify (v1)
def zh_simplify(line):
	line["seg"] = [HanLP.convertToSimplifiedChinese(item)
	               for item in line["seg"]]
    return line


# Chinese traditional -> Chinese simplify (v2)
def zh_simplify_v2(line):
	line["seg"] = [SnowNLP(item).han for item in line["seg"]]
    return line
