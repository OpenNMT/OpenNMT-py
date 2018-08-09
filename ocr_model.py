#!/usr/bin/env python
from __future__ import division, unicode_literals
import argparse
from onmt.translate.translator import build_translator
import numpy as np
import cv2
from skimage.filters import threshold_niblack, rank
from skimage.morphology import disk
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms

def nick_binarize(img_list):
    '''Binarize linecut images using two differently sized local threshold kernels

    Args:
        img_list: list of grayscale linecut images
    Returns:
        results: binarized images in the same order as the input'''

    results = []
    for img in img_list:
        assert img.ndim == 2, 'Image must be grayscale'
        height = img.shape[0]

        # Resize the images to 200 pixel height
        scaling_factor = 200/img.shape[0]
        new_w = int(scaling_factor*img.shape[1])
        new_h = int(scaling_factor*img.shape[0])
        img = cv2.resize(img, (new_w, new_h))

        # First pass thresholding
        th1 = threshold_niblack(img, 13, 0.0)

        # Second pass thresholding
        radius = 201
        structured_elem = disk(radius)
        th2 =  rank.otsu(img, structured_elem)

        # Masking
        img = (img > th1) | (img > th2)
        img = img.astype('uint8')*255

        img = cv2.resize(img, (int(img.shape[1] *height/img.shape[0]), height))
        results.append(img)

    return results


def enhance(img):
    imgEnhance = np.array(ImageEnhance.Contrast(Image.fromarray(img)).enhance(5))
    gray = cv2.cvtColor(imgEnhance, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((1, 1), np.uint8)
    # binary = cv2.dilate(binary,kernel,iterations = 1)

    imgEnhance = np.array(ImageEnhance.Contrast(Image.fromarray(img)).enhance(3))
    imgEnhance = cv2.fastNlMeansDenoisingColored(imgEnhance, None, 10, 10, 7, 21)

    imgEnhance[binary > 0] = img[binary > 0]

    # return np.array(Image.fromarray(imgEnhance).filter(ImageFilter.SHARPEN))

    return imgEnhance, binary

class OPT:
    gpu = 0
    data_type = "img"
    model = "demo-model_step_97000.pt"

    dropout = 0.0
    alpha = 0.00
    batch_size = 30
    beam_size = 3
    beta = -0.00
    block_ngram_repeat = 100
    coverage_penalty = 'avg'
    dump_beam = ''
    dynamic_dict = False
    ignore_when_blocking = []
    length_penalty = 'avg'
    max_length = 100
    max_sent_length = None
    min_length = 0
    n_best = 2
    output = 'pred.txt'
    replace_unk = False
    report_bleu = False
    report_rouge = False
    sample_rate = 16000
    share_vocab = True
    src = 'data/im2text/src-test.txt'
    src_dir = 'data/'
    stepwise_penalty = True
    tgt = None
    window = 'hamming'
    window_size = 0.02
    window_stride = 0.01

    # Debug
    # verbose = True
    # attn_debug = True
    # fast = False
    # report_score = True

    # Production
    verbose = False
    attn_debug = False
    fast = False
    report_score = False

class Img2LatexModel:
    def __init__(self):
        self.opt = OPT()

        self.model = build_translator(self.opt,report_score= self.opt.report_score)
        print("Initalized model")

    def normalize(self, img):
        # add_padding(img, padding=10)
        min_h = 50
        max_h = 75
        if (img.shape[0] < min_h):
            # img = cv2.resize(img, (int(img.shape[1]*min_h/img.shape[0]), min_h))
            img = Image.fromarray(img).resize((int(img.shape[1] * min_h / img.shape[0]), min_h), Image.ANTIALIAS)
        elif (img.shape[0] > max_h):
            # img = cv2.resize(img, (int(img.shape[1] * max_h / img.shape[0]), max_h))
            img = Image.fromarray(img).resize((int(img.shape[1] * max_h / img.shape[0]), max_h), Image.ANTIALIAS)
        else:
            img = Image.fromarray(img)

        img = ImageEnhance.Contrast(img).enhance(5)
        return img

    def normalize_PIL(self, img):
        try:
            img = Image.fromarray(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY))
        except:
            pass
        # add_padding(img, padding=10)
        min_h = 50
        max_h = 75

        width, height = img.size
        if (height < min_h):
            # img = cv2.resize(img, (int(img.shape[1]*min_h/img.shape[0]), min_h))
            img = img.resize((int(width * min_h / height), min_h), Image.ANTIALIAS)

        elif (height > max_h):
            # img = cv2.resize(img, (int(img.shape[1] * max_h / img.shape[0]), max_h))
            img = img.resize((int(width * max_h / height), max_h), Image.ANTIALIAS)

        img = ImageEnhance.Contrast(img).enhance(1)

        return img

    def predict(self, imgs:[], batch_size = 1):
        #Need grayscale images
        #imgs = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in imgs]

        imgs = [(transforms.ToTensor()(self.normalize_PIL(img)),"xxx") for img in imgs]
        result = self.model.translate(src_data_iter=imgs, batch_size=batch_size, src_dir=self.opt.src_dir)[1]
        result = [[" ".join("".join(x.replace(r"\,", "").split()).split("\;")) for x in k] for k in result]
        print(result)

        result = result[0]
        if (len(result[1]) == 0):
            return [""]
        else:
            return [result[0]]

def main(opt):
    translator = build_translator(opt, report_score=True)
    translator.translate(opt.src_dir, opt.src, opt.tgt,
                         opt.batch_size, opt.attn_debug)

def add_padding(image, padding = 5, padding_left = 2):
    temp = np.ones((image.shape[0] + padding * 2, image.shape[1] + padding + padding_left), np.uint8) * 255
    temp[padding:padding + image.shape[0],padding_left:padding_left + image.shape[1]] = image
    return temp

if __name__ == "__main__":
    model = Img2LatexModel()

    import glob
    import shutil
    import os

    input_dir = "data/sjnk/images/".replace('/', os.path.sep)

    im_names = glob.glob(os.path.join(input_dir, '*.png')) + \
               glob.glob(os.path.join(input_dir, '*.jpg')) + \
               glob.glob(os.path.join(input_dir, '*.jpeg')) + \
               glob.glob(os.path.join(input_dir, '*.tif'))

    im_names = sorted(im_names)

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        result = model.predict([Image.fromarray(cv2.imread(im_name,0))])
        print(result)
