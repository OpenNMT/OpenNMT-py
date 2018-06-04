#!/usr/bin/env python
"""
    Tools to convert models created using OpenNMT-py < 04
    It requires to have sources of both versions of OpenNMT-py 

    Example:
        ```
        git clone https://github.com/OpenNMT/OpenNMT-py onmt

        # set a legacy repository using <0.4 commit
        cp -R onmt onmt_legacy
        cd onmt_legacy
        git reset hard 0ecec8b4c16fdec7d8ce2646a0ea47ab6535d308

        # get >= 0.4 
        cd ../onmt
        git remote add ubiqus https://github.com/Ubiqus/OpenNMT-py
        git pull ubiqus master

        # finally convert your file $model_file using this tool:
        cd ..
        ./03to04.py --onmt_legacy ./onmt_legacy \
                    --onmt_new ./onmt \
                    -f $model_file \
                    -o "onmt04_${model_file}"
        ```

    Tested with:
        - python 3.6
        - torch 0.4
"""
import argparse
import importlib
import torch
import sys


def convert(onmt_legacy, onmt_new, file_path, output_path):
    print("*** 0) Requirements")
    print("Using path:\n\t%s" % str(sys.path))
    torchtext = importlib.import_module('torchtext')
    print("With torchtext:\n\t%s" % str(torchtext))

    # Load model file with legacy ONMT
    print("\n\n*** 1) Loading model with legacy ONMT")
    sys.path = [onmt_legacy]
    print("\nUsing path:\n\t%s" % str(sys.path))

    onmt = importlib.import_module('onmt')
    print("With ONMT:\n\t%s" % str(onmt))
    print("Loading model file:\n\t%s" % file_path)
    m = torch.load(file_path, map_location='cpu')

    # Load 0.4 compatible onmt and modify the model
    print("\n\n*** 2) Converting model to ONMT>=0.4")
    sys.path = [onmt_new]
    print("\nUsing path:\n\t%s" % str(sys.path))
    importlib.reload(onmt)
    print("With ONMT:\n\t%s" % str(onmt))
    m['optim'].__class__ = onmt.utils.optimizers.Optimizer

    with open(output_path, 'wb') as output:
        print("Saving new model to:\n\t%s" % output_path)
        torch.save(m, output)

    print("\n\n***\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Makes model compatible with "
                                     + "OpenNMT-py + Pytorch 0.4")
    parser.add_argument("--onmt_legacy", "-l", required=True,
                        help="Path to ONMT < 0.4")
    parser.add_argument("--onmt_new", "-n", required=True,
                        help="Path to ONMT >= 0.4")
    parser.add_argument("--file_path", "-f", required=True,
                        help="Model file (.pt)")
    parser.add_argument("--output_path", "-o", required=True,
                        help="Output model file (.pt)")

    args = parser.parse_args()
    convert(**vars(args))
