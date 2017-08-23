import argparse
import copy
import unittest
import onmt
import torch
from train_opts import add_model_arguments, add_optim_arguments
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='preprocess.py')
onmt.Markdown.add_md_help_argument(parser)