#!/usr/bin/env python

from setuptools import setup

setup(name='OpenNMT_py',
      description='A python implementation of OpenNMT',
      version='0.7.0',

      packages=['onmt', 'onmt.encoders', 'onmt.modules', 'onmt.tests',
                'onmt.translate', 'onmt.decoders', 'onmt.inputters',
                'onmt.models', 'onmt.utils'])
