'''
Usage:

	python join_jfleg_dev_set.py --path "../../../datasets/jfleg/dev/" --ref 4

Will generate 2 new files in **--path**: "dev.all.src" and "dev.all.ref", which:
	1. Duplicate aand concat all sentences in "dev.src" for **--ref** times
	2. Concat the files dev.ref[0-N] to a single file, where N is specified by **--ref**

'''

import argparse
import os

if __name__ == '__main__':

	argparser = argparse.ArgumentParser()
	argparser.add_argument('--path', dest="PATH")
	argparser.add_argument('--ref', dest="REF_COUNT", type=int)
	args = argparser.parse_args()

	src_input_path = os.path.join(args.PATH, 'dev.src')
	src_output_path = os.path.join(args.PATH, 'dev.all.src')
	ref_input_paths = [os.path.join(args.PATH, 'dev.ref'+str(i)) for i in range(args.REF_COUNT)]
	ref_output_path = os.path.join(args.PATH, 'dev.all.ref')

	# Pre-process src
	with open(src_input_path, 'r', encoding='utf-8') as src_input:
		with open(src_output_path, 'w', encoding='utf-8') as src_output:
			sents = ''.join(src_input.readlines())
			for i in range(args.REF_COUNT):
				src_output.write(sents)

	# Pre-process references
	with open(ref_output_path, 'w', encoding='utf-8') as ref_output:

		for ref_input_path in ref_input_paths:
			with open(ref_input_path, 'r', encoding='utf-8') as ref_input:
				sents = ''.join(ref_input.readlines())
				ref_output.write(sents)
