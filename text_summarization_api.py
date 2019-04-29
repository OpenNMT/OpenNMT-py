import os

# Text Summarization
# Runnign existing translate api for conversion 
def summarize(input_file, use_gpu = False):
  gpu_string = ""
  print('IN Summarization')
  if use_gpu:
    gpu_string = "-gpu 0"
  out_file = input_file + '_summarized.txt'
  os.system("python translate.py {} -log_file_level NOTSET --report_every  -batch_size 1  -beam_size 5  -model summary/gigaword_copy_acc_51.78_ppl_11.71_e20.pt  -src {}  -share_vocab  -output {}  -min_length 6  -verbose  -stepwise_penalty  -coverage_penalty summary  -beta 5  -length_penalty wu  -alpha 0.9  -block_ngram_repeat 3  -ignore_when_blocking \".\" ".format(gpu_string, input_file, out_file))
  try:  
    summarized = open(out_file, 'r')
    sum_text = summarized.readlines()
    summarized.close()
    output   = ". ".join(sum_text).replace('\n', '')
    summarized = open(out_file, 'w')
    summarized.write(output)
    summarized.close()
  except FileNotFoundError:
    print('Summarized file not found')

summarize('test.txt')