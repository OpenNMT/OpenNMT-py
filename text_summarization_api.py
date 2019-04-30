import os
# Text Summarization
# Runnign existing translate api for conversion 
def summarize(input_file, suffix = '_summarized.txt', use_gpu = False):
  gpu_string = ""
  if use_gpu:
    gpu_string = "-gpu 0"

  # constants
  model = 'available_models/gigaword_copy_acc_51.78_ppl_11.71_e20.pt'
  out_file = input_file + suffix
  os.system("python translate.py {}  -batch_size 1  -beam_size 5  -model {}  -src {}  -share_vocab  -output {}  -min_length 6  -stepwise_penalty  -coverage_penalty summary  -beta 5  -length_penalty wu  -alpha 0.9  -block_ngram_repeat 3  -ignore_when_blocking \".\" ".format(gpu_string, model, input_file, out_file))
  try:  
    summarized = open(out_file, 'r')
    sum_text = summarized.readlines()
    summarized.close()
    output   = ". ".join(sum_text).replace('\n', '')
    summarized = open(out_file, 'w')
    summarized.write(output)
    summarized.close()
  except IOError:
    print('Summarized file not found')

# summarize('test.txt')