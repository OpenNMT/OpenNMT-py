import onmt
import onmt.Markdown
import argparse
import torch


def loadImageLibs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms


parser = argparse.ArgumentParser(description='preprocess.py')
onmt.Markdown.add_md_help_argument(parser)

# **Preprocess Options**

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-src_type', default="bitext",
                    choices=["bitext", "monotext", "img"],
                    help="""Type of the source input.
                         This affects all the subsequent operations
                         Options are [bitext|monotext|img].""")
parser.add_argument('-src_img_dir', default=".",
                    help="Location of source images")


parser.add_argument('-train',
                    help="""Path to the monolingual training data""")
parser.add_argument('-train_src', required=False,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=False,
                    help="Path to the training target data")
parser.add_argument('-valid',
                    help="""Path to the monolingual validation data""")
parser.add_argument('-valid_src', required=False,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=False,
                    help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")

parser.add_argument('-src_seq_length', type=int, default=50,
                    help="Maximum source sequence length")
parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                    help="Truncate source sequence length.")
parser.add_argument('-tgt_seq_length', type=int, default=50,
                    help="Maximum target sequence length to keep.")
parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                    help="Truncate target sequence length.")

parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)


def makeVocabulary(filename, size):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                      lower=opt.lower)

    with open(filename) as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize)

        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeBilingualData(srcFile, tgtFile, srcDicts, tgtDicts):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile)
    tgtF = open(tgtFile)

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: src and tgt do not have the same # of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            continue

        srcWords = sline.split()
        tgtWords = tline.split()

        if len(srcWords) <= opt.src_seq_length \
           and len(tgtWords) <= opt.tgt_seq_length:

            # Check truncation condition.
            if opt.src_seq_length_trunc != 0:
                srcWords = srcWords[:opt.src_seq_length_trunc]
            if opt.tgt_seq_length_trunc != 0:
                tgtWords = tgtWords[:opt.tgt_seq_length_trunc]

            if opt.src_type == "bitext":
                src += [srcDicts.convertToIdx(srcWords,
                                              onmt.Constants.UNK_WORD)]
            elif opt.src_type == "img":
                loadImageLibs()
                src += [transforms.ToTensor()(
                    Image.open(opt.src_img_dir + "/" + srcWords[0]))]

            tgt += [tgtDicts.convertToIdx(tgtWords,
                                          onmt.Constants.UNK_WORD,
                                          onmt.Constants.BOS_WORD,
                                          onmt.Constants.EOS_WORD)]
            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    print(('Prepared %d sentences ' +
          '(%d ignored due to length == 0 or src len > %d or tgt len > %d)') %
          (len(src), ignored, opt.src_seq_length, opt.tgt_seq_length))

    return src, tgt


def makeMonolingualData(srcFile, srcDicts):
    src = []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s ...' % (srcFile))

    with open(srcFile) as srcF:
        for sline in srcF:
            sline = sline.strip()

            # source and/or target are empty
            if sline == "":
                print('WARNING: ignoring an empty line ('+str(count+1)+')')
                continue

            srcWords = sline.split()

            if len(srcWords) <= opt.src_seq_length:

                # Check truncation condition.
                if opt.src_seq_length_trunc != 0:
                    srcWords = srcWords[:opt.src_seq_length_trunc]

                src += [srcDicts.convertToIdx(srcWords,
                                              onmt.Constants.UNK_WORD,
                                              onmt.Constants.BOS_WORD,
                                              onmt.Constants.EOS_WORD)]
                sizes += [len(srcWords)]
            else:
                ignored += 1

            count += 1

            if count % opt.report_every == 0:
                print('... %d sentences prepared' % count)

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]

    print(('Prepared %d sentences ' +
          '(%d ignored due to length == 0 or src len > %d)') %
          (len(src), ignored, opt.src_seq_length))

    return src

def main():

    if opt.src_type in ['bitext', 'img']:
        assert None not in [opt.train_src, opt.train_tgt,
                               opt.valid_src, opt.valid_tgt], \
            "With source type %s the following parameters are" \
            "required: -train_src, -train_tgt, " \
            "-valid_src, -valid_tgt" % (opt.src_type)

    elif opt.src_type == 'monotext':
        assert None not in [opt.train, opt.valid], \
            "With source type monotext the following " \
            "parameters are required: -train, -valid"

    dicts = {}
    dicts['src'] = onmt.Dict()
    if opt.src_type == 'bitext':
        dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                                      opt.src_vocab_size)
        dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                  opt.tgt_vocab_size)

    elif opt.src_type == 'monotext':
        dicts['src'] = initVocabulary('source', opt.train, opt.src_vocab,
                                      opt.src_vocab_size)

    elif opt.src_type == 'img':
        dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                      opt.tgt_vocab_size)

    print('Preparing training ...')
    train = {}
    valid = {}

    if opt.src_type in ['bitext', 'img']:
        train['src'], train['tgt'] = makeBilingualData(opt.train_src, opt.train_tgt,
                                          dicts['src'], dicts['tgt'])

        print('Preparing validation ...')
        valid['src'], valid['tgt'] = makeBilingualData(opt.valid_src, opt.valid_tgt,
                                              dicts['src'], dicts['tgt'])

    elif opt.src_type == 'monotext':
        train = makeMonolingualData(opt.train, dicts['src'])
        print('Preparing validation ...')
        valid = makeMonolingualData(opt.valid, dicts['src'])

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.src_type in ['bitext', 'img'] and opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')

    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'type':  opt.src_type,
                 'train': train,
                 'valid': valid}
    torch.save(save_data, opt.save_data + '.train.pt')


if __name__ == "__main__":
    main()
