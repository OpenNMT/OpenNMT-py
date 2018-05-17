import torch
import torch.nn as nn
import onmt
import onmt.io
import onmt.modules
from allennlp.commands.elmo import ElmoEmbedder
from test.Options import Opt

from test.aux_train import lazily_load_dataset, load_fields, DatasetLazyIter, make_dataset_iter,make_loss_compute


opt = Opt()


ee = ElmoEmbedder()
embeddings  = ee.embed_sentence("bitcoin alone has a sixty percent share of global search".split())
path = '/Users/ugan/PycharmProjects/language-agnostic-bot/junk_corpus'
vocab = dict(torch.load(path+"/demo.vocab.pt"))
src_padding = vocab["src"].stoi[onmt.io.PAD_WORD]
tgt_padding = vocab["tgt"].stoi[onmt.io.PAD_WORD]


emb_size = 10
rnn_size = 6
# Specify the core model.
encoder_embeddings = onmt.modules.Embeddings(emb_size, len(vocab["src"]), word_padding_idx=src_padding)

encoder = onmt.modules.RNNEncoder(hidden_size=rnn_size, num_layers=1,
                                 rnn_type="LSTM", bidirectional=True,
                                 embeddings=encoder_embeddings)

decoder_embeddings = onmt.modules.Embeddings(emb_size, len(vocab["tgt"]),
                                             word_padding_idx=tgt_padding)
decoder = onmt.modules.InputFeedRNNDecoder(hidden_size=rnn_size, num_layers=1,
                                           bidirectional_encoder=True,
                                           rnn_type="LSTM", embeddings=decoder_embeddings)
model = onmt.modules.NMTModel(encoder, decoder)

# Specify the tgt word generator and loss computation module
model.generator = nn.Sequential(nn.Linear(rnn_size, len(vocab["tgt"])), nn.LogSoftmax())
loss = onmt.Loss.NMTLossCompute(model.generator, vocab["tgt"])

optim = onmt.Optim(method="sgd", lr=1, max_grad_norm=2)
optim.set_parameters(model.parameters())


# Load some data
data = torch.load(path + "/demo.train.1.pt")
valid_data = torch.load(path + "/demo.valid.1.pt")
data.load_fields(vocab)
valid_data.load_fields(vocab)
data.examples = data.examples[:100]

#use this make_dataset_iter, actually this lazily_load_dataset
# train_iter = onmt.io.OrderedIterator(
#                 dataset=data, batch_size=10,
#                 device=-1,
#                 repeat=False)


checkpoint = None
first_dataset = next(lazily_load_dataset("train", opt))
data_type = first_dataset.data_type

# Load fields generated from preprocess phase.
fields = load_fields(first_dataset, data_type, checkpoint, opt)

train_iter = make_dataset_iter(lazily_load_dataset("train", opt), fields, opt)

# valid_iter = onmt.io.OrderedIterator(
#                 dataset=valid_data, batch_size=10,
#                 device=-1,
#                 train=False)

valid_iter = make_dataset_iter(lazily_load_dataset("valid", opt), fields, opt, is_train=False)

train_loss = make_loss_compute(model, fields["tgt"].vocab, opt)
valid_loss = make_loss_compute(model, fields["tgt"].vocab, opt)
trainer = onmt.Trainer(model, train_loss, valid_loss, loss, loss, optim)

def report_func(*args):
    stats = args[-1]
    stats.output(args[0], args[1], 10, 0)
    return stats

for epoch in range(2):
    trainer.train(train_iter, epoch, report_func)
    val_stats = trainer.validate()

    print("Validation")
    val_stats.output(epoch, 11, 10, 0)
    trainer.epoch_step(val_stats.ppl(), epoch)

