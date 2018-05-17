from allennlp.modules.elmo import Elmo, batch_to_ids
import torch.nn as nn
import torch

class ElmoEmbedder(nn.Module):

    def __init__(self):
        super(ElmoEmbedder, self).__init__()
        self.options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        self.weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo_embedder = Elmo(self.options_file, self.weight_file, 1, dropout=0)
        self.embedding_size = self.elmo_embedder.get_output_dim()


    def load_pretrained_vectors(self, emb_file, fixed):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False

    def forward(self, batch):
        character_ids = batch_to_ids(batch)
        embeddings = self.elmo_embedder(character_ids)
        return embeddings['elmo_representations'][0]

