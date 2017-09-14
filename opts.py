import argparse
from onmt.modules.SRU import CheckSRU


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """
    # Model options
    parser.add_argument('-model_type', default='text',
                        help="Type of encoder to use. Options are [text|img].")
    # Embedding Options
    parser.add_argument('-word_vec_size', type=int, default=-1,
                        help='Word embedding for both.')
    parser.add_argument('-src_word_vec_size', type=int, default=500,
                        help='Src word embedding sizes')
    parser.add_argument('-tgt_word_vec_size', type=int, default=500,
                        help='Tgt word embedding sizes')

    parser.add_argument('-feat_merge', type=str, default='concat',
                        choices=['concat', 'sum', 'mlp'],
                        help='Merge action for the features embeddings')
    parser.add_argument('-feat_vec_size', type=int, default=-1,
                        help="""If specified, feature embedding sizes
                        will be set to this. Otherwise, feat_vec_exponent
                        will be used.""")
    parser.add_argument('-feat_vec_exponent', type=float, default=0.7,
                        help="""If -feat_merge_size is not set, feature
                        embedding sizes will be set to N^feat_vec_exponent
                        where N is the number of values the feature takes.""")
    parser.add_argument('-position_encoding', action='store_true',
                        help='Use a sin to mark relative words positions.')
    parser.add_argument('-share_decoder_embeddings', action='store_true',
                        help='Share the word and out embeddings for decoder.')

    # RNN Options
    parser.add_argument('-encoder_type', type=str, default='rnn',
                        choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
                        help="""Type of encoder layer to use.""")
    parser.add_argument('-decoder_type', type=str, default='rnn',
                        choices=['rnn', 'transformer', 'cnn'],
                        help='Type of decoder layer to use.')

    parser.add_argument('-layers', type=int, default=-1,
                        help='Number of layers in enc/dec.')
    parser.add_argument('-enc_layers', type=int, default=2,
                        help='Number of layers in the encoder')
    parser.add_argument('-dec_layers', type=int, default=2,
                        help='Number of layers in the decoder')

    parser.add_argument('-cnn_kernel_width', type=int, default=3,
                        help="""Size of windows in the cnn, the kernel_size is
                         (cnn_kernel_width, 1) in conv layer""")

    parser.add_argument('-rnn_size', type=int, default=500,
                        help='Size of LSTM hidden states')
    parser.add_argument('-input_feed', type=int, default=1,
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")

    parser.add_argument('-rnn_type', type=str, default='LSTM',
                        choices=['LSTM', 'GRU', 'SRU'],
                        action=CheckSRU,
                        help="""The gate type to use in the RNNs""")
    # parser.add_argument('-residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")

    parser.add_argument('-brnn', action="store_true",
                        help="Deprecated, use `encoder_type`.")
    parser.add_argument('-brnn_merge', default='concat',
                        choices=['concat', 'sum'],
                        help="Merge action for the bidir hidden states")

    parser.add_argument('-context_gate', type=str, default=None,
                        choices=['source', 'target', 'both'],
                        help="""Type of context gate to use.
                        Do not select for no context gate.""")

    # Attention options
    parser.add_argument('-global_attention', type=str, default='general',
                        choices=['dot', 'general', 'mlp'],
                        help="""The attention type to use:
                        dotprot or general (Luong) or MLP (Bahdanau)""")

    # Genenerator and loss options.
    parser.add_argument('-copy_attn', action="store_true",
                        help='Train copy attention layer.')
    parser.add_argument('-copy_attn_force', action="store_true",
                        help='When available, train to copy.')
    parser.add_argument('-coverage_attn', action="store_true",
                        help='Train a coverage attention layer.')
    parser.add_argument('-lambda_coverage', type=float, default=1,
                        help='Lambda value for coverage.')


def train_opts(parser):
    parser.add_argument('-save_model', default='model',
                        help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""")
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")
    # GPU
    parser.add_argument('-gpuid', default=[], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-seed', type=int, default=-1,
                        help="""Random seed used for the experiments
                        reproducibility.""")

    # Init options
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init).
                        Use 0 to not use initialization""")

    # Pretrained word vectors
    parser.add_argument('-pre_word_vecs_enc',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""")
    parser.add_argument('-pre_word_vecs_dec',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the decoder side.
                        See README for specific formatting instructions.""")
    # Fixed word vectors
    parser.add_argument('-fix_word_vecs_enc',
                        action='store_true',
                        help="Fix word embeddings on the encoder side.")
    parser.add_argument('-fix_word_vecs_dec',
                        action='store_true',
                        help="Fix word embeddings on the encoder side.")

    # Optimization options
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-max_generator_batches', type=int, default=32,
                        help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but
                        uses more memory.""")
    parser.add_argument('-epochs', type=int, default=13,
                        help='Number of training epochs')
    parser.add_argument('-optim', default='sgd',
                        choices=['sgd', 'adagrad', 'adadelta', 'adam'],
                        help="""Optimization method.""")
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.3,
                        help="Dropout probability; applied in LSTM stacks.")
    parser.add_argument('-truncated_decoder', type=int, default=0,
                        help="""Truncated bptt.""")
    # learning rate
    parser.add_argument('-learning_rate', type=float, default=1.0,
                        help="""Starting learning rate. If adagrad/adadelta/adam
                        is used, then this is the global learning rate.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this
                        epoch""")
    parser.add_argument('-start_checkpoint_at', type=int, default=0,
                        help="""Start checkpointing every epoch after and including
                        this epoch""")
    parser.add_argument('-decay_method', type=str, default="",
                        choices=['noam'], help="Use a custom decay rate.")
    parser.add_argument('-warmup_steps', type=int, default=4000,
                        help="""Number of warmup steps for custom decay.""")

    parser.add_argument('-report_every', type=int, default=50,
                        help="Print stats at this interval.")
    parser.add_argument('-exp_host', type=str, default="",
                        help="Send logs to this crayon server.")
    parser.add_argument('-exp', type=str, default="",
                        help="Name of the experiment for logging.")


def preprocess_opts(parser):
    # Dictionary Options
    parser.add_argument('-src_vocab_size', type=int, default=50000,
                        help="Size of the source vocabulary")
    parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                        help="Size of the target vocabulary")

    parser.add_argument('-src_words_min_frequency', type=int, default=0)
    parser.add_argument('-tgt_words_min_frequency', type=int, default=0)

    # Truncation options
    parser.add_argument('-src_seq_length', type=int, default=50,
                        help="Maximum source sequence length")
    parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                        help="Truncate source sequence length.")
    parser.add_argument('-tgt_seq_length', type=int, default=50,
                        help="Maximum target sequence length to keep.")
    parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                        help="Truncate target sequence length.")

    # Data processing options
    parser.add_argument('-shuffle', type=int, default=1,
                        help="Shuffle data")
    parser.add_argument('-lower', action='store_true', help='lowercase data')

    # Options most relevant to summarization
    parser.add_argument('-dynamic_dict', action='store_true',
                        help="Create dynamic dictionaries")
    parser.add_argument('-share_vocab', action='store_true',
                        help="Share source and target vocabulary")


def add_md_help_argument(parser):
    parser.add_argument('-md', action=MarkdownHelpAction,
                        help='print Markdown-formatted help text and exit.')


# MARKDOWN boilerplate

# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
class MarkdownHelpFormatter(argparse.HelpFormatter):
    """A really bare-bones argparse help formatter that generates valid markdown.
    This will generate something like:
    usage
    # **section heading**:
    ## **--argument-one**
    ```
    argument-one help text
    ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        usage_text = super(MarkdownHelpFormatter, self)._format_usage(
            usage, actions, groups, prefix)
        return '\n```\n%s\n```\n\n' % usage_text

    def format_help(self):
        self._root_section.heading = '# %s' % self._prog
        return super(MarkdownHelpFormatter, self).format_help()

    def start_section(self, heading):
        super(MarkdownHelpFormatter, self).start_section('## **%s**' % heading)

    def _format_action(self, action):
        lines = []
        action_header = self._format_action_invocation(action)
        lines.append('### **%s** ' % action_header)
        if action.help:
            lines.append('')
            lines.append('```')
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
            lines.append('```')
        lines.extend(['', ''])
        return '\n'.join(lines)


class MarkdownHelpAction(argparse.Action):
    def __init__(self, option_strings,
                 dest=argparse.SUPPRESS, default=argparse.SUPPRESS,
                 **kwargs):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()
