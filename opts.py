import configargparse
from onmt.modules.SRU import CheckSRU


def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """
    # Model options
    parser.add('-model_type', '--model_type', default='text',
               help="Type of encoder to use. Options are [text|img].")
    # Embedding Options
    parser.add('-word_vec_size', '--word_vec_size', type=int, default=-1,
               help='Word embedding for both.')
    parser.add(
        '-src_word_vec_size',
        '--src_word_vec_size',
        type=int,
        default=500,
        help='Src word embedding sizes')
    parser.add(
        '-tgt_word_vec_size',
        '--tgt_word_vec_size',
        type=int,
        default=500,
        help='Tgt word embedding sizes')

    parser.add('-feat_merge', '--feat_merge', type=str, default='concat',
               choices=['concat', 'sum', 'mlp'],
               help='Merge action for the features embeddings')
    parser.add('-feat_vec_size', '--feat_vec_size', type=int, default=-1,
               help="""If specified, feature embedding sizes
                        will be set to this. Otherwise, feat_vec_exponent
                        will be used.""")
    parser.add('-feat_vec_exponent', '--feat_vec_exponent', type=float,
               default=0.7, help="""If -feat_merge_size is not set, feature
                        embedding sizes will be set to N^feat_vec_exponent
                        where N is the number of values the feature takes.""")
    parser.add(
        '-position_encoding',
        '--position_encoding',
        action='store_true',
        help='Use a sin to mark relative words positions.')
    parser.add(
        '-share_decoder_embeddings',
        '--share_decoder_embeddings',
        action='store_true',
        help='Share the word and out embeddings for decoder.')
    parser.add('-share_embeddings', '--share_embeddings', action='store_true',
               help="""Share the word embeddings between encoder
                         and decoder.""")

    # RNN Options
    parser.add('-encoder_type', '--encoder_type', type=str, default='rnn',
               choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
               help="""Type of encoder layer to use.""")
    parser.add('-decoder_type', '--decoder_type', type=str, default='rnn',
               choices=['rnn', 'transformer', 'cnn'],
               help='Type of decoder layer to use.')

    parser.add('-layers', '--layers', type=int, default=-1,
               help='Number of layers in enc/dec.')
    parser.add('-enc_layers', '--enc_layers', type=int, default=2,
               help='Number of layers in the encoder')
    parser.add('-dec_layers', '--dec_layers', type=int, default=2,
               help='Number of layers in the decoder')

    parser.add('-cnn_kernel_width', '--cnn_kernel_width', type=int, default=3,
               help="""Size of windows in the cnn, the kernel_size is
                         (cnn_kernel_width, 1) in conv layer""")

    parser.add('-rnn_size', '--rnn_size', type=int, default=500,
               help='Size of LSTM hidden states')
    parser.add('-input_feed', '--input_feed', type=int, default=1,
               help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")

    parser.add('-rnn_type', '--rnn_type', type=str, default='LSTM',
               choices=['LSTM', 'GRU', 'SRU'],
               action=CheckSRU,
               help="""The gate type to use in the RNNs""")
    # parser.add('-residual', '--residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")

    parser.add('-brnn', '--brnn', action="store_true",
                        help="Deprecated, use `encoder_type`.")
    parser.add('-brnn_merge', '--brnn_merge', default='concat',
               choices=['concat', 'sum'],
               help="Merge action for the bidir hidden states")

    parser.add('-context_gate', '--context_gate', type=str, default=None,
               choices=['source', 'target', 'both'],
               help="""Type of context gate to use.
                        Do not select for no context gate.""")

    # Attention options
    parser.add(
        '-global_attention',
        '--global_attention',
        type=str,
        default='general',
        choices=[
            'dot',
            'general',
            'mlp'],
        help="""The attention type to use:
                        dotprot or general (Luong) or MLP (Bahdanau)""")

    # Genenerator and loss options.
    parser.add('-copy_attn', '--copy_attn', action="store_true",
               help='Train copy attention layer.')
    parser.add('-copy_attn_force', '--copy_attn_force', action="store_true",
               help='When available, train to copy.')
    parser.add('-coverage_attn', '--coverage_attn', action="store_true",
               help='Train a coverage attention layer.')
    parser.add('-lambda_coverage', '--lambda_coverage', type=float, default=1,
               help='Lambda value for coverage.')


def preprocess_opts(parser):
    parser.add('-data_type', '--data_type', default="text",
               help="Type of the source input. Options are [text|img].")
    parser.add('-data_img_dir', '--data_img_dir', default=".",
               help="Location of source images")

    parser.add('-train_src', '--train_src', required=True,
               help="Path to the training source data")
    parser.add('-train_tgt', '--train_tgt', required=True,
               help="Path to the training target data")
    parser.add('-valid_src', '--valid_src', required=True,
               help="Path to the validation source data")
    parser.add('-valid_tgt', '--valid_tgt', required=True,
               help="Path to the validation target data")

    parser.add('-save_data', '--save_data', required=True,
               help="Output file for the prepared data")

    parser.add('-src_vocab', '--src_vocab',
               help="Path to an existing source vocabulary")
    parser.add('-tgt_vocab', '--tgt_vocab',
               help="Path to an existing target vocabulary")
    parser.add(
        '-features_vocabs_prefix',
        '--features_vocabs_prefix',
        type=str,
        default='',
        help="Path prefix to existing features vocabularies")
    parser.add('-seed', '--seed', type=int, default=3435,
                        help="Random seed")
    parser.add('-report_every', '--report_every', type=int, default=100000,
               help="Report status every this many sentences")

    # Dictionary Options
    parser.add('-src_vocab_size', '--src_vocab_size', type=int, default=50000,
               help="Size of the source vocabulary")
    parser.add('-tgt_vocab_size', '--tgt_vocab_size', type=int, default=50000,
               help="Size of the target vocabulary")

    parser.add('-src_words_min_frequency',
               '--src_words_min_frequency', type=int, default=0)
    parser.add('-tgt_words_min_frequency',
               '--tgt_words_min_frequency', type=int, default=0)

    # Truncation options
    parser.add('-src_seq_length', '--src_seq_length', type=int, default=50,
               help="Maximum source sequence length")
    parser.add(
        '-src_seq_length_trunc',
        '--src_seq_length_trunc',
        type=int,
        default=0,
        help="Truncate source sequence length.")
    parser.add('-tgt_seq_length', '--tgt_seq_length', type=int, default=50,
               help="Maximum target sequence length to keep.")
    parser.add(
        '-tgt_seq_length_trunc',
        '--tgt_seq_length_trunc',
        type=int,
        default=0,
        help="Truncate target sequence length.")

    # Data processing options
    parser.add('-shuffle', '--shuffle', type=int, default=1,
               help="Shuffle data")
    parser.add('-lower', '--lower', action='store_true', help='lowercase data')

    # Options most relevant to summarization
    parser.add('-dynamic_dict', '--dynamic_dict', action='store_true',
               help="Create dynamic dictionaries")
    parser.add('-share_vocab', '--share_vocab', action='store_true',
               help="Share source and target vocabulary")


def train_opts(parser):
    # Model loading/saving options
    parser.add('-data', '--data', required=True,
                        help="""Path prefix to the ".train.pt" and
                        ".valid.pt" file path from preprocess.py""")

    parser.add('-save_model', '--save_model', default='model',
               help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""")
    parser.add('-train_from', '--train_from', default='', type=str,
               help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")
    # GPU
    parser.add('-gpuid', '--gpuid', default=[], nargs='+', type=int,
               help="Use CUDA on the listed devices.")
    parser.add('-seed', '--seed', type=int, default=-1,
                        help="""Random seed used for the experiments
                        reproducibility.""")

    # Init options
    parser.add('-start_epoch', '--start_epoch', type=int, default=1,
               help='The epoch from which to start')
    parser.add('-param_init', '--param_init', type=float, default=0.1,
               help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init).
                        Use 0 to not use initialization""")

    # Pretrained word vectors
    parser.add('-pre_word_vecs_enc', '--pre_word_vecs_enc',
               help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""")
    parser.add('-pre_word_vecs_dec', '--pre_word_vecs_dec',
               help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the decoder side.
                        See README for specific formatting instructions.""")
    # Fixed word vectors
    parser.add('-fix_word_vecs_enc', '--fix_word_vecs_enc',
               action='store_true',
               help="Fix word embeddings on the encoder side.")
    parser.add('-fix_word_vecs_dec', '--fix_word_vecs_dec',
               action='store_true',
               help="Fix word embeddings on the encoder side.")

    # Optimization options
    parser.add('-batch_size', '--batch_size', type=int, default=64,
               help='Maximum batch size')
    parser.add(
        '-max_generator_batches',
        '--max_generator_batches',
        type=int,
        default=32,
        help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but
                        uses more memory.""")
    parser.add('-epochs', '--epochs', type=int, default=13,
               help='Number of training epochs')
    parser.add('-optim', '--optim', default='sgd',
               choices=['sgd', 'adagrad', 'adadelta', 'adam'],
               help="""Optimization method.""")
    parser.add('-max_grad_norm', '--max_grad_norm', type=float, default=5,
               help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")
    parser.add('-dropout', '--dropout', type=float, default=0.3,
               help="Dropout probability; applied in LSTM stacks.")
    parser.add(
        '-truncated_decoder',
        '--truncated_decoder',
        type=int,
        default=0,
        help="""Truncated bptt.""")
    # learning rate
    parser.add('-learning_rate', '--learning_rate', type=float, default=1.0,
               help="""Starting learning rate. If adagrad/adadelta/adam
                        is used, then this is the global learning rate.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add(
        '-learning_rate_decay',
        '--learning_rate_decay',
        type=float,
        default=0.5,
        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add('-start_decay_at', '--start_decay_at', type=int, default=8,
               help="""Start decaying every epoch after and including this
                        epoch""")
    parser.add(
        '-start_checkpoint_at',
        '--start_checkpoint_at',
        type=int,
        default=0,
        help="""Start checkpointing every epoch after and including
                        this epoch""")
    parser.add('-decay_method', '--decay_method', type=str, default="",
               choices=['noam'], help="Use a custom decay rate.")
    parser.add('-warmup_steps', '--warmup_steps', type=int, default=4000,
               help="""Number of warmup steps for custom decay.""")

    parser.add('-report_every', '--report_every', type=int, default=50,
               help="Print stats at this interval.")
    parser.add('-exp_host', '--exp_host', type=str, default="",
               help="Send logs to this crayon server.")
    parser.add('-exp', '--exp', type=str, default="",
               help="Name of the experiment for logging.")


def translate_opts(parser):
    parser.add('-model', '--model', required=True,
               help='Path to model .pt file')
    parser.add('-src', '--src', required=True,
               help="""Source sequence to decode (one line per
                        sequence)""")
    parser.add('-src_img_dir', '--src_img_dir', default="",
               help='Source image directory')
    parser.add('-tgt', '--tgt',
               help='True target sequence (optional)')
    parser.add('-output', '--output', default='pred.txt',
               help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add('-beam_size', '--beam_size', type=int, default=5,
               help='Beam size')
    parser.add('-batch_size', '--batch_size', type=int, default=30,
               help='Batch size')
    parser.add('-max_sent_length', '--max_sent_length', type=int, default=100,
               help='Maximum sentence length.')
    parser.add('-replace_unk', '--replace_unk', action="store_true",
               help="""Replace the generated UNK tokens with the
                        source token that had highest attention weight. If
                        phrase_table is provided, it will lookup the
                        identified source token and give the corresponding
                        target token. If it is not provided(or the identified
                        source token does not exist in the table) then it
                        will copy the source token""")
    parser.add('-verbose', '--verbose', action="store_true",
               help='Print scores and predictions for each sentence')
    parser.add('-attn_debug', '--attn_debug', action="store_true",
               help='Print best attn for each word')
    parser.add('-dump_beam', '--dump_beam', type=str, default="",
               help='File to dump beam information to.')
    parser.add('-n_best', '--n_best', type=int, default=1,
               help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add('-gpu', '--gpu', type=int, default=-1,
               help="Device to run on")
    # Options most relevant to summarization.
    parser.add('-dynamic_dict', '--dynamic_dict', action='store_true',
               help="Create dynamic dictionaries")
    parser.add('-share_vocab', '--share_vocab', action='store_true',
               help="Share source and target vocabulary")


def add_md_help_argument(parser):
    parser.add('-md', '--md', action=MarkdownHelpAction,
               help='print Markdown-formatted help text and exit.')


# MARKDOWN boilerplate

# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
class MarkdownHelpFormatter(configargparse.HelpFormatter):
    """
    A really bare-bones configargparse help formatter that generates valid
    markdown.  This will generate something like:
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


class MarkdownHelpAction(configargparse.Action):
    def __init__(self, option_strings,
                 dest=configargparse.SUPPRESS, default=configargparse.SUPPRESS,
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
