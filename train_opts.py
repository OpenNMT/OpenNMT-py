def add_model_arguments(parser):
    # Model options
    parser.add_argument('-layers', type=int, default=2,
                        help='Number of layers in the LSTM encoder/decoder')
    parser.add_argument('-rnn_size', type=int, default=500,
                        help='Size of LSTM hidden states')
    parser.add_argument('-word_vec_size', type=int, default=500,
                        help='Word embedding sizes')
    parser.add_argument('-feature_vec_size', type=int, default=100,
                        help='Feature vec sizes')

    parser.add_argument('-input_feed', type=int, default=1,
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")
    parser.add_argument('-rnn_type', type=str, default='LSTM',
                        choices=['LSTM', 'GRU'],
                        help="""The gate type to use in the RNNs""")
    parser.add_argument('-brnn', action='store_true',
                        help='Use a bidirectional encoder')
    parser.add_argument('-brnn_merge', default='concat',
                        help="""Merge action for the bidirectional hidden states:
                        [concat|sum]""")
    parser.add_argument('-copy_attn', action="store_true",
                        help='Train copy attention layer.')
    parser.add_argument('-copy_attn_force', action="store_true",
                        help="""Train copy attention layer to copy even
                        if word is in the src vocab.
                        .""")

    parser.add_argument('-coverage_attn', action="store_true",
                        help='Train a coverage attention layer.')
    parser.add_argument('-lambda_coverage', type=float, default=1,
                        help='Lambda value for coverage.')

    parser.add_argument('-encoder_layer', type=str, default='rnn',
                        help="""Type of encoder layer to use.
                        Options: [rnn|mean|transformer]""")
    parser.add_argument('-decoder_layer', type=str, default='rnn',
                        help='Type of decoder layer to use. [rnn|transformer]')
    parser.add_argument('-context_gate', type=str, default=None,
                        choices=['source', 'target', 'both'],
                        help="""Type of context gate to use [source|target|both].
                        Do not select for no context gate.""")
    parser.add_argument('-attention_type', type=str, default='dot',
                        choices=['dot', 'mlp'],
                        help="""The attention type to use:
                        dotprot (Luong) or MLP (Bahdanau)""")
    parser.add_argument('-encoder_type', default='text',
                        help="Type of encoder to use. Options are [text|img].")
    parser.add_argument('-dropout', type=float, default=0.3,
                        help='Dropout probability.')
    parser.add_argument('-position_encoding', action='store_true',
                        help='Use a sinusoids for words positions.')
    parser.add_argument('-share_decoder_embeddings', action='store_true',
                        help='Share the word and softmax embeddings..')
    parser.add_argument('-share_embeddings', action='store_true',
                        help='Share word embeddings between encoder/decoder')


def add_optim_arguments(parser):
    # Optimization options
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-max_generator_batches', type=int, default=32,
                        help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster,
                        but uses more memory.""")
    parser.add_argument('-epochs', type=int, default=13,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init).
                        Use 0 to not use initialization""")
    parser.add_argument('-optim', default='sgd',
                        help="""Optimization method.
                        [sgd|adagrad|adadelta|adam]""")
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm
                        """)
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
                        help="""Start checkpointing every epoch after and including this
                        epoch""")
    parser.add_argument('-decay_method', type=str, default="",
                        help="""Use a custom learning rate
                        decay [restart|noam]""")
    parser.add_argument('-warmup_steps', type=int, default=4000,
                        help="""Number of warmup steps for custom decay.""")
