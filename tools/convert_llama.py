#!/usr/bin/env python
# flake8: noqa
import json
import torch
import argparse
import pyonmttok
from argparse import Namespace
from onmt.inputters.inputter import vocabs_to_dict
from onmt.constants import DefaultTokens
from sentencepiece import SentencePieceProcessor
import os


class Tokenizer:
    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
        self.vocab = [self.sp_model.id_to_piece(i) for i in range(self.n_words)]  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help="""Path to the model directory""")
    parser.add_argument('--tokenizer_model', type=str, required=True,
                        help="""Path to the tokenizer model""")
    parser.add_argument('--output', type=str, required=True,
                        help="""Path to the model directory""")
    opt = parser.parse_args()

    checkpoint = torch.load(os.path.join(opt.model_dir,
                            "consolidated.00.pth"), map_location=torch.device('cpu'))

    params_json = os.path.join(opt.model_dir, "params.json")
    with open(params_json, encoding="utf-8") as fparam:
        params = json.load(fparam)

    onmt_cp = {}
    onmt_cp['model'] = {}

    decoder_layers = params['n_layers']
    src_word_vec_size = params['dim']
    tgt_word_vec_size = params['dim']
    hidden_size = params['dim']
    heads = params['n_heads']

    onmt_cp['model']['decoder.embeddings.make_embedding.emb_luts.0.weight'] =\
        checkpoint['tok_embeddings.weight']

    for i in range(decoder_layers):
        onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_keys.weight'] =\
            checkpoint['layers.' + str(i) + '.attention.wk.weight']

        onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_values.weight'] =\
            checkpoint['layers.'+ str(i) + '.attention.wv.weight']

        onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_query.weight'] =\
            checkpoint['layers.' + str(i) + '.attention.wq.weight']

        onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.self_attn.final_linear.weight'] =\
            checkpoint['layers.' + str(i) + '.attention.wo.weight']

        onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.layer_norm_1.weight'] =\
            checkpoint['layers.' + str(i) + '.attention_norm.weight'].clone()

        onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_1.weight'] =\
            checkpoint['layers.' + str(i) + '.feed_forward.w1.weight']
        onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_1.bias'] =\
            torch.zeros(onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_1.weight'].size(0), dtype=torch.float16)

        onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_2.weight'] =\
            checkpoint['layers.' + str(i) + '.feed_forward.w2.weight']
        onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_2.bias'] =\
            torch.zeros(onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_2.weight'].size(0), dtype=torch.float16)

        onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_3.weight'] =\
            checkpoint['layers.' + str(i) + '.feed_forward.w3.weight']
        onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_3.bias'] =\
            torch.zeros(onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_3.weight'].size(0), dtype=torch.float16)

        onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.layer_norm.weight'] =\
            checkpoint['layers.' + str(i) + '.ffn_norm.weight'].clone()
        onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.layer_norm.bias'] =\
            torch.zeros(onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.layer_norm.weight'].size(0), dtype=torch.float16)

    onmt_cp['model']['decoder.layer_norm.weight'] = checkpoint['norm.weight']
    onmt_cp['model']['decoder.layer_norm.bias'] = torch.zeros(onmt_cp['model']['decoder.layer_norm.weight'].size(0), dtype=torch.float16)

    onmt_cp['generator'] = {}
    onmt_cp['generator']['weight'] = checkpoint['output.weight']
    onmt_cp['generator']['bias'] = torch.zeros(onmt_cp['generator']['weight'].size(0), dtype=torch.float16)

    tokenizer = Tokenizer(model_path=opt.tokenizer_model)
    vocabs = {}
    vocab = tokenizer.vocab
    vocab[3] = DefaultTokens.PAD
    src_vocab = pyonmttok.build_vocab_from_tokens(
        vocab,
        maximum_size=tokenizer.n_words,
        special_tokens=['<unk>',
                        '<s>',
                        '</s>'
                        ])
    vocabs['src'] = src_vocab
    vocabs['tgt'] = src_vocab
    vocabs['data_task'] = 'lm'
    vocabs['decoder_start_token'] = '<s>'

    onmt_cp['vocab'] = {}
    onmt_cp['vocab'] = vocabs_to_dict(vocabs)

    if os.path.exists(os.path.join(opt.model_dir, "consolidated.01.pth")):
        checkpoint = torch.load(os.path.join(opt.model_dir,
                                "consolidated.01.pth"), map_location=torch.device('cpu'))

        onmt_cp['model']['decoder.embeddings.make_embedding.emb_luts.0.weight'] =\
            torch.cat((onmt_cp['model']['decoder.embeddings.make_embedding.emb_luts.0.weight'], checkpoint['tok_embeddings.weight']), dim=1)

        for i in range(decoder_layers):
            onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_keys.weight'] =\
                torch.cat((onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_keys.weight'], checkpoint['layers.' + str(i) + '.attention.wk.weight']), dim=0)

            onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_values.weight'] =\
                torch.cat((onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_values.weight'], checkpoint['layers.' + str(i) + '.attention.wv.weight']), dim=0)

            onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_query.weight'] =\
                torch.cat((onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_query.weight'], checkpoint['layers.' + str(i) + '.attention.wq.weight']), dim=0)

            onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.self_attn.final_linear.weight'] =\
                torch.cat((onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.self_attn.final_linear.weight'], checkpoint['layers.' + str(i) + '.attention.wo.weight']), dim=1)

            onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_1.weight'] =\
                torch.cat((onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_1.weight'], checkpoint['layers.' + str(i) + '.feed_forward.w1.weight']), dim=0)
            onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_1.bias'] =\
                torch.zeros(onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_1.weight'].size(0), dtype=torch.float16)

            onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_2.weight'] =\
                torch.cat((onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_2.weight'], checkpoint['layers.' + str(i) + '.feed_forward.w2.weight']), dim=1)
            onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_2.bias'] =\
                torch.zeros(onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_2.weight'].size(0), dtype=torch.float16)

            onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_3.weight'] =\
                torch.cat((onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_3.weight'], checkpoint['layers.' + str(i) + '.feed_forward.w3.weight']), dim=0)
            onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_3.bias'] =\
                torch.zeros(onmt_cp['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_3.weight'].size(0), dtype=torch.float16)

        onmt_cp['generator']['weight'] = torch.cat((onmt_cp['generator']['weight'], checkpoint['output.weight']), dim=0)
        onmt_cp['generator']['bias'] = torch.zeros(onmt_cp['generator']['weight'].size(0), dtype=torch.float16) 

    transformer_ff = onmt_cp['model']['decoder.transformer_layers.0.feed_forward.w_1.weight'].size(0)
    vocab_size = onmt_cp['generator']['weight'].size(0)

    onmt_cp['opt'] = Namespace(config='', save_config=None, data={}, skip_empty_level='silent', save_data='', overwrite=False, n_sample=0, dump_transforms=False, src_vocab='', tgt_vocab='', share_vocab=True, src_feats_vocab=None, src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, vocab_size_multiple=1, src_words_min_frequency=1, tgt_words_min_frequency=1, decoder_start_token='<s>', src_seq_length_trunc=None, tgt_seq_length_trunc=None, both_embeddings=None, src_embeddings=None, tgt_embeddings=None, embeddings_type=None, switchout_temperature=1.0, tokendrop_temperature=1.0, tokenmask_temperature=1.0, reversible_tokenization='joiner', prior_tokenization=False, src_subword_model='', tgt_subword_model='', src_subword_nbest=1, tgt_subword_nbest=1, src_subword_alpha=0.0, tgt_subword_alpha=0.0, src_subword_vocab='', tgt_subword_vocab='', src_vocab_threshold=0, tgt_vocab_threshold=0, src_subword_type='none', tgt_subword_type='none', src_onmttok_kwargs="{'mode': 'none'}", tgt_onmttok_kwargs="{'mode': 'none'}", src_seq_length=150, tgt_seq_length=150, src_prefix='', tgt_prefix='', permute_sent_ratio=0.0, rotate_ratio=0.0, insert_ratio=0.0, random_ratio=0.0, mask_ratio=0.0, mask_length='subword', poisson_lambda=3.0, replace_length=-1, src_word_vec_size=src_word_vec_size, tgt_word_vec_size=tgt_word_vec_size, word_vec_size=src_word_vec_size, share_decoder_embeddings=False, share_embeddings=True, position_encoding=False, update_vocab=False, feat_merge='concat', feat_vec_size=-1, feat_vec_exponent=0.7, model_task='lm', model_type='text', model_dtype='fp16', decoder_type='transformer_lm', freeze_encoder=False, freeze_decoder=False, layers=-1, dec_layers=decoder_layers, hidden_size=hidden_size, enc_hid_size=hidden_size, dec_hid_size=hidden_size, cnn_kernel_width=3, layer_norm='rms', pos_ffn_activation_fn='silu', input_feed=1, bridge=False, rnn_type='LSTM', context_gate=None, bridge_extra_node=True, bidir_edges=True, state_dim=hidden_size, n_edge_types=2, n_node=2, n_steps=2, src_ggnn_size=0, global_attention='general', global_attention_function='softmax', self_attn_type='scaled-dot', max_relative_positions=-1, heads=heads, transformer_ff=transformer_ff, aan_useffn=False, add_qkvbias=False, lambda_align=0.0, alignment_layer=-3, alignment_heads=0, full_context_alignment=False, copy_attn=False, copy_attn_type='general', generator_function='softmax', copy_attn_force=False, reuse_copy_attn=False, copy_loss_by_seqlength=False, coverage_attn=False, lambda_coverage=0.0, lm_prior_model=None, lm_prior_lambda=0.0, lm_prior_tau=1.0, loss_scale=0, apex_opt_level='', data_type='text', save_model='nllb', save_checkpoint_steps=5000, keep_checkpoint=50, gpu_ranks=[0], world_size=1, gpu_backend='nccl', gpu_verbose_level=0, master_ip='localhost', master_port=10000, seed=1234, param_init=0.0, param_init_glorot=True, train_from='', reset_optim='none', pre_word_vecs_enc=None, pre_word_vecs_dec=None, freeze_word_vecs_enc=False, freeze_word_vecs_dec=False, num_workers=4, batch_size=8192, batch_size_multiple=1, batch_type='tokens', normalization='tokens', accum_count=[4], accum_steps=[0], valid_steps=5000, valid_batch_size=4096, train_steps=100000, single_pass=False, early_stopping=0, early_stopping_criteria=None, optim='fusedadam', adagrad_accumulator_init=0, max_grad_norm=0.0, dropout=[0.1], attention_dropout=[0.1], dropout_steps=[0], truncated_decoder=0, adam_beta1=0.9, adam_beta2=0.998, label_smoothing=0.1, average_decay=0.0005, average_every=1, learning_rate=2.0, learning_rate_decay=0.5, start_decay_steps=50000, decay_steps=10000, decay_method='', warmup_steps=4000, log_file='', log_file_level='0', verbose=False, train_eval_steps=200, train_metrics=[], valid_metrics=[], scoring_debug=True, dump_preds='', report_every=100, exp_host='', exp='', tensorboard=False, tensorboard_log_dir='runs/onmt', bucket_size=262144, bucket_size_init=-1, bucket_size_increment=0, prefetch_factor=400, brnn=False, data_task='lm', _all_transform={'filtertoolong'})

    totalsize = 0
    for m in ['model', 'generator']:
        for item in onmt_cp[m].keys():
            item2 = onmt_cp[m][item]
            totalsize += item2.nelement() * item2.element_size()
    print("Saving parameters: ", totalsize)

    torch.save(onmt_cp, opt.output)
