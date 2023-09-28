import argparse
import ctranslate2
import gradio as gr
import numpy as np
import os
import sentencepiece as spm
import time

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.utils.misc import use_gpu, set_random_seed

from onmt.inference_engine import InferenceEngine

CACHE = {}
tokenizer_dir = "llama"
max_context_length = 4096


parser = argparse.ArgumentParser()
parser.add_argument("-inference_config_file", help="Inference config file",
                    required=True,
                    type=str)
parser.add_argument("-inference_mode", help="Inference mode",
                    choices=['-py', 'ct2'],
                    default='-py',
                    type=str)
parser.add_argument("-server_port", help="Gradio app server port",
                    default='-py',
                    type=int)

args = parser.parse_args()
inference_config_file = args.inference_config_file
inference_mode = args.inference_mode
server_port = args.server_port


def make_prompt(chat_history):
    task_description = "Below is an instruction that describes a task. Write a response that appropriately completes the request.｟newline｠｟newline｠"  # noqa:E501
    sp = CACHE["tokenizer"]
    nb_user_tokens = []
    nb_bot_tokens = [0]

    parsed_instructions = []
    parsed_responses = []

    def parse_instruction(text):
        parsed_text = f"### Instruction:｟newline｠ {text} ｟newline｠｟newline｠"
        parsed_text_sp = parsed_text.replace("｟newline｠", "\n")
        tokens = sp.encode(parsed_text_sp, out_type=str)
        nb_user_tokens.append(len(tokens))
        return parsed_text

    def parse_response(text):
        parsed_text = f"### Response:｟newline｠{text}"
        tokens = sp.encode(parsed_text, out_type=str)
        nb_bot_tokens.append(len(tokens))
        return parsed_text

    out = [task_description]
    for _user_message, _bot_message in chat_history:
        parsed_instructions.append(parse_instruction(_user_message))
        if _bot_message is not None:
            parsed_responses.append(parse_response(_bot_message))
        else:
            parsed_responses.append("### Response:｟newline｠")

        keep_indices = prune_history(
            nb_user_tokens, nb_bot_tokens, max_context_length - len(task_description)
        )
        for i in keep_indices:
            out.append(parsed_instructions[i])
            out.append(parsed_responses[i])
    prompt = "".join(out)
    return prompt


def prune_history(x, y, L):
    reversed_indices = list(range(len(x)))[::-1]
    keep_indices = []
    _x, _y = x[::-1], y[::-1]
    z = [sum(i) for i in zip(_x, _y)]
    for i, n in enumerate(np.cumsum(z)):
        if n < L:
            keep_indices.append(reversed_indices[i])
    keep_indices.reverse()
    return keep_indices


######################
# Inference with CT2 #
######################
model_dir = "finetuned_llama7B/llama7B-vicuna-onmt_step_4000.concat_CT2"


def load_models(model_dir, tokenizer_dir):
    if CACHE.get("generator", None) is None:
        CACHE["generator"] = ctranslate2.Generator(model_dir, device="cuda")
        CACHE["tokenizer"] = spm.SentencePieceProcessor(
            os.path.join(tokenizer_dir, "tokenizer.model")
        )


def generate_words(prompt, add_bos=True):
    generator, sp = CACHE["generator"], CACHE["tokenizer"]
    prompt_tokens = sp.encode(prompt, out_type=str)

    if add_bos:
        prompt_tokens.insert(0, "<s>")

    step_results = generator.generate_tokens(
        prompt_tokens, sampling_temperature=0.1, sampling_topk=40, max_length=512
    )

    output_ids = []
    for step_result in step_results:
        is_new_word = step_result.token.startswith("▁")

        if is_new_word and output_ids:
            yield " " + sp.decode(output_ids)
            output_ids = []

        output_ids.append(step_result.token_id)

    if output_ids:
        yield " " + sp.decode(output_ids)


def make_bot_message_ct2(prompt):
    prompt = prompt.replace("｟newline｠", "\n")
    words = []
    for _out in generate_words(prompt):
        words.append(_out)
    bot_message = "".join(words[:-1])
    return bot_message


######################
# Inference with -py #
######################
ckpt_path = "finetuned_llama7B/llama7B-vicuna-onmt_step_4000.concat_added_key.pt"


def _get_parser():
    parser = ArgumentParser(description="translate.py")
    opts.config_opts(parser)
    opts.translate_opts(parser, dynamic=True)
    return parser


def load_translator(opt):
    if CACHE.get("translator", None) is None:
        ArgumentParser.validate_translate_opts(opt)
        ArgumentParser._get_all_transform_translate(opt)
        ArgumentParser._validate_transforms_opts(opt)
        ArgumentParser.validate_translate_opts_dynamic(opt)
        set_random_seed(opt.seed, use_gpu(opt))
        # Build the translator (along with the model)
        CACHE["inference_engine"] = InferenceEngine(opt)
        CACHE["tokenizer"] = spm.SentencePieceProcessor(
            os.path.join(tokenizer_dir, "tokenizer.model")
        )


def make_bot_message_py(prompt):
    # we receive a text box content
    # might be good to split also based on full period (later)
    src = [prompt.replace("\n", "｟newline｠")]
    scores, predictions = CACHE["inference_engine"].infer_list(src)
    # print("\n".join([predictions[i][0] for i in range(len(predictions))]))
    bot_message = "\n".join(sent[0] for sent in predictions)
    bot_message = bot_message.replace("｟newline｠", "\n")
    return bot_message


######
# UI #
######

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    submit = gr.Button("Submit")
    clear = gr.Button("Clear")

    if inference_mode == "ct2":
        load_models(model_dir, tokenizer_dir)
    elif inference_mode == "-py":
        parser = _get_parser()
        base_args = (
            ["-model", ckpt_path]
            + ["-src", "dummy"]
            + ["-config", inference_config_file]
        )
        opt = parser.parse_args(base_args)

        load_translator(opt)

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        prompt = make_prompt(history)

        if inference_mode == "ct2":
            bot_message = make_bot_message_ct2(prompt)
        elif inference_mode == "-py":
            bot_message = make_bot_message_py(prompt)
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0)
            yield history

    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch(server_port=server_port, server_name="0.0.0.0")

# What are the 3 best french cities ?
# Which one is better if I like outdoor activities ?
# Which one is better if I like cultural outings?
# What are the best neighborhoods in these 5 cities?
