import argparse
import gradio as gr
import numpy as np
import time

import onmt.opts as opts
from onmt.transforms.tokenize import SentencePieceTransform
from onmt.utils.parse import ArgumentParser
from onmt.utils.misc import use_gpu, set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument(
    "-inference_config_file", help="Inference config file", required=True, type=str
)
parser.add_argument(
    "-inference_mode",
    help="Inference mode",
    required=True,
    type=str,
    choices=["py", "ct2"],
)
parser.add_argument(
    "-max_context_length",
    help="Maximum size of the chat history.",
    type=int,
    default=4096,
)
parser.add_argument(
    "-server_port", help="Server port for the gradio app.", default=6006, type=int
)

args = parser.parse_args()
inference_config_file = args.inference_config_file
inference_mode = args.inference_mode
max_context_length = args.max_context_length
server_port = args.server_port

CACHE = {}


def make_prompt(chat_history):
    task_description = "Below is an instruction that describes a task. Write a response that appropriately completes the request.｟newline｠｟newline｠"  # noqa:E501
    nb_user_tokens = []
    nb_bot_tokens = [0]

    parsed_instructions = []
    parsed_responses = []

    def parse_instruction(text):
        parsed_text = f"### Instruction:｟newline｠ {text} ｟newline｠｟newline｠"
        tokens = CACHE["tokenizer"]._tokenize(parsed_text)
        nb_user_tokens.append(len(tokens))
        return parsed_text

    def parse_response(text):
        parsed_text = f"### Response:｟newline｠{text}"
        tokens = CACHE["tokenizer"]._tokenize(parsed_text)
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


def prune_history(user_messages_sizes, bot_messages_sizes, max_history_size):
    """Prune the history from the beginning not to exceed the maximum context length."""
    nb_rounds = len(user_messages_sizes)
    # Put messages sizes in antichronological order
    reversed_user_messages_sizes = user_messages_sizes[::-1]
    reversed_bot_messages_sizes = bot_messages_sizes[::-1]
    reversed_rounds_indices = list(range(nb_rounds))[::-1]
    # Caluculate antichronological history sizes
    reversed_round_sizes = [
        sum(i) for i in zip(reversed_user_messages_sizes, reversed_bot_messages_sizes)
    ]
    reversed_history_sizes = np.cumsum(reversed_round_sizes)
    keep_rounds_indices = []
    # Prune the history from the beginning
    for i, n in enumerate(np.cumsum(reversed_history_sizes)):
        if n < max_history_size:
            keep_rounds_indices.append(reversed_rounds_indices[i])
    # Put back indices in chronological order.
    keep_rounds_indices.reverse()
    return keep_rounds_indices


def _get_parser():
    parser = ArgumentParser(description="chatbot.py")
    opts.translate_opts(parser)
    opts.model_opts(parser)
    return parser


def load_models(opt, inference_mode):
    if CACHE.get("inference_engine", None) is None:
        ArgumentParser.validate_translate_opts(opt)
        ArgumentParser._get_all_transform_translate(opt)
        ArgumentParser._validate_transforms_opts(opt)
        ArgumentParser.validate_translate_opts_dynamic(opt)
        set_random_seed(opt.seed, use_gpu(opt))
        # Build the translator (along with the model)
        if inference_mode == "py":
            print("Inference with py ...")
            from onmt.inference_engine import InferenceEnginePY

            CACHE["inference_engine"] = InferenceEnginePY(opt)
        elif inference_mode == "ct2":
            print("Inference with ctranslate2 ...")
            from onmt.inference_engine import InferenceEngineCT2

            CACHE["inference_engine"] = InferenceEngineCT2(opt)
        # We need to build the Llama tokenizer to count tokens and prune the history.
        CACHE["tokenizer"] = SentencePieceTransform(opt)
        CACHE["tokenizer"].warm_up()


def make_bot_message(prompt, inference_mode):
    src = [prompt.replace("\n", "｟newline｠")]
    if inference_mode == "py":
        scores, predictions = CACHE["inference_engine"].infer_list(src)
        # The hypotheses are lists of one element but we still need to take the first one.
        bot_message = "\n".join(sent[0] for sent in predictions)
    elif inference_mode == "ct2":
        scores, predictions = CACHE["inference_engine"].infer_list(src)
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
    base_args = ["-config", inference_config_file]
    parser = _get_parser()
    opt = parser.parse_args(base_args)
    load_models(opt, inference_mode)

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        prompt = make_prompt(history)
        bot_message = make_bot_message(prompt, inference_mode)
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
