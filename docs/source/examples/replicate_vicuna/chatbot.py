import ctranslate2
import gradio as gr
import numpy as np
import os
import sentencepiece as spm
import time

CACHE = {}
model_dir = "finetuned_llama7B/llama7B-vicuna-onmt_step_4000.concat_CT2"
tokenizer_dir = "llama"
max_context_length = 1000  # # 4096


def load_models(model_dir, tokenizer_dir):
    if CACHE.get("generator", None) is None:
        CACHE["generator"] = ctranslate2.Generator(model_dir, device="cuda")
        CACHE["tokenizer"] = spm.SentencePieceProcessor(
            os.path.join(tokenizer_dir, "tokenizer.model")
        )


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
    prompt = prompt.replace("｟newline｠", "\n")
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


def make_bot_message(prompt):
    words = []
    for _out in generate_words(prompt):
        words.append(_out)
    bot_message_length = _out
    return "".join(words[:-1]), bot_message_length


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    load_models(model_dir, tokenizer_dir)

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        prompt = make_prompt(history)
        bot_message = make_bot_message(prompt)

        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.001)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch(server_port=1851, server_name="0.0.0.0")

# What are the 3 best french cities ?
# Which one is better if I like outdoor activities ?
# Which one is better if I like cultural outings?
# What are the best neighborhoods in these 5 cities?
