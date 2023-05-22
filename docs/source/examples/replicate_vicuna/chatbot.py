import ctranslate2
import gradio as gr
import os
import sentencepiece as spm
import time

CACHE = {}
model_dir = "finetuned_llama7B/llama7B-vicuna-onmt_step_4000.concat_CT2"
tokenizer_dir = "llama"
out_dir = "outputs/ct2"


def load_models(model_dir, tokenizer_dir):
    if CACHE.get("generator", None) is None:
        CACHE["generator"] = ctranslate2.Generator(model_dir, device="cuda")
        CACHE["tokenizer"] = spm.SentencePieceProcessor(
            os.path.join(tokenizer_dir, "tokenizer.model")
        )


def make_prompt(chat_history):
    task_description = "Below is an instruction that describes a task. Write a response that appropriately completes the request.｟newline｠｟newline｠"  # noqa:E501

    def parse_instruction(text):
        return f"### Instruction:｟newline｠ {text} ｟newline｠｟newline｠"

    def parse_response(text):
        return f"### Response:｟newline｠{text}"

    out = [task_description]
    for _user_message, _bot_message in chat_history:
        out.append(parse_instruction(_user_message))
        if _bot_message is not None:
            out.append(parse_response(_bot_message))
        else:
            out.append("### Response:｟newline｠")
    prompt = "".join(out)
    return prompt


def generate_words(generator, sp, prompt, add_bos=True):
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
    out = []
    for word in generate_words(CACHE["generator"], CACHE["tokenizer"], prompt):
        out.append(word)
    return "".join(out)


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    load_models(model_dir, tokenizer_dir)

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        with open(os.path.join(out_dir, "history"), "w") as f:
            f.write(str(history))
        prompt = make_prompt(history)
        with open(os.path.join(out_dir, "prompt"), "w") as f:
            f.write(str(prompt))
        bot_message = make_bot_message(prompt)
        with open(os.path.join(out_dir, "bot_message"), "w") as f:
            f.write(str(bot_message))

        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.03)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch(server_port=1851, server_name="0.0.0.0")

# What are the 3 best french cities ?
