# Supervised Finetuning of llama 7B to replicate Vicuna
This tutorial shows how to finetune a LLaMA 7B foundation model on instruction data including multi-round conversations.

Different features will be enabled:
- Application of the LoRa method to the attention layers.
- 8bit compression of the position-wise feed-forward layers.
- Architectural improvements used during the training of the llama models (RMS normalisation, Rotary Embeddings, SwiGLU activation).

The maximal context length will be set to 512.

Here is a short description of the content of your current directory:

- The OpenNMT-py repository.
- The `replicate_vicuna.yaml` file with the finetuning options
- A subdirectory named "llama" with the llama chekpoints.
- The llama7B checkpoint converted to `OpenNMT-py` format (`llama7B-vicuna-onmt`) and the vocabulary (`vocab.txt`). They will be genenerated with `OpenNMT-py` tools.
- A subdirectory named "dataAI" with the datasets for the finetuning.
- A subdirectory named "finetuned_llama7B" that will contain the finetuning samples, the tensorboard logs and the checkpoints.
- The `translate_opts_py.yaml` file with the translation options for the inference with `translate.py`.
- The `translate_opts_ct2.yaml` file with the translation options for the inference with `cranslate2`.
- The `input_examples.txt` file with a few input examples.
- A subdirectory named "outputs" that will contain the inferred outputs of the finetuned model.
- The `simple_inference.py` file to compute vicuna's predictions from the `input_examples.txt` file, for the 2 different modes.
- The `chatbot.py` script (for the ctranslate2 inference with a gradio application).

## Dependencies
Apex is highly recommended to have fast performance.

```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
cd ..
```

You must also have gradio and ctranslate2 installed in your environment:

```shell
pip install gradio
pip install ctranslate2==3.14.0
```

## Data

### Checkpoints
The procedure to retrieve the llama checkpoints as well the llama legacy sentencepiece tokenizer is described on the official llama repository:  https://github.com/facebookresearch/llama/

Let us save them in a local folder that we will name "llama".

We need to convert the llama 7B checkpoint to the `onmt` format, using the `convert_llama.py` tool:

```shell
python3 OpenNMT-py/tools/convert_llama.py \
    --model_dir llama/7B/ \
    --tokenizer_model llama/tokenizer.model \
    --output llama7B-vicuna-onmt
```

The converted checkpoint is named `llama7B-vicuna-onmt`.

### Vocabulary 
As the subword model is a sentencepiece model, the vocabulary can be retrieved from the tokenizer. The `convert_llama.py` script saved a copy of the vocabulary with slight modifications but you can also extract the vocabulary from the newly created checkpoint as follow:

```shell
python3 OpenNMT-py/tools/extract_vocabulary.py -model llama7B-vicuna-onmt -out_file vocab.txt -side src
```



### Datasets 

The original [*alpaca*](https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json) and *vicuna* datasets are JSON files. This 

Here is the first element of the original alpaca_data.json dataset :
```json
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.\n\n3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night."
    },
```

The *vicuna* dataset

**The datasets that will be used in this tutorial are slightly modified versions of the original datasets.**
They have been flattened into plain text files. Moreover all occurences of the “\n” symbol, which acts as example break in the OpenNMT world, have been replaced with '｟newline｠'.

The onmt datasets can be retrieved at the links below:

- [alpaca](https://opennmt-models.s3.amazonaws.com/llama/alpaca_clean.txt) (51751 examples)
  
- [vicuna](https://opennmt-models.s3.amazonaws.com/llama/sharegpt.txt) (28800 examples)

Let us save them in a  local folder that we will name `dataAI`.

Each example is a prompt that contains:
- a short description of the task
- an instrunction following the pattern `### Instruction`
- a proposal of answer following the pattern `### Response` 

Here is the first example in the onmt alpaca dataset:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.｟newline｠｟newline｠### Instruction:｟newline｠Give three tips for staying healthy.｟newline｠｟newline｠### Response:｟newline｠1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.｟newline｠｟newline｠2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.｟newline｠｟newline｠3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night.
```

## Finetuning

We provide an example of a finetuning configuration (`replicate_vicuna.yaml`). To enable the application of the LoRa method to the attention layers, the options of the checkpoint need to be overriden.


The finetuning can be launched with this command:

```shell 
nohup python3 OpenNMT-py/onmt/bin/train.py -config replicate_-vicuna.yaml > finetenune-llama7B-vicuna-onmt.log &
```

We can start by generating some samples (by turning `dump_samples` to True and `n_samples` to a strictly positive value).

It is worth noting that the he sentencepiece vocabulary does not map the custom substring ｟newline｠with a specific token. However it maps the new line symbol '\n' with the token '<0x0A>'. To handle properly our datasets without changing the vocabulary and training new embddings from scratch, the Tokenize transform replaces on-the-fly the token '｟newline｠' token with  '<0x0A>'.

For instance the first training example is transformed in:
```
▁Below ▁is ▁an ▁instruction ▁that ▁describes ▁a ▁task . ▁Write ▁a ▁response ▁that ▁appropri ately ▁comple tes ▁the ▁request . <0x0A> <0x0A> ## # ▁Inst ruction : <0x0A> G ive ▁three ▁tips ▁for ▁stay ing ▁health y . <0x0A> <0x0A> ## # ▁Response : <0x0A> 1 . ▁E at ▁a ▁bal anced ▁and ▁nut rit ious ▁di et : ▁Make ▁sure ▁your ▁me als ▁are ▁inclus ive ▁of ▁a ▁variety ▁of ▁f ruits ▁and ▁veget ables , ▁lean ▁protein , ▁whole ▁gra ins , ▁and ▁health y ▁f ats . ▁This ▁helps ▁to ▁provide ▁your ▁body ▁with ▁the ▁essential ▁nut ri ents ▁to ▁function ▁at ▁its ▁best ▁and ▁can ▁help ▁prevent ▁chron ic ▁dise ases . <0x0A> <0x0A> 2 . ▁Eng age ▁in ▁regular ▁physical ▁activity : ▁Ex erc ise ▁is ▁cru cial ▁for ▁maintain ing ▁strong ▁b ones , ▁mus cles , ▁and ▁card i ov asc ular ▁health . ▁A im ▁for ▁at ▁least ▁ 1 5 0 ▁minutes ▁of ▁moder ate ▁aer ob ic ▁exercise ▁or ▁ 7 5 ▁minutes ▁of ▁vig orous ▁exercise ▁each ▁week . <0x0A> <0x0A> 3 . ▁Get ▁enough ▁sleep : ▁Getting ▁enough ▁quality ▁sleep ▁is ▁cru cial ▁for ▁physical ▁and ▁mental ▁well - be ing . ▁It ▁helps ▁to ▁reg ulate ▁m ood , ▁improve ▁cogn itive ▁function , ▁and ▁supports ▁health y ▁growth ▁and ▁imm une ▁function . ▁A im ▁for ▁ 7 - 9 ▁hours ▁of ▁sleep ▁each ▁night .
```


## Inference

### Concatenation of the checkpoints

As we applied the LoRa method, we first need to merge the finetuned `llama7B-vicuna-onmt.pt` checkpoint in the original `llama7B-onmt.pt` model, using the `lora_weights.py tool`. :

```shell
python3 OpenNMT-py/tools/lora_weights.py\
    --action merge \
    --base_model llama7B-vicuna-onmt \
    --lora_weights finetuned_llama7B/llama7B-vicuna-onmt_step_4000.pt \
    --output finetuned_llama7B/llama7B-vicuna-onmt_step_4000.concat.pt
```

### Conversion to ctranslate format

To convert the concatenated checkpoint to ctranslate2 format, run the following command:

```shell
python3 OpenNMT-py/onmt/bin/release_model.py \
    --model finetuned_llama7B/llama7B-vicuna-onmt_step_4000.concat.pt \
    --output finetuned_llama7B/llama7B-vicuna-onmt_step_4000.concat_CT2 \
    --format ctranslate2 \
    --quantization int8_float16
```

### Multi-round conversations with vicuna

We provide a gradio chatbot application that can be run with two different inference modes ("py" or ctranslate2).

Run one of the following commands:
```shell
python3 chatbot.py \
-inference_config_file translate_opts_py.yaml \
-inference_mode py \
-max_context_length 4096 \
-server_port 5000
```
Or:

```shell
python3 chatbot.py \
-inference_config_file translate_opts_ct2.yaml \
-inference_mode ct2 \
-max_context_length 4096 \
-server_port 5000
```
Where `translate_opts_ct2.yaml`  and `translate_opts_py.yaml` are the provided config with the translation options.
You can test other decoding methods and paramaters.

###  Simple inference

To obtain the model's inference you can run this command:


```shell
python3 simple_inference.py \
    -input_file input_examples.txt \
    -inference_config_file translate_opts_py.yaml \
    -inference_mode py \
    -output_dir outputs
```
Or:

```shell
python3 simple_inference.py \
    -input_file input_examples.txt \
    -inference_config_file translate_opts_ct2.yaml \
    -inference_mode ct2 \
    -output_dir outputs
```
