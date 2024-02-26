These are perplexity computed on wikitext2.

Numbers are not comparable to lm-evaluation-harness since they compute word / byte / bit perplexity like this:

hf-auto (pretrained=mistralai/Mistral-7B-Instruct-v0.2), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 8
| Tasks  |Version|Filter|n-shot|    Metric     |Value |   |Stderr|
|--------|------:|------|------|---------------|-----:|---|------|
|wikitext|      2|none  |None  |word_perplexity|9.8183|±  |N/A   |
|        |       |none  |None  |byte_perplexity|1.5329|±  |N/A   |
|        |       |none  |None  |bits_per_byte  |0.6163|±  |N/A   |


hf-auto (pretrained=meta-llama/Llama-2-7b-hf), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
| Tasks  |Version|Filter|n-shot|    Metric     |Value |   |Stderr|
|--------|------:|------|------|---------------|-----:|---|------|
|wikitext|      2|none  |None  |word_perplexity|8.7921|±  |N/A   |
|        |       |none  |None  |byte_perplexity|1.5016|±  |N/A   |
|        |       |none  |None  |bits_per_byte  |0.5865|±  |N/A   |


Numbers are not comparable to perplexity reported by llama.cpp because we use a smaller context window but also we detokenize the raw corpus (thing that they shoudl do but they don't)

| 7B Family        |                       | PPL   | Time (sec) |
| ---------------- | --------------------- | ----- | ---------- |
| Base             | llama2                | 5.78  | 152        |
|                  | mistral v0.1          | 5.70  | 162        |
|                  |          awq          | 5.81  | 165        |
|                  | Yi-6B-200K            | 7.76  | 133        |
|                  | xgen-7B               | 8.64  | 129        |
|                  | mpt-7B                | 8.43  | 147        |
|                  |                       |       |            |
| Instruct / Tuned | llama2-chat           | 7.37  | 148        |
|                  | mistral-instr-v0.2    | 6.98  | 160        |
|                  |           gemm-awq    | 7.07  | 164        |
|                  |           gemv-awq    | 7.07  | 237        |
|                  |                       |       |            |
|                  | Alma-7B-R             | 6.82  | 156        |
|                  | TowerInstruct-7B      | 6.45  | 157        |
|                  | codellama-7B          | 8.56  | 154        |
|                  |                       |       |            |
| 3B Family        | Phi-2                 | 9.74  | 52         |
|                  | Phi-2-psy             | 10.44 | 53         |
|                  |                       |       |            |
| 13B Family       | llama2 (4-bit)        | 5.31  | 296        |
|                  | llama2-chat (4-bit)   | 6.59  | 292        |
|                  |                       |       |            |
| 34B Family       | codellama-34B (4-bit) | 6.00  | 706        |


We note that llama2 and Mistral are in fact very close for their base model. However there is a shift between their chat model.

All others are quite below which is surprising for Yi given their results on the Open llm leaderboard.

I need to check why Mistral seems a little slower than llama2, it should be the opposite.
