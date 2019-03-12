from collections import defaultdict
import collections

src = open("ldc2015e86/dev-dfs-linear_src.txt").read().splitlines()
gold = open("ldc2015e86/dev-dfs-linear_targ.txt").read().splitlines()
#reen = open("reentrs_and_deps.txt").read().splitlines()
s_out = open("seq_model_output.txt").read().splitlines()
t_out = open("tree_model_output.txt").read().splitlines()
g_out = open("graph_model_output.txt").read().splitlines()

#s_sco = open("seq_model_scores.txt").read().splitlines()
#t_sco = open("tree_model_scores.txt").read().splitlines()
#g_sco = open("graph_model_scores.txt").read().splitlines()

ex = defaultdict(list)
for s, g, s_o, t_o, g_o in zip(src, gold, s_out, t_out, g_out):
    if "he" in g and "he" in g_o and ("they" in s_o or "they" in t_o):
        ex[len(g)].append((s, g, s_o, t_o, g_o))
for k in sorted(ex):
    print(k)
    for item in ex[k]:
        for x in item:
            print(x)
    input()
