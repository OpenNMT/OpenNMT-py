import sys
import spacy
nlp = spacy.load('en_coref_lg')
count = 0

mistake_type = "NUM"

fsrc = open(mistake_type + "_contr.src.txt", "w")
fref = open(mistake_type + "_contr.ref.txt", "w")
fcon = open(mistake_type + "_contr.contr.txt", "w")

males = ['he','his','him','himself']
females = ['she','hers','her','herself']
subj_pro = ['i', 'you', 'he', 'she', 'it', 'we', 'you', 'they']
obj_pro = ['me', 'you', 'him', 'her', 'it', 'us', 'you', 'them']
poss_adj = ['my', 'your', 'his', 'her', 'its', 'our', 'your', 'their']
poss_pro = ['mine', 'yours', 'his', 'hers', None, 'ours', 'yours', 'theirs']
ref_pro = ['myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves']
pros = [subj_pro, obj_pro, poss_adj, poss_pro, ref_pro]

for src, line in zip(open(sys.argv[1]), open(sys.argv[2])):
    line = line.strip()
    doc = nlp(line)
    corefs = {}
    for i, tok in enumerate(doc):
        mentions = []
        try:
            for c in tok._.coref_clusters:
                mentions.append(c.main)
        except:
            pass
        mentions = [x for x in mentions if tok not in x]
        if mentions != []:
            corefs[i] = mentions

    if corefs != {}:
        original = [str(t) for t in doc] 
        examples = []
        for k in corefs:
            for v in corefs[k]:
                    if mistake_type == "ANTEC":
                        if original[k] != str(v):
                            examples.append(" ".join(original[:k] + [str(v)] + original[k + 1:]))
                            
                    elif mistake_type == "TYPE":
                        for p in pros:
                            if original[k] in p:
                                idx = p.index(original[k])
                                for p2 in pros:
                                    if p[idx] != p2[idx] and p2[idx] is not None:
                                        examples.append(" ".join(original[:k] + [str(p2[idx])] + original[k + 1:]))
                    elif mistake_type == "NUM":
                        for p in pros:
                            if original[k] in p:
                                for item in p:
                                    if item != original[k] and item is not None:
                                        if (item in females and original[k] in females) or (item in males and original[k] in males) or (item not in males and item not in females) or (original[k] not in males and original[k] not in females):
                                            examples.append(" ".join(original[:k] + [str(item)] + original[k + 1:]))
#                    elif mistake_type == "PERS":
#                        for p in pros:
#                            if original[k] in p:
#                                for item in p:
#                                    if item != original[k] and item is not None:
#                                        if item not in thirds or original[k] not in thirds:
#                                            examples.append(" ".join(original[:k] + [str(item)] + original[k + 1:]))                                        
                    elif mistake_type == "GEND":
                        for p in pros:
                            if original[k] in p:
                                for item in p:
                                    if item != original[k] and item is not None:
                                        if (item in females and original[k] in males) or  (item in males and original[k] in females):
                                            examples.append(" ".join(original[:k] + [str(item)] + original[k + 1:]))

                                    
        for example in examples:
            count += 1
            fsrc.write(src.strip() + "\n")
            fref.write(line + "\n")
            fcon.write(example + "\n")

fsrc.close()
fref.close()
fcon.close()
print(mistake_type, count) 
