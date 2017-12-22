#!/usr/bin/env python
#encoding: utf-8

__author__ = 'Keisuke Sakaguchi'
__version__ = "0.1"
__descripstion__ = "calculate edit distance for given two sequences (e.g. lists or strings) "
__usage__ = "python edit_dist.py seqence1 sequence2"#

import os
import sys
import pattern.en
from nltk.corpus import wordnet as wn

#Constant Values
ins_cost = 1 # insertion cost
del_cost = 1 # deletion cost
sub_cost = 1 # substitution cost
partial_sub_cost = 0.7
close_sub_cost = 0.3
edThreshold = 2 # edit distance threshold 

def get_synonyms(word, pos):
    if pos[0] == 'N':
        pos = 'n'
    elif pos[0] == 'J':
        pos = 'a'
    elif pos[0] == 'V':
        pos = 'v'
    else:
        return []

    word_synsets = wn.synsets(word, pos)
    #print word_synsets
    synonymList = []
    for synset in word_synsets:
        for synonym in synset.lemmas():
            cand = synonym.name()
            if (cand.isalpha()):
                synonymList.append(cand)
    synonymSet = set(synonymList)
    return list(synonymSet)

def are_synonyms(t1, t2, pos):
    t1_synonyms = get_synonyms(t1, pos)
    t2_synonyms = get_synonyms(t2, pos)
    if len(set(t1_synonyms).intersection(set(t2_synonyms))) > 0:
        return True
    else:
        return False

def getEditDist(s1, s2):
    if isinstance(s1, str) or isinstance(s1, unicode):
        s1 = list(s1)
    if isinstance(s2, str) or isinstance(s2, unicode):
        s2 = list(s2)

    s1 = ['#'] + s1
    s2 = ['#'] + s2
    
    # distance matrix
    dm = [[0 for n in range(len(s1))] for m in range(len(s2))]
    for m in range(len(s1)):
        dm[0][m] = m
    for n in range(len(s2)):
        dm[n][0] = n
    for n in range(1, len(s1)):
        for m in range(1, len(s2)):
            # compute cost for insertion, deletion, and replace(substitution)
            c1 = dm[m][n-1] + ins_cost
            c2 = dm[m-1][n] + del_cost
            if s1[n] == s2[m]:
                c3 = dm[m-1][n-1]
            else:
                c3 = dm[m-1][n-1]+sub_cost

            if c3 <= c2 and c3 <= c1:
                dm[m][n] = c3
            elif c2 <= c3 and c2 <= c1:
                dm[m][n] = c2
            else:
                dm[m][n] = c1
 
    return dm[-1][-1]


def getEditDistWithPOS(seq1, seq2, pos1, pos2):
    if isinstance(seq1, str):
        seq1 = list(seq1)
    if isinstance(seq2, str):
        seq2 = list(seq2)

    seq1 = ['#'] + seq1
    seq2 = ['#'] + seq2
    pos1 = ['#'] + pos1
    pos2 = ['#'] + pos2

    dist_matrix = [[0 for j in range(len(seq1))] for i in range(len(seq2))]
    for i in range(len(seq1)):
        dist_matrix[0][i] = i
    for j in range(len(seq2)):
        dist_matrix[j][0] = j

    move_matrix = [[0 for j in range(len(seq1))] for i in range(len(seq2))]
    move_matrix[0][0] = '#'
    for i in range(1,len(seq1)):
        move_matrix[0][i] = "I"
    for j in range(1,len(seq2)):
        move_matrix[j][0] = "D"
   
    #calculate edit distance using DP
    for j in range(1, len(seq1)):
        for i in range(1, len(seq2)):

            # compute cost for insertion, deletion, and replace(substitution)
            w1 = seq1[j].lower()
            w2 = seq2[i].lower()
            cost1 = dist_matrix[i][j-1] + ins_cost
            cost2 = dist_matrix[i-1][j] + del_cost
            which_replace = ""
            if w1 == w2 and pos1[j] == pos2[i]:
                cost3 = dist_matrix[i-1][j-1]
                which_replace = "R"
            elif w1 == w2:
                cost3 = dist_matrix[i-1][j-1]
                which_replace = "R"
            elif pattern.en.lemma(w1) == pattern.en.lemma(w2):
                which_replace = "Ri" # Replace with inflection
                cost3 = dist_matrix[i-1][j-1] + close_sub_cost
            elif ((len(w1) >3 and len(w2) > 3)
                    and (getEditDist(w1, w2) < edThreshold)):
                cost3 = dist_matrix[i-1][j-1] + close_sub_cost
                which_replace = "Rc" # Replace with close spelling
            elif (pos1[j] == pos2[i] and are_synonyms(w1, w2, pos1[j])):
                cost3 = dist_matrix[i-1][j-1] + close_sub_cost
                which_replace = "Rs" # Replace with semantics
            elif pos1[j] == pos2[i]:
                cost3 = dist_matrix[i-1][j-1] + partial_sub_cost
                which_replace = "Rp" # Replace with POS
            else:
                cost3 = dist_matrix[i-1][j-1] + ins_cost + del_cost + 0.1
                which_replace = "Rd"

            # decide the move
            if cost3 <= cost2 and cost3 <= cost1:
                dist_matrix[i][j] = cost3
                #move_matrix[i][j] = 'R'
                move_matrix[i][j] = which_replace
            elif cost2 <= cost3 and cost2 <= cost1:
                dist_matrix[i][j] = cost2
                move_matrix[i][j] = "D"
            else:
                dist_matrix[i][j] = cost1
                move_matrix[i][j] = "I"
    
    # get operations
    j = len(seq1)-1
    i = len(seq2)-1
    prev_move = move_matrix[i][j]
    moves = [prev_move]
    while not (prev_move == "#"):
        if prev_move == "I": 
            prev_move = move_matrix[i][j-1]
            j -= 1
            moves.append("I")
        elif prev_move == "D":
            prev_move = move_matrix[i-1][j]
            i -= 1
            moves.append("D")
        elif prev_move[0] == "R":
            if dist_matrix[i][j] == dist_matrix[i-1][j-1]:
                moves.append("_")
            else:
                moves.append(prev_move)

            prev_move = move_matrix[i-1][j-1]
            j -= 1
            i -= 1
        else:
            raise

    return dist_matrix[-1][-1], moves[::-1]


if __name__ == '__main__':
    # unit test
    #print getEditDist('dog', 'bag')    # replace*2
    #print getEditDist('dog', 'cat')    # replace*3
    #print getEditDist('dog', 'doing')  # replace*1 + insert*1
    #print getEditDist('dog', 'do')     # delete*1
    #print getEditDist([1,3], [2,3])    # replace*1
    #print getEditDist([1,2,3,5,6], [1,2,4,5])    # replace*1 + insert*1
    #print are_synonyms("dog", "frump", "NN")
    #print are_synonyms("dog", "cat", "NN")
    #print are_synonyms("dog", "cat", "VB")
    #print are_synonyms("drink", "toast", "V")

    print getEditDist(sys.argv[1], sys.argv[2])
