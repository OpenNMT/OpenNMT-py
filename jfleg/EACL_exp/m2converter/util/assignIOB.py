#!/usr/bin/env python
#encoding: utf-8

__author__ = ""
__version__ = ""
__copyright__ = ""
__license__ = ""
__descripstion__ = ""
__usage__ = ""

import sys
import os

def getIOBorg(l):
    #print l
    prev_oper = ""
    edits = []

    for i, l_i in enumerate(l):
        oper = l_i
        if prev_oper == oper:
            pass
            #edits[-1][1] = i-1 # end_index
        else:
            if not i==0:
                edits[-1][1] = i
            edits.append([0,0,oper])
            edits[-1][0] = i
            prev_oper = oper
    edits[-1][1] = i+1

    # additional touch for considering D operations (which don't change source-side index)
    for j, e in enumerate(edits):
        if e[2] == "D":
            # get difference
            diff_idx = e[1] - e[0]
            for h in range(j+1, len(edits)):
                edits[h][0] = edits[h][0] - diff_idx
                edits[h][1] = edits[h][1] - diff_idx
        else:
            pass

    return edits


def getIOB(l):
    #print l
    prev_oper = ""
    edits = []

    for i, l_i in enumerate(l):
        oper = l_i
        if prev_oper == oper:
            pass
            #edits[-1][1] = i-1 # end_index
        else:
            if not i==0:
                edits[-1][1] = i
                edits[-1][4] = i
            edits.append([0,0,oper,0,0]) # src_begin_idx, src_end_idx, operation, trg_begin_idx, trg_end_idx
            edits[-1][0] = i
            edits[-1][3] = i
            prev_oper = oper
    edits[-1][1] = i+1
    edits[-1][4] = i+1

    # additional touch for considering D operations (which don't change source-side index)
    for j, e in enumerate(edits):
        if e[2] == "D":
            # get difference
            diff_idx = e[1] - e[0]
            for h in range(j+1, len(edits)):
                edits[h][0] = edits[h][0] - diff_idx
                edits[h][1] = edits[h][1] - diff_idx
        if e[2] == "I":
            # get difference
            diff_idx = e[4] - e[3]
            for h in range(j+1, len(edits)):
                edits[h][3] = edits[h][3] - diff_idx
                edits[h][4] = edits[h][4] - diff_idx
        else:
            pass

    return edits

if __name__ == "__main__":
    ## unit tests
    #l = [4,3,3,2,2,1,1,1]
    #iob = getIOB(l)
    #print iob
    #assert iob == [[0,1,4],[1,3,3],[3,5,2],[5,8,1]]

    l = ['_', '_', 'D', 'D', 'D', 'R', '_', 'I', 'I', 'R', 'R', 'R', '_']
    iob = getIOB(l)
    print iob
    #assert iob == [[0,2,'_'],[2,5,'D'],[5,6,'R'],[6,7,'_'],[7,9,'I'],[9,12,'R'],[12,13,'_']]
    assert iob == [[0,2,'_',0,2],[2,5,'D',2,5],[2,3,'R',5,6],[3,4,'_',6,7],[4,6,'I',7,9],[6,9,'R',7,10],[9,10,'_',10,11]]
    
