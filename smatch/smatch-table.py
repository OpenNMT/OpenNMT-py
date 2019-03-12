#!/usr/bin/env python

import amr
import sys
import subprocess
import smatch
import os
import random
import time

ERROR_LOG = sys.stderr

DEBUG_LOG = sys.stderr

verbose = False

# directory on isi machine
# change if needed
isi_dir_pre = "/nfs/web/isi.edu/cgi-bin/div3/mt/save-amr"


def get_names(file_dir, files):
    """
    Get the annotator name list based on a list of files
    Args:
    file_dir: AMR file folder
    files: a list of AMR names, e.g. nw_wsj_0001_1

    Returns:
   a list of user names who annotate all the files
    """
    # for each user, check if they have files available
    # return user name list
    total_list = []
    name_list = []
    get_sub = False
    for path, subdir, dir_files in os.walk(file_dir):
        if not get_sub:
            total_list = subdir[:]
            get_sub = True
        else:
            break
    for user in total_list:
        has_file = True
        for f in files:
            file_path = file_dir + user + "/" + f + ".txt"
            if not os.path.exists(file_path):
                has_file = False
                break
        if has_file:
            name_list.append(user)
    if len(name_list) == 0:
        print >> ERROR_LOG, "********Error: Cannot find any user who completes the files*************"
    return name_list


def compute_files(user1, user2, file_list, dir_pre, start_num):

    """
    Compute the smatch scores for a file list between two users
    Args:
    user1: user 1 name
    user2: user 2 name
    file_list: file list
    dir_pre: the file location prefix
    start_num: the number of restarts in smatch
    Returns:
    smatch f score.

    """
    match_total = 0
    test_total = 0
    gold_total = 0
    for fi in file_list:
        file1 = dir_pre + user1 + "/" + fi + ".txt"
        file2 = dir_pre + user2 + "/" + fi + ".txt"
        if not os.path.exists(file1):
            print >> ERROR_LOG, "*********Error: ", file1, "does not exist*********"
            return -1.00
        if not os.path.exists(file2):
            print >> ERROR_LOG, "*********Error: ", file2, "does not exist*********"
            return -1.00
        try:
            file1_h = open(file1, "r")
            file2_h = open(file2, "r")
        except IOError:
            print >> ERROR_LOG, "Cannot open the files", file1, file2
            break
        cur_amr1 = smatch.get_amr_line(file1_h)
        cur_amr2 = smatch.get_amr_line(file2_h)
        if cur_amr1 == "":
            print >> ERROR_LOG, "AMR 1 is empty"
            continue
        if cur_amr2 == "":
            print >> ERROR_LOG, "AMR 2 is empty"
            continue
        amr1 = amr.AMR.parse_AMR_line(cur_amr1)
        amr2 = amr.AMR.parse_AMR_line(cur_amr2)
        test_label = "a"
        gold_label = "b"
        amr1.rename_node(test_label)
        amr2.rename_node(gold_label)
        (test_inst, test_rel1, test_rel2) = amr1.get_triples()
        (gold_inst, gold_rel1, gold_rel2) = amr2.get_triples()
        if verbose:
            print >> DEBUG_LOG, "Instance triples of file 1:", len(test_inst)
            print >> DEBUG_LOG, test_inst
            print >> DEBUG_LOG, "Attribute triples of file 1:", len(test_rel1)
            print >> DEBUG_LOG, test_rel1
            print >> DEBUG_LOG, "Relation triples of file 1:", len(test_rel2)
            print >> DEBUG_LOG, test_rel2
            print >> DEBUG_LOG, "Instance triples of file 2:", len(gold_inst)
            print >> DEBUG_LOG, gold_inst
            print >> DEBUG_LOG, "Attribute triples of file 2:", len(gold_rel1)
            print >> DEBUG_LOG, gold_rel1
            print >> DEBUG_LOG, "Relation triples of file 2:", len(gold_rel2)
            print >> DEBUG_LOG, gold_rel2
        (best_match, best_match_num) = smatch.get_best_match(test_inst, test_rel1, test_rel2,
                                                             gold_inst, gold_rel1, gold_rel2,
                                                             test_label, gold_label)
        if verbose:
            print >> DEBUG_LOG, "best match number", best_match_num
            print >> DEBUG_LOG, "Best Match:", smatch.print_alignment(best_match, test_inst, gold_inst)
        match_total += best_match_num
        test_total += (len(test_inst) + len(test_rel1) + len(test_rel2))
        gold_total += (len(gold_inst) + len(gold_rel1) + len(gold_rel2))
        smatch.match_triple_dict.clear()
    (precision, recall, f_score) = smatch.compute_f(match_total, test_total, gold_total)
    return "%.2f" % f_score


def get_max_width(table, index):
    return max([len(str(row[index])) for row in table])


def pprint_table(table):
    """
    Print a table in pretty format

    """
    col_paddings = []
    for i in range(len(table[0])):
        col_paddings.append(get_max_width(table,i))
    for row in table:
        print row[0].ljust(col_paddings[0] + 1),
        for i in range(1, len(row)):
            col = str(row[i]).rjust(col_paddings[i]+2)
            print col,
        print "\n"

def build_arg_parser():
    """
    Build an argument parser using argparse. Use it when python version is 2.7 or later.

    """
    parser = argparse.ArgumentParser(description="Smatch table calculator -- arguments")
    parser.add_argument("--fl", type=argparse.FileType('r'), help='AMR ID list file')
    parser.add_argument('-f', nargs='+', help='AMR IDs (at least one)')
    parser.add_argument("-p", nargs='*', help="User list (can be none)")
    parser.add_argument("--fd", default=isi_dir_pre, help="AMR File directory. Default=location on isi machine")
    parser.add_argument('-r', type=int, default=4, help='Restart number (Default:4)')
    parser.add_argument('-v', action='store_true', help='Verbose output (Default:False)')
    return parser

def build_arg_parser2():
    """
    Build an argument parser using optparse. Use it when python version is 2.5 or 2.6.

    """
    usage_str = "Smatch table calculator -- arguments"
    parser = optparse.OptionParser(usage=usage_str)
    parser.add_option("--fl", dest="fl", type="string", help='AMR ID list file')
    parser.add_option("-f", dest="f", type="string", action="callback", callback=cb, help="AMR IDs (at least one)")
    parser.add_option("-p", dest="p", type="string", action="callback", callback=cb, help="User list")
    parser.add_option("--fd", dest="fd", type="string", help="file directory")
    parser.add_option("-r", "--restart", dest="r", type="int", help='Restart number (Default: 4)')
    parser.add_option("-v", "--verbose", action='store_true', dest="v", help='Verbose output (Default:False)')
    parser.set_defaults(r=4, v=False, ms=False, fd=isi_dir_pre)
    return parser


def cb(option, value, parser):
    """
    Callback function to handle variable number of arguments in optparse

    """
    arguments = [value]
    for arg in parser.rargs:
        if arg[0] != "-":
            arguments.append(arg)
        else:
            del parser.rargs[:len(arguments)]
            break
    if getattr(parser.values, option.dest):
        arguments.extend(getattr(parser.values, option.dest))
    setattr(parser.values, option.dest, arguments)


def check_args(args):
    """
    Parse arguments and check if the arguments are valid

    """
    if not os.path.exists(args.fd):
        print >> ERROR_LOG, "Not a valid path", args.fd
        return [], [], False
    if args.fl is not None:
        # we already ensure the file can be opened and opened the file
        file_line = args.fl.readline()
        amr_ids = file_line.strip().split()
    elif args.f is None:
        print >> ERROR_LOG, "No AMR ID was given"
        return [], [], False
    else:
        amr_ids = args.f
    names = []
    check_name = True
    if args.p is None:
        names = get_names(args.fd, amr_ids)
        # no need to check names
        check_name = False
        if len(names) == 0:
            print >> ERROR_LOG, "Cannot find any user who tagged these AMR"
            return [], [], False
        else:
            names = args.p
    if len(names) == 0:
        print >> ERROR_LOG, "No user was given"
        return [], [], False
    if len(names) == 1:
        print >> ERROR_LOG, "Only one user is given. Smatch calculation requires at least two users."
        return [], [], False
    if "consensus" in names:
        con_index = names.index("consensus")
        names.pop(con_index)
        names.append("consensus")
    # check if all the AMR_id and user combinations are valid
    if check_name:
        pop_name = []
        for i, name in enumerate(names):
            for amr in amr_ids:
                amr_path = args.fd + name + "/" + amr + ".txt"
                if not os.path.exists(amr_path):
                    print >> ERROR_LOG, "User", name, "fails to tag AMR", amr
                    pop_name.append(i)
                    break
        if len(pop_name) != 0:
            pop_num = 0
            for p in pop_name:
                print >> ERROR_LOG, "Deleting user", names[p - pop_num], "from the name list"
                names.pop(p - pop_num)
                pop_num += 1
        if len(names) < 2:
            print >> ERROR_LOG, "Not enough users to evaluate. Smatch requires >2 users who tag all the AMRs"
            return "", "", False
    return amr_ids, names, True


def main(arguments):
    global verbose
    (ids, names, result) = check_args(arguments)
    if arguments.v:
        verbose = True
    if not result:
        return 0
    acc_time = 0
    len_name = len(names)
    table = []
    for i in range(0, len_name + 1):
        table.append([])
    table[0].append("")
    for i in range(0, len_name):
        table[0].append(names[i])
    for i in range(0, len_name):
        table[i+1].append(names[i])
        for j in range(0, len_name):
            if i != j:
                start = time.clock()
                table[i+1].append(compute_files(names[i], names[j], ids, args.fd, args.r))
                end = time.clock()
                if table[i+1][-1] != -1.0:
                    acc_time += end-start
            else:
                table[i+1].append("")
    # check table
    for i in range(0, len_name + 1):
        for j in range(0, len_name + 1):
            if i != j:
                if table[i][j] != table[j][i]:
                    if table[i][j] > table[j][i]:
                        table[j][i] = table[i][j]
                    else:
                        table[i][j] = table[j][i]
    pprint_table(table)
    return acc_time


if __name__ == "__main__":
    whole_start = time.clock()
    parser = None
    args = None
    if sys.version_info[:2] != (2, 7):
        # requires python version >= 2.5
        if sys.version_info[0] != 2 or sys.version_info[1] < 5:
            print >> ERROR_LOG, "This program requires python 2.5 or later to run. "
            exit(1)
        import optparse
        parser = build_arg_parser2()
        (args, opts) = parser.parse_args()
        file_handle = None
        if args.fl is not None:
            try:
                file_handle = open(args.fl, "r")
                args.fl = file_handle
            except IOError:
                print >> ERROR_LOG, "The ID list file", args.fl, "does not exist"
                args.fl = None
    # python version 2.7
    else:
        import argparse
        parser = build_arg_parser()
        args = parser.parse_args()
    # Regularize fd, add "/" at the end if needed
    if args.fd[-1] != "/":
        args.fd += "/"
    # acc_time is the smatch calculation time
    acc_time = main(args)
    whole_end = time.clock()
    # time of the whole running process
    whole_time = whole_end - whole_start
    # print if needed
    # print >> ERROR_LOG, "Accumulated computation time", acc_time
    # print >> ERROR_LOG, "Total time", whole_time
    # print >> ERROR_LOG, "Percentage", float(acc_time)/float(whole_time)

