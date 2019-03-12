# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
This script computes smatch score between two AMRs.
For detailed description of smatch, see http://www.isi.edu/natural-language/amr/smatch-13.pdf

"""
import cPickle as pickle
import amr
import os
import random
import sys
import time

# total number of iteration in smatch computation
iteration_num = 5

# verbose output switch.
# Default false (no verbose output)
verbose = False

# single score output switch.
# Default true (compute a single score for all AMRs in two files)
single_score = True 

# precision and recall output switch.
# Default false (do not output precision and recall, just output F score)
pr_flag = False

# Error log location
ERROR_LOG = sys.stderr

# Debug log location
DEBUG_LOG = sys.stderr

# dictionary to save pre-computed node mapping and its resulting triple match count
# key: tuples of node mapping
# value: the matching triple count
match_triple_dict = {}

def parse_relations(rels, v2c):
	var_list = []
	conc_list = []
	for r in rels:
		if str(r[1]) not in var_list and str(r[1]) != "TOP" and r[1] in v2c:
			var_list.append(str(r[1]))
			conc_list.append(str(v2c[r[1]]))
		if str(r[2]) not in var_list and r[2] in v2c:
			var_list.append(str(r[2]))
			conc_list.append(str(v2c[r[2]]))
	k = 0
	rel_dict = []*len(var_list)
	att_dict = []*len(var_list)
	for v in var_list:
		rel_dict.append({})
		att_dict.append({})
		for i in rels:
			if str(i[1]) == str(v) and i[2] in v2c:
				rel_dict[k][str(i[2])] = i[0]
				att_dict[k][i[0]] = str(v2c[i[2]])
		k += 1
	return amr.AMR(var_list, conc_list, rel_dict, att_dict)

def get_amr_line(input_f):
    """
    Read the file containing AMRs. AMRs are separated by a blank line.
    Each call of get_amr_line() returns the next available AMR (in one-line form).
    Note: this function does not verify if the AMR is valid

    """
    cur_amr = []
    has_content = False
    for line in input_f:
        line = line.strip()
        if line == "":
            if not has_content:
                # empty lines before current AMR
                continue
            else:
                # end of current AMR
                break
        if line.strip().startswith("#"):
            # ignore the comment line (starting with "#") in the AMR file
            continue
        else:
            has_content = True
            cur_amr.append(line.strip())
    return "".join(cur_amr)


def build_arg_parser():
    """
    Build an argument parser using argparse. Use it when python version is 2.7 or later.

    """
    parser = argparse.ArgumentParser(description="Smatch calculator -- arguments")
    parser.add_argument('-f', nargs=2, required=True,
                        help='Two files containing AMR pairs. AMRs in each file are separated by a single blank line')
    parser.add_argument('-r', type=int, default=4, help='Restart number (Default:4)')
    parser.add_argument('-v', action='store_true', help='Verbose output (Default:false)')
    parser.add_argument('--ms', action='store_true', default=False,
                        help='Output multiple scores (one AMR pair a score)' \
                             'instead of a single document-level smatch score (Default: false)')
    parser.add_argument('--pr', action='store_true', default=False,
                        help="Output precision and recall as well as the f-score. Default: false")
    return parser


def build_arg_parser2():
    """
    Build an argument parser using optparse. Use it when python version is 2.5 or 2.6.

    """
    usage_str = "Smatch calculator -- arguments"
    parser = optparse.OptionParser(usage=usage_str)
    parser.add_option("-f", "--files", nargs=2, dest="f", type="string",
                      help='Two files containing AMR pairs. AMRs in each file are ' \
                           'separated by a single blank line. This option is required.')
    parser.add_option("-r", "--restart", dest="r", type="int", help='Restart number (Default: 4)')
    parser.add_option("-v", "--verbose", action='store_true', dest="v", help='Verbose output (Default:False)')
    parser.add_option("--ms", "--multiple_score", action='store_true', dest="ms",
                      help='Output multiple scores (one AMR pair a score) instead of ' \
                           'a single document-level smatch score (Default: False)')
    parser.add_option('--pr', "--precision_recall", action='store_true', dest="pr",
                      help="Output precision and recall as well as the f-score. Default: false")
    parser.set_defaults(r=4, v=False, ms=False, pr=False)
    return parser


def get_best_match(instance1, attribute1, relation1,
                   instance2, attribute2, relation2,
                   prefix1, prefix2):
    """
    Get the highest triple match number between two sets of triples via hill-climbing.
    Arguments:
        instance1: instance triples of AMR 1 ("instance", node name, node value)
        attribute1: attribute triples of AMR 1 (attribute name, node name, attribute value)
        relation1: relation triples of AMR 1 (relation name, node 1 name, node 2 name)
        instance2: instance triples of AMR 2 ("instance", node name, node value)
        attribute2: attribute triples of AMR 2 (attribute name, node name, attribute value)
        relation2: relation triples of AMR 2 (relation name, node 1 name, node 2 name)
        prefix1: prefix label for AMR 1
        prefix2: prefix label for AMR 2
    Returns:
        best_match: the node mapping that results in the highest triple matching number
        best_match_num: the highest triple matching number

    """
    # Compute candidate pool - all possible node match candidates.
    # In the hill-climbing, we only consider candidate in this pool to save computing time.
    # weight_dict is a dictionary that maps a pair of node
    (candidate_mappings, weight_dict) = compute_pool(instance1, attribute1, relation1,
                                                     instance2, attribute2, relation2,
                                                     prefix1, prefix2)

    if verbose:
        print >> DEBUG_LOG, "Candidate mappings:"
        print >> DEBUG_LOG, candidate_mappings
        print >> DEBUG_LOG, "Weight dictionary"
        print >> DEBUG_LOG, weight_dict
    best_match_num = 0
    # initialize best match mapping
    # the ith entry is the node index in AMR 2 which maps to the ith node in AMR 1
    best_mapping = [-1] * len(instance1)
    for i in range(0, iteration_num):
        if verbose:
            print >> DEBUG_LOG, "Iteration", i
        if i == 0:
            # smart initialization used for the first round
            cur_mapping = smart_init_mapping(candidate_mappings, instance1, instance2)
        else:
            # random initialization for the other round
            cur_mapping = random_init_mapping(candidate_mappings)
        # compute current triple match number
        match_num = compute_match(cur_mapping, weight_dict)
        if verbose:
            print >> DEBUG_LOG, "Node mapping at start", cur_mapping
            print >> DEBUG_LOG, "Triple match number at start:", match_num
        while True:
            # get best gain
            (gain, new_mapping) = get_best_gain(cur_mapping, candidate_mappings, weight_dict,
                                                len(instance2), match_num)
            if verbose:
                print >> DEBUG_LOG, "Gain after the hill-climbing", gain
            # hill-climbing until there will be no gain for new node mapping
            if gain <= 0:
                break
            # otherwise update match_num and mapping
            match_num += gain
            cur_mapping = new_mapping[:]
            if verbose:
                print >> DEBUG_LOG, "Update triple match number to:", match_num
                print >> DEBUG_LOG, "Current mapping:", cur_mapping
        if match_num > best_match_num:
            best_mapping = cur_mapping[:]
            best_match_num = match_num
    return best_mapping, best_match_num


def compute_pool(instance1, attribute1, relation1,
                 instance2, attribute2, relation2,
                 prefix1, prefix2):
    """
    compute all possible node mapping candidates and their weights (the triple matching number gain resulting from
    mapping one node in AMR 1 to another node in AMR2)

    Arguments:
        instance1: instance triples of AMR 1
        attribute1: attribute triples of AMR 1 (attribute name, node name, attribute value)
        relation1: relation triples of AMR 1 (relation name, node 1 name, node 2 name)
        instance2: instance triples of AMR 2
        attribute2: attribute triples of AMR 2 (attribute name, node name, attribute value)
        relation2: relation triples of AMR 2 (relation name, node 1 name, node 2 name
        prefix1: prefix label for AMR 1
        prefix2: prefix label for AMR 2
    Returns:
      candidate_mapping: a list of candidate nodes.
                       The ith element contains the node indices (in AMR 2) the ith node (in AMR 1) can map to.
                       (resulting in non-zero triple match)
      weight_dict: a dictionary which contains the matching triple number for every pair of node mapping. The key
                   is a node pair. The value is another dictionary. key {-1} is triple match resulting from this node
                   pair alone (instance triples and attribute triples), and other keys are node pairs that can result
                   in relation triple match together with the first node pair.


    """
    candidate_mapping = []
    weight_dict = {}
    for i in range(0, len(instance1)):
        # each candidate mapping is a set of node indices
        candidate_mapping.append(set())
        for j in range(0, len(instance2)):
            # if both triples are instance triples and have the same value
            if instance1[i][0].lower() == instance2[j][0].lower() \
                    and instance1[i][2].lower() == instance2[j][2].lower():
                # get node index by stripping the prefix
                node1_index = int(instance1[i][1][len(prefix1):])
                node2_index = int(instance2[j][1][len(prefix2):])
                candidate_mapping[node1_index].add(node2_index)
                node_pair = (node1_index, node2_index)
                # use -1 as key in weight_dict for instance triples and attribute triples
                if node_pair in weight_dict:
                    weight_dict[node_pair][-1] += 1
                else:
                    weight_dict[node_pair] = {}
                    weight_dict[node_pair][-1] = 1
    for i in range(0, len(attribute1)):
        for j in range(0, len(attribute2)):
            # if both attribute relation triple have the same relation name and value
            if attribute1[i][0].lower() == attribute2[j][0].lower() \
                    and attribute1[i][2].lower() == attribute2[j][2].lower():
                node1_index = int(attribute1[i][1][len(prefix1):])
                node2_index = int(attribute2[j][1][len(prefix2):])
                candidate_mapping[node1_index].add(node2_index)
                node_pair = (node1_index, node2_index)
                # use -1 as key in weight_dict for instance triples and attribute triples
                if node_pair in weight_dict:
                    weight_dict[node_pair][-1] += 1
                else:
                    weight_dict[node_pair] = {}
                    weight_dict[node_pair][-1] = 1
    for i in range(0, len(relation1)):
        for j in range(0, len(relation2)):
            # if both relation share the same name
            if relation1[i][0].lower() == relation2[j][0].lower():
                node1_index_amr1 = int(relation1[i][1][len(prefix1):])
                node1_index_amr2 = int(relation2[j][1][len(prefix2):])
                node2_index_amr1 = int(relation1[i][2][len(prefix1):])
                node2_index_amr2 = int(relation2[j][2][len(prefix2):])
                # add mapping between two nodes
                candidate_mapping[node1_index_amr1].add(node1_index_amr2)
                candidate_mapping[node2_index_amr1].add(node2_index_amr2)
                node_pair1 = (node1_index_amr1, node1_index_amr2)
                node_pair2 = (node2_index_amr1, node2_index_amr2)
                if node_pair2 != node_pair1:
                    # update weight_dict weight. Note that we need to update both entries for future search
                    # i.e weight_dict[node_pair1][node_pair2]
                    #     weight_dict[node_pair2][node_pair1]
                    if node1_index_amr1 > node2_index_amr1:
                        # swap node_pair1 and node_pair2
                        node_pair1 = (node2_index_amr1, node2_index_amr2)
                        node_pair2 = (node1_index_amr1, node1_index_amr2)
                    if node_pair1 in weight_dict:
                        if node_pair2 in weight_dict[node_pair1]:
                            weight_dict[node_pair1][node_pair2] += 1
                        else:
                            weight_dict[node_pair1][node_pair2] = 1
                    else:
                        weight_dict[node_pair1] = {}
                        weight_dict[node_pair1][-1] = 0
                        weight_dict[node_pair1][node_pair2] = 1
                    if node_pair2 in weight_dict:
                        if node_pair1 in weight_dict[node_pair2]:
                            weight_dict[node_pair2][node_pair1] += 1
                        else:
                            weight_dict[node_pair2][node_pair1] = 1
                    else:
                        weight_dict[node_pair2] = {}
                        weight_dict[node_pair2][-1] = 0
                        weight_dict[node_pair2][node_pair1] = 1
                else:
                    # two node pairs are the same. So we only update weight_dict once.
                    # this generally should not happen.
                    if node_pair1 in weight_dict:
                        weight_dict[node_pair1][-1] += 1
                    else:
                        weight_dict[node_pair1] = {}
                        weight_dict[node_pair1][-1] = 1
    return candidate_mapping, weight_dict


def smart_init_mapping(candidate_mapping, instance1, instance2):
    """
    Initialize mapping based on the concept mapping (smart initialization)
    Arguments:
        candidate_mapping: candidate node match list
        instance1: instance triples of AMR 1
        instance2: instance triples of AMR 2
    Returns:
        initialized node mapping between two AMRs

    """
    random.seed()
    matched_dict = {}
    result = []
    # list to store node indices that have no concept match
    no_word_match = []
    for i, candidates in enumerate(candidate_mapping):
        if len(candidates) == 0:
            # no possible mapping
            result.append(-1)
            continue
        # node value in instance triples of AMR 1
        value1 = instance1[i][2]
        for node_index in candidates:
            value2 = instance2[node_index][2]
            # find the first instance triple match in the candidates
            # instance triple match is having the same concept value
            if value1 == value2:
                if node_index not in matched_dict:
                    result.append(node_index)
                    matched_dict[node_index] = 1
                    break
        if len(result) == i:
            no_word_match.append(i)
            result.append(-1)
    # if no concept match, generate a random mapping
    for i in no_word_match:
        candidates = list(candidate_mapping[i])
        while len(candidates) > 0:
            # get a random node index from candidates
            rid = random.randint(0, len(candidates) - 1)
            if candidates[rid] in matched_dict:
                candidates.pop(rid)
            else:
                matched_dict[candidates[rid]] = 1
                result[i] = candidates[rid]
                break
    return result
        

def random_init_mapping(candidate_mapping):
    """
    Generate a random node mapping.
    Args:
        candidate_mapping: candidate_mapping: candidate node match list
    Returns:
        randomly-generated node mapping between two AMRs

    """
    # if needed, a fixed seed could be passed here to generate same random (to help debugging)
    random.seed()
    matched_dict = {}
    result = []
    for c in candidate_mapping:
        candidates = list(c)
        if len(candidates) == 0:
            # -1 indicates no possible mapping
            result.append(-1)
            continue
        found = False
        while len(candidates) > 0:
            # randomly generate an index in [0, length of candidates)
            rid = random.randint(0, len(candidates) - 1)
            # check if it has already been matched
            if candidates[rid] in matched_dict:
                candidates.pop(rid)
            else:
                matched_dict[candidates[rid]] = 1
                result.append(candidates[rid])
                found = True
                break
        if not found:
            result.append(-1)
    return result

 
def compute_match(mapping, weight_dict):
    """
    Given a node mapping, compute match number based on weight_dict.
    Args:
    mappings: a list of node index in AMR 2. The ith element (value j) means node i in AMR 1 maps to node j in AMR 2.
    Returns:
    matching triple number
    Complexity: O(m*n) , m is the node number of AMR 1, n is the node number of AMR 2

    """
    # If this mapping has been investigated before, retrieve the value instead of re-computing.
    if verbose:
        print >> DEBUG_LOG, "Computing match for mapping"
        print >> DEBUG_LOG, mapping
    if tuple(mapping) in match_triple_dict:
        if verbose:
            print >> DEBUG_LOG, "saved value", match_triple_dict[tuple(mapping)]
        return match_triple_dict[tuple(mapping)]
    match_num = 0
    # i is node index in AMR 1, m is node index in AMR 2
    for i, m in enumerate(mapping):
        if m == -1:
            # no node maps to this node
            continue
        # node i in AMR 1 maps to node m in AMR 2
        current_node_pair = (i, m)
        if current_node_pair not in weight_dict:
            continue
        if verbose:
            print >> DEBUG_LOG, "node_pair", current_node_pair
        for key in weight_dict[current_node_pair]:
            if key == -1:
                # matching triple resulting from instance/attribute triples
                match_num += weight_dict[current_node_pair][key]
                if verbose:
                    print >> DEBUG_LOG, "instance/attribute match", weight_dict[current_node_pair][key]
            # only consider node index larger than i to avoid duplicates
            # as we store both weight_dict[node_pair1][node_pair2] and
            #     weight_dict[node_pair2][node_pair1] for a relation
            elif key[0] < i:
                continue
            elif mapping[key[0]] == key[1]:
                match_num += weight_dict[current_node_pair][key]
                if verbose:
                    print >> DEBUG_LOG, "relation match with", key, weight_dict[current_node_pair][key]
    if verbose:
        print >> DEBUG_LOG, "match computing complete, result:", match_num
    # update match_triple_dict
    match_triple_dict[tuple(mapping)] = match_num
    return match_num  


def move_gain(mapping, node_id, old_id, new_id, weight_dict, match_num):
    """
    Compute the triple match number gain from the move operation
    Arguments:
        mapping: current node mapping
        node_id: remapped node in AMR 1
        old_id: original node id in AMR 2 to which node_id is mapped
        new_id: new node in to which node_id is mapped
        weight_dict: weight dictionary
        match_num: the original triple matching number
    Returns:
        the triple match gain number (might be negative)

    """
    # new node mapping after moving
    new_mapping = (node_id, new_id)
    # node mapping before moving
    old_mapping = (node_id, old_id)
    # new nodes mapping list (all node pairs)
    new_mapping_list = mapping[:]
    new_mapping_list[node_id] = new_id
    # if this mapping is already been investigated, use saved one to avoid duplicate computing
    if tuple(new_mapping_list) in match_triple_dict:
        return match_triple_dict[tuple(new_mapping_list)] - match_num
    gain = 0
    # add the triple match incurred by new_mapping to gain
    if new_mapping in weight_dict:
        for key in weight_dict[new_mapping]:
            if key == -1:
                # instance/attribute triple match
                gain += weight_dict[new_mapping][-1]
            elif new_mapping_list[key[0]] == key[1]:
                # relation gain incurred by new_mapping and another node pair in new_mapping_list
                gain += weight_dict[new_mapping][key]
    # deduct the triple match incurred by old_mapping from gain
    if old_mapping in weight_dict:
        for k in weight_dict[old_mapping]:
            if k == -1:
                gain -= weight_dict[old_mapping][-1]
            elif mapping[k[0]] == k[1]:
                gain -= weight_dict[old_mapping][k]
    # update match number dictionary
    match_triple_dict[tuple(new_mapping_list)] = match_num + gain
    return gain


def swap_gain(mapping, node_id1, mapping_id1, node_id2, mapping_id2, weight_dict, match_num):
    """
    Compute the triple match number gain from the swapping
    Arguments:
    mapping: current node mapping list
    node_id1: node 1 index in AMR 1
    mapping_id1: the node index in AMR 2 node 1 maps to (in the current mapping)
    node_id2: node 2 index in AMR 1
    mapping_id2: the node index in AMR 2 node 2 maps to (in the current mapping)
    weight_dict: weight dictionary
    match_num: the original matching triple number
    Returns:
    the gain number (might be negative)

    """
    new_mapping_list = mapping[:]
    # Before swapping, node_id1 maps to mapping_id1, and node_id2 maps to mapping_id2
    # After swapping, node_id1 maps to mapping_id2 and node_id2 maps to mapping_id1
    new_mapping_list[node_id1] = mapping_id2
    new_mapping_list[node_id2] = mapping_id1
    if tuple(new_mapping_list) in match_triple_dict:
        return match_triple_dict[tuple(new_mapping_list)] - match_num
    gain = 0
    new_mapping1 = (node_id1, mapping_id2)
    new_mapping2 = (node_id2, mapping_id1)
    old_mapping1 = (node_id1, mapping_id1)
    old_mapping2 = (node_id2, mapping_id2)
    if node_id1 > node_id2:
        new_mapping2 = (node_id1, mapping_id2)
        new_mapping1 = (node_id2, mapping_id1)
        old_mapping1 = (node_id2, mapping_id2)
        old_mapping2 = (node_id1, mapping_id1)
    if new_mapping1 in weight_dict:
        for key in weight_dict[new_mapping1]:
            if key == -1:
                gain += weight_dict[new_mapping1][-1]
            elif new_mapping_list[key[0]] == key[1]:
                gain += weight_dict[new_mapping1][key]
    if new_mapping2 in weight_dict:
        for key in weight_dict[new_mapping2]:
            if key == -1:
                gain += weight_dict[new_mapping2][-1]
            # to avoid duplicate
            elif key[0] == node_id1:
                continue
            elif new_mapping_list[key[0]] == key[1]:
                gain += weight_dict[new_mapping2][key]
    if old_mapping1 in weight_dict:
        for key in weight_dict[old_mapping1]:
            if key == -1:
                gain -= weight_dict[old_mapping1][-1]
            elif mapping[key[0]] == key[1]:
                gain -= weight_dict[old_mapping1][key]
    if old_mapping2 in weight_dict:
        for key in weight_dict[old_mapping2]:
            if key == -1:
                gain -= weight_dict[old_mapping2][-1]
            # to avoid duplicate
            elif key[0] == node_id1:
                continue
            elif mapping[key[0]] == key[1]:
                gain -= weight_dict[old_mapping2][key]
    match_triple_dict[tuple(new_mapping_list)] = match_num + gain
    return gain


def get_best_gain(mapping, candidate_mappings, weight_dict, instance_len, cur_match_num):
    """
    Hill-climbing method to return the best gain swap/move can get
    Arguments:
    mapping: current node mapping
    candidate_mappings: the candidates mapping list
    weight_dict: the weight dictionary
    instance_len: the number of the nodes in AMR 2
    cur_match_num: current triple match number
    Returns:
    the best gain we can get via swap/move operation

    """
    largest_gain = 0
    # True: using swap; False: using move
    use_swap = True
    # the node to be moved/swapped
    node1 = None
    # store the other node affected. In swap, this other node is the node swapping with node1. In move, this other
    # node is the node node1 will move to.
    node2 = None
    # unmatched nodes in AMR 2
    unmatched = set(range(0, instance_len))
    # exclude nodes in current mapping
    # get unmatched nodes
    for nid in mapping:
        if nid in unmatched:
            unmatched.remove(nid)
    for i, nid in enumerate(mapping):
        # current node i in AMR 1 maps to node nid in AMR 2
        for nm in unmatched:
            if nm in candidate_mappings[i]:
                # remap i to another unmatched node (move)
                # (i, m) -> (i, nm)
                if verbose:
                    print >> DEBUG_LOG, "Remap node", i, "from ", nid, "to", nm
                mv_gain = move_gain(mapping, i, nid, nm, weight_dict, cur_match_num)
                if verbose:
                    print >> DEBUG_LOG, "Move gain:", mv_gain
                    new_mapping = mapping[:]
                    new_mapping[i] = nm
                    new_match_num = compute_match(new_mapping, weight_dict)
                    if new_match_num != cur_match_num + mv_gain:
                        print >> ERROR_LOG, mapping, new_mapping
                        print >> ERROR_LOG, "Inconsistency in computing: move gain", cur_match_num, mv_gain, \
                            new_match_num
                if mv_gain > largest_gain:
                    largest_gain = mv_gain
                    node1 = i
                    node2 = nm
                    use_swap = False
    # compute swap gain
    for i, m in enumerate(mapping):
        for j in range(i+1, len(mapping)):
            m2 = mapping[j]
            # swap operation (i, m) (j, m2) -> (i, m2) (j, m)
            # j starts from i+1, to avoid duplicate swap
            if verbose:
                print >> DEBUG_LOG, "Swap node", i, "and", j
                print >> DEBUG_LOG, "Before swapping:", i, "-", m, ",", j, "-", m2
                print >> DEBUG_LOG, mapping
                print >> DEBUG_LOG, "After swapping:", i, "-", m2, ",", j, "-", m
            sw_gain = swap_gain(mapping, i, m, j, m2, weight_dict, cur_match_num)
            if verbose:
                print >> DEBUG_LOG, "Swap gain:", sw_gain
                new_mapping = mapping[:]
                new_mapping[i] = m2
                new_mapping[j] = m
                print >> DEBUG_LOG, new_mapping
                new_match_num = compute_match(new_mapping, weight_dict)
                if new_match_num != cur_match_num + sw_gain:
                    print >> ERROR_LOG, match, new_match
                    print >> ERROR_LOG, "Inconsistency in computing: swap gain", cur_match_num, sw_gain, new_match_num
            if sw_gain > largest_gain:
                largest_gain = sw_gain
                node1 = i
                node2 = j
                use_swap = True
    # generate a new mapping based on swap/move
    cur_mapping = mapping[:]
    if node1 is not None:
        if use_swap:
            if verbose:
                print >> DEBUG_LOG, "Use swap gain"
            temp = cur_mapping[node1]
            cur_mapping[node1] = cur_mapping[node2]
            cur_mapping[node2] = temp
        else:
            if verbose:
                print >> DEBUG_LOG, "Use move gain"
            cur_mapping[node1] = node2
    else:
        if verbose:
            print >> DEBUG_LOG, "no move/swap gain found"
    if verbose:
        print >> DEBUG_LOG, "Original mapping", mapping
        print >> DEBUG_LOG, "Current mapping", cur_mapping
    return largest_gain, cur_mapping


def print_alignment(mapping, instance1, instance2):
    """
    print the alignment based on a node mapping
    Args:
        match: current node mapping list
        instance1: nodes of AMR 1
        instance2: nodes of AMR 2

    """
    result = []
    for i, m in enumerate(mapping):
        if m == -1:
            result.append(instance1[i][1] + "(" + instance1[i][2] + ")" + "-Null")
        else:
            result.append(instance1[i][1] + "(" + instance1[i][2] + ")" + "-"
                          + instance2[m][1] + "(" + instance2[m][2] + ")")
    return " ".join(result)


def compute_f(match_num, test_num, gold_num):
    """
    Compute the f-score based on the matching triple number,
                                 triple number of AMR set 1,
                                 triple number of AMR set 2
    Args:
        match_num: matching triple number
        test_num:  triple number of AMR 1 (test file)
        gold_num:  triple number of AMR 2 (gold file)
    Returns:
        precision: match_num/test_num
        recall: match_num/gold_num
        f_score: 2*precision*recall/(precision+recall)
    """
    if test_num == 0 or gold_num == 0:
        return 0.00, 0.00, 0.00
    precision = (0.000 + match_num) / (test_num + 0.000)
    recall = (0.000 + match_num) / (gold_num + 0.000)
    if (precision + recall) != 0:
        f_score = 2 * precision * recall / (precision + recall)
        if verbose:
            print >> DEBUG_LOG, "F-score:", f_score
        return precision, recall, f_score
    else:
        if verbose:
            print >> DEBUG_LOG, "F-score:", "0.0"
        return precision, recall, 0.00


def main(list1, list2):
    """
    Main function of smatch score calculation

    """
    global verbose
    global iteration_num
    global single_score
    global pr_flag
    global match_triple_dict
    # set the iteration number
    # total iteration number = restart number + 1
    iteration_num = 5
    #if arguments.ms:
    #    single_score = False
    #if arguments.v:
    #    verbose = True
    #if arguments.pr:
    pr_flag = True
    # matching triple number
    total_match_num = 0
    # triple number in test file
    total_test_num = 0
    # triple number in gold file
    total_gold_num = 0
    # sentence number
    sent_num = 1
    for l1, l2 in zip(list1,list2):
        lst_amr1, dic_amr1 = l1
        lst_amr2, dic_amr2 = l2
	amr1 = parse_relations(lst_amr1, dic_amr1)
        amr2 = parse_relations(lst_amr2, dic_amr2)
	prefix1 = "a"
        prefix2 = "b"
        # Rename node to "a1", "a2", .etc
        amr1.rename_node(prefix1)
        # Renaming node to "b1", "b2", .etc
        amr2.rename_node(prefix2)
        (instance1, attributes1, relation1) = amr1.get_triples()
        (instance2, attributes2, relation2) = amr2.get_triples()
        if verbose:
            # print parse results of two AMRs
            print >> DEBUG_LOG, "AMR pair", sent_num
            print >> DEBUG_LOG, "============================================"
            #print >> DEBUG_LOG, "AMR 1 (one-line):", cur_amr1
            #print >> DEBUG_LOG, "AMR 2 (one-line):", cur_amr2
            print >> DEBUG_LOG, "Instance triples of AMR 1:", len(instance1)
            print >> DEBUG_LOG, instance1
            print >> DEBUG_LOG, "Attribute triples of AMR 1:", len(attributes1)
            print >> DEBUG_LOG, attributes1
            print >> DEBUG_LOG, "Relation triples of AMR 1:", len(relation1)
            print >> DEBUG_LOG, relation1
            print >> DEBUG_LOG, "Instance triples of AMR 2:", len(instance2)
            print >> DEBUG_LOG, instance2
            print >> DEBUG_LOG, "Attribute triples of AMR 2:", len(attributes2)
            print >> DEBUG_LOG, attributes2
            print >> DEBUG_LOG, "Relation triples of AMR 2:", len(relation2)
            print >> DEBUG_LOG, relation2
        (best_mapping, best_match_num) = get_best_match(instance1, attributes1, relation1,
                                                        instance2, attributes2, relation2,
                                                        prefix1, prefix2)
        if verbose:
            print >> DEBUG_LOG, "best match number", best_match_num
            print >> DEBUG_LOG, "best node mapping", best_mapping
            print >> DEBUG_LOG, "Best node mapping alignment:", print_alignment(best_mapping, instance1, instance2)
	test_triple_num = len(instance1) + len(attributes1) + len(relation1)
        gold_triple_num = len(instance2) + len(attributes2) + len(relation2)
	if not single_score:
            # if each AMR pair should have a score, compute and output it here
            (precision, recall, best_f_score) = compute_f(best_match_num,
                                                          test_triple_num,
                                                          gold_triple_num)
            #print "Sentence", sent_num
            if pr_flag:
                print "Precision: %.2f" % precision
                print "Recall: %.2f" % recall
#            print "Smatch score: %.2f" % best_f_score
            print "%.4f" % best_f_score
	total_match_num += best_match_num
        total_test_num += test_triple_num
        total_gold_num += gold_triple_num
        # clear the matching triple dictionary for the next AMR pair
        match_triple_dict.clear()
        sent_num += 1
    if verbose:
        print >> DEBUG_LOG, "Total match number, total triple number in AMR 1, and total triple number in AMR 2:"
        print >> DEBUG_LOG, total_match_num, total_test_num, total_gold_num
        print >> DEBUG_LOG, "---------------------------------------------------------------------------------"
    # output document-level smatch score (a single f-score for all AMR pairs in two files)
    return compute_f(total_match_num, total_test_num, total_gold_num)
