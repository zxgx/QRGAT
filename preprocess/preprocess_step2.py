import pickle
import time
import sys
import os
from collections import Counter

from utils import is_ent
from deal_cvt import is_cvt, load_cvt


def load_seed(filename):
    f = open(filename)
    seed_set = set()
    for line in f:
        seed_set.add(line.strip())
    f.close()
    return seed_set


def fetch_triple_1hop(kb_file, seed_file, output, cvt_nodes):
    seed_set = load_seed(seed_file)
    f_out = open(output, "w")

    cvt_set = set()
    f = open(kb_file)
    for line in f:
        spline = line.strip().split("\t")

        if spline[0] in seed_set or spline[2] in seed_set:  # bidirectional
            f_out.write(line)

            if spline[0] in seed_set and spline[2] not in seed_set and is_cvt(spline[2], cvt_nodes):
                cvt_set.add(spline[2])

    f.close()

    f = open(kb_file)
    for line in f:
        head, rel, tail = line.strip().split('\t')
        if head in cvt_set:
            f_out.write(line)
    f.close()
    f_out.close()


def filter_ent_from_triple(in_file, out_file):
    f = open(in_file)
    ent_set = set()
    for line in f:
        line = line.strip().split("\t")
        if is_ent(line[0]):
            ent_set.add(line[0])
        if is_ent(line[2]):
            ent_set.add(line[2])
        # ent_set.update([line[0], line[2]])
    f.close()
    f = open(out_file, "w")
    for ent in ent_set:
        f.write(ent + "\n")
    f.close()


def build_index(subgraph_file, seed_file, idx_graph_path, ent_path, rel_path, stop_ent_path):
    ent_dict, rel_dict = dict(), dict()
    ent_counter = Counter()
    f = open(subgraph_file)
    f_out = open(idx_graph_path, 'w')
    for line in f:
        head, rel, tail = line.strip().split('\t')

        if is_ent(tail):
            ent_counter.update([tail])

        if head not in ent_dict:
            ent_dict[head] = len(ent_dict)
        head = ent_dict[head]

        if tail not in ent_dict:
            ent_dict[tail] = len(ent_dict)
        tail = ent_dict[tail]

        if rel not in rel_dict:
            rel_dict[rel] = len(rel_dict)
        rel = rel_dict[rel]

        f_out.write('\t'.join([str(head), str(rel), str(tail)]) + '\n')
    f.close()
    f_out.close()

    print("Entity size: %d, Relation Size: %d" % (len(ent_dict), len(rel_dict)))

    f_out = open(ent_path, 'w')
    for ent in ent_dict:
        f_out.write(ent+'\n')
    f_out.close()

    f_out = open(rel_path, 'w')
    for rel in rel_dict:
        f_out.write(rel+'\n')
    f_out.close()

    f_out = open(stop_ent_path, 'w')
    for e, c in ent_counter.most_common(150):
        f_out.write("%s\t%d\n" % (e, c))
    f_out.close()

    with open(seed_file) as f:
        for line in f:
            line = line.strip()
            if line not in ent_dict:
                print("Seed: '%s' not in ent.txt" % line)


if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    seed_file = os.path.join(dataset_dir, 'seed.txt')

    kb_file = 'manual_fb_filter.txt'

    cvt_pkl = 'data/cvt.pkl'
    tick = time.time()
    if not os.path.exists(cvt_pkl):
        cvt_file = 'data/cvtnodes.bin'
        cvt_nodes = load_cvt(cvt_file)
        with open(cvt_pkl, 'wb') as f:
            pickle.dump(cvt_nodes, f)
    else:
        with open(cvt_pkl, 'rb') as f:
            cvt_nodes = pickle.load(f)
    print("%.2fs for loading cvt nodes" % (time.time() - tick))

    output_hop1 = os.path.join(dataset_dir, 'subgraph_hop1.txt')
    if os.path.exists(output_hop1):
        print("Skip 1st hop")
    else:
        fetch_triple_1hop(kb_file=kb_file, seed_file=seed_file, output=output_hop1, cvt_nodes=cvt_nodes)

    hop1_ent_file = os.path.join(dataset_dir, 'ent_hop1.txt')
    if os.path.exists(hop1_ent_file):
        print("Skip ent fetch for 1st hop subgraph")
    else:
        filter_ent_from_triple(in_file=output_hop1, out_file=hop1_ent_file)

    output_hop2 = os.path.join(dataset_dir, "subgraph_hop2.txt")
    if os.path.exists(output_hop2):
        print("Skip 2nd hop")
    else:
        fetch_triple_1hop(kb_file=kb_file, seed_file=hop1_ent_file, output=output_hop2, cvt_nodes=cvt_nodes)

    subgraph_file = os.path.join(dataset_dir, 'subgraph.txt')
    ent_path = os.path.join(dataset_dir, 'ent.txt')
    rel_path = os.path.join(dataset_dir, 'rel.txt')
    stop_ent_path = os.path.join(dataset_dir, 'stop_ent.txt')
    if os.path.exists(subgraph_file) and os.path.exists(ent_path) and os.path.exists(rel_path) \
            and os.path.exists(stop_ent_path):
        print("Skip index subgraph")
    else:
        build_index(output_hop2, seed_file, subgraph_file, ent_path, rel_path, stop_ent_path)
