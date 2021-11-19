import sys
import os

from deal_cvt import load_cvt, is_cvt
from utils import is_ent


def load_seed(filename):
    f = open(filename)
    seed_set = set()
    for line in f:
        seed_set.add(line.strip())
    f.close()
    return seed_set


def fetch_triple_1hop(kb_file, seed_file, output, cvt_nodes, cvt_hop=True):
    seed_set = load_seed(seed_file)
    cvt_set = set()
    f_out = open(output, "w")

    num_tot = 0
    num_res = 0
    f = open(kb_file)
    for line in f:
        spline = line.strip().split("\t")
        num_tot += 1

        if spline[0] in seed_set or spline[2] in seed_set:  # bidirectional
            f_out.write(line)
            num_res += 1

            if cvt_hop:
                if spline[0] in seed_set and spline[2] not in seed_set and is_cvt(spline[2], cvt_nodes):
                    cvt_set.add(spline[2])
                elif spline[2] in seed_set and spline[0] not in seed_set and is_cvt(spline[0], cvt_nodes):
                    cvt_set.add(spline[0])

        # if num_tot % 1000000 == 0:
        #     print("seed-hop", num_tot, num_res)
    f.close()

    num_tot = 0
    num_res = 0
    if cvt_hop:
        cvt_set = cvt_set - seed_set
        f = open(kb_file)
        for line in f:
            num_tot += 1
            spline = line.strip().split("\t")

            if (spline[0] in cvt_set and spline[2] not in seed_set) or \
                    (spline[2] in cvt_set and spline[0] not in seed_set):
                f_out.write(line)
                num_res += 1
            # if num_tot % 1000000 == 0:
            #     print("cvt-hop", num_tot, num_res)
        f.close()
    f_out.close()
    return cvt_set


def filter_ent_from_triple(in_file, out_file):
    f = open(in_file)
    ent_set = set()
    for line in f:
        line = line.strip().split("\t")
        # if is_ent(line[0]):
        #     ent_set.add(line[0])
        # if is_ent(line[2]):
        #     ent_set.add(line[2])
        ent_set.update([line[0], line[2]])
    f.close()
    f = open(out_file, "w")
    for ent in ent_set:
        f.write(ent + "\n")
    f.close()


if __name__ == "__main__":
    cvt_nodes = load_cvt('data/cvtnodes.bin')

    dataset_dir = sys.argv[1]
    seed_file = os.path.join(dataset_dir, 'seed.txt')

    kb_file = 'manual_fb_filter.txt'

    output_hop1 = os.path.join(dataset_dir, 'subgraph_hop1.txt')
    if os.path.exists(output_hop1):
        print("Skip 1st hop")
    else:
        fetch_triple_1hop(kb_file=kb_file, seed_file=seed_file, output=output_hop1, cvt_nodes=cvt_nodes, cvt_hop=True)

    hop1_ent_file = os.path.join(dataset_dir, 'ent_hop1.txt')
    if os.path.exists(hop1_ent_file):
        print("Skip ent fetch for 1st hop subgraph")
    else:
        filter_ent_from_triple(in_file=output_hop1, out_file=hop1_ent_file)

    output_hop2 = os.path.join(dataset_dir, "subgraph_hop2.txt")
    if os.path.exists(output_hop2):
        print("Skip 2nd hop")
    else:
        fetch_triple_1hop(kb_file=kb_file, seed_file=hop1_ent_file, output=output_hop2, cvt_nodes=cvt_nodes,
                          cvt_hop=True)
