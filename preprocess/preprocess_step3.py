import sys
import os
import time
import json
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from deal_cvt import load_cvt, is_cvt
from ppr_util import rank_ppr_ents
from utils import load_dict


def _get_answer_coverage(answers, entities):
    if len(answers) == 0:
        return 1.0
    found, total = 0., 0
    for answer in answers:
        if answer["kb_id"] in entities:
            found += 1.
        total += 1
    return found / total


def get_subgraph(cand_ents, triples, idx2ent, idx2rel, cvt_add_flag=True):
    triple_set, reserve_set = set(), set()

    for ent in cand_ents:
        if ent in triples:
            triple_set |= triples[ent]

    tp_edges = {}
    for triple in triple_set:
        head, rel, tail = triple
        if head not in cand_ents or tail not in cand_ents:
            continue
        reserve_set.add((head, rel, tail))

        if is_cvt(idx2ent[head], cvt_nodes):
            tp_edges.setdefault(head, set())
            tp_edges[head].add((head, rel, tail))
        elif is_cvt(idx2ent[tail], cvt_nodes):
            tp_edges.setdefault(tail, set())
            tp_edges[tail].add((head, rel, tail))

    # remove or augment single cvt node
    for ent in tp_edges:
        if len(tp_edges[ent]) == 1:
            triple = tp_edges[ent].pop()

            if cvt_add_flag and len(triples[ent]) > 1:
                reserve_set |= triples[ent]
            else:
                reserve_set.remove(triple)

    ent_set, triple_list = set(), list()
    for triple in reserve_set:
        head, rel, tail = triple
        head = idx2ent[head]
        rel = idx2rel[rel]
        tail = idx2ent[tail]
        triple_list.append((head, rel, tail))
        ent_set.add(head)
        ent_set.add(tail)

    return list(ent_set), triple_list


def build_sp_mat(subgraph):
    ent2id = {}
    rel2id = {}
    rows = []
    cols = []
    for triple in subgraph:
        head, rel, tail = triple
        if head not in ent2id:
            ent2id[head] = len(ent2id)
        if tail not in ent2id:
            ent2id[tail] = len(ent2id)
        if rel not in rel2id:
            rel2id[rel] = len(rel2id)
        head_id = ent2id[head]
        tail_id = ent2id[tail]
        rows.append(head_id)
        rows.append(tail_id)
        cols.append(tail_id)
        cols.append(head_id)
    print("Number Entity : {}, Relation : {}".format(len(ent2id), len(rel2id)))
    print("Number Triple : {}".format(len(subgraph)))
    vals = np.ones((len(rows),))
    rows = np.array(rows)
    cols = np.array(cols)
    sp_mat = csr_matrix((vals, (rows, cols)), shape=(len(ent2id), len(ent2id)))

    return ent2id, rel2id, normalize(sp_mat, norm="l1", axis=1)


def get_2hop_triples(triples, idx2ent, seed_set):
    # all cvt cases are taken into consideration
    triple_set, hop1_triples = set(), set()
    for seed_ent in seed_set:
        if seed_ent not in triples:
            continue
        hop1_triples |= triples[seed_ent]
    triple_set |= hop1_triples
    for head, rel, tail in hop1_triples:
        if tail in triples:
            triple_set |= triples[tail]
        if head in triples:
            triple_set |= triples[head]

    # remove edges that contains single cvt node
    cvt_edges = dict()
    for triple in triple_set:
        head, rel, tail = triple

        if is_cvt(idx2ent[head], cvt_nodes):
            cvt_edges.setdefault(head, set())
            cvt_edges[head].add(triple)
        elif is_cvt(idx2ent[tail], cvt_nodes):
            cvt_edges.setdefault(tail, set())
            cvt_edges[tail].add(triple)

    for cvt_node in cvt_edges:
        if len(cvt_edges[cvt_node]) == 1:
            triple = cvt_edges[cvt_node].pop()
            triple_set.remove(triple)

    return triple_set


def load_kb(kb_file, ent_file, rel_file):
    ent2idx, idx2ent = load_dict(ent_file)
    _, idx2rel = load_dict(rel_file)

    triples = {}
    f = open(kb_file)
    for line in f:
        head, rel, tail = line.strip().split("\t")
        head, rel, tail = int(head), int(rel), int(tail)

        triples.setdefault(head, set())
        triples[head].add((head, rel, tail))
        triples.setdefault(tail, set())
        triples[tail].add((head, rel, tail))
    f.close()

    return triples, ent2idx, idx2ent, idx2rel


def retrieve_subgraph(kb_file, ent_file, rel_file, in_file, out_file=None, max_ent=2000):
    st = time.time()
    triples, ent2idx, idx2ent, idx2rel = load_kb(kb_file, ent_file, rel_file)
    print("Load KB", time.time() - st)  # ~37 min and ~160G memory for webqsp

    f = open(in_file)
    f_out = open(out_file, "w")
    answer_coverage = []
    for i, line in enumerate(f):
        data = json.loads(line)

        entity_set = set()
        for entity in data['entities']:
            if entity in ent2idx:
                entity_set.add(ent2idx[entity])
        tick = time.time()
        triple_set = get_2hop_triples(triples, idx2ent, entity_set)
        print("get 2 hop subgraph:", time.time() - tick)

        if len(entity_set) == 0 or len(triple_set) == 0:
            extracted_tuples, extracted_entities = [], []
            print("Bad question", data['id'])
        else:
            tick = time.time()
            ent2id, rel2id, sp_mat = build_sp_mat(triple_set)
            print("build sparse matrix:", time.time() - tick)

            seed_list = []
            for ent in entity_set:
                if ent in ent2id:
                    seed_list.append(ent2id[ent])
            tick = time.time()
            extracted_ents = rank_ppr_ents(seed_list, sp_mat, max_ent=max_ent)
            print("PPR:", time.time() - tick)
            id2ent = {v: k for k, v in ent2id.items()}
            ent_set_new = set(id2ent[ent] for ent in extracted_ents.tolist()) | entity_set
            tick = time.time()
            extracted_entities, extracted_tuples = get_subgraph(ent_set_new, triples, idx2ent, idx2rel, cvt_add_flag=False)
            print("result subgrpah:", time.time() - tick)

        data['subgraph'] = {}
        data['subgraph']['tuples'] = extracted_tuples
        data['subgraph']['entities'] = extracted_entities
        f_out.write(json.dumps(data) + "\n")

        answer_coverage.append(_get_answer_coverage(data["answers"], extracted_entities))
        if i % 100 == 0:
            print("{} samples cost time:{:.1f} with coverage{:.3f}".format(i,
                                                                           time.time() - st, np.mean(answer_coverage)))
    f.close()
    f_out.close()
    print("Finish", time.time() - st)
    print("Answer coverage in retrieved subgraphs = %.3f" % (np.mean(answer_coverage)))


cvt_nodes = load_cvt('data/cvtnodes.bin')

dataset_dir = sys.argv[1]
graph_file = os.path.join(dataset_dir, 'subgraph.txt')
ent_file = os.path.join(dataset_dir, 'ent.txt')
rel_file = os.path.join(dataset_dir, 'rel.txt')
qa_file = os.path.join(dataset_dir, 'step1.json')
output_file = os.path.join(dataset_dir, 'qa.raw.json')

retrieve_subgraph(graph_file, ent_file, rel_file, qa_file, output_file, max_ent=2000)
