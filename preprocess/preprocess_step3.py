import pickle
import sys
import os
import time
import json
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from deal_cvt import load_cvt, is_cvt
from ppr_util import rank_ppr_ents
from utils import load_dict, is_ent


def _get_answer_coverage(answers, entities):
    if len(answers) == 0:
        return 1.0
    found, total = 0., 0
    for answer in answers:
        if answer["kb_id"] in entities:
            found += 1.
        total += 1
    return found / total


def get_subgraph(cand_ents, triple_set, idx2ent, idx2rel, cvt_nodes, cvt_add_flag=False):
    reserve_set = set()

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


def get_2hop_triples(triples, idx2ent, seed_set, cvt_nodes, stop_ent):
    # all cvt cases are taken into consideration
    triple_set, hop1_triples = set(), set()
    trp_cache, ent_cache = set(), set()
    for seed_ent in seed_set:
        if seed_ent in triples:
            if seed_ent not in stop_ent and is_ent(idx2ent[seed_ent]):
                hop1_triples |= triples[seed_ent]
            else:
                trp_cache |= triples[seed_ent]
                ent_cache.add(seed_ent)

    if len(hop1_triples) == 0:
        triple_set |= trp_cache
        hop1_triples = trp_cache
    else:
        triple_set |= hop1_triples

    for head, rel, tail in hop1_triples:
        if tail not in seed_set and tail in triples and tail not in stop_ent and is_ent(idx2ent[tail]):
            triple_set |= triples[tail]
        if head not in seed_set and head in triples and head not in stop_ent and is_ent(idx2ent[head]):
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


def retrieve_subgraph(cvt_nodes, triples, ent2idx, idx2ent, idx2rel, stop_ent, qa_data, start, end, output_file,
                      max_ent=2000):
    st = time.time()

    f_out = open(output_file, "w")
    for idx in range(start, end):
        data = qa_data[idx]
        print("\n\nProcessing: %d (%d - %d), id: %s" % (idx, start, end-1, data['id']))

        entity_set = set()
        for entity in data['entities']:
            if entity in ent2idx:
                entity_set.add(ent2idx[entity])
        tick = time.time()
        triple_set = get_2hop_triples(triples, idx2ent, entity_set, cvt_nodes, stop_ent)
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
            extracted_entities, extracted_tuples = get_subgraph(ent_set_new, triple_set, idx2ent, idx2rel, cvt_nodes)
            print("result subgrpah:", time.time() - tick)

        data['subgraph'] = {}
        data['subgraph']['tuples'] = extracted_tuples
        data['subgraph']['entities'] = extracted_entities
        f_out.write(json.dumps(data) + "\n")

    f_out.close()
    print("Finish", time.time() - st)


if __name__ == '__main__':
    dataset_dir = sys.argv[1]
    pid = int(sys.argv[2])
    num_process = 1

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

    graph_pkl = os.path.join(dataset_dir, 'subgraph.pkl')
    tick = time.time()
    if not os.path.exists(graph_pkl):
        graph_file = os.path.join(dataset_dir, 'subgraph.txt')
        triples = {}
        with open(graph_file) as f:
            for line in f:
                head, rel, tail = line.strip().split("\t")
                head, rel, tail = int(head), int(rel), int(tail)

                triples.setdefault(head, set())
                triples[head].add((head, rel, tail))
                triples.setdefault(tail, set())
                triples[tail].add((head, rel, tail))

        with open(graph_pkl, 'wb') as f:
            pickle.dump(triples, f)
    else:
        with open(graph_pkl, 'rb') as f:
            triples = pickle.load(f)
    print("%.2fs for loading indexed graph" % (time.time() - tick))

    ent2idx_pkl = os.path.join(dataset_dir, 'ent2idx.pkl')
    idx2ent_pkl = os.path.join(dataset_dir, 'idx2ent.pkl')
    tick = time.time()
    if not (os.path.exists(ent2idx_pkl) and os.path.exists(idx2ent_pkl)):
        ent_file = os.path.join(dataset_dir, 'ent.txt')
        ent2idx, idx2ent = load_dict(ent_file)

        with open(ent2idx_pkl, 'wb') as f:
            pickle.dump(ent2idx, f)

        with open(idx2ent_pkl, 'wb') as f:
            pickle.dump(idx2ent, f)
    else:
        with open(ent2idx_pkl, 'rb') as f:
            ent2idx = pickle.load(f)
        with open(idx2ent_pkl, 'rb') as f:
            idx2ent = pickle.load(f)
    print("%.2fs for loading entity dict" % (time.time() - tick))

    idx2rel_pkl = os.path.join(dataset_dir, 'idx2rel.pkl')
    tick = time.time()
    if not os.path.exists(idx2rel_pkl):
        rel_file = os.path.join(dataset_dir, 'rel.txt')
        _, idx2rel = load_dict(rel_file)

        with open(idx2rel_pkl, 'wb') as f:
            pickle.dump(idx2rel, f)
    else:
        with open(idx2rel_pkl, 'rb') as f:
            idx2rel = pickle.load(f)
    print("%.2fs for loading relation dict" % (time.time() - tick))

    stop_ent_pkl = os.path.join(dataset_dir, 'stop_ent.pkl')
    tick = time.time()
    if not os.path.exists(stop_ent_pkl):
        stop_ent_path = os.path.join(dataset_dir, 'stop_ent.txt')
        stop_ent = set()
        with open(stop_ent_path) as f:
            for i, line in enumerate(f):
                ent, _ = line.strip().split('\t')
                stop_ent.add(ent2idx[ent])

        with open(stop_ent_pkl, 'wb') as f:
            pickle.dump(stop_ent, f)
    else:
        with open(stop_ent_pkl, 'rb') as f:
            stop_ent = pickle.load(f)
    print("%.2fs for loading stop entities" % (time.time() - tick))

    qa_pkl = os.path.join(dataset_dir, 'step1.pkl')
    tick = time.time()
    if not os.path.exists(qa_pkl):
        qa_file = os.path.join(dataset_dir, 'step1.json')
        qa_data = []
        with open(qa_file) as f:
            for line in f:
                qa_data.append(json.loads(line))
        num_data = len(qa_data)

        with open(qa_pkl, 'wb') as f:
            pickle.dump(qa_data, f)
    else:
        with open(qa_pkl, 'rb') as f:
            qa_data = pickle.load(f)
            num_data = len(qa_data)
    print("%.2fs for loading qa data" % (time.time() - tick))

    batch_size = num_data // num_process
    start = pid * batch_size
    end = num_data if pid == num_process - 1 else (pid+1) * batch_size
    output_file = os.path.join(dataset_dir, 'qa.raw.%d.json' % pid)
    retrieve_subgraph(cvt_nodes, triples, ent2idx, idx2ent, idx2rel, stop_ent, qa_data, start, end, output_file)
