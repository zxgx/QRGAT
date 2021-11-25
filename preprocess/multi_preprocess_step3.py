from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
import pickle
import copy
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


def get_subgraph(cand_ents, triples, idx2ent, idx2rel, cvt_nodes, cvt_add_flag=False):
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
    vals = np.ones((len(rows),))
    rows = np.array(rows)
    cols = np.array(cols)
    sp_mat = csr_matrix((vals, (rows, cols)), shape=(len(ent2id), len(ent2id)))

    return ent2id, rel2id, normalize(sp_mat, norm="l1", axis=1)


def get_2hop_triples(triples, idx2ent, seed_set, cvt_nodes):
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


def retrieve_subgraph(cvt_name, graph_name, ent2idx_name, idx2ent_name, idx2rel_name, qa_data_name, start_idx, end_idx,
                      pid, output_file, max_ent=2000):
    st = time.time()
    s_cvt_nodes = SharedMemory(cvt_name)
    cvt_nodes = pickle.loads(s_cvt_nodes.buf.tobytes())
    print("PID: %d | load CVT nodes: %.2fs" % (pid, time.time() - st))

    tick = time.time()
    s_triples = SharedMemory(graph_name)
    triples = pickle.loads(s_triples.buf.tobytes())
    s_ent2idx = SharedMemory(ent2idx_name)
    ent2idx = pickle.loads(s_ent2idx.buf.tobytes())
    s_idx2ent = SharedMemory(idx2ent_name)
    idx2ent = pickle.loads(s_idx2ent.buf.tobytes())
    s_idx2rel = SharedMemory(idx2rel_name)
    idx2rel = pickle.loads(s_idx2rel.buf.tobytes())
    print("PID: %d | load indexed KB: %.2fs" % (pid, time.time() - tick))

    tick = time.time()
    s_qa_data = SharedMemory(qa_data_name)
    qa_data = pickle.loads(s_qa_data.buf.tobytes())
    print("PID: %d | load QA data: %.2fs" % (pid, time.time() - tick))

    f_out = open(output_file, 'w')
    answer_coverage = []
    for idx in range(start_idx, end_idx):
        data = copy.deepcopy(qa_data[idx])
        print("PID: %d | processing %s" % (pid, data['id']))

        entity_set = set()
        for entity in data['entities']:
            if entity in ent2idx:
                entity_set.add(ent2idx[entity])
        tick = time.time()
        triple_set = get_2hop_triples(triples, idx2ent, entity_set, cvt_nodes)
        print("PID: %d | get 2 hop subgraph: %.2fs" % (pid, time.time() - tick))

        if len(entity_set) == 0 or len(triple_set) == 0:
            extracted_tuples, extracted_entities = [], []
            print("PID: %d | Bad question: %s" % (pid, data['id']))
        else:
            tick = time.time()
            ent2id, rel2id, sp_mat = build_sp_mat(triple_set)
            print("PID: %d | build sparse matrix: %.2fs, num ent: %d, num relation: %d, num edge: %d" %
                  (pid, time.time() - tick, len(ent2id), len(rel2id), len(triple_set)))

            seed_list = []
            for ent in entity_set:
                if ent in ent2id:
                    seed_list.append(ent2id[ent])
            tick = time.time()
            extracted_ents = rank_ppr_ents(seed_list, sp_mat, max_ent=max_ent)
            print("PID: %d | PPR: %.2fs" % (pid, time.time() - tick))
            id2ent = {v: k for k, v in ent2id.items()}
            ent_set_new = set(id2ent[ent] for ent in extracted_ents.tolist()) | entity_set
            tick = time.time()
            extracted_entities, extracted_tuples = get_subgraph(ent_set_new, triples, idx2ent, idx2rel, cvt_nodes)
            print("PID: %d | extract result subgrpah: %.2fs" % (pid, time.time() - tick))

        data['subgraph'] = {}
        data['subgraph']['tuples'] = extracted_tuples
        data['subgraph']['entities'] = extracted_entities
        f_out.write(json.dumps(data) + "\n")

        answer_coverage.append(_get_answer_coverage(data["answers"], extracted_entities))

    f_out.close()
    s_cvt_nodes.close()
    s_triples.close()
    s_ent2idx.close()
    s_idx2ent.close()
    s_idx2rel.close()
    s_qa_data.close()
    print("PID: %d | Finish: %.2fs" % (pid, time.time() - st))
    return answer_coverage


if __name__ == '__main__':
    cvt_file = 'data/cvtnodes.bin'
    cvt_pkl = 'data/cvt.pkl'

    dataset_dir = sys.argv[1]
    num_process = int(sys.argv[2])

    graph_pkl = os.path.join(dataset_dir, 'subgraph.pkl')
    ent2idx_pkl = os.path.join(dataset_dir, 'ent2idx.pkl')
    idx2ent_pkl = os.path.join(dataset_dir, 'idx2ent.pkl')
    idx2rel_pkl = os.path.join(dataset_dir, 'idx2rel.pkl')
    qa_pkl = os.path.join(dataset_dir, 'step1.pkl')

    tick = time.time()
    if not os.path.exists(cvt_pkl):
        cvt_nodes = load_cvt(cvt_file)
        with open(cvt_pkl, 'wb') as f:
            pickle.dump(cvt_nodes, f)
        b_cvt_nodes = pickle.dumps(cvt_nodes)
        del cvt_nodes
    else:
        with open(cvt_pkl, 'rb') as f:
            b_cvt_nodes = pickle.dumps(pickle.load(f))
    b_cvt_nodes = memoryview(b_cvt_nodes)
    print("%.2fs for loading cvt nodes in bytes" % (time.time() - tick))

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
        b_triples = pickle.dumps(triples)
        del triples
    else:
        with open(graph_pkl, 'rb') as f:
            b_triples = pickle.dumps(pickle.load(f))
    b_triples = memoryview(b_triples)
    print("%.2fs for loading indexed graph in bytes" % (time.time() - tick))

    tick = time.time()
    if not (os.path.exists(ent2idx_pkl) and os.path.exists(idx2ent_pkl)):
        ent_file = os.path.join(dataset_dir, 'ent.txt')
        ent2idx, idx2ent = load_dict(ent_file)

        with open(ent2idx_pkl, 'wb') as f:
            pickle.dump(ent2idx, f)
        b_ent2idx = pickle.dumps(ent2idx)
        del ent2idx

        with open(idx2ent_pkl, 'wb') as f:
            pickle.dump(idx2ent, f)
        b_idx2ent = pickle.dumps(idx2ent)
        del idx2ent
    else:
        with open(ent2idx_pkl, 'rb') as f:
            b_ent2idx = pickle.dumps(pickle.load(f))
        with open(idx2ent_pkl, 'rb') as f:
            b_idx2ent = pickle.dumps(pickle.load(f))
    b_ent2idx = memoryview(b_ent2idx)
    b_idx2ent = memoryview(b_idx2ent)
    print("%.2fs for loading entity dict in bytes" % (time.time() - tick))

    tick = time.time()
    if not os.path.exists(idx2rel_pkl):
        rel_file = os.path.join(dataset_dir, 'rel.txt')
        _, idx2rel = load_dict(rel_file)

        with open(idx2rel_pkl, 'wb') as f:
            pickle.dump(idx2rel, f)
        b_idx2rel = pickle.dumps(idx2rel)
        del idx2rel
    else:
        with open(idx2rel_pkl, 'rb') as f:
            b_idx2rel = pickle.dumps(pickle.load(f))
    b_idx2rel = memoryview(b_idx2rel)
    print("%.2fs for loading relation dict in bytes" % (time.time() - tick))

    tick = time.time()
    if not os.path.exists(qa_pkl):
        qa_file = os.path.join(dataset_dir, 'step1.json')
        qa_data = []
        with open(qa_file) as f:
            for line in f:
                qa_data.append(json.loads(line))
        b_qa_data = pickle.dumps(qa_data)
        num_data = len(qa_data)
        with open(qa_pkl, 'wb') as f:
            pickle.dump(qa_data, f)
        del qa_data
    else:
        with open(qa_pkl, 'rb') as f:
            b_qa_data = pickle.load(f)
            num_data = len(b_qa_data)
            b_qa_data = pickle.dumps(b_qa_data)
    b_qa_data = memoryview(b_qa_data)
    print("%.2fs for loading qa data in bytes" % (time.time() - tick))

    with SharedMemoryManager() as smm:
        shared_cvt_nodes = smm.SharedMemory(b_cvt_nodes.nbytes)
        shared_cvt_nodes.buf[:] = b_cvt_nodes

        shared_triples = smm.SharedMemory(b_triples.nbytes)
        shared_triples.buf[:] = b_triples

        shared_ent2idx = smm.SharedMemory(b_ent2idx.nbytes)
        shared_ent2idx.buf[:] = b_ent2idx

        shared_idx2ent = smm.SharedMemory(b_idx2ent.nbytes)
        shared_idx2ent.buf[:] = b_idx2ent

        shared_idx2rel = smm.SharedMemory(b_idx2rel.nbytes)
        shared_idx2rel.buf[:] = b_idx2rel

        shared_qa_data = smm.SharedMemory(b_qa_data.nbytes)
        shared_qa_data.buf[:] = b_qa_data

        pool = Pool(num_process)
        stride = num_data // num_process
        results = []
        for i in range(num_process):
            start = i * stride
            end = num_data if i == (num_process-1) else (i+1) * stride
            output_file = os.path.join(dataset_dir, 'qa.raw.%d.json' % i)
            result = pool.apply_async(
                retrieve_subgraph, args=(
                    shared_cvt_nodes.name, shared_triples.name, shared_ent2idx.name, shared_idx2ent.name,
                    shared_idx2rel.name, shared_qa_data.name, start, end, i, output_file
                )
            )
            results.append(result)
        pool.close()
        pool.join()

    coverage = []
    for res in results:
        coverage.extend(res.get())
    print("Answer coverage in %d data = %.3f" % (num_data, np.mean(coverage).item()))
