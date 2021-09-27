import random
import sys
import pickle
import os
from collections import defaultdict
import json
import numpy as np


def get_dict(path):
    word2idx, idx2word = dict(), dict()
    with open(path, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            idx = len(word2idx)
            word2idx[word] = idx
            idx2word[idx] = word
    return word2idx, idx2word


def weighted_sample(weights):
    assert sum(weights) == 1
    weights = np.add.accumulate(weights)
    gen = np.random.sample()
    for i, each in enumerate(weights):
        if gen < each:
            return i


class SubgraphSampler:
    """
    Num of paths:
        10%:    1
        20%:    2
        30%:    3
        40%:    4
    Pattern:
        13%:    ent1 -> answer
        12%:    answer -> ent1
        25%:    ent1 -> ent2 -> answer
        25%:    ent1 -> answer -> ent2
        25%:    answer -> ent1 -> ent2
    """
    num_path_weights = [0.1, 0.2, 0.3, 0.4]
    pattern_weights = [0.13, 0.12, 0.25, 0.25, 0.25]

    def __init__(self, dataset, mid2name, qg_train, sample_size=20):
        self.dataset = dataset
        self.mid2name = pickle.load(open(mid2name, 'rb'))
        self.rel_pool = self.load_rel_pool(qg_train)
        self.ent2idx, self.idx2ent = get_dict(os.path.join(dataset, 'entities.txt'))
        self.rel2idx, self.idx2rel = get_dict(os.path.join(dataset, 'relations.txt'))
        self.sample_size = sample_size

        self.id_counter = 0

    def load_rel_pool(self, path):
        rel_pool = set()
        with open(path) as f:
            for line in f:
                line = json.loads(line)
                for rel in line['inGraph']['g_edge_types']:
                    rel_pool.add(rel[1:].replace('/', '.'))
        return list(rel_pool)

    def run(self):
        for split in ['train', 'dev', 'test']:
            src_path = os.path.join(self.dataset, split+'_simple.json')
            out_path = os.path.join(self.dataset, split+'_gen.json')
            self.generate(src_path, out_path)

    def generate(self, src_path, out_path):
        data = []
        with open(src_path) as f:
            for line in f:
                line = json.loads(line)

                graph, rev_graph = defaultdict(set), defaultdict(set)
                ent_list = set()
                for head, rel, tail in line['subgraph']['tuples']:
                    if self.idx2rel[rel] not in self.rel_pool:
                        continue
                    ent_list.update([head, tail])
                    graph[head].add((head, rel, tail))
                    rev_graph[tail].add((head, rel, tail))

                if len(ent_list) == 0:
                    print('empty subgraph:', line['id'])
                else:
                    data.extend(self.sample_subgraph(line['id'], list(ent_list), graph, rev_graph))

        with open(out_path, 'w', encoding='utf-8') as f:
            for each in data:
                f.write(json.dumps(each) + '\n')

    def sample_subgraph(self, src_id, ent_list, graph, rev_graph):
        samples = []
        for _ in range(self.sample_size):
            ans = random.choice(ent_list)
            ans_id = self.idx2ent[ans]
            if self.mid2name[ans_id] == 'none':
                continue

            num_paths = weighted_sample(self.num_path_weights) + 1  # offset

            paths, i = set(), 0
            while i < num_paths:
                pattern = weighted_sample(self.pattern_weights)
                if pattern == 0:
                    if ans not in rev_graph:
                        continue
                    paths.add(random.choice(list(rev_graph[ans])))
                elif pattern == 1:
                    if ans not in graph:
                        continue
                    paths.add(random.choice(list(graph[ans])))
                elif pattern == 2:
                    if ans not in rev_graph:
                        continue
                    path1 = random.choice(list(rev_graph[ans]))
                    tmp = path1[0]
                    if tmp not in rev_graph:
                        continue
                    path2 = random.choice(list(rev_graph[tmp]))
                    paths.update([path1, path2])
                elif pattern == 3:
                    if ans not in rev_graph or ans not in graph:
                        continue
                    path1 = random.choice(list(rev_graph[ans]))
                    path2 = random.choice(list(graph[ans]))
                    paths.update([path1, path2])
                elif pattern == 4:
                    if ans not in graph:
                        continue
                    path1 = random.choice(list(graph[ans]))
                    tmp = path1[2]
                    if tmp not in graph:
                        continue
                    path2 = random.choice(list(graph[tmp]))
                    paths.update([path1, path2])
                i += 1

            nodes, edges, adj = set(), set(), {}
            for head, rel, tail in paths:
                head, rel, tail = self.idx2ent[head], self.idx2rel[rel], self.idx2ent[tail]
                rel = '/' + rel.replace('.', '/')
                # only support single relation, overwrite relations for same head and tail entity pairs.
                # assert head not in adj or tail not in adj[head], str(adj) + '\n' + '\n'.join([head, rel, tail])
                adj[head] = {tail: rel}
                nodes.update([head, tail])
                edges.add(rel)

            g_node_names, g_edge_types = {}, {}
            for node in nodes:
                g_node_names[node] = self.mid2name[node]
            for edge in edges:
                g_edge_types[edge] = edge
            samples.append({
                'answers': [self.mid2name[ans_id]],
                'answer_ids': [ans_id],
                'qId': self.id_counter,
                'inGraph': {
                    'g_node_names': g_node_names,
                    'g_edge_types': g_edge_types,
                    'g_adj': adj
                },
                'outSeq': '',
                'src_id': src_id
            })
            self.id_counter += 1

        return samples


if __name__ == '__main__':
    # min_g, max_g, avg_g = 1e30, 0, 0
    # num_data = 0
    # for split in ['train.json', 'dev.json', 'test.json']:
    #     with open(os.path.join('mywq', split)) as f:
    #         for line in f:
    #             line = json.loads(line)
    #             g_size = len(line['inGraph']['g_adj'])
    #             if g_size < min_g:
    #                 min_g = g_size
    #             if g_size > max_g:
    #                 max_g = g_size
    #             avg_g += g_size
    #             num_data += 1
    # print(min_g, max_g, avg_g/num_data)  # 1 / 70 / 3

    random.seed(1020)
    np.random.seed(1020)

    dataset = sys.argv[1]
    mid2name = sys.argv[2]  # mid2name.dict
    assert dataset == 'CWQ' or dataset == 'webqsp'

    sampler = SubgraphSampler(dataset, mid2name, qg_train='mywq/train.json', sample_size=10)
    # sampler.generate(os.path.join(dataset, 'case_study.json'), 'to_gen.json')
    sampler.run()
