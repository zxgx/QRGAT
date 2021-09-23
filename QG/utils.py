import json
import random
from collections import OrderedDict, defaultdict, Counter
import re
import string
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import os
import pickle
from scipy.sparse import lil_matrix
import torch

from evaluation.eval import QGEvalCap


re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[%s]' % re.escape(string.punctuation))
_PAD_TOKEN = '[PAD]'
_UNK_TOKEN = '[UNK]'
_SOS_TOKEN = '[SOS]'
_EOS_TOKEN = '[EOS]'


def normalize_answer(s):
    """Lower text and remove extra whitespace."""
    def remove_articles(text):
        return re_art.sub(' ', text)

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def load_data(path, levi_graph):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = json.loads(line, object_pairs_hook=OrderedDict)

            answers = line['answers']  # [name], maybe EMPTY
            normalized_answers = {normalize_answer(x) for x in answers}
            answers = [wordpunct_tokenize(x.lower()) for x in answers]
            answer_ids = set(line['answer_ids'])  # mid

            question = wordpunct_tokenize(line['outSeq'].lower())
            question.append(_EOS_TOKEN)

            graph = {
                'node_ids': {}, 'node_name_words': [], 'node_type_words': [], 'node_type_ids': [],
                'answer_indicator': [], 'edge_name_words': [], 'edge_ids': [], 'adj_mat': defaultdict(dict)
            }

            for idx, node_id in enumerate(line['inGraph']['g_node_names']):
                graph['node_ids'][node_id] = idx
                node_name = line['inGraph']['g_node_names'][node_id]
                graph['node_name_words'].append(wordpunct_tokenize(node_name.lower()))

                if 'g_node_types' in line['inGraph']:
                    graph['node_type_ids'].append(line['inGraph']['g_node_types'][node_id])
                    node_type = ' '.join(line['inGraph']['g_node_types'][node_id].split('/')[-1].split('_'))
                    graph['node_type_words'].append(wordpunct_tokenize(node_type.lower()))

                if len(answer_ids) > 0:
                    graph['answer_indicator'].append(1 if node_id in answer_ids else 2)
                else:
                    graph['answer_indicator'].append(
                        1 if normalize_answer(line['inGraph']['g_node_names'][node_id]) in normalized_answers else 2
                    )

            # Levi graph
            if levi_graph:
                num_nodes = len(graph['node_ids'])
                edge_index = num_nodes
                virtual_edge_idx = 0
                for node_id, val in line['inGraph']['g_adj'].items():
                    idx1 = graph['node_ids'][node_id]
                    for node_id2, edge_id in val.items():
                        idx2 = graph['node_ids'][node_id2]
                        assert isinstance(edge_id, str)

                        graph['adj_mat'][idx1][edge_index] = virtual_edge_idx
                        virtual_edge_idx += 1
                        graph['adj_mat'][edge_index][idx2] = virtual_edge_idx
                        virtual_edge_idx += 1

                        graph['edge_ids'].append(line['inGraph']['g_edge_types'][edge_id])
                        edge_name = ' '.join(line['inGraph']['g_edge_types'][edge_id].split('/')[-1].split('_'))
                        graph['edge_name_words'].append(wordpunct_tokenize(edge_name.lower()))
                        edge_index += 1

                assert len(graph['edge_name_words']) == edge_index - num_nodes
                graph['num_virtual_nodes'] = edge_index
                graph['num_virtual_edges'] = virtual_edge_idx
            else:
                edge_index = 0
                for node_id, val in line['inGraph']['g_adj'].items():
                    idx1 = graph['node_ids'][node_id]
                    for node_id2, edge_id in val.items():
                        idx2 = graph['node_ids'][node_id2]
                        assert isinstance(edge_id, str)
                        graph['adj_mat'][idx1][idx2] = edge_index

                        graph['edge_ids'].append(line['inGraph']['g_edge_types'][edge_id])
                        edge_name = ' '.join(line['inGraph']['g_edge_types'][edge_id].split('/')[-1].split('_'))
                        graph['edge_name_words'].append(wordpunct_tokenize(edge_name.lower()))
                        edge_index += 1
                assert len(graph['edge_name_words']) == edge_index

            data.append((graph, question, answers))

    lens = [len(x[1]) for x in data]
    print("Read %d samples from %s, max seq len: %d, min seq len: %d, mean seq len: %.2f" %
          (len(data), path, np.max(lens), np.min(lens), np.mean(lens)))
    return data


class Vocab:
    def __init__(self, dataset, max_vocab_size=None, min_freq=1):
        self.PAD = 0
        self.SOS = 1
        self.EOS = 2
        self.UNK = 3
        self.pad_token = _PAD_TOKEN
        self.sos_token = _SOS_TOKEN
        self.eos_token = _EOS_TOKEN
        self.unk_token = _UNK_TOKEN

        self.index2word = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        self.word2index = dict(zip(self.index2word, range(len(self.index2word))))

        word2count = Vocab.count_words(dataset)
        self._add_words(word2count, max_vocab_size, min_freq)

    @classmethod
    def build(cls, saved_vocab_file, dataset=None, max_vocab_size=None, min_freq=1):
        if os.path.exists(saved_vocab_file):
            print('Loading pre-built vocab model stored in {}'.format(saved_vocab_file))
            vocab = pickle.load(open(saved_vocab_file, 'rb'))
        else:
            vocab = Vocab(dataset, max_vocab_size, min_freq)
            print('Saving vocab model to {}'.format(saved_vocab_file))
            pickle.dump(vocab, open(saved_vocab_file, 'wb'))
        return vocab

    @staticmethod
    def count_words(dataset):
        words = Counter()
        for graph, question, answers in dataset:
            for node_words in graph['node_name_words']:
                words.update(node_words)
            for node_type_words in graph['node_type_words']:
                words.update(node_type_words)
            for edge_name_words in graph['edge_name_words']:
                words.update(edge_name_words)

            words.update(question)

            for answer in answers:
                words.update(answer)
        return words

    def _add_words(self, word2count, max_vocab_size, min_freq):
        ordered = sorted(word2count.items(), key=lambda x: x[1], reverse=True)
        if max_vocab_size is not None:
            ordered = ordered[:max_vocab_size]

        for i, (word, count) in enumerate(ordered):
            if count < min_freq:
                break
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        assert len(self.word2index) == len(self.index2word)
        print('There are %d tokens in training data, and %d tokens are reserved in vocab' %
              (len(word2count), len(self.word2index)))

    def get_index(self, word):
        return self.word2index.get(word, self.UNK)

    def get_word(self, idx):
        return self.index2word[idx] if idx < len(self) else '[OOV: %d]' % idx

    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)


class OOVDict:
    def __init__(self, offset):
        self.word2index = {}
        self.index2word = {}
        self.next_index = {}
        self.base_oov_idx = offset
        self.ext_vocab_size = offset

    def add_word(self, idx_in_batch, word):
        key = (idx_in_batch, word)
        index = self.word2index.get(key, None)
        if index is not None:
            return index

        index = self.next_index.get(idx_in_batch, self.base_oov_idx)
        self.next_index[idx_in_batch] = index + 1
        self.word2index[key] = index
        self.index2word[(idx_in_batch, index)] = word
        self.ext_vocab_size = max(self.ext_vocab_size, index+1)
        return index


class Batch:
    def __init__(self, samples, vocab, device):
        self.samples = samples
        self.vocab = vocab
        self.device = device

        self.oov_dict = OOVDict(len(vocab))

        self.graphs = self.batch_graphs([x[0] for x in samples])

        questions, question_lens, org_questions = [], [], []
        for i, (_, question, _) in enumerate(samples):
            org_questions.append(' '.join(question[:-1]))  # strip EOS
            question = self.question2ids(i, question)
            questions.append(question)
            question_lens.append(len(question))

        self.org_questions = org_questions  # [ str ],
        self.questions = self.pad_2d(questions, fill=vocab.PAD)  # batch size, max seq len
        self.question_lens = torch.tensor(question_lens).long()  # batch size

    def batch_graphs(self, graphs):
        max_node_size = max([g.get('num_virtual_nodes', len(g['node_name_words'])) for g in graphs])
        max_edge_size = max([g.get('num_virtual_edges', len(g['edge_name_words'])) for g in graphs])

        num_nodes, num_edges = [], []
        node_name_words, node_name_lens = [], []
        node_type_words, node_type_lens = [], []
        edge_name_words, edge_name_lens = [], []
        answer_indicator = []
        batch_oov_idx = []

        batch_node2edge, batch_edge2node = [], []

        for batch_idx, graph in enumerate(graphs):
            node_name_idx, oov_idx = [], []
            # node name
            for each in graph['node_name_words']:
                oov_idx.append(self.oov_dict.add_word(batch_idx, tuple(each)))

                current_node_name = [self.vocab.get_index(w) for w in each]
                assert len(current_node_name) > 0
                node_name_idx.append(current_node_name)
            node_name_words.append(node_name_idx)
            node_name_lens.append([max(len(x), 1) for x in node_name_idx])
            answer_indicator.append(graph['answer_indicator'])

            # node type
            node_type_idx = []
            for each in graph['node_type_words']:
                current_node_type = [self.vocab.get_index(w) for w in each]
                assert len(current_node_type) > 0
                node_type_idx.append(current_node_type)
            node_type_words.append(node_type_idx)
            node_type_lens.append(max(len(x), 1) for x in node_type_idx)

            # edge name
            edge_name_idx = []
            for each in graph['edge_name_words']:
                # oov_idx.append(self.oov_dict.add_word(batch_idx, tuple(each)))

                current_edge_name = [self.vocab.get_index(w) for w in each]
                assert len(current_edge_name) > 0
                edge_name_idx.append(current_edge_name)
            edge_name_words.append(edge_name_idx)
            edge_name_lens.append([max(len(x), 1) for x in edge_name_idx])

            num_nodes.append(len(node_name_idx))
            num_edges.append(len(edge_name_idx))

            batch_oov_idx.append(oov_idx)

            # adjacent matrix
            node2edge = lil_matrix(np.zeros((max_edge_size, max_node_size)), dtype=np.float32)
            edge2node = lil_matrix(np.zeros((max_node_size, max_edge_size)), dtype=np.float32)
            for head, val in graph['adj_mat'].items():
                for tail, edge in val.items():
                    if head == tail:  # check
                        continue
                    node2edge[edge, head] = 1
                    edge2node[tail, edge] = 1
            batch_node2edge.append(node2edge)
            batch_edge2node.append(edge2node)

        ret = {
            'max_node_size': max_node_size,  # scalar
            'num_nodes': np.array(num_nodes, dtype=np.int32),  # batch size
            'num_edges': np.array(num_edges, dtype=np.int32),  # batch size
            'node_name_words': self.pad_3d(node_name_words, fill=self.vocab.PAD),  # batch size, node size, max node len
            'node_name_lens': self.pad_2d(node_name_lens, fill=1),   # batch size, node size, > 0
            # 'node_type_words': self.pad_3d(node_type_words, fill=self.vocab.PAD),  # batch size, type size, max type len
            # 'node_type_lens': self.pad_2d(node_type_lens, fill=0),   # batch size, type size
            'edge_name_words': self.pad_3d(edge_name_words, fill=self.vocab.PAD),  # batch size, edge size, max edge len
            'edge_name_lens': self.pad_2d(edge_name_lens, fill=1),   # batch size, edge size > 0
            # batch size, virtual edge size, virtual node size
            'node2edge': torch.stack([torch.tensor(x.A) for x in batch_node2edge], dim=0).to(self.device),
            # batch size, virtual node size, virtual edge size
            'edge2node': torch.stack([torch.tensor(x.A) for x in batch_edge2node], dim=0).to(self.device),
            'answer_indicator': self.pad_2d(answer_indicator, fill=0),  # batch size, node size
            'oov_idx': self.pad_2d(batch_oov_idx, fill=self.vocab.PAD)  # batch size, node size
        }

        return ret

    def pad_2d(self, mat, fill=0):
        dim1 = len(mat)
        dim2 = max([len(x) for x in mat])

        tensor = np.ones((dim1, dim2), dtype=np.int32) * fill
        for i in range(dim1):
            tensor[i, :len(mat[i])] = mat[i]
        return torch.from_numpy(tensor).long().to(self.device)

    def pad_3d(self, mat, fill=0):
        dim1 = len(mat)
        dim2 = max([len(x) for x in mat])
        dim3 = max([max([len(x) for x in y]) for y in mat])

        tensor = np.ones((dim1, dim2, dim3), dtype=np.int32) * fill
        for i in range(dim1):
            for j in range(len(mat[i])):
                tensor[i, j, :len(mat[i][j])] = mat[i][j]
        return torch.from_numpy(tensor).long().to(self.device)

    def question2ids(self, idx_in_batch, question):
        matched_pos = {}
        for key in self.oov_dict.word2index:
            if key[0] == idx_in_batch:
                indices, word = [], list(key[1])
                for i in range(len(question)):
                    if question[i: i+len(word)] == word:
                        indices.append((i, i+len(word)))
                for pos in indices:
                    matched_pos[pos] = key

        matched_pos = sorted(matched_pos.items(), key=lambda d: d[0][0])

        ret, i = [], 0
        while i < len(question):
            if len(matched_pos) == 0 or i < matched_pos[0][0][0]:
                ret.append(self.vocab.get_index(question[i]))
                i += 1
            else:
                pos, key = matched_pos.pop(0)
                ret.append(self.oov_dict.word2index.get(key))
                i += len(key[1])
        return ret

    def __len__(self):
        return len(self.samples)

    def decode_single(self, predict, batch_idx):
        sample = []
        for tok in predict:
            if tok == self.vocab.SOS:
                continue
            if tok == self.vocab.EOS:
                break

            if tok >= len(self.vocab):
                word = self.oov_dict.index2word.get((batch_idx, tok), self.vocab.unk_token)
                word = ' '.join(word)
            else:
                word = self.vocab.get_word(tok)
            sample.append(word)
        return ' '.join(sample)

    def decode_batch(self, predict):
        decoded = []
        if not isinstance(predict, list):
            predict = predict.transpose(0, 1).tolist()
        for batch_idx, each in enumerate(predict):
            decoded.append(self.decode_single(each, batch_idx))
        return decoded


def evaluate_predictions(target_src, decoded_text):
    assert len(target_src) == len(decoded_text)
    eval_targets = {}
    eval_predictions = {}
    for idx in range(len(target_src)):
        eval_targets[idx] = [target_src[idx]]
        eval_predictions[idx] = [decoded_text[idx]]

    QGEval = QGEvalCap(eval_targets, eval_predictions)
    scores = QGEval.evaluate()
    return scores


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.history = []
        self.last = None
        self.val = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.last = self.mean()
        self.history.append(self.last)
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def mean(self):
        if self.count == 0:
            return 0.
        return self.sum / self.count

    @staticmethod
    def update_metrics(full, batch, batch_size):
        for k in batch:
            full[k].update(batch[k]*100, batch_size)

    @staticmethod
    def reset_metrics(metrics):
        for k in metrics:
            metrics[k].reset()

    @staticmethod
    def format(metrics):
        results = []
        for k in metrics:
            results.append("%s: %.4f" % (k, metrics[k].mean()))
        return ', '.join(results)


class QGDataset:
    def __init__(self, raw, is_sort, batch_size, vocab, device, shuffle):
        self.num_data = len(raw)
        self.num_batch = np.ceil(len(raw)/batch_size).astype(int)

        if is_sort:  # sort by the number of nodes
            raw = sorted(raw, key=lambda x: len(x[0]['node_ids']))

        self.batches = []
        for start_idx in range(0, len(raw), batch_size):
            self.batches.append(
                Batch(raw[start_idx:start_idx+batch_size], vocab, device)
            )
        assert len(self.batches) == self.num_batch

        self.shuffle = shuffle

    def batching(self):
        indices = list(range(self.num_batch))

        if self.shuffle:
            random.shuffle(indices)

        for i in indices:
            yield self.batches[i]
