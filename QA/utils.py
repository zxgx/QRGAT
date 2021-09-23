import json
import numpy as np
import random
from nltk.tokenize import RegexpTokenizer
import torch


def get_dict(path):
    word2idx, idx2word = dict(), dict()
    with open(path, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            idx = len(word2idx)
            word2idx[word] = idx
            idx2word[idx] = word
    return word2idx, idx2word


class Tokenizer:
    def __init__(self, vocab_path):
        self.word2idx, self.idx2word, self.num_token = self.preprocess(vocab_path)

    def preprocess(self, path):
        word2idx, idx2word, count = {}, {}, 0

        word2idx['#pad#'] = count
        idx2word[count] = '#pad#'
        count += 1
        word2idx['<unk>'] = count
        idx2word[count] = '<unk>'
        count += 1

        with open(path, encoding='utf-8') as f:
            for each in f:
                word = each.strip()
                if word in word2idx:
                    raise ValueError(word+' have already in vocab dict!')
                word2idx[word] = count
                idx2word[count] = word
                count += 1
        return word2idx, idx2word, count

    def __call__(self, text):
        tok_ids = []
        for token in text.split():
            tok_ids.append(self.word2idx.get(token, self.word2idx['<unk>']))
        return tok_ids

    def decode(self, text):
        return ' '.join([self.idx2word[t] for t in text])

    @staticmethod
    def load_glove_emb(path):
        pad = np.zeros(300)
        unk = np.random.uniform(-1., 1., 300)
        word_emb = np.load(path)
        return np.vstack([pad, unk, word_emb])


class QADataset:
    def __init__(self, data_path, ent2idx, rel2idx, tokenizer, batch_size, training, device,
                 fact_dropout=0., token_path=None):
        self.ent2idx = ent2idx
        self.rel2idx = rel2idx
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.training = training
        self.device = device
        self.fact_dropout = fact_dropout

        if token_path is None:
            self.reg_tokenizer = RegexpTokenizer(r'\d{1}|\w+|[^\w\s]+')

        self.max_seq_len = 0
        data = self.load_data(data_path, token_path)
        self.num_data = len(data)

        self.max_local_entity = 0
        self.global2local_maps = self.build_global2local_maps(data)

        self.data_id = np.empty(self.num_data, dtype=object)
        self.question = np.zeros((self.num_data, self.max_seq_len), dtype=int)
        self.question_mask = np.zeros((self.num_data, self.max_seq_len), dtype=float)
        self.topic_label = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.entity_mask = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.subgraph = np.empty(self.num_data, dtype=object)
        self.answer_label = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.answer_list = np.empty(self.num_data, dtype=object)

        self.buffer_data(data)

    def load_data(self, data_path, token_path):
        idx2question = {}
        if token_path is not None:
            with open(token_path, encoding='utf-8') as f:
                for each in f:
                    each = json.loads(each)
                    processed_question = []
                    for tok in each['dep']:
                        processed_question.append(tok[0])
                    idx2question[each['id']] = ' '.join(processed_question)

        omitted, data = [], []
        with open(data_path, encoding='utf-8') as f:
            for each in f:
                each = json.loads(each)

                if len(each['entities']) == 0:
                    omitted.append(each['id'])
                    continue

                if token_path is not None:
                    if each['id'] not in idx2question:
                        raise ValueError(each['id'] + 'don\'t have tokenized question!')
                    each['question'] = self.tokenizer(idx2question[each['id']])
                else:
                    each['question'] = self.tokenizer(' '.join(self.reg_tokenizer.tokenize(each['question'].lower())))
                self.max_seq_len = max(self.max_seq_len, len(each['question']))

                data.append({
                    'id': each['id'],
                    'question': each['question'],
                    'topic entities': each['entities'],  # topic entity id
                    'subgraph': each['subgraph']['tuples'],
                    'candidates': each['subgraph']['entities'],  # subgraph entity id
                    'answers': each['answers']
                })
        print('Read %d data from %s' % (len(data), data_path))
        print('Omit %d questions without any topic entity:\n%s' % (len(omitted), str(omitted)))
        return data

    def build_global2local_maps(self, data):
        global2local, total_local_entity = {}, 0.
        for each in data:
            g2l = dict()
            self._add_entity_to_map(each['topic entities'], g2l)
            self._add_entity_to_map(each['candidates'], g2l)
            assert each['id'] not in global2local, 'Duplicate data id: ' + each['id']
            global2local[each['id']] = g2l
            total_local_entity += len(g2l)
            self.max_local_entity = max(self.max_local_entity, len(g2l))
        print('Average local entity: %.2f' % (total_local_entity / len(data)))
        print('Max local entity: %d' % self.max_local_entity)
        return global2local

    @staticmethod
    def _add_entity_to_map(entities, g2l):
        for each in entities:
            if each not in g2l:
                g2l[each] = len(g2l)

    def buffer_data(self, data):
        answerable = 0
        for i, each in enumerate(data):
            self.data_id[i] = each['id']
            g2l = self.global2local_maps[each['id']]
            assert len(g2l) > 0

            for j, word in enumerate(each['question']):
                self.question[i, j] = word
                self.question_mask[i, j] = 1

            topic_ent = set([g2l[x] for x in each['topic entities']])
            for ent in topic_ent:
                self.topic_label[i, ent] = 1

            for g, l in g2l.items():
                self.entity_mask[i, l] = 1

            head_list, rel_list, tail_list = [], [], []
            for head, rel, tail in each['subgraph']:
                head_list.append(g2l[head])
                tail_list.append(g2l[tail])
                rel_list.append(rel)
            self.subgraph[i] = (
                np.array(head_list, dtype=int),
                np.array(rel_list, dtype=int),
                np.array(tail_list, dtype=int)
            )

            answer_set, local_answer_set = set(), set()
            for answer in each['answers']:
                answer = answer['text'] if type(answer['kb_id']) == int else answer['kb_id']
                answer_id = self.ent2idx[answer]
                answer_set.add(answer_id)
                if answer_id in g2l:
                    local_answer_set.add(g2l[answer_id])
            self.answer_list[i] = list(answer_set)

            for answer in local_answer_set:
                self.answer_label[i, answer] = 1.
            if len(local_answer_set) > 0:
                answerable += 1

        print("There are %d / %d answerable questions" % (answerable, len(data)))

    def __getitem__(self, item):
        return (
            self.data_id[item], self.question[item], self.question_mask[item], self.topic_label[item],
            self.entity_mask[item], self.subgraph[item], self.answer_label[item], self.answer_list[item]
        )

    def __len__(self):
        return self.num_data

    def batching(self):
        indices = list(range(self.num_data))
        if self.training:
            random.shuffle(indices)

        for start_index in range(0, self.num_data, self.batch_size):
            batch_indices = indices[start_index: start_index+self.batch_size]

            data_id = self.data_id[batch_indices]
            question = torch.tensor(self.question[batch_indices], dtype=torch.long).to(self.device)
            question_mask = torch.tensor(self.question_mask[batch_indices], dtype=torch.float).to(self.device)
            topic_label = torch.tensor(self.topic_label[batch_indices], dtype=torch.float).to(self.device)
            entity_mask = torch.tensor(self.entity_mask[batch_indices], dtype=torch.float).to(self.device)
            subgraph = self.build_subgraph(data_id, batch_indices)
            answer_label = torch.tensor(self.answer_label[batch_indices], dtype=torch.float).to(self.device)
            answer_list = self.answer_list[batch_indices]

            yield data_id, question, question_mask, topic_label, entity_mask, subgraph, answer_label, answer_list

    def build_subgraph(self, data_id, batch_indices):
        batch_heads, batch_tails, batch_ids = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
        batch_relations = np.array([], dtype=int)
        subgraph = self.subgraph[batch_indices]

        for i, (head_list, rel_list, tail_list) in enumerate(subgraph):
            g2l = self.global2local_maps[data_id[i]]
            offset = i * self.max_local_entity

            fact_size = len(head_list)
            if self.training:
                fact_size_in_use = np.ceil(fact_size * (1-self.fact_dropout)).astype(int)
                mask_index = np.random.permutation(fact_size)[:fact_size_in_use]
            else:
                fact_size_in_use = fact_size
                mask_index = np.arange(fact_size)

            batch_heads = np.append(batch_heads, head_list[mask_index]+offset)
            batch_relations = np.append(batch_relations, rel_list[mask_index])
            batch_tails = np.append(batch_tails, tail_list[mask_index]+offset)
            batch_ids = np.append(batch_ids, np.full(fact_size_in_use, i, dtype=int))

        batch_ids = torch.from_numpy(batch_ids).long().to(self.device)
        batch_relations = torch.from_numpy(batch_relations).long().to(self.device)

        fact_ids = np.array(range(len(batch_heads)), dtype=int)
        values = torch.ones(batch_ids.shape[0], dtype=torch.float)
        batch_size, fact_size = len(batch_indices), fact_ids.shape[0]

        head2edge_idx = torch.tensor([fact_ids, batch_heads], dtype=torch.long)
        head2edge = torch.sparse.FloatTensor(head2edge_idx, values,
                                             (fact_size, batch_size * self.max_local_entity)).to(self.device)

        tail2edge_idx = torch.tensor([fact_ids, batch_tails], dtype=torch.long)
        tail2edge = torch.sparse.FloatTensor(tail2edge_idx, values,
                                             (fact_size, batch_size * self.max_local_entity)).to(self.device)

        return batch_ids, batch_relations, head2edge, tail2edge
