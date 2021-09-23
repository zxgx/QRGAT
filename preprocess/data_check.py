import re
import string
import json
import os
from nltk.tokenize import wordpunct_tokenize
from thefuzz import process
import pickle
from tqdm import tqdm


re_punc = re.compile(r'[%s]' % re.escape(string.punctuation))


def normalize(ent):
    if ent[0] == '/':
        return ent[1:].replace('/', '.')
    return ent


def load_raw_mhqg(path):
    data = {}
    # ent_set, rel_set = set(), set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            # print(line['qId'])
            question = re_punc.sub(' ', line['outSeq']).lower()
            question = ' '.join(wordpunct_tokenize(question))
            # answers = [normalize(x) for x in line['answer_ids']]
            # entities = [normalize(x) for x in line['inGraph']['g_node_names']]
            # relations = [normalize(x) for x in line['inGraph']['g_edge_types']]
            # ent_set.update(entities + answers)
            # rel_set.update(relations)

            assert line['qId'] not in data
            data[line['qId']] = question
    return data


def load_qa(path):
    train_data = []
    with open(os.path.join(path, 'train_simple.json')) as f:
        for line in f:
            line = json.loads(line)
            question = ' '.join(wordpunct_tokenize(re_punc.sub(' ', line['question']).lower()))
            if question in train_data:
                print("Dup question in train:", question)
            else:
                train_data.append(question)

    dev_data = []
    with open(os.path.join(path, 'dev_simple.json')) as f:
        for line in f:
            line = json.loads(line)
            question = ' '.join(wordpunct_tokenize(re_punc.sub(' ', line['question']).lower()))
            if question in dev_data:
                print("Dup question in dev:", question)
            else:
                dev_data.append(question)

    test_data = []
    with open(os.path.join(path, 'test_simple.json')) as f:
        for line in f:
            line = json.loads(line)
            question = ' '.join(wordpunct_tokenize(re_punc.sub(' ', line['question']).lower()))
            if question in test_data:
                print("Dup question in test:", question)
            else:
                test_data.append(question)
    return train_data, dev_data, test_data


if __name__ == '__main__':
    # Step 1:
    # mhqg_data = load_raw_mhqg('mhqg-wq/raw/data.json')  # id : question
    # print('mhqg data size: %d' % (len(mhqg_data)))
    #
    # cwq_train, cwq_dev, cwq_test = load_qa('CWQ')
    # print('cwq train data size: %d, dev data size: %d, test data size: %d' %
    #       (len(cwq_train), len(cwq_dev), len(cwq_test)))
    # wq_train, wq_dev, wq_test = load_qa('webqsp')
    # print('wq train data size: %d, dev data size: %d, test data size: %d' %
    #       (len(wq_train), len(wq_dev), len(wq_test)))
    #
    # total_train = cwq_train + wq_train
    # total_dev = cwq_dev + wq_dev
    # total_test = cwq_test + wq_test
    #
    # pickle.dump(mhqg_data, open('workspace/qg.pkl', 'wb'))
    # pickle.dump(total_train, open('workspace/qa_train.pkl', 'wb'))
    # pickle.dump(total_dev, open('workspace/qa_dev.pkl', 'wb'))
    # pickle.dump(total_test, open('workspace/qa_test.pkl', 'wb'))

    # mhqg_data = pickle.load(open('workspace/qg.pkl', 'rb'))
    # total_train = pickle.load(open('workspace/qa_train.pkl', 'rb'))
    # total_dev = pickle.load(open('workspace/qa_dev.pkl', 'rb'))
    # total_test = pickle.load(open('workspace/qa_test.pkl', 'rb'))

    # Step 2
    # _train, _dev, _test, miss = [], [], [], []
    # # exact match
    # for k, v in mhqg_data.items():
    #     if v in total_train and v not in total_dev and v not in total_test:
    #         _train.append(k)
    #     elif v in total_dev and v not in total_train and v not in total_test:
    #         _dev.append(k)
    #     elif v in total_test and v not in total_train and v not in total_dev:
    #         _test.append(k)
    #     else:
    #         miss.append(k)
    #
    # # fuzzy match, Warning: this costs around 10 hours
    # ambiguous = []
    # for k in miss:
    #     v = mhqg_data[k]
    #     _, train_ratio = process.extractOne(v, total_train)
    #     _, dev_ratio = process.extractOne(v, total_dev)
    #     _, test_ratio = process.extractOne(v, total_test)
    #     max_ratio = max(train_ratio, dev_ratio, test_ratio)
    #     if max_ratio <= 80:
    #         print("WARNING - ratio: %d, %s: %s" % (max_ratio, k, v))
    #
    #     if train_ratio > dev_ratio and train_ratio > test_ratio:
    #         _train.append(k)
    #     elif dev_ratio > train_ratio and dev_ratio > test_ratio:
    #         _dev.append(k)
    #     elif test_ratio > train_ratio and test_ratio > dev_ratio:
    #         _test.append(k)
    #     else:
    #         ambiguous.append((k, train_ratio, dev_ratio, test_ratio))
    # for each in ambiguous:
    #     print(each)
    #
    # saved = {
    #     'train': _train,
    #     'dev': _dev,
    #     'test': _test,
    #     'ambiguous': ambiguous
    # }
    # pickle.dump(saved, open("workspace/splits.pkl", 'wb'))

    splits = pickle.load(open('workspace/splits.pkl', 'rb'))
    _train = splits['train']
    _dev = splits['dev']
    _test = splits['test']
    ambiguous = splits['ambiguous']

    print(len(_train), len(_dev), len(_test), len(ambiguous))

    # Step 3
    raw = {}
    with open('mhqg-wq/raw/data.json', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            raw[line['qId']] = line

    train_split = [raw[x] for x in _train]
    dev_split = [raw[x] for x in _dev]
    test_split = [raw[x] for x in _test]

    def dump(path, split):
        with open(path, 'w', encoding='utf-8') as f:
            for each in split:
                f.write(json.dumps(each) + '\n')

    dump('workspace/train.txt', train_split)
    dump('workspace/dev.txt', dev_split)
    dump('workspace/test.txt', test_split)
