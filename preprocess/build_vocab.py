import re
import sys
import os
import json
from nltk.tokenize import RegexpTokenizer
import numpy as np


dataset = sys.argv[1]
glove_path = 'glove.840B.300d.txt'
assert dataset == 'CWQ' or dataset == 'webqsp'

vocab = {}
re_question_tokenizer = RegexpTokenizer(r'\d{1}|\w+|[^\w\s]+')
re_relation_tokenizer = re.compile('[_.]')

# question
splits = ['train.json', 'dev.json', 'test.json']
for split in splits:
    path = os.path.join(dataset, split)
    f = open(path)
    for line in f:
        line = json.loads(line)
        question = line['question']
        tokens = re_question_tokenizer.tokenize(question.lower())
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    f.close()

# relations
f = open(os.path.join(dataset, 'relations.txt'))
for line in f:
    relation_tokens = re_relation_tokenizer.split(line.strip().lower())
    for token in relation_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
f.close()

glove_f = open(glove_path)
vocab_f = open(os.path.join(dataset, 'vocab.txt'), 'w', encoding='utf-8')
hit_words, word_emb = set(), []
for line in glove_f:
    line = line.strip().split()
    if line[0] not in vocab or line[0] in hit_words:
        continue
    else:
        try:
            vocab_f.write(line[0] + '\n')
            word_emb.append(np.array(line[1:], dtype=np.float32))
            hit_words.add(line[0])
        except Exception:
            print(line)
            exit(-1)

glove_f.close()
vocab_f.close()
print('%d tokens in %s, %d tokens reserved' % (len(vocab), dataset, len(word_emb)))

word_emb = np.vstack(word_emb)
print(word_emb.shape)
np.save(os.path.join(dataset, 'word_emb.npy'), word_emb)

discard = set(vocab.keys()) - hit_words
with open(os.path.join(dataset, 'discard.txt'), 'w', encoding='utf-8') as f:
    for each in discard:
        f.write(each+'\n')
