import sys
import os
import numpy as np
import re


dataset = sys.argv[1]
assert dataset == 'CWQ' or dataset == 'webqsp'

vocab = dict()
with open(os.path.join(dataset, 'vocab.txt')) as f:
    for line in f:
        line = line.strip()
        vocab[line] = len(vocab)
print("Vocab size: %d" % len(vocab))
word_emb = np.load(os.path.join(dataset, 'word_emb.npy'))

re_relation_tokenizer = re.compile('[_.]')
rel_emb = []
tot_num, hit_num, rand, lines = 0, 0, 0, 0
with open(os.path.join(dataset, 'relations.txt')) as f:
    for line in f:
        rel_words = re_relation_tokenizer.split(line.strip().lower())
        this_emb = []
        for w in rel_words:
            if w in vocab:
                this_emb.append(word_emb[vocab[w]])
                hit_num += 1
            tot_num += 1
        if this_emb:
            this_emb = np.vstack(this_emb).mean(axis=0)
        else:
            rand += 1
            this_emb = np.random.uniform(-1., 1., word_emb.shape[1])
        lines += 1
        rel_emb.append(this_emb)
print("Hit tokens: %d / %d, randomly init relations: %d / %d" % (hit_num, tot_num, rand, lines))

rel_emb = np.vstack(rel_emb)
assert rel_emb.shape[0] == lines
np.save(os.path.join(dataset, 'rel_emb.npy'), rel_emb)
