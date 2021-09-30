import json
import pickle


ent_pool = dict()
with open('CWQ/entities.txt') as f:
    for line in f:
        line = line.strip()
        if line not in ent_pool:
            ent_pool[line] = None
with open('webqsp/entities.txt') as f:
    for line in f:
        line = line.strip()
        if line not in ent_pool:
            ent_pool[line] = None

fb = open('mid2name.txt', encoding='utf-8')
for line in fb:
    line = json.loads(line)
    for k in line:
        if k in ent_pool:
            assert ent_pool[k] is None
            ent_pool[k] = line[k][0]
fb.close()

count = 0
for k in ent_pool:
    if ent_pool[k] is None:
        count += 1
        ent_pool[k] = 'none'
print("%d / %d" % (count, len(ent_pool)))  # 1654779 / 2839896
pickle.dump(ent_pool, open('mid2name.dict', 'wb'))

