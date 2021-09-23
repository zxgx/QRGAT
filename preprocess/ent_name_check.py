import json


ent_pool = dict()
with open('CWQ/entities.txt') as f:
    for line in f:
        line = line.strip()
        if line not in ent_pool:
            ent_pool[line] = 'none'
with open('webqsp/entities.txt') as f:
    for line in f:
        line = line.strip()
        if line not in ent_pool:
            ent_pool[line] = 'none'

fb = open('workspace/mid2name.txt', encoding='utf-8')
for line in fb:
    line = json.loads(line)
    for k in line:
        if k in ent_pool:
            ent_pool[k] = line[k][0]
fb.close()
