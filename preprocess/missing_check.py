import os
import json


def update_dict(pool, path):
    with open(path, encoding='utf-8') as f_in:
        for line in f_in:
            pool.add(line.strip())


def normalize(entity):
    if entity.startswith('/'):
        entity = entity[1:].replace('/', '.')
    else:
        entity = '-'.join(entity.split(' '))
    return entity


ent_pool, rel_pool = set(), set()

update_dict(ent_pool, 'CWQ/entities.txt')
update_dict(ent_pool, 'webqsp/entities.txt')

update_dict(rel_pool, 'CWQ/relations.txt')
update_dict(rel_pool, 'webqsp/relations.txt')

missed_ent, missed_rel = set(), set()
for split in ['train.json', 'dev.json', 'test.json']:
    path = os.path.join('mywq', split)
    f = open(path, encoding='utf-8')
    for line in f:
        line = json.loads(line)
        for ans in line['answer_ids']:
            ans = normalize(ans)
            if ans not in ent_pool:
                missed_ent.add(ans)

        for ent in line['inGraph']['g_node_names']:
            ent = normalize(ent)
            if ent not in ent_pool:
                missed_ent.add(ent)
        for _, v in line['inGraph']['g_adj'].items():
            for _, rel in v.items():
                assert type(rel) == str
                rel = normalize(rel)
                if rel not in rel_pool:
                    missed_rel.add(rel)

    f.close()
print(len(missed_ent), len(missed_rel))  # 2996, 108
with open('missed_ent.txt', 'w', encoding='utf-8') as f:
    for each in missed_ent:
        f.write(each+'\n')
with open('missed_rel.txt', 'w', encoding='utf-8') as f:
    for each in missed_rel:
        f.write(each+'\n')
