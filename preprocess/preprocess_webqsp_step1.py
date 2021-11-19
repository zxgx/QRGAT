import json
import sys
import os

from utils import find_entity

dataset_dir = sys.argv[1]  # WebQSP directory
files = ['WebQSP.test.json', 'WebQSP.train.json']

output_file = os.path.join(dataset_dir, 'step1.json')
f_out = open(output_file, 'w')

seed_ent = set()
for file in files:
    filename = os.path.join(dataset_dir, file)
    with open(filename) as f_in:
        data = json.load(f_in)
        for each in data['Questions']:
            ID = each['QuestionId']
            question = each['ProcessedQuestion']

            ans_list, ent_list = list(), set()
            for parse in each['Parses']:
                sparql = parse['Sparql']
                ent_set = find_entity(sparql)

                for ans in parse['Answers']:
                    ans_list.append({
                        'kb_id': ans['AnswerArgument'],
                        'text': ans['EntityName']
                    })
                ent_list.update(ent_set)
            seed_ent.update(ent_list)

            sample = {
                'id': ID,
                'question': question,
                'entities': list(ent_list),
                'answers': ans_list  # may have duplicate items
            }
            f_out.write(json.dumps(sample)+'\n')

            # if len(ent_list) == 0:
            #     # WebQTest-1133, WebQTrn-1466
            #     print(ID)

f_out.close()

# print(len(seed_ent))  # 2643
seed_file = os.path.join(dataset_dir, 'seed.txt')
f_seed = open(seed_file, 'w')
for each in seed_ent:
    f_seed.write(each+'\n')
f_seed.close()
