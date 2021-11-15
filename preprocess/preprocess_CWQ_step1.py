import sys
import os
import json

from utils import find_entity

dataset_dir = sys.argv[1]  # CWQ directory
# preprocessed test split from https://github.com/lanyunshi/KBQA-GST
files = ["ComplexWebQuestions_train.json", "ComplexWebQuestions_test_wans.json", "ComplexWebQuestions_dev.json"]

output_file = os.path.join(dataset_dir, "step1.json")
f_out = open(output_file, "w")
seed_ent = set()

for file in files:
    with open(os.path.join(dataset_dir, file)) as f_in:
        data = json.load(f_in)
        for each in data:
            ID = each['ID']
            question = each['question']

            ans_list = list()
            for ans in each['answers']:
                ans_list.append({
                    'kb_id': ans['answer_id'],
                    'text': ans['answer']
                })

            sparql = each['sparql']
            ent_list = find_entity(sparql)
            seed_ent.update(ent_list)

            sample = {
                'id': ID,
                'question': question,
                'entities': list(ent_list),
                'answers': ans_list
            }
            f_out.write(json.dumps(sample)+'\n')

            if len(ent_list) == 0:  # manual check
                """
                WebQTest-1956_5610ad35b96b7e38d0836d9c9bd8cd17
                WebQTest-1956_3d784c638d4b044d17ab942ee87db9cf
                WebQTest-1956_650372fbf8339023d462b452320d9f26
                WebQTest-1956_2d8b1c60ea068727728bfdf699ec0ed7
                WebQTest-1956_92dc128ba132d9910b46ff7002918caa
                WebQTest-1956_7df8cb1d6655be1b311c324761de106b
                WebQTest-1956_ed9842b42608ed4538f187dc18a3902b
                WebQTest-1956_174a491f694bb264c4ccbe56d70144f1
                """
                print(ID)

f_out.close()

print(len(seed_ent))  # 11843
seed_file = os.path.join(dataset_dir, 'seed.txt')
f_seed = open(seed_file, 'w')
for each in seed_ent:
    f_seed.write(each+'\n')
f_seed.close()
