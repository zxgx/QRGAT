# the entity linking data can be downloaded from
# https://github.com/lanyunshi/Multi-hopComplexKBQA

import json
import os


def get_dict(path):
    word2idx, idx2word = dict(), dict()
    with open(path, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            idx = len(word2idx)
            word2idx[word] = idx
            idx2word[idx] = word
    return word2idx, idx2word


def main(qgg_dir, my_dir, output_dir):
    os.makedirs(output_dir, exist_ok=False)

    inp_dirs = ['train_CWQ', 'dev_CWQ', 'test_CWQ']    
    split_data = [[], [], []]
    for i, inp in enumerate(inp_dirs):
        with open(os.path.join(qgg_dir, inp, 'te.json')) as fp:
            for line in fp:
                line = json.loads(line)
                split_data[i].append(list(line.keys()))
        
        print(f"{inp} has {len(split_data[i])} samples")
    
    my_files = ['train_simple.json', 'dev_simple.json', 'test_simple.json']
    ent2idx, idx2ent = get_dict(os.path.join(my_dir, "entities.txt"))
    
    for i, split in enumerate(my_files):
        idx = 0
        zero_cnt = 0
        with open(os.path.join(my_dir, split)) as ref, open(os.path.join(output_dir, split), 'w') as oup:
            for line in ref:
                each = json.loads(line)

                intersect = set(split_data[i][idx]).intersection([idx2ent[ent] for ent in each['entities']])
                if len(intersect) == 0:
                    zero_cnt += 1

                each['entities'] = [ent2idx[ent] for ent in intersect]
                oup.write(json.dumps(each)+'\n')
                idx += 1
        print(f"{split} has {zero_cnt}/{idx} empty linking results")


if __name__ == "__main__":
    main(f"dump/QGG_CWQ/", 
         f"dump/nsm_dataset/CWQ", 
         f"datasets/CWQ")
