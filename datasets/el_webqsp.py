# the entity linking data can be downloaded from
# https://github.com/salesforce/rng-kbqa

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


def main(rng_dir, my_dir, output_dir):
    os.makedirs(output_dir, exist_ok=False)

    inp_files = ['webqsp_train_elq-5_mid.json', 'webqsp_test_elq-5_mid.json']    
    rng_data = {}
    for rng_file in inp_files:
        with open(os.path.join(rng_dir, rng_file)) as fp:
            data = json.load(fp)
        for each in data:
            assert each["id"] not in rng_data
            rng_data[each["id"]] = each["freebase_ids"]
    
    my_files = ['train_simple.json', 'dev_simple.json', 'test_simple.json']
    ent2idx, idx2ent = get_dict(os.path.join(my_dir, "entities.txt"))
    
    for split in my_files:
        with open(os.path.join(my_dir, split)) as ref, open(os.path.join(output_dir, split), 'w') as oup:
            for line in ref:
                each = json.loads(line)

                sample_id = each["id"]
                assert sample_id in rng_data, f'{sample_id} not in rng data'
                
                intersect = set(rng_data[sample_id]).intersection([idx2ent[ent] for ent in each['entities']])
                each['entities'] = [ent2idx[ent] for ent in intersect]

                oup.write(json.dumps(each)+'\n')


if __name__ == "__main__":
    main(f"dump/rng-kbqa/WebQSP/misc", 
         f"dump/nsm_dataset/webqsp", 
         f"datasets/webqsp")
