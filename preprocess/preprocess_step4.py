import json
import sys
import os


def load_ids(path):
    ids = list()
    with open(path) as f:
        for line in f:
            line = json.loads(line)
            ids.append(line['id'])
    return ids


def simplify_data(qa_files, train_path, dev_path, test_path, new_ent_path, new_rel_path,
                  new_train_path, new_dev_path, new_test_path):
    train_ids = load_ids(train_path)
    dev_ids = load_ids(dev_path)
    test_ids = load_ids(test_path)

    ent2idx, rel2idx = dict(), dict()
    f_ent = open(new_ent_path, 'w')
    f_rel = open(new_rel_path, 'w')
    f_train = open(new_train_path, 'w')
    f_dev = open(new_dev_path, 'w')
    f_test = open(new_test_path, 'w')

    for qa_file in qa_files:
        f = open(qa_file)
        for line in f:
            data = json.loads(line)

            for ent in data['entities']:
                if ent not in ent2idx:
                    ent2idx[ent] = len(ent2idx)
                    f_ent.write(ent+'\n')
            data['entities'] = [ent2idx[ent] for ent in data['entities']]

            for head, rel, tail in data['subgraph']['tuples']:
                if head not in ent2idx:
                    ent2idx[head] = len(ent2idx)
                    f_ent.write(head+'\n')
                if rel not in rel2idx:
                    rel2idx[rel] = len(rel2idx)
                    f_rel.write(rel+'\n')
                if tail not in ent2idx:
                    ent2idx[tail] = len(ent2idx)
                    f_ent.write(tail+'\n')

            for ans in data['answers']:
                if ans['kb_id'] not in ent2idx:
                    ent2idx[ans['kb_id']] = len(ent2idx)
                    f_ent.write(ans['kb_id']+'\n')

            data['subgraph']['tuples'] = [(ent2idx[head], rel2idx[rel], ent2idx[tail]) for head, rel, tail in data['subgraph']['tuples']]
            data['subgraph']['entities'] = [ent2idx[ent] for ent in data['subgraph']['entities']]

            if data['id'] in train_ids:
                f_train.write(json.dumps(data)+"\n")
            elif data['id'] in dev_ids:
                f_dev.write(json.dumps(data)+"\n")
            elif data['id'] in test_ids:
                f_test.write(json.dumps(data)+"\n")
            else:
                print("Exceptional sample: %s" % data['id'])
        f.close()

    f_ent.close()
    f_rel.close()
    f_train.close()
    f_dev.close()
    f_test.close()


if __name__ == '__main__':
    dataset_dir = sys.argv[1]
    num_files = int(sys.argv[2])
    ref_dir = sys.argv[3]

    qa_files = [os.path.join(dataset_dir, 'qa.raw.%d.json' % i) for i in range(num_files)]

    train_path = os.path.join(ref_dir, 'train_simple.json')
    dev_path = os.path.join(ref_dir, 'dev_simple.json')
    test_path = os.path.join(ref_dir, 'test_simple.json')

    ent_path = os.path.join(dataset_dir, 'entities.txt')
    rel_path = os.path.join(dataset_dir, 'relations.txt')
    new_train_path = os.path.join(dataset_dir, 'train.json')
    new_dev_path = os.path.join(dataset_dir, 'dev.json')
    new_test_path = os.path.join(dataset_dir, 'test.json')
    simplify_data(
        qa_files, train_path, dev_path, test_path, ent_path, rel_path, new_train_path, new_dev_path, new_test_path
    )
