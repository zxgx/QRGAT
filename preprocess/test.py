import json
import numpy as np
from collections import Counter
import os
import pickle
from deal_cvt import is_cvt


def _get_answer_coverage(answers, entities):
    if len(answers) == 0:
        return 1.0
    found, total = 0., 0
    for answer in answers:
        if answer["kb_id"] in entities:
            found += 1.
        total += 1
    return found / total


def _get_ans_hits(answers, entities):
    if len(answers) == 0:
        return 1
    for ans in answers:
        if ans['kb_id'] in entities:
            return 1
    return 0


def get_answer_coverage():
    splits = list(range(16))
    tot_coverage = []
    tot_hits = []
    f_debug = open("debug.log", 'w')
    for split in splits:
        coverage = []
        hits = []
        with open('webqsp/qa.raw.%d.json' % split) as f:
            for line in f:
                data = json.loads(line)
                coverage.append(_get_answer_coverage(data['answers'], data['subgraph']['entities']))
                hit = _get_ans_hits(data['answers'], data['subgraph']['entities'])
                hits.append(hit)
                if hit == 0:
                    f_debug.write(line)
        print("split: %s, coverage: %.3f in %d data, hits: %.3f" % (split, np.mean(coverage), len(coverage), np.mean(hits)))
        tot_coverage.extend(coverage)
        tot_hits.extend(hits)
    print("Total coverage: %.3f in %d data, hits: %.3f" % (np.mean(tot_coverage), len(tot_coverage), np.mean(tot_hits)))
    f_debug.close()


def get_subgraph_coverage(dataset_dir, entity_path):
    coverage = []
    ent_set = set()
    with open(entity_path) as f:
        for line in f:
            ent_set.add(line.strip())

    for split in ['train', 'dev', 'test']:
        split_coverage = []
        f = open(os.path.join(dataset_dir, split+'_simple.json'))
        for line in f:
            line = json.loads(line)
            split_coverage.append(_get_answer_coverage(line['answers'], ent_set))
        f.close()
        print("%s coverage: %.3f" % (split, np.mean(split_coverage)))
        coverage.extend(split_coverage)
    print("Total coverage: %.3f" % (np.mean(coverage)))


def count_relations():
    rel_counter = Counter()
    with open('webqsp/subgraph_hop2.txt') as f:
        for line in f:
            _, rel, _ = line.strip().split('\t')
            rel_counter.update([rel])

    with open('rel_freq.txt', 'w') as f:
        for rel, freq in rel_counter.most_common():
            f.write("%s\t%d\n" % (rel, freq))


def count_entities(path):
    # outward_counter = Counter()
    inward_counter = Counter()
    with open(path) as f:
        for line in f:
            head, rel, tail = line.strip().split('\t')
            # outward_counter.update([head])
            inward_counter.update([tail])

    def output_stat(top, output_path):
        with open(output_path, 'w') as f:
            for k, v in top.items():
                f.write("%s\t%d\n" % (k, v))

    # outward_top = dict(outward_counter.most_common(100))
    # output_stat(outward_top, 'outward_top_entities.txt')
    # del outward_top

    inward_top = dict(inward_counter.most_common(100))
    output_stat(inward_top, 'inward_top_entities.txt')

    # outward_counter.clear()
    # inward_counter.clear()
    # with open(path) as f:
    #     for line in f:
    #         head, rel, tail = line.split('\t')
    #         if head in outward_top:
    #             outward_counter.update([rel])
    #         if tail in inward_top:
    #             inward_counter.update([rel])
    #
    # outward_top = dict(outward_counter.most_common(100))  # 36w
    # output_stat(outward_top, 'outward_top_relations.txt')
    # inward_top = dict(inward_counter.most_common(100))
    # output_stat(inward_top, 'inward_top_relations.txt')


def output_sparql(dataset_dir):
    if dataset_dir == 'CWQ':
        files = ["ComplexWebQuestions_train.json", "ComplexWebQuestions_test_wans.json", "ComplexWebQuestions_dev.json"]
    elif dataset_dir == 'webqsp':
        files = ['WebQSP.test.json', 'WebQSP.train.json']
    else:
        raise ValueError("Unknown dataset: "+dataset_dir)

    f_out = open(os.path.join(dataset_dir, 'sparql.txt'), 'w')
    for file in files:
        with open(os.path.join(dataset_dir, file)) as f:
            data = json.load(f)
            for each in data['Questions']:
                ID = each['QuestionId']
                question = each['ProcessedQuestion']
                f_out.write("ID: %s\n" % ID)
                f_out.write("Question: %s\n\n" % question)
                for parse in each['Parses']:
                    sparql = parse['Sparql']
                    f_out.write(sparql+'\n')
                f_out.write('='*80 + '\n')

    f_out.close()


if __name__ == '__main__':
    # count_relations()
    # count_entities('webqsp/subgraph_hop2.txt')
    get_answer_coverage()

    # get_subgraph_coverage('../QA/datasets/webqsp', 'webqsp/ent.txt')

    # with open('data/cvt.pkl', 'rb') as f:
    #     cvt_nodes = pickle.load(f)
    #
    # count = 0
    # with open('webqsp/seed.txt') as f:
    #     for line in f:
    #         if is_cvt(line.strip(), cvt_nodes):
    #             count += 1
    # print(count)
