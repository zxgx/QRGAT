import os
import torch
import json
import copy

from main import parse_args
from utils import get_dict, Tokenizer, QADataset
from model import QAModel


def load_subgraph(path):
    graphs = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            graph = {}
            for i, (head, relation, tail) in enumerate(data['subgraph']['tuples']):
                if (head, tail) in graph:
                    graph[(head, tail)].append(relation)
                else:
                    graph[(head, tail)] = [relation]

            graphs[data['id']] = graph
    return graphs


def candidates_process(scores, l2g, ignore_prob, max_count=-1):
    candidates = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    retrieved, counter = [], 0
    for c, prob in candidates:
        if c >= len(l2g):
            continue

        if prob < ignore_prob or (max_count > 0 and counter == max_count):
            break

        retrieved.append((l2g[c], prob))
        counter += 1
    return retrieved


paths = []


def search_path(topics, candidate_list, graph, ends):
    for start in topics:
        stack = [start]
        search_step(0, start, stack, candidate_list, graph, ends)
        stack.pop()
        if len(paths) == 10:
            break


def search_step(step, current, stack, candidate_list, graph, ends):
    if len(paths) == 10:
        return

    if step == len(candidate_list):
        if current in ends:
            paths.append(copy.deepcopy(stack))
        return

    for c, probc in candidate_list[step]:
        if len(paths) == 10:
            break
        if (current, c) in graph:
            stack.append((current, graph[(current, c)], c, probc))
            search_step(step+1, c, stack, candidate_list, graph, ends)
            stack.pop()
        elif (c, current) in graph:
            stack.append((current, [-x for x in graph[(c, current)]], c, probc))
            search_step(step+1, c, stack, candidate_list, graph, ends)
            stack.pop()
        elif c == current:
            stack.append((current, [], c, probc))
            search_step(step + 1, c, stack, candidate_list, graph, ends)
            stack.pop()


def main():
    args = parse_args()
    print(args)

    dataset_dir = os.path.join('datasets', args.dataset)
    cache_dir = os.path.join('cache', args.dataset+'_test_results')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    correct_f = open(os.path.join(cache_dir, 'correct.log'), 'w')
    wrong_f = open(os.path.join(cache_dir, 'wrong.log'), 'w')

    # KB data
    ent_path = os.path.join(dataset_dir, 'entities.txt')
    ent2idx, idx2ent = get_dict(ent_path)
    rel_path = os.path.join(dataset_dir, 'relations.txt')
    rel2idx, idx2rel = get_dict(rel_path)

    print("There are %d entities and %d relations" % (len(ent2idx), len(rel2idx)))

    tokenizer = Tokenizer(os.path.join(dataset_dir, 'vocab.txt'))
    print('Adopt pre-defined vocabulary of size: %d in tokenizer' % tokenizer.num_token)

    test_data_path = os.path.join(dataset_dir, 'test_simple.json')
    device = torch.device(args.device)
    test_data = QADataset(
        data_path=test_data_path, ent2idx=ent2idx, rel2idx=rel2idx, tokenizer=tokenizer, batch_size=args.batch_size,
        training=False, device=device,
    )
    graphs = load_subgraph(test_data_path)

    if args.word_emb_path is not None:
        word_emb_path = os.path.join(dataset_dir, args.word_emb_path)
        word_emb = torch.from_numpy(tokenizer.load_glove_emb(word_emb_path)).float()
    else:
        word_emb = None
    model = QAModel(
        word_size=tokenizer.num_token, word_dim=args.word_dim, hidden_dim=args.hidden_dim,
        question_dropout=args.question_dropout, linear_dropout=args.linear_dropout, num_step=args.num_step,
        relation_size=len(rel2idx), relation_dim=args.relation_dim, direction=args.direction, rnn_type=args.rnn_type,
        num_layers=args.num_layers, pretrained_emb=word_emb
    )
    print(model)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
        print('Load model checkpoint from:', args.checkpoint)
    else:
        print('Randomly initialize model')
    model = model.to(device)
    model.eval()

    eps = 0.95
    ignore_prob = (1 - eps) / test_data.max_local_entity
    hits = 0.
    with torch.no_grad():
        for batch in test_data.batching():
            data_id, question, question_mask, topic_label, entity_mask, subgraph, answer_label, answer_list = batch

            # batch size, max local entity
            inter_scores, scores = model((question, question_mask, topic_label, entity_mask, subgraph))

            predict_dist = torch.softmax(scores, dim=1)

            for i, (d_id, pred_dist, _q, t_dist, a_list) in enumerate(zip(data_id, predict_dist, question, topic_label, answer_list)):
                g2l = test_data.global2local_maps[d_id]
                l2g = {v: k for k, v in g2l.items()}
                subgraph = graphs[d_id]

                t_idx = t_dist.nonzero().squeeze(1)
                pred_dist[t_idx] = 0
                _inter_scores = [x[i].tolist() for x in inter_scores]

                # top 10 prediction, [(ent id, prob)]
                ans_retrieved = candidates_process(pred_dist.tolist(), l2g, ignore_prob, max_count=10)
                if len(a_list) == 0:
                    cur_hit = True
                    hits += 1
                elif len(ans_retrieved) > 0 and ans_retrieved[0][0] in a_list:
                    cur_hit = False
                    hits += 1

                log_f = correct_f if cur_hit else wrong_f
                log_f.write('ID: ' + d_id + '\n')
                log_f.write('Question: ' + tokenizer.decode(_q.tolist()) + '\n')
                topic_ents = [l2g[x] for x in t_idx.tolist()]
                log_f.write('Topic Entity: ' + str([idx2ent[x] for x in topic_ents]) + '\n')
                log_f.write('Ground Truth: ' + str([idx2ent[x] for x in a_list]) + '\n')

                log_f.write('Top 10 Candidates for Each Step:\n')
                hopwise_scores = []
                for step, each in enumerate(_inter_scores):
                    log_f.write('Step:\t' + str(step+1) + ': ')
                    candidates = candidates_process(each, l2g, ignore_prob)
                    hopwise_scores.append(candidates)
                    for x in candidates[:10]:
                        log_f.write('(%s, %.2f)\t' % (idx2ent[x[0]], x[1]))
                    log_f.write('\n')

                global paths
                paths = []
                search_path(topic_ents, hopwise_scores, subgraph, a_list)
                log_f.write('Top 10 Reasoning Path:\n')
                for p_idx, each in enumerate(paths):
                    log_f.write('Path %d:\t' % p_idx)
                    for x in each[1:]:
                        rels = []
                        if len(x[1]) > 0:
                            for _r in x[1]:
                                rels.append(idx2rel[_r] if _r >= 0 else 'REV.'+idx2rel[-_r])
                        else:
                            rels = 'SELF_LOOP'
                        log_f.write('(%s, %s, %s, %.2f)\t' % (idx2ent[x[0]], rels, idx2ent[x[2]], x[3]))
                    log_f.write('\n')

                log_f.write('Answer Probabilities:\n')
                for x in ans_retrieved:
                    log_f.write('(%s, %.2f)\t' % (idx2ent[x[0]], x[1]))
                log_f.write('\n')
                log_f.write('=' * 40 + '\n\n')

    hits /= test_data.num_data
    print('Hits@1: %.4f' % hits)
    correct_f.close()
    wrong_f.close()


if __name__ == '__main__':
    main()
