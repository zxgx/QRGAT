import argparse
import time
import os
import random

import torch
import numpy as np

from transformers import AutoTokenizer, AutoConfig
from utils import get_dict, Tokenizer, QADataset
from model import QAModel

model_name_map = {
    'BERT': 'bert-base-uncased', 
    'XLNet': 'xlnet-base-cased', 
    'T5': 'T5-base',
}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1020)

    # Dataset
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fact_dropout', type=float, default=0.)

    # Model
    parser.add_argument('--word_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--question_dropout', type=float, default=0.3)
    parser.add_argument('--linear_dropout', type=float, default=0.2)
    parser.add_argument('--num_step', type=int, default=3)
    parser.add_argument('--emb_entity', action='store_true')
    parser.add_argument('--entity_dim', type=int, default=50)
    parser.add_argument('--relation_dim', type=int, default=200)
    parser.add_argument('--direction', type=str, default='all')
    parser.add_argument('--word_emb_path', type=str, default=None)
    parser.add_argument('--relation_emb_path', type=str, default=None)

    parser.add_argument('--graph_encoder_type', choices=['GAT', 'NSM', 'MIX'], default='GAT', type=str)
    parser.add_argument('--gat_head_dim', type=int, default=25)
    parser.add_argument('--gat_head_size', type=int, default=8)
    parser.add_argument('--gat_dropout', type=float, default=0.6)
    parser.add_argument('--gat_skip', action='store_true')
    parser.add_argument('--gat_bias', action='store_true')

    parser.add_argument('--attn_key', type=str, default='r')
    parser.add_argument('--attn_value', type=str, default='rn')

    # Train & Eval
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--evaluate_every', type=int, default=2)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--decay_rate', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--label_smooth', type=float, default=0.)

    parser.add_argument('--fine_tune', action='store_true')

    # Log
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)

    # pretrained model
    parser.add_argument('--pretrained_model_type', type=str, default=None, 
                        choices=[None, 'BERT', 'XLNet', 'T5' ])
    parser.add_argument('--hugging_face_cache', type=str, default='hugging_face_cache')

    return parser.parse_args()


def train(train_data, dev_data, model, lr, weight_decay, decay_rate, early_stop, epochs, evaluate_every, model_path):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    # criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    if decay_rate >= 1.:
        scheduler = None
    else:
        print('Using scheduler')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max', factor=decay_rate, patience=early_stop // 2, verbose=True)

    best_hit1, best_f1, best_model, stop_increase = 0., 0., model.state_dict(), 0
    for epoch in range(epochs):
        model.train()
        st, tot_loss = time.time(), 0.
        for batch in train_data.batching():
            data_id, question, question_mask, topic_label, candidate_entity, entity_mask, subgraph, answer_label, \
                answer_list = batch

            # batch size, max local entity
            scores = model((question, question_mask, topic_label, candidate_entity, entity_mask, subgraph))

            # smoothed cross entropy
            mask = torch.sum(answer_label, dim=1, keepdim=True)
            mask = (mask > 0.).float()

            # loss = criterion(scores, answer_label) * mask
            scores = torch.log(torch.softmax(scores, dim=1) + 1e-20)
            loss = -(scores * answer_label) * mask
            loss = torch.sum(loss) / loss.shape[0]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            tot_loss += loss.item()

        print('Epoch: %d / %d | Training loss: %.4f ' %
              (epoch+1, epochs, tot_loss), end='')
        if (epoch+1) % evaluate_every == 0:
            hit1, _, _, f1 = evaluate(model, dev_data)
            if hit1 > best_hit1:
                best_hit1, best_model = hit1, model.state_dict()
                stop_increase = 0
            else:
                stop_increase += 1
            if f1 > best_f1:
                best_f1 = f1
            print('| Develop hit@1: %.4f / %.4f, f1: %.4f / %.4f' %
                  (hit1, best_hit1, f1, best_f1), end='')
            if scheduler is not None:
                scheduler.step(hit1)
        print('| Time Cost: %.2fs' % (time.time() - st), end='\n' + '=' * 40 + '\n')

        if stop_increase == early_stop:
            print('Early stop at epoch:', epoch + 1 - early_stop * evaluate_every)
            break

    torch.save(best_model, model_path)
    print('Model is saved to', model_path)


def evaluate(model, data_loader, eps=0.95):
    model.eval()
    ignore_prob = (1 - eps) / data_loader.max_local_entity
    hits, hits5, hits10, f1s = 0., 0., 0., 0.

    with torch.no_grad():
        for batch in data_loader.batching():
            data_id, question, question_mask, topic_label, candidate_entity, entity_mask, subgraph, answer_label, \
                answer_list = batch

            # batch size, max local entity
            scores = model((question, question_mask, topic_label, candidate_entity, entity_mask, subgraph))
            predict_dist = torch.softmax(scores, dim=1)

            for d_id, pred_dist, _q, t_dist, a_list in zip(data_id, predict_dist, question, topic_label, answer_list):
                g2l = data_loader.global2local_maps[d_id]
                l2g = {v: k for k, v in g2l.items()}
                t_idx = torch.nonzero(t_dist, as_tuple=False).squeeze(1)

                pred_dist[t_idx] = 0
                retrieved, correct, acc_prob = [], 0., 0.
                candidates = sorted(enumerate(pred_dist.tolist()), key=lambda x: x[1], reverse=True)

                for c, prob in candidates:
                    if c >= len(g2l):
                        continue

                    if prob < ignore_prob or acc_prob > eps:
                        break

                    retrieved.append(l2g[c])
                    if l2g[c] in a_list:
                        correct += 1
                    acc_prob += prob

                if len(a_list) == 0:
                    if len(retrieved) == 0:
                        f1s += 1.
                    hits += 1.
                    hits5 += 1.
                    hits10 += 1.
                elif len(retrieved) > 0:
                    if retrieved[0] in a_list:
                        hits += 1
                    if len(set(retrieved[:5]) & set(a_list)) > 0:
                        hits5 += 1
                    if len(set(retrieved[:10]) & set(a_list)) > 0:
                        hits10 += 1
                    precision, recall = correct / len(retrieved), correct / len(a_list)
                    if precision > 0. and recall > 0.:
                        f1s += 2. / (1. / precision + 1. / recall)

    hits /= data_loader.num_data
    hits5 /= data_loader.num_data
    hits10 /= data_loader.num_data
    f1s /= data_loader.num_data
    return hits, hits5, hits10, f1s


def main():
    args = parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if args.emb_entity:
            torch.backends.cudnn.enabled = False

    start_time = time.time()
    dataset_dir = os.path.join('datasets', args.dataset)

    # KB data
    ent_path = os.path.join(dataset_dir, 'entities.txt')
    ent2idx, idx2ent = get_dict(ent_path)
    rel_path = os.path.join(dataset_dir, 'relations.txt')
    rel2idx, idx2rel = get_dict(rel_path)

    print("There are %d entities and %d relations" % (len(ent2idx), len(rel2idx)))

    pretrained_model_name = None if args.pretrained_model_type is None else model_name_map[args.pretrained_model_type]
    if pretrained_model_name is None:
        tokenizer = Tokenizer(os.path.join(dataset_dir, 'vocab.txt'))
        print('Adopt pre-defined vocabulary of size: %d in tokenizer' % tokenizer.num_token)
    else:
        if not os.path.exists(args.hugging_face_cache):
            os.makedirs(args.hugging_face_cache)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, cache_dir=args.hugging_face_cache)
        print(f'Adopt {pretrained_model_name} tokenizer with vocab size: {tokenizer}')

    # QA data splits
    train_data_path = os.path.join(dataset_dir, 'train_simple.json')
    # train_data_path = os.path.join(dataset_dir, 'train_small.json')
    train_token_path = None  # os.path.join(dataset_dir, 'train.dep')
    dev_data_path = os.path.join(dataset_dir, 'dev_simple.json')
    dev_token_path = None  # os.path.join(dataset_dir, 'dev.dep')
    test_data_path = os.path.join(dataset_dir, 'test_simple.json')
    test_token_path = None  # os.path.join(dataset_dir, 'test.dep')

    train_data, dev_data, test_data = None, None, None
    device = torch.device(args.device)
    if args.train:
        if args.fine_tune:
            train_data_path = os.path.join(dataset_dir, 'finetune.json')
        train_data = QADataset(
            data_path=train_data_path, token_path=train_token_path, ent2idx=ent2idx, rel2idx=rel2idx,
            tokenizer=tokenizer, batch_size=args.batch_size, training=True, device=device,
            fact_dropout=args.fact_dropout
        )
        dev_data = QADataset(
            data_path=dev_data_path, token_path=dev_token_path, ent2idx=ent2idx, rel2idx=rel2idx,
            tokenizer=tokenizer, batch_size=args.batch_size, training=False, device=device,
        )

    if args.eval:
        test_data = QADataset(
            data_path=test_data_path, token_path=test_token_path, ent2idx=ent2idx, rel2idx=rel2idx,
            tokenizer=tokenizer, batch_size=args.batch_size, training=False, device=device,
        )

    if args.word_emb_path is not None and pretrained_model_name is None:
        word_emb_path = os.path.join(dataset_dir, args.word_emb_path)
        word_emb = torch.from_numpy(tokenizer.load_glove_emb(word_emb_path)).float()
    else:
        word_emb = None

    if args.relation_emb_path is not None:
        rel_emb = os.path.join(dataset_dir, args.relation_emb_path)
        rel_emb = torch.from_numpy(np.load(rel_emb)).float()
    else:
        rel_emb = None

    if args.emb_entity:
        ent_size = len(ent2idx)
    else:
        ent_size = -1
    if args.pretrained_model_type is None:
        word_size = tokenizer.num_token
        word_dim = args.word_dim
    else:
        config = AutoConfig.from_pretrained(
            pretrained_model_name, cache_dir=args.hugging_face_cache)
        word_size = None
        word_dim = config.hidden_size
    model = QAModel(
        word_size=word_size, word_dim=word_dim, hidden_dim=args.hidden_dim,
        question_dropout=args.question_dropout, linear_dropout=args.linear_dropout, num_step=args.num_step,
        pretrained_emb=word_emb, entity_size=ent_size, entity_dim=args.entity_dim, relation_size=len(rel2idx),
        relation_dim=args.relation_dim, pretrained_relation=rel_emb, direction=args.direction,
        graph_encoder_type=args.graph_encoder_type, gat_head_dim=args.gat_head_dim, gat_head_size=args.gat_head_size,
        gat_dropout=args.gat_dropout, gat_skip=args.gat_skip, gat_bias=args.gat_bias,
        attn_key=args.attn_key, attn_value=args.attn_value, 
        pretrained_model_name=pretrained_model_name, hugging_face_cache=args.hugging_face_cache
    )
    print(model)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
        print('Load model checkpoint from:', args.checkpoint)
    else:
        print('Randomly initialize model')
    model = model.to(device)

    if args.save_path is not None:
        model_path = args.save_path
    else:
        tmp = '-'.join([args.dataset, str(int(time.time()))]) + '.pt'
        model_path = os.path.join('cache', tmp)

    print('Initialization finished, time cost: %.2f' % (time.time() - start_time))

    ######################### Sanity Check #####################################
    # for batch in train_data.batching():
    #     data_id, question, question_mask, topic_label, entity_mask, subgraph, answer_label, answer_list = batch
    #     print(question.shape)
    #     for _data_id, _question, _question_mask, _topic_label, _entity_mask, _ans_label, _ans_list in zip(data_id, question, question_mask, topic_label, entity_mask, answer_label, answer_list):
    #         print("ID: %s" % _data_id)
    #
    #         print("Question: %s" % tokenizer.decode(_question.tolist()))
    #         print("Question Mask: %s" % _question_mask.long().tolist())
    #
    #         g2l = train_data.global2local_maps[_data_id]
    #         l2g = {v: k for k, v in g2l.items()}
    #         t_idx = torch.nonzero(_topic_label, as_tuple=False).squeeze(1)
    #         print("Topic entities: %s" % [l2g[x] for x in t_idx.tolist()])
    #
    #         print("Answer List: %s" % ([idx2ent[x] for x in _ans_list]))
    #         ans = torch.nonzero(_ans_label, as_tuple=False).squeeze(1)
    #         print("Answer Label: %s" % ([idx2ent[l2g[x]] for x in ans.tolist()]))
    #         print('=' * 50 + '\n')
    #
    #     print("Subgraphs:")
    #     batch_ids, batch_relations, edge_index = subgraph
    #     edge_counts = torch.bincount(batch_ids).tolist()
    #     batch_start = 0
    #     for i, edge_count in enumerate(edge_counts):
    #         off_set = i * train_data.max_local_entity
    #
    #         heads = (edge_index[0][batch_start: batch_start+5] - off_set).tolist()
    #         rels = batch_relations[batch_start: batch_start+5].tolist()
    #         tails = (edge_index[1][batch_start: batch_start+5] - off_set).tolist()
    #         batch_start += edge_count
    #
    #         _data_id = data_id[i]
    #         g2l = train_data.global2local_maps[_data_id]
    #         l2g = {v: k for k, v in g2l.items()}
    #         print("ID: %s" % _data_id)
    #         for head, rel, tail in zip(heads, rels, tails):
    #             print("[%s, %s, %s]" % (l2g[head], rel, l2g[tail]))
    #         print('=' * 50 + '\n')
    # exit(0)

    # model.eval()
    # with torch.no_grad():
    #     for e in range(3):
    #         for batch in train_data.batching():
    #             data_id, question, question_mask, topic_label, entity_mask, subgraph, answer_label, answer_list = batch
    #
    #             # batch size, max local entity
    #             scores = model((question, question_mask, topic_label, entity_mask, subgraph))
    #             print("Scores: %s" % scores[0, :5].tolist())
    #             print("=" * 50 + '\n')
    # exit(0)

    # optimizer = torch.optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    # model.train()
    # for e in range(50):
    #     tot_loss = 0.
    #     for batch in train_data.batching():
    #         data_id, question, question_mask, topic_label, entity_mask, subgraph, answer_label, answer_list = batch
    #
    #         # batch size, max local entity
    #         scores = model((question, question_mask, topic_label, entity_mask, subgraph))
    #
    #         # smoothed cross entropy
    #         mask = torch.sum(answer_label, dim=1, keepdim=True)
    #         mask = (mask > 0.).float()
    #
    #         # loss = criterion(scores, answer_label) * mask
    #         scores = torch.log(torch.softmax(scores, dim=1) + 1e-20)
    #         loss = -(scores * answer_label) * mask
    #         loss = torch.sum(loss) / loss.shape[0]
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    #         optimizer.step()
    #
    #         tot_loss += loss.item()
    #     print("Epoch: %d, Loss: %.4f" % (e+1, tot_loss))
    # exit(0)
    ##############################################################################
    if args.train:
        print('Model will be saved to', model_path)
        train(
            train_data, dev_data, model, lr=args.lr, weight_decay=args.weight_decay, decay_rate=args.decay_rate,
            early_stop=args.early_stop, epochs=args.epochs, evaluate_every=args.evaluate_every, model_path=model_path,
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    if args.eval:
        hits1, hits5, hits10, f1 = evaluate(model, test_data)
        print("Test hits@1: %.4f, hits@5: %.4f, hits@10: %.4f, f1: %.4f" % (hits1, hits5, hits10, f1))


if __name__ == '__main__':
    main()
