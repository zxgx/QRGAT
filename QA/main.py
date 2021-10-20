import argparse
import time
import os
import random

import torch
import numpy as np

from utils import get_dict, Tokenizer, QADataset
from model import QAModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=None)

    # Dataset
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fact_dropout', type=float, default=0.)

    # Model
    parser.add_argument('--word_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--question_dropout', type=float, default=0.3)
    parser.add_argument('--linear_dropout', type=float, default=0.2)
    parser.add_argument('--num_step', type=int, default=3)
    parser.add_argument('--relation_dim', type=int, default=200)
    parser.add_argument('--direction', type=str, default='all')
    parser.add_argument('--rnn_type', type=str, default='RNN')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--word_emb_path', type=str, default=None)
    parser.add_argument('--kge_func', type=str, default='ComplEx')
    parser.add_argument('--kge_weight', type=float, default=1e-3)

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

    # Log
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)

    return parser.parse_args()


def train(train_data, dev_data, model, lr, weight_decay, decay_rate, early_stop, epochs, evaluate_every, model_path,
          kge_weight):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
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
        st, tot_loss, tot_kge_loss = time.time(), 0., 0.
        for batch in train_data.batching():
            data_id, question, question_mask, topic_label, entity_mask, subgraph, answer_label, answer_list = batch

            # batch size, max local entity
            _, scores, kge_loss = model((question, question_mask, topic_label, entity_mask, subgraph))

            # smoothed cross entropy
            mask = torch.sum(answer_label, dim=1, keepdim=True)
            mask = (mask > 0.).float()

            # loss = criterion(scores, answer_label) * mask
            scores = torch.log(torch.softmax(scores, dim=1) + 1e-20)
            loss = -(scores * answer_label) * mask
            loss = torch.sum(loss) / loss.shape[0]

            for each in kge_loss:
                loss += kge_weight * each

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            tot_loss += loss.item()
            tot_kge_loss += kge_loss[0].item()

        print('Epoch: %d / %d | Training loss: %.4f, kge loss: %.4f ' %
              (epoch+1, epochs, tot_loss, tot_kge_loss), end='')
        if (epoch+1) % evaluate_every == 0:
            hit1, _, _, f1, eval_kge_loss = evaluate(model, dev_data)
            if hit1 > best_hit1:
                best_hit1, best_model = hit1, model.state_dict()
                stop_increase = 0
            else:
                stop_increase += 1
            if f1 > best_f1:
                best_f1 = f1
            print('| Develop hit@1: %.4f / %.4f, f1: %.4f / %.4f, kge loss: %.4f' %
                  (hit1, best_hit1, f1, best_f1, eval_kge_loss), end='')
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
    tot_kge_loss = 0.

    with torch.no_grad():
        for batch in data_loader.batching():
            data_id, question, question_mask, topic_label, entity_mask, subgraph, answer_label, answer_list = batch

            # batch size, max local entity
            _, scores, kge_loss = model((question, question_mask, topic_label, entity_mask, subgraph))
            tot_kge_loss += kge_loss[0].item()
            predict_dist = torch.softmax(scores, dim=1)

            for d_id, pred_dist, _q, t_dist, a_list in zip(data_id, predict_dist, question, topic_label, answer_list):
                g2l = data_loader.global2local_maps[d_id]
                l2g = {v: k for k, v in g2l.items()}
                t_idx = t_dist.nonzero().squeeze(1)

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
    return hits, hits5, hits10, f1s, tot_kge_loss


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

    start_time = time.time()
    dataset_dir = os.path.join('datasets', args.dataset)

    # KB data
    ent_path = os.path.join(dataset_dir, 'entities.txt')
    ent2idx, idx2ent = get_dict(ent_path)
    rel_path = os.path.join(dataset_dir, 'relations.txt')
    rel2idx, idx2rel = get_dict(rel_path)

    print("There are %d entities and %d relations" % (len(ent2idx), len(rel2idx)))

    tokenizer = Tokenizer(os.path.join(dataset_dir, 'vocab.txt'))
    print('Adopt pre-defined vocabulary of size: %d in tokenizer' % tokenizer.num_token)

    # QA data splits
    train_data_path = os.path.join(dataset_dir, 'train_simple.json')
    train_token_path = None  # os.path.join(dataset_dir, 'train.dep')
    dev_data_path = os.path.join(dataset_dir, 'dev_simple.json')
    dev_token_path = None  # os.path.join(dataset_dir, 'dev.dep')
    test_data_path = os.path.join(dataset_dir, 'test_simple.json')
    test_token_path = None  # os.path.join(dataset_dir, 'test.dep')

    train_data, dev_data, test_data = None, None, None
    device = torch.device(args.device)
    if args.train:
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

    if args.word_emb_path is not None:
        word_emb_path = os.path.join(dataset_dir, args.word_emb_path)
        word_emb = torch.from_numpy(tokenizer.load_glove_emb(word_emb_path)).float()
    else:
        word_emb = None
    model = QAModel(
        word_size=tokenizer.num_token, word_dim=args.word_dim, hidden_dim=args.hidden_dim,
        question_dropout=args.question_dropout, linear_dropout=args.linear_dropout, num_step=args.num_step,
        relation_size=len(rel2idx), relation_dim=args.relation_dim, direction=args.direction, rnn_type=args.rnn_type,
        num_layers=args.num_layers, pretrained_emb=word_emb, kge_f=args.kge_func
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

    if args.train:
        print('Model will be saved to', model_path)
        train(
            train_data, dev_data, model, lr=args.lr, weight_decay=args.weight_decay, decay_rate=args.decay_rate,
            early_stop=args.early_stop, epochs=args.epochs, evaluate_every=args.evaluate_every, model_path=model_path,
            kge_weight=args.kge_weight
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    if args.eval:
        hits1, hits5, hits10, f1, _ = evaluate(model, test_data)
        print("Test hits@1: %.4f, hits@5: %.4f, hits@10: %.4f, f1: %.4f" % (hits1, hits5, hits10, f1))


if __name__ == '__main__':
    main()
