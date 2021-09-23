import argparse
import random
import torch
import numpy as np
import os
import time

from utils import load_data, Vocab, QGDataset, AverageMeter, evaluate_predictions
from model import QGModel


def parse_args():
    parser = argparse.ArgumentParser()

    # environment config
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)

    # dataset
    parser.add_argument('--dataset', type=str, default='data/mhqg-wq')
    parser.add_argument('--levi_graph', action='store_true')
    parser.add_argument('--vocab_path', type=str, default='data/mhqg-wq/vocab.pkl')
    parser.add_argument('--max_vocab_size', type=int, default=20000)
    parser.add_argument('--min_freq', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)

    # model
    parser.add_argument('--max_dec_step', type=int, default=37)  # include EOS
    parser.add_argument('--word_dim', type=int, default=300)
    parser.add_argument('--word_dropout', type=float, default=0.4)
    parser.add_argument('--encoder_hidden_dim', type=int, default=300)
    parser.add_argument('--bidir_encoder', action='store_true')
    parser.add_argument('--encoder_layer', type=int, default=1)
    parser.add_argument('--encoder_dropout', type=float, default=0.3)
    parser.add_argument('--answer_indicator_dim', type=int, default=32)
    parser.add_argument('--gnn_hops', type=int, default=4)
    parser.add_argument('--direction', type=str, default='all')
    parser.add_argument('--decoder_hidden_dim', type=int, default=300)

    # train & eval
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--init_forcing', type=float, default=0.8)
    parser.add_argument('--forcing_decay', type=float, default=0.9999)
    parser.add_argument('--eps_label_smooth', type=float, default=0.2)

    parser.add_argument('--rl_ratio', type=float, default=0.)  # 0.02
    parser.add_argument('--reward_metric', type=str, default='Bleu_4,ROUGE_L')
    parser.add_argument('--reward_ratio', type=str, default='1,0.02')

    parser.add_argument('--grad_clipping', type=float, default=10)
    parser.add_argument('--early_stop', type=int, default=10)

    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--min_output_len', type=int, default=4)
    parser.add_argument('--max_output_len', type=int, default=36)

    return parser.parse_args()


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


def train(model, train_data, dev_data, lr, weight_decay, lr_decay_factor, patience, epochs, init_forcing, forcing_decay,
          eps_label_smooth, rl_ratio, reward_metric, reward_ratio, grad_clipping, early_stop, model_path):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    if lr_decay_factor >= 1.:
        scheduler = None
    else:
        print('Using scheduler')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max', factor=lr_decay_factor, patience=patience, verbose=True
        )

    train_tot_loss, best_metric, best_param, stop_epochs = AverageMeter(), {}, model.state_dict(), 0
    train_metrics = {
        'Bleu_1': AverageMeter(), 'Bleu_2': AverageMeter(), 'Bleu_3': AverageMeter(),
        'Bleu_4': AverageMeter(), 'ROUGE_L': AverageMeter()
    }
    dev_metrics = {
        'Bleu_1': AverageMeter(), 'Bleu_2': AverageMeter(), 'Bleu_3': AverageMeter(),
        'Bleu_4': AverageMeter(), 'ROUGE_L': AverageMeter()
    }
    for k in dev_metrics:
        best_metric[k] = dev_metrics[k].mean()

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        for step, batch in enumerate(train_data.batching()):
            forcing_ratio = init_forcing * (forcing_decay ** step)
            output = model(batch, target_tensor=batch.questions, forcing_ratio=forcing_ratio, compute_loss=True,
                           eps_label_smooth=eps_label_smooth)

            if rl_ratio > 0:
                sample_output = model(batch, saved_out=output, compute_loss=True, loss_reduction=False, sample=True,
                                      eps_label_smooth=eps_label_smooth)
                baseline_output = model(batch, saved_out=output)

                sample_decoded = sample_output.decode_tokens.transpose(0, 1)  # batch size, out seq len
                baseline_decoded = baseline_output.decode_tokens.transpose(0, 1)  # batch size, out seq len

                reward_metric_list = reward_metric.split(',')
                if reward_ratio is not None:
                    reward_ratio = [float(x) for x in reward_ratio.split(',')]
                else:
                    reward_ratio = None

                rl_rewards = []
                for batch_idx in range(len(batch)):
                    sample_predict = batch.decode_single(sample_decoded[batch_idx].tolist(), batch_idx)
                    baseline_predict = batch.decode_single(baseline_decoded[batch_idx].tolist(), batch_idx)
                    sample_scores = evaluate_predictions([batch.org_questions[batch_idx]], [sample_predict])
                    baseline_scores = evaluate_predictions([batch.org_questions[batch_idx]], [baseline_predict])
                    sample_reward = 0.
                    for idx, metric in enumerate(reward_metric_list):
                        _reward = sample_scores[metric] - baseline_scores[metric]
                        if reward_ratio:
                            _reward *= reward_ratio[idx]
                        sample_reward += _reward
                    rl_rewards.append(sample_reward)
                rl_rewards = torch.tensor(rl_rewards).to(model.device)  # batch size
                rl_loss = torch.sum(rl_rewards * sample_output.loss) / len(batch)

                train_loss = (1-rl_ratio) * output.nll_loss + rl_ratio * rl_loss
                train_metric = evaluate_predictions(batch.org_questions, batch.decode_batch(baseline_output.decode_tokens))
            else:
                train_loss = output.nll_loss
                train_metric = evaluate_predictions(batch.org_questions, batch.decode_batch(output.decode_tokens))

            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)
            optimizer.step()

            train_tot_loss.update(train_loss.item())
            AverageMeter.update_metrics(train_metrics, train_metric, len(batch))

        model.eval()
        with torch.no_grad():
            for batch in dev_data.batching():
                output = model(batch)
                dev_metric = evaluate_predictions(batch.org_questions, batch.decode_batch(output.decode_tokens))
                AverageMeter.update_metrics(dev_metrics, dev_metric, len(batch))

        if scheduler is not None:
            scheduler.step(dev_metrics['Bleu_4'].mean())

        epoch_time = time.time() - start_time
        output_str = "Epoch: %d / %d, Time consumed: %.2fs" % (epoch+1, epochs, epoch_time)
        train_str = "\tTraining loss: %.4f, " % train_tot_loss.mean() + AverageMeter.format(train_metrics)
        dev_str = "\tDev " + AverageMeter.format(dev_metrics)
        print('\n'.join([output_str, train_str, dev_str]))

        if best_metric['Bleu_4'] <= dev_metrics['Bleu_4'].mean():
            print("Found best model")
            for k in dev_metrics:
                best_metric[k] = dev_metrics[k].mean()
            best_param = model.state_dict()
            stop_epochs = 0
        else:
            stop_epochs += 1

        if stop_epochs == early_stop:
            print("Early stop at epoch:", (epoch+1-early_stop))
            break
        train_tot_loss.reset()
        AverageMeter.reset_metrics(train_metrics)
        AverageMeter.reset_metrics(dev_metrics)

    print("Done!")
    best_str = 'Best results on dev set:'
    for k in best_metric:
        best_str += ' %s: %.4f' % (k, best_metric[k])
    print(best_str)
    if model_path is not None:
        torch.save(best_param, model_path)
        print("Model is saved to " + model_path)
    model.load_state_dict(best_param)

    return train_tot_loss, train_metrics, dev_metrics


def beam_search(model, data, beam_size, min_output_len, max_output_len):
    model.eval()
    with torch.no_grad():
        predict, target = [], []
        for batch in data.batching():
            hypotheses = model.beam_search(batch, max_output_len, min_output_len, beam_size)

            batch_toks = [each[0].tokens[1:] for each in hypotheses]
            decoded = batch.decode_batch(batch_toks)
            predict.extend(decoded)
            target.extend(batch.org_questions)
    metrics = evaluate_predictions(target, predict)
    return metrics


def main():
    args = parse_args()
    print(args)

    set_random_seed(args.seed)

    if args.save_path is not None:
        if os.path.exists(args.save_path):
            raise ValueError('Directory already exists:', args.save_path)
        os.makedirs(args.save_path)

    device = torch.device(args.device)

    # Dataset
    train_set, vocab = None, None
    train_data, dev_data, test_data = None, None, None
    if args.train:
        train_set = load_data(os.path.join(args.dataset, 'train.json'), args.levi_graph)
        vocab = Vocab.build(args.vocab_path, train_set, args.max_vocab_size, args.min_freq)
        train_data = QGDataset(train_set, True, args.batch_size, vocab, device, shuffle=True)
        dev_set = load_data(os.path.join(args.dataset, 'dev.json'), args.levi_graph)
        dev_data = QGDataset(dev_set, True, args.batch_size, vocab, device, shuffle=False)
    if args.eval:
        test_set = load_data(os.path.join(args.dataset, 'test.json'), args.levi_graph)
        if not args.train:
            if args.vocab_path is None and args.checkpoint:
                raise ValueError("vocab and checkpoint path should be specified")
            vocab = Vocab.build(args.vocab_path)
        test_data = QGDataset(test_set, True, args.batch_size, vocab, device, shuffle=False)

    # Model
    model = QGModel(
        vocab, args.levi_graph, device, args.max_dec_step, args.word_dim, args.word_dropout, args.encoder_hidden_dim,
        args.bidir_encoder, args.encoder_layer, args.encoder_dropout, args.answer_indicator_dim, args.gnn_hops,
        args.direction, args.decoder_hidden_dim
    )
    if args.checkpoint is not None:
        model_path = os.path.join(args.checkpoint, 'model.pt')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)

    '''
    # Sanity check for data
    for batch in dev_data.batching():
        for batch_id, question in enumerate(batch.questions):
            converted, count = [], 0
            for tok_id in question:
                tok_id = tok_id.item()
                if tok_id >= len(vocab):
                    converted.append(batch.oov_dict.index2word[(batch_id, tok_id)])
                else:
                    converted.append(vocab.get_word(tok_id))
                if tok_id != vocab.PAD:
                    count += 1
            print('Question:\n\t%s' % converted)
            assert batch.question_lens[batch_id] == count

            print("Nodes:")
            for node_idx, node_name in enumerate(batch.graphs['node_name_words'][batch_id]):
                converted, count = [], 0
                for tok_id in node_name:
                    tok_id = tok_id.item()
                    converted.append(vocab.get_word(tok_id))
                    if tok_id != vocab.PAD:
                        count += 1
                print("\t%s" % converted)
                # assert count == batch.graphs['node_name_lens'][batch_id][node_idx], \
                #     "encode: %d, decode: %d" % (count, batch.graphs['node_name_lens'][batch_id][node_idx])

            print("Edges:")
            for edge_idx, edge_name in enumerate(batch.graphs['edge_name_words'][batch_id]):
                converted, count = [], 0
                for tok_id in edge_name:
                    tok_id = tok_id.item()
                    converted.append(vocab.get_word(tok_id))
                    if tok_id != vocab.PAD:
                        count += 1
                print("\t%s" % converted)
                # assert count == batch.graphs['edge_name_lens'][batch_id][edge_idx]        
        ret = model(batch, target_tensor=batch.questions)

        print('passed!')
        exit(0)
    '''
    '''
    # Sanity check for model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    train_tot_loss = AverageMeter()
    train_metrics = {
        'Bleu_1': AverageMeter(), 'Bleu_2': AverageMeter(), 'Bleu_3': AverageMeter(),
        'Bleu_4': AverageMeter(), 'ROUGE_L': AverageMeter()
    }
    gold, pred = None, None
    for epoch in range(100):
        model.train()
        start_time = time.time()
        for step, batch in enumerate(train_data.batching()):
            forcing_ratio = args.init_forcing * (args.forcing_decay ** step)
            output = model(batch, target_tensor=batch.questions, forcing_ratio=forcing_ratio, compute_loss=True,
                           eps_label_smooth=0.)

            train_loss = output.nll_loss
            decode_tokens = batch.decode_batch(output.decode_tokens)
            train_metric = evaluate_predictions(batch.org_questions, decode_tokens)
            gold = batch.org_questions
            pred = decode_tokens

            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)
            optimizer.step()

            train_tot_loss.update(train_loss.item())
            AverageMeter.update_metrics(train_metrics, train_metric, len(batch))

        epoch_time = time.time() - start_time
        output_str = "Epoch: %d / %d, Time consumed: %.2fs" % (epoch + 1, 100, epoch_time)
        train_str = "\tTraining loss: %.4f, " % train_tot_loss.mean() + AverageMeter.format(train_metrics)
        print('\n'.join([output_str, train_str]))
        train_tot_loss.reset()
        AverageMeter.reset_metrics(train_metrics)

    print("Gold:")
    print('\n'.join(batch.org_questions))
    print("Pred:")
    print('\n'.join(decode_tokens) + '\n')
    print("passed")
    exit(0)
    '''

    if args.train:
        model_path = os.path.join(args.save_path, 'model.pt') if args.save_path is not None else None
        train(model, train_data, dev_data, args.lr, args.weight_decay, args.lr_decay_factor, args.patience, args.epochs,
              args.init_forcing, args.forcing_decay, args.eps_label_smooth, args.rl_ratio, args.reward_metric,
              args.reward_ratio, args.grad_clipping, args.early_stop, model_path)

    if args.eval:
        metrics = beam_search(model, test_data, args.beam_size, args.min_output_len, args.max_output_len)
        print("Test: ")
        for k, v in metrics.items():
            print("%s: %.4f" % (k, v))


if __name__ == '__main__':
    main()
