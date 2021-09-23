from sklearn.model_selection import ParameterGrid
import subprocess


param_config = {
    '--fact_dropout': [0.],

    '--word_dim': [300],
    '--hidden_dim': [100],
    '--question_dropout': [0.3],
    '--linear_dropout': [0.2],
    '--num_step': [3],
    '--relation_dim': [200],
    '--direction': ['all'],
    '--rnn_type': ['LSTM'],

    '--lr': [5e-4],
    '--decay_rate': [1., 0.9, 0.5, 0.1],
    '--dataset': ['webqsp'],
    '--weight_decay': [1e-5],
    '--word_emb_path': ['word_emb.npy'],
    '--label_smooth': [0., 0.1, 0.2]
}

process_str = 'python -u main.py --train --eval --batch_size 32 --epochs 200 --evaluate_every 2 --early_stop 10 --seed 1020'

possible_param_list = list(ParameterGrid(param_config))
print(f'There will be {len(possible_param_list)} runs')

for i, param in enumerate(possible_param_list):
    print(f'{i}/{len(possible_param_list)}:', end='\t')
    # param is a dict
    run_str = process_str
    for arg, arg_val in param.items():
        run_str += f' {arg} {arg_val}'

    print(run_str)
    process = subprocess.run(run_str.split(), encoding='UTF-8')
