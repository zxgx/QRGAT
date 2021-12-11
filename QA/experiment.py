from sklearn.model_selection import ParameterGrid
import subprocess


param_config = {
    '--dataset': ['CWQ'],
    '--fact_dropout': [0.],

    '--word_dim': [300],
    '--hidden_dim': [100],
    '--question_dropout': [0.3],
    '--linear_dropout': [0.2],
    '--num_step': [4],
    '--relation_dim': [200],
    '--direction': ['all'],
    '--word_emb_path': ['word_emb.npy'],

    '--graph_encoder_type': ['NSM'],
    '--gat_head_dim': [25],
    '--gat_head_size': [8],
    '--gat_dropout': [0.0],
    '': ['--gat_skip'],

    '--lr': [1e-3],
    '--decay_rate': [0.5],
    '--weight_decay': [1e-5],
    '--label_smooth': [0.2],
    '--batch_size': [24]
}

process_str = 'python -u main.py --train --eval'

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
