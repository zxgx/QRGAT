from sklearn.model_selection import ParameterGrid
import subprocess


param_config = {
    '--dataset': ['webqsp'],

    '--num_step': [3],
    '--word_emb_path': ['word_emb.npy'],

    '--graph_encoder_type': ['NSM'],
    '--gat_dropout': [0.0],
    '': ['--gat_skip'],

    '--lr': [1e-3, 1e-4, 1e-5],
    '--decay_rate': [0.5],
    '--weight_decay': [1e-5],
    '--label_smooth': [0.2],
    '--batch_size': [32],

    '--pretrained_model_type': ['BERT']
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
