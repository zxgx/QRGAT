# QRGAT
Source code for "Question-directed Reasoning with Relation-aware Graph Attention Network for Complex Question Answering over Knowledge Graph"

## Preprocessed data
There are two datasets used in this work, WebQuestionsSP and ComplexWebQuestions.
Preprocessed data can be directly accessed in [this link](https://drive.google.com/file/d/1OL45pvg5EflFI8wqYu8MJa48spnYKgRD/view?usp=sharing) (~1.05GB).

## Experiment logs & model weights
All the training logs for the experiments in the paper and the corresponding model checkpoints can be accesses in [this link](https://drive.google.com/file/d/19aGf1OF2XrQI75yjn54BoLkvdt9mdP47/view?usp=sharing) (~3.14GB).

## Commands & reproducing results
All the commands are listed in the `scripts` directory. Before rerunning the training process, please make sure the pretrained data are downloaded and untared into `datasets` directory; before doing the inference, please make sure the model checkpoints are downloaded and untared into `cache` directory.

Followers can rerun the training process by the shell scripts with `train` in their name, or reproduce the experiment results by scripts with `inference` in their name.


```
git clone https://github.com/zxgx/QRGAT.git
cd QRGAT

# download the preprocessed data
tar -zxf preprocessed_data.tgz -C <data_dir>
cd datasets
ln -s <data_dir>/CWQ CWQ
ln -s <data_dir>/webqsp webqsp
cd ..

# download the model checkpoints
tar -zxf qrgat.tgz -C cache

bash scripts/<any shell script>
```

# update on EL
To get the results with entity linking (EL), we use the EL results from [RnG-KBQA](https://github.com/salesforce/rng-kbqa) for WebQSP.  
The scripts to annotate our datasets with entity linking are in `datasets` dir.  
To run this setup, please (1) replace the dataset softlink (explained below) with the downloaded dataset and (2) use `--enable_entity_linking`

# Citation
If you find this repository helpful, kindly cite:
```
@article{zhang2024question,
  title={Question-Directed Reasoning With Relation-Aware Graph Attention Network for Complex Question Answering Over Knowledge Graph},
  author={Zhang, Geng and Liu, Jin and Zhou, Guangyou and Zhao, Kunsong and Xie, Zhiwen and Huang, Bo},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2024},
  publisher={IEEE}
}
```
