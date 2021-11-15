import sys
import os


dataset = sys.argv[1]
train_data_path = os.path.join(dataset, 'train_simple.json')
dev_data_path = os.path.join(dataset, 'dev_simple.json')

data = []
with open(train_data_path) as f1, open(dev_data_path) as f2:
    for line in f1:
        data.append(line)
    for line in f2:
        data.append(line)

output_path = os.path.join(dataset, 'finetune.json')
with open(output_path, 'w') as f:
    for each in data:
        f.write(each)
