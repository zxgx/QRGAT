import sys
import os


dataset = sys.argv[1]
data_path = os.path.join(dataset, 'train_simple.json')

data = []
with open(data_path) as f:
    for i, line in enumerate(f):
        data.append(line)
        if i == 20:
            break


with open(os.path.join(dataset, 'train_small.json'), 'w', encoding='utf-8') as f:
    for each in data:
        f.write(each)
