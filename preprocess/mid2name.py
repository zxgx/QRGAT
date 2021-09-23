import sys
from collections import defaultdict
import json

# data/fb_en.txt
# mid2name.txt
fb_path = sys.argv[1]
output_path = sys.argv[2]

fb = open(fb_path)
mid2name = defaultdict(list)
num_line, num_reserve, multi_name = 0, 0, 0
for line in fb:
    splitline = line.strip().split("\t")
    num_line += 1
    if len(splitline) < 3:
        continue

    rel = splitline[1]
    if rel == 'type.object.name' or rel == 'common.topic.alias':
        if splitline[0] in mid2name:
            multi_name += 1
        mid2name[splitline[0]].append(splitline[2])
        num_reserve += 1
fb.close()
print('num triple: %d, num name: %d, num multi name: %d' % (num_line, num_reserve, multi_name))

output_f = open(output_path, 'w', encoding='utf-8')
for k, v in mid2name.items():
    item = json.dumps({k: v})
    output_f.write(item + '\n')
output_f.close()
