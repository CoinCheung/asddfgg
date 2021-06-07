
import os
import os.path as osp
import random


annpth = './datasets/seti/train_labels.csv'
save_root = './datasets/seti/'
n_test = 4000


with open(annpth, 'r') as fr:
    lines = fr.read().splitlines()[1:]

random.shuffle(lines)

train_lines = lines[n_test:]
test_lines = lines[:n_test]

with open(osp.join(save_root, 'train_split.txt'), 'w') as fw:
    fw.write('\n'.join(train_lines))
with open(osp.join(save_root, 'val_split.txt'), 'w') as fw:
    fw.write('\n'.join(test_lines))
