
import os
import os.path as osp
import random


n_folds = 5

if not osp.exists('./folds'): os.makedirs('./folds')

with open('./train_all.txt', 'r') as fr:
    lines = fr.read().splitlines()

random.shuffle(lines)

size = len(lines) // n_folds

for ind in range(n_folds):
    test_lines = lines[ind * size:(ind + 1) * size]
    train_lines = lines[:ind * size] + lines[(ind + 1) * size:]
    with open(f'./folds/train_fold_{ind + 1}.txt', 'w') as fw:
        fw.write('\n'.join(train_lines))
    with open(f'./folds/val_fold_{ind + 1}.txt', 'w') as fw:
        fw.write('\n'.join(test_lines))
