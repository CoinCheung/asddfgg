

import os
import os.path as osp
import random


rootpth = './train'
annpth = './train_labels.csv'
save_root = './'
n_test = 4000


with open(annpth, 'r') as fr:
    lines = fr.read().splitlines()[1:]

md5_lbs = {}
for line in lines:
    md5, lb = line.split(',')
    md5_lbs[md5] = lb

md5_pths = {}
for root, folders, files in os.walk(rootpth):
    for fl in files:
        md5 = osp.splitext(fl)[0]
        pth = osp.join(root, fl)
        md5_pths[md5] = pth

md5s = list(md5_pths.keys())
print(len(md5s))

random.shuffle(md5s)

train_md5s = md5s[n_test:]
test_md5s = md5s[:n_test]

train_lines = [
        ','.join([md5_pths[el], md5_lbs[el]])
        for el in train_md5s
        ]
test_lines = [
        ','.join([md5_pths[el], md5_lbs[el]])
        for el in test_md5s
        ]

#  with open(osp.join(save_root, 'train_all.txt'), 'w') as fw:
#      fw.write('\n'.join(train_lines + test_lines))
#  with open(osp.join(save_root, 'train_split.txt'), 'w') as fw:
#      fw.write('\n'.join(train_lines))
#  with open(osp.join(save_root, 'val_split.txt'), 'w') as fw:
#      fw.write('\n'.join(test_lines))


test_root = './test/'
test_paths = []
for root, folders, files in os.walk(test_root):
    for fl in files:
        fpth = osp.join(root, fl)
        test_paths.append(fpth)
with open(osp.join(save_root, 'test.txt'), 'w') as fw:
    fw.write('\n'.join(test_paths))
