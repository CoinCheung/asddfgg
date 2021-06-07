
import os
import os.path as osp
import random
import numpy as np
from tqdm import tqdm


rootpth = './'


with open(osp.join(rootpth, 'train_all.txt'), 'r') as fr:
    lines = fr.read().splitlines()

lb_count = {0:0, 1: 0}
ch_count, ch_val = 0, np.zeros(6)
v_max, v_min = -10000., 10000.
for line in tqdm(lines):
    pth, lb = line.split(',')
    lb_count[int(lb)] += 1
    arr = np.load(pth)
    ch_count += arr[1] * arr[2]
    ch_val += arr.sum(axis=(1, 2))
    a_max, a_min = np.max(arr), np.min(arr)
    if a_max > v_max: v_max = a_max
    if a_min < v_min: v_min = a_min

print('max val: ', v_max)
print('min val: ', v_min)



