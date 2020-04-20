from config import cfg
import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
from progressBar import ShowProcess

root_path = os.path.join(cfg.DATA_DIR, 'primary', 'train_image')

sub_dirs = os.walk(root_path)

im_paths = []
for sub_dir in sub_dirs:
    im_paths.extend([os.path.join(sub_dir[0], sub_dir[2][j]) for j in range(len(sub_dir[2])) if sub_dir[1] == []])

for im_path in im_paths:
    assert im_path[-7:-4] == im_path[-18:-15]

count=0
iters = ShowProcess(len(im_paths))
for i, im_path in enumerate(im_paths):
    iters.show_process(i)
    im=cv2.imread(im_path)
    a = np.sum(im[:5,:5,:])
    b = np.sum(im[95:100,:5,:])
    c = np.sum(im[95:100,95:100,:])
    d = np.sum(im[:5,95:100,:])
    if a==0 or b==0 or c==0 or d==0:
        os.remove(im_path)

im_paths = []
for sub_dir in sub_dirs:
    im_paths.extend([os.path.join(sub_dir[0], sub_dir[2][j]) for j in range(len(sub_dir[2])) if sub_dir[1] == []])

ann_dir = os.path.join(cfg.DATA_DIR, 'primary', 'train_visit')
ann_paths = [os.path.join(ann_dir, i) for i in os.listdir(ann_dir)]

im_paths.sort(key=lambda x: int(x[-14:-8]))
ann_paths.sort(key=lambda x: int(x[-14:-8]))
for i in range(len(im_paths)):
    assert im_paths[i][-14:-8] == ann_paths[i][-14:-8]

