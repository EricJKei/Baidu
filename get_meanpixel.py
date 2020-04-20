import os, cv2
from config import cfg
from factory import imdb
from progressBar import ShowProcess
import numpy as np

def gm(paths):
    r = 0
    g = 0
    b = 0
    #placeholder = np.zeros((100, 100, 3), dtype = np.float32)
    iters = ShowProcess(len(paths))
    for i, path in enumerate(paths):
        iters.show_process(i)
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        r += np.sum(im[:, :, 0])
        g += np.sum(im[:, :, 1])
        b += np.sum(im[:, :, 2])
    num = len(paths)*10000
    print(r/num, g/num, b/num)


if __name__=='__main__':
    data = imdb('train')
    gm(data._getImgPaths())
    # 120.4110524053897
    # 138.29312131860354
    # 159.7861642692823

