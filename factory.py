from config import cfg
from progressBar import ShowProcess
from datetime import datetime
import numpy as np
import pickle
import random
import cv2
import os
import skimage
import time


class imdb():
    def __init__(self, trainable='train'):
        self.train = trainable
        self.im_dir = os.path.join(cfg.DATA_DIR, 'primary', self.train + '_image')
        self.aug_im_dir = os.path.join(cfg.DATA_DIR, 'primary', self.train + '_image_aug')
        self.txt_dir = os.path.join(cfg.DATA_DIR, 'primary', self.train + '_visit_pkl')
        self.aug_txt_dir = os.path.join(cfg.DATA_DIR, 'primary', self.train + '_visit_aug')
        self.allImgPaths = self._getImgPaths()
        self.allAnnPaths = self._getAnnPaths()

        if self.train == 'train':
            for i in range(len(self.allImgPaths)):
                assert self.allImgPaths[i][-14: -4] == self.allAnnPaths[i][-14:-4]

        if self.train=='train':
            self.trainImgPaths=self.allImgPaths[:31000]
            self.trainAnnPaths=self.allAnnPaths[:31000]
            self.valImgPaths = self.allImgPaths[31000:]
            self.valAnnPaths = self.allAnnPaths[31000:]
            for i in range(len(self.trainImgPaths)):
                assert self.trainImgPaths[i][-14: -4] == self.trainAnnPaths[i][-14:-4]
            for j in range(len(self.valImgPaths)):
                assert self.valImgPaths[j][-14: -4] == self.valAnnPaths[j][-14:-4]
        else:
            self.testImgPaths = self.allImgPaths
            self.testAnnPaths = self.allAnnPaths
            for i in range(len(self.testImgPaths)):
                assert self.testImgPaths[i][-10: -4] == self.testAnnPaths[i][-10:-4]


    # 数据平衡
    # {1: 9542, 2:7538, 3:3590, 4:1358, 5:3464, 6:5507, 7:3517, 8:2617, 9:2867}
    def dataBalance(self, im_paths, ann_paths):
        for i in range(len(im_paths)):
            assert im_paths[i][-14:-4]==ann_paths[i][-14:-4]
        res={}
        for i in range(len(im_paths)):
            lb = int(im_paths[i][-5]) - 1
            if lb not in res.keys():
                res[lb] = 1
            else:
                res[lb] += 1
        add_im_paths = []
        add_ann_paths = []
        print("Starting to load balanced data...")
        for label in list(res.keys()):
            nums=res[label]
            name = str(label + 1).zfill(3)

            im_ls = [im_path for im_path in im_paths if im_path.endswith(name+'.jpg')]
            ann_ls = [ann_path for ann_path in ann_paths if ann_path.endswith(name+'.pkl')]

            if nums<7000:
                add_im_paths.extend(im_ls)
                add_ann_paths.extend(ann_ls)

            elif nums >= 7000:
                add_im_paths.extend(im_ls[:7000])
                add_ann_paths.extend(ann_ls[:7000])

        print("Starting to load augment data...")

        aug_paths = os.listdir(self.aug_im_dir)
        add_im_paths.extend([os.path.join(self.aug_im_dir, aug_path) for aug_path in aug_paths])

        aug_paths = os.listdir(self.aug_txt_dir)
        add_ann_paths.extend([os.path.join(self.aug_txt_dir, aug_path) for aug_path in aug_paths])

        add_im_paths.sort(key=lambda x: int(x[-14:-8]))
        add_ann_paths.sort(key=lambda x: int(x[-14:-8]))

        np.random.seed(0)
        np.random.shuffle(add_im_paths)
        np.random.seed(0)
        np.random.shuffle(add_ann_paths)

        print("Validate the cooperation of multi-model data...")

        for i in range(len(add_ann_paths)):
            assert add_im_paths[i][-14: -4]==add_ann_paths[i][-14:-4]
        for i in range(len(im_paths)):
            assert im_paths[i][-14:-4] == ann_paths[i][-14:-4]
        self.trainImgPaths=add_im_paths
        self.trainAnnPaths=add_ann_paths


    #获取图像绝对路径
    def _getImgPaths(self):
        im_paths=[]
        paths = os.walk(self.im_dir)
        for sub_paths in paths:
            im_paths.extend([os.path.join(sub_paths[0], sub_paths[2][j]) \
                             for j in range(len(sub_paths[2])) if sub_paths[1] == []])

        if self.train=='train':
            im_paths.sort(key=lambda x: int(x[-14:-8]))
            np.random.seed(12)
            np.random.shuffle(im_paths)
        else:
            im_paths.sort(key=lambda x: int(x[-10:-4]))

        return im_paths

    #获取txt绝对路径
    def _getAnnPaths(self):
        txt_paths=[]
        paths = os.walk(self.txt_dir)
        for sub_paths in paths:
            txt_paths.extend([os.path.join(sub_paths[0], sub_paths[2][j]) \
                              for j in range(len(sub_paths[2]))])

        if self.train == 'train':
            txt_paths.sort(key=lambda x: int(x[-14:-8]))
            np.random.seed(12)
            np.random.shuffle(txt_paths)
        else:
            txt_paths.sort(key=lambda x: int(x[-10:-4]))

        return txt_paths

    def getImgs(self, paths):
        ims = []
        # iters = ShowProcess(len(paths))
        for i, path in enumerate(paths):
            # iters.show_process(i)
            im = cv2.imread(path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            sd = random.randint(0, 99)
            if self.train == 'train':
                if sd<10:
                    im=cv2.flip(im, 1)
            im = im.astype('float32')
            im -= cfg.IMG_MEAN_PIXEL
            #im /= 255
            ims.append(im)

        return np.array(ims)

    # 获取txt文件内容和标签
    def getAnns(self, paths):
        res = []
        for path in paths:
            with open(path, 'rb') as f:
                txt2im = pickle.load(f)
            res.append(txt2im)
        return np.array(res)
        
    def getLabels(self, paths):
        labels = [float(path[-5])-1 for path in paths]
        return np.array(labels)

if __name__=='__main__':
    data=imdb('train')
    # data.dataBalance(data.trainImgPaths, data.trainAnnPaths)

    data.dataBalance(data.trainImgPaths, data.trainAnnPaths)

    print(data.getAnns(data.trainAnnPaths[:2]))
    

