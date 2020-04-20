from config import cfg
from factory import imdb
from progressBar import ShowProcess
import numpy as np
from datetime import datetime
import time
import cv2
import os
import pickle

# 获取txt文件内容和标签
def getAnns(paths, trainable):

    res=[]
    iters = ShowProcess(len(paths))
    for j, path in enumerate(paths):
        iters.show_process(j)
        
        if trainable == 'train':
            new_path = path.replace('train_visit', 'train_visit_pkl').replace('txt', 'pkl')
            rt = os.path.join(cfg.DATA_DIR, 'primary', 'train_visit_pkl')
        elif trainable == 'test':
            new_path = path.replace('test_visit', 'test_visit_pkl').replace('txt', 'pkl')
            rt = os.path.join(cfg.DATA_DIR, 'primary', 'test_visit_pkl')
        
        if not os.path.exists(rt):
           os.mkdir(rt)

        txt2im = np.zeros((12, 24, 7))
        with open(path, 'r') as f:
            contents = f.readlines()

        instances = []
        for content in contents:
            instances.append(content.split('\t')[1])

        days = []
        for instance in instances:
            days.extend(instance.strip().split(','))

        # '20181221&09|10|11|12|13|14|15'
        for day in days:
            when, hours = day.split('&')
            month = when[4:6]
            week = datetime.strptime(when, "%Y%m%d").weekday()
            hours = hours.split('|')
            for hour in hours:
                m, n, k = int(month) - 1, int(hour), int(week)
                txt2im[int(m), int(n), int(k)] += 1

        txt2im = txt2im.astype('float32')
        with open(new_path, 'wb') as f:
            pickle.dump(txt2im, f)


if __name__=='__main__':
    data=imdb('train')
    tmps = data.allAnnPaths
    print(len(tmps))
    getAnns(tmps, data.train)
