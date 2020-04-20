import os
from config import cfg
from datetime import datetime
import numpy as np
from factory import imdb
import cv2
import pickle
import argparse

parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--model1', dest='model1',
                      help='test model',
                      default=None, type=str)

args = parser.parse_args()

test_data=imdb('test')

name=args.model
path=os.getcwd()
path=os.path.join(path, 'saved_models', name+'.pkl')
with open(path, 'rb') as f:
    demo=pickle.load(f)

res = np.argmax(demo, axis=1)
res=res.reshape((10000,))

for i in range(10000):
    pre=test_data.testAnnPaths[i][-10:-4]+'\t'+str(res[i]+1).zfill(3)+'\n'
    with open('saved_models/' + name + '.txt', 'a') as f:
        f.writelines(pre)


