import os
import os.path as osp
import numpy as np

from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.ROOT_DIR = os.getcwd()

__C.DATA_DIR = os.path.join(__C.ROOT_DIR, 'data')

__C.CACHE_DIR = os.path.join(__C.DATA_DIR, 'cache')

__C.WEIGHTS_DIR = os.path.join(__C.ROOT_DIR, 'saved_models')

__C.WEIGHTS = os.path.join(__C.DATA_DIR, 'weights', 'resnet50_weights.hdf5')

__C.TB = os.path.join(__C.ROOT_DIR, 'tensorboard')

__C.COMBINE_WEIGHTS = [os.path.join(__C.DATA_DIR, 'weights', 'resnet50_image_weights.hdf5'),
                       os.path.join(__C.DATA_DIR, 'weights', 'resnet50_txt_weights.hdf5')]

__C.ALL_WEIGHTS = [os.path.join(__C.DATA_DIR, 'weights', 'resnet50_imtxt_weights.hdf5')]

__C.IMG_MEAN_PIXEL = np.array([[[119.5723, 137.2779, 158.5521]]])

__C.TXT_MEAN_PIXEL = np.array([[[0.0043, 0.0043, 0.0044, 0.0045, 0.0046, 0.0054,
                                 0.0070, 0.0074, 0.0083, 0.0083, 0.0094, 0.0097]]])


#
# Training options
#
__C.TRAIN = edict()

#Adan=1e-3, SGD=1e-2
__C.TRAIN.LR=1e-3



#
# Testing options
#
__C.TEST = edict()
