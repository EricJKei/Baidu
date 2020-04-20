# -*- coding: utf-8 -*-
"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""
from keras import backend as K
from keras import layers
from keras.models import Sequential
from keras import regularizers
from keras.layers import Activation, concatenate
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, AveragePooling2D, Embedding, Lambda
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from keras.optimizers import Adam, SGD
from config import cfg
from keras_applications.resnet_common import ResNeXt50

import pickle
import os

def identity_block(input_tensor, kernel_size, filters, stage,
                   block, activation=PReLU(), suffix=None):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = suffix + 'res' + str(stage) + block + '_branch'
    bn_name_base = suffix + 'bn' + str(stage) + block + '_branch'
    ac_name_base = suffix + 'activation' + str(stage) + block
    # the default values of strides and padding are (1,1) and 'valid' respectively
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu', name=ac_name_base+'2a')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu', name=ac_name_base+'2b')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor], name=suffix + str(stage) + block)
    x = Activation('relu', name=ac_name_base+'2c')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), activation=PReLU(), suffix=None):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = suffix + 'res' + str(stage) + block + '_branch'
    bn_name_base = suffix + 'bn' + str(stage) + block + '_branch'
    ac_name_base = suffix + 'activation' + str(stage) + block
    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu', name=ac_name_base+'2a')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu', name=ac_name_base+'2b')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut], name=suffix + str(stage) + block)
    x = Activation('relu', name=ac_name_base+'2c')(x)
    return x


def get_model1(input_shape1=[100, 100, 3], class_num=9, model_name=None, retrain_model=None, 
               saved_model=None, weights=None):

    bn_model = 0

    suffix = 'img_'
    kernel_regularizer = regularizers.l2(1e-5)
    # kernel_regularizer = None

	
    img_input = Input(shape=input_shape1, name='Input1')

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='img_conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='img_bn_conv1')(x)
    x = Activation('relu', name='img_activation_1')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='img_MaxPooling_1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), suffix=suffix)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', suffix=suffix)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', suffix=suffix)

    # the feature is halved here if strides is edfault
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', suffix=suffix)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', suffix=suffix)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', suffix=suffix)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', suffix=suffix)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', suffix=suffix)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', suffix=suffix)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', suffix=suffix)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', suffix=suffix)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', suffix=suffix)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', suffix=suffix)
	
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', suffix=suffix)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', suffix=suffix)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', suffix=suffix)

    if model_name == 'image+txt':
        return img_input, x
	
    # x = AveragePooling2D((7, 7), name='avg_pool')(x)
	
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(1024, activation='relu', kernel_regularizer=kernel_regularizer, name='dense_2048')(x)
    #x = BatchNormalization()(x)
    #x= Dropout(0.5)(x)

    output = Dense(class_num, activation='softmax', kernel_regularizer=kernel_regularizer, name='dense_output1')(x)

    model = Model(img_input, output)

    if saved_model != None:
        model.load_weights(os.path.join(cfg.WEIGHTS_DIR, saved_model), by_name=True)
    elif retrain_model != None:
        model.load_weights(os.path.join(cfg.WEIGHTS_DIR, retrain_model), by_name=True)

    return img_input, model

def get_model2(input_shape2=[12, 24, 7], class_num=9, model_name=None, retrain_model=None, 
               saved_model=None, weights=None):

    bn_model = 0

    suffix = 'txt_'
    kernel_regularizer = regularizers.l2(1e-5)
    #kernel_regularizer = None

    img_input = Input(shape=input_shape2, name='Input2')

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='txt_conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='txt_bn_conv1')(x)
    x = Activation('relu', name='txt_activation_1')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='txt_MaxPooling_1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), suffix=suffix)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', suffix=suffix)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', suffix=suffix)

    # the feature is halved here if strides is edfault
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', suffix=suffix)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', suffix=suffix)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', suffix=suffix)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', suffix=suffix)

    '''
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', suffix=suffix)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', suffix=suffix)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', suffix=suffix)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', suffix=suffix)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', suffix=suffix)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', suffix=suffix)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', suffix=suffix)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', suffix=suffix)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', suffix=suffix)
    '''

    if model_name == 'image+txt':
        return img_input, x
    # x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(1024, activation='relu', kernel_regularizer=kernel_regularizer, name='dense_512')(x)
    #x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    #(1,1,2048)

    output = Dense(class_num, activation='softmax', kernel_regularizer=kernel_regularizer, name='dense_output2')(x)

    model = Model(img_input, output)

    if saved_model != None:
        model.load_weights(os.path.join(cfg.WEIGHTS_DIR, saved_model), by_name=True)
    elif retrain_model != None:
        model.load_weights(os.path.join(cfg.WEIGHTS_DIR, retrain_model), by_name=True)

    return img_input, model

def get_model(input_shape1=[100, 100, 3], input_shape2 = [12, 24, 7], class_num=9,
              model_name = 'image+txt', retrain_model=None, saved_model=None):

    if model_name == 'image':
        _, model = get_model1(input_shape1=input_shape1, class_num=class_num, model_name=model_name,
                              retrain_model=retrain_model, saved_model=saved_model)

        print("Successfully loading image model.")
        return model

    elif model_name == 'txt':
        #_, model = DPN26(input_shape2=input_shape2, class_num=class_num, model_name=model_name,
        #                      retrain_model=retrain_model, saved_model=saved_model)

        _, model = get_model2(input_shape2=input_shape2, class_num=class_num, model_name=model_name,
                              retrain_model=retrain_model, saved_model=saved_model)

        print("Successfully loading txt model.")
        return model

    elif model_name == 'image+txt':
        kernel_regularizer = regularizers.l2(1e-5)
        # kernel_regularizer = None
        model1, model2 = model_name.split('+')
        input1, x1 = get_model1(input_shape1=input_shape1, model_name=model_name)
        input2, x2 = get_model2(input_shape2=input_shape2, model_name=model_name)

        x = concatenate([x1, x2],name='c_concat')
        x = GlobalAveragePooling2D(name='c_GAP')(x)
        x = BatchNormalization(name='c_bn1')(x)

        x = Dense(2560, activation='relu', kernel_regularizer=kernel_regularizer, name='c_dense')(x)
        x = BatchNormalization(name='c_bn2')(x)
        x = Dropout(0.5)(x)

        output = Dense(class_num, activation='softmax', name='c_output')(x)

        model = Model(inputs=[input1, input2], outputs=output)
        if saved_model != None:
            model.load_weights(os.path.join(cfg.WEIGHTS_DIR, saved_model), by_name=True)

        elif retrain_model != None:
            weights = retrain_model.split(',')
            if len(weights)==1:
                model.load_weights(os.path.join(cfg.WEIGHTS_DIR, weights[0]), by_name=True)
            elif len(weights)==2:
                for weight in weights:
                    model.load_weights(os.path.join(cfg.WEIGHTS_DIR, weight), by_name=True)

        print("Successfully loading image and txt model.")
        return model
    else:
        raise Exception("No model are loaded.")
