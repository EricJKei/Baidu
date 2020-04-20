from multimodels import get_model
from config import cfg
from factory import imdb
from datetime import datetime
from data_loader import dataloader
from progressBar import ShowProcess

import numpy as np
import keras
import h5py
import pickle
import argparse
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, LambdaCallback
from keras.datasets import cifar10


os.environ["CUDA_VISIBLE_DEVICES"] ="4"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

def parse_args():

    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('--datasets', dest='datasets',
                        help='train or test',
                        default='train', type=str)
    parser.add_argument('--model', dest='model',
                        help='image, txt or image+txt',
                        default='image+txt', type=str)
    parser.add_argument('--retrain_model', dest='retrain_model',
                        help='hdf5 file',
                        default=None, type=str)
    parser.add_argument('--test_model', dest='test_model',
                        help='initialize with post-trained model weights',
                        default=None, type=str)
    parser.add_argument('--batch_size', dest='batch_size', default=64, type=int)
    parser.add_argument('--train_data', default='../BDXJTU/data/primary/resample_train.tfrecord', type=str)
    parser.add_argument('--val_data', default='../BDXJTU/data/primary/val.tfrecord', type=str)
    parser.add_argument('--test_data', default='../BDXJTU/data/primary/test.tfrecord', type=str)

    args = parser.parse_args()
    return args

def train_gen(imdb, model):

    sess.run(imdb.init_op['tr_init'])

    while True:

        image, visit, lb = sess.run(imdb.next_batch)
        if model == 'image+txt':
            x1 = image
            x2 = visit
            yield [{'Input1':image, 'Input2':visit}, lb]
        elif model == 'image':
            x1 = image
            yield [image, lb]
        elif model == 'txt':
            x2 = visit
            yield [visit, lb]
        else:
            raise Exception("No data")

def val_gen(imdb, model):

    sess.run(imdb.init_op['val_init'])

    while True:

        image, visit, lb = sess.run(imdb.next_batch)
        if model == 'image+txt':
            x1 = image
            x2 = visit
            yield [{'Input1':image, 'Input2':visit}, lb]
        elif model == 'image':
            x1 = image
            yield [image, lb]
        elif model == 'txt':
            x2 = visit
            yield [visit, lb]
        else:
            raise Exception("No data")


if __name__=='__main__':
    args = parse_args()

    lr = cfg.TRAIN.LR

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                                  cooldown=0, patience=5, min_lr=1e-6)

    opt = 'output'

    filepath = 'weights.{epoch:02d}-{loss:.2f}.hdf5'
    weights_dir = os.path.join(cfg.WEIGHTS_DIR, opt)
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    checkpointer = ModelCheckpoint(filepath=os.path.join(weights_dir, filepath),
                                   verbose=1, save_weights_only=True, period=10)


    model = get_model(input_shape1=[100, 100, 3], input_shape2 = [26, 24, 7], class_num=9,
                      model_name = args.model, retrain_model=args.retrain_model, saved_model=args.test_model)

    for i, layer in enumerate(model.layers):
        #if layer.name.startswith('img') or layer.name.startswith('txt'):
        #    model.layers[i].trainable = False
        #if layer.name.startswith('img'):
        #    model.layers[i].trainable = False
        model.layers[i].trainable = True
 
    #model.summary()
    optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #optimizer=SGD(lr=lr, momentum=0.9, nesterov=True)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

	
    if args.datasets=='train':
        train_dataset=dataloader(args)
        train_dataset.build_loader()
        tmp = args.datasets
        args.datasets='val'
        val_dataset=dataloader(args)
        val_dataset.build_loader()
        args.datasets=tmp

        model.fit_generator(generator=train_gen(train_dataset, args.model),
                            samples_per_epoch=63000//args.batch_size, epochs=100, callbacks=[reduce_lr, checkpointer],
                            validation_data=val_gen(val_dataset, args.model),
                            validation_steps=100)

    elif args.datasets=='val':
        dataset=dataloader(args)
        dataset.build_loader()
        sess.run(dataset.init_op['val_init'])
        steps = 100
        pbar = ShowProcess(steps)
        for step in range(steps):
            pbar.show_process(step)
            image, visit, lb = sess.run(dataset.next_batch)
            if args.model == 'image+txt':
                predict = model.predict([image, visit])
            elif args.model == 'image':
                predict = model.predict(image)
            elif args.model == 'txt':
                predict = model.predict(visit)
            if step == 0:
                predicts=predict
                labels = lb
            else:
                predicts=np.vstack((predicts, predict))
                labels=np.vstack((labels, lb))
        pre =np.equal(np.argmax(predicts, 1), np.argmax(labels, 1))
        accuracy = np.mean(pre)
        print(accuracy)
        

    elif args.datasets=='test':
        sess.run(dataset.init_op['te_init'])
        steps = 10000//100
        for step in range(steps):

            image, visit, lb = sess.run(dataset.next_batch)

            print("Starting to predict, please waiting for it.")
            if args.model == 'image+txt':
                predict = model.predict([image, visit])
            elif args.model == 'image':
                predict = model.predict(image)
            elif args.model == 'txt':
                predict = model.predict(visit)
            if step == 0:
                predicts=predict
            else:
                predicts=np.vstack((predicts, predict))
           
        with open(os.path.join(cfg.WEIGHTS_DIR, args.test_model+'.pkl'), 'wb') as f:
            pickle.dump(predicts, f)

    #    score = model.evaluate(data.getImgs(data.trainImgPaths),
    #                           data.getLabels(data.trainAnnPaths), batch_size=32)

    # history.loss_plot('epoch')
