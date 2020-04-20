import tensorflow as tf
import numpy as np
import pickle
import cv2

from imgaug import augmenters as iaa
from progressBar import ShowProcess


class dataloader():
    def __init__(self, args):
        self.mode = args.datasets
        self.batch_size = args.batch_size
        self.train_data = args.train_data
        self.val_data = args.val_data
        self.test_data = args.test_data
        self.train_nums = 68199
        self.val_nums = 7912
        self.te_nums = 10000
        self.mean = [0.4685483813116096, 0.538136651819416, 0.6217816988531444]
        self.std = [0.1016119525359456, 0.0900060860845122, 0.08024531900661314]
        self.vis_max = 1297.0


    def build_loader(self):

        if self.mode == 'train':
            #train data
            self.tr_dataset = tf.data.TFRecordDataset([self.train_data])
            self.tr_dataset = self.tr_dataset.map(self._extract_fn, num_parallel_calls=2).prefetch(self.batch_size)
            self.tr_dataset = self.tr_dataset.map(self._normalization_trainval, num_parallel_calls=2).prefetch(self.batch_size)
            self.tr_dataset = self.tr_dataset.shuffle(self.batch_size)
            self.tr_dataset = self.tr_dataset.repeat()
            self.tr_dataset = self.tr_dataset.batch(self.batch_size)

            #generate iterator
            iterator = tf.data.Iterator.from_structure(self.tr_dataset.output_types, self.tr_dataset.output_shapes)
            self.next_batch = iterator.get_next()
            self.init_op = {}
            self.init_op['tr_init'] = iterator.make_initializer(self.tr_dataset)

        elif self.mode == 'val':
            self.val_dataset = tf.data.TFRecordDataset([self.val_data])
            self.val_dataset = self.val_dataset.map(self._extract_fn, num_parallel_calls=2).prefetch(self.batch_size)
            self.val_dataset = self.val_dataset.map(self._normalization_trainval, num_parallel_calls=2).prefetch(self.batch_size)
            self.val_dataset = self.val_dataset.shuffle(self.batch_size)
            self.val_dataset = self.val_dataset.repeat()
            self.val_dataset = self.val_dataset.batch(self.batch_size)

            iterator = tf.data.Iterator.from_structure(self.val_dataset.output_types, self.val_dataset.output_shapes)
            self.next_batch = iterator.get_next()
            self.init_op = {}
            self.init_op['val_init'] = iterator.make_initializer(self.val_dataset)

        elif self.mode == 'test':

            self.te_dataset = tf.data.TFRecordDataset([self.test_data])
            self.te_dataset = self.te_dataset.map(self._extract_fn, num_parallel_calls=2).prefetch(1000)
            self.te_dataset = self.te_dataset.map(self._normalization_test, num_parallel_calls=2).prefetch(1000)
            self.te_dataset = self.te_dataset.batch(1000)

            iterator = tf.data.Iterator.from_structure(self.te_dataset.output_types, self.te_dataset.output_shapes)
            self.next_batch = iterator.get_next()
            self.init_op = {}
            self.init_op['te_init'] = iterator.make_initializer(self.te_dataset)

    def _extract_fn(self, tfrecord):

        if self.mode == 'train' or self.mode == 'val':
            features = tf.parse_single_example(tfrecord,
                                               features={
                                                   'image': tf.FixedLenFeature([], tf.string),
                                                   'visit': tf.FixedLenFeature([], tf.string),
                                                   'label': tf.FixedLenFeature([], tf.int64)
                                               })
        elif self.mode == 'test':
            features = tf.parse_single_example(tfrecord,
                                               features={
                                                   'image': tf.FixedLenFeature([], tf.string),
                                                   'visit': tf.FixedLenFeature([], tf.string),
                                               })
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, (100, 100, 3))
        image = tf.cast(image, tf.float32)

        visit = tf.decode_raw(features['visit'], tf.float64)
        visit = tf.reshape(visit, (26, 24, 7))
        visit = tf.cast(visit, tf.float32)

        if self.mode == 'test':
            return image, visit

        else:
            label = tf.cast(features['label'], tf.int64)
            label = tf.one_hot(label, 9, 1, 0)

            return image, visit, label

    def _normalization_trainval(self, image, visit, label):

        image= image / 255.
        image = (image - self.mean)/self.std

        return image, visit, label

    def _normalization_test(self, image, visit):

        image= image / 255.
        image = (image - self.mean)/self.std

        return image, visit

    def augumentor(self, image):
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.SomeOf((0,4),[
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
            ]),
            iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    #iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
            #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            ], random_order=True)

        image_aug = augment_img.augment_images(image)
        return image_aug


if __name__=='__main__':
    pass



