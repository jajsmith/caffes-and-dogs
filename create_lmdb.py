'''
Title			:create_lmdb.py
Description		:Divides training images into 2 sets and stores them in lmdb dbs for training & validation
usage			:python create_lmdb.py
'''

import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

# Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    # Histogram Equalization
    for depth in range(0,3):
        print('Equalizing image at depth', depth)
        img[:, :, depth] = cv2.equalizeHist(img[:, :, depth])	

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

def make_datum(img, label):
    # image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

def db_write(train_data, in_db, validation_write=False):
    with in_db.begin(write=True) as in_txn:
        for in_idx, img_path in enumerate(train_data):
            if not validation_write and in_idx % VALIDATION == 0:
                continue;
            elif validation_write and in_idx % VALIDATION != 0:
                continue;
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = transform_img(img)
            if 'cat' in img_path:
                label = 0
            else:
                label = 1
            datum = make_datum(img, label)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
            print('{:0>5d}'.format(in_idx), ':', img_path)
    

# MAIN

train_lmdb = '/mnt/git/caffes-and-dogs/input/train_lmdb'
validation_lmdb = '/mnt/git/caffes-and-dogs/input/validation_lmdb'
#train_lmdb = '/user/jonathansmith/source/caffe-and-dogs/input/train_lmdb'
#validation_lmdb = '/user/jonathansmith/source/caffe-and-dogs/input/validation_lmdb'

os.system('rm -rf ' + train_lmdb)
os.system('rm -rf ' + validation_lmdb)

train_data = [img for img in glob.glob("./input/train/*jpg")]
test_data = [img for img in glob.glob("./input/test1/*jpg")]

# Shuffle train_data
random.shuffle(train_data)

print('Creating train_lmdb')

# Set aside every sixth image for validation
VALIDATION = 6

in_db = lmdb.open(train_lmdb, map_size=int(1e12)) # TODO: Explain map_size 
db_write(train_data, in_db, validation_write=False)
in_db.close()

in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
db_write(train_data, in_db, validation_write=True)
in_db.close()

print('\nFinished processing all images')









