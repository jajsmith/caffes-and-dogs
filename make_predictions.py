'''
Title           :make_predictions.py
Description     :Makes predictions using trained caffe model on test data and generates kaggle submission file
Author          :Adil Moujahid, Jonathan Smith
usage           :python make_predictions.py
python_version  :2.7
'''

import os
import argparse
import sys
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_gpu()

# Arguments 

parser = argparse.ArgumentParser(description = 'Process arguments')
parser.add_argument('-i', nargs = '?', const = 5000, default = 5000)
parser.add_argument('-t', nargs = '?', const = 'test1', default = 'test1')
args = parser.parse_args()

test_folder = str(args.t)
model_iterations = str(args.i)

print 'model_iters: ' + model_iterations

# Constants
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227
REPOSITORY_DIR = '/mnt/git/caffes-and-dogs/'

# HELPERS

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    
    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img

#
# LOAD MODEL AND WEIGHTS
#

print 'Loading model and weights...'

# Read mean image

mean_img_path = REPOSITORY_DIR + 'input/mean.binaryproto'

mean_blob = caffe_pb2.BlobProto()
with open(mean_img_path) as f:
    mean_blob.ParseFromString(f.read())

mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (
     mean_blob.channels,
     mean_blob.height,
     mean_blob.width
    )
)

# Read model arch and trained model's weights

model_architecture_path = REPOSITORY_DIR + 'caffe_models/caffe_model_1/caffenet_deploy_1.prototxt'
model_weights_path = REPOSITORY_DIR + 'caffe_models/caffe_model_1/caffe_model_1_iter_' + model_iterations + '.caffemodel'

net = caffe.Net(model_architecture_path, model_weights_path, caffe.TEST)

# Define image transformers

transformer = caffe.io.Transformer({ 'data' : net.blobs['data'].data.shape })
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2, 0, 1))

print 'Caffe is ready to predict.'

#
# PREDICT
#

print 'Predicting whether each image has a cat or dog in it...'

test_folder_path = REPOSITORY_DIR + 'input/' + test_folder + '/'
test_img_paths = [img_path for img_path in glob.glob(test_folder_path + '*jpg')]

test_ids = []
preds = []
for img_path in test_img_paths:
    
    # Get image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

    # Calculate prediction
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probs = out['prob']

    # Store outcome
    test_ids = test_ids + [img_path.split('/')[-1][:-4]]
    preds = preds + [pred_probs.argmax()]

    print img_path
    print pred_probs.argmax()
    print '--------'

print 'Done predictions.'

#
# SAVE FILE
#

predictions_path = REPOSITORY_DIR + 'caffe_models/caffe_model_1/submission_model_1.csv'
with open(predictions_path, 'w') as f:
    f.write('id,label\n')
    for i in range(len(test_ids)):
        f.write(str(test_ids[i]) + ',' + str(preds[i]) + '\n')
f.close()

print 'Saved predictions to file: ' + predictions_path
