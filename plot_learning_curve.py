'''
Title			:plot_learning_curve.py
Description		:Generates learning curves for acaffe models
Authors			:Adil Moujahid, Jonathan Smith
usage			:python plot_learning_curve.py model_training.log ./model_learning_curve.png
python_version		:2.7
'''

import os
import sys
import subprocess
import pandas as pd

import matplotlib
matplotlib.use('AGG') # TODO: Explain
import matplotlib.pylab as plt

plt.style.use('ggplot')

# Get necessary paths

caffe_path = '/home/ubuntu/src/caffe_python_2/'
model_log_path = sys.argv[1]
learning_curve_path = sys.argv[2]

# CONFIG -- Situate program in model log directory

print 'PLOT LEARNING CURVE'
print 'Beginning plot for ' + model_log_path
model_log_dir_path = os.path.dirname(model_log_path)
os.chdir(model_log_dir_path)


# CONFIG -- Parsing training/validation logs

cmd_parse_log = caffe_path + 'tools/extra/parse_log.sh ' + model_log_path
process = subprocess.Popen(cmd_parse_log, shell=True, stdout=subprocess.PIPE)
process.wait()

# CONFIG -- Read training/test logs

train_log_path = model_log_path + '.train'
test_log_path = model_log_path + '.test'
train_log = pd.read_csv(train_log_path, delim_whitespace=True)
test_log = pd.read_csv(test_log_path, delim_whitespace=True)

# PLOTTING -- training and test losses

fig, ax1 = plt.subplots()

train_loss, = ax1.plot(train_log['#Iters'], train_log['TrainingLoss'], color='red', alpha=0.5)
test_loss, = ax1.plot(test_log['#Iters'], test_log['TestLoss'], linewidth=2, color='green', alpha=0.5)

ax1.set_ylim(ymin=0, ymax=1)
ax1.set_xlabel('Iterations', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.tick_params(labelsize=12)

# PLOTTING -- test accuracy

ax2 = ax1.twinx()

test_accuracy, = ax2.plot(test_log['#Iters'], test_log['TestAccuracy'], linewidth=2, color='blue')

ax2.set_ylim(ymin=0, ymax = 1)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.tick_params(labelsize=15)

# PLOTTING -- legend

plt.legend(
    [train_loss, test_loss, test_accuracy],
    ['Training Loss', 'Test Loss', 'Test Accuracy'],
    bbox_to_anchor=(1, 0.8)
)
plt.title('Training Curve', fontsize=18)

# SAVING

print 'Saving plot to ' + learning_curve_path
plt.savefig(learning_curve_path)

# CLEANING UP

cmd_rm_train = 'rm ' + train_log_path
process = subprocess.Popen(cmd_rm_train, shell=True, stdout=subprocess.PIPE)
process.wait()

cmd_rm_test = 'rm ' + test_log_path
process = subprocess.Popen(cmd_rm_test, shell=True, stdout=subprocess.PIPE)
process.wait()

print 'Done plotting learning curve for ' + model_log_path


