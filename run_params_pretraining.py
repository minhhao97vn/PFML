# Load packages
from utils import *
from attacks import *
from train import *

import random
import tensorflow as tf
from keras import backend as K
import os
import matplotlib.pyplot as plt
import csv
import sys
import copy
import matplotlib.pyplot as plt
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('--random_state', default=1)
parser.add_argument('--lmd', default=5)
parser.add_argument('--learning_rate', default=0.001)
parser.add_argument('--fair_measure', default='equalized_odds')
parser.add_argument('--dataset', default='compas')
args = parser.parse_args()

# Parameters
num_pretrain_iters = 2000
SEED = int(args.random_state)
lmd = int(args.lmd)
l_rate = float(args.learning_rate)
measure = args.fair_measure
dataset = args.dataset

if dataset != 'compas' and dataset != 'adult':
    print('{} is not supported. Please try compas or adult'.format(dataset))
    exit()

if measure == 'equalized_odds':
    gradient_train_params = gradient_train_params_equalized_odds
elif measure == 'demographic_parity':
    gradient_train_params = gradient_train_params_demographic_parity
else:
    print('{} is not supported. Please try equalized_odds or demographic_parity'.format(measure))
    exit()

# Set random seeds
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
random.seed(SEED)
np.random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

# Generate data using generate_dataset method from [1]
data = generate_data(dataset=dataset, measure=measure, random_seed=SEED)

# Prepare data for training
X_TRAIN = data['x_train']
Y_TRAIN = data['y_train']
G_TRAIN = data['g_train']

datatrain = {
    'x_train': X_TRAIN,
    'y_train': Y_TRAIN,
    'g_train': G_TRAIN
}

A = np.random.random((X_TRAIN.shape[1], 1))
b = np.random.random()

x_train = copy.deepcopy(datatrain['x_train'])
y_train = copy.deepcopy(datatrain['y_train']).reshape((-1, 1))
g_train = copy.deepcopy(datatrain['g_train']).reshape((-1, 1))

start_time = time.time()

for i in range(0, num_pretrain_iters):
    grad = gradient_train_params(A, b, x_train, y_train, x_train, y_train, g_train, L=lmd, r=1, n=len(x_train))
    A -= grad['dA'] * l_rate
    b -= grad['db'] * l_rate

end_time = time.time()

print("Time for this run: ", end_time - start_time)

np.save("{}_{}_paramA_L{}_lr{}_iter{}_Dc.npy".format(dataset, measure, lmd, str(l_rate).replace('.', ''), num_pretrain_iters), A)
np.save("{}_{}_paramb_L{}_lr{}_iter{}_Dc.npy".format(dataset, measure, lmd, str(l_rate).replace('.', ''), num_pretrain_iters), b)
