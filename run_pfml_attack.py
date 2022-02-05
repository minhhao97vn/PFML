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
parser.add_argument('--epsilon', default=0.1)
parser.add_argument('--random_state', default=1)
parser.add_argument('--lmd', default=5)
parser.add_argument('--gamma', default=150)
parser.add_argument('--learning_rate', default=0.001)
parser.add_argument('--flip_label', action='store_true')
parser.add_argument('--flip_feature', action='store_true')
parser.add_argument('--fair_measure', default='equalized_odds')
parser.add_argument('--dataset', default='compas')
args = parser.parse_args()

# Parameters
trained_iters = 2000
SEED = int(args.random_state)
ep = float(args.epsilon)  # fraction of poisoning data
base_model = LogisticRegression(max_iter=2000, solver='lbfgs')
lmd = int(args.lmd)
l_rate = float(args.learning_rate)
gamma = int(args.gamma)
measure = args.fair_measure
dataset = args.dataset

is_flip = args.flip_label
flip_ratio = 0.15
is_feature_flip = args.flip_feature
number_of_feature_flip = 1 if is_feature_flip else 0

# print(is_flip)
# print(is_feature_flip)

if dataset == 'compas':
    one_hot_features = [5, 6, 7, 8, 9, 10]
elif dataset == 'adult':
    one_hot_features = np.concatenate([np.arange(6, 79), np.arange(80, 107)], axis=0)
else:
    print('{} is not supported. Please try compas or adult'.format(measure))
    exit()

if measure == 'equalized_odds':
    fair_constraint = EqualizedOdds
    train_fair_model_reduction = train_fair_model_reduction_Equalized_Odds
    train_fair_model_pp = train_fair_model_post_processing_Equalized_Odds
    fair_gap = Equalized_Odds
elif measure == 'demographic_parity':
    fair_constraint = DemographicParity
    train_fair_model_reduction = train_fair_model_reduction_Demographic_Parity
    train_fair_model_pp = train_fair_model_post_processing_Demographic_Parity
    fair_gap = Demographic_Parity
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

X_TEST = data['x_test']
Y_TEST = data['y_test']
G_TEST = data['g_test']

X_ATTACK = data['x_attacker']
Y_ATTACK = data['y_attacker']
G_ATTACK = data['g_attacker']

# hard examples
X_NOISE = data['x_noise']
Y_NOISE = data['y_noise']
G_NOISE = data['g_noise']

# attacker set is augmented by hard examples
X_ATTACK = np.concatenate([X_ATTACK, X_NOISE])
Y_ATTACK = np.append(Y_ATTACK, Y_NOISE)
G_ATTACK = np.append(G_ATTACK, G_NOISE)

datatrain = {
    'x_train': X_TRAIN,
    'y_train': Y_TRAIN,
    'g_train': G_TRAIN
}
attacker_set = {
    'x_train': X_ATTACK,
    'y_train': Y_ATTACK,
    'g_train': G_ATTACK
}
hard_set = {
    'x_train': X_NOISE,
    'y_train': Y_NOISE,
    'g_train': G_NOISE
}

print('Clean size: ', X_TRAIN.shape)

num_attack_iters = int(ep * X_TRAIN.shape[0])

data_acc = []
data_fair = []

A = np.load(
    "{}_{}_paramA_L{}_lr{}_iter{}_Dc.npy".format(dataset, measure, lmd, str(l_rate).replace('.', ''), trained_iters))
b = np.load(
    "{}_{}_paramb_L{}_lr{}_iter{}_Dc.npy".format(dataset, measure, lmd, str(l_rate).replace('.', ''), trained_iters))

point_loss_acc = []
point_loss_fair = []
avg_loss_acc = []
avg_loss_fair = []

# Iterate over alpha values
for t in [0, 2, 5, 8, 10]:

    datatrain_c = copy.deepcopy(datatrain)
    attacker_set_c = copy.deepcopy(attacker_set)
    A_c = copy.deepcopy(A)
    b_c = copy.deepcopy(b)

    alpha = t / 10
    row_acc = ['alpha_' + str(alpha)]
    row_fair = ['alpha_' + str(alpha)]

    start_time = time.time()

    X_POISON, Y_POISON, G_POISON, avg_loss_acc, avg_loss_fair, point_loss_acc, point_loss_fair, point_loss_total = PFML(
        datatrain=datatrain_c, attacker_set=attacker_set_c, epsilon=ep, L=lmd, al=alpha, num_iters=num_attack_iters,
        r=1, lr=l_rate, A=A_c, b=b_c, gamma=gamma, label_flip=is_flip, label_flip_ratio=flip_ratio,
        number_of_feature_flip=number_of_feature_flip, one_hot_features=one_hot_features, measure=measure)

    end_time = time.time()

    print("Alpha: {} - Time for this run: {} ".format(alpha, end_time - start_time))

    # Train fair model - reduction
    for gap in [0.12, 0.1, 0.07, 0.05]:
        fair_model = train_fair_model_reduction(base_model, X_POISON, Y_POISON, G_POISON, fair_constraint(), gap)
        pred_fair = np.array(fair_model(X_TEST))
        ############ need revision
        row_acc.append(accuracy(Y_TEST, pred_fair))
        if measure == 'equalized_odds':
            row_fair.append(max(fair_gap(G_TEST, pred_fair, Y_TEST)))
        else:
            row_fair.append(fair_gap(G_TEST, pred_fair))

    # Train fair model - post processing
    fair_model_PP = train_fair_model_pp(base_model, X_POISON, Y_POISON.astype(int), G_POISON.astype(int),
                                        measure)
    pred_fair_PP = np.array(fair_model_PP(X_TEST, G_TEST.astype(int)))
    ######## need revision
    row_acc.append(accuracy(Y_TEST, pred_fair_PP))
    if measure == 'equalized_odds':
        row_fair.append(max(fair_gap(G_TEST, pred_fair_PP, Y_TEST)))
    else:
        row_fair.append(fair_gap(G_TEST, pred_fair_PP))

    # Save values
    data_acc.append(row_acc)
    data_fair.append(row_fair)

    del X_POISON, Y_POISON, G_POISON, fair_model, pred_fair, fair_model_PP, pred_fair_PP

# Save to files
with open(
        'Gamma_{}_FlipFeature_{}_FLipLabel_{}_Pretrained_{}_iter_{}_epsilon_{}_L_{}_rd_{}_lr_{}_{}_PFML_{}_acc.csv'.format(
            gamma, is_feature_flip, is_flip, trained_iters, num_attack_iters, ep, lmd, SEED, l_rate, dataset, measure),
        'w') as csvfile:
    csv_writer = csv.writer(csvfile)

    csv_writer.writerows(data_acc)

with open(
        'Gamma_{}_FlipFeature_{}_FLipLabel_{}_Pretrained_{}_iter_{}_epsilon_{}_L_{}_rd_{}_lr_{}_{}_PFML_{}_fairness.csv'.format(
            gamma, is_feature_flip, is_flip, trained_iters, num_attack_iters, ep, lmd, SEED, l_rate, dataset, measure),
        'w') as csvfile:
    csv_writer = csv.writer(csvfile)

    csv_writer.writerows(data_fair)
