import copy
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from numpy import linalg
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import log_loss, mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from time import time

from utils import *


def uniform_sampling(datatrain, attacker_set, epsilon, seed, flip=False):
    if seed is not None:
        np.random.seed(seed)

    X_TRAIN = datatrain['x_train']
    Y_TRAIN = datatrain['y_train']
    G_TRAIN = datatrain['g_train']

    X_ATTACK = attacker_set['x_train']
    Y_ATTACK = attacker_set['y_train']
    G_ATTACK = attacker_set['g_train']

    if flip:
        Y_ATTACK = 1 - Y_ATTACK

    num_points = int(epsilon * len(X_TRAIN))

    idx = np.random.choice(np.arange(len(X_ATTACK)), num_points, replace=False)

    X_TRAIN_p = X_ATTACK[idx]
    Y_TRAIN_p = Y_ATTACK[idx]
    G_TRAIN_p = G_ATTACK[idx]

    X_TRAIN_n = np.concatenate([X_TRAIN, X_TRAIN_p], axis=0)
    Y_TRAIN_n = np.append(Y_TRAIN, Y_TRAIN_p)
    G_TRAIN_n = np.append(G_TRAIN, G_TRAIN_p)

    return X_TRAIN_n, Y_TRAIN_n, G_TRAIN_n


def PFML(datatrain, attacker_set, epsilon, L, num_iters, al, r, lr, A, b, gamma, label_flip=False,
         label_flip_ratio=0.15, number_of_feature_flip=0, one_hot_features=None, measure='equalized_odds'):
    eval_loss_pfml = eval_loss_pfml_equalized_odds if measure == 'equalized_odds' else eval_loss_pfml_demographic_parity
    gradient_pfml = gradient_pfml_equalized_odds if measure == 'equalized_odds' else gradient_pfml_demographic_parity

    # D_c: Clean data
    X_TRAIN = datatrain['x_train']
    Y_TRAIN = datatrain['y_train'].reshape((-1, 1))
    G_TRAIN = datatrain['g_train'].reshape((-1, 1))

    # F(D_k): feasible poisoning data points
    X_ATTACK = attacker_set['x_train']
    Y_ATTACK = attacker_set['y_train'].reshape((-1, 1))
    G_ATTACK = attacker_set['g_train'].reshape((-1, 1))

    X_ATTACK_UF = copy.deepcopy(X_ATTACK)
    Y_ATTACK_UF = copy.deepcopy(Y_ATTACK)
    G_ATTACK_UF = copy.deepcopy(G_ATTACK)

    if label_flip:
        attack_size = X_ATTACK.shape[0]
        flip_size = int(attack_size * label_flip_ratio)
        flip_indices = np.random.choice(np.arange(attack_size), flip_size, replace=False)
        Y_ATTACK[flip_indices] = 1 - Y_ATTACK[flip_indices]

    if number_of_feature_flip and one_hot_features is not None:
    
        # For DP, sensitive feature must not be included in X so it is not included in one_hot_features
        if measure == 'demographic_parity':
            one_hot_features = one_hot_features[:-1]

        one_hot_value_dict = {}
        for idx in one_hot_features:
            one_hot_value_dict[idx] = np.unique(X_TRAIN[:, idx])

        while number_of_feature_flip > 0:
            flip_feature = np.random.choice(one_hot_features, X_ATTACK.shape[0], replace=True)
            for idx in range(0, X_ATTACK.shape[0]):
                value = X_ATTACK[idx, flip_feature[idx]]
                one_hot_value = one_hot_value_dict[flip_feature[idx]]
                flipped_value = one_hot_value[1 - np.where(one_hot_value == value)[0][0]]
                X_ATTACK[idx, flip_feature[idx]] = flipped_value

            number_of_feature_flip -= 1

        X_ATTACK = np.concatenate([X_ATTACK_UF, X_ATTACK], axis=0)
        Y_ATTACK = np.concatenate([Y_ATTACK_UF, Y_ATTACK], axis=0)
        G_ATTACK = np.concatenate([G_ATTACK_UF, G_ATTACK], axis=0)

    # Number of poisoning points (epsilon*n)
    num_points = int(epsilon * len(X_TRAIN))

    # Number of features
    num_feats = X_TRAIN.shape[1]

    # X_poison, Y_poison, G_poison will contain poisoning points selected from F(D_k)
    X_poison = []
    Y_poison = []
    G_poison = []

    # Indices of selected poisoning points at line 3 in pseudo code
    selected_idx = []
    n = len(X_TRAIN)

    # Temporary variables that will contain D_C U D_p
    X_TRAIN_N = X_TRAIN
    Y_TRAIN_N = Y_TRAIN
    G_TRAIN_N = G_TRAIN

    #####
    point_loss_acc = []
    point_loss_fair = []
    point_loss_total = []
    avg_loss_acc = []
    avg_loss_fair = []
    #####

    for j in range(num_iters):

        # If the number of iterations is higher than num_points, the selected_idx will eliminate the
        # oldest one when the condition below is satisfied
        if len(selected_idx) >= num_points:
            selected_idx = selected_idx[-num_points + 1:]

        # Concatenation of D_c and D_p in PFML before calculating loss at line 7 in pseudo code
        if len(X_poison) != 0:
            X_TRAIN_N = np.concatenate([X_TRAIN, X_poison], axis=0)
            Y_TRAIN_N = np.concatenate([Y_TRAIN, Y_poison], axis=0)
            G_TRAIN_N = np.concatenate([G_TRAIN, G_poison], axis=0)

        # eval_loss_pfml() is called to evaluate loss of (x,y) in F(D_k) according to line 7 in pseudo code
        loss, avg_acc, avg_fair, point_acc, point_fair = eval_loss_pfml(A, b, X_ATTACK, Y_ATTACK,
                                                                        G_ATTACK,
                                                                        X_TRAIN_N, Y_TRAIN_N, G_TRAIN_N,
                                                                        gamma,
                                                                        al)
        loss = loss.flatten()

        # We sort the loss of all feasible poisoning data points
        s_loss = sorted(loss, key=lambda x: -x)

        # We choose the feasible poisoning data point that having highest loss and add to poisoning set
        idx_i = 0
        for i in range(len(s_loss)):
            c = False
            for k in np.arange(len(loss))[loss == s_loss[i]]:
                if k not in selected_idx:
                    idx_i = k
                    c = True
                    break
            if c:
                break

        idx = idx_i
        selected_idx.append(idx)
        selected_point_acc = point_acc[idx]
        selected_point_fair = point_fair[idx]
        selected_point_total = loss[idx]

        #####
        point_loss_acc.append(selected_point_acc)
        point_loss_fair.append(selected_point_fair)
        point_loss_total.append(selected_point_total)
        avg_loss_acc.append(avg_acc)
        avg_loss_fair.append(avg_fair)
        #####

        # Add new selected poisoning point to D_p
        X_poison.append(X_ATTACK[idx])
        Y_poison.append(Y_ATTACK[idx])
        G_poison.append(G_ATTACK[idx])

        # Concatenate D_c and new D_p before calculating gradient (line 9 in pseudo code)
        x_new = np.concatenate([X_TRAIN, X_poison], axis=0)
        y_new = np.concatenate([Y_TRAIN, Y_poison], axis=0)
        g_new = np.concatenate([G_TRAIN, G_poison], axis=0)

        # Calculate gradient at line 10 in pseudo code.
        # The L and Delta functions are both calculated with input D_c U D_p.
        # The number of poisoning points t = len(selected_idx)
        grad = gradient_pfml(A, b, x_new, y_new, x_new, y_new, g_new, L, r, n, len(selected_idx))
        A -= grad['dA'] * lr
        b -= grad['db'] * lr

    # Reshape variables to numpy arrays
    X_P = np.array(X_poison).reshape((-1, num_feats))[-num_points:]
    Y_P = np.array(Y_poison).reshape((-1, 1))[-num_points:]
    G_P = np.array(G_poison).reshape((-1, 1))[-num_points:]

    # Concatenate D_c U D_p before returning finished data set
    X_TRAIN_POISON = np.concatenate([X_TRAIN, X_P], axis=0)
    Y_TRAIN_POISON = np.concatenate([Y_TRAIN, Y_P], axis=0).flatten()
    G_TRAIN_POISON = np.concatenate([G_TRAIN, G_P], axis=0).flatten()

    return X_TRAIN_POISON, Y_TRAIN_POISON, G_TRAIN_POISON, avg_loss_acc, avg_loss_fair, point_loss_acc, point_loss_fair, point_loss_total
