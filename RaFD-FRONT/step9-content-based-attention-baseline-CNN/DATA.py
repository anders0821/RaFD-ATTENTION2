# -*- coding:utf-8 -*-

import numpy as np
import h5py

def load_cv(FOLD_FOR_TEST):
    # X
    h5 = h5py.File('../DATA-CROP-RAW-IN-IS.mat', 'r')
    X = h5['X'][:]
    X = np.reshape(X, [X.shape[0], X.shape[1], X.shape[2], 1])# 1 channel

    # Y
    Y = h5['LBL'][:]
    Y = np.hstack(Y)
    Y = np.float32(np.eye(7)[Y])# Y -> onehot
    
    # train val
    FOLD = h5['FOLD'][:]
    FOLD = np.hstack(FOLD)
    TRAIN_X = X[FOLD!=FOLD_FOR_TEST]
    VAL_X = X[FOLD==FOLD_FOR_TEST]
    TRAIN_Y = Y[FOLD!=FOLD_FOR_TEST]
    VAL_Y = Y[FOLD==FOLD_FOR_TEST]

    # print shape
    print 'TRAIN_X.shape', TRAIN_X.shape
    print 'TRAIN_Y.shape', TRAIN_Y.shape
    print 'VAL_X.shape', VAL_X.shape
    print 'VAL_Y.shape', VAL_Y.shape

    return [TRAIN_X, TRAIN_Y, VAL_X, VAL_Y]
