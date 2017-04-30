# -*- coding:utf-8 -*-

import numpy as np
import h5py

def load_cv(FOLD_FOR_TEST):
    # X
    h5 = h5py.File('../DATA-CROP-FIX-IN-IS.mat', 'r')
    X = h5['X'][:]
    X = np.reshape(X, [X.shape[0], X.shape[1], X.shape[2], 1])# 1 channel

    # Y_EXP
    Y_EXP = h5['LBL_EXP'][:]
    Y_EXP = np.hstack(Y_EXP)
    Y_EXP = np.float32(np.eye(7)[Y_EXP])# Y_EXP -> onehot
    
    # Y_PSE
    Y_PSE = h5['LBL_PSE'][:]
    Y_PSE = np.hstack(Y_PSE)
    Y_PSE = np.float32(np.eye(5)[Y_PSE])# Y_PSE -> onehot
    
    # train val
    FOLD = h5['FOLD'][:]
    FOLD = np.hstack(FOLD)
    TRAIN_X = X[FOLD!=FOLD_FOR_TEST]
    VAL_X = X[FOLD==FOLD_FOR_TEST]
    TRAIN_Y_EXP = Y_EXP[FOLD!=FOLD_FOR_TEST]
    VAL_Y_EXP = Y_EXP[FOLD==FOLD_FOR_TEST]
    TRAIN_Y_PSE = Y_PSE[FOLD!=FOLD_FOR_TEST]
    VAL_Y_PSE = Y_PSE[FOLD==FOLD_FOR_TEST]

    # print shape
    print 'TRAIN_X.shape', TRAIN_X.shape
    print 'TRAIN_Y_EXP.shape', TRAIN_Y_EXP.shape
    print 'TRAIN_Y_PSE.shape', TRAIN_Y_PSE.shape
    print 'VAL_X.shape', VAL_X.shape
    print 'VAL_Y_EXP.shape', VAL_Y_EXP.shape
    print 'VAL_Y_PSE.shape', VAL_Y_PSE.shape

    # 使用random_flip_up_down会影响逐pose可视化X-Y-MASK 翻转改在数据加载中先做一遍增广 需要两倍内存
    TRAIN_X_FLIP = TRAIN_X[:, ::-1, :, :]
    TRAIN_Y_EXP_FLIP = TRAIN_Y_EXP
    TRAIN_Y_PSE_FLIP = TRAIN_Y_PSE[:, ::-1]
    VAL_X_FLIP = VAL_X[:, ::-1, :, :]
    VAL_Y_EXP_FLIP = VAL_Y_EXP
    VAL_Y_PSE_FLIP = VAL_Y_PSE[:, ::-1]
    TRAIN_X = np.concatenate([TRAIN_X, TRAIN_X_FLIP])
    TRAIN_Y_EXP = np.concatenate([TRAIN_Y_EXP, TRAIN_Y_EXP_FLIP])
    TRAIN_Y_PSE = np.concatenate([TRAIN_Y_PSE, TRAIN_Y_PSE_FLIP])
    VAL_X = np.concatenate([VAL_X, VAL_X_FLIP])
    VAL_Y_EXP = np.concatenate([VAL_Y_EXP, VAL_Y_EXP_FLIP])
    VAL_Y_PSE = np.concatenate([VAL_Y_PSE, VAL_Y_PSE_FLIP])

    # print shape
    print 'TRAIN_X.shape', TRAIN_X.shape
    print 'TRAIN_Y_EXP.shape', TRAIN_Y_EXP.shape
    print 'TRAIN_Y_PSE.shape', TRAIN_Y_PSE.shape
    print 'VAL_X.shape', VAL_X.shape
    print 'VAL_Y_EXP.shape', VAL_Y_EXP.shape
    print 'VAL_Y_PSE.shape', VAL_Y_PSE.shape

    return [TRAIN_X, TRAIN_Y_EXP, TRAIN_Y_PSE, VAL_X, VAL_Y_EXP, VAL_Y_PSE]
