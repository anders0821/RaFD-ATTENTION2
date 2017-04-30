#!/usr/bin/python
# -*- coding:utf-8 -*-

import random
import time
import os
import numpy as np
import tensorflow as tf
import h5py
import DATA


def weight(shape, name=None):
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(0.0, 0.1))


def positive_bias(shape, name=None):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))


def zero_bias(shape, name=None):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


def conv(x, w, name=None):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', name=name)


def pool(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def build_graph(train):
    with tf.device("/cpu:0"):
        with tf.name_scope('X'):
            x = tf.placeholder(tf.float32, [None, 284, 284, 1], name='x')
            mlp = x

        if train:
            with tf.name_scope('RANDOM-CROP-FLIP'):
                crop_x = tf.map_fn(lambda img: tf.random_crop(img, [272, 272, 1]), mlp)
                # crop_x = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), crop_x)  # 使用random_flip_up_down会影响逐pose可视化X-Y-MASK 翻转改在数据加载中先做一遍增广 需要两倍内存
                mlp = crop_x
        else:
            with tf.name_scope('CENTER-CROP'):
                crop_x = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img, 272, 272), mlp)
                mlp = crop_x

    with tf.name_scope('CONV-1'):
        c1 = 16
        res = tf.pad(mlp, [[0, 0], [0, 0], [0, 0], [0, c1-1]])
        mlp = conv(mlp, weight([3, 3, 1, c1], name='w11')) + positive_bias([c1], name='b11')
        mlp = tf.nn.relu(mlp, name='conv1')
        mlp = conv(mlp, weight([3, 3, c1, c1], name='w12')) + positive_bias([c1], name='b12')
        mlp = tf.nn.relu(mlp, name='conv2')
        mlp = conv(mlp, weight([3, 3, c1, c1], name='w13')) + positive_bias([c1], name='b13')
        mlp = tf.nn.relu(mlp, name='conv3')
        # mlp = conv(mlp, weight([3, 3, c1, c1], name='w14')) + positive_bias([c1], name='b14')
        # mlp = tf.nn.relu(mlp, name='conv4')
        mlp = tf.add(mlp, res, name='res')
        mlp = pool(mlp, name='pool')

    with tf.name_scope('CONV-2'):
        c2 = 32
        res = tf.pad(mlp, [[0, 0], [0, 0], [0, 0], [0, c2-c1]])
        mlp = conv(mlp, weight([3, 3, c1, c2], name='w21')) + positive_bias([c2], name='b21')
        mlp = tf.nn.relu(mlp, name='conv1')
        mlp = conv(mlp, weight([3, 3, c2, c2], name='w22')) + positive_bias([c2], name='b22')
        mlp = tf.nn.relu(mlp, name='conv2')
        mlp = conv(mlp, weight([3, 3, c2, c2], name='w23')) + positive_bias([c2], name='b23')
        mlp = tf.nn.relu(mlp, name='conv3')
        # mlp = conv(mlp, weight([3, 3, c2, c2], name='w24')) + positive_bias([c2], name='b24')
        # mlp = tf.nn.relu(mlp, name='conv4')
        mlp = tf.add(mlp, res, name='res')
        mlp = pool(mlp, name='pool')

    with tf.name_scope('CONV-3'):
        c3 = 64
        res = tf.pad(mlp, [[0, 0], [0, 0], [0, 0], [0, c3-c2]])
        mlp = conv(mlp, weight([3, 3, c2, c3], name='w31')) + positive_bias([c3], name='b31')
        mlp = tf.nn.relu(mlp, name='conv1')
        mlp = conv(mlp, weight([3, 3, c3, c3], name='w32')) + positive_bias([c3], name='b32')
        mlp = tf.nn.relu(mlp, name='conv2')
        # mlp = conv(mlp, weight([3, 3, c3, c3], name='w33')) + positive_bias([c3], name='b33')
        # mlp = tf.nn.relu(mlp, name='conv3')
        # mlp = conv(mlp, weight([3, 3, c3, c3], name='w34')) + positive_bias([c3], name='b34')
        # mlp = tf.nn.relu(mlp, name='conv4')
        mlp = tf.add(mlp, res, name='res')
        mlp = pool(mlp, name='pool')

    with tf.name_scope('CONV-4'):
        c4 = 128
        res = tf.pad(mlp, [[0, 0], [0, 0], [0, 0], [0, c4-c3]])
        mlp = conv(mlp, weight([3, 3, c3, c4], name='w41')) + positive_bias([c4], name='b41')
        mlp = tf.nn.relu(mlp, name='conv1')
        mlp = conv(mlp, weight([3, 3, c4, c4], name='w42')) + positive_bias([c4], name='b42')
        mlp = tf.nn.relu(mlp, name='conv2')
        # mlp = conv(mlp, weight([3, 3, c4, c4], name='w43')) + positive_bias([c4], name='b43')
        # mlp = tf.nn.relu(mlp, name='conv3')
        # mlp = conv(mlp, weight([3, 3, c4, c4], name='w44')) + positive_bias([c4], name='b44')
        # mlp = tf.nn.relu(mlp, name='conv4')
        mlp = tf.add(mlp, res, name='res')
        mlp = pool(mlp, name='pool')

    with tf.name_scope('MASK'):
        mask = tf.constant([
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
            [1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0, 1.0 / 289.0],
        ], tf.float32)
        mask = tf.reshape(mask, [1, 17, 17, 1])
        mask = tf.tile(mask, [tf.shape(mlp)[0], 1, 1, tf.shape(mlp)[3]])
        mlp = tf.mul(mlp, mask)
        mlp = tf.reduce_sum(mlp, [1, 2], True)

    if train:
        with tf.name_scope('DROPOUT'):
            mlp = tf.nn.dropout(mlp, 0.5, noise_shape=tf.shape(mlp)*[1, 0, 0, 1]+[0, 1, 1, 0], name='dropout')  # dropout by map

    with tf.name_scope('FLAT'):
        mlp = tf.reshape(mlp, [-1, c4], name='flat')

    '''
    if train:
        with tf.name_scope('DROPOUT'):
            mlp = tf.nn.dropout(mlp, 0.5, name='dropout')
    '''

    # 1FC
    with tf.name_scope('FC'):
        logit_exp = tf.nn.xw_plus_b(mlp, weight([c4, 7], name='w7_exp'), zero_bias([7], name='b7_exp'), name='logit_exp')
        logit_pse = tf.nn.xw_plus_b(mlp, weight([c4, 5], name='w7_pse'), zero_bias([5], name='b7_pse'), name='logit_pse')
        del mlp

    with tf.name_scope('Y'):
        y_exp = tf.placeholder(tf.float32, [None, 7], name='y_exp')
        y_pse = tf.placeholder(tf.float32, [None, 5], name='y_pse')

    with tf.name_scope('SOFTMAX-WITH-LOSS'):
        loss_exp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit_exp, y_exp), name='loss_exp')
        loss_pse = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit_pse, y_pse), name='loss_pse')
        lambda_ = 0
        loss = loss_exp + lambda_ * loss_pse

    with tf.name_scope('SOFTMAX'):
        prob_exp = tf.nn.softmax(logit_exp, name='prob_exp')
        prob_pse = tf.nn.softmax(logit_pse, name='prob_pse')

    with tf.name_scope('ACC'):
        acc_exp = tf.equal(tf.argmax(prob_exp, 1), tf.argmax(y_exp, 1), name='correct_exp')
        acc_exp = tf.reduce_mean(tf.cast(acc_exp, tf.float32), name='acc_exp')
        acc_pse = tf.equal(tf.argmax(prob_pse, 1), tf.argmax(y_pse, 1), name='correct_pse')
        acc_pse = tf.reduce_mean(tf.cast(acc_pse, tf.float32), name='acc_pse')

    if train:
        with tf.name_scope('OPT'):
            opt = tf.train.AdamOptimizer(name='opt')
            train_op = opt.minimize(loss, name='train_op')
    else:
        train_op = None

    return [x, y_exp, y_pse, loss, acc_exp, acc_pse, train_op]


def train_val():
    # 常量
    OUTPUT_DIR = './100-1/'
    MB = 30
    SNAPSHOT_RESUME_FROM = 0
    EPOCH_MAX = 130
    FOLD_FOR_VAL = 1

    # 加载数据集
    [TRAIN_X, TRAIN_Y_EXP, TRAIN_Y_PSE, VAL_X, VAL_Y_EXP, VAL_Y_PSE] = DATA.load_cv(FOLD_FOR_VAL)

    # 创建计算图
    with tf.Graph().as_default():
        # 为重现使用固定的随机数种子
        # 不同版本TF结果不同  同一版本下cpu/gpu结果相同
        # 可能和快照功能冲突
        seed = 1
        np.random.seed(seed)
        tf.set_random_seed(seed)
        random.seed(seed)

        # 创建计算图
        with tf.variable_scope('GRAPH', reuse=None):
            [train_x, train_y_exp, train_y_pse, train_loss, train_acc_exp, train_acc_pse, train_op] = build_graph(True)
        with tf.variable_scope('GRAPH', reuse=True):
            [val_x, val_y_exp, val_y_pse, val_loss, val_acc_exp, val_acc_pse, _] = build_graph(False)

        # 创建会话
        with tf.Session() as sess:
            # 训练初始化或加载快照
            if SNAPSHOT_RESUME_FROM == 0:
                tf.global_variables_initializer().run()
            else:
                tf.train.Saver().restore(sess, OUTPUT_DIR+'snapshot-'+str(SNAPSHOT_RESUME_FROM))
                print 'load snapshot'

            # 训练循环
            # 1 ~ EPOCH_MAX 或 SNAPSHOT_RESUME_FROM+1 ~ EPOCH_MAX
            for epoch in xrange(SNAPSHOT_RESUME_FROM+1, EPOCH_MAX+1):
                print '---------- epoch %d ----------' % epoch
                t = time.time()
                mean_train_loss = 0.0
                mean_train_acc_exp = 0.0
                mean_train_acc_pse = 0.0
                mean_train_count = 0
                mean_val_loss = 0.0
                mean_val_acc_exp = 0.0
                mean_val_acc_pse = 0.0
                mean_val_count = 0

                # 打乱训练集
                idx = np.random.permutation(TRAIN_X.shape[0])
                TRAIN_X = TRAIN_X[idx, :, :]
                TRAIN_Y_EXP = TRAIN_Y_EXP[idx, :]
                TRAIN_Y_PSE = TRAIN_Y_PSE[idx, :]

                # 训练
                # 抛弃训练集尾部 担心变化的MB会影响ADAM BATCHNORM等计算
                ITER_COUNT = TRAIN_X.shape[0] / MB
                for itr in xrange(ITER_COUNT):
                    train_x_val = TRAIN_X[itr * MB:itr * MB + MB, :, :]
                    train_y_exp_val = TRAIN_Y_EXP[itr * MB:itr * MB + MB, :]
                    train_y_pse_val = TRAIN_Y_PSE[itr * MB:itr * MB + MB, :]
                    [_, train_loss_val, train_acc_exp_val, train_acc_pse_val] =\
                        sess.run([train_op, train_loss, train_acc_exp, train_acc_pse], feed_dict={train_x: train_x_val, train_y_exp: train_y_exp_val, train_y_pse: train_y_pse_val})
                    mean_train_loss += train_loss_val * MB
                    mean_train_acc_exp += train_acc_exp_val * MB
                    mean_train_acc_pse += train_acc_pse_val * MB
                    mean_train_count += MB

                print 'mean train loss %g, mean train acc exp %g, mean train acc pse %g' % (mean_train_loss / mean_train_count, mean_train_acc_exp / mean_train_count, mean_train_acc_pse / mean_train_count)

                # 验证
                # 保留验证集尾部
                ITER_COUNT = ((VAL_X.shape[0] - 1) / MB) + 1
                for itr in xrange(ITER_COUNT):
                    mb = min(itr * MB + MB, VAL_X.shape[0]) - itr * MB
                    val_x_val = VAL_X[itr * MB:itr * MB + mb, :, :]
                    val_y_exp_val = VAL_Y_EXP[itr * MB:itr * MB + mb, :]
                    val_y_pse_val = VAL_Y_PSE[itr * MB:itr * MB + mb, :]
                    [val_loss_val, val_acc_exp_val, val_acc_pse_val] =\
                        sess.run([val_loss, val_acc_exp, val_acc_pse], feed_dict={val_x: val_x_val, val_y_exp: val_y_exp_val, val_y_pse: val_y_pse_val})
                    mean_val_loss += val_loss_val * mb
                    mean_val_acc_exp += val_acc_exp_val * mb
                    mean_val_acc_pse += val_acc_pse_val * mb
                    mean_val_count += mb

                print 'mean val loss %g, mean val acc exp %g, mean val acc pse %g' % (mean_val_loss / mean_val_count, mean_val_acc_exp / mean_val_count, mean_val_acc_pse / mean_val_count)

                print 't %g' % (time.time() - t)


train_val()
