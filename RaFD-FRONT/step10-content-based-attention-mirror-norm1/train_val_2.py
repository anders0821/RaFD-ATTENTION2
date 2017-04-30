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
            x = tf.placeholder(tf.float32, [None, 125, 160, 1], name='x')
            mlp = x

        if train:
            with tf.name_scope('RANDOM-CROP-FLIP'):
                crop_x = tf.map_fn(lambda img: tf.random_crop(img, [112, 144, 1]), mlp)
                crop_x = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), crop_x)
                mlp = crop_x
        else:
            with tf.name_scope('CENTER-CROP'):
                crop_x = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img, 112, 144), mlp)
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
        ca = 66
        mask = tf.reshape(mlp, [-1, c4])
        mask = tf.nn.xw_plus_b(mask, weight([c4, ca], 'w5'), zero_bias([ca], 'b5'))
        mask = tf.tanh(mask)
        mask = tf.nn.xw_plus_b(mask, weight([ca, 1], 'w6'), zero_bias([1], 'b6'))
        mask = tf.reshape(mask, [-1, 7, 9])
        mask = (mask + tf.reverse(mask, [False, True, False])) / 2
        mask = tf.reshape(mask, [-1, 7*9])
        mask = tf.nn.softmax(mask)
        mask = tf.reshape(mask, [-1, 7, 9, 1])

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
        mlp = tf.nn.xw_plus_b(mlp, weight([c4, 7], name='w7'), zero_bias([7], name='b7'), name='fc')

    with tf.name_scope('Y'):
        y = tf.placeholder(tf.float32, [None, 7], name='y')

    with tf.name_scope('SOFTMAX-WITH-LOSS'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(mlp, y), name='loss')

    with tf.name_scope('SOFTMAX'):
        prob = tf.nn.softmax(mlp, name='prob')

    with tf.name_scope('ACC'):
        acc = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1), name='correct')
        acc = tf.reduce_mean(tf.cast(acc, tf.float32), name='acc')

    if train:
        with tf.name_scope('OPT'):
            opt = tf.train.AdamOptimizer(name='opt')
            train_op = opt.minimize(loss, name='train_op')
    else:
        train_op = None

    # 创建summary
    with tf.name_scope('SUM'):
        # mask 为方便展示仅尺度变化
        # mask_m = tf.reduce_min(mask, [1, 2], True)
        mask_M = tf.reduce_max(mask, [1, 2], True)
        mask_visual = mask / mask_M * 255.0  # mask_visual = (mask-mask_m) / (mask_M-mask_m) * 255.0

        # prj_mask 为方便展示仅尺度变化
        prj_mask = mask
        prj_mask = tf.nn.conv2d(prj_mask, tf.ones([3, 3, 1, 1])/9.0, strides=[1, 1, 1, 1], padding='SAME')
        prj_mask = tf.nn.conv2d(prj_mask, tf.ones([3, 3, 1, 1])/9.0, strides=[1, 1, 1, 1], padding='SAME')
        prj_mask = tf.image.resize_nearest_neighbor(prj_mask, tf.shape(prj_mask)[1:3]*2)
        prj_mask = tf.nn.conv2d(prj_mask, tf.ones([3, 3, 1, 1])/9.0, strides=[1, 1, 1, 1], padding='SAME')
        prj_mask = tf.nn.conv2d(prj_mask, tf.ones([3, 3, 1, 1])/9.0, strides=[1, 1, 1, 1], padding='SAME')
        prj_mask = tf.image.resize_nearest_neighbor(prj_mask, tf.shape(prj_mask)[1:3]*2)
        prj_mask = tf.nn.conv2d(prj_mask, tf.ones([3, 3, 1, 1])/9.0, strides=[1, 1, 1, 1], padding='SAME')
        prj_mask = tf.nn.conv2d(prj_mask, tf.ones([3, 3, 1, 1])/9.0, strides=[1, 1, 1, 1], padding='SAME')
        prj_mask = tf.nn.conv2d(prj_mask, tf.ones([3, 3, 1, 1])/9.0, strides=[1, 1, 1, 1], padding='SAME')
        prj_mask = tf.image.resize_nearest_neighbor(prj_mask, tf.shape(prj_mask)[1:3]*2)
        prj_mask = tf.nn.conv2d(prj_mask, tf.ones([3, 3, 1, 1])/9.0, strides=[1, 1, 1, 1], padding='SAME')
        prj_mask = tf.nn.conv2d(prj_mask, tf.ones([3, 3, 1, 1])/9.0, strides=[1, 1, 1, 1], padding='SAME')
        prj_mask = tf.nn.conv2d(prj_mask, tf.ones([3, 3, 1, 1])/9.0, strides=[1, 1, 1, 1], padding='SAME')
        prj_mask = tf.image.resize_nearest_neighbor(prj_mask, tf.shape(prj_mask)[1:3]*2)
        # prj_mask_m = tf.reduce_min(prj_mask, [1, 2], True)
        prj_mask_M = tf.reduce_max(prj_mask, [1, 2], True)
        prj_mask_visual = prj_mask / prj_mask_M * 255.0   # prj_mask_visual = (prj_mask-prj_mask_m) / (prj_mask_M-prj_mask_m) * 255.0

        # mask_crop_x 为方便展示动态范围变化
        mask_crop_x = prj_mask * crop_x
        mask_crop_x_m = tf.reduce_min(mask_crop_x, [1, 2], True)
        mask_crop_x_M = tf.reduce_max(mask_crop_x, [1, 2], True)
        mask_crop_x_visual = (mask_crop_x - mask_crop_x_m) / (mask_crop_x_M - mask_crop_x_m) * 255.0

        # y
        y_visual = tf.reshape(y, [-1, 1, 7, 1]) * 255.0

        # prob
        prob_visual = tf.reshape(prob, [-1, 1, 7, 1]) * 255.0

    if train:
        summary = tf.merge_summary([
                tf.image_summary('train mask', mask_visual),  # 1 7 9 1
                tf.image_summary('train prj_mask', prj_mask_visual),  # 1 112 144 1
                tf.image_summary('train crop_x', crop_x),  # 1 112 144 1
                tf.image_summary('train mask_crop_x', mask_crop_x_visual),  # 1 112 144 1
                tf.image_summary('train y', y_visual),  # 1 1 7 1
                tf.image_summary('train prob', prob_visual),  # 1 1 7 1
                tf.scalar_summary('train loss', loss),
                tf.scalar_summary('train acc', acc),
            ])
    else:
        summary = tf.merge_summary([
                tf.image_summary('val mask', mask_visual),  # 1 7 9 1
                tf.image_summary('val prj_mask', prj_mask_visual),  # 1 112 144 1
                tf.image_summary('val crop_x', crop_x),  # 1 112 144 1
                tf.image_summary('val mask_crop_x', mask_crop_x_visual),  # 1 112 144 1
                tf.image_summary('val y', y_visual),  # 1 1 7 1
                tf.image_summary('val prob', prob_visual),  # 1 1 7 1
                tf.scalar_summary('val loss', loss),
                tf.scalar_summary('val acc', acc),
            ])

    return [x, y, loss, prob, acc, train_op, summary, crop_x, mask]


def train_val():
    # 常量
    OUTPUT_DIR = './100-2/'
    MB = 100
    SNAPSHOT_RESUME_FROM = 0
    EPOCH_MAX = 10000
    # SNAPSHOT_INTERVAL = 1
    FOLD_FOR_VAL = 2

    # 加载数据集
    [TRAIN_X, TRAIN_Y, VAL_X, VAL_Y] = DATA.load_cv(FOLD_FOR_VAL)

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
            [train_x, train_y, train_loss, _, train_acc, train_op, train_summary, train_crop_x, train_mask] = build_graph(True)
        with tf.variable_scope('GRAPH', reuse=True):
            [val_x, val_y, val_loss, _, val_acc, _, val_summary, val_crop_x, val_mask] = build_graph(False)

        # 创建会话
        with tf.Session() as sess:
            # 创建summary_writer
            summary_writer = tf.train.SummaryWriter(OUTPUT_DIR, sess.graph)
            summary_writer.flush()

            # 训练初始化或加载快照
            if SNAPSHOT_RESUME_FROM == 0:
                tf.initialize_all_variables().run()
            else:
                tf.train.Saver().restore(sess, OUTPUT_DIR+'snapshot-'+str(SNAPSHOT_RESUME_FROM))
                print 'load snapshot'

            # 训练循环
            # 1 ~ EPOCH_MAX 或 SNAPSHOT_RESUME_FROM+1 ~ EPOCH_MAX
            for epoch in xrange(SNAPSHOT_RESUME_FROM+1, EPOCH_MAX+1):
                print '---------- epoch %d ----------' % epoch
                t = time.time()
                mean_train_loss = 0.0
                mean_train_acc = 0.0
                mean_train_count = 0
                mean_val_loss = 0.0
                mean_val_acc = 0.0
                mean_val_count = 0

                # 打乱训练集
                idx = np.random.permutation(TRAIN_X.shape[0])
                TRAIN_X = TRAIN_X[idx, :, :]
                TRAIN_Y = TRAIN_Y[idx, :]

                # 训练
                # 抛弃训练集尾部 担心变化的MB会影响ADAM BATCHNORM等计算
                ITER_COUNT = TRAIN_X.shape[0] / MB
                TRAIN_CROP_X_VAL = np.zeros([TRAIN_X.shape[0], 112, 144, 1])
                TRAIN_MASK_VAL = np.zeros([TRAIN_X.shape[0], 7, 9, 1])
                for itr in xrange(ITER_COUNT):
                    train_x_val = TRAIN_X[itr * MB:itr * MB + MB, :, :]
                    train_y_val = TRAIN_Y[itr * MB:itr * MB + MB, :]
                    [_, train_loss_val, train_acc_val, train_summary_val, train_crop_x_val, train_mask_val] = sess.run([train_op, train_loss, train_acc, train_summary, train_crop_x, train_mask], feed_dict={train_x: train_x_val, train_y: train_y_val})
                    mean_train_loss += train_loss_val * MB
                    mean_train_acc += train_acc_val * MB
                    mean_train_count += MB

                    summary_writer.add_summary(train_summary_val, epoch)
                    summary_writer.flush()
                    TRAIN_CROP_X_VAL[itr * MB:itr * MB + MB, :, :] = train_crop_x_val
                    TRAIN_MASK_VAL[itr * MB:itr * MB + MB, :, :, :] = train_mask_val

                print 'mean train loss %g, mean train acc %g' % (mean_train_loss / mean_train_count, mean_train_acc / mean_train_count)

                # 验证
                # 保留验证集尾部
                ITER_COUNT = ((VAL_X.shape[0] - 1) / MB) + 1
                VAL_CROP_X_VAL = np.zeros([VAL_X.shape[0], 112, 144, 1])
                VAL_MASK_VAL = np.zeros([VAL_X.shape[0], 7, 9, 1])
                for itr in xrange(ITER_COUNT):
                    mb = min(itr * MB + MB, VAL_X.shape[0]) - itr * MB
                    val_x_val = VAL_X[itr * MB:itr * MB + mb, :, :]
                    val_y_val = VAL_Y[itr * MB:itr * MB + mb, :]
                    [val_loss_val, val_acc_val, val_summary_val, val_crop_x_val, val_mask_val] = sess.run([val_loss, val_acc, val_summary, val_crop_x, val_mask], feed_dict={val_x: val_x_val, val_y: val_y_val})
                    mean_val_loss += val_loss_val * mb
                    mean_val_acc += val_acc_val * mb
                    mean_val_count += mb

                    summary_writer.add_summary(val_summary_val, epoch)
                    summary_writer.flush()
                    VAL_CROP_X_VAL[itr * MB:itr * MB + mb, :, :] = val_crop_x_val
                    VAL_MASK_VAL[itr * MB:itr * MB + mb, :, :, :] = val_mask_val

                print 'mean val loss %g, mean val acc %g' % (mean_val_loss / mean_val_count, mean_val_acc / mean_val_count)

                # save X-Y-MASK-100.mat
                if ((epoch % 100) == 0):
                    with h5py.File(OUTPUT_DIR+'X-Y-MASK-'+str(epoch)+'.mat', 'w') as h5:
                        h5['TRAIN_CROP_X_VAL'] = TRAIN_CROP_X_VAL
                        h5['TRAIN_Y'] = TRAIN_Y
                        h5['TRAIN_MASK_VAL'] = TRAIN_MASK_VAL
                        h5['VAL_CROP_X_VAL'] = VAL_CROP_X_VAL
                        h5['VAL_Y'] = VAL_Y
                        h5['VAL_MASK_VAL'] = VAL_MASK_VAL

                # 人工申请的save snapshot
                try:
                    os.remove(OUTPUT_DIR+'requestsave')
                    tf.train.Saver().save(sess, OUTPUT_DIR+'snapshot-'+str(epoch))
                    print 'save snapshot'
                except:
                    pass

                # 计划的save snapshot
                # if (epoch in []) or ((epoch % SNAPSHOT_INTERVAL) == 0):
                #     tf.train.Saver().save(sess, OUTPUT_DIR+'snapshot-'+str(epoch))
                #     print 'save snapshot'

                print 't %g' % (time.time() - t)


train_val()
