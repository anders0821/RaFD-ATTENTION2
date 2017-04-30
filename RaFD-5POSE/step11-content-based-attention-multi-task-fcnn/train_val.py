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


def build_graph(mode):
    with tf.device("/cpu:0"):
        if mode == 'train':
            with tf.name_scope('X'):
                x = tf.placeholder(tf.float32, [None, 284, 284, 1], name='x')
                mlp = x
            with tf.name_scope('RANDOM-CROP-FLIP'):
                crop_x = tf.map_fn(lambda img: tf.random_crop(img, [272, 272, 1]), mlp)
                # crop_x = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), crop_x)  # 使用random_flip_up_down会影响逐pose可视化X-Y-MASK 翻转改在数据加载中先做一遍增广 需要两倍内存
                mlp = crop_x
        elif mode == 'val':
            with tf.name_scope('X'):
                x = tf.placeholder(tf.float32, [None, 284, 284, 1], name='x')
                mlp = x
            with tf.name_scope('CENTER-CROP'):
                crop_x = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img, 272, 272), mlp)
                mlp = crop_x
        elif mode == 'fcnn':
            # FCNN模式下输入数据较前两个模式大
            # CONV层本身就是对输入自适应的，三个模式的输入维度不同 参数维度相同 输出维度不同
            # MASK层实现时需要保证自适应，三个模式的输入维度不同 参数维度相同 输出维度相同
            # 最后的FC层是写死的，三个模式的输入维度相同 参数维度相同 输出维度相同
            # 三个模式在输入尺寸变化情况下参数维度都相同 故可以共享参数
            with tf.name_scope('X'):
                x = tf.placeholder(tf.float32, [None, None, None, 1], name='x')
                mlp = x
            with tf.name_scope('NO-CROP'):
                crop_x = mlp + 0
                mlp = crop_x
        else:
            assert False

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

        mask = tf.reshape(mask, [-1, tf.shape(mlp)[1]*tf.shape(mlp)[2]])
        mask = tf.nn.softmax(mask)
        mask = tf.reshape(mask, [-1, tf.shape(mlp)[1], tf.shape(mlp)[2], 1])

        mlp = tf.mul(mlp, mask)
        mlp = tf.reduce_sum(mlp, [1, 2], True)

    if mode == 'train':
        with tf.name_scope('DROPOUT'):
            mlp = tf.nn.dropout(mlp, 0.5, noise_shape=tf.shape(mlp)*[1, 0, 0, 1]+[0, 1, 1, 0], name='dropout')  # dropout by map
    elif mode == 'val':
        pass
    elif mode == 'fcnn':
        pass
    else:
        assert False

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

    if mode == 'train':
        with tf.name_scope('OPT'):
            opt = tf.train.AdamOptimizer(name='opt')
            train_op = opt.minimize(loss, name='train_op')
    elif mode == 'val':
        train_op = None
    elif mode == 'fcnn':
        train_op = None
    else:
        assert False

    # 创建summary
    '''
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

        # y_exp
        y_exp_visual = tf.reshape(y_exp, [-1, 1, 7, 1]) * 255.0

        # y_pse
        y_pse_visual = tf.reshape(y_pse, [-1, 1, 7, 1]) * 255.0

        # prob
        prob_visual = tf.reshape(prob, [-1, 1, 7, 1]) * 255.0
        '''

    if mode == 'train':
        summary = tf.merge_summary([
            # tf.image_summary('train mask', mask_visual),  # 1 17 17 1
            # tf.image_summary('train prj_mask', prj_mask_visual),  # 1 272 272 1
            # tf.image_summary('train crop_x', crop_x),  # 1 272 272 1
            # tf.image_summary('train mask_crop_x', mask_crop_x_visual),  # 1 272 272 1
            # tf.image_summary('train y_exp', y_exp_visual),  # 1 1 7 1
            # tf.image_summary('train y_pse', y_pse_visual),  # 1 1 5 1
            # tf.image_summary('train prob', prob_visual),  # 1 1 7 1
            tf.scalar_summary('train loss', loss),
            tf.scalar_summary('train loss_exp', loss_exp),
            tf.scalar_summary('train loss_pse', loss_pse),
            tf.scalar_summary('train acc_exp', acc_exp),
            tf.scalar_summary('train acc_pse', acc_pse),
        ])
    elif mode == 'val':
        summary = tf.merge_summary([
            # tf.image_summary('val mask', mask_visual),  # 1 17 17 1
            # tf.image_summary('val prj_mask', prj_mask_visual),  # 1 272 272 1
            # tf.image_summary('val crop_x', crop_x),  # 1 272 272 1
            # tf.image_summary('val mask_crop_x', mask_crop_x_visual),  # 1 272 272 1
            # tf.image_summary('val y_exp', y_exp_visual),  # 1 1 7 1
            # tf.image_summary('val y_pse', y_pse_visual),  # 1 1 5 1
            # tf.image_summary('val prob', prob_visual),  # 1 1 7 1
            tf.scalar_summary('val loss', loss),
            tf.scalar_summary('val loss_exp', loss_exp),
            tf.scalar_summary('val loss_pse', loss_pse),
            tf.scalar_summary('val acc_exp', acc_exp),
            tf.scalar_summary('val acc_pse', acc_pse),
        ])
    elif mode == 'fcnn':
        summary = tf.merge_summary([
            # tf.image_summary('val mask', mask_visual),  # 1 17 17 1
            # tf.image_summary('val prj_mask', prj_mask_visual),  # 1 272 272 1
            # tf.image_summary('val crop_x', crop_x),  # 1 272 272 1
            # tf.image_summary('val mask_crop_x', mask_crop_x_visual),  # 1 272 272 1
            # tf.image_summary('val y_exp', y_exp_visual),  # 1 1 7 1
            # tf.image_summary('val y_pse', y_pse_visual),  # 1 1 5 1
            # tf.image_summary('val prob', prob_visual),  # 1 1 7 1
            tf.scalar_summary('val loss', loss),
            tf.scalar_summary('val loss_exp', loss_exp),
            tf.scalar_summary('val loss_pse', loss_pse),
            tf.scalar_summary('val acc_exp', acc_exp),
            tf.scalar_summary('val acc_pse', acc_pse),
        ])
    else:
        assert False

    return [x, y_exp, y_pse, loss, acc_exp, acc_pse, train_op, summary, crop_x, mask]


def train_val():
    # 常量1
    OUTPUT_DIR = './100/'
    MB = 30
    SNAPSHOT_RESUME_FROM = 0
    EPOCH_MAX = 1000000
    SNAPSHOT_INTERVAL = 1000000
    FOLD_FOR_VAL = 0

    # 加载数据集
    [TRAIN_X, TRAIN_Y_EXP, TRAIN_Y_PSE, VAL_X, VAL_Y_EXP, VAL_Y_PSE] = DATA.load_cv(FOLD_FOR_VAL)

    # 加载HAPPEI
    HAPPEI = []
    for i in xrange(4):
        h5 = h5py.File("./HAPPEI/HAPPEI-%d.mat" % (i+1), 'r')
        happei = h5['HAPPEI'][:]
        happei = np.reshape(happei, [1, happei.shape[0], happei.shape[1], 1])  # 1 channel
        print 'happei.shape', happei.shape
        HAPPEI.append(happei)

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
            [train_x, train_y_exp, train_y_pse, train_loss, train_acc_exp, train_acc_pse, train_op, train_summary, train_crop_x, train_mask] = build_graph('train')
        with tf.variable_scope('GRAPH', reuse=True):
            [val_x, val_y_exp, val_y_pse, val_loss, val_acc_exp, val_acc_pse, _, val_summary, val_crop_x, val_mask] = build_graph('val')
        with tf.variable_scope('GRAPH', reuse=True):
            [fcnn_x, fcnn_y_exp, fcnn_y_pse, fcnn_loss, fcnn_acc_exp, fcnn_acc_pse, _, fcnn_summary, fcnn_crop_x, fcnn_mask] = build_graph('fcnn')

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
                TRAIN_CROP_X_VAL = np.zeros([TRAIN_X.shape[0], 272, 272, 1], np.float32)
                TRAIN_MASK_VAL = np.zeros([TRAIN_X.shape[0], 17, 17, 1], np.float32)
                for itr in xrange(ITER_COUNT):
                    train_x_val = TRAIN_X[itr * MB:itr * MB + MB, :, :]
                    train_y_exp_val = TRAIN_Y_EXP[itr * MB:itr * MB + MB, :]
                    train_y_pse_val = TRAIN_Y_PSE[itr * MB:itr * MB + MB, :]
                    [_, train_loss_val, train_acc_exp_val, train_acc_pse_val, train_summary_val, train_crop_x_val, train_mask_val] =\
                        sess.run([train_op, train_loss, train_acc_exp, train_acc_pse, train_summary, train_crop_x, train_mask], feed_dict={train_x: train_x_val, train_y_exp: train_y_exp_val, train_y_pse: train_y_pse_val})
                    mean_train_loss += train_loss_val * MB
                    mean_train_acc_exp += train_acc_exp_val * MB
                    mean_train_acc_pse += train_acc_pse_val * MB
                    mean_train_count += MB

                    summary_writer.add_summary(train_summary_val, epoch)
                    summary_writer.flush()
                    TRAIN_CROP_X_VAL[itr * MB:itr * MB + MB, :, :] = train_crop_x_val
                    TRAIN_MASK_VAL[itr * MB:itr * MB + MB, :, :, :] = train_mask_val

                print 'mean train loss %g, mean train acc exp %g, mean train acc pse %g' % (mean_train_loss / mean_train_count, mean_train_acc_exp / mean_train_count, mean_train_acc_pse / mean_train_count)

                # 验证
                # 保留验证集尾部
                ITER_COUNT = ((VAL_X.shape[0] - 1) / MB) + 1
                VAL_CROP_X_VAL = np.zeros([VAL_X.shape[0], 272, 272, 1], np.float32)
                VAL_MASK_VAL = np.zeros([VAL_X.shape[0], 17, 17, 1], np.float32)
                for itr in xrange(ITER_COUNT):
                    mb = min(itr * MB + MB, VAL_X.shape[0]) - itr * MB
                    val_x_val = VAL_X[itr * MB:itr * MB + mb, :, :]
                    val_y_exp_val = VAL_Y_EXP[itr * MB:itr * MB + mb, :]
                    val_y_pse_val = VAL_Y_PSE[itr * MB:itr * MB + mb, :]
                    [val_loss_val, val_acc_exp_val, val_acc_pse_val, val_summary_val, val_crop_x_val, val_mask_val] =\
                        sess.run([val_loss, val_acc_exp, val_acc_pse, val_summary, val_crop_x, val_mask], feed_dict={val_x: val_x_val, val_y_exp: val_y_exp_val, val_y_pse: val_y_pse_val})
                    mean_val_loss += val_loss_val * mb
                    mean_val_acc_exp += val_acc_exp_val * mb
                    mean_val_acc_pse += val_acc_pse_val * mb
                    mean_val_count += mb

                    summary_writer.add_summary(val_summary_val, epoch)
                    summary_writer.flush()
                    VAL_CROP_X_VAL[itr * MB:itr * MB + mb, :, :] = val_crop_x_val
                    VAL_MASK_VAL[itr * MB:itr * MB + mb, :, :, :] = val_mask_val

                print 'mean val loss %g, mean val acc exp %g, mean val acc pse %g' % (mean_val_loss / mean_val_count, mean_val_acc_exp / mean_val_count, mean_val_acc_pse / mean_val_count)

                # fcnn
                FCNN_CROP_X_VAL = []
                FCNN_MASK_VAL = []
                for happei in HAPPEI:
                    fcnn_x_val = happei
                    fcnn_y_exp_val = np.zeros([1, 7], np.float32)
                    fcnn_y_pse_val = np.zeros([1, 5], np.float32)
                    [fcnn_loss_val, fcnn_acc_exp_val, fcnn_acc_pse_val, fcnn_summary_val, fcnn_crop_x_val, fcnn_mask_val] = \
                        sess.run([fcnn_loss, fcnn_acc_exp, fcnn_acc_pse, fcnn_summary, fcnn_crop_x, fcnn_mask], feed_dict={fcnn_x: fcnn_x_val, fcnn_y_exp: fcnn_y_exp_val, fcnn_y_pse: fcnn_y_pse_val})

                    FCNN_CROP_X_VAL.append(fcnn_crop_x_val)
                    FCNN_MASK_VAL.append(fcnn_mask_val)

                # save X-Y-MASK.mat
                if (epoch in [1]) or ((epoch % 1) == 0):### if (epoch in [1]) or ((epoch % 10) == 0):
                    with h5py.File(OUTPUT_DIR+'X-Y-MASK-'+str(epoch)+'.mat', 'w') as h5:
                        h5['TRAIN_CROP_X_VAL'] = TRAIN_CROP_X_VAL
                        h5['TRAIN_Y_EXP'] = TRAIN_Y_EXP
                        h5['TRAIN_Y_PSE'] = TRAIN_Y_PSE
                        h5['TRAIN_MASK_VAL'] = TRAIN_MASK_VAL
                        h5['VAL_CROP_X_VAL'] = VAL_CROP_X_VAL
                        h5['VAL_Y_EXP'] = VAL_Y_EXP
                        h5['VAL_Y_PSE'] = VAL_Y_PSE
                        h5['VAL_MASK_VAL'] = VAL_MASK_VAL
                        for i in xrange(4):
                            h5["FCNN_CROP_X_VAL_%d" % i] = FCNN_CROP_X_VAL[i]
                            h5["FCNN_MASK_VAL_%d" % i] = FCNN_MASK_VAL[i]

                # 降低内存峰值
                del TRAIN_CROP_X_VAL
                del VAL_CROP_X_VAL
                del FCNN_CROP_X_VAL
                del TRAIN_MASK_VAL
                del VAL_MASK_VAL
                del FCNN_MASK_VAL

                # 人工申请的save snapshot
                try:
                    os.remove(OUTPUT_DIR+'requestsave')
                    tf.train.Saver().save(sess, OUTPUT_DIR+'snapshot-'+str(epoch))
                    print 'save snapshot'
                except:
                    pass

                # 计划的save snapshot
                if (epoch in []) or ((epoch % SNAPSHOT_INTERVAL) == 0):
                    tf.train.Saver().save(sess, OUTPUT_DIR+'snapshot-'+str(epoch))
                    print 'save snapshot'

                print 't %g' % (time.time() - t)


train_val()
