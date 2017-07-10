#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class Model():
    def __init__(self, args, deterministic=False):
        self.args = args
        # cell 定义
        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception('model type not supported: {}'.format(args.model))

        deterministic = tf.Variable(deterministic,
                                    name='deterministic')  # when training, set to False; when testing, set to True
        # rnn_size指的是每个rnn单元中的神经元个数（虽然RNN途中只有一个圆圈代表，但这个圆圈代表了rnn_size个神经元）
        cell = cell_fn(args.rnn_size)
        # 固定格式，有几层rnn:num_layers
        self.cell = cell = rnn.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)
        # 输入 X
        self.input_data = tf.placeholder(tf.int64, [None, args.seq_length])
        # self.targets = tf.placeholder(tf.int64, [None, args.seq_length])  # seq2seq model
        # 输入 Y
        self.targets = tf.placeholder(tf.int64, [None, ])  # target is class label
        # cell的初始状态设为0，因为在前面设置cell时，cell_size已经设置好了，因此这里只需给出batch_size即可
        # （一个batch内有batch_size个sequence的输入）
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        # W矩阵是将输入转换到了cell_size，因此这样的大小设置:词向量矩阵的维度应该是 vocab_size * rnn_size
        with tf.variable_scope('embeddingLayer'):
            with tf.device('/cpu:0'):
                W = tf.get_variable('W', [args.vocab_size, args.rnn_size])
                embedded = tf.nn.embedding_lookup(W, self.input_data)
                # 关于tf.nn.embedding_lookup(W, self.input_data)：
                # 调用tf.nn.embedding_lookup，索引与train_dataset对应的向量，相当于用train_dataset作为一个id，去检索矩阵中与这个id对应的embedding
                # embeddinglookup得到的look_up尺寸是[batch_size, seq_length, rnn_size]
                # shape: (batch_size, seq_length, cell.input_size) => (seq_length, batch_size, cell.input_size)
                ########################
                # Prepare data shape to match `rnn` function requirements
                # Current data input shape: (batch_size, n_steps, n_input)
                # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
                ########################
                # 在 seq_length 维度切分，得到 seq_length 个 [batch_size, 1, cell.input_size]
                inputs = tf.split(embedded, args.seq_length, 1)
                # https://www.tensorflow.org/api_docs/python/tf/squeeze
                # 将 1 squeeze掉 -> [batch_size, cell.input_size]
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs, last_state = rnn.static_rnn(cell, inputs, self.initial_state, scope='rnnLayer')

        with tf.variable_scope('softmaxLayer'):
            softmax_w = tf.get_variable('w', [args.rnn_size, args.label_size])
            softmax_b = tf.get_variable('b', [args.label_size])
            logits = tf.matmul(outputs[-1], softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits)

        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.targets))  # Softmax loss
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets,
                                                           logits=logits))  # Softmax loss
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.cost)  # Adam Optimizer
        # 准确率
        self.correct_pred = tf.equal(tf.argmax(self.probs, 1), self.targets)
        self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def predict_label(self, sess, labels, text):
        x = np.array(text)
        state = self.cell.zero_state(len(text), tf.float32).eval()
        feed = {self.input_data: x, self.initial_state: state}
        probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)

        results = np.argmax(probs, 1)
        id2labels = dict(zip(labels.values(), labels.keys()))
        labels = map(id2labels.get, results)
        return labels

    def predict_class(self, sess, text):
        x = np.array(text)
        state = sess.run(self.cell.zero_state(len(text), tf.float32))
        # state = self.cell.zero_state(len(text), tf.float32)
        # feed = {self.input_data: x,self.initial_state[0]:state[0].eval(),self.initial_state[1]:state[1].eval()}
        feed = {self.input_data: x, self.initial_state: state}
        probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)

        results = np.argmax(probs, 1)
        return results
