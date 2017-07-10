#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import time
import csv
import collections
import cPickle as pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn


class TextLoader(object):
    def __init__(self, utils_dir, data_path, batch_size, seq_length, vocab, labels, encoding='utf8'):
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        if utils_dir is not None:
            self.utils_dir = utils_dir

            label_file = os.path.join(utils_dir, 'labels.pkl')
            vocab_file = os.path.join(utils_dir, 'vocab.pkl')
            corpus_file = os.path.join(utils_dir, 'corpus.txt')

            # 标签映射文件加载
            with open(label_file, 'r') as f:
                self.labels = pickle.load(f)
            self.label_size = len(self.labels)
            # 数据预处理
            if not os.path.exists(vocab_file):
                print 'reading corpus and processing data'
                self.preprocess(vocab_file, corpus_file, data_path)
            # 加载数据
            else:
                print 'loading vocab and processing data'
                self.load_preprocessed(vocab_file, data_path)
        # 数据加载
        elif vocab is not None and labels is not None:
            self.vocab = vocab
            self.vocab_size = len(vocab) + 1
            self.labels = labels
            self.label_size = len(self.labels)

            self.load_preprocessed(None, data_path)

        self.reset_batch_pointer()

    # 文本to向量
    # text -> vector
    def transform(self, d):
        new_d = map(self.vocab.get, d[:self.seq_length])
        new_d = map(lambda i: i if i else 0, new_d)

        if len(new_d) >= self.seq_length:
            # 截断
            new_d = new_d[:self.seq_length]
        else:
            # 补齐
            new_d = new_d + [0] * (self.seq_length - len(new_d))
        return new_d

    # 数据预处理
    def preprocess(self, vocab_file, corpus_file, data_path):
        # 加载词典
        with open(corpus_file, 'r') as f:
            corpus = f.readlines()
            corpus = ''.join(map(lambda i: i.strip(), corpus))

        try:
            corpus = corpus.decode('utf8')
        except Exception as e:
            # print e
            pass
        # 统计每个字符次数
        counter = collections.Counter(corpus)
        # count_pairs = sorted(counter.items(), key=lambda i: -i[1])
        # 按照次数排序
        count_pairs = counter.most_common()
        # 得到的char就是(letters)，里面不重复的按照从高到低的顺序存放着每个字符
        # zip() in conjunction with the * operator can be used to unzip a list:
        self.chars, _ = zip(*count_pairs)
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.chars, f)

        self.vocab_size = len(self.chars) + 1
        # vocab是一个字典，存放着每个字符对应着的index
        self.vocab = dict(zip(self.chars, range(1, len(self.chars)+1)))

        data = pd.read_csv(data_path, encoding='utf8')
        # text -> vector
        tensor_x = np.array(list(map(self.transform, data['text'])))
        tensor_y = np.array(list(map(self.labels.get, data['label'])))
        # Translates slice objects to concatenation along the second axis.
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html
        self.tensor = np.c_[tensor_x, tensor_y].astype(int)

    # 数据加载
    def load_preprocessed(self, vocab_file, data_path):
        if vocab_file is not None:
            with open(vocab_file, 'rb') as f:
                self.chars = pickle.load(f)
            self.vocab_size = len(self.chars) + 1
            self.vocab = dict(zip(self.chars, range(1, len(self.chars)+1)))

        data = pd.read_csv(data_path, encoding='utf8')
        tensor_x = np.array(list(map(self.transform, data['text'])))
        tensor_y = np.array(list(map(self.labels.get, data['label'])))
        self.tensor = np.c_[tensor_x, tensor_y].astype(int)


    def create_batches(self):
        # 样本数/batch_size
        self.num_batches = int(self.tensor.shape[0] / self.batch_size)
        if self.num_batches == 0:
            assert False, 'Not enough data, make batch_size small.'
        # This function only shuffles the array along the first axis of a multi-dimensional array.
        # The order of sub-arrays is changed but their contents remains the same.
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.shuffle.html
        np.random.shuffle(self.tensor)
        tensor = self.tensor[:self.num_batches * self.batch_size]
        # why -1 ?
        self.x_batches = np.split(tensor[:, :-1], self.num_batches, 0)
        self.y_batches = np.split(tensor[:, -1], self.num_batches, 0)


    def next_batch(self):
        x = self.x_batches[self.pointer]
        y = self.y_batches[self.pointer]
        self.pointer += 1
        return x, y


    def reset_batch_pointer(self):
        self.create_batches()
        self.pointer = 0

if __name__ == '__main__':
    data_loader = TextLoader(utils_dir='data', data_path='data/input.csv', batch_size=32, seq_length=20,vocab=None,labels=None)