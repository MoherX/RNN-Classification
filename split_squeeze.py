#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
#
import numpy as np

data_1= np.c_[np.array([1,2,3]), np.array([4,5,6.9]),np.array([4,5,6.9])].astype(int)
data_2= np.c_[np.array([5,6,3]), np.array([9,5,6.9]),np.array([2,0,6.9])].astype(int)
data=np.array([data_1,data_2])
print data_1.shape,data.shape
print data
data = np.split(data,3,1)
print data
print data[0].shape
print np.squeeze(data[0],1).shape