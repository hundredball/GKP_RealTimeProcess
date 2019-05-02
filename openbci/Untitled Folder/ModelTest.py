#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:57:07 2019

@author: jodie
"""

#import Model
import torch
import torch.utils.data as Data
import classifier_energy
import numpy as np

scale_filt_split_data, t = classifier_energy.test()

test_data = np.reshape(scale_filt_split_data,(len(scale_filt_split_data), 1, np.size(scale_filt_split_data,2), np.size(scale_filt_split_data,1)))
#test_label = Y_val-1

test_dataTS = torch.from_numpy(test_data)

model = torch.load('EEGNet_ReLU.pt')
model.eval()
test_predict = torch.argmax(model(test_dataTS.float()), dim=1)

