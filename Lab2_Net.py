#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 17:26:33 2019

@author: jodie
"""

import torch
import numpy as np

class EEGNet(torch.nn.Module):
    def __init__(self, activation):
        super(EEGNet, self).__init__()
        self.firstconv = torch.nn.Sequential(
            torch.nn.Conv2d(1,16, kernel_size=(1,30), stride=(1,1), padding=(0,15), bias=True),
            torch.nn.BatchNorm2d(16))
        self.depthwiseConv = torch.nn.Sequential(
#            torch.nn.Conv2d(16,32, kernel_size=(2,1), stride=(1,1), groups=1, bias=True),
        
            torch.nn.Conv2d(16,32, kernel_size=(5,1), stride=(1,1), groups=16, bias=False),
            
            torch.nn.BatchNorm2d(32),
            activation,
            torch.nn.AvgPool2d(kernel_size=(1,4), stride=(1,4)),
            torch.nn.Dropout(p=0.25))
        self.separableConv = torch.nn.Sequential(
            torch.nn.Conv2d(32,32, kernel_size=(1,15), stride=(1,1), padding=(0,7), bias=True),
            torch.nn.BatchNorm2d(32),
            activation,
            torch.nn.AvgPool2d(kernel_size=(1,8), stride=(1,8)),
            torch.nn.Dropout(p=0.25))
#        self.classify = torch.nn.Linear(in_features=736,out_features=2)
        
        self.classify = torch.nn.Linear(in_features=224,out_features=3)
            
    def forward(self,x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        # flatten the data
        x = torch.reshape(x, (x.size()[0], np.prod(x.size()[1:])))
        x = self.classify(x)
        
#        x = self.layers(x)
        return x
            
    
#class DeepConvNet:
#    def __init__(self):
#        
        