# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 01:45:09 2019

@author: 榮
"""

from sklearn.model_selection import train_test_split
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.integrate import simps
import scipy
from sklearn.preprocessing import normalize, scale

def load_data(file):
    data = []
    with open(file,'r') as fin:
        for row in fin:
            val = [float(i) for i in row.split(' ')[0:-1]]
            #row = np.array(row, dtype='float')
            data.append(val)
            
        data = np.array(data, dtype='float')
        data = np.delete(data,0,0)
    return data
    
def load_log(file):
    logdata = []
    #input is byte, convert to string
    with open(file,'rb') as flog:
        for row in flog:
            try:
                t = row.decode('utf-8').replace('\x00','')
                logdata.append(t)
            except:
                continue
    return logdata
        
#the time is at the beginning of a trial
def get_event_and_time(logdata):
    trial_event = []
    trial_time = []
    
    for row in logdata:
        if "num:" in row:
            trial_event.append(int(row[6]))
        if "OnsetTime:" in row:
            trial_time.append(int(re.split('\r| ', row)[1]))
            
    start_time = trial_time[-2]
    end_time = trial_time[-1]
    trial_time = trial_time[0:-2]
    
    #all_time minus start_time
    offset_trial_time = []
    for t in trial_time:
        offset_trial_time.append(t-start_time)
    
    return trial_event, offset_trial_time 

def split_data(data, time):
    rate = 125
    
    # take data from startPoint~startPoint+interval seconds
    startPoint = int(rate*0)
    interval = rate*6
    
    fine_data = []
    for i in range(len(time)):
        trig = int(round(time[i]/1000*rate))
        #minus the reference (0.6s) before task start
        m = np.mean(data[int(trig - 0.6*rate) : trig], axis=0)
        fine_data.append(data[trig+startPoint:trig+startPoint+interval]-m)
        
#        fine_data.append(data[trig+startPoint:trig+startPoint+interval])
    
    return np.array(fine_data)

def butter_bandpass(lowcut, highcut, fs, order = 5):
    nyq = 0.5*fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order = 5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b,a,data, axis=0)
    return y

def save(data):
    with open('fine_data.txt', 'w') as outfile:
        for slice_2d in data:
            np.savetxt(outfile, slice_2d)

def show_f(data):
    N = 750
    T = 1.0/125
    #x = np.linspace(0.0, N*T, N)
    yf = scipy.fft(data)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.show()
    
def split_windows(x,y):
    '''split trials into frames (2s duration (250 points), 50 pointsX_train, split_datax
 gap)'''
    duration = 250
    gap = 50
    X_train = []
    Y_train = []
    
    for i in range(x.shape[0]):
        for j in range(0, x.shape[1]-duration+1, gap):
            X_train.append(x[i, j:j+duration, :])
            Y_train.append(y[i])
            
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train, Y_train


def get_raw_data(dataName, logName):
    data = load_data(dataName)
    log = load_log(logName)
    event, fine_time = get_event_and_time(log)
    splited_data = split_data(data, fine_time)
    
    print('get raw data, 90*750*5 for data, 90 for event')    
    return splited_data, event

def get_split_data(dataName, logName, saveParaName):
    
    splited_data, event = get_raw_data(dataName, logName)
    # split data into training and validation => X_train(80x750x5), X_val(10x750x5)
    X_train, X_val, Y_train, Y_val = train_test_split(splited_data, event, test_size = 0.1, random_state=42)

    # split windows of 2 sec => X_train(?x750x5), X_val(??x750x5)
    X_train, Y_train = split_windows(X_train, Y_train)
    X_val, Y_val = split_windows(X_val, Y_val)
    
    # bandpass filter 1-50
    X_train_pass = [0]*X_train.shape[0]
    for i in range(X_train.shape[0]):
        X_train_pass[i] = butter_bandpass_filter(X_train[i], 1, 50, 125, 5)

    print(X_train_pass[0][0])
    X_train_specificPoint = []
    time_point = 0
    for i in range(len(X_train)):
        if time_point == 0 or time_point == 5 or time_point == 10:
            X_train_specificPoint.append(X_train_pass[i])
            if time_point == 10:
                time_point = -1
            
        time_point += 1
    
    mean = np.mean(X_train_specificPoint)
    std = np.std(X_train_specificPoint)
    print(mean)
    print(std)
    
    print('get split data before bandpass and normalization, (X*250*5) for data, X for event')
    print('save mean and std of bandpassed data in ', saveParaName)
    with open(saveParaName, 'w') as f:
        f.write(str(mean)+'\n')
        f.write(str(std))
    
    return X_train, Y_train, X_val, Y_val

def get_processed_data(dataName, logName, saveParamName):
    X_train, Y_train, X_val, Y_val = get_split_data(dataName, logName, saveParamName)

    # bandpass filter 1-50
    for i in range(X_train.shape[0]):
        X_train[i] = butter_bandpass_filter(X_train[i], 1, 50, 125, 5)
    for i in range(X_val.shape[0]):
        X_val[i] = butter_bandpass_filter(X_val[i], 1, 50, 125, 5)
        
        # standardize data using training data, take data point in 0,2,4 secs
    X_train_specificPoint = []
    time_point = 0
    for i in range(len(X_train)):
        if time_point == 0 or time_point == 5 or time_point == 10:
            X_train_specificPoint.append(X_train[i])
            if time_point == 10:
                time_point = -1
            
        time_point += 1
    
    mean = np.mean(X_train_specificPoint)
    std = np.std(X_train_specificPoint)
    X_train = (X_train-mean)/std
    X_val = (X_val-mean)/std
    
    # randomize the data
    #X_train, t, Y_train, tt = train_test_split(X_train,Y_train,test_size=0.0,random_state=0)
    #X_val, t, Y_val, tt = train_test_split(X_val,Y_val,test_size=0.0,random_state=0)
    
    print('return all preprocessed data')
    return X_train, Y_train, X_val, Y_val