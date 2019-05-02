# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:15:49 2019

@author: John
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.integrate import simps
import scipy
from sklearn.preprocessing import normalize, scale
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
#from xgboost import XGBClassifier
import functools

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

def SVMclassify(X_train, Y_train, X_val, Y_val):
    clf = SVC(kernel = 'poly', degree=3)
    clf.fit(X_train, Y_train)
    correct_num = sum(clf.predict(X_val) == Y_val)
    print('Validation')
    print('Total:', len(Y_val), ' | Correct:', correct_num)
    print('Accuracy:', correct_num/len(Y_val))
    
    correct_num = sum(clf.predict(X_train) == Y_train)
    print('Train')
    print('Total:', len(Y_train), ' | Correct:', correct_num)
    print('Accuracy:', correct_num/len(Y_train))
    
def XGBClassify(X_train, Y_train, X_val, Y_val):
    XG = XGBClassifier(
            colsample_bytree= 0.9, gamma= 2, max_delta_step= 5, max_depth= 8,
            min_child_weight= 2, n_estimators= 50)
    
    XG.fit(X_train,Y_train,verbose=True)
    
    correct_num = sum(XG.predict(X_val) == Y_val)
    print('Validation')
    print('Total:', len(Y_val), ' | Correct:', correct_num)
    print('Accuracy:', correct_num/len(Y_val))
    
    correct_num = sum(XG.predict(X_train) == Y_train)
    print('Train')
    print('Total:', len(Y_train), ' | Correct:', correct_num)
    print('Accuracy:', correct_num/len(Y_train))

# ------------------------ output data for EEGNet ---------------------
def test(dataName, logName, saveParaName):
    # load data
    data = load_data(dataName)
    log = load_log(logName)
    event, fine_time = get_event_and_time(log)
    
    # split data with respect to fine_time , sample rate 125Hz, remove baseline => 90x750x5
    splited_data = split_data(data, fine_time)
    
    # split data into training and validation => X_train(80x750x5), X_val(10x750x5)
    X_train, X_val, Y_train, Y_val = train_test_split(splited_data, event, test_size = 0.1, random_state=42)

    # split windows of 2 sec => X_train(800x250x5), X_val(100x250x5)
    X_train, Y_train = split_windows(X_train, Y_train)
    X_val, Y_val = split_windows(X_val, Y_val)

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
    print(mean)
    print(std)
    # randomize the data
    X_train, t, Y_train, tt = train_test_split(X_train,Y_train,test_size=0.0,random_state=0)
    X_val, t, Y_val, tt = train_test_split(X_val,Y_val,test_size=0.0,random_state=0)
    
    with open(saveParaName, 'w') as f:
        f.write(str(mean)+'\n')
        f.write(str(std))
    
    return X_train, Y_train, X_val, Y_val

def load_testData(dataName, logName, loadParaName):
    data = load_data(dataName)
    log = load_log(logName)
    event, fine_time = get_event_and_time(log)
    
    splited_data = split_data(data, fine_time)
    
    # split windows of 2 sec
    X_test, Y_test = split_windows(splited_data, event)
    
    # bandpass filter 1-50
    for i in range(len(X_test)):
        X_test[i] = butter_bandpass_filter(X_test[i], 1, 50, 125, 5)
    
    # standardize data
    with open(loadParaName, 'r') as f:
        mean = float(f.readline())
        std = float(f.readline())

#    print(filt_split_data.shape)
    X_test = (X_test-mean)/std
    
    return X_test, Y_test

def output_testData(dataName, logName):
    data = load_data(dataName)
    log = load_log(logName)
    event, fine_time = get_event_and_time(log)
    
    splited_data = split_data(data, fine_time)
    
    return splited_data, event

def getBandPower(X):
    # Define sampling rate and window length
#    X = filt_split_data
    fs = 125
#    time = np.arange(X.shape[1])/fs
    multiplier = 2/0.5
    win = multiplier*fs
    freq_res = 1/multiplier
    
    X_bandpower = np.zeros((X.shape[0], 4, X.shape[2]))
    simps_dx = functools.partial(simps, dx=freq_res)
    
    for sample in range(X.shape[0]):
        for channel in range(X.shape[2]):
            freqs, psd = scipy.signal.welch(X[sample,:,channel], fs, nperseg=win)
            [idx_delta, idx_theta, idx_alpha, idx_beta] = map(
                    np.logical_and, [freqs>=1, freqs>=4, freqs>=9, freqs>=13], [freqs<=4, freqs<=8, freqs<=13, freqs<=30])
            [X_bandpower[sample,0,channel], X_bandpower[sample,1,channel], X_bandpower[sample,2,channel], X_bandpower[sample,3,channel]] = map(
                    simps_dx, [psd[idx_delta], psd[idx_theta], psd[idx_alpha], psd[idx_beta]])
            
            
#    plt.figure(figsize=(8,4))
#    plt.plot(freqs, psd, color='k', lw=2)
#    plt.xlabel('Frequency(HZ)')
#    plt.ylabel('Power spectral density (V^2 / HZ)')
    
    return X_bandpower
    
#def test

if __name__ == '__main__':
    data = load_data('tongue_move_5channel_10-1.txt')
    log = load_log('GKP_Exp0412.txt')
    event, fine_time = get_event_and_time(log)
    
    #split data with respect to fine_time , sample rate 125Hz
    splited_data = split_data(data, fine_time)
    
    '''concate data
    con_split_data = np.zeros((90,800,5))
    #save(fine_data)
    #把後半部data複製一份接至data前面以做discrete filter
    for i in range(split_data.shape[0]):
        con = split_data[i,700:,:]
        con_split_data[i] = np.concatenate((con,split_data[i]),axis=0)
    
    '''
    
    filt_split_data = np.zeros(splited_data.shape)
    for i in range(splited_data.shape[0]):
        filt_split_data[i] = butter_bandpass_filter(splited_data[i], 1, 50, 125, 5)
    
#    scale_filt_split_data = np.zeros(filt_split_data.shape)
#    for i in range(filt_split_data.shape[0]):
#        scale_filt_split_data[i] = scale(filt_split_data[i],axis=1) #axis=1 normalize each sample independently
   
    
    ''' random data and split train/val and split window'''
    random_seed = 42
    X_train, X_val, Y_train, Y_val = train_test_split(filt_split_data,event,test_size=0.1,random_state=random_seed)
    
#    mean = np.mean(X_train)
#    std = np.std(X_train)
#    X_train = (X_train-mean)/std
#    X_val = (X_val-mean)/std
    
    X_train, Y_train = split_windows(X_train, Y_train)
    X_val, Y_val = split_windows(X_val, Y_val)
    
    X_train, t, Y_train, tt = train_test_split(X_train,Y_train,test_size=0.0,random_state=random_seed)
    X_val, t, Y_val, tt = train_test_split(X_val,Y_val,test_size=0.0,random_state=random_seed)
    
    '''calculate sum of square of each frames'''
#    X_train_energy = []
#    X_val_energy = []
#    for sample in X_train:
#        X_train_energy.append(np.sum(np.square(sample),axis=0)/np.size(sample,0))
#        
#    for sample in X_val:
#        X_val_energy.append(np.sum(np.square(sample),axis=0)/np.size(sample,0))
#    
#    X_train_energy = np.array(X_train_energy)
#    X_val_energy = np.array(X_val_energy)
    
    ''' Specific feature (0-4, 1-3, 2) '''
#    X_train_feature = np.array([X_train_energy[:,0]-X_train_energy[:,4] , X_train_energy[:,1]-X_train_energy[:,3],
#                                X_train_energy[:,2]]).T
#    X_val_feature = np.array([X_val_energy[:,0]-X_val_energy[:,4] , X_val_energy[:,1]-X_val_energy[:,3],
#                                X_val_energy[:,2]]).T
    
    ''' concate energy and difference '''
#    X_train_energy_feature = np.delete(np.concatenate((X_train_energy, X_train_feature), axis=1), 7, 1)
#    X_val_energy_feature = np.delete(np.concatenate((X_val_energy, X_val_feature), axis=1), 7, 1)
    
    ''' flatten X_train and X_val '''
#    X_train_flatten = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
#    X_val_flatten = np.reshape(X_val, (X_val.shape[0], X_val.shape[1]*X_val.shape[2]))
    
    ''' Calculate band power '''
    X_train_bandpower = getBandPower(X_train)
    X_val_bandpower = getBandPower(X_val)
    X_train_bandpower = np.reshape(X_train_bandpower, (X_train_bandpower.shape[0], X_train_bandpower.shape[1]*X_train_bandpower.shape[2]))
    X_val_bandpower = np.reshape(X_val_bandpower, (X_val_bandpower.shape[0], X_val_bandpower.shape[1]*X_val_bandpower.shape[2]))
    
    mean = np.mean(X_train_bandpower)
    std = np.std(X_train_bandpower)
    X_train_bandpower = (X_train_bandpower-mean)/std
    X_val_bandpower = (X_val_bandpower-mean)/std
    
    '''raw data to SVM, or energy to SVM'''
#    SVMclassify(X_train_energy, Y_train, X_val_energy, Y_val)
#    SVMclassify(X_train_feature, Y_train, X_val_feature, Y_val)
#    SVMclassify(X_train_energy_feature, Y_train, X_val_energy_feature, Y_val)
    print('------------SVM-------------')
    SVMclassify(X_train_bandpower, Y_train, X_val_bandpower, Y_val)
#    XGBClassify(X_train_flatten, Y_train, X_val_flatten, Y_val)
    print('------------XGB-------------')
    XGBClassify(X_train_bandpower, Y_train, X_val_bandpower, Y_val)
    
    ''' based on energy '''
#    right_energy = 0
#    left_energy = 0
#    for i in range(len(X_train_energy_part)):
        
    
    
    #normalize
#    fine_data = np.zeros(split_data.shape)
#    for i in range(fine_data.shape[0]):
#        fine_data[i] = scale(split_data[i],axis=1) #axis=1 normalize each sample independently
    
#    
    #channelwise normalize data to 0~1
    #for i in range(fine_data.shape[2]):
        
    
    
