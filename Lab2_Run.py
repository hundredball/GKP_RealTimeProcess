#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 17:39:54 2019

@author: jodie
"""

from Lab2_Net import *
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
import time
import classifier_energy as CE
#import Lab2_Run as LR2

'''
runtest(): 
    test the model which is already trained
    
main():
    train a new model with data
'''


def loss_batch(model, loss_func, xb, yb, opt=None):
        _, predicted = torch.max(model(xb),1)
        correct = (predicted == yb.long()).sum().item()
        
        if opt is not None:
            loss = loss_func(model(xb), yb.long())
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            return loss.item(), correct, len(xb)
            
        return correct, len(xb)
        
def fit(epochs, model, loss_func, opt, train_dl, test_dl, device):
    startTime = time.time()
    train_accuracy_list = [0]*epochs
    test_accuracy_list = [0]*epochs
    for epoch in range(0,epochs):
        
        
        # train phase
        model.train()
        losses, num_correct, nums = zip(*[loss_batch(model, loss_func, xb.to(device), yb.to(device), opt) for xb, yb in train_dl])
#        train_loss_list[epoch] = np.sum(np.multiply(losses, nums)) / np.sum(nums)
#        print('Train Loss: '+str(train_loss_list[epoch]))
        train_accuracy_list[epoch] = sum(num_correct) / sum(nums) * 100
        
#        print('Train Accuracy: '+str(train_accuracy_list[epoch]))
        
        # test phase
        model.eval()
        num_correct, nums = zip(*[loss_batch(model, loss_func, xb.to(device), yb.to(device)) for xb, yb in test_dl])
        test_accuracy_list[epoch] = sum(num_correct) / sum(nums) * 100
#        print('Test Accuracy: '+str(test_accuracy_list[epoch]))
        if ((epoch+1) % 30 == 0):
            print('Epoch ', (epoch+1), ': ', train_accuracy_list[epoch], ' | ', test_accuracy_list[epoch])          
        
#    plt.figure()
#    plt.subplot(2,1,1)
#    plt.plot(train_accuracy_list)
#    plt.title('Train Accuracy')
#    
#    plt.subplot(2,1,2)
#    plt.plot(test_accuracy_list)
#    plt.title('Test Accuracy')
    
    print('Highest Test Accuracy: ', max(test_accuracy_list))
    endTime = time.time()
    print('It costs '+str(endTime-startTime)+' seconds.')
    
    return train_accuracy_list, test_accuracy_list

# Test data in a selected model
def runTest(subject_session, testDate, modelDate):
    device = torch.device('cuda')
    
    dataFile = testDate + '/tongue_move_5channel_' + subject_session + '.txt'
    eventFile = testDate + '/GKP_Exp' + testDate + '.txt'
    paramFile = modelDate +'/param_' + modelDate + '.txt'
    X_test, Y_test = CE.load_testData(dataFile, eventFile, paramFile)
    test_data = np.reshape(X_test,(len(X_test), 1, np.size(X_test,2), np.size(X_test,1)))
    test_label = Y_test-1
    (test_dataTS, test_labelTS) = map(torch.from_numpy, (test_data, test_label))
    [test_dataTS, test_labelTS] = [x.to(device=device) for x in [test_dataTS, test_labelTS]]
    
    modelName = modelDate + '/EEGNet_ReLU_' + modelDate + '.pt'
    model = torch.load(modelName).to(device=device)
    model.eval()
    _, predicted = torch.max(model(test_dataTS.float()),1)
    correct = (predicted == test_labelTS.long()).sum().item()
    print('Accuracy: ', correct/test_labelTS.shape[0])

def outputTestData(subject_session, testDate):
    
    dataFile = testDate + '/tongue_move_5channel_' + subject_session + '.txt'
    eventFile = testDate + '/GKP_Exp' + testDate + '.txt'
    testData, testLabel = CE.output_testData(dataFile, eventFile)
    
    saveData = 'RawData_' + subject_session
    saveLabel = 'Labels_' + subject_session
    np.save(saveData, testData)
    np.save(saveLabel, testLabel)
#    print(testData.shape) 

# train a model based on data in one day
if __name__ == '__main__':
    
    # wrap up training and testing data
    device = torch.device('cuda')
    
    subject_session = '11-2'
    date = '0424'
    
    dataFile = date + '/tongue_move_5channel_' + subject_session + '.txt'
    eventFile = date + '/GKP_Exp' + date + '.txt'
    paramFile = date + '/param_' + date + '.txt'
    X_train, train_label, X_val, test_label = CE.test(dataFile, eventFile, paramFile)
#    train_data = np.reshape(X_train,(len(X_train), 1, np.size(X_train,2), np.size(X_train,1)))
    train_data = np.reshape(X_train,(len(X_train), np.size(X_train,1), np.size(X_train,2), 1)).swapaxes(1,3)
    train_label = train_label-1
#    test_data = np.reshape(X_val,(len(X_val), 1, np.size(X_val,2), np.size(X_val,1)))
    test_data = np.reshape(X_val,(len(X_val), np.size(X_val,1), np.size(X_val,2), 1)).swapaxes(1,3)
    test_label = test_label-1
    
    (train_dataTS, train_labelTS, test_dataTS, test_labelTS) = map(
            torch.from_numpy, (train_data, train_label, test_data, test_label))
    [train_dataTS, train_labelTS, test_dataTS, test_labelTS] = [x.to(device=device) for x in [train_dataTS, train_labelTS, test_dataTS, test_labelTS]]
     
    [train_dataset,test_dataset] = map(
            Data.TensorDataset, [train_dataTS.float(),test_dataTS.float()], [train_labelTS,test_labelTS])
    batchSize = 64
    train_dl = Data.DataLoader(train_dataset, batch_size=batchSize)
    test_dl = Data.DataLoader(test_dataset, batch_size=batchSize)
    
    #--------------------EEGNet---------------------
    EEGNetModel_ELU = EEGNet(torch.nn.ELU()).to(device=device)
    EEGNetModel_ReLU = EEGNet(torch.nn.ReLU()).to(device=device)
    EEGNetModel_Leaky = EEGNet(torch.nn.LeakyReLU()).to(device=device)
    loss_func = F.cross_entropy
    epochs = 300
    learning_rate = 0.01
    opt = torch.optim.Adam(EEGNetModel_ELU.parameters(),
                             lr=learning_rate)
    train_accuracy_ELU, test_accuracy_ELU = fit(epochs, EEGNetModel_ELU, loss_func, opt, train_dl, test_dl, device)
    opt = torch.optim.Adam(EEGNetModel_ReLU.parameters(),lr=learning_rate)
    train_accuracy_ReLU, test_accuracy_ReLU = fit(epochs, EEGNetModel_ReLU, loss_func, opt, train_dl, test_dl, device)
    opt = torch.optim.Adam(EEGNetModel_Leaky.parameters(),
                             lr=learning_rate)
    train_accuracy_Leaky, test_accuracy_Leaky = fit(epochs, EEGNetModel_Leaky, loss_func, opt, train_dl, test_dl, device)
    
    # plot results
    
    epoch_range = [i for i in range(1,epochs+1)]
    plt.figure()
    plt.plot(epoch_range,train_accuracy_ELU, epoch_range,test_accuracy_ELU,
             epoch_range,train_accuracy_ReLU, epoch_range,test_accuracy_ReLU,
             epoch_range,train_accuracy_Leaky, epoch_range,test_accuracy_Leaky)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Activation function comparison(EEGNet)')
    plt.legend(loc='lower right',labels=['elu_train','elu_test','relu_train','relu_test','leaky_relu_train','leaky_relu_test'])
    print('ELU max test accuracy:', max(test_accuracy_ELU),'%')
    print('ReLU max test accuracy:', max(test_accuracy_ReLU),'%')
    print('Leaky ReLU max test accuracy:', max(test_accuracy_Leaky),'%')
    
    # save model
    modelName = date + '/EEGNet_ReLU_' + date + '.pt'
    torch.save(EEGNetModel_ReLU, modelName)
    