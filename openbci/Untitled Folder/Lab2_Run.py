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
#    plt.plot(train_accuracy_list)
#    plt.title('Train Accuracy')
    
#    plt.figure()
#    plt.plot(test_accuracy_list)
#    plt.title('Test Accuracy')
    
    endTime = time.time()
    print('It costs '+str(endTime-startTime)+' seconds.')
    
    return train_accuracy_list, test_accuracy_list

if __name__ == '__main__':
    
    # wrap up training and testing data
    device = torch.device('cuda')
#    train_data, train_label, test_data, test_label = read_bci_data()
    
    X_train, train_label, X_val, test_label = CE.test()
    train_data = np.reshape(X_train,(len(X_train), 1, np.size(X_train,2), np.size(X_train,1)))
    train_label = train_label-1
    test_data = np.reshape(X_val,(len(X_val), 1, np.size(X_val,2), np.size(X_val,1)))
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
    opt = torch.optim.Adam(EEGNetModel_ReLU.parameters(),
                             lr=learning_rate)
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
    
    #save model
    torch.save(EEGNetModel_ReLU, 'EEGNet_ReLU.pt')