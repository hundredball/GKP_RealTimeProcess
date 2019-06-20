#Model file

import numpy as np
import torch
import classifier_energy
from scipy.signal import butter, lfilter
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import Lab2_Run as LR
import time

class Model(object):
	
	def __init__(self, path=None, paramPath = None, socket=None, client=None, address=None):
		self.so = True
		if(not path):
			print('please given the path to your model')
		if(not socket):
			self.so = False
			print('no socket connected')
		
		mean = None
		std = None
		with open(paramPath,'r') as f:
			mean = float(f.readline())
			std = float(f.readline())
		self.mean =  mean
		self.std = std

		self.model = torch.load(path)
		self.model.eval()
		self.socket = socket
		self.client = client
		self.address = address
		self.device = torch.device("cuda")
		self.ref = None
		self.z = None
		self.flag_z = False
		
	def butter_bandpass(self, lowcut, highcut, fs, order = 5):
		nyq = 0.5*fs
		low = lowcut / nyq
		high = highcut / nyq
		b, a = butter(order, [low, high], btype='band')
		return b, a

	def butter_bandpass_filter(self, data, lowcut, highcut, fs, order = 5):
		b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
#		print('b:', b.shape)
#		print('a:', a.shape)
		if self.flag_z == False:
			self.z = np.zeros((max(len(a), len(b))-1, 2), dtype=np.float)
			self.flag_z = True
		
		#print(self.z.shape)
		#data has 2 channels, so z also needs to be 2*?
		y, self.z = lfilter(b,a,data,axis=0, zi = self.z)
		return y
	
	def preprocessing(self, data):
		data = np.delete(data, [i for i in range(5,16)], axis=1)
		data = np.array(data, dtype='float')
		data = np.delete(data,0,0) #remove first row since it is all 0
#		print('shape',data.shape)
		#data is (250*5) #modify at 0424

		data = self.butter_bandpass_filter(data, 0.5, 30, 125, 5)
		data -= self.ref
	
		#for i in range(data.shape[0]):
		#data = scale(data,axis=1) #axis=1 normalize each sample independently		
		mean = np.mean(data)
		std = np.std(data)
		data = data-mean
		data = data/std

		#test_data = np.reshape(data,(len(data), 1, np.size(data,2), np.size(data,1)))
#		test_data = np.reshape(data,(1, 1, np.size(data,1), np.size(data,0)))
		#test_data = np.reshape(data, (len(data), np.size(data,1), np.size(data, 2), 1)).swapaxes(1,3)
		test_data = data.T
		test_data = np.reshape(test_data, (1,1,test_data.shape[0],test_data.shape[1]))
        
		test_dataTS = torch.from_numpy(test_data)
		return test_dataTS
		
	def predict(self, data, label=None, s=None):
#		print('-------Data------')
		#print(data[2:5, 0:15])

		
		X = self.preprocessing(data)
		#X = self.test_preprocessing(data, label)
#		print('-------X---------')
#		print(X.shape)
		
		X = X.to(self.device)
		
		Y = self.model(X.float())
#		print('Y shape: ', Y.shape)        
		Y = torch.argmax(self.model(X.float()), dim=1).cpu().numpy()
		
		print('ans',Y)
#		print('socket:', self.socket)
		if self.so:
			self.socket.sendall(bytes(Y))
		return Y
    
	def test_preprocessing(self, data, label):
		'''data, label = getData.get_split_data(dataFile, eventFile, paramPath)'''
		'''注意此函數會修改吃進來的data變數= ='''
		for i in range(data.shape[0]):
			data[i] = self.butter_bandpass_filter(data[i], 1, 50, 125, 5)
	
		#for i in range(data.shape[0]):
		#data = scale(data,axis=1) #axis=1 normalize each sample independently		
		data = data-self.mean
		data = data/self.std
		
		# randomize the data
		#data, t, Y_train, tt = train_test_split(data,label,test_size=0.0,random_state=0)
        
		test_data = np.reshape(data, (len(data), np.size(data,1), np.size(data, 2), 1)).swapaxes(1,3)
        
		test_dataTS = torch.from_numpy(test_data)
		print('test model prediction using prepared data(X*250*5)')
		return test_dataTS
    
	def setReference(self, data):
		data = np.delete(data, [i for i in range(5,16)], axis=1)
		data = np.array(data, dtype='float')
		data = np.delete(data,0,0) #remove first row since it is all 0
#		print('shape',data.shape)
		
		data = self.butter_bandpass_filter(data, 0.5, 30, 125, 5)
		
		fs = 125
		interval = 0.6
		self.ref = np.mean(data[-int(fs*interval):], axis=0)
		print('Set ref: ', self.ref)
        
	def runThread(self, isRestMode, data):
#		since = time.time()
		if isRestMode:
			self.setReference(data)
		else:
			self.predict(data)
#		print('Spend %f seconds' %(time.time()-since))
		return True

'''
if __name__ == '__main__':
#	t, data = classifier_energy.test()
    
    data = np.load('RawData_11-1.npy')
    label = np.load('Labels_11-1.npy')
    label -= 1
    data, label = classifier_energy.split_windows(data, label)
    
    m = Model('EEGNet_ReLU_0420.pt', 'param_0420.txt', None)
    # demo model result
    Y = m.predict(data)
    
    # train model result
    original_model_Y = LR.runTest('11-1', '0420', '0420')
    
    correct = sum(label == Y)
    print('label == demo_model: ', correct)
    
    correct = sum(original_model_Y == Y)
    print('demo_model == train_model: ', correct)
'''
if __name__ == '__main__':
    import getData
    dataFile = '0424/tongue_move_5channel_11-2.txt'
    logFile = '0424/GKP_Exp0424.txt'
    paramFile = '0424/param_0424.txt'
    modelName = 'EEGNet_ReLU_0424.pt'
    
    data, label, datav, labelv = getData.get_split_data(dataFile, logFile, paramFile)
    m = Model('EEGNet_ReLU_0424.pt', '0424/param_0424.txt', None)
    Y_demo = m.predict(data, label)
    
    device = torch.device('cuda')
    data2, label2, data2v, label2v = getData.get_processed_data(dataFile, logFile, paramFile)
    test_data = np.reshape(data2, (len(data2), np.size(data2,1), np.size(data2, 2), 1)).swapaxes(1,3)
    test_dataTS = torch.from_numpy(test_data)
    test_dataTS = test_dataTS.to(device)
    model = torch.load(modelName).to(device=device)
    model.eval()
    
    Y = model(test_dataTS.float())