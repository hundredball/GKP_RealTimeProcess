#Model file

import numpy as np
import torch
import classifier_energy
from scipy.signal import butter, lfilter
from sklearn.preprocessing import scale

class Model(object):
	
	def __init__(self, path=None, socket=None, client=None, address=None):
		self.so = True
		if(not path):
			print('please given the path to your model')
		if(not socket):
			self.so = False
			print('no socket connected')
		
		self.model = torch.load(path)
		self.model.eval()
		self.socket = socket
		self.client = client
		self.address = address
		
	def butter_bandpass(self, lowcut, highcut, fs, order = 5):
		nyq = 0.5*fs
		low = lowcut / nyq
		high = highcut / nyq
		b, a = butter(order, [low, high], btype='band')
		return b, a

	def butter_bandpass_filter(self, data, lowcut, highcut, fs, order = 5):
		b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
		y = lfilter(b,a,data)
		return y
	
	def preprocessing(self, data):
		data = np.array(data, dtype='float')
		#data = np.delete(data,0,0) #remove first row since it is all 0
		for i in range(data.shape[0]):
			data[i] = self.butter_bandpass_filter(data[i], 1, 50, 125, 5)
	
		for i in range(data.shape[0]):
			data[i] = scale(data[i],axis=1) #axis=1 normalize each sample independently
			
		test_data = np.reshape(data,(len(data), 1, np.size(data,2), np.size(data,1)))
		
		test_dataTS = torch.from_numpy(test_data)
		return test_dataTS
		
	def predict(self, data):
		X = self.preprocessing(data)
		print('predict data')
        
		Y = self.model(X.float())
		
		print(Y)
#		if self.so:
#			self.socket.sendall(Y.encode('utf-8'))
		
		
if __name__ == '__main__':
	t, data = classifier_energy.test()
	m = Model('EEGNet_ReLU.pt', None)
	Y = m.predict(data)
    
	Y = torch.argmax(Y, dim=1)