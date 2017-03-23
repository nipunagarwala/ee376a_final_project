import matplotlib.pyplot as plt
import numpy as np

import time

from utils import *
from scipy.io import wavfile
from scipy import misc

from scipy import misc
class Algorithm(object):

	def __init__(self, input_func, input_func_args, A_func, N=0, M=0, A_param=None):
		"""
		@input_func 		- function / data generation function
		@input_func_args 	- tuple / tuple passed in as the 
									  argument 'params' to @input_func
		@A_func				- function / how 'A' matrix is generated
		@N 					- Size of the input (x)
		@M 					- Size of the output (y)

		'A' matrix is M by N
		"""
		self.input_func = input_func
		self.input_func_args = input_func_args
		self.A_func = A_func
		self.A_param = A_param
		self.N = N
		self.M = M

	def changeInputSz(self, N, M):
		self.N = N
		self.M = M

	# Load real data
	#------------------------------------------------------------
	def load_real_dataset(self, params):
		"""
		Loads a real data
		"""

		filename,use_fft = params
		if '.wav' in filename:			
			fs,data = wavfile.read(filename)
			data = data[fs*6:fs*16]
		elif ('.png' in filename) or ('.jpg' in filename):
			data = misc.imread(filename, flatten=True)/255.0
			self.img_shape = data.shape

			if use_fft:
				data = np.fft.fft2(data)

			data = data.flatten()
			self.N = data.size
			self.M = self.N/2

		return data
	#------------------------------------------------------------

	def generateData(self):
		"""
		@x 	- Numpy 1-D array of floats / Original input signal
		@y	- Numpy 1-D array of floats / Array to be decoded
		@A 	- Numpy 2-D array of floats / Array used to encode y
		"""

		if self.input_func==None:
			self.x = self.load_real_dataset(self.input_func_args)
		else:
			self.x = self.input_func(self.N, self.input_func_args)

		self.A = self.A_func((self.M,self.N), self.A_param)
		self.y = np.matmul(self.A, self.x)

	def predict_long_wav_data(self, fs, outname):
		self.x = self.load_real_dataset(self.input_func_args)

		mse = 0
		sparsity_err = 0
		total_x = self.x
		wav_x_pred = np.array([])
		basis_mtrx = np.array([])
		numIter = int(len(total_x)/self.N)+1
		for i in range(numIter):
			if i%1==0:
				print 'Segment %d output %d iterations' %(i,numIter)

			self.x = total_x[i*self.N:min((i+1)*self.N, len(total_x))]
			self.N = len(self.x)

			# transform the signal to frequency domain, so it is sparse
			if basis_mtrx.shape[0]!=self.x.shape[0]:
				basis_mtrx = fft_basis_matrix(time2freq=True, size=self.N)
			self.x = np.matmul(basis_mtrx, self.x)

			self.A = self.A_func((self.M, self.N))
			self.y = np.matmul(self.A, self.x)

			self.predict()

			# convert to time domain
			time_signal = np.real(np.matmul(np.conj(basis_mtrx), self.x_pred))
			wav_x_pred = np.concatenate((wav_x_pred, time_signal))

			new_mse,new_sparsity_err = self.eval_error()
			mse += new_mse
			sparsity_err += new_sparsity_err

		save_wave(wav_x_pred, fs, outname)
		return mse/numIter, sparsity_err/numIter
		
	def predict_perf(self, repNum=5):
		"""
		Measures time and error performance predictions.
		The result is averaged for @repNum runs.

		Returns	: time_elapsed, error
		"""
		print '-'*30

		mse_error = 0
		sparsity_error = 0
		time_elapsed = 0
		for i in range(repNum):
			print 'Iteration %d' % i

			self.generateData()
			start = time.time()
			self.predict()
			new_time = time.time()-start
			new_mse,new_sparsity_err = self.eval_error()

			print 'Time: %0.3f' % new_time
			print 'MSE Error: %0.3f' % new_mse
			print 'Sparsity Error: %0.3f' % new_sparsity_err
			print '-'*30

			time_elapsed += new_time
			mse_error += new_mse
			sparsity_error += new_sparsity_err

		return time_elapsed/repNum, mse_error/repNum, sparsity_error/repNum

	def predict(self):
		"""
		To be implemented by each algorithms.

		Remember to store the prediction result in self.x_pred
		"""
		raise Exception('[ERROR] : please implement predict()')

	def plot(self):
		plt.subplot(2, 1, 1)
		plt.plot(range(len(self.x)), self.x)

		plt.subplot(2, 1, 2)
		plt.plot(range(self.N), self.x_pred)

		plt.show()

	def eval_error(self):
		"""
		Evaluates the reconstruction error.
		Returns both MSE and sparsity error (i.e. the number of elements in 
		@self.x and @self.x_pred that are not both nonzero or zero)

		Returns	: (MSE, sparsity error)
		"""
		MSE = ((np.abs(self.x_pred-self.x))**2).mean()

		matches = np.logical_xor((self.x_pred!=0),(self.x!=0))
		sparsity = np.sum(matches)/float(len(self.x))

		return MSE,sparsity