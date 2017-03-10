import numpy as np

from data_gen import *
from algorithms import Algorithm

class null_hyp(Algorithm):
	"""
	You say that the signal is sparse?
	Then just predict all 0!

	... yeah it's a baseline
	"""

	def __init__(self, input_func, input_func_args, A_func, N, M):
		Algorithm.__init__(self, input_func, input_func_args, A_func, N, M)
		pass

	def predict(self):
		self.x_pred = np.zeros(self.N)

		return self.x_pred

class OMP(Algorithm):
	"""
	Implementation of Orthogonal Matching Pursuit (OMP)
	Also known as greedy algorithm
	"""

	def __init__(self, input_func, input_func_args, A_func, N, M, threshold=0.1):
		Algorithm.__init__(self, input_func, input_func_args, A_func, N, M)
		self.threshold = threshold

	def predict(self):
		self.x_pred = np.zeros(self.N, dtype=np.complex)

		residual = self.y[:]
		T = []
		counter = 0
		while np.sum(np.abs(residual))>self.threshold:
			dotP = np.abs(np.matmul(np.conj(residual), self.A))
			new_indx = np.argmax(dotP)

			T.append(new_indx)

			At = self.A[:,T]
			self.x_pred[T] = np.matmul(np.linalg.pinv(At), self.y)

			residual = self.y - np.matmul(At, self.x_pred[T])

			counter += 1
			if counter%20==0:
				print np.sum(np.abs(residual))

		return self.x_pred

class IST(Algorithm):
	"""
	Implementation of the Iterative Soft Thresholding
	Also known as L1 regularized-Least Squares

	Details here: 
		https://pdfs.semanticscholar.org/b2ff/10caa8005521bd8a4d165d44d14df3d841e1.pdf
	"""

	def __init__(self, input_func, input_func_args, A_func, N, M, thresholdR=0.05, lam=5):
		Algorithm.__init__(self, input_func, input_func_args, A_func, N, M)
		self.thresholdR = thresholdR
		self.lam = lam

	def predict(self):
		# determine c to be larger than the largest singular value of AtA.
		largest_s = np.linalg.eigvalsh(np.matmul(self.A,self.A.T)).max()
		c_s_ratio = 1.2
		c = c_s_ratio * largest_s
		eta_theta = self.lam/c

		self.x_pred = np.zeros(self.N, dtype=np.complex)

		zero_array = np.zeros(self.N, dtype=np.complex)
		
		counter = 0
		# set residual values to at least run the loop twice
		last_resid = 10
		resid = 1
		while True:
			eta_x = self.x_pred + 1/c*np.matmul(self.A.T, (self.y - np.matmul(self.A, self.x_pred)))
			self.x_pred = np.sign(eta_x)*np.maximum(np.abs(eta_x)-eta_theta, zero_array)

			counter += 1
			if counter%25==0:
				last_resid = resid
				resid = np.sum(np.abs(self.y - np.matmul(self.A, self.x_pred)))
				print resid

				if ((resid<last_resid) and (abs(resid-last_resid)/resid < self.thresholdR)) or resid<1:
					break

			if counter%500==1:
				print 'writing...'
				save_img(self.x_pred, self.img_shape, 
						 '%s_%d.png'%(self.input_func_args[0].replace('.png',''),counter),
						 use_fft=self.input_func_args[1])

		return self.x_pred

def usages():
	"""
	alg = OMP(input_func=gen_basic_dataset, 
		  input_func_args=20, 
		  A_func=create_A_matrix, 
		  N=N, M=M)

	alg = IST(input_func=gen_basic_dataset, 
			  input_func_args=20, 
			  A_func=create_A_matrix, 
			  N=N, M=M, thresholdR=0.05, lam=10)
	"""
	pass

if __name__ == "__main__":
	pass