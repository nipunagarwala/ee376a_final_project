import numpy as np 
import os
import sys
from algorithms import Algorithm




class AMP(Algorithm):

	def __init__(self, input_func, input_func_args, A_func, N, M, epsilon=1e-2, lam = 2, num_iter = 2000):
		Algorithm.__init__(self, input_func, input_func_args, A_func, N, M)
		self.epsilon = epsilon
		self.num_iter = 1000
		self.lam = lam


	def predict(self):

		def eta(gamma, tau):
			res = np.zeros_like(gamma)
			# for i in range(0, len(gamma)):
			# 	if gamma[i] > tau:
			# 		res[i] = gamma[i] - tau
			# 	elif gamma[i] < -tau:
			# 		res[i] = gamma[i] + tau

			# return res
			return np.sign(gamma)*np.maximum(np.abs(gamma) - tau, np.zeros_like(gamma))

		def eta_prime(gamma, tau):
			return ((gamma > tau) + (gamma < -tau)).astype(int)

		n,N = self.A.shape
		k = 10
		xhat = np.zeros((N, 1))
		# z = np.expand_dims(self.y, axis=1)
		z = np.expand_dims(np.zeros_like(self.y), axis=1)
		cur_y = np.expand_dims(self.y, axis=1)
		thresh = 1.
		for i in range(0, self.num_iter):
			xold = xhat
			thresh_old = thresh

			r = xhat + np.dot(self.A.T, z)
			xhat  = eta(r, thresh)

			# thresh = np.sqrt(np.sum((xhat)**2)/N)*self.lam
			thresh = thresh*1.0/n*np.mean(eta_prime(r, thresh))


			print "Error is: {0}".format(np.mean((self.x-xhat)**2))
			# print "L2 Norm of x: {0}".format(np.sum(self.x**2))
			# print "L2 Norm of xhat: {0}".format(np.sum(xhat**2))

			# print "Checking random norms: {0}".format(cur_y - np.dot(self.A,xhat))

			z = cur_y - np.dot(self.A,xhat) + (z*1.0/n)*np.mean(eta_prime(xold + np.dot(self.A.T, z), thresh_old))

			print "Done with iteration {0} \n".format(i)
			if np.sum((cur_y - np.dot(self.A,xhat))**2)/np.sum(cur_y**2) < self.epsilon:
				self.x_pred = np.round(xhat)
				print "Our estimate xhat is: {0}".format(xhat)
				break




# def main():



# if __name__ == '__main__'


