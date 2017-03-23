import numpy as np 
import os
import sys




class AMP(Algorithm):

	def __init__(self, input_func, input_func_args, A_func, N, M, epsilon=1e-4, num_iter = 2000):
		Algorithm.__init__(self, input_func, input_func_args, A_func, N, M)
		self.epsilon = epsilon
		self.num_iter = 1000


	def predict(self):
		print self.A

		def eta(gamma, tau):
			return np.sign(gamma)*np.maximum(np.abs(gamma) - tau, np.zeros_like(gamma))

		def eta_prime(gamma, tau):
			return (gamma > tau) + (gamma < tau)

		n,N = self.A.shape

		xhat = zeros(N, 1);
		z = y

		for i in range(0, self.num_iter):
			gamma = xhat + np.dot(A.T, z)

			threshold = (np.sort(abs(gamma))[::-1])[n]

			xhat  = eta(gamma, threshold)

			print "Error is: {0}".format(np.mean((x-xhat)**2))

			z = y - np.dot(A,xhat) + (z/n)*np.sum(eta_prime(gamma, threshold))

			if np.sum((y - np.dot(A,xhat))**2)/np.sum(y**2) < self.epsilon:
				print "Our estimate xhat is: {0}".format(xhat)
				break




# def main():



# if __name__ == '__main__'


