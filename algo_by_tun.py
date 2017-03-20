import numpy as np

from data_gen import *
from algorithms import Algorithm
from cvxpy import *

class L1(Algorithm):
	"""
	Implementation of Least-norm (L1) minimization
	minimize |x|_1
	subject to Ax = y
	Also known as greedy algorithm
	"""

	def __init__(self, input_func, input_func_args, A_func, N, M):
		Algorithm.__init__(self, input_func, input_func_args, A_func, N, M)

	def predict(self):

		print 'begin CVX'

		x = Variable(self.N)
		obj = Minimize(norm(x,1))
		constraints = [self.A*x == self.y]

		problem = Problem(obj, constraints)
		problem.solve()

		print 'end CVX'

		self.x_pred = np.array(x.value)

		return self.x_pred

class LASSO(Algorithm):
	"""
	minimize 0.5/M*|Ax-y|_2^2 + alpha*|x|_1
	where M is the number of samples
	"""

	def __init__(self, input_func, input_func_args, A_func, N, M, lam=5):
		Algorithm.__init__(self, input_func, input_func_args, A_func, N, M)
		self.lam = lam

	def predict(self):

		x = Variable(self.N)
		obj = Minimize(self.lam*norm(x,1) + 0.5*norm(self.A*x-self.y,2)**2)

		problem = Problem(obj)
		problem.solve()

		self.x_pred = np.array(x.value)

		return self.x_pred

class AMP(Algorithm):
	"""
	x(t+1) = eta(A_T*z(t) + x(t))
	z(t) = y - A*x(t) + 1/delta*z(t-1)*mean(eta'(A_T*z(t-1)+x_t-1))
	eta(a) = sign(a)(|a| - lambda)
	"""

	def __init__(self, input_func, input_func_args, A_func, N, M, lam=10, epsilon=1e-4, delta=0.5):
		Algorithm.__init__(self, input_func, input_func_args, A_func, N, M)
		self.epsilon = epsilon
		self.delta = delta
		self.lam = lam


	def predict(self):
		print self.A

		def eta(w, tau):
			return np.sign(w)*np.maximum(np.abs(w) - tau, np.zeros(w.size, dtype=np.complex))

		def eta_prime(w, tau):
			if tau <= 0: return np.ones(w.size)
			w_prime = np.zeros(w.size)
			for i in np.arange(w.size):
				if w[i] > tau or w[i] < -1*tau : w_prime[i] = 1.0
			return w_prime

		# largest_s = np.linalg.eigvalsh(np.matmul(self.A,self.A.T)).max()
		# c_s_ratio = 1.2
		# c = c_s_ratio * largest_s
		# eta_theta = self.lam/c
		# print 'c = ', c

		x = np.zeros(self.N, dtype=np.complex)
		x_prev = np.zeros(self.N, dtype=np.complex)
		z = self.y - np.matmul(self.A,x)
		z_prev = np.zeros(self.M, dtype=np.complex)
		tau = 2


		while True:

			## update x
			new_x = eta(np.matmul(self.A.T, z)+x, tau)

			## update z
			new_z = self.y - np.matmul(self.A, new_x) \
			+ 1/self.delta*z*np.mean(eta_prime(np.matmul(self.A.T, z) + x, tau))
			print 'new_z = ', new_z

			## update tau
			new_tau = tau*1/self.delta*np.mean(eta_prime(np.matmul(self.A.T, z) + new_x, tau))
			print 'new_tau = ', new_tau

			diff = np.linalg.norm(new_x - x)
			print 'diff = ', diff
			if diff < self.epsilon : break

			tau = new_tau
			z = new_z
			x = new_x



			"""

			### update z
			new_z = self.y - np.matmul(self.A, self.x_pred) + 1/self.delta*z*np.mean(eta_prime(np.matmul(self.A.T, z) + x_pred_prev, tau))
			res = self.y - np.matmul(self.A, self.x_pred)
			# print 'res = ', res
			# print 'z = ', new_z


			### update tau
			f = np.mean(eta_prime(np.matmul(self.A.T, z) + self.x_pred, tau))
			print f
			new_tau = (tau)*1/self.delta*f
			# print 'tau = ', new_tau

			### update x
			x_pred_prev = self.x_pred
			self.x_pred = eta(np.matmul(self.A.T, z) + x_pred_prev, tau)
			diff = np.linalg.norm(x_pred_prev - self.x_pred)

			print 'diff = ', diff

			if diff < self.epsilon: break
      
      tau = new_tau
			z = new_z
			x = new_x

		self.x_pred = x


		return self.x_pred

def usages():
	"""
	alg = L1(input_func=gen_basic_dataset,
		  input_func_args=20,
		  A_func=create_A_matrix,
		  N=N, M=M)
	"""

	"""
	alg = AMP(input_func=gen_basic_dataset,
		  input_func_args=20,
		  A_func=create_A_matrix,
		  N=N, M=M,
		  lam=10,
		  epsilon=1e-4,
		  delta=1)
	"""
	pass

if __name__ == "__main__":
	pass
