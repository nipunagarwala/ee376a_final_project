import numpy as np
import scipy
import pickle

from utils import *

# Music generation
#------------------------------------------------------------
NOTES = {'C':261.63, 'C#':277.18, 'D':293.66, 'D#':311.13, 'E':329.63, 'F':349.23,
		 'F#':369.99, 'G':392.0, 'G#':415.3, 'A':440.0, 'A#':466.16, 'B':493.88}

def sine_gen(amplitude, f, phase, fs=0, delT=0, x=None):
	"""
	Generates a sine wave

	@amplitude	: int / sin amplitude
	@f 			: int / sin frequency
	@phase		: int / sin phase (in degrees)
	@fs			: int / sample rate per second
	@delT		: int / duration of the sin wave (in seconds)
	@x			: int list / time list
	"""

	if (fs==0) and (delT==0) and (x is None):
		print '[ERROR] sine_gen(): please input either @fs and @delT or @ X...'

	if x is None:
		x = np.arange(delT*fs)/float(fs)
	y = [amplitude * np.sin(2*np.pi*f*i + np.pi/180*phase) for i in x]

	return x,y

def complex_sine_gen(x, sine_tup):
	y = np.zeros(x.shape)
	for amp,f,phase in sine_tup:
		_,y_new = sine_gen(amp, f, phase, x=x)
		y += y_new

	return x,y

def notes2wav(data, fs, duration, outname):
	dat_list = []
	for amp,f,phase in data:
		dat_list.append((amp,NOTES[f],phase))
	
	x,y = complex_sine_gen(np.arange(duration*fs)/float(fs), dat_list)

	pickle.dump((x,y),open(outname+'.p', 'wb'))

	y = process_for_wav(y)
	
	save_wave(y, fs, outname+'.wav')

def gen_sound_dataset(size, params):
	note_list,fs = params

	dat_list = []
	for amp,f,phase in note_list:
		dat_list.append((amp,NOTES[f],phase))
	
	x,data = complex_sine_gen(np.arange(size)/float(fs), dat_list)

	# transform the signal to frequency domain, so it is sparse
	basis_mtrx = fft_basis_matrix(time2freq=True, size=size)
	data = np.matmul(basis_mtrx, data)

	return data

def gen_random_sound_dataset(size, params):
	fs = params

	num_notes = np.random.randint(1,4)
	notes = np.random.choice(NOTES.keys(), size=num_notes, replace=False)
	amps = np.random.uniform(low=0.2, high=1, size=num_notes)

	note_list = [(amp,note,0) for amp,note in zip(amps,notes)]

	return gen_sound_dataset(size, (note_list,fs))
#------------------------------------------------------------

# Verdu et al. Dataset
#------------------------------------------------------------
from scipy.stats import bernoulli
from scipy.stats import norm
def gen_verdu_dataset(size, params):
	# unpack the parameters
	gamma,sigma,mean = params

	shape = None
	if type(size)==tuple:
		shape = size
		size = size[0]*size[1]

	data = np.zeros(size)
	ber_rv = bernoulli.rvs(p=gamma, size=size)
	normal_rv = norm.rvs(loc=mean, scale=sigma, size=size)

	data = ber_rv*normal_rv
	
	# if shape==None:
	# 	np.save('verdu_size_%d_gamma_%0.1f_sigma_%0.1f_mean_%d.npy'%(size,gamma,sigma,mean), data)
	# else:
	# 	np.save('verdu_size_%dby%d_gamma_%0.1f_sigma_%0.1f_mean_%d.npy'%(shape[0],shape[1],gamma,sigma,mean), data.reshape(shape))

	return data
#------------------------------------------------------------

# Basic x signal
#------------------------------------------------------------
def gen_basic_dataset(size, params):
	"""
	Generates a signal of size @size, with K elements being 1's (i.e. K-sparse)
	"""
	K = params

	data = np.zeros(size)
	indx = np.random.choice(size, size=K, replace=False)
	data[indx] = 1

	return data
#------------------------------------------------------------

# Random signal (not sparse)
#------------------------------------------------------------
def gen_random_dataset(size, params):
	"""
	Generates a signal of size @size, with each element distributed Gaussian,
	as parametrized by @params
	"""
	mu,sig = params
	data = norm.rvs(loc=mu, scale=sig, size=size)

	return data
#------------------------------------------------------------

def create_A_matrix(sz, param=None):
	"""
	Generates the 'A' matrix.

	@sz - tuple / size of A
	"""
	A = np.random.random(sz)

	return A

def create_normal_A_matrix(sz, param=None):
	"""
	Generates the 'A' matrix.

	@sz - tuple / size of A
	"""
	A = np.random.normal(0,1,sz)

	return A

def create_0_1_A_matrix(sz, param=0.002):
	"""
	Generates the 'A' matrix.

	@sz - tuple / size of A
	"""
	A = bernoulli.rvs(p=param, size=sz)

	return A

if __name__ == "__main__":
	pass