import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import *

# Music generation
#------------------------------------------------------------
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

	if fs==0 and delT==0 and x==None:
		print '[ERROR] sine_gen(): please input either @fs and @delT or @ X...'

	if x==None:
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
	notes = {'C':261.63, 'C#':277.18, 'D':293.66, 'D#':311.13, 'E':329.63, 'F':349.23,
			 'F#':369.99, 'G':392.0, 'G#':415.3, 'A':440.0, 'A#':466.16, 'B':493.88}

	dat_list = []
	for amp,f,phase in data:
		dat_list.append((amp,notes[f],phase))
	
	x,y = complex_sine_gen(np.arange(duration*fs)/float(fs), dat_list)
	bits = x.dtype.itemsize

	pickle.dump((x,y),open(outname+'.p', 'wb'))

	y = process_for_wav(y)
	
	save_wave(y, fs, outname+'.wav')
#------------------------------------------------------------

# Verdu et al. Dataset
#------------------------------------------------------------
from scipy.stats import bernoulli
from scipy.stats import norm
def gen_verdu_dataset(size, gamma, sigma, mean=0):
	shape = None
	if type(size)==tuple:
		shape = size
		size = size[0]*size[1]

	data = np.zeros(size)
	ber_rv = bernoulli.rvs(p=gamma, size=size)
	normal_rv = norm.rvs(loc=mean, scale=sigma, size=size)

	data = ber_rv*normal_rv
	
	if shape==None:
		np.save('verdu_size_%d_gamma_%0.1f_sigma_%0.1f_mean_%d.npy'%(size,gamma,sigma,mean), data)
	else:
		np.save('verdu_size_%dby%d_gamma_%0.1f_sigma_%0.1f_mean_%d.npy'%(shape[0],shape[1],gamma,sigma,mean), data.reshape(shape))
#------------------------------------------------------------

if __name__ == "__main__":
	# fs = 1000
	# duration = 10
	# data = ((1,'C',0),(1,'G',0))
	# notes2wav(data, fs, duration, "c_fifth")
	pass