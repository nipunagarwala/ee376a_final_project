import matplotlib.pyplot as plt
import numpy as np
import wave
import struct

def plot(x, y):
	plt.stem(x,y, 'r', )
	plt.plot(x,y)
	plt.show()

def process_for_wav(data):
	data = data/max(abs(data))
	# convert between -32767 to 32767
	data = [int(x * 32767.0) for x in data]
	# convert to binary
	data = struct.pack("h" * len(data), *data)

	return data

def save_wave(data, fs, filename):
	"""
	Generates a .wav file. Taken from: http://denshi.blog.jp/signal_processing/python/save_sine_wave
	"""
	wf = wave.open(filename, "w")
	wf.setnchannels(1)
	wf.setsampwidth(2)
	wf.setframerate(fs)
	wf.writeframes(data)
	wf.close()

def plot_2d(data):
	plt.matshow(data, aspect='auto', origin='lower', cmap='Greys')
	plt.show()

def eval_error(reconstruction, original):
	"""
	Evaluates the reconstruction error.

	@reconstruction	- numpy array / reconstructed data
	@original		- numpy array / original data
	"""

	return ((reconstruction-original)**2).mean()

if __name__ == "__main__":
	plot_2d(np.load('verdu_size_20by20_gamma_0.9_sigma_1_mean_0.npy'))
	pass
