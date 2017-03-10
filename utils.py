import matplotlib.pyplot as plt
import numpy as np
import wave
import struct

from scipy.io.wavfile import write

def fft_basis_matrix(time2freq, size):
	dft = np.fft.fft(np.eye(size))
	basis_mtrx = dft if time2freq else np.conj(dft)/size

	return basis_mtrx

def plot(x, y):
	plt.stem(x,y, 'r', )
	plt.plot(x,y)
	plt.show()

def save_wave(data, fs, outname):
	scaled = np.int16(data/np.max(np.abs(data)) * 32767)
	write(outname, fs, scaled)

def save_freq_domain_wav(freq_signal, fs, outname=None):
	basis_mtrx = fft_basis_matrix(time2freq=False, size=len(freq_signal))

	# convert the @freq_signal to time domain
	time_signal = np.matmul(basis_mtrx, freq_signal)

	# save the signal
	save_wave(np.real(time_signal), fs, outname)

def plot_2d(data):
	plt.matshow(data, aspect='auto', origin='lower', cmap='Greys')
	plt.show()

from scipy import misc
def save_img(img_vec, shape, outfile, use_fft):
	img_mat = np.reshape(img_vec, shape)
	
	if use_fft:
		img_mat = np.real(np.fft.ifft2(img_mat))

	img_mat = np.clip(np.abs(img_mat), a_min=0, a_max=1)*255
	misc.imsave(outfile, img_mat)

if __name__ == "__main__":
	plot_2d(np.load('verdu_size_20by20_gamma_0.9_sigma_1_mean_0.npy'))
	pass
