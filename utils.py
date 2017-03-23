import matplotlib.pyplot as plt
import numpy as np
import wave
import struct
import pywt

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
def save_img(img_vec, alg, outfile, use_transform):
	if use_transform:
		cA = img_vec[:alg.cA_end]
		cH = img_vec[alg.cA_end:alg.cH_end]
		cV = img_vec[alg.cH_end:alg.cV_end]
		cD = img_vec[alg.cV_end:]

		cA = np.reshape(cA, alg.cA_shape)
		cH = np.reshape(cH, alg.cH_shape)
		cV = np.reshape(cV, alg.cV_shape)
		cD = np.reshape(cD, alg.cD_shape)

		img_mat = pywt.idwt2((cA,(cH,cV,cD)), 'haar')
	else:
		img_mat = np.reshape(img_vec, alg.img_shape)

	img_mat = np.clip(np.abs(img_mat), a_min=0, a_max=1)*255
	misc.imsave(outfile, img_mat)

from scipy import signal
from scipy.io import wavfile
def plot_spectrogram_from_file(filename):
	fs,data = wavfile.read(filename)
	data = data/np.mean(data)
	
	f, t, Sxx = signal.spectrogram(data[8000*6:8000*9], fs)
	# f, t, Sxx = signal.spectrogram(data[0:8000*3], fs)

	plt.pcolormesh(t, f, Sxx)
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')

	plt.show()

if __name__ == "__main__":
	# plot_2d(np.load('verdu_size_20by20_gamma_0.9_sigma_1_mean_0.npy'))
	# plot_spectrogram_from_file('results/nb_results/wind_lq_predicted.wav')
	# plot_spectrogram_from_file('results/omp_results/wind_lq_predicted.wav')
	plot_spectrogram_from_file('dataset/wind_lq.wav')
	pass
