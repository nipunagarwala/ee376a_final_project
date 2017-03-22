from algo_by_yuki import OMP,IST
from algo_by_tun import AMP

from data_gen import *

def test_sound(alg, repNum, soundType):
	"""
	Generates sound according to @soundType, and applies the algorithm on it.
	Saves the result in .wav format.

	Testing is repeated @repNum times.
	"""
	fs = 1000

	if soundType=='c_maj':
		alg.input_func = gen_sound_dataset
		alg.input_func_args=([(1,'C',0),(1,'E',0),(1,'G',0)], fs)
	elif soundType=='c_min':
		alg.input_func = gen_sound_dataset
		alg.input_func_args=([(1,'C',0),(1,'D#',0),(1,'G',0)], fs)
	elif soundType=='c_fifth':
		alg.input_func = gen_sound_dataset
		alg.input_func_args=([(1,'C',0),(1,'G',0)], fs)
	elif soundType=='random':
		alg.input_func = gen_random_sound_dataset
		alg.input_func_args=fs
	else:
		print '[ERROR] test_sound() : "%s" not a recognized @soundType' % soundType
		exit(0)
	
	performance = alg.predict_perf(repNum=repNum)
	print performance

	save_freq_domain_wav(alg.x, fs, '%s_sound.wav'%soundType)
	save_freq_domain_wav(alg.x_pred, fs, '%s_sound_predicted.wav'%soundType)

	return performance

def test_real_song(alg):
	"""
	Loads 10 seconds of 8000Hz music ('dataset/wind_lq_predicted.wav'),
	applies algorithm on windows of size alg.N, and outputs the result
	in a .wav file. 
	"""
	alg.input_func = None
	alg.input_func_args = 'dataset/wind_lq.wav',True

	alg.predict_long_wav_data(fs=8000, outname='wind_lq_predicted.wav')

def test_random_noise_img(alg, repNum=1):
	"""
	Uses the data generation function as defined prior to this function call
	(set alg.input_func and alg.input_func_args), and creates an image of it
	for easier inspection of the performance.

	Testing is repeated @repNum times.
	"""
	import math

	performance = alg.predict_perf(repNum=repNum)
	print performance
	alg.plot()

	img_sz = (math.sqrt(alg.N),math.sqrt(alg.N))
	save_img(alg.x, img_sz, 'random_original.png', False)
	save_img(alg.x_pred, img_sz, 'random_predicted.png', False)

	return performance

def test_real_img(alg, img_name, use_fft):
	"""
	Loads a real image as defined in @img_name, and applies @alg.
	Set @use_fft to True/False to convert the image to frequency
	domain (i.e. apply np.fft.fft2()).
	"""
	alg.input_func = None 
	alg.input_func_args = img_name,use_fft

	performance = alg.predict_perf(repNum=1)
	print performance
	alg.plot()

	save_img(alg.x_pred, alg.img_shape, '%s_predicted.png'%img_name, use_fft)

	return performance

def test_any(alg, repNum=1):
	"""
	General testing module. Manually set the data generation method.
	(set alg.input_func and alg.input_func_args)

	Testing is repeated @repNum times.
	"""
	performance = alg.predict_perf(repNum=repNum)
	print performance
	alg.plot()

	return performance

if __name__ == "__main__":
	# alg = OMP(input_func=gen_basic_dataset, 
	# 		  input_func_args=50, 
	# 		  A_func=create_A_matrix, 
	# 		  N=1024, M=512, threshold=0.5)

	# test_real_img(alg, 'dataset/ee376a.png', use_fft=False)
	
	# alg = OMP(input_func=gen_verdu_dataset, 
	# 		  input_func_args=(0.2, 1, 0), 
	# 		  A_func=create_A_matrix, 
	# 		  N=1024, M=512, threshold=0.5)

	alg = AMP(input_func=gen_basic_dataset, 
			  input_func_args=10, 
			  A_func=create_A_matrix, 
			  N=1024, M=512,
			  lam=100, epsilon=1e-4, delta=1000000000)

	test_any(alg)